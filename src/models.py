from .utils import build_mlp, TanhTransform
import torch
from torchvision import transforms as T
nn = torch.nn
F = nn.functional
td = torch.distributions


class DenseNormal(nn.Module):
    def __init__(self, in_features, out_features, layers=(32, 32), min_std=1e-2):
        super().__init__()
        self.fc = build_mlp(in_features, *layers, 2*out_features)
        self.min_std = min_std

    def forward(self, x):
        mu, std = self.fc(x).chunk(2, -1)
        std = torch.maximum(std, torch.full_like(std, -18.))
        std = F.softplus(std) + self.min_std
        return mu, std


class NormalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, layers, mean_scale=5., init_std=5.):
        super().__init__()
        self.model = build_mlp(obs_dim, *layers, 2*act_dim)
        self.init_std = torch.log(torch.tensor(init_std).exp() - 1)
        self.mean_scale = mean_scale

    def forward(self, obs):
        mu, std = self.model(obs).chunk(2, -1)
        mu = self.mean_scale * torch.tanh(mu / self.mean_scale)
        std = F.softplus(std + self.init_std) + 1e-7
        dist = td.Normal(mu, std)
        dist = td.transformed_distribution.TransformedDistribution(dist, TanhTransform())
        return td.Independent(dist, 1)


class RSSM(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 deter_dim,
                 stoch_dim,
                 layers,
                 ):
        super().__init__()
        self.obs_dim, self.act_dim, self.deter_dim, self.stoch_dim = obs_dim, act_dim, deter_dim, stoch_dim
        self.prior = DenseNormal(self.deter_dim, self.stoch_dim, layers)
        self.infer = DenseNormal(self.obs_dim + self.deter_dim, self.stoch_dim, layers)
        self.cell = nn.GRUCell(self.act_dim + self.stoch_dim, self.deter_dim)

    def img_step(self, prev_action, prev_state):
        stoch, deter = self.split_state(prev_state)[-2:]
        x = torch.cat([stoch, prev_action], -1)
        deter = self.cell(x, deter)
        mu, std = self.prior(deter)
        return self.make_state(mu, std, deter)

    def obs_step(self, obs, prev_action, prev_state):
        prior = self.img_step(prev_action, prev_state)
        deter = self.split_state(prior)[-1]
        x = torch.cat([obs, deter], -1)
        mu, std = self.infer(x)
        return self.make_state(mu, std, deter), prior

    def imagine(self, actions, state):
        if not torch.is_tensor(state):
            state = self.init_state(actions.size(1))
        states = []
        for act in actions:
            state = self.img_step(act, state)
            states.append(state)
        return torch.stack(states)

    def observe(self, observations, actions, post):
        if not torch.is_tensor(post):
            post = self.init_state(actions.size(1))
        posts, priors = [], []
        for obs, act in zip(observations, actions):
            post, prior = self.obs_step(obs, act, post)
            posts.append(post)
            priors.append(prior)
        return torch.stack(posts), torch.stack(priors)

    def split_state(self, state):
        mu, std, stoch, deter = state.split(3*(self.stoch_dim,)+(self.deter_dim,), -1)
        return mu, std, stoch, deter

    @staticmethod
    def make_state(mu, std, deter):
        stoch = mu + torch.randn_like(std)*std
        return torch.cat([mu, std, stoch, deter], -1)

    def init_state(self, batch_size):
        return torch.zeros(batch_size, 3*self.stoch_dim+self.deter_dim)

    def get_feat(self, state):
        return state.split((2*self.stoch_dim, self.stoch_dim+self.deter_dim), -1)[-1]

    def get_dist(self, state):
        mu, std = self.split_state(state)[:2]
        dist = td.Normal(mu, std)
        return td.Independent(dist, 1)

# class PointCloudEncoder(nn.Module):
#     def __init__(self, in_features, depth, layers):
#         super().__init__()
#         coef = 2 ** (layers - 1)
#         self.model = nn.ModuleList([nn.Sequential(nn.Linear(in_features, coef*depth), nn.ReLU())])
#         for i in range(layers-1):
#             m = nn.Sequential(nn.Linear(coef*depth, coef // 2 * depth), nn.ReLU())
#             self.model.append(m)
#             coef //= 2
#
#         self.fc = nn.Sequential(
#             nn.Linear(depth, depth),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         for layer in self.model:
#             x = layer(x)
#         values, idx = torch.max(x, -2)
#         return self.fc(values)
#
#
# class PointCloudDecoder(nn.Module):
#     def __init__(self, in_features, out_features, depth, layers, pn_number):
#         super().__init__()
#
#         self.coef = 2**(layers-1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_features, self.coef * depth * pn_number),
#             nn.ELU(inplace=True),
#         )
#
#         self.deconvs = nn.ModuleList([nn.Unflatten(-1, (pn_number, self.coef * depth))])
#         for _ in range(layers-1):
#             self.deconvs.append(nn.Linear(self.coef * depth, self.coef * depth // 2))
#             self.deconvs.append(nn.ELU(inplace=True))
#             self.coef //= 2
#
#         self.deconvs.append(nn.Linear(depth, out_features))
#
#     def forward(self, x):
#         x = self.fc(x)
#         for layer in self.deconvs:
#             x = layer(x)
#         return x

class PointCloudDecoder(nn.Module):
    def __init__(self, in_features, pn_number, depth=32, act=nn.ELU):
        super().__init__()

        self.deconvs = nn.Sequential(
            nn.Linear(in_features, 2*depth*pn_number),
            act(),
            nn.Unflatten(-1, (pn_number, 2*depth)),
            nn.Linear(2*depth, depth),
            act(),
            nn.Linear(depth, 3)
        )

    def forward(self, x):
        return self.deconvs(x)


class PointCloudEncoder(nn.Module):
    def __init__(self, in_features, out_features, layers, dropout=0., act=nn.ELU):
        super().__init__()
        self.convs = nn.Sequential()

        sizes = (in_features,) + layers
        for i in range(len(sizes)-1):
            block = nn.Sequential(
                nn.Linear(sizes[i], sizes[i+1]),
                act(),
                nn.Dropout(dropout)
            )
            self.convs.add_module(f'conv{i}', block)

        self.fc = nn.Sequential(
            nn.Linear(sizes[-1], out_features),
            nn.LayerNorm(out_features),
            nn.Tanh()
        )

        self.soft = False

    def forward(self, x):
        x = self.convs(x)
        # try soft sampling
        if self.soft:
            indices = F.gumbel_softmax(x, hard=True).argmax(-2)
            values = torch.gather(x, -2, indices.unsqueeze(-2)).squeeze(-2)
        else:
            values, indices = torch.max(x, -2)
        return self.fc(values)


class PixelEncoder(nn.Module):
    def __init__(self, in_channels=3, out_features=64, depth=32, act=nn.ELU):
        super().__init__()

        self.convs = nn.Sequential(
            T.Normalize(.5, 1),
            nn.Conv2d(in_channels, depth, 3, 2),
            act(),
            nn.Conv2d(depth, depth, 3, 1),
            act(),
            nn.Conv2d(depth, depth, 3, 1),
            act(),
            nn.Flatten(),
            nn.Linear(depth*27*27, out_features),  # 37 if size = 84, 27 if 64
            nn.LayerNorm(out_features),
            nn.Tanh()
        )

    def forward(self, img):
        reshape = img.ndimension() > 4  # hide temporal axis
        if reshape:
            seq_len, batch_size = img.shape[:2]
            img = img.flatten(0, 1)
        img = self.convs(img)
        if reshape:
            img = img.reshape(seq_len, batch_size, -1)
        return img


class PixelDecoder(nn.Module):
    def __init__(self, in_features, out_channels=3, depth=32, act=nn.ELU):
        super().__init__()
        self.deconvs = nn.Sequential(
            nn.Linear(in_features, depth*27*27),
            act(),
            nn.Unflatten(-1, (depth, 27, 27)),
            nn.ConvTranspose2d(depth, depth, 3, 1),
            act(),
            nn.ConvTranspose2d(depth, depth, 3, 1),
            act(),
            nn.ConvTranspose2d(depth, out_channels, 3, 2, output_padding=1),
            T.Normalize(-.5, 1)
        )

    def forward(self, x):
        reshape = x.ndimension() > 2  # hide temporal axis
        if reshape:
            seq_len, batch_size = x.shape[:2]
            x = x.flatten(0, 1)
        img = self.deconvs(x)
        if reshape:
            img = img.reshape(seq_len, batch_size, 3, 64, 64)
        return img
