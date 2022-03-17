from .utils import build_mlp, TanhTransform
import torch
nn = torch.nn
td = torch.distributions


class DenseNormal(nn.Module):
    def __init__(self, in_features, out_features, layers=[32, 32], min_std=1e-2):
        super().__init__()
        self.fc = build_mlp([in_features]+layers+[2*out_features])
        self.min_std = min_std

    def forward(self, x):
        mu, std = self.fc(x).chunk(2, -1)
        std = torch.maximum(std, torch.full_like(std, -18.))
        std = nn.functional.softplus(std) + self.min_std
        return mu, std


class NormalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, layers, mean_scale=5., init_std=5.):
        super().__init__()
        self.model = build_mlp([obs_dim] + layers + [2*act_dim])
        self.init_std = torch.log(torch.tensor(init_std).exp() - 1)
        self.mean_scale = mean_scale

    def forward(self, obs):
        mu, std = self.model(obs).chunk(2, -1)
        mu = self.mean_scale * torch.tanh(mu / self.mean_scale)
        std = nn.functional.softplus(std + self.init_std) + 1e-3
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


class PointCloudEncoder(nn.Module):
    def __init__(self, in_features, depth, layers):
        super().__init__()
        coef = 2 ** (layers - 1)
        self.model = nn.ModuleList([nn.Sequential(nn.Linear(in_features, coef*depth), nn.ReLU())])
        for i in range(layers-1):
            m = nn.Sequential(nn.Linear(coef*depth, coef // 2 * depth), nn.ReLU())
            self.model.append(m)
            coef //= 2

        self.fc = nn.Sequential(
            nn.Linear(depth, depth),
            nn.Tanh(),
        )

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        values, idx = torch.max(x, -2)
        return self.fc(values)


class PointCloudDecoder(nn.Module):
    def __init__(self, in_features, out_features, depth, layers, pn_number):
        super().__init__()

        self.coef = 2**(layers-1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, self.coef * depth * pn_number),
            nn.ELU(inplace=True),
        )

        self.deconvs = nn.ModuleList([nn.Unflatten(-1, (pn_number, self.coef * depth))])
        for _ in range(layers-1):
            self.deconvs.append(nn.Linear(self.coef * depth, self.coef * depth // 2))
            self.deconvs.append(nn.ELU(inplace=True))
            self.coef //= 2

        self.deconvs.append(nn.Linear(depth, out_features))

    def forward(self, x):
        x = self.fc(x)
        for layer in self.deconvs:
            x = layer(x)
        return x