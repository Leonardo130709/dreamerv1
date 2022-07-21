from .utils import build_mlp, TruncatedTanhTransform
import torch
from collections import Iterable
nn = torch.nn
F = nn.functional
td = torch.distributions


class DenseNormal(nn.Module):
    def __init__(self, in_features, out_features, layers=(32, 32), min_std=1e-1, act=nn.ELU):
        super().__init__()
        self.fc = build_mlp(in_features, *layers, 2*out_features, act=act)
        self.min_std = min_std

    def forward(self, x):
        mu, std = self.fc(x).chunk(2, -1)
        std = torch.maximum(std, torch.full_like(std, -18.))
        std = F.softplus(std) + self.min_std
        return mu, std


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, layers, mean_scale=1., init_std=5., act=nn.ELU):
        super().__init__()
        self.model = build_mlp(obs_dim, *layers, 2*act_dim, act=act)
        self.init_std = torch.log(torch.tensor(init_std).exp() - 1.)
        self.mean_scale = mean_scale

    def forward(self, obs):
        mu, std = self.model(obs).chunk(2, -1)
        mu = self.mean_scale * torch.tanh(mu / self.mean_scale)
        std = F.softplus(std + self.init_std) + 1e-3
        dist = td.Normal(mu, std)
        dist = td.transformed_distribution.TransformedDistribution(
            dist,
            td.transforms.IndependentTransform(
                TruncatedTanhTransform(cache_size=1),
                # td.transforms.TanhTransform(cache_size=1),
                reinterpreted_batch_ndims=1,
                cache_size=1)
        )
        return dist


#TODO: RSSM State better be named -- use NamedTuple
#   right now the only reason to use torch.Tensor is simplified batching throughout
#   since there is no tf.nested
class RSSM(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 deter_dim: int,
                 stoch_dim: int,
                 layers: Iterable[int],
                 ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.prior = DenseNormal(self.deter_dim, self.stoch_dim, layers)
        self.post = DenseNormal(self.obs_dim + self.deter_dim, self.stoch_dim, layers)
        self.cell = nn.GRUCell(self.act_dim + self.stoch_dim, self.deter_dim)

    def img_step(self, prev_action, prev_state):
        # h_t = f(a_tm1, h_tm1, s_tm1)
        # s_t ~ p(s|h_t)
        stoch, deter = self.split_state(prev_state)[-2:]
        x = torch.cat([stoch, prev_action], -1)
        deter = self.cell(x, deter)
        mu, std = self.prior(deter)
        return self.make_state(mu, std, deter)

    def obs_step(self, obs, prev_action, prev_state):
        # s_t ~ q(s | h_t, o_t)
        prior = self.img_step(prev_action, prev_state)
        deter = self.split_state(prior)[-1]
        x = torch.cat([obs, deter], -1)
        mu, std = self.post(x)
        return self.make_state(mu, std, deter), prior

    def imagine(self, actions, prior):
        if not torch.is_tensor(prior):
            prior = self.init_state(actions.size(1))
        priors = []
        for act in actions:
            prior = self.img_step(act, prior)
            priors.append(prior)
        return torch.stack(priors)

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
        mu, std, stoch, deter = state.split(3*[self.stoch_dim]+[self.deter_dim], -1)
        return mu, std, stoch, deter

    @staticmethod
    def make_state(mu, std, deter):
        stoch = td.Normal(mu, std).rsample()
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
    """PointNet with an option to process global features of selected points."""
    def __init__(self, in_channels, out_features, layers, act=nn.ELU, features_from_layers=()):
        super().__init__()

        layers = (in_channels,) + layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            block = nn.Sequential(
                nn.Linear(layers[i], layers[i + 1]),
                act(),
            )
            self.layers.append(block)

        if isinstance(features_from_layers, int):
            features_from_layers = (features_from_layers, )
        self.selected_layers = features_from_layers

        self.fc_size = layers[-1] * (1 + sum([layers[i] for i in self.selected_layers]))
        self.ln_emb = nn.Sequential(
            nn.Linear(self.fc_size, out_features),
            nn.LayerNorm(out_features),
            nn.Tanh()
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        values, indices = x.max(-2)
        if len(self.selected_layers):
            selected_features = torch.cat(
                [self._gather(features[ind], indices) for ind in self.selected_layers],
                -1)
            values = torch.cat((values.unsqueeze(-1), selected_features), -1).flatten(-2)
        return self.ln_emb(values)

    @staticmethod
    def _gather(features, indices):
        indices = torch.repeat_interleave(indices.unsqueeze(-1), features.size(-1), -1)
        return torch.gather(features, -2, indices)


class PointCloudDecoder(nn.Module):
    def __init__(self, in_features: int, pn_number: int, out_channels: int, layers: tuple, act=nn.ELU):
        super().__init__()

        layers = layers + (out_channels,)
        deconvs = [
            nn.Linear(in_features, pn_number*layers[0]),
            nn.Unflatten(-1, (pn_number, layers[0]))
                   ]
        for i in range(len(layers)-1):
            deconvs.extend([
                act(),
                nn.Linear(layers[i], layers[i+1])
            ])

        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, x):
        return self.deconvs(x)


class PixelsEncoder(nn.Module):
    def __init__(self, in_channels=3, out_features=64, depth=32, act=nn.ELU):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, depth, 3, 2),
            act(),
            nn.Conv2d(depth, depth, 3, 1),
            act(),
            nn.Flatten(),
            nn.Linear(depth*39*39, out_features),
            nn.LayerNorm(out_features),
            nn.Tanh()
            # nn.Conv2d(depth, depth, 3, 1),
            # act(),
            # nn.Conv2d(depth, depth, 3, 1),
            # act(),
            # nn.Flatten(),
            # LayerNormTanhEmbedding(depth*35*35, out_features)
        )

    def forward(self, img):
        prefix_shape = img.shape[:-3]
        img = img.flatten(0, len(prefix_shape)-1)
        img = self.convs(img)
        img = img.reshape(*prefix_shape, -1)
        return img


class PixelsDecoder(nn.Module):
    def __init__(self, in_features, out_channels=3, depth=32, act=nn.ELU):
        super().__init__()
        dim = 39 # 39 - for two conv layers, 35 for 4 layers
        self.out_channels = out_channels
        self.deconvs = nn.Sequential(
            nn.Linear(in_features, depth*dim**2),
            act(),
            nn.Unflatten(-1, (depth, dim, dim)),
            nn.ConvTranspose2d(depth, depth, 3, 1),
            act(),
            # nn.ConvTranspose2d(depth, depth, 3, 1),
            # act(),
            # nn.ConvTranspose2d(depth, depth, 3, 1),
            # act(),
            nn.ConvTranspose2d(depth, out_channels, 3, 2, output_padding=1),
        )

    def forward(self, x):
        prefix_shape = x.shape[:-1]
        x = x.flatten(0, len(prefix_shape)-1)
        img = self.deconvs(x)
        img = img.reshape(*prefix_shape, self.out_channels, 84, 84)
        return img
