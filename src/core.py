from dataclasses import dataclass
from .agent import Agent
from . import utils
import gym
import torch

@dataclass
class Config:
    #task
    discount = .99
    disclam = .95
    horizon = 5
    free_nats = 3.
    kl_scale = 1.
    expl_scale = .3
    alpha = .8

    #model
    actor_layers = [32, 32]
    critic_layers = [32, 32]
    deter_dim = 100
    stoch_dim = 30
    emb_dim = 32

    #train
    actor_lr = 1e-3
    critic_lr = 1e-3
    wm_lr = 1e-3
    batch_size = 20
    max_grad = 100
    burn_in = 5
    seq_len = 50

    device = 'cpu'


class Dreamer:
    def __init__(self, config):
        self.c = config
        obs_dim, act_dim, enc_obs_dim = self._make_env()
        self.callback = None
        self.agent = Agent(enc_obs_dim, act_dim, self.encoder, self.decoder, self.callback, config)

    def learn(self):
        def policy(obs, state, training):
            if not torch.is_tensor(obs):
                obs = torch.from_numpy(obs[None]).to(self.agent.device)
            action, state = self.agent.act(obs, state, training)
            return action.detach().cpu().flatten().numpy(), state

        while True:
            tr = utils.simulate(self.env, policy, True)
            for k, v in tr.items():
                tr[k] = torch.from_numpy(v)
            observations, actions, rewards, states = map(tr.get,
                                                         ('observations', 'actions', 'rewards', 'states'))
            self.agent.learn(observations, actions, rewards, states)
            print(tr['rewards'].sum().item())

    def _make_env(self):
        self.env = gym.make('MountainCarContinuous-v0')
        sample_obs = self.env.reset()
        obs_dim = sample_obs.shape[0]
        act_dim = self.env.action_space.sample().shape[0]
        #self.encoder = utils.layer_norm_emb(obs_dim, self.c.emb_dim)
        from torch import nn
        self.encoder = nn.Identity()
        self.decoder = utils.build_mlp([self.c.deter_dim + self.c.stoch_dim, obs_dim])
        return obs_dim, act_dim, obs_dim
