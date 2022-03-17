from dataclasses import dataclass
from .agent import Agent
from . import utils, wrappers
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from statistics import mean
from dm_control import suite
import datetime
import pdb

torch.autograd.set_detect_anomaly(True)

@dataclass
class Config:
    #task
    discount = .99
    disclam = .95
    horizon = 15
    kl_scale = 1.
    expl_scale = .5
    alpha = .8
    rho = 0.
    eta = 1e-4
    action_repeat = 2

    #model
    actor_layers = 2*[128]
    critic_layers = 3*[200]
    wm_layers = 2*[200]
    deter_dim = 200
    stoch_dim = 32
    emb_dim = 32

    pn_layers = 2
    pn_depth = 32
    pn_number = 800

    #train
    actor_lr = 8e-5
    critic_lr = 8e-5
    wm_lr = 6e-4
    batch_size = 50
    max_grad = 100
    seq_len = 50
    buffer_capacity = 10**2
    training_steps = 300
    target_update = 99

    device = 'cuda'
    encoder = 'PointNet'
    task = 'walker_stand'


class Dreamer:
    def __init__(self, config):
        self.c = config
        obs_dim, act_dim, enc_obs_dim = self._make_env()
        self.callback = SummaryWriter(log_dir=f'./dreamer_logdir/{config.encoder}/{config.task}/{datetime.datetime.now()}')
        self.agent = Agent(enc_obs_dim, act_dim, self.encoder, self.decoder, self.callback, config)
        self.buffer = utils.TrajectoryBuffer(config.buffer_capacity, config.seq_len)

    def learn(self):
        def policy(obs, state, training):
            if not torch.is_tensor(obs):
                obs = torch.from_numpy(obs[None]).to(self.agent.device)
            action, state = self.agent.act(obs, state, training)
            return action.detach().cpu().flatten().numpy(), state
        t = 0
        while True:
            tr = utils.simulate(self.env, policy, True)
            self.buffer.add(tr)

            counter = 0
            dl = DataLoader(self.buffer, batch_size=self.c.batch_size)
            for tr in dl:
                observations, actions, rewards, states = map(lambda k: tr[k].transpose(0, 1).to(self.agent.device),
                                                             ('observations', 'actions', 'rewards', 'states'))
                self.agent.learn(observations, actions, rewards, states)
                counter += 1
                if counter > self.c.training_steps:
                    break

            if t % 10 == 0:
                res = []
                for _ in range(5):
                    tr = utils.simulate(self.env, policy, training=False)
                    res.append(tr['rewards'].sum().item())
                self.callback.add_scalar('eval/reward', mean(res), t*1000)
            t += 1

    def _make_env(self):
        env = utils.make_env(self.c.task)
        if self.c.encoder == 'MLP':
            env = wrappers.dmWrapper(env)
        else:
            env = wrappers.depthMapWrapper(env, device=self.c.device, points=self.c.pn_number)
        self.env = wrappers.FrameSkip(env, self.c.action_repeat)
        sample_obs = self.env.reset()
        act_dim = self.env.action_spec().shape[0]
        self.encoder, self.decoder = utils.build_encoder_decoder(self.c, sample_obs.shape)
        enc_obs = self.encoder(torch.from_numpy(sample_obs[None]))
        return sample_obs.shape[0], act_dim, enc_obs.shape[1]
