import pathlib
from .agent import Dreamer
from . import utils, wrappers
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

torch.autograd.set_detect_anomaly(True)

class RLAlg:
    def __init__(self, config):
        self.config = config
        obs_dim, act_dim = self._make_env()
        self._task_path = pathlib.Path('.').joinpath(f'./{config.logdir}/{config.task}/{config.observe}')
        self.callback = SummaryWriter(log_dir=self._task_path)
        self.agent = Agent(obs_dim, act_dim, self.callback, config)
        self.buffer = utils.TrajectoryBuffer(config.buffer_capacity, config.seq_len)
        self.interactions_count = 0

    def learn(self):
        self.config.save(self._task_path / 'config')

        def policy(obs, state, training):
            if not torch.is_tensor(obs):
                obs = torch.from_numpy(obs[None]).to(self.agent.device)
            action, state = self.agent.act(obs, state, training)
            return action.detach().cpu().flatten().numpy(), state
        
        while True:
            tr = utils.simulate(self.env, policy, True)
            self.interactions_count += 1000
            self.buffer.add(tr)

            dl = DataLoader(self.buffer, batch_size=self.config.batch_size)
            for i, tr in enumerate(dl):
                observations, actions, rewards, states = map(lambda k: tr[k].transpose(0, 1).to(self.agent.device),
                                                             ('observations', 'actions', 'rewards', 'states'))
                self.agent.learn(observations, actions, rewards, states)
                if i > self.config.training_steps:
                    break

            if self.interactions_count % self.config.eval_freq == 0:
                res = [utils.simulate(self.env, policy, training=False)['rewards'].sum() for _ in range(5)]
                self.callback.add_scalar('eval/reward', np.mean(res), self.interactions_count)
                self.callback.add_scalar('eval/std', np.std(res), self.interactions_count)

    def save(self):
        pass

    def load(self, path):
        pass

    def _make_env(self):
        env = utils.make_env(self.config.task)
        if self.config.observe == 'states':
            env = wrappers.dmWrapper(env)
        elif self.config.observe == 'pixels':
            env = wrappers.PixelsToGym(env)
        elif self.config.observe == 'point_cloud':
            env = wrappers.depthMapWrapper(env, device=self.config.device, points=self.config.pn_number, camera_id=0)
        else:
            raise NotImplementedError
        self.env = wrappers.FrameSkip(env, self.config.action_repeat)
        act_dim = self.env.action_space.shape[0]
        obs_dim = self.env.observation_space.shape[0]
        return obs_dim, act_dim
