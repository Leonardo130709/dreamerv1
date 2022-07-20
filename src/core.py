import pathlib
from .agent import Dreamer
from . import utils, wrappers
from .config import Config
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import time

# #TODO: remove after debug
torch.autograd.set_detect_anomaly(True)


class RLAlg:
    def __init__(self, config):
        utils.set_seed(config.seed)
        self.config = config
        self.env = self.make_env(task_kwargs={'random': self.config.seed})
        self.task_path = pathlib.Path(config.logdir)
        self.callback = SummaryWriter(log_dir=config.logdir)
        self.agent = Dreamer(self.env, config, self.callback)
        self.buffer = utils.TrajectoryBuffer(config.buffer_capacity, config.seq_len+config.burn_in)
        self.interactions_count = 0

    def learn(self):
        dur = time.time()
        while self.interactions_count <= self.config.total_steps:
            tr = utils.simulate(self.env, self.policy, True)
            interactions = len(tr['actions'])
            self.interactions_count += interactions
            self.buffer.add(tr)

            dl = DataLoader(
                self.buffer.sample_subset(
                    self.config.spi * interactions // self.config.seq_len
                ),
                batch_size=self.config.batch_size,
                drop_last=True
            )
            for tr in dl:
                observations, actions, rewards, states = map(
                    lambda k: tr[k].transpose(0, 1).to(self.agent.device),
                    ('observations', 'actions', 'rewards', 'states')
                )
                self.agent.learn(observations, actions, rewards, states)

            if self.interactions_count % self.config.eval_freq == 0:
                res = [utils.simulate(self.make_env(), self.policy, training=False)['rewards'].sum()
                       for _ in range(10)]
                self.callback.add_scalar('eval/return_mean', np.mean(res), self.interactions_count)
                self.callback.add_scalar('eval/return_std', np.std(res), self.interactions_count)

                self.save()

        dur = time.time() - dur
        self.callback.add_hparams(
            {k: v for k, v in vars(self.config).items()
             if any(map(lambda t: isinstance(v, t), (int, float, bool)))},
            dict(duration=dur, score=np.mean(res))
        )

    def save(self):
        self.config.save(self.task_path / 'config.yml')
        torch.save({
            'interactions': self.interactions_count,
            'params': self.agent.state_dict(),
            'actor_optim': self.agent.actor_optim.state_dict(),
            'critic_optim': self.agent.critic_optim.state_dict(),
            'wm_optim': self.agent.wm_optim.state_dict()
        }, self.task_path / 'checkpoint')
        # TODO: restore buffer saving
        # with open(self.task_path / 'buffer', 'wb') as buffer:
        #     pickle.dump(self.buffer, buffer)

    @classmethod
    def load(cls, path, **kwargs):
        path = pathlib.Path(path)
        # [f.unlink() for f in path.iterdir() if f.match('*tfevents*')]
        config = Config.load(path / 'config.yml', **kwargs)
        alg = cls(config)

        if (path / 'checkpoint').exists():
            chkp = torch.load(
                path / 'checkpoint',
                map_location=torch.device(config.device if torch.cuda.is_available() else 'cpu')
            )
            with torch.no_grad():
                alg.agent.load_state_dict(chkp['params'], strict=False)
                alg.agent.actor_optim.load_state_dict(chkp['actor_optim'])
                alg.agent.critic_optim.load_state_dict(chkp['critic_optim'])
                alg.agent.wm_optim.load_state_dict(chkp['wm_optim'])

            alg.interactions_count = chkp['interactions']

        # if (path / 'buffer').exists():
        #     with open(path / 'buffer', 'rb') as b:
        #         alg.buffer = pickle.load(b)
        return alg

    def make_env(self, task_kwargs=None, environment_kwargs=None):
        env = utils.make_env(self.config.task, task_kwargs, environment_kwargs)
        if self.config.observe == 'states':
            env = wrappers.StatesWrapper(env)
        elif self.config.observe in wrappers.PixelsWrapper.channels.keys():
            env = wrappers.PixelsWrapper(env, mode=self.config.observe)
        elif self.config.observe == 'point_cloud':
            env = wrappers.PointCloudWrapperV2(
                env,
                pn_number=self.config.pn_number,
                render_kwargs=dict(camera_id=0, height=240, width=320),
                stride=self.config.stride
                )
        else:
            raise NotImplementedError
        env = wrappers.ActionRepeat(env, self.config.action_repeat)
        return env

    def policy(self, obs, state, training):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs[None]).to(self.agent.device)
        action, state = self.agent.act(obs, state, training)
        return action.detach().cpu().flatten().numpy(), state
