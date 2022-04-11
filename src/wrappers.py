import numpy as np
import gym
from gym.spaces import Box
import torch
from .utils import PointCloudGenerator
import ctypes
from dm_control.mujoco.wrapper import MjvOption
from dm_control.suite.wrappers import pixels


class Wrapper:
    """ Partially solves problem with  compatibility"""

    def __init__(self, env):
        self.env = env
        self._observation_space, self._action_space = self._infer_spaces(env)

    def observation(self, timestamp):
        return timestamp

    def reward(self, timestamp):
        return np.float32(timestamp.reward)

    def done(self, timestamp):
        return timestamp.last()

    def step(self, action):
        timestamp = self.env.step(action)
        obs = self.observation(timestamp)
        r = self.reward(timestamp)
        d = self.done(timestamp)
        return obs, r, d, None

    def reset(self):
        return self.observation(self.env.reset())

    @staticmethod
    def _infer_spaces(env):
        lim = float('inf')
        spec = env.action_spec()
        action_space = Box(low=spec.minimum.astype(np.float32), dtype=np.float32,
                           high=spec.maximum.astype(np.float32), shape=spec.shape)
        ar = list(env.observation_spec().values())[0]

        obs_sample = np.concatenate(list(map(lambda ar: ar.generate_value() if ar.shape != () else [1],
                                             env.observation_spec().values())))

        obs_space = Box(low=-lim, high=lim, shape=obs_sample.shape, dtype=np.float32)#ar.dtype)
        return obs_space, action_space

    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def unwrapped(self):
        env = self
        while hasattr(env, 'env'):
            env = env.env
        return env

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class dmWrapper(Wrapper):
    def observation(self, timestamp):
        obs = np.array([])
        for v in timestamp.observation.values():
            if not v.ndim:
                v = v[None]
            obs = np.concatenate((obs, v))
        return obs.astype(np.float32)


class FrameSkip(gym.Wrapper):
    def __init__(self, env, frames_number):
        super().__init__(env)
        self.fn = frames_number

    def step(self, action):
        R = 0
        for i in range(self.fn):
            next_obs, reward, done, info = self.env.step(action)
            R += reward
            if done:
                break
        return np.float32(next_obs), np.float32(R), done, info

    def reset(self):
        return np.float32(self.env.reset())


class depthMapWrapper(Wrapper):

    def __init__(self, env,
                 camera_id=0,
                 height=240,
                 width=320,
                 device='cpu',
                 return_pos=False,
                 points=1000,
                 ):
        super().__init__(env)
        self.env = env
        self.points = points
        self._depth_kwargs = dict(camera_id=camera_id, height=height, width=width,
                                  depth=True, scene_option=self._prepare_scene())
        self.return_pos = return_pos
        self.pcg = PointCloudGenerator(**self.pc_params, device=device)

    def observation(self, timestamp):
        depth = self.env.physics.render(**self._depth_kwargs)
        pc = self.pcg.get_PC(depth)
        pc = self._segmentation(pc)
        if self.return_pos:
            pos = self.env.physics.position()
            return pc, pos
        return pc.detach().cpu().numpy()

    def _segmentation(self, pc):
        dist_thresh = 19
        pc = pc[pc[..., 2] < dist_thresh] # smth like infty cutting
        if self.points:
            amount = pc.size(-2)
            if amount > self.points:
                ind = torch.randperm(amount, device=self.pcg.device)[:self.points]
                pc = torch.index_select(pc, -2, ind)
            elif amount < self.points:
                zeros = torch.zeros(self.points - amount, *pc.shape[1:], device=self.pcg.device)
                pc = torch.cat([pc, zeros])
        return pc

    def _prepare_scene(self):
        scene = MjvOption()
        scene.flags = (ctypes.c_uint8*22)(0)

        return scene

    @property
    def pc_params(self):
        # device
        fovy = self.env.physics.model.cam_fovy[0]
        return dict(
            camera_fovy=fovy,
            image_height=self._depth_kwargs.get('height', 240),
            image_width=self._depth_kwargs.get('width', 320)
        )


class PixelsToGym(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = pixels.Wrapper(self.env, render_kwargs={'camera_id': 0, 'height': 64, 'width': 64})

    def observation(self, timestamp):
        obs = timestamp.observation['pixels']
        obs = np.array(obs) / 255.
        obs = np.array(obs)
        return obs.transpose((2, 1, 0))

    @property
    def observation_space(self):
        # correspondent space have to be extracted from the dm_control API -> gym API
        return Box(low=0., high=1., shape=(64, 64, 3))
