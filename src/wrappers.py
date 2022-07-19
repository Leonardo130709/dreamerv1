import collections
import numpy as np
import dm_env
from dm_control.suite.wrappers import pixels


class Wrapper(dm_env.Environment):
    """This allows to modify attributes which agent observes and to pack it back."""
    def __init__(self, env: dm_env.Environment):
        self.env = env

    @staticmethod
    def observation(timestep: dm_env.TimeStep):
        return timestep.observation

    @staticmethod
    def reward(timestep: dm_env.TimeStep) -> float:
        return timestep.reward

    def step(self, action) -> dm_env.TimeStep:
        timestep = self.env.step(action)
        return self._wrap_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        return self._wrap_timestep(self.env.reset())

    def _wrap_timestep(self, timestep) -> dm_env.TimeStep:
        return timestep._replace(
            reward=self.reward(timestep),
            observation=self.observation(timestep)
        )

    def action_spec(self) -> dm_env.specs.Array:
        return self.env.action_spec()

    def observation_spec(self) -> dm_env.specs.Array:
        return self.env.observation_spec()

    @property
    def physics(self):
        return self.env.physics


class StatesWrapper(Wrapper):
    # Use dm_control flat_observation environment kwarg instead.
    """ Converts OrderedDict obs to 1-dim np.ndarray[np.float32]. """
    def __init__(self, env):
        super().__init__(env)
        self._observation_spec = self._infer_obs_specs(env)

    def observation(self, timestamp):
        obs = []
        for v in timestamp.observation.values():
            if v.ndim == 0:
                v = v[None]
            obs.append(v.flatten())
        obs = np.concatenate(obs)
        return obs.astype(np.float32)

    @staticmethod
    def _infer_obs_specs(env) -> dm_env.specs.Array:
        dim = sum((np.prod(ar.shape) for ar in env.observation_spec().values()))
        return dm_env.specs.Array(shape=(dim,), dtype=np.float32, name='states')

    def observation_spec(self):
        return self._observation_spec


class ActionRepeat(Wrapper):
    """Repeat the same action multiple times."""
    def __init__(self, env, frames_number: int):
        super().__init__(env)
        self.fn = frames_number

    def step(self, action):
        rew_sum = 0.
        for _ in range(self.fn):
            timestep = self.env.step(action)
            rew_sum += timestep.reward
            if timestep.last():
                break
        return timestep._replace(reward=rew_sum)


class PointCloudWrapperV2(Wrapper):
    def __init__(self,
                 env,
                 pn_number: int = 1000,
                 render_kwargs=None,
                 append_rgb=False,
                 stride: int = -1
                 ):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=240, width=320)
        assert {'camera_id', 'height', 'width'}.issubset(self.render_kwargs.keys())

        self._grid = 1. + np.mgrid[:self.render_kwargs['height'], :self.render_kwargs['width']]

        self.stride = stride
        self.pn_number = pn_number
        self.append_rgb = append_rgb
        self._selected_geoms = np.array(self._segment_by_name(
            env.physics, ('ground', 'wall', 'floor'), **self.render_kwargs
        ))

    def observation(self, timestep):
        depth = self.env.physics.render(depth=True, **self.render_kwargs)
        pcd = self._point_cloud_from_depth(depth)
        mask = self._mask(pcd)

        if self.append_rgb:
            rgb = self._get_colours()
            pcd = np.concatenate((pcd, rgb), axis=1)

        pcd = self._downsampling(pcd[mask])
        return self._to_fixed_number(pcd).astype(np.float32)

    def _point_cloud_from_depth(self, depth):
        f_inv, cx, cy = self._inverse_intrinsic_matrix_params()
        x, y = (depth * self._grid)
        x = (x - cx) * f_inv
        y = (y - cy) * f_inv

        pc = np.stack((x, y, depth), axis=-1)
        return pc.reshape(-1, 3)
        # rot_mat = self.env.physics.data.cam_xmat[self.render_kwargs['camera_id']].reshape(3, 3)
        # return np.einsum('ij, hwi->hwj', rot_mat, pc).reshape(-1, 3)

    def _to_fixed_number(self, pc):
        n = pc.shape[0]
        if n == 0:
            pc = np.zeros((self.pn_number, 3))
        elif n <= self.pn_number:
            pc = np.pad(pc, ((0, self.pn_number - n), (0, 0)), mode='edge')
        else:
            pc = np.random.permutation(pc)[:self.pn_number]
        return pc

    def _inverse_intrinsic_matrix_params(self):
        height = self.render_kwargs['height']
        cx = (height - 1) / 2.
        cy = (self.render_kwargs['width'] - 1) / 2.
        fov = self.env.physics.model.cam_fovy[self.render_kwargs['camera_id']]
        f_inv = 2 * np.tan(np.deg2rad(fov) / 2.) / height
        return f_inv, cx, cy

    def _mask(self, point_cloud):
        seg = self.env.physics.render(segmentation=True, **self.render_kwargs)
        segmentation = np.isin(seg[..., 0].flatten(), self._selected_geoms)
        truncate = point_cloud[..., 2] < 10.
        return np.logical_and(segmentation, truncate)

    def observation_spec(self):
        return dm_env.specs.Array(shape=(self.pn_number, 3 + 3*self.append_rgb),
                                  dtype=np.float32,
                                  name='point_cloud' + '+rgb'*self.append_rgb)

    @staticmethod
    def _segment_by_name(physics, bad_geoms_names, **render_kwargs):
        geom_ids = physics.render(segmentation=True, **render_kwargs)[..., 0]

        def _predicate(geom_id):
            if geom_id == -1:  # infinity
                return False
            return all(
                map(
                    lambda name: name not in physics.model.id2name(geom_id, 'geom'),
                    bad_geoms_names
                )
            )

        return list(filter(_predicate, np.unique(geom_ids).tolist()))

    def _downsampling(self, pcd):
        if self.stride < 0:
            adaptive_stride = pcd.shape[0] // self.pn_number
            return pcd[::max(adaptive_stride, 1)]
        else:
            return pcd[::self.stride]

    def _get_colours(self):
        rgb = self.env.physics.render(**self.render_kwargs).reshape(3, -1).astype(np.float32)
        rgb /= 255.
        return rgb.T


class PixelsToGym(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = pixels.Wrapper(self.env, render_kwargs={'camera_id': 0, 'height': 64, 'width': 64})

    def observation(self, timestamp):
        obs = timestamp.observation['pixels']
        obs = np.array(obs) / 255.
        obs = np.array(obs)
        return obs.transpose((2, 1, 0))

    def observation_spec(self) -> dm_env.specs.Array:
        spec = self.env.observation_spec()
        return spec.replace_(shape=spec.shape.transpose((2, 1, 0)))
