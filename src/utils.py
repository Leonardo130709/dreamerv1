import torch
from torchaudio.functional import lfilter
import numpy as np
from collections import deque
import random
from torch.utils.data import Dataset
from dm_control import suite
import math
import dataclasses
from abc import ABC
from ruamel.yaml import YAML
nn = torch.nn
F = nn.functional
td = torch.distributions


def build_mlp(*sizes, act=nn.ELU):
    mlp = []
    if len(sizes) == 1:
        sizes = sizes[0]
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(act())
    return nn.Sequential(*mlp[:-1])


def grads_sum(model):
    s = 0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.sum().item()
    return s


def make_env(name, **kwargs):
    domain, task = name.split('_', 1)
    if domain == 'ball':
        # the only task with double underline
        domain = 'ball_in_cup'
        task = 'catch'
    return suite.load(domain, task, **kwargs)


def gve(rewards, values, discount, disclam):
    # seq_len, batch_size, _ = rewards.shape
    # assert values.shape == rewards.shape
    last_val = values[-1].unsqueeze(0)
    device = last_val.device
    inp = rewards[:-1] + discount*(1. - disclam)*values[1:]
    inp = torch.cat([inp, last_val])
    inp = inp.transpose(0, -1)
    target_values = lfilter(inp.flip(-1), torch.tensor([1., - disclam*discount], device=device),
                            torch.tensor([1, 0.], device=device), clamp=False).flip(-1)
    return target_values.transpose(0, -1)[:-1]


def gve2(rewards, values, discount, disclam):
    target_values = []
    last_val = values[-1]
    for r, v in zip(rewards[:-1].flip(0), values[1:].flip(0)):
        last_val = r + discount*(disclam*last_val + (1.-disclam)*v)
        target_values.append(last_val)
    return torch.stack(target_values).flip(0)


def soft_update(target, online, rho):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.copy_(rho * pt + (1. - rho) * po)


def simulate(env, policy, training):
    obs = env.reset()
    done = False
    prev_state = None
    observations, actions, rewards, states = [], [], [], []  # also states for recurrent agent
    while not done:
        if prev_state is not None:
            states.append(prev_state[0].detach().cpu().flatten().numpy())
            action, prev_state = policy(obs, prev_state, training)
        else:
            action, prev_state = policy(obs, prev_state, training)
            states.append(torch.zeros_like(prev_state[0]).detach().cpu().flatten().numpy())
        new_obs, reward, done, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append([reward])
        obs = new_obs

    tr = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        states=states,
    )
    for k, v in tr.items():
        tr[k] = np.stack(v, 0)
    return tr


class TrajectoryBuffer(Dataset):
    def __init__(self, capacity, seq_len):
        self._data = deque(maxlen=capacity)
        self.seq_len = seq_len

    def add(self, trajectory):
        if len(trajectory['actions']) < self.seq_len:
            return
        self._data.append(trajectory)

    def __getitem__(self, idx):
        tr = random.choice(self._data)
        start = random.randint(0, len(tr['actions']) - self.seq_len)
        return {k: v[start:start+self.seq_len] for k, v in tr.items()}

    def __len__(self):
        return 10*len(self._data)


class TanhTransform(td.transforms.TanhTransform):
    lim = 0.9999997

    def _inverse(self, y):
        y = torch.clamp(y, min=-self.lim, max=self.lim)
        return torch.atanh(y)


class PointCloudGenerator:
    def __init__(self, camera_fovy, image_height, image_width, device, cam_matrix=None, rot_matrix=None, position=None):
        super(PointCloudGenerator, self).__init__()

        self.fovy = math.radians(camera_fovy)
        self.height = image_height
        self.width = image_width
        self.device = device

        if rot_matrix != None:
            self.rot_matrix = torch.tensor(rot_matrix, dtype=torch.float32, device=device, requires_grad=False)

        if position != None:
            self.position = torch.tensor(position, dtype=torch.float32, device=device, requires_grad=False)

        if cam_matrix != None:
            self.cam_matrix = cam_matrix
        else:
            self.cam_matrix = self.get_cam_matrix()

        self.fx = self.cam_matrix[0, 0]
        self.fy = self.cam_matrix[1, 1]
        self.cx = self.cam_matrix[0, 2]
        self.cy = self.cam_matrix[1, 2]

        self.uv1 = torch.ones((self.height, self.width, 3), dtype=torch.float32, device=device, requires_grad=False)
        for i in range(self.height):
            for j in range(self.width):
                self.uv1[i][j][0] = ((i + 1) - self.cx) / self.fx
                self.uv1[i][j][1] = ((j + 1) - self.cy) / self.fy
        #print(self.uv1.shape)
        self.uv1 = self.uv1.reshape(-1, 3)
        #   print(self.uv1.shape)

    def get_cam_matrix(self):
        f = self.height / (2 * math.tan(self.fovy / 2))

        return torch.tensor(((f, 0, self.width / 2), (0, f, self.height / 2), (0, 0, 1)),
                            dtype=torch.float32, device=self.device, requires_grad=False)

    def reshape_depth(self, depth):
        depth = torch.tensor(np.flip(depth, axis=0).copy(), dtype=torch.float32, device=self.device,
                             requires_grad=False)
        depth = depth.reshape(-1, 1)
        return torch.cat((depth, depth, depth), dim=-1)

    def get_PC(self, depth):
        depth = self.reshape_depth(depth)
        xyz = depth * self.uv1
        return xyz


@dataclasses.dataclass
class AbstractConfig(ABC):
    def save(self, file_path):
        yaml = YAML()
        with open(file_path, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f)

    def load(self, file_path):
        yaml = YAML()
        with open(file_path) as f:
            config_dict = yaml.load(f)
        return dataclasses.replace(self, **config_dict)
