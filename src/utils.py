import torch
#from torchaudio.functional import lfilter
import numpy as np
from collections import deque
import random
from torch.utils.data import Dataset, Subset
from dm_control import suite
from itertools import chain
nn = torch.nn
F = nn.functional
td = torch.distributions


def build_mlp(*sizes, act=nn.ELU):
    mlp = []
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(act())
    return nn.Sequential(*mlp[:-1])


def grads_sum(model):
    s = 0
    for p in model.parameters():
        if torch.is_tensor(p.grad):
            s += p.grad.pow(2).sum().item()
    return np.sqrt(s)


def make_env(name, task_kwargs=None, environment_kwargs=None):
    domain, task = name.split('_', 1)
    if domain == 'ball':
        domain = 'ball_in_cup'
        task = 'catch'
    return suite.load(domain, task, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs)


def set_seed(seed):
    #TODO: fix seed in dataloaders
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# def gve(rewards, values, discount, disclam):
#     # seq_len, batch_size, _ = rewards.shape
#     # assert values.shape == rewards.shape
#     last_val = values[-1].unsqueeze(0)
#     device = last_val.device
#     inp = rewards[:-1] + discount*(1. - disclam)*values[1:]
#     inp = torch.cat([inp, last_val])
#     inp = inp.transpose(0, -1)
#     target_values = lfilter(inp.flip(-1), torch.tensor([1., - disclam*discount], device=device),
#                             torch.tensor([1, 0.], device=device), clamp=False).flip(-1)
#     return target_values.transpose(0, -1)[:-1]


def gve2(rewards, values, discount, disclam):
    target_values = []
    last_val = values[-1]
    for r, v in zip(rewards[:-1].flip(0), values[1:].flip(0)):
        last_val = r + discount*(disclam*last_val + (1.-disclam)*v)
        target_values.append(last_val)
    return torch.stack(target_values).flip(0)


def soft_update(target, online, rho):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.data.copy_((1. - rho) * pt.data + rho * po.detach())


def simulate(env, policy, training):
    obs = env.reset().observation
    done = False
    prev_state = None
    observations, actions, rewards, dones, states = [], [], [], [], []  # also states for recurrent agent
    while not done:
        if prev_state is not None:
            states.append(prev_state[0].detach().cpu().flatten().numpy())
            action, prev_state = policy(obs, prev_state, training)
        else:
            action, prev_state = policy(obs, prev_state, training)
            states.append(torch.zeros_like(prev_state[0]).detach().cpu().flatten().numpy())
        timestep = env.step(action)
        reward = np.float32(timestep.reward)[None]
        done = timestep.last()
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(np.float32(done)[None])
        obs = timestep.observation

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
        tr = self._data[idx]
        start = random.randint(0, len(tr['actions']) - self.seq_len)
        return {k: v[start:start+self.seq_len] for k, v in tr.items()}

    def __len__(self):
        return len(self._data)

    def sample_subset(self, size):
        idx = np.random.randint(0, len(self._data), size=size)
        return Subset(self, idx)


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        nn.init.zeros_(module.bias)

