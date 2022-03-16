import torch
from torchaudio.functional import lfilter
import numpy as np
nn = torch.nn
F = nn.functional
td = torch.distributions


def build_mlp(sizes, act=nn.ELU):
    mlp = []
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(act())
    return nn.Sequential(*mlp[:-1])


def layer_norm_emb(in_features, out_features):
    raise NotImplementedError
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(), #specify dimension
        nn.Tanh())


def safe_softplus(x):
    x = torch.maximum(x, torch.full_like(x, -18.))
    return F.softplus(x) + 1e-8


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
    for r, v in zip(rewards[:-1].flip(-1), values[1:].flip(-1)):
        last_val = r + discount*(disclam*last_val + (1-disclam)*v)
        target_values.append(last_val)

    target_values = torch.stack(target_values)
    return target_values.flip(-1)


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
        tr[k] = np.stack(v, 0)[:, None, :]
    return tr
