import torch
from . import models, utils
from copy import deepcopy
from itertools import chain
from torch.nn.utils import clip_grad_norm_
import pdb
nn = torch.nn
F = nn.functional
td = torch.distributions


# todo check if burn_in and storing hidden_states will be useful

class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, encoder, decoder, callback,
                 config):
        super().__init__()
        self.c = config
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.encoder = encoder
        self.decoder = decoder or nn.Linear(self.c.deter_dim + self.c.stoch_dim, obs_dim)
        self.callback = callback
        self._build()

    def act(self, obs, prev_state, training):
        if prev_state is None:
            latent = self.wm.init_state(obs.size(0))
            action = torch.zeros((obs.size(0), self.act_dim))
        else:
            latent, action = prev_state

        obs = self.encoder(obs)
        latent, _ = self.wm.obs_step(obs, action, latent)
        feat = self.wm.get_feat(latent)
        dist = self.actor(feat)
        action = dist.rsample()

        if training:
            action = action + self.c.expl_scale*torch.randn_like(action)

        action = torch.clamp(action, -1, 1)
        return action, (latent, action)

    def learn(self, observations, actions, rewards, states):
        #introduce discounts, burn_in, truncate last value
        # todo discounts, burn_in, truncate properly, separate_kl, categorical latent
        self._model_params.requires_grad_(True)
        observations = self.encoder(observations)
        next_observations = observations.roll(-1, 0)
        next_observations, actions, rewards = map(lambda t: t[:-1], (next_observations, actions, rewards))
        posts, priors = self.wm.observe(next_observations, actions, states[0])
        feat = self.wm.get_feat(posts)
        obs_pred = self.decoder(feat)
        reward_pred = self.reward_model(feat)
        prior_dist = self.wm.get_dist(priors)
        post_dist = self.wm.get_dist(posts)
        fixed_prior_dist = self.wm.get_dist(priors.detach())
        fixed_post_dist = self.wm.get_dist(posts.detach())
        div = self.c.alpha*td.kl_divergence(fixed_post_dist, prior_dist) + \
              (1. - self.c.alpha)*td.kl_divergence(post_dist, fixed_prior_dist)
        div = div.mean()
        div = torch.maximum(div, torch.full_like(div, self.c.free_nats))
        model_loss = (obs_pred - next_observations).pow(2).mean() + (reward_pred - rewards).pow(2).mean() + \
                     self.c.kl_scale*div

        self._wm_optim.zero_grad()
        model_loss.backward()
        clip_grad_norm_(self._model_params, self.c.max_grad)
        self._model_params.requires_grad_(False)

        imag_feat = self.imagine_ahead(posts.detach())
        rewards = self.reward_model(imag_feat)
        values = self.critic(imag_feat)
        #pdb.set_trace()
        target_values = utils.gve2(rewards, values, self.c.discount, self.c.disclam)
        discount = self.masked_discount(target_values, 0)
        actor_loss = - (discount*target_values).mean()
        critic_loss = (discount*(values[:-1] - target_values.detach()).pow(2)).mean()

        self._actor_optim.zero_grad()
        self._critic_optim.zero_grad()
        actor_loss.backward(retain_graph=True) # careful values reused twice
        critic_loss.backward()

        self._wm_optim.step()
        self._actor_optim.step()
        self._critic_optim.step()
        return model_loss, actor_loss, critic_loss

    def imagine_ahead(self, posts):
        def policy(state):
            feat = self.wm.get_feat(state)
            dist = self.actor(feat)
            return dist.rsample()

        state = posts[:-1].reshape(-1, posts.size(-1))
        states = []
        for _ in range(self.c.horizon):
            action = policy(state)
            state = self.wm.img_step(action, state)
            states.append(state)

        states = torch.stack(states)
        return self.wm.get_feat(states)

    def masked_discount(self, x, mask_size):
        mask = torch.cat([torch.zeros(mask_size, device=self.device),
                          torch.ones(x.size(0) - mask_size, device=self.device)])
        discount = self.c.discount**torch.arange(0, x.size(0), device=self.device)
        discount = discount*mask
        shape = (x.ndimension()-1)*(1,)
        return discount.view(-1, *shape)

    def _build(self):
        feat = self.c.deter_dim + self.c.stoch_dim
        self.actor = models.NormalActor(feat, self.act_dim, self.c.actor_layers)
        self.critic = utils.build_mlp([feat] + self.c.critic_layers + [1])
        self.reward_model = utils.build_mlp([feat] + self.c.critic_layers + [1])
        self.wm = models.RSSM(self.obs_dim, self.act_dim, self.c.deter_dim, self.c.stoch_dim)
        self._target_actor, self._target_critic = map(lambda m: deepcopy(m).requires_grad_(False),
                                                      (self.actor, self.critic))
        self._model_params = nn.ParameterList(
            chain(*map(nn.Module.parameters, (self.encoder, self.wm, self.decoder, self.reward_model)))
        )
        self._actor_optim = torch.optim.Adam(self.actor.parameters(), self.c.actor_lr)
        self._critic_optim = torch.optim.Adam(self.critic.parameters(), self.c.critic_lr)
        self._wm_optim = torch.optim.Adam(self._model_params, self.c.wm_lr)
        self.device = torch.device(self.c.device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
