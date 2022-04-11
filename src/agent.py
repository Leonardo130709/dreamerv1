import torch
from . import models, utils
from copy import deepcopy
from itertools import chain
from torch.nn.utils import clip_grad_norm_
from pytorch3d.loss import chamfer_distance
import pdb

nn = torch.nn
F = nn.functional
td = torch.distributions
torch.autograd.set_detect_anomaly(True)


# todo check if burn_in and storing hidden_states will be useful


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, callback, config):
        super().__init__()
        self._c = config
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.callback = callback
        self._step = 0
        self._build()

    @torch.no_grad()
    def act(self, obs, prev_state, training):
        if prev_state is None:
            batch_size = obs.size(0)
            latent = self.wm.init_state(batch_size).to(self.device)
            action = torch.zeros((batch_size, self.act_dim), device=self.device)
        else:
            latent, action = prev_state

        obs = self.encoder(obs)
        latent, _ = self.wm.obs_step(obs, action, latent)
        feat = self.wm.get_feat(latent)
        dist = self.actor(feat)
        if training:
            action = dist.sample()
            action = action + self._c.expl_scale * torch.randn_like(action)
        else:
            action = dist.sample([100]).mean(0)
        action = torch.clamp(action, -1, 1)
        return action, (latent, action)

    def learn(self, observations, actions, rewards, states):
        # todo discounts, burn_in, truncate properly, separate_kl, categorical latent
        #   gumbel, target_ntwotrk, reinforce gradients, entropy, layer norm
        self._model_params.requires_grad_(True)
        observations_emb = self.encoder(observations)

        posts, priors = self.wm.observe(observations_emb, actions, states[0])
        feat = self.wm.get_feat(posts)
        obs_pred = self.decoder(feat)
        reward_pred = self.reward_model(feat)
        post_dist, fixed_post_dist, prior_dist, fixed_prior_dist = map(self.wm.get_dist,
                                                                       (posts, posts.detach(), priors, priors.detach()))
        div = self._c.alpha * td.kl_divergence(fixed_post_dist, prior_dist) + \
              (1. - self._c.alpha) * td.kl_divergence(post_dist, fixed_prior_dist)
        div = div.mean()

        if self._c.observe == 'point_cloud':
            obs_loss = chamfer_distance(obs_pred.flatten(0, 1), observations.flatten(0, 1))[0].mean()
        else:
            obs_loss = (obs_pred - observations).pow(2).mean()
        reward_loss = (reward_pred - rewards).pow(2).mean()
        model_loss = obs_loss + reward_loss + self._c.kl_scale * div
        self.callback.add_scalar('train/obs_loss', obs_loss, self._step)
        self.callback.add_scalar('train/reward_loss', reward_loss, self._step)
        self.callback.add_scalar('train/kl_div', div, self._step)
        self.callback.add_scalar('train/model_loss', model_loss, self._step)
        self.callback.add_scalar('train/mean_rewards', rewards.mean(), self._step)
        self.callback.add_scalar('train/model_grads', utils.grads_sum(self._model_params), self._step)

        self.wm_optim.zero_grad()
        # todo: careful with retain_graph
        model_loss.backward(retain_graph=True)
        clip_grad_norm_(self._model_params, self._c.max_grad)
        self._model_params.requires_grad_(False)

        imag_feat, actions = self.imagine_ahead(posts.detach())  # could try with priors
        rewards = self.reward_model(imag_feat)
        values = self._target_critic(imag_feat)
        target_values = utils.gve2(rewards, values, self._c.discount, self._c.disclam)
        values, imag_feat, actions = map(lambda t: t[:-1], (values, imag_feat, actions))
        dist = self.actor(imag_feat)
        log_prob = dist.log_prob(actions).unsqueeze(-1)
        ent = -log_prob
        # reinforce_loss = -log_probs*(target_values - values).detach()
        reinforce_loss = 0
        actor_gain = -self._c.rho * reinforce_loss + (1 - self._c.rho) * target_values + self._c.eta * ent
        discount = self.masked_discount(target_values, 0)
        actor_loss = - torch.mean(discount * actor_gain)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self._c.max_grad)
        self.callback.add_scalar('train/actor_ent', ent.mean(), self._step)
        self.callback.add_scalar('train/actor_loss', actor_gain.mean(), self._step)
        self.callback.add_scalar('train/mean_value', values.mean(), self._step)
        self.callback.add_scalar('train/actor_grads', utils.grads_sum(self.actor), self._step)

        values = self.critic(imag_feat.detach())
        critic_loss = (discount * (values - target_values.detach()).pow(2)).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self._c.max_grad)
        self.callback.add_scalar('train/critic_loss', critic_loss, self._step)
        self.callback.add_scalar('train/critic_grads', utils.grads_sum(self.critic), self._step)

        self.wm_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        utils.soft_update(self._target_critic, self.critic, self._c.critic_polyak)
        self._step += 1

        return model_loss, actor_loss, critic_loss

    def imagine_ahead(self, posts):
        def policy(state):
            feat = self.wm.get_feat(state)
            dist = self.actor(feat)
            return dist.rsample()

        state = posts[:-1].reshape(-1, posts.size(-1))
        # cause last could be terminal state
        states, actions = [], []
        for _ in range(self._c.horizon):
            action = policy(state)
            states.append(state)
            actions.append(action)
            state = self.wm.img_step(action, state)

        states = torch.stack(states)
        actions = torch.stack(actions)
        return self.wm.get_feat(states), actions

    def masked_discount(self, x, mask_size):
        mask = torch.cat([torch.zeros(mask_size, device=self.device),
                          torch.ones(x.size(0) - mask_size, device=self.device)])
        discount = self._c.discount ** torch.arange(0, x.size(0), device=self.device)
        discount = discount * mask
        shape = (x.ndimension() - 1) * (1,)
        return discount.reshape(-1, *shape)

    def _build(self):
        feat = self._c.deter_dim + self._c.stoch_dim
        self.actor = models.NormalActor(feat, self.act_dim, self._c.actor_layers)
        # self.actor = models.OrdinalActor(feat, self.act_dim, utils.Spec(-1, 1, 10))
        self.critic = utils.build_mlp(feat, *self._c.critic_layers, 1)
        self._target_critic = deepcopy(self.critic).requires_grad_(False)
        self.reward_model = utils.build_mlp(feat, *self._c.wm_layers, 1)
        self.wm = models.RSSM(self._c.obs_emb_dim, self.act_dim, self._c.deter_dim, self._c.stoch_dim,
                              self._c.wm_layers)
        if self._c.observe == 'states':
            self.encoder = nn.Sequential(
                nn.Linear(self.obs_dim, self._c.obs_emb_dim),
                nn.ELU(),
                nn.Linear(self._c.obs_emb_dim, self._c.obs_emb_dim),
                nn.Tanh(),
            )
            self.decoder = nn.Linear(feat, self.obs_dim)
        elif self._c.observe == 'pixels':
            self.encoder = models.PixelEncoder(3, self._c.obs_emb_dim)
            self.decoder = models.PixelDecoder(feat, 3)
        elif self._c.observe == 'point_cloud':
            self.encoder = models.PointCloudEncoder(3, self._c.obs_emb_dim,
                                                    layers=self._c.pn_layers, dropout=self._c.pn_dropout)
            self.decoder = models.PointCloudDecoder(feat, 3, self._c.pn_number)
        else:
            raise NotImplementedError

        self._model_params = nn.ParameterList(
            chain(*map(nn.Module.parameters, (self.encoder, self.wm, self.decoder, self.reward_model)))
        )
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self._c.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self._c.critic_lr)
        self.wm_optim = torch.optim.Adam(self._model_params, self._c.wm_lr)
        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
