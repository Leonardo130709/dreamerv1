import torch
from . import models, utils
from copy import deepcopy
from itertools import chain
from torch.nn.utils import clip_grad_norm_
from pytorch3d.loss import chamfer_distance

nn = torch.nn
F = nn.functional
td = torch.distributions


#TODO: while continuous control may not require discounts,
# it is useful to consider discounts in the alg.
class Dreamer(nn.Module):
    def __init__(self, env, config, callback):
        super().__init__()
        self._c = config
        self.observation_spec = env.observation_spec()
        self.act_dim = env.action_spec().shape[0]
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
        else:
            action = torch.tanh(dist.base_dist.mean)
        return action, (latent, action)

    def learn(self, observations, actions, rewards, states):
        # todo truncate properly, categorical latent
        #   gumbel, reinforce gradients, entropy, layer norm

        init_state = states[0]
        if self._c.burn_in > 0:
            with torch.no_grad():
                obs_enc = self.encoder(observations[:self._c.burn_in])
                posts, priors = self.wm.observe(obs_enc, actions[:self._c.burn_in], init_state)
                init_state = posts[-1]
                observations, actions, rewards = map(lambda t: t[self._c.burn_in:],
                                                     (observations, actions, rewards))

        self._wm_parameters.requires_grad_(True)
        wm_loss, posts = self._model_learning(observations, actions, rewards, init_state)

        self.wm_optim.zero_grad()
        wm_loss.backward()
        clip_grad_norm_(self._wm_parameters, self._c.max_grad)
        self.wm_optim.step()
        self._wm_parameters.requires_grad_(False)

        actor_loss, critic_loss = self._policy_learning_and_improvement(posts.detach())

        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self._c.max_grad)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self._c.max_grad)
        self.critic_optim.step()

        with torch.no_grad():
            utils.soft_update(self._target_critic, self.critic, self._c.critic_polyak)
        self._step += 1

        if self._c.debug:
            self.callback.add_scalar('train/wm_grads', utils.grads_sum(self._wm_parameters), self._step)
            self.callback.add_scalar('train/critic_grads', utils.grads_sum(self.critic), self._step)
            self.callback.add_scalar('train/actor_grads', utils.grads_sum(self.actor), self._step)

    def _model_learning(self, observations, actions, rewards, init_state):
        obs_emb = self.encoder(observations)
        posts, priors = self.wm.observe(obs_emb, actions, init_state)
        feat = self.wm.get_feat(posts)
        obs_pred = self.decoder(feat)
        reward_pred = self.reward_model(feat)

        post_dist, fixed_post_dist, prior_dist, fixed_prior_dist =\
            map(self.wm.get_dist,
                (posts, posts.detach(), priors, priors.detach())
                )
        div = self._c.alpha * td.kl_divergence(fixed_post_dist, prior_dist) + \
              (1. - self._c.alpha) * td.kl_divergence(post_dist, fixed_prior_dist)
        div = div.mean()

        if self._c.observe == 'point_cloud':
            obs_loss = chamfer_distance(obs_pred.flatten(0, 1), observations.flatten(0, 1))[
                0].mean()
        else:
            obs_loss = (obs_pred - observations).pow(2).mean()

        reward_loss = (reward_pred - rewards).pow(2).mean()
        model_loss = obs_loss + reward_loss + self._c.kl_scale * div

        if self._c.debug:
            self.callback.add_scalar('train/obs_loss', obs_loss, self._step)
            self.callback.add_scalar('train/reward_loss', reward_loss, self._step)
            self.callback.add_scalar('train/kl_div', div, self._step)
            self.callback.add_scalar('train/mean_rewards', rewards.mean(), self._step)

        return model_loss, posts

    def _policy_learning_and_improvement(self, states):
        assert not states.requires_grad
        # could try with priors instead of posts
        imag_feat, actions = self._imagine_ahead(states)

        rewards = self.reward_model(imag_feat)
        values = self._target_critic(imag_feat)
        target_values = utils.gve2(rewards, values, self._c.discount, self._c.disclam)
        assert target_values.requires_grad
        values, imag_feat, actions = map(lambda t: t[:-1], (values, imag_feat, actions))

        dist = self.actor(imag_feat.detach())
        log_prob = dist.log_prob(actions.detach())

        # remove scale from values -> to lr only.
        # normalized_values = (target_values - target_values.deatch().mean()) / target_values.deatch().std()

        actor_loss = - target_values + self._c.entropy_coef * log_prob.unsqueeze(-1)
        discount = self._sequence_discount(actor_loss)
        actor_loss = torch.mean(discount * actor_loss)
        values = self.critic(imag_feat.detach()) # couple of unnecessary stop_grads: remove them
        critic_loss = (values - target_values.detach()).pow(2)
        critic_loss = torch.mean(discount * critic_loss)

        if self._c.debug:
            self.callback.add_scalar('train/actor_ent', (-log_prob).detach().mean(), self._step)
            self.callback.add_scalar('train/actor_loss', actor_loss.detach().mean(), self._step)
            self.callback.add_scalar('train/mean_value', values.detach().mean(), self._step)
            self.callback.add_scalar('train/critic_loss', critic_loss.detach(), self._step)

        return actor_loss, critic_loss

    def _imagine_ahead(self, posts):

        def policy(state):
            feat = self.wm.get_feat(state)
            dist = self.actor(feat.detach())
            return dist.rsample()

        # Imagine from the every state.
        # Avoid terminal state. Not used in continuous control.
        state = posts[:-1].reshape(-1, posts.size(-1))
        # First step also should be avoided since it comes from buffer
        states, actions = [], []
        for _ in range(self._c.horizon):
            states.append(state)
            action = policy(state)
            actions.append(action)
            state = self.wm.img_step(action, state)

        #first state is from buffer, it is removed in dreamer v2
        states = torch.stack(states)
        actions = torch.stack(actions)
        return self.wm.get_feat(states), actions

    def _sequence_discount(self, x):
        discount = self._c.discount ** torch.arange(x.size(0), device=self.device)
        shape = (x.ndimension() - 1) * (1,)
        return discount.reshape(-1, *shape)

    def _build(self):
        feat_dim = self._c.deter_dim + self._c.stoch_dim

        # RL
        self.actor = models.Actor(feat_dim,
                                  self.act_dim,
                                  layers=self._c.actor_layers,
                                  mean_scale=1.,
                                  init_std=1.)
        self.critic = utils.build_mlp(feat_dim, *self._c.critic_layers, 1)

        self._target_critic = deepcopy(self.critic).requires_grad_(False)

        # WM
        self.reward_model = utils.build_mlp(feat_dim, *self._c.wm_layers, 1)
        self.wm = models.RSSM(self._c.obs_emb_dim, self.act_dim, self._c.deter_dim,
                              self._c.stoch_dim, self._c.wm_layers)

        if self._c.observe == "states":
            obs_dim = self.observation_spec.shape[0]
            self.encoder = nn.Linear(obs_dim, self._c.obs_emb_dim)
            self.decoder = nn.Linear(feat_dim, obs_dim)
        elif self._c.observe == 'pixels':
            _, _, channels = self.observation_spec.shape
            self.encoder = models.PixelsEncoder(channels, self._c.obs_emb_dim)
            self.decoder = models.PixelsDecoder(feat_dim, channels)
        elif self._c.observe == 'point_cloud':
            pn_number, channels = self.observation_spec.shape
            self.encoder = models.PointCloudEncoder(channels,
                                                    self._c.obs_emb_dim,
                                                    layers=self._c.pn_layers,
                                                    features_from_layers=(),
                                                    )
            self.decoder = models.PointCloudDecoder(feat_dim, self._c.pn_number, channels,
                                                    layers=self._c.pn_layers)
        else:
            raise NotImplementedError

        self._wm_parameters = nn.ParameterList(
            chain(*map(nn.Module.parameters,
                       (self.encoder, self.wm, self.decoder, self.reward_model))
                  )
        )

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self._c.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self._c.critic_lr)
        self.wm_optim = torch.optim.Adam(self._wm_parameters, self._c.wm_lr)

        self.apply(utils.weight_init)

        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
