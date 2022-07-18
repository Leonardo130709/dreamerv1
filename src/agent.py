import torch
from . import models, utils
from copy import deepcopy
from itertools import chain
from torch.nn.utils import clip_grad_norm_
from pytorch3d.loss import chamfer_distance

nn = torch.nn
F = nn.functional
td = torch.distributions
torch.autograd.set_detect_anomaly(True)


# todo check if burn_in and storing hidden_states will be useful


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
            action = action + self._c.expl_scale * torch.randn_like(action)
        else:
            action = dist.sample([100]).mean(0)
        action = torch.clamp(action, -1, 1)
        return action, (latent, action)

    def learn(self, observations, actions, rewards, states):
        # todo discounts, burn_in, truncate properly, separate_kl, categorical latent
        #   gumbel, target_ntwotrk, reinforce gradients, entropy, layer norm

        import pdb; pdb.set_trace()

        self._model_params.requires_grad_(True)

        wm_loss, posts = self._model_learning(observations, actions, rewards, states)

        #
        # self.wm_optim.zero_grad()
        # model_loss.backward()
        # clip_grad_norm_(self._model_params, self._c.max_grad)
        # self._model_params.requires_grad_(False)



        self.wm_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        with torch.no_grad():
            utils.soft_update(self._target_critic, self.critic, self._c.critic_polyak)
        self._step += 1

        return model_loss, actor_loss, critic_loss

    def _model_learning(self, observations, actions, rewards, states):
        observations_emb = self.encoder(observations)

        posts, priors = self.wm.observe(observations_emb, actions, states[0])
        feat = self.wm.get_feat(posts)
        obs_pred = self.decoder(feat)
        reward_pred = self.reward_model(feat)
        post_dist, fixed_post_dist, prior_dist, fixed_prior_dist =\
            map(self.wm.get_dist,
                (posts, posts.detach(), priors, priors.detach()))
        div = self._c.alpha * td.kl_divergence(fixed_post_dist, prior_dist) + \
              (1. - self._c.alpha) * td.kl_divergence(post_dist, fixed_prior_dist)
        div = div.mean(axis=-1)

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
            self.callback.add_scalar('train/model_loss', model_loss, self._step)
            self.callback.add_scalar('train/mean_rewards', rewards.mean(), self._step)
            self.callback.add_scalar('train/model_grads', utils.grads_sum(self._model_params),
                                     self._step)

        return model_loss, posts

    def _policy_learning_and_improvement(self, states):
        self.wm_
        imag_feat, actions = self.imagine_ahead(states.detach())  # could try with priors
        rewards = self.reward_model(imag_feat)
        values = self._target_critic(imag_feat)
        target_values = utils.gve2(rewards, values, self._c.discount, self._c.disclam)
        assert target_values.requires_grad
        values, imag_feat, actions = map(lambda t: t[:-1], (values, imag_feat, actions))
        with torch.no_grad():
            dist = self.actor(imag_feat)
            log_prob = dist.log_prob(actions)
            ent = -log_prob.mean()

        actor_loss = - target_values  # remove scale from values -> to lr
        actor_loss = self._sequence_discount(actor_loss) * actor_loss

        values = self.critic(imag_feat.detach())
        critic_loss = (values - target_values.detach()).pow(2)
        critic_loss = self._sequence_discount(critic_loss) * critic_loss

        if self._c.debug:
            self.callback.add_scalar('train/actor_ent', ent.detach().mean(), self._step)
            self.callback.add_scalar('train/actor_loss', actor_loss.detach().mean(), self._step)
            self.callback.add_scalar('train/mean_value', values.detach().mean(), self._step)
            self.callback.add_scalar('train/actor_grads', utils.grads_sum(self.actor), self._step)
            self.callback.add_scalar('train/critic_loss', critic_loss.detach(), self._step)
            self.callback.add_scalar('train/critic_grads', utils.grads_sum(self.critic), self._step)

    def imagine_ahead(self, posts):
        def policy(state):
            feat = self.wm.get_feat(state)
            dist = self.actor(feat)
            return dist.rsample()

        # Avoid terminal state.
        state = posts[:-1].reshape(-1, posts.size(-1))
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

    def _sequence_discount(self, x):
        discount = self._c.discount ** torch.arange(x.size(0), device=self.device)
        shape = (x.ndimension() - 1) * (1,)
        return discount.reshape(-1, *shape)

    def _build(self):
        feat = self._c.deter_dim + self._c.stoch_dim

        # RL
        self.actor = models.Actor(feat, self.act_dim, self._c.actor_layers)
        self.critic = utils.build_mlp(feat, *self._c.critic_layers, 1)

        self._target_critic = deepcopy(self.critic).requires_grad_(False)

        # WM
        self.reward_model = utils.build_mlp(feat, *self._c.wm_layers, 1)
        self.wm = models.RSSM(self._c.obs_emb_dim, self.act_dim, self._c.deter_dim,
                              self._c.stoch_dim, self._c.wm_layers)

        if self._c.observe == 'pixels':
            self.encoder = models.PixelsEncoder(3, self._c.obs_emb_dim)
            self.decoder = models.PixelsDecoder(feat, 3)
        elif self._c.observe == 'point_cloud':
            self.encoder = models.PointCloudEncoder(3, self._c.obs_emb_dim,
                                                    layers=self._c.pn_layers)
            self.decoder = models.PointCloudDecoder(feat, 3, self._c.pn_number)
        else:
            raise NotImplementedError

        self._wm_parameters = nn.ParameterList(
            chain(*map(nn.Module.parameters,
                       (self.encoder, self.wm, self.decoder, self.reward_model))
                  )
        )

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self._c.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self._c.critic_lr)
        self.wm_optim = torch.optim.Adam(self._model_params, self._c.wm_lr)
        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
