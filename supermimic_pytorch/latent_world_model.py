from __future__ import annotations
from collections import namedtuple

import torch
from torch import Tensor
from torch.nn import Module, Linear, GELU
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from einops import rearrange, reduce
from ema_pytorch import EMA
from torch_einops_utils import binary_cross_entropy_with_logits, freeze, unfreeze
from torch_einops_utils.shape import size

from x_mlps_pytorch import MLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from supermimic_pytorch.supermimic import (
    State,
    add_state_noise,
    batch_size_of,
    default,
    flatten_state,
    flatten_state_with_inverse,
    prepend_state,
    reduce_explorative_loss,
    repeat_batch,
    resolve_action_dim,
    resolve_explorative_latent,
    resolve_normalizer,
    rollout_step_weights,
    sample_explorative_latent,
    to_one_hot
)
from supermimic_pytorch.welford_normalizer import WelfordNormalizer

# helpers

def exists(v):
    return v is not None

def identity(t):
    return t

def last(arr):
    return arr[-1]

def compact(*arr):
    return tuple(filter(exists, arr))

# predictions

Predictions = namedtuple('Predictions', ('state', 'latent', 'terminal', 'reward'))

Losses = namedtuple('Losses', ('state', 'latent', 'recon', 'terminal', 'reward'))

# latent world model

class LatentWorldModel(Module):
    def __init__(
        self,
        *,
        state_encoder: Module,
        state_decoder: Module,
        dim_action: int | None = None,
        num_discrete_actions: int | tuple[int, ...] | None = None,
        dim_sim_context = 33,
        dim_latent = 128,
        dim_hiddens = (2048, 2048, 2048),
        activation: Module = GELU(),
        normed_mlp = False,
        pred_residual_latent = True,
        pred_terminal = True,
        pred_reward = False,
        terminal_init_bias = -4.,
        state_loss_weight = 1.,
        latent_loss_weight = 1.,
        recon_loss_weight = 1.,
        reward_loss_weight = 1.,
        terminal_loss_weight = 1.,
        rollout_loss_decay: float = 1.,
        state_noise_scale: float = 0.,
        ema_decay = 0.999,
        freeze_autoencoder = True,
        num_explorative_rollouts: int | None = None,
        dim_explorative_latent: int = 0,
        explorative_latent_scale: float = 0.25,
        antithetic_explorative_latent: bool = True,
        state_normalizer: WelfordNormalizer | bool | None = None
    ):
        super().__init__()
        assert len(dim_hiddens) > 0
        assert 0. < rollout_loss_decay <= 1., '`rollout_loss_decay` must be in (0, 1]'
        assert state_noise_scale >= 0., '`state_noise_scale` must be non-negative'

        # explorative modeling - Gladstone et al. (2026); antithetic pairing - Jia et al. (2025)

        num_explorative_rollouts = default(num_explorative_rollouts, 2 if dim_explorative_latent > 0 else 1)
        assert num_explorative_rollouts >= 1 and dim_explorative_latent >= 0

        self.num_explorative_rollouts = num_explorative_rollouts
        self.dim_explorative_latent = dim_explorative_latent
        self.explorative_latent_scale = explorative_latent_scale
        self.antithetic_explorative_latent = antithetic_explorative_latent

        dim_action, self.num_discrete_actions = resolve_action_dim(dim_action, num_discrete_actions)
        self.dim_action = dim_action
        self.has_discrete_actions = len(self.num_discrete_actions) > 0

        self.state_normalizer = resolve_normalizer(state_normalizer)

        mlp_klass = MLP if not normed_mlp else NormedMLP

        # state <-> latent autoencoder (frozen, gradients still flow for actor bptt)

        self.state_encoder = state_encoder
        self.state_decoder = state_decoder

        self.freeze_autoencoder = freeze_autoencoder

        if freeze_autoencoder:
            freeze(self.state_encoder).eval()
            freeze(self.state_decoder).eval()

        # ema target encoder -> next latent target

        self.target_state_encoder = EMA(
            self.state_encoder,
            include_online_model = False,
            beta = ema_decay,
            update_after_step = 0,
            update_every = 1
        )

        freeze(self.target_state_encoder.ema_model)
        self.target_state_encoder.ema_model.eval()

        # latent transition backbone

        dim_in = dim_latent + dim_action + dim_sim_context + dim_explorative_latent

        self.transition_mlp = mlp_klass(dim_in, *dim_hiddens, activation = activation, activate_last = True)

        # predictions

        dim_embed = last(dim_hiddens)
        self.to_latent_pred = Linear(dim_embed, dim_latent)
        self.to_reward_pred = Linear(dim_embed, 1) if pred_reward else None
        self.to_terminal_pred = Linear(dim_embed, 1) if pred_terminal else None

        self.pred_reward = pred_reward
        self.pred_terminal = pred_terminal

        # zero init residual head -> starts as identity dynamics in latent space

        if pred_residual_latent:
            torch.nn.init.zeros_(self.to_latent_pred.weight)
            torch.nn.init.zeros_(self.to_latent_pred.bias)

        if exists(self.to_terminal_pred):
            torch.nn.init.constant_(self.to_terminal_pred.bias, terminal_init_bias)

        self.has_sim_context = dim_sim_context > 0
        self.pred_residual_latent = pred_residual_latent

        # losses (temporal decay rho^t - Hansen et al. 2024; training noise - Sanchez-Gonzalez et al. 2020)

        self.state_loss_weight = state_loss_weight
        self.latent_loss_weight = latent_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.terminal_loss_weight = terminal_loss_weight
        self.rollout_loss_decay = float(rollout_loss_decay)
        self.state_noise_scale = float(state_noise_scale)

    @property
    def has_recon_loss(self):
        return self.recon_loss_weight > 0 and not self.freeze_autoencoder

    def train(self, mode = True):
        super().train(mode)
        self.target_state_encoder.ema_model.eval()

        if self.freeze_autoencoder:
            self.state_encoder.eval()
            self.state_decoder.eval()

        return self

    def unfreeze_autoencoder(self):
        self.freeze_autoencoder = False
        unfreeze(self.state_encoder).train()
        unfreeze(self.state_decoder).train()
        return self

    def encode(self, state: State):
        if exists(self.state_normalizer):
            state = self.state_normalizer.normalize(state)
        return self.state_encoder(flatten_state(state))

    def decode(self, latent):
        decoded = self.state_decoder(latent)
        return self.state_normalizer.denormalize(decoded) if exists(self.state_normalizer) else decoded

    @torch.no_grad()
    def encode_target(self, state: State):
        if exists(self.state_normalizer):
            state = self.state_normalizer.normalize(state)
        return self.target_state_encoder(flatten_state(state))

    @torch.no_grad()
    def update_ema(self):
        # call after optimizer.step() once the autoencoder is unfrozen

        self.target_state_encoder.update()
        return self

    def rollout(
        self,
        state: State,
        actions,                                 # (b time dim_action)
        next_states: State,                      # (b time dim_state)
        sim_context: Tensor | None = None,       # (b dim_sim_context)
        next_terminals: Tensor | None = None,    # (b time ...)
        next_rewards: Tensor | None = None,      # (b time ...)
        *,
        num_explorative_rollouts: int | None = None,
        explorative_latent_scale: float | None = None,
        rollout_loss_decay: float | None = None,
        state_noise_scale: float | None = None,
        update_normalizer: bool | None = None
    ):
        num_steps = size(actions, 'b [t] ...')
        k = default(num_explorative_rollouts, self.num_explorative_rollouts)
        assert k >= 1

        decay = float(default(rollout_loss_decay, self.rollout_loss_decay))
        noise_scale = float(default(state_noise_scale, self.state_noise_scale))
        step_weights = rollout_step_weights(num_steps, decay)

        orig_batch = batch_size_of(state)

        if exists(self.state_normalizer):
            if default(update_normalizer, self.training):
                self.state_normalizer.update(prepend_state(state, next_states))
            else:
                self.state_normalizer.init_stats(state)

        state, inverse_state = flatten_state_with_inverse(state)
        next_states = flatten_state(next_states, match = inverse_state)

        if self.training:
            state = add_state_noise(state, noise_scale, self.state_normalizer)

        state, actions, next_states, sim_context, next_terminals, next_rewards = repeat_batch(
            (state, actions, next_states, sim_context, next_terminals, next_rewards),
            k
        )

        latent = self.encode(state)

        explorative_latent = None

        if self.dim_explorative_latent > 0:
            explorative_latent = sample_explorative_latent(
                orig_batch,
                k,
                self.dim_explorative_latent,
                antithetic = self.antithetic_explorative_latent,
                scale = default(explorative_latent_scale, self.explorative_latent_scale),
                device = latent.device,
                dtype = latent.dtype
            )

        # accumulants

        all_preds = []
        all_losses = []
        total_loss = latent.new_zeros(orig_batch * k)

        # latent rollout

        for time_index, step_weight in enumerate(step_weights):
            action = actions[:, time_index]
            next_state = next_states[:, time_index]

            next_terminal = next_reward = None

            if exists(next_terminals):
                next_terminal = next_terminals[:, time_index]

            if exists(next_rewards):
                next_reward = next_rewards[:, time_index]

            loss, pred, losses = self.forward_from_latent(
                latent,
                action,
                sim_context,
                next_state,
                next_terminal,
                next_reward,
                explorative_latent = explorative_latent,
                reduce_loss = False,
                update_normalizer = False
            )

            total_loss = total_loss + step_weight * loss

            all_preds.append(pred._replace(state = inverse_state(pred.state)))
            all_losses.append(Losses(*(t.mean() if exists(t) else None for t in losses)))

            latent = pred.latent # feed predicted latent back in

        return reduce_explorative_loss(total_loss, k), all_preds, all_losses

    def forward_from_latent(
        self,
        latent,
        action,
        sim_context: Tensor | None = None,
        next_state: State | None = None,
        next_terminal: Tensor | None = None,
        next_reward: Tensor | None = None,
        inverse_state = None,
        *,
        explorative_latent: Tensor | None = None,
        reduce_loss: bool = True,
        update_normalizer: bool | None = None
    ):
        if self.has_discrete_actions and not action.is_floating_point():
            action = to_one_hot(action, self.num_discrete_actions, latent.dtype)

        assert action.ndim == 2, 'use `rollout` for a rollout loss'
        assert self.has_sim_context == exists(sim_context)

        explorative_latent = resolve_explorative_latent(latent, explorative_latent, self.dim_explorative_latent)

        embed = self.transition_mlp(compact(latent, action, sim_context, explorative_latent))

        pred_latent = self.to_latent_pred(embed)

        if self.pred_residual_latent:
            pred_latent = pred_latent + latent

        inverse_next_state = None

        if exists(next_state):
            next_state_flat, inverse_next_state = flatten_state_with_inverse(next_state, match = inverse_state)
            if exists(self.state_normalizer) and default(update_normalizer, self.training):
                self.state_normalizer.update(next_state)
            next_state = next_state_flat

        # decode predicted next latent back to state space

        pred_norm_next_state = self.state_decoder(pred_latent)
        pred_next_state = self.state_normalizer.denormalize(pred_norm_next_state) if exists(self.state_normalizer) else pred_norm_next_state

        # maybe rewards

        pred_reward = None

        if exists(self.to_reward_pred):
            pred_reward = rearrange(self.to_reward_pred(embed), '... 1 -> ...')

        # maybe terminal

        pred_terminal_logit = pred_terminal = None

        if exists(self.to_terminal_pred):
            pred_terminal_logit = rearrange(self.to_terminal_pred(embed), '... 1 -> ...')
            pred_terminal = pred_terminal_logit.sigmoid()

        reconstitute = inverse_state or inverse_next_state or identity

        # outputs

        pred = Predictions(reconstitute(pred_next_state), pred_latent, pred_terminal, pred_reward)

        # maybe loss

        need_loss = exists(next_state) or exists(next_terminal) or exists(next_reward)

        if not need_loss:
            return pred

        total_loss = latent.new_zeros(size(latent, '[b] d'))
        state_loss = latent_loss = recon_loss = terminal_loss = reward_loss = None

        if exists(next_state):
            target_norm_next_state = self.state_normalizer.normalize(next_state) if exists(self.state_normalizer) else next_state
            state_loss = reduce(F.mse_loss(target_norm_next_state, pred_norm_next_state, reduction = 'none'), 'b d -> b', 'mean')

            target_latent = self.encode_target(next_state)
            latent_loss = reduce(F.smooth_l1_loss(pred_latent, target_latent, reduction = 'none'), 'b d -> b', 'mean')

            total_loss = total_loss + state_loss * self.state_loss_weight + latent_loss * self.latent_loss_weight

            # e2e recon, keeps the unfrozen autoencoder honest

            if self.has_recon_loss:
                recon_loss = reduce(F.mse_loss(target_norm_next_state, self.state_decoder(self.state_encoder(target_norm_next_state)), reduction = 'none'), 'b d -> b', 'mean')
                total_loss = total_loss + recon_loss * self.recon_loss_weight

        if exists(next_reward):
            assert exists(pred_reward)
            reward_loss = F.mse_loss(next_reward, pred_reward, reduction = 'none')
            total_loss = total_loss + reward_loss * self.reward_loss_weight

        if exists(next_terminal):
            assert exists(pred_terminal)
            terminal_loss = binary_cross_entropy_with_logits(pred_terminal_logit, next_terminal, reduction = 'none')
            total_loss = total_loss + terminal_loss * self.terminal_loss_weight

        losses = Losses(state_loss, latent_loss, recon_loss, terminal_loss, reward_loss)

        if reduce_loss:
            total_loss = total_loss.mean()
            losses = Losses(*(t.mean() if exists(t) else None for t in losses))

        return total_loss, pred, losses

    def forward(
        self,
        state: State,                           # (b dim_state)
        action,                                 # (b dim_action)
        sim_context: Tensor | None = None,      # (b dim_sim_context)
        next_state: State | None = None,
        next_terminal: Tensor | None = None,
        next_reward: Tensor | None = None,
        *,
        explorative_latent: Tensor | None = None,
        reduce_loss: bool = True,
        update_normalizer: bool | None = None
    ):
        flat_state, inverse_state = flatten_state_with_inverse(state)

        if exists(self.state_normalizer):
            if default(update_normalizer, self.training and exists(next_state)):
                real_states = tree_map(lambda a, b: torch.stack((a, b), dim = 1), state, next_state) if exists(next_state) else state
                self.state_normalizer.update(real_states)
            else:
                self.state_normalizer.init_stats(state)

        latent = self.encode(flat_state)

        return self.forward_from_latent(
            latent,
            action,
            sim_context,
            next_state,
            next_terminal,
            next_reward,
            inverse_state = inverse_state,
            explorative_latent = explorative_latent,
            reduce_loss = reduce_loss,
            update_normalizer = False
        )
