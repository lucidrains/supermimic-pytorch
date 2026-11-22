from __future__ import annotations
from collections import namedtuple

import torch
from torch import Tensor
from torch.nn import Module, Linear, GELU
import torch.nn.functional as F

from einops import rearrange
from ema_pytorch import EMA
from torch_einops_utils import binary_cross_entropy_with_logits, freeze, unfreeze
from torch_einops_utils.shape import size

from x_mlps_pytorch import MLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from supermimic_pytorch.supermimic import (
    State,
    flatten_state,
    flatten_state_with_inverse,
    resolve_action_dim,
    to_one_hot
)

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
        ema_decay = 0.999,
        freeze_autoencoder = True
    ):
        super().__init__()
        assert len(dim_hiddens) > 0

        dim_action, self.num_discrete_actions = resolve_action_dim(dim_action, num_discrete_actions)
        self.dim_action = dim_action
        self.has_discrete_actions = len(self.num_discrete_actions) > 0

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

        dim_in = dim_latent + dim_action + dim_sim_context

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

        # losses

        self.state_loss_weight = state_loss_weight
        self.latent_loss_weight = latent_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.terminal_loss_weight = terminal_loss_weight

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
        return self.state_encoder(flatten_state(state))

    def decode(self, latent):
        return self.state_decoder(latent)

    @torch.no_grad()
    def encode_target(self, state: State):
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
        next_rewards: Tensor | None = None       # (b time ...)
    ):
        num_steps = size(actions, 'b [t] ...')

        state, inverse_state = flatten_state_with_inverse(state)
        next_states = flatten_state(next_states, match = inverse_state)

        latent = self.encode(state)

        # accumulants

        all_preds = []
        all_losses = []
        total_loss = 0.

        # latent rollout

        for time_index in range(num_steps):
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
                next_reward
            )

            total_loss = total_loss + loss

            all_preds.append(pred._replace(state = inverse_state(pred.state)))
            all_losses.append(losses)

            latent = pred.latent # feed predicted latent back in

        return total_loss, all_preds, all_losses

    def forward_from_latent(
        self,
        latent,
        action,
        sim_context: Tensor | None = None,
        next_state: State | None = None,
        next_terminal: Tensor | None = None,
        next_reward: Tensor | None = None,
        inverse_state = None
    ):
        if self.has_discrete_actions and not action.is_floating_point():
            action = to_one_hot(action, self.num_discrete_actions, latent.dtype)

        assert action.ndim == 2, 'use `rollout` for a rollout loss'
        assert self.has_sim_context == exists(sim_context)

        embed = self.transition_mlp(compact(latent, action, sim_context))

        pred_latent = self.to_latent_pred(embed)

        if self.pred_residual_latent:
            pred_latent = pred_latent + latent

        # decode predicted next latent back to state space

        pred_next_state = self.decode(pred_latent)

        # maybe rewards

        pred_reward = None

        if exists(self.to_reward_pred):
            pred_reward = rearrange(self.to_reward_pred(embed), '... 1 -> ...')

        # maybe terminal

        pred_terminal_logit = pred_terminal = None

        if exists(self.to_terminal_pred):
            pred_terminal_logit = rearrange(self.to_terminal_pred(embed), '... 1 -> ...')
            pred_terminal = pred_terminal_logit.sigmoid()

        # maybe flatten next_state

        inverse_next_state = None

        if exists(next_state):
            next_state, inverse_next_state = flatten_state_with_inverse(next_state, match = inverse_state)

        reconstitute = inverse_state or inverse_next_state or identity

        # outputs

        pred = Predictions(reconstitute(pred_next_state), pred_latent, pred_terminal, pred_reward)

        # maybe loss

        need_loss = exists(next_state) or exists(next_terminal) or exists(next_reward)

        if not need_loss:
            return pred

        total_loss = 0.
        state_loss = latent_loss = recon_loss = terminal_loss = reward_loss = None

        if exists(next_state):
            state_loss = F.mse_loss(next_state, pred_next_state)

            target_latent = self.encode_target(next_state)
            latent_loss = F.smooth_l1_loss(pred_latent, target_latent)

            total_loss = total_loss + state_loss * self.state_loss_weight + latent_loss * self.latent_loss_weight

            # e2e recon, keeps the unfrozen autoencoder honest

            if self.has_recon_loss:
                recon_loss = F.mse_loss(next_state, self.decode(self.encode(next_state)))
                total_loss = total_loss + recon_loss * self.recon_loss_weight

        if exists(next_reward):
            assert exists(pred_reward)
            reward_loss = F.mse_loss(next_reward, pred_reward)
            total_loss = total_loss + reward_loss * self.reward_loss_weight

        if exists(next_terminal):
            assert exists(pred_terminal)
            terminal_loss = binary_cross_entropy_with_logits(pred_terminal_logit, next_terminal)
            total_loss = total_loss + terminal_loss * self.terminal_loss_weight

        losses = Losses(state_loss, latent_loss, recon_loss, terminal_loss, reward_loss)

        return total_loss, pred, losses

    def forward(
        self,
        state: State,                           # (b dim_state)
        action,                                 # (b dim_action)
        sim_context: Tensor | None = None,      # (b dim_sim_context)
        next_state: State | None = None,
        next_terminal: Tensor | None = None,
        next_reward: Tensor | None = None
    ):
        state, inverse_state = flatten_state_with_inverse(state)
        latent = self.encode(state)

        return self.forward_from_latent(
            latent,
            action,
            sim_context,
            next_state,
            next_terminal,
            next_reward,
            inverse_state = inverse_state
        )
