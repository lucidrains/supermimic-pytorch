from __future__ import annotations
from collections import namedtuple

import torch
from torch import Tensor
from torch.nn import Module, Linear
import torch.nn.functional as F

# ein equations

from einops import rearrange
from torch_einops_utils import binary_cross_entropy_with_logits, frozen
from torch_einops_utils.shape import size, shape

# mlps

from x_mlps_pytorch import MLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def last(arr):
    return arr[-1]

def compact(*arr):
    return tuple(filter(exists, arr))

# world model

Predictions = namedtuple('Predictions', ('state', 'terminal', 'reward'))

Losses = namedtuple('Losses', ('state', 'terminal', 'reward'))

class WorldModel(Module):
    def __init__(
        self,
        *,
        dim_state = 71,
        dim_action = 29,
        dim_sim_context = 33,
        dim_hiddens: tuple[int, ...] = (2048, 2048, 2048),
        normed_mlp = False,
        pred_residual_state = True,
        pred_terminal = True,
        pred_reward = False,
        state_loss_weight = 1.,
        reward_loss_weight = 1.,
        terminal_loss_weight = 1.
    ):
        super().__init__()
        assert len(dim_hiddens) > 0

        mlp_klass = MLP if not normed_mlp else NormedMLP

        dim_in = dim_state + dim_action + dim_sim_context

        # backbone

        self.transition_mlp = mlp_klass(dim_in, *dim_hiddens)

        # predictions

        dim_embed = last(dim_hiddens)
        self.to_state_pred = Linear(dim_embed, dim_state, bias = False)
        self.to_reward_pred = Linear(dim_embed, 1, bias = False) if pred_reward else None
        self.to_terminal_pred = Linear(dim_embed, 1, bias = False) if pred_terminal else None

        self.pred_reward = pred_reward
        self.pred_terminal = pred_terminal

        # zero init residual head -> starts as identity dynamics

        if pred_residual_state:
            torch.nn.init.zeros_(self.to_state_pred.weight)

        self.has_sim_context = dim_sim_context > 0
        self.pred_residual_state = pred_residual_state

        # losses

        self.state_loss_weight = state_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.terminal_loss_weight = terminal_loss_weight

    def forward(
        self,
        state,                                  # (b dim_state)
        actions,                                # (b dim_action)
        sim_context: Tensor | None = None,      # (b dim_sim_context)
        next_state: Tensor | None = None,
        next_terminal: Tensor | None = None,
        next_reward: Tensor | None = None
    ):
        assert self.has_sim_context == exists(sim_context)

        # world model is just a forward dynamics model f(s, a) -> s'
        # with simulation context (domain randomization presumably)

        embed = self.transition_mlp(compact(state, actions, sim_context))

        # whether predicting residual state or not

        pred_state = self.to_state_pred(embed)

        pred_next_state = pred_state

        if self.pred_residual_state:
            pred_next_state = pred_next_state + state

        # maybe rewards

        pred_reward = None

        if exists(self.to_reward_pred):
            pred_reward = self.to_reward_pred(embed)
            pred_reward = rearrange(pred_reward, '... 1 -> ...')

        # maybe terminal

        pred_terminal = None

        if exists(self.to_terminal_pred):
            pred_terminal = self.to_terminal_pred(embed)
            pred_terminal = rearrange(pred_terminal, '... 1 -> ...')

        # outputs

        pred = Predictions(pred_next_state, pred_terminal, pred_reward)

        # maybe loss

        need_loss = exists(next_state) or exists(next_terminal) or exists(next_reward)

        if not need_loss:
            return pred

        total_loss = 0.
        state_loss = reward_loss = terminal_loss = None

        if exists(next_state):
            state_loss = F.mse_loss(next_state, pred_next_state)
            total_loss = total_loss + state_loss * self.state_loss_weight

        if exists(next_reward):
            assert exists(pred_reward)
            reward_loss = F.mse_loss(next_reward, pred_reward)
            total_loss = total_loss + reward_loss * self.reward_loss_weight

        if exists(next_terminal):
            assert exists(pred_terminal)
            terminal_loss = binary_cross_entropy_with_logits(pred_terminal, next_terminal)
            total_loss = total_loss + terminal_loss * self.terminal_loss_weight

        losses = Losses(state_loss, terminal_loss, reward_loss)

        return total_loss, pred, losses

# main class

Rollout = namedtuple('Rollout', ('states', 'actions', 'terminals', 'rewards'))

class SuperMimic(Module):
    def __init__(
        self,
        *,
        policy: MLP | Module,
        world_model: WorldModel | Module
    ):
        super().__init__()

        self.policy = policy
        self.world_model = world_model

    def forward(
        self,
        initial_state,
        sim_context: Tensor | None = None,
        num_rollout_steps = 16
    ):
        policy, world_model = self.policy, self.world_model

        pred_reward, pred_terminal = self.world_model.pred_reward, self.world_model.pred_terminal

        # rollout policy within the world model

        states = [initial_state]
        actions = []
        rewards = []
        terminals = []

        state = initial_state

        with frozen(world_model, eval = True):

            for _ in range(num_rollout_steps):
                action = policy(state)

                pred = world_model(state, action, sim_context)

                next_state = pred.state

                states.append(next_state)
                actions.append(action)
                rewards.append(pred.reward)
                terminals.append(pred.terminal)

                state = next_state

        # return all intermediates

        states = rearrange(states, 't b ... -> b t ...')
        actions = rearrange(actions, 't b ... -> b t ...')

        rewards = rearrange(rewards, 't b -> b t') if pred_reward else None
        terminals = rearrange(terminals, 't b -> b t') if pred_terminal else None

        return Rollout(states, actions, terminals, rewards)
