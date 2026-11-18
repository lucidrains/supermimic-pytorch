from __future__ import annotations
from collections import namedtuple

import torch
from torch import Tensor, as_tensor
from torch.nn import Module, Linear, Sequential, Tanh, GELU, Identity
import torch.nn.functional as F

# ein equations

from einops import rearrange, pack
from torch_einops_utils import binary_cross_entropy_with_logits, frozen, uniform_like
from torch_einops_utils.shape import size, shape

# mlps

from x_mlps_pytorch import MLP, MimoMLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

# gae for learning the critic for the horizon = 16 version, but not necessary

from x_ppo import calc_gae
from assoc_scan import AssocScan

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def last(arr):
    return arr[-1]

def compact(*arr):
    return tuple(filter(exists, arr))

# actor

class HumanoidActor(Module):
    """unitree humanoid actor"""

    def __init__(
        self,
        *,
        dim_obs = 93,
        dim_action = 29,
        dim_goal = 71,
        dim_command = 3,
        num_obs_frames = 10,
        num_goal_frames = 10,
        dim_latent = 32,
        dim_hiddens = (2048, 1024, 512, 256),
        dim_velocity_hiddens = (256, 256),
        activation = GELU()
    ):
        super().__init__()

        motion_encoder = Sequential(
            MLP(num_goal_frames * dim_goal, *dim_hiddens, dim_latent, activation = activation),
            Tanh()
        )

        velocity_encoder = Sequential(
            MLP(dim_command + num_obs_frames * dim_obs, *dim_velocity_hiddens, dim_latent, activation = activation),
            Tanh()
        )

        self.net = MimoMLP(
            dim_latent + num_obs_frames * dim_obs,
            *dim_hiddens,
            dim_action,
            inputs = (motion_encoder, velocity_encoder, Identity()),
            activation = activation
        )

    def forward(
        self,
        obs,                                # (b num_obs_frames dim_obs)
        goal = None,                        # (b num_goal_frames dim_goal)
        command = None                      # (b dim_command) normalized velocity command
    ):
        is_motion = exists(goal)
        assert is_motion != exists(command), 'pass either goal or command'

        obs = rearrange(obs, 'b h d -> b (h d)')
        goal = rearrange(goal, 'b k d -> b (k d)') if is_motion else None

        velocity_input = (command, obs) if not is_motion else None
        select_inputs = (0 if is_motion else 1, 2)

        # (motion goal, (velocity command, obs history), obs history)

        inputs = (goal, velocity_input, obs)
        return self.net(inputs, select_inputs = select_inputs)

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

    def rollout(
        self,
        state,
        actions,                                 # (b time dim_action)
        next_states: Tensor,                     # (b time ...)
        sim_context: Tensor | None = None,       # (b dim_sim_context)
        next_terminals: Tensor | None = None,    # (b time ...)
        next_rewards: Tensor | None = None       # (b time ...)
    ):
        num_steps = size(actions, 'b [t] ...')

        # accumulants

        all_preds = []
        all_losses = []
        total_loss = 0.

        # rollout across timesteps

        for time_index in range(num_steps):
            action = actions[:, time_index]
            next_state = next_states[:, time_index]

            next_terminal = next_reward = None

            if exists(next_terminals):
                next_terminal = next_terminals[:, time_index]

            if exists(next_rewards):
                next_reward = next_rewards[:, time_index]

            loss, pred, losses = self(
                state,
                action,
                sim_context,
                next_state,
                next_terminal,
                next_reward
            )

            total_loss = total_loss + loss

            all_preds.append(pred)
            all_losses.append(losses)

            state = pred.state # feed prediction back in

        return total_loss, all_preds, all_losses

    def forward(
        self,
        state,                                  # (b dim_state)
        action,                                 # (b dim_action)
        sim_context: Tensor | None = None,      # (b dim_sim_context)
        next_state: Tensor | None = None,
        next_terminal: Tensor | None = None,
        next_reward: Tensor | None = None
    ):
        assert action.ndim == 2, 'use `rollout` for a rollout loss'
        assert self.has_sim_context == exists(sim_context)

        # world model is just a forward dynamics model f(s, a) -> s'
        # with simulation context (domain randomization presumably)

        embed = self.transition_mlp(compact(state, action, sim_context))

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
            pred_terminal_logit = self.to_terminal_pred(embed)
            pred_terminal_logit = rearrange(pred_terminal_logit, '... 1 -> ...')
            pred_terminal = pred_terminal_logit.sigmoid()

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
            terminal_loss = binary_cross_entropy_with_logits(pred_terminal_logit, next_terminal)
            total_loss = total_loss + terminal_loss * self.terminal_loss_weight

        losses = Losses(state_loss, terminal_loss, reward_loss)

        return total_loss, pred, losses

# main class

Rollout = namedtuple('Rollout', ('states', 'actions', 'terminals', 'rewards'))

class SuperMimic(Module):
    def __init__(
        self,
        *,
        actor: MLP | NormedMLP | Module,
        world_model: WorldModel | Module,
        observation_noise_scale = None,
        discount_factor = 0.99
    ):
        super().__init__()

        self.actor = actor
        self.world_model = world_model

        # discount factor, for loss as well as gae (once critic is involved)

        self.assoc_scan = AssocScan(reverse = True)
        self.discount_factor = discount_factor

        # observation noise for the actor

        if exists(observation_noise_scale):
            observation_noise_scale = as_tensor(observation_noise_scale)

        self.register_buffer('observation_noise_scale', observation_noise_scale)

    def actor_loss(
        self,
        rewards,                # (b t)
        pred_terminal,          # (b t)
        bootstrap_value = None, # (b)
        discount_factor = None
    ):
        # rewards[t] = r_{t+1}, pred_terminal[t] = d_{t+1} -> aligned, as in eq (2)

        assert size(pred_terminal, 'b [t]') == size(rewards, 'b [t]')

        # following eq (2) in the paper
        # rewards are derived analytically from world model next state predictions, actions, mocap related references

        discount_factor = default(discount_factor, self.discount_factor)

        gates = discount_factor * (1. - pred_terminal)

        # maybe append the bootstrap value, for horizons 16 or less

        if exists(bootstrap_value):
            rewards, _ = pack([rewards, bootstrap_value], 'b *')
            gates, _ = pack([gates, torch.ones_like(bootstrap_value)], 'b *')

        # reverse assoc scan

        scanned = self.assoc_scan(gates, rewards)
        discounted_sum_rewards = scanned[:, 0]

        # maximize rewards

        loss = -discounted_sum_rewards.mean()

        return loss

    def forward(
        self,
        initial_state,
        sim_context: Tensor | None = None,
        actor_goals: Tensor | None = None,
        actor_cond: Tensor | None = None,
        num_rollout_timesteps = 32
    ):
        actor, world_model = self.actor, self.world_model

        pred_reward, pred_terminal = self.world_model.pred_reward, self.world_model.pred_terminal

        # rollout actor within the world model

        states = [initial_state]
        actions = []
        rewards = []
        terminals = []

        state = initial_state

        has_observation_noise = exists(self.observation_noise_scale)
        actor_has_goal = exists(actor_goals)

        with frozen(world_model, eval = True):

            for time_index in range(num_rollout_timesteps):

                # actor forward

                obs = state

                if has_observation_noise:
                    obs = state + uniform_like(state, self.observation_noise_scale)

                actor_goal = actor_goals[:, time_index] if actor_has_goal else None

                action = actor(compact(obs, actor_goal, actor_cond))

                # world model prediction

                pred = world_model(state, action, sim_context)

                # next state

                next_state = pred.state

                # accumulate

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
