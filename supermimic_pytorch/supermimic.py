from __future__ import annotations
from collections import namedtuple
from typing import Callable

import torch
from torch import Tensor, as_tensor, cat, is_tensor
from torch.nn import Module, Linear, Sequential, Tanh, GELU, Identity
import torch.nn.functional as F
from torch.utils._pytree import tree_map

# ein equations

from einops import rearrange, pack, repeat
from torch_einops_utils import binary_cross_entropy_with_logits, freeze, frozen, masked_mean, tree_flatten_with_inverse, tree_map_tensor, uniform_like
from torch_einops_utils.shape import assert_shape, size

# mlps

from x_mlps_pytorch import MLP, MimoMLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from ema_pytorch import EMA
from assoc_scan import AssocScan
from discrete_continuous_embed_readout import MultiCategorical

from supermimic_pytorch.history import (
    HistoryWindow,
    action_smoothness_penalty
)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def last(arr):
    return arr[-1]

def compact(*arr):
    return tuple(filter(exists, arr))

def cast_tuple(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v,)

# state helpers (Tensor, dict[Name, Tensor], or nested dict)

Name = str
State = Tensor | dict[Name, 'State']

def state_schema(state: State):
    if is_tensor(state):
        assert state.ndim > 0, 'state leaves must have at least 1 dimension'
        return state.shape[-1]

    assert isinstance(state, dict) and len(state) > 0, 'state must be a Tensor or non-empty dict'
    assert all(isinstance(k, str) for k in state.keys()), 'state dict keys must be str'

    return tuple((k, state_schema(v)) for k, v in state.items())

def flatten_state_with_inverse(state: State, match = None):
    schema = None if is_tensor(state) else state_schema(state)

    if exists(match):
        expected = getattr(match, 'schema', None) if callable(match) else match
        assert expected == schema, 'state dict structure mismatch'

    if is_tensor(state):
        return state, identity

    tensors, unflatten = tree_flatten_with_inverse(state)
    dims = [t.shape[-1] for t in tensors]

    def inverse(t: Tensor):
        return unflatten(t.split(dims, dim = -1))

    inverse.schema = schema

    return cat(tensors, dim = -1), inverse

def flatten_state(state: State, match = None):
    flat, _ = flatten_state_with_inverse(state, match = match)
    return flat

def flatten_window(x: State | None) -> Tensor | None:
    if not exists(x):
        return None
    flat = flatten_state(x)
    return rearrange(flat, '... w d -> ... (w d)') if flat.ndim >= 3 else flat

def select_key(state: State, key: str | tuple[str, ...]) -> State:
    out = state
    for k in cast_tuple(key):
        assert isinstance(out, dict) and k in out, f'key {k!r} not found in state dict'
        out = out[k]
    return out

def add_observation_noise(obs: State, scale) -> State:
    if not exists(scale):
        return obs

    def add_noise(tensor, noise_scale):
        if not exists(noise_scale):
            return tensor

        if is_tensor(noise_scale):
            noise_scale = noise_scale.to(device = tensor.device, dtype = tensor.dtype)

        return tensor + uniform_like(tensor, noise_scale)

    if isinstance(scale, dict) or is_tensor(obs):
        return tree_map(add_noise, obs, scale)

    if is_tensor(scale) and scale.ndim > 0:
        flat, inverse = flatten_state_with_inverse(obs)
        return inverse(add_noise(flat, scale))

    return tree_map_tensor(lambda tensor: add_noise(tensor, scale), obs)

def resolve_action_dim(
    dim_action: int | None,
    num_discrete_actions: int | tuple[int, ...] | None,
    default_dim_action = 29
):
    num_discrete_actions = cast_tuple(num_discrete_actions) if num_discrete_actions else ()
    assert all(n > 0 for n in num_discrete_actions), 'discrete action counts must be positive'

    if not num_discrete_actions:
        return default(dim_action, default_dim_action), ()

    assert not (exists(dim_action) and dim_action > 0), 'cannot specify both `dim_action` and `num_discrete_actions`'
    return sum(num_discrete_actions), num_discrete_actions

def to_one_hot(indices: Tensor, num_classes: tuple[int, ...], dtype):
    indices = indices.long()

    if len(num_classes) == 1:
        if indices.ndim == 2 and indices.shape[-1] == 1:
            indices = rearrange(indices, 'b 1 -> b')
        return F.one_hot(indices, num_classes[0]).to(dtype)

    assert indices.ndim >= 2 and indices.shape[-1] == len(num_classes)
    return cat([F.one_hot(i, n) for i, n in zip(indices.unbind(dim = -1), num_classes)], dim = -1).to(dtype)

# actor

class HumanoidActor(Module):
    """unitree humanoid actor"""

    structured_input = True

    def __init__(
        self,
        *,
        dim_obs = 93,
        dim_action = 29,
        dim_goal = 71,
        dim_command = 3,
        num_obs_frames = 10,
        num_action_frames = 0,
        num_goal_frames = 10,
        dim_latent = 32,
        dim_hiddens = (2048, 1024, 512, 256),
        dim_velocity_hiddens = (256, 256),
        activation = GELU()
    ):
        super().__init__()

        dim_obs_history = num_obs_frames * dim_obs + num_action_frames * dim_action

        motion_encoder = Sequential(
            MLP(num_goal_frames * dim_goal, *dim_hiddens, dim_latent, activation = activation),
            Tanh()
        )

        velocity_encoder = Sequential(
            MLP(dim_command + dim_obs_history, *dim_velocity_hiddens, dim_latent, activation = activation),
            Tanh()
        )

        self.net = MimoMLP(
            dim_latent + dim_obs_history,
            *dim_hiddens,
            dim_action,
            inputs = (motion_encoder, velocity_encoder, Identity()),
            activation = activation
        )

    def forward(
        self,
        obs: State,                         # (... num_obs_frames dim_obs) or nested dict
        goal: State | None = None,          # (... num_goal_frames dim_goal)
        command = None,                     # (... dim_command) normalized velocity command
        past_action: Tensor | None = None   # (... num_action_frames dim_action)
    ):
        is_motion = exists(goal)
        assert is_motion != exists(command), 'pass either goal or command'

        obs = flatten_window(obs)

        if exists(past_action):
            obs = cat((obs, flatten_window(past_action)), dim = -1)

        goal = flatten_window(goal) if is_motion else None

        velocity_input = (command, obs) if not is_motion else None
        select_inputs = (0 if is_motion else 1, 2)

        # (motion goal, (velocity command, obs history), obs history)

        inputs = (goal, velocity_input, obs)
        return self.net(inputs, select_inputs = select_inputs)

# critic

class Critic(Module):
    assoc_scan = AssocScan(reverse = True)

    def __init__(
        self,
        *,
        dim_state = 71,
        dim_goal = 71,
        dim_sim_context = 33,
        dim_hiddens = (256, 128),
        activation = GELU(),
        normed_mlp = False
    ):
        super().__init__()

        mlp_klass = MLP if not normed_mlp else NormedMLP

        self.dim_in = dim_state + dim_goal + dim_sim_context
        self.net = mlp_klass(self.dim_in, *dim_hiddens, 1, activation = activation)

        last_layer = self.net.layers[-1]
        last_linear = last_layer[0] if normed_mlp else last_layer
        torch.nn.init.zeros_(last_linear.bias)
        torch.nn.init.orthogonal_(last_linear.weight, gain = 0.01)

    @staticmethod
    def calc_returns(
        rewards,                # (b n)
        values,                 # (b n + 1)
        pred_terminal,          # (b n)
        discount_factor = 0.99,
        lam = 0.95
    ):
        # eq 3

        num_steps = size(rewards, 'b n -> n')

        assert_shape(values, 'b n -> n', n = num_steps + 1)
        assert_shape(pred_terminal, 'b n', n = num_steps)

        values, next_values = values[..., :-1], values[..., 1:]
        survival = 1. - pred_terminal

        deltas = rewards + discount_factor * survival * next_values - values
        gates = discount_factor * survival * lam

        gae = Critic.assoc_scan(gates, deltas)
        return gae + values

    def forward(
        self,
        state: State,                       # (b d) or (b t d)
        goal = None,                        # (b g) or (b t g)
        sim_context = None,                 # (b c)
        returns = None,                     # (b n) -> loss
        mask = None                         # (b n)
    ):
        state = flatten_state(state)

        num_steps = size(state, 'b t d -> t') if state.ndim == 3 else None

        def align(t):
            if not exists(t) or not exists(num_steps) or t.ndim != 2:
                return t
            return repeat(t, 'b d -> b t d', t = num_steps)

        x = cat(compact(state, align(goal), align(sim_context)), dim = -1)
        assert_shape(x, '... d', d = self.dim_in)

        values = rearrange(self.net(x), '... 1 -> ...')

        if not exists(returns):
            return values

        loss = F.mse_loss(
            values,
            returns.detach(),
            reduction = 'none'
        )

        return masked_mean(loss, mask)

# world model

Predictions = namedtuple('Predictions', ('state', 'terminal', 'reward'))

Losses = namedtuple('Losses', ('state', 'terminal', 'reward'))

class WorldModel(Module):
    def __init__(
        self,
        *,
        dim_state = 71,
        dim_action: int | None = None,
        num_discrete_actions: int | tuple[int, ...] | None = None,
        dim_sim_context = 33,
        dim_hiddens: tuple[int, ...] = (2048, 2048, 2048),
        activation: Module = GELU(),
        normed_mlp = False,
        pred_residual_state = True,
        pred_terminal = True,
        pred_reward = False,
        terminal_init_bias = -4.,
        state_loss_weight = 1.,
        reward_loss_weight = 1.,
        terminal_loss_weight = 1.
    ):
        super().__init__()
        assert len(dim_hiddens) > 0

        dim_action, self.num_discrete_actions = resolve_action_dim(dim_action, num_discrete_actions)
        self.dim_action = dim_action
        self.has_discrete_actions = len(self.num_discrete_actions) > 0

        mlp_klass = MLP if not normed_mlp else NormedMLP

        dim_in = dim_state + dim_action + dim_sim_context

        # backbone

        self.transition_mlp = mlp_klass(dim_in, *dim_hiddens, activation = activation, activate_last = True)

        # predictions

        dim_embed = last(dim_hiddens)
        self.to_state_pred = Linear(dim_embed, dim_state)
        self.to_reward_pred = Linear(dim_embed, 1) if pred_reward else None
        self.to_terminal_pred = Linear(dim_embed, 1) if pred_terminal else None

        self.pred_reward = pred_reward
        self.pred_terminal = pred_terminal

        # zero init residual head -> starts as identity dynamics

        if pred_residual_state:
            torch.nn.init.zeros_(self.to_state_pred.weight)
            torch.nn.init.zeros_(self.to_state_pred.bias)

        if exists(self.to_terminal_pred):
            torch.nn.init.constant_(self.to_terminal_pred.bias, terminal_init_bias)

        self.has_sim_context = dim_sim_context > 0
        self.pred_residual_state = pred_residual_state

        # losses

        self.state_loss_weight = state_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.terminal_loss_weight = terminal_loss_weight

    def rollout(
        self,
        state: State,
        actions,                                 # (b time dim_action)
        next_states: State,                      # (b time ...)
        sim_context: Tensor | None = None,       # (b dim_sim_context)
        next_terminals: Tensor | None = None,    # (b time ...)
        next_rewards: Tensor | None = None       # (b time ...)
    ):
        num_steps = size(actions, 'b [t] ...')

        state, inverse_state = flatten_state_with_inverse(state)
        next_states = flatten_state(next_states, match = inverse_state)

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

            all_preds.append(pred._replace(state = inverse_state(pred.state)))
            all_losses.append(losses)

            state = pred.state # feed prediction back in

        return total_loss, all_preds, all_losses

    def forward(
        self,
        state: State,                           # (b dim_state)
        action,                                 # (b dim_action) or discrete indices
        sim_context: Tensor | None = None,      # (b dim_sim_context)
        next_state: State | None = None,
        next_terminal: Tensor | None = None,
        next_reward: Tensor | None = None
    ):
        state, inverse_state = flatten_state_with_inverse(state)

        if self.has_discrete_actions and not action.is_floating_point():
            action = to_one_hot(action, self.num_discrete_actions, state.dtype)

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
            pred_reward = rearrange(self.to_reward_pred(embed), '... 1 -> ...')

        # maybe terminal

        pred_terminal_logit = pred_terminal = None

        if exists(self.to_terminal_pred):
            pred_terminal_logit = rearrange(self.to_terminal_pred(embed), '... 1 -> ...')
            pred_terminal = pred_terminal_logit.sigmoid()

        # outputs

        pred = Predictions(inverse_state(pred_next_state), pred_terminal, pred_reward)

        # maybe loss

        need_loss = exists(next_state) or exists(next_terminal) or exists(next_reward)

        if not need_loss:
            return pred

        total_loss = 0.
        state_loss = reward_loss = terminal_loss = None

        if exists(next_state):
            next_state = flatten_state(next_state, match = inverse_state)
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
        critic: Module | None = None,
        use_ema_critic = False,
        critic_ema_decay = 0.99,
        critic_ema_kwargs: dict = dict(),
        observation_noise_scale = None,
        actor_obs_key: str | tuple[str, ...] | None = None,
        state_to_obs: Callable[[State, Tensor | None], State] | None = None,
        critic_state_key: str | tuple[str, ...] | None = None,
        obs_history: HistoryWindow | int | None = None,
        action_history: HistoryWindow | int | None = None,
        goal_window: HistoryWindow | int | None = None,
        action_smoothness_weight: float = 0.,
        discount_factor = 0.99,
        weight_critic_by_survival = True,
        discrete_action_temperature = 1.
    ):
        super().__init__()
        assert not (exists(actor_obs_key) and exists(state_to_obs)), 'pass either `actor_obs_key` or `state_to_obs`'

        self.actor = actor
        self.world_model = world_model
        self.critic = critic

        self.dim_action = getattr(world_model, 'dim_action', None)
        self.num_discrete_actions = getattr(world_model, 'num_discrete_actions', ())
        self.has_discrete_actions = len(self.num_discrete_actions) > 0

        assert discrete_action_temperature > 0.
        self.discrete_action_temperature = discrete_action_temperature

        # asymmetric selectors, unfold history windows & smoothness

        self.actor_obs_key = actor_obs_key
        self.state_to_obs = state_to_obs
        self.critic_state_key = critic_state_key

        self.obs_history = (HistoryWindow(obs_history) if obs_history > 0 else None) if isinstance(obs_history, int) else obs_history
        self.action_history = (HistoryWindow(action_history, self.dim_action, lag = 1) if action_history > 0 else None) if isinstance(action_history, int) else action_history
        self.goal_window = (HistoryWindow(goal_window, future = True, pad_mode = 'replicate') if goal_window > 0 else None) if isinstance(goal_window, int) else goal_window

        self.action_smoothness_weight = action_smoothness_weight

        # optional ema target critic

        self.ema_critic = None

        if use_ema_critic:
            assert exists(critic), '`critic` must be passed when `use_ema_critic = True`'

            self.ema_critic = EMA(
                critic,
                include_online_model = False,
                beta = critic_ema_decay,
                update_after_step = 0,
                update_every = 1,
                **critic_ema_kwargs
            )

            freeze(self.ema_critic.ema_model).eval()

        # discount factor, for loss as well as gae (once critic is involved)

        self.assoc_scan = AssocScan(reverse = True)
        self.discount_factor = discount_factor
        self.weight_critic_by_survival = weight_critic_by_survival

        # observation noise for the actor

        self.observation_noise_schema = None
        self.observation_noise_dict = None

        if exists(observation_noise_scale):
            if isinstance(observation_noise_scale, dict):
                leaves = tree_flatten_with_inverse(observation_noise_scale)[0]
                can_flatten_state = not (exists(actor_obs_key) or exists(state_to_obs)) and all(is_tensor(t) and t.ndim > 0 for t in leaves)

                if can_flatten_state:
                    observation_noise_scale, inverse_noise = flatten_state_with_inverse(observation_noise_scale)
                    self.observation_noise_schema = inverse_noise.schema
                    observation_noise_scale = as_tensor(observation_noise_scale)
                else:
                    self.observation_noise_dict = observation_noise_scale
                    observation_noise_scale = None
            else:
                observation_noise_scale = as_tensor(observation_noise_scale)

        self.register_buffer('observation_noise_scale', observation_noise_scale)

    def train(self, mode = True):
        super().train(mode)

        if exists(self.ema_critic):
            self.ema_critic.ema_model.eval()

        return self

    @property
    def target_critic(self):
        return self.ema_critic.ema_model if exists(self.ema_critic) else self.critic

    @torch.no_grad()
    def update_ema(self):
        if exists(self.ema_critic):
            self.ema_critic.update()

        return self

    def critic_loss(
        self,
        states: State,          # (b t + 1 d)
        rewards,                # (b t)
        pred_terminal,          # (b t)
        goals = None,           # (b g) or (b t + 1 g)
        sim_context = None,     # (b c)
        lam = 0.95,
        discount_factor = None,
        weight_by_survival: bool | None = None,
        mask = None             # (b t)
    ):
        assert exists(self.critic), '`critic` was not passed to `SuperMimic`'

        discount_factor = default(discount_factor, self.discount_factor)
        weight_by_survival = default(weight_by_survival, self.weight_critic_by_survival)

        if exists(self.critic_state_key):
            states = select_key(states, self.critic_state_key)

        states = flatten_state(states)
        states, rewards, pred_terminal = states.detach(), rewards.detach(), pred_terminal.detach()

        if exists(goals):
            goals = flatten_state(goals)
            if goals.ndim == 3 and goals.shape[1] > states.shape[1]:
                goals = goals[:, :states.shape[1]]

        with torch.no_grad():
            values = self.target_critic(states, goals, sim_context)
            returns = Critic.calc_returns(rewards, values, pred_terminal, discount_factor = discount_factor, lam = lam)

        # survival weights c_t = prod_{i < t} (1 - d_i)

        if weight_by_survival:
            survival = 1. - pred_terminal
            survival_weights = F.pad(survival[:, :-1], (1, 0), value = 1.).cumprod(dim = -1)
            mask = mask * survival_weights if exists(mask) else survival_weights

        step_goals = goals[:, :-1] if exists(goals) and goals.ndim == 3 else goals

        return self.critic(
            states[:, :-1],
            step_goals,
            sim_context,
            returns = returns,
            mask = mask
        )

    def actor_loss(
        self,
        rewards,                # (b t)
        pred_terminal,          # (b t)
        bootstrap_value = None, # (b)
        final_state: State | None = None,     # (b d)
        final_goal = None,      # (b g)
        sim_context = None,     # (b c)
        actions: State | None = None,         # (b t d_a)
        past_actions: State | None = None,    # (b d_a) or (b p d_a)
        past_mask: Tensor | None = None,      # (b p) or (b,)
        action_smoothness_weight: float | None = None,
        discount_factor = None
    ):
        # rewards[t] = r_{t+1}, pred_terminal[t] = d_{t+1} -> aligned, as in eq (2)

        assert size(pred_terminal, 'b [t]') == size(rewards, 'b [t]')

        smoothness_weight = default(action_smoothness_weight, self.action_smoothness_weight)
        if smoothness_weight > 0.:
            assert exists(actions), '`actions` required when `action_smoothness_weight > 0`'
            rewards = rewards - smoothness_weight * action_smoothness_penalty(
                actions, past_actions, past_mask = past_mask
            )

        # following eq (2) in the paper
        # rewards are derived analytically from world model next state predictions, actions, mocap related references

        discount_factor = default(discount_factor, self.discount_factor)

        gates = discount_factor * (1. - pred_terminal)

        # bootstrap value from frozen critic, unless passed in

        if not exists(bootstrap_value) and exists(self.critic):
            assert exists(final_state), '`final_state` must be passed when `critic` is set'

            if exists(self.critic_state_key):
                final_state = select_key(final_state, self.critic_state_key)

            final_state = flatten_state(final_state)
            final_goal = flatten_state(final_goal) if exists(final_goal) else None

            with frozen(self.target_critic, eval = True):
                bootstrap_value = self.target_critic(final_state, final_goal, sim_context)

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
        initial_state: State,
        sim_context: Tensor | None = None,
        actor_goals: State | None = None,
        actor_cond: Tensor | None = None,
        past_obs: State | None = None,
        past_actions: Tensor | None = None,
        past_mask: Tensor | None = None,
        num_rollout_timesteps = 32
    ):
        actor, world_model = self.actor, self.world_model

        pred_reward, pred_terminal = self.world_model.pred_reward, self.world_model.pred_terminal

        # maybe flatten dict state

        initial_state, inverse_state = flatten_state_with_inverse(initial_state, match = self.observation_noise_schema)
        batch_size = size(initial_state, 'b d -> b')
        device, dtype = initial_state.device, initial_state.dtype

        # maybe unfold future goal windows & init rolling caches

        actor_has_goal = exists(actor_goals)

        if actor_has_goal and exists(self.goal_window):
            actor_goals = tree_map_tensor(lambda t: t[:, :num_rollout_timesteps], self.goal_window(actor_goals))

        obs_cache = act_cache = None

        if exists(self.obs_history) and exists(past_obs):
            _, obs_cache = self.obs_history.step(past_obs, mask = past_mask)

        if exists(self.action_history):
            if exists(past_actions) and self.has_discrete_actions and not past_actions.is_floating_point():
                past_actions = to_one_hot(past_actions, self.num_discrete_actions, dtype)
            _, act_cache = self.action_history.step(
                past_actions, mask = past_mask, batch_shape = (batch_size, self.dim_action), device = device, dtype = dtype
            )

        # rollout actor within the world model

        states = [initial_state]
        actions = []
        rewards = []
        terminals = []

        state = initial_state

        obs_noise = default(self.observation_noise_dict, self.observation_noise_scale)
        structured = getattr(actor, 'structured_input', False)
        is_latent_world_model = hasattr(world_model, 'encode') and hasattr(world_model, 'forward_from_latent')

        with frozen(world_model, eval = True):

            latent = world_model.encode(state) if is_latent_world_model else None

            for time_index in range(num_rollout_timesteps):

                # actor forward

                obs = inverse_state(state)

                if exists(self.state_to_obs):
                    obs = self.state_to_obs(obs, sim_context)
                elif exists(self.actor_obs_key):
                    obs = select_key(obs, self.actor_obs_key)

                obs = add_observation_noise(obs, obs_noise)

                if exists(self.obs_history):
                    obs, obs_cache = self.obs_history.step(obs, obs_cache)

                past_action = None
                if exists(self.action_history):
                    past_action, _ = self.action_history.step(cache = act_cache)

                actor_goal = tree_map_tensor(lambda t: t[:, time_index], actor_goals) if actor_has_goal else None

                if structured:
                    action = actor(obs, goal = actor_goal, command = actor_cond, past_action = past_action)
                else:
                    actor_inputs = compact(
                        flatten_window(obs),
                        flatten_window(past_action),
                        flatten_window(actor_goal),
                        flatten_window(actor_cond)
                    )
                    action = actor(actor_inputs)

                if self.has_discrete_actions:
                    if is_tensor(action) and len(self.num_discrete_actions) > 1:
                        action = action.split(self.num_discrete_actions, dim = -1)

                    action = MultiCategorical(action).sample(
                        temperature = self.discrete_action_temperature,
                        differentiable = True
                    )

                if exists(self.action_history):
                    _, act_cache = self.action_history.step(action, act_cache)

                # world model prediction

                if is_latent_world_model:
                    pred = world_model.forward_from_latent(latent, action, sim_context)
                    latent = pred.latent
                else:
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

        states = inverse_state(rearrange(states, 't b ... -> b t ...'))
        actions = rearrange(actions, 't b ... -> b t ...')

        rewards = rearrange(rewards, 't b -> b t') if pred_reward else None
        terminals = rearrange(terminals, 't b -> b t') if pred_terminal else None

        return Rollout(states, actions, terminals, rewards)
