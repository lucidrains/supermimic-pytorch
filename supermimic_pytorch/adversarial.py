from __future__ import annotations
from collections import namedtuple
from typing import Literal

import torch
from torch import Tensor, autograd, cat, stack
from torch.nn import Module, GELU
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from einops import rearrange, reduce, repeat
from torch_einops_utils import (
    frozen,
    pad_left_at_dim,
    safe_cat,
    slice_right_at_dim,
    tree_map_detach
)
from torch_einops_utils.shape import assert_shape

from x_mlps_pytorch import MLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from supermimic_pytorch.supermimic import (
    Rollout,
    State,
    SuperMimic,
    default,
    exists,
    flatten_state
)
from supermimic_pytorch.rewards import (
    projected_gravity,
    select,
    unpack_unitree_g1_state,
    world_to_heading_frame
)

# namedtuples

LossType = Literal['r3gan', 'lsgan']

DiscriminatorLosses = namedtuple('DiscriminatorLosses', ('total', 'adversarial', 'r1', 'r2', 'real_logit', 'fake_logit'))
AdversarialRollout = namedtuple('AdversarialRollout', (*Rollout._fields, 'task_rewards', 'motion_rewards', 'realism_rewards'))
WorldModelRollout = namedtuple('WorldModelRollout', ('loss', 'real_states', 'imagined_states', 'preds', 'losses', 'adv_loss'))

# helpers

def stack_states(states: list[State]) -> State:
    return tree_map(lambda *xs: stack(xs, dim = 1), *states)

def prepend_state(s0: State, next_states: State) -> State:
    return tree_map(lambda a, b: cat((rearrange(a, 'b ... -> b 1 ...'), b), dim = 1), s0, next_states)

def fit_seq_left(seq: Tensor, length: int, replicate = True) -> Tensor:
    diff = length - seq.shape[1]
    if diff <= 0:
        return slice_right_at_dim(seq, length, dim = 1)
    if not replicate:
        return pad_left_at_dim(seq, diff, dim = 1, value = 0.)

    pad = repeat(seq[:, :1], 'b 1 ... -> b p ...', p = diff)
    return cat((pad, seq), dim = 1)

def to_windows(seq: Tensor, window_size: int) -> Tensor:
    assert seq.shape[1] >= window_size, f'sequence length {seq.shape[1]} < window_size {window_size}'
    return rearrange(seq.unfold(1, window_size, 1), 'b n d w -> b n (w d)')

def align_cond(cond: State, num_windows: int) -> Tensor:
    cond = flatten_state(cond)
    if cond.ndim == 2:
        return repeat(cond, 'b d -> b n d', n = num_windows)

    assert cond.shape[1] >= num_windows, f'cond length {cond.shape[1]} < {num_windows}'
    return slice_right_at_dim(cond, num_windows, dim = 1)

# zero-centered gradient penalty (R1 / R2, Huang et al. 2025)

def zero_centered_gradient_penalty(inputs: Tensor, logits: Tensor) -> Tensor:
    grads, = autograd.grad(
        outputs = logits.sum(),
        inputs = inputs,
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )
    return reduce(grads.square(), '... d -> ...', 'sum').mean()

# adversarial trajectory critic (standalone, embodiment-agnostic)

class AdversarialCritic(Module):
    def __init__(
        self,
        *,
        dim_feature: int,
        window_size: int = 2,
        dim_action: int = 0,
        dim_cond: int = 0,
        dim_hiddens: tuple[int, ...] = (512, 256),
        activation: Module = GELU(),
        normed_mlp: bool = True,
        net: Module | None = None,
        feature_fn = None,
        loss_type: LossType = 'r3gan',
        r1_weight: float = 5.,
        r2_weight: float = 5.,
        clamp_amp_reward: bool = False,
        real_logit_ema_decay: float = 0.99
    ):
        super().__init__()
        assert window_size >= 1 and loss_type in ('r3gan', 'lsgan')

        self.window_size = window_size
        self.include_actions = dim_action > 0
        self.has_cond = dim_cond > 0
        self.num_action_frames = max(window_size - 1, 1) if self.include_actions else 0

        self.feature_fn = feature_fn
        self.loss_type = loss_type
        self.r1_weight = float(r1_weight)
        self.r2_weight = float(r2_weight)
        self.clamp_amp_reward = clamp_amp_reward
        self.real_logit_ema_decay = float(real_logit_ema_decay)

        self.dim_in = window_size * dim_feature + self.num_action_frames * dim_action + dim_cond

        if exists(net):
            self.net = net
        else:
            mlp_klass = NormedMLP if normed_mlp else MLP
            self.net = mlp_klass(self.dim_in, *dim_hiddens, 1, activation = activation)

            last_layer = self.net.layers[-1]
            last_linear = last_layer[0] if normed_mlp else last_layer
            torch.nn.init.zeros_(last_linear.bias)
            torch.nn.init.orthogonal_(last_linear.weight, gain = 0.01)

        self.register_buffer('real_logit_ema', torch.zeros(()))

    def pack_windows(
        self,
        states: State,
        actions: Tensor | None = None,
        cond: State | None = None,
        horizon: int | None = None
    ) -> Tensor:
        feats = flatten_state(select(states, self.feature_fn) if exists(self.feature_fn) else states)

        if exists(horizon):
            feats = fit_seq_left(feats, horizon + self.window_size - 1, replicate = True)

        state_windows = to_windows(feats, self.window_size)
        num_windows = state_windows.shape[1]

        action_windows = None
        if self.include_actions:
            assert exists(actions), '`actions` required when `dim_action > 0`'
            act = flatten_state(actions)
            if exists(horizon):
                act = fit_seq_left(act, horizon + self.num_action_frames - 1, replicate = False)
            action_windows = slice_right_at_dim(to_windows(act, self.num_action_frames), num_windows, dim = 1)

        cond_aligned = None
        if self.has_cond:
            assert exists(cond), '`cond` required when `dim_cond > 0`'
            cond_aligned = align_cond(cond, num_windows)

        windows = safe_cat((state_windows, action_windows, cond_aligned), dim = -1)
        assert_shape(windows, 'b n d', d = self.dim_in)
        return windows

    def score(self, windows: Tensor) -> Tensor:
        logits = self.net(windows)
        return rearrange(logits, '... 1 -> ...') if logits.ndim == windows.ndim and logits.shape[-1] == 1 else logits

    def forward(
        self,
        states: State,
        actions: Tensor | None = None,
        cond: State | None = None,
        *,
        horizon: int | None = None
    ) -> Tensor:
        return self.score(self.pack_windows(states, actions, cond, horizon))

    def reward(
        self,
        states: State,
        actions: Tensor | None = None,
        cond: State | None = None,
        *,
        horizon: int | None = None,
        detach: bool = False
    ) -> Tensor:
        if detach:
            states = tree_map_detach(states)
            actions = actions.detach() if exists(actions) else None
            cond = tree_map_detach(cond) if exists(cond) else None

        with frozen(self, eval = True):
            logits = self(states, actions, cond, horizon = horizon)

            if self.loss_type == 'lsgan':
                rew = 1. - 0.25 * (logits - 1.).square()
                rew = rew.clamp(min = 0.) if self.clamp_amp_reward else rew
            else:
                rew = -F.softplus(-(logits - self.real_logit_ema.to(logits)))

        return rew.detach() if detach else rew

    def loss_from_windows(
        self,
        real_x: Tensor,
        fake_x: Tensor,
        *,
        apply_gp: bool = True,
        return_breakdown: bool = False
    ):
        do_r1 = apply_gp and self.r1_weight > 0.
        do_r2 = apply_gp and self.r2_weight > 0.

        real_x = real_x.detach().requires_grad_(do_r1)
        fake_x = fake_x.detach().requires_grad_(do_r2)

        real_logits = self.score(real_x)
        fake_logits = self.score(fake_x)

        real_mean = real_logits.mean()
        fake_mean = fake_logits.mean()

        if self.loss_type == 'r3gan':
            if self.training:
                self.real_logit_ema.lerp_(real_mean.detach().to(self.real_logit_ema), 1. - self.real_logit_ema_decay)

            adv_loss = 0.5 * (
                F.softplus(-(real_logits - fake_mean)).mean() +
                F.softplus(fake_logits - real_mean).mean()
            )
        else:
            adv_loss = 0.5 * (
                (real_logits - 1.).square().mean() +
                (fake_logits + 1.).square().mean()
            )

        zero = adv_loss.new_zeros(())
        r1_loss = 0.5 * self.r1_weight * zero_centered_gradient_penalty(real_x, real_logits) if do_r1 else zero
        r2_loss = 0.5 * self.r2_weight * zero_centered_gradient_penalty(fake_x, fake_logits) if do_r2 else zero

        total = adv_loss + r1_loss + r2_loss

        if not return_breakdown:
            return total

        return DiscriminatorLosses(total, adv_loss.detach(), r1_loss.detach(), r2_loss.detach(), real_mean.detach(), fake_mean.detach())

    def loss(
        self,
        real_states: State,
        fake_states: State,
        *,
        real_actions: Tensor | None = None,
        fake_actions: Tensor | None = None,
        real_cond: State | None = None,
        fake_cond: State | None = None,
        apply_gp: bool = True,
        return_breakdown: bool = False
    ):
        real_x = self.pack_windows(real_states, real_actions, real_cond)
        fake_x = self.pack_windows(fake_states, fake_actions, fake_cond)
        return self.loss_from_windows(real_x, fake_x, apply_gp = apply_gp, return_breakdown = return_breakdown)

# adapter wrapping SuperMimic for AMP (rollout vs ref) and real-vs-imagined dynamics

class AdversarialSuperMimic(Module):
    def __init__(
        self,
        supermimic: SuperMimic,
        *,
        motion_critic: AdversarialCritic | None = None,
        dynamics_critic: AdversarialCritic | None = None,
        task_reward_weight: float = 1.,
        motion_reward_weight: float = 1.,
        pessimism_weight: float = 0.,
        world_model_adv_weight: float = 0.,
        detach_motion_reward: bool = False,
        detach_pessimism_reward: bool = False
    ):
        super().__init__()
        assert exists(motion_critic) or exists(dynamics_critic), 'pass `motion_critic` and/or `dynamics_critic`'

        self.supermimic = supermimic
        self.motion_critic = motion_critic
        self.dynamics_critic = dynamics_critic

        self.task_reward_weight = float(task_reward_weight)
        self.motion_reward_weight = float(motion_reward_weight)
        self.pessimism_weight = float(pessimism_weight)
        self.world_model_adv_weight = float(world_model_adv_weight)
        self.detach_motion_reward = detach_motion_reward
        self.detach_pessimism_reward = detach_pessimism_reward

    def actor_loss(self, *args, **kwargs):
        return self.supermimic.actor_loss(*args, **kwargs)

    def critic_loss(self, *args, **kwargs):
        return self.supermimic.critic_loss(*args, **kwargs)

    @torch.no_grad()
    def update_ema(self):
        self.supermimic.update_ema()
        return self

    def _format_actions(self, actions: Tensor | None, state: State) -> Tensor | None:
        if not exists(actions):
            return None
        act = self.supermimic.maybe_one_hot_actions(actions, flatten_state(state).dtype)
        return rearrange(act, 'b d -> b 1 d') if act.ndim == 2 else act

    def _score(
        self,
        critic: AdversarialCritic,
        states: State,
        actions: Tensor | None = None,
        cond: State | None = None,
        horizon: int | None = None,
        detach: bool = False
    ) -> Tensor:
        return critic.reward(
            states,
            actions = actions if critic.include_actions else None,
            cond = cond if critic.has_cond else None,
            horizon = horizon,
            detach = detach
        )

    # world model fit (+ optional real-vs-imagined realism loss)

    def world_model_rollout(
        self,
        state: State,
        actions: Tensor,
        next_states: State,
        sim_context: Tensor | None = None,
        next_terminals: Tensor | None = None,
        next_rewards: Tensor | None = None,
        *,
        dynamics_cond: State | None = None
    ) -> WorldModelRollout:
        wm_loss, preds, losses = self.supermimic.world_model.rollout(
            state, actions, next_states,
            sim_context = sim_context,
            next_terminals = next_terminals,
            next_rewards = next_rewards
        )

        real_states = prepend_state(state, next_states)
        imagined_states = stack_states([state, *(p.state for p in preds)])
        adv_loss = None

        if exists(self.dynamics_critic) and self.world_model_adv_weight > 0.:
            act = self._format_actions(actions, state)
            adv_loss = -self._score(self.dynamics_critic, imagined_states, act, default(dynamics_cond, sim_context)).mean()
            wm_loss = wm_loss + self.world_model_adv_weight * adv_loss

        return WorldModelRollout(wm_loss, real_states, imagined_states, preds, losses, adv_loss)

    # imagined rollout (+ optional AMP motion prior & real-vs-imagined pessimism)

    def forward(
        self,
        initial_state: State,
        sim_context: Tensor | None = None,
        *args,
        motion_cond: State | None = None,
        dynamics_cond: State | None = None,
        **kwargs
    ) -> AdversarialRollout:
        rollout = self.supermimic(initial_state, sim_context, *args, **kwargs)
        horizon = rollout.actions.shape[1]

        task_rewards = rollout.rewards
        motion_rewards = realism_rewards = None
        terms = []

        if exists(task_rewards) and self.task_reward_weight != 0.:
            terms.append(self.task_reward_weight * task_rewards)

        if exists(self.motion_critic) and self.motion_reward_weight != 0.:
            motion_rewards = self._score(
                self.motion_critic, rollout.states, rollout.actions, motion_cond,
                horizon = horizon, detach = self.detach_motion_reward
            )
            terms.append(self.motion_reward_weight * motion_rewards)

        if exists(self.dynamics_critic) and self.pessimism_weight != 0.:
            realism_rewards = self._score(
                self.dynamics_critic, rollout.states, rollout.actions, default(dynamics_cond, sim_context),
                horizon = horizon, detach = self.detach_pessimism_reward
            )
            terms.append(self.pessimism_weight * realism_rewards)

        total_rewards = sum(terms[1:], terms[0]) if terms else task_rewards
        return AdversarialRollout(*rollout[:3], total_rewards, task_rewards, motion_rewards, realism_rewards)

    # AMP critic loss: unpaired reference mocap vs. policy rollout

    def motion_critic_loss(
        self,
        ref_states: State,
        fake: Rollout | AdversarialRollout | State,
        *,
        ref_actions: Tensor | None = None,
        fake_actions: Tensor | None = None,
        ref_cond: State | None = None,
        fake_cond: State | None = None,
        apply_gp: bool = True,
        return_breakdown: bool = False
    ):
        critic = self.motion_critic
        assert exists(critic), '`motion_critic` was not passed'

        if isinstance(fake, (Rollout, AdversarialRollout)):
            fake_states = fake.states
            fake_actions = default(fake_actions, fake.actions if critic.include_actions else None)
        else:
            fake_states = fake

        return critic.loss(
            ref_states, fake_states,
            real_actions = ref_actions,
            fake_actions = fake_actions,
            real_cond = ref_cond,
            fake_cond = fake_cond,
            apply_gp = apply_gp,
            return_breakdown = return_breakdown
        )

    # dynamics critic loss: real simulator vs. imagined WorldModel / SuperMimic rollout

    def dynamics_critic_loss(
        self,
        real_states: State,
        imagined: Rollout | AdversarialRollout | WorldModelRollout | State,
        *,
        real_actions: Tensor | None = None,
        imagined_actions: Tensor | None = None,
        real_cond: State | None = None,
        imagined_cond: State | None = None,
        apply_gp: bool = True,
        return_breakdown: bool = False
    ):
        critic = self.dynamics_critic
        assert exists(critic), '`dynamics_critic` was not passed'

        if isinstance(imagined, WorldModelRollout):
            imagined_states = imagined.imagined_states
        elif isinstance(imagined, (Rollout, AdversarialRollout)):
            imagined_states = imagined.states
            imagined_actions = default(imagined_actions, imagined.actions)
        else:
            imagined_states = imagined

        if critic.include_actions:
            real_actions = self._format_actions(real_actions, real_states)
            imagined_actions = self._format_actions(default(imagined_actions, real_actions), real_states)

        return critic.loss(
            real_states, imagined_states,
            real_actions = real_actions if critic.include_actions else None,
            fake_actions = imagined_actions if critic.include_actions else None,
            real_cond = real_cond,
            fake_cond = default(imagined_cond, real_cond) if critic.has_cond else None,
            apply_gp = apply_gp,
            return_breakdown = return_breakdown
        )

# unitree preset: heading-invariant AMP style features (68-d)

def unitree_g1_amp_features(state: State) -> Tensor:
    s = unpack_unitree_g1_state(state)
    q = s['root_quat']

    return cat((
        s['root_pos'][..., 2:3],
        projected_gravity(q),
        world_to_heading_frame(q, s['root_lin_vel']),
        world_to_heading_frame(q, s['root_ang_vel']),
        s['joint_pos'],
        s['joint_vel']
    ), dim = -1)
