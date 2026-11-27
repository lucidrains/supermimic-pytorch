from __future__ import annotations
from math import ceil
from collections import namedtuple
from typing import Callable, Literal

import torch
from torch import Tensor, as_tensor, autograd, cat, is_tensor
from torch.nn import Module, Parameter, Linear, Sequential, Tanh, GELU, Identity
import torch.nn.functional as F
from torch.utils._pytree import tree_map, tree_leaves

# ein equations

import einx
from einops import rearrange, reduce, pack, repeat
from torch_einops_utils import (
    binary_cross_entropy_with_logits,
    exclusive_cumprod,
    freeze,
    frozen,
    masked_mean,
    pad_left_at_dim,
    pad_right_at_dim,
    pad_right_ndim_to,
    safe_cat,
    slice_left_at_dim,
    slice_right_at_dim,
    tree_flatten_with_inverse,
    tree_map_detach,
    tree_map_tensor,
    uniform_like
)
from torch_einops_utils.shape import assert_shape, is_shape, shape, size

# mlps

from x_mlps_pytorch import MLP, MimoMLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from ema_pytorch import EMA
from assoc_scan import AssocScan
from discrete_continuous_embed_readout import MultiCategorical

from supermimic_pytorch.welford_normalizer import WelfordNormalizer

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

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
        return size(state, '... [d]')

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
    dims = [size(t, '... [d]') for t in tensors]

    def inverse(t: Tensor):
        return unflatten(t.split(dims, dim = -1))

    inverse.schema = schema

    return cat(tensors, dim = -1), inverse

def flatten_state(state: State, match = None):
    flat, _ = flatten_state_with_inverse(state, match = match)
    return flat

def batch_size_of(tree) -> int:
    return size(next(t for t in tree_leaves(tree) if is_tensor(t)), '[b] ...')

def prepend_state(s0: State, next_states: State) -> State:
    return tree_map(lambda a, b: cat((rearrange(a, 'b ... -> b 1 ...'), b), dim = 1), s0, next_states)

def repeat_batch(tree, k: int):
    if k <= 1:
        return tree

    return tree_map_tensor(lambda t: repeat(t, 'b ... -> (b k) ...', k = k), tree)

def align_batch(tree, target_batch: int):
    def _align(t: Tensor):
        if t.ndim == 0:
            return t

        batch = size(t, '[b] ...')
        if batch == target_batch:
            return t

        assert divisible_by(target_batch, batch), f'cannot align batch {batch} to {target_batch}'
        return repeat(t, 'b ... -> (b k) ...', k = target_batch // batch)

    return tree_map_tensor(_align, tree)

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
        if is_shape(indices, 'b 1'):
            indices = rearrange(indices, 'b 1 -> b')
        return F.one_hot(indices, num_classes[0]).to(dtype)

    assert_shape(indices, 'b ... k', k = len(num_classes))
    return cat([F.one_hot(i, n) for i, n in zip(indices.unbind(dim = -1), num_classes)], dim = -1).to(dtype)

# history window & action smoothness

PadMode = Literal['zero', 'replicate', 'learned']

HistoryCache = namedtuple('HistoryCache', ('frames', 'valid', 'inverse'))

def to_mask(mask: Tensor | None, mask_shape: tuple[int, int], device) -> Tensor:
    if not exists(mask):
        return torch.ones(mask_shape, dtype = torch.bool, device = device)
    mask = pad_right_ndim_to(mask, 2).to(device = device, dtype = torch.bool)
    assert mask.shape == mask_shape
    return mask

def to_sequence(tensor: Tensor, mask: Tensor | None = None):
    tensor = rearrange(tensor, 'b d -> b 1 d') if tensor.ndim == 2 else tensor
    assert tensor.ndim == 3
    return tensor, to_mask(mask, shape(tensor, '[b t] d').shape, tensor.device)

class HistoryWindow(Module):
    def __init__(
        self,
        window_size: int,
        dim: int | dict | None = None,
        *,
        lag: int = 0,
        future: bool = False,
        pad_mode: PadMode = 'zero',
        per_slot_pad: bool = True
    ):
        super().__init__()
        dim = sum(tree_flatten_with_inverse(dim)[0]) if isinstance(dim, dict) else dim

        assert window_size >= 1 and lag >= 0 and pad_mode in ('zero', 'replicate', 'learned')
        assert pad_mode != 'learned' or exists(dim), '`dim` required for learned pad embed'

        self.window_size = window_size
        self.dim = dim
        self.lag = lag
        self.future = future
        self.pad_mode = pad_mode
        self.per_slot_pad = per_slot_pad
        self.cache_size = window_size + max(lag - 1, 0) # trailing slots hold lag > 1 frames before read()

        pad_shape = (window_size, dim) if per_slot_pad else (dim,)
        self.pad_embed = Parameter(torch.zeros(pad_shape)) if pad_mode == 'learned' else None

    @property
    def dim_out(self) -> int:
        assert exists(self.dim)
        return self.window_size * self.dim

    def fill(self, windows: Tensor, valid: Tensor) -> Tensor:
        if exists(self.pad_embed):
            pattern = '... w, ... w d, w d -> ... w d' if self.per_slot_pad else '... w, ... w d, d -> ... w d'
            return einx.where(pattern, valid, windows, self.pad_embed.to(windows))

        if self.pad_mode == 'replicate':
            if self.future:
                anchor_idx = self.window_size - 1 - valid.flip(-1).int().argmax(dim = -1)
            else:
                anchor_idx = valid.int().argmax(dim = -1)

            anchor = einx.get_at('... [w] d, ... -> ... d', windows, anchor_idx)
            anchor = einx.where('..., ... d, -> ... d', valid.any(dim = -1), anchor, 0.)
            return einx.where('... w, ... w d, ... d -> ... w d', valid, windows, anchor)

        return einx.where('... w, ... w d, -> ... w d', valid, windows, 0.)

    def unfold(self, tensor: Tensor, pad_value = 0.) -> Tensor:
        pad_fn, slice_fn = (pad_right_at_dim, slice_right_at_dim) if self.future else (pad_left_at_dim, slice_left_at_dim)
        windows = pad_fn(tensor, self.window_size - 1 + self.lag, dim = 1, value = pad_value).unfold(1, self.window_size, 1)
        return rearrange(slice_fn(windows, size(tensor, 'b [t] ...'), dim = 1), 'b t ... w -> b t w ...')

    # streaming

    def init_cache(
        self,
        batch_size: int,
        dim: int | None = None,
        *,
        past: State | None = None,
        mask: Tensor | None = None,
        device = None,
        dtype = None
    ) -> HistoryCache:
        assert not self.future
        if exists(past):
            return self.push(past, mask = mask)

        dim = default(dim, self.dim)
        assert exists(dim), '`dim` required for empty cache'
        return HistoryCache(
            torch.zeros((batch_size, self.cache_size, dim), device = device, dtype = dtype),
            torch.zeros((batch_size, self.cache_size), device = device, dtype = torch.bool),
            None
        )

    def reset(self, cache: HistoryCache | None, done: Tensor) -> HistoryCache | None:
        if not exists(cache):
            return None
        keep = ~done.to(device = cache.valid.device, dtype = torch.bool)
        return cache._replace(valid = einx.logical_and('b w, b -> b w', cache.valid, keep))

    def push(
        self,
        frame: State,
        cache: HistoryCache | None = None,
        *,
        mask: Tensor | None = None,
        reset: Tensor | None = None
    ) -> HistoryCache:
        assert not self.future
        cache = self.reset(cache, reset) if exists(reset) else cache
        flat, inverse = flatten_state_with_inverse(frame, match = cache.inverse if exists(cache) else None)
        flat, frame_mask = to_sequence(flat, mask)

        if not exists(cache):
            batch, dim = shape(flat, '[b] t [d]')
            cache = self.init_cache(batch, dim, device = flat.device, dtype = flat.dtype)

        frames = slice_right_at_dim(cat((cache.frames.to(flat), flat), dim = 1), self.cache_size, dim = 1)
        valid = slice_right_at_dim(cat((cache.valid.to(frame_mask), frame_mask), dim = 1), self.cache_size, dim = 1)
        return HistoryCache(frames, valid, inverse)

    def read(self, cache: HistoryCache) -> State:
        filled = self.fill(cache.frames[:, :self.window_size], cache.valid[:, :self.window_size])
        return cache.inverse(filled) if exists(cache.inverse) else filled

    def step(
        self,
        frame: State,
        cache: HistoryCache | None = None,
        *,
        mask: Tensor | None = None,
        reset: Tensor | None = None
    ) -> tuple[State, HistoryCache]:
        assert self.lag == 0, 'use `read()` then `push()` for lagged windows'
        cache = self.push(frame, cache, mask = mask, reset = reset)
        return self.read(cache), cache

    # parallel unfold

    def forward(
        self,
        inputs: State,
        *,
        past: State | None = None,
        past_mask: Tensor | None = None,
        mask: Tensor | None = None,
        episode_starts: Tensor | None = None,
        return_mask: bool = False
    ):
        flat, inverse = flatten_state_with_inverse(inputs)
        seq, seq_mask = to_sequence(flat, mask)
        seq_len = size(seq, 'b [t] d')

        if exists(episode_starts):
            episode_starts = to_mask(episode_starts, shape(seq, '[b t] d').shape, seq.device)

        if exists(past):
            past_seq, past_seq_mask = to_sequence(flatten_state(past, match = inverse), past_mask)
            seq, seq_mask = cat((past_seq, seq), dim = 1), cat((past_seq_mask, seq_mask), dim = 1)
            if exists(episode_starts):
                episode_starts = pad_left_at_dim(episode_starts, size(past_seq, 'b [t] d'), dim = 1, value = False)

        windows, valid = self.unfold(seq, 0.), self.unfold(seq_mask, False)

        if exists(episode_starts):
            episode_ids = episode_starts.long().cumsum(dim = 1)
            valid = valid & einx.equal('b t w, b t -> b t w', self.unfold(episode_ids, -1), episode_ids)

        windows, valid = slice_right_at_dim(windows, seq_len, dim = 1), slice_right_at_dim(valid, seq_len, dim = 1)
        out = inverse(self.fill(windows, valid))
        return (out, valid) if return_mask else out

def resolve_history(window: HistoryWindow | int | None, **kwargs) -> HistoryWindow | None:
    return (HistoryWindow(window, **kwargs) if window > 0 else None) if isinstance(window, int) else window

def resolve_normalizer(
    normalizer: WelfordNormalizer | bool | None,
    data_shape: int | tuple[int, ...] | dict | None = None
) -> WelfordNormalizer | None:
    if isinstance(normalizer, WelfordNormalizer):
        return normalizer
    return WelfordNormalizer(data_shape = data_shape) if normalizer else None

def action_smoothness_penalty(
    actions: State,
    past_actions: State | None = None,
    *,
    past_mask: Tensor | None = None,
    mask: Tensor | None = None,
    order: int = 1,
    episode_starts: Tensor | None = None
) -> Tensor:
    assert order >= 1
    flat_actions, inverse = flatten_state_with_inverse(actions)
    flat_past = flatten_state(past_actions, match = inverse) if exists(past_actions) else None

    windows, valid = HistoryWindow(order + 1)(
        flat_actions, past = flat_past, past_mask = past_mask,
        mask = mask, episode_starts = episode_starts, return_mask = True
    )

    squared_diff = reduce(torch.diff(windows, n = order, dim = 2).square(), 'b t 1 d -> b t', 'sum')
    return einx.where('b t, b t, -> b t', valid.all(dim = -1), squared_diff, 0.)

# explorative latent sampling - Gladstone et al. (2026); antithetic pairing - Jia et al. (2025)

def sample_explorative_latent(
    batch_size: int,
    num_rollouts: int,
    dim_latent: int,
    *,
    antithetic: bool = True,
    scale: float = 0.25,
    device = None,
    dtype = torch.float32
) -> Tensor:
    if num_rollouts <= 1 or scale == 0.:
        return torch.zeros((batch_size * num_rollouts, dim_latent), device = device, dtype = dtype)

    num_samples = ceil(num_rollouts / 2) if antithetic else num_rollouts
    z = torch.randn((batch_size, num_samples, dim_latent), device = device, dtype = dtype)

    if antithetic:
        z = cat((z, -z), dim = 1)[:, :num_rollouts]

    z = z * (scale * (dim_latent ** -0.5))
    return rearrange(z, 'b k d -> (b k) d')

def resolve_explorative_latent(ref: Tensor, explorative_latent: Tensor | None, dim_latent: int) -> Tensor | None:
    if dim_latent <= 0:
        return None

    batch = size(ref, '[b] ...')
    if exists(explorative_latent):
        assert_shape(explorative_latent, 'b d', b = batch, d = dim_latent)
        return explorative_latent

    return ref.new_zeros((batch, dim_latent))

def select_explorative_loss(losses: Tensor, k: int) -> Tensor:
    if k > 1:
        assert divisible_by(size(losses, '[b]'), k), f'batch {size(losses, "[b]")} must be divisible by num_explorative_rollouts {k}'
        losses = reduce(losses, '(b k) -> b', 'min', k = k)
    return losses

def reduce_explorative_loss(losses: Tensor, k: int) -> Tensor:
    return select_explorative_loss(losses, k).mean()

# world model rollout helpers - temporal decay (Hansen et al. 2024) & training state noise (Sanchez-Gonzalez et al. 2020)

def add_state_noise(
    state: Tensor,
    scale: float,
    normalizer: WelfordNormalizer | None = None
) -> Tensor:
    assert scale >= 0., '`state_noise_scale` must be non-negative'
    if scale == 0.:
        return state

    noise_std = normalizer.std.to(state) * scale if exists(normalizer) else scale
    return state + torch.randn_like(state) * noise_std

def rollout_step_weights(num_steps: int, decay: float) -> tuple[float, ...]:
    assert 0. < decay <= 1., '`rollout_loss_decay` must be in (0, 1]'
    if decay == 1. or num_steps == 0:
        return (1.,) * num_steps

    weights = tuple(decay ** t for t in range(num_steps))
    scale = num_steps / sum(weights)
    return tuple(w * scale for w in weights)

# per-rollout median gradient clipping - Dang & Edelkamp (2026)

def median_clip_rollout_losses(
    losses: Tensor,             # (b,)
    action_probe: Tensor,       # (b * k, t, ...)
    *,
    clip_mult = 10.,
    eps = 1e-8
) -> Tensor:
    assert clip_mult > 0.

    if not losses.requires_grad:
        return losses

    action_grads, = autograd.grad(losses.sum(), action_probe, retain_graph = True, only_inputs = True, allow_unused = True)

    if not exists(action_grads):
        return losses

    grad_norms = reduce(action_grads.float().square(), '(b k) ... -> b', 'sum', b = size(losses, '[b]')).sqrt()
    cap = clip_mult * grad_norms.median()

    weights = (cap / grad_norms.clamp(min = eps)).clamp(max = 1.)
    weights = torch.where(cap > 0., weights, 1.).to(losses.dtype).detach()

    return losses * weights

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
        dim_explorative_latent = 0,
        dim_hiddens = (2048, 1024, 512, 256),
        dim_velocity_hiddens = (256, 256),
        activation = GELU()
    ):
        super().__init__()
        assert dim_explorative_latent >= 0
        self.dim_explorative_latent = dim_explorative_latent

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
            dim_latent + dim_obs_history + dim_explorative_latent,
            *dim_hiddens,
            dim_action,
            inputs = (motion_encoder, velocity_encoder, Identity()),
            activation = activation
        )

    def forward(
        self,
        obs: State,                                # (... num_obs_frames dim_obs) or nested dict
        goal: State | None = None,                 # (... num_goal_frames dim_goal)
        command = None,                            # (... dim_command) normalized velocity command
        past_action: Tensor | None = None,         # (... num_action_frames dim_action)
        explorative_latent: Tensor | None = None   # (... dim_explorative_latent)
    ):
        is_motion = exists(goal)
        assert is_motion != exists(command), 'pass either goal or command'

        obs = safe_cat((flatten_window(obs), flatten_window(past_action)), dim = -1)

        goal = flatten_window(goal) if is_motion else None

        velocity_input = (command, obs) if not is_motion else None
        select_inputs = (0 if is_motion else 1, 2)

        policy_obs = obs

        if self.dim_explorative_latent > 0:
            policy_obs = (
                cat((obs, explorative_latent), dim = -1)
                if exists(explorative_latent) else
                pad_right_at_dim(obs, self.dim_explorative_latent, dim = -1)
            )

        # (motion goal, (velocity command, obs history), obs history [+ explorative latent])

        inputs = (goal, velocity_input, policy_obs)
        return self.net(inputs, select_inputs = select_inputs)

# critic

class Critic(Module):
    assoc_scan = AssocScan(reverse = True)
    structured_input = True

    def __init__(
        self,
        *,
        dim_state = 71,
        dim_goal = 71,
        dim_sim_context = 33,
        dim_hiddens = (256, 128),
        activation = GELU(),
        normed_mlp = False,
        state_normalizer: WelfordNormalizer | bool | None = None
    ):
        super().__init__()

        mlp_klass = MLP if not normed_mlp else NormedMLP

        self.dim_in = dim_state + dim_goal + dim_sim_context
        self.net = mlp_klass(self.dim_in, *dim_hiddens, 1, activation = activation)
        self.state_normalizer = resolve_normalizer(state_normalizer, data_shape = dim_state)

        last_layer = self.net.layers[-1]
        last_linear = last_layer[0] if normed_mlp else last_layer
        torch.nn.init.zeros_(last_linear.bias)
        torch.nn.init.orthogonal_(last_linear.weight, gain = 0.01)

    @staticmethod
    def calc_returns(
        rewards,                # (b n)
        values,                 # (b n + 1)
        pred_terminal = None,   # (b n)
        discount_factor = 0.99,
        lam = 0.95
    ):
        # eq 3

        num_steps = size(rewards, 'b n -> n')

        assert_shape(values, 'b n -> n', n = num_steps + 1)

        if exists(pred_terminal):
            assert_shape(pred_terminal, 'b n', n = num_steps)
            survival = 1. - pred_terminal
            gates = discount_factor * survival * lam
        else:
            survival = 1.
            gates = torch.full_like(rewards, discount_factor * lam)

        values, next_values = values[..., :-1], values[..., 1:]
        deltas = rewards + discount_factor * survival * next_values - values

        gae = Critic.assoc_scan(gates, deltas)
        return gae + values

    def forward(
        self,
        state: State,                       # (b d) or (b t d)
        goal = None,                        # (b g) or (b t g)
        sim_context = None,                 # (b c)
        returns = None,                     # (b n) -> loss
        mask = None,                        # (b n)
        *,
        update_normalizer: bool | None = None
    ):
        if exists(self.state_normalizer):
            state = self.state_normalizer(state, update = default(update_normalizer, self.training and exists(returns)))

        state = flatten_state(state)

        num_steps = size(state, 'b t d -> t') if state.ndim == 3 else None

        def align(t):
            if not exists(t) or not exists(num_steps) or t.ndim != 2:
                return t
            return repeat(t, 'b d -> b t d', t = num_steps)

        x = safe_cat((state, align(goal), align(sim_context)), dim = -1)
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
        terminal_loss_weight = 1.,
        rollout_loss_decay: float = 1.,
        state_noise_scale: float = 0.,
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

        self.state_normalizer = resolve_normalizer(state_normalizer, data_shape = dim_state)

        mlp_klass = MLP if not normed_mlp else NormedMLP

        dim_in = dim_state + dim_action + dim_sim_context + dim_explorative_latent

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

        # losses (temporal decay rho^t - Hansen et al. 2024; training noise - Sanchez-Gonzalez et al. 2020)

        self.state_loss_weight = state_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.terminal_loss_weight = terminal_loss_weight
        self.rollout_loss_decay = float(rollout_loss_decay)
        self.state_noise_scale = float(state_noise_scale)

    def rollout(
        self,
        state: State,
        actions,                                 # (b time dim_action)
        next_states: State,                      # (b time ...)
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

        # update running stats once on real logged states [s_0, ..., s_k], never on predicted states

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

        explorative_latent = None

        if self.dim_explorative_latent > 0:
            explorative_latent = sample_explorative_latent(
                orig_batch,
                k,
                self.dim_explorative_latent,
                antithetic = self.antithetic_explorative_latent,
                scale = default(explorative_latent_scale, self.explorative_latent_scale),
                device = state.device,
                dtype = state.dtype
            )

        # accumulants

        all_preds = []
        all_losses = []
        total_loss = state.new_zeros(orig_batch * k)

        # rollout across timesteps

        for time_index, step_weight in enumerate(step_weights):
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
                next_reward,
                explorative_latent = explorative_latent,
                reduce_loss = False,
                update_normalizer = False
            )

            total_loss = total_loss + step_weight * loss

            all_preds.append(pred._replace(state = inverse_state(pred.state)))
            all_losses.append(Losses(*(t.mean() if exists(t) else None for t in losses)))

            state = pred.state # feed prediction back in

        return reduce_explorative_loss(total_loss, k), all_preds, all_losses

    def forward(
        self,
        state: State,                           # (b dim_state)
        action,                                 # (b dim_action) or discrete indices
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
        flat_next_state = flatten_state(next_state, match = inverse_state) if exists(next_state) else None

        if exists(self.state_normalizer):
            if default(update_normalizer, self.training and exists(next_state)):
                real_states = tree_map(lambda a, b: torch.stack((a, b), dim = 1), state, next_state) if exists(next_state) else state
                self.state_normalizer.update(real_states)
            else:
                self.state_normalizer.init_stats(state)

        state = flat_state

        if self.has_discrete_actions and not action.is_floating_point():
            action = to_one_hot(action, self.num_discrete_actions, state.dtype)

        assert action.ndim == 2, 'use `rollout` for a rollout loss'
        assert self.has_sim_context == exists(sim_context)

        explorative_latent = resolve_explorative_latent(state, explorative_latent, self.dim_explorative_latent)

        # world model is just a forward dynamics model f(s, a, [xi, z]) -> s'

        norm_state = self.state_normalizer.normalize(state) if exists(self.state_normalizer) else state

        embed = self.transition_mlp(compact(norm_state, action, sim_context, explorative_latent))

        # whether predicting residual state or not

        pred_state = self.to_state_pred(embed)

        pred_norm_next_state = pred_state + norm_state if self.pred_residual_state else pred_state
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

        # outputs

        pred = Predictions(inverse_state(pred_next_state), pred_terminal, pred_reward)

        # maybe loss

        need_loss = exists(flat_next_state) or exists(next_terminal) or exists(next_reward)

        if not need_loss:
            return pred

        total_loss = state.new_zeros(size(state, '[b] d'))
        state_loss = reward_loss = terminal_loss = None

        if exists(flat_next_state):
            target_norm_next_state = self.state_normalizer.normalize(flat_next_state) if exists(self.state_normalizer) else flat_next_state
            state_loss = reduce(F.mse_loss(target_norm_next_state, pred_norm_next_state, reduction = 'none'), 'b d -> b', 'mean')
            total_loss = total_loss + state_loss * self.state_loss_weight

        if exists(next_reward):
            assert exists(pred_reward)
            reward_loss = F.mse_loss(next_reward, pred_reward, reduction = 'none')
            total_loss = total_loss + reward_loss * self.reward_loss_weight

        if exists(next_terminal):
            assert exists(pred_terminal)
            terminal_loss = binary_cross_entropy_with_logits(pred_terminal_logit, next_terminal, reduction = 'none')
            total_loss = total_loss + terminal_loss * self.terminal_loss_weight

        losses = Losses(state_loss, terminal_loss, reward_loss)

        if reduce_loss:
            total_loss = total_loss.mean()
            losses = Losses(*(t.mean() if exists(t) else None for t in losses))

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
        obs_normalizer: WelfordNormalizer | bool | None = None,
        reward_fn: Module | Callable | None = None,
        termination_fn: Module | Callable | None = None,
        action_smoothness_weight: float = 0.,
        discount_factor = 0.99,
        weight_critic_by_survival = True,
        discrete_action_temperature = 1.,
        num_explorative_rollouts: int | None = None,
        dim_explorative_latent: int = 0,
        explorative_latent_scale: float = 0.25,
        antithetic_explorative_latent: bool = True,
        wm_explorative_latent_scale: float = 0.,
        use_median_grad_clip: bool | None = None,
        median_grad_clip_mult = 10.
    ):
        super().__init__()
        assert not (exists(actor_obs_key) and exists(state_to_obs)), 'pass either `actor_obs_key` or `state_to_obs`'

        # explorative rollouts - Gladstone et al. (2026)

        num_explorative_rollouts = default(num_explorative_rollouts, 2 if dim_explorative_latent > 0 else 1)
        assert num_explorative_rollouts >= 1 and dim_explorative_latent >= 0

        self.num_explorative_rollouts = num_explorative_rollouts
        self.dim_explorative_latent = dim_explorative_latent
        self.explorative_latent_scale = explorative_latent_scale
        self.antithetic_explorative_latent = antithetic_explorative_latent
        self.wm_explorative_latent_scale = wm_explorative_latent_scale

        # per-rollout median-relative gradient clipping - Dang & Edelkamp (2026)

        assert median_grad_clip_mult > 0.
        self.use_median_grad_clip = use_median_grad_clip
        self.median_grad_clip_mult = median_grad_clip_mult
        self._action_probe = None

        self.actor = actor
        self.world_model = world_model
        self.critic = critic
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn

        self.dim_action = getattr(world_model, 'dim_action', None)
        self.num_discrete_actions = getattr(world_model, 'num_discrete_actions', ())
        self.has_discrete_actions = len(self.num_discrete_actions) > 0

        assert discrete_action_temperature > 0.
        self.discrete_action_temperature = discrete_action_temperature

        # asymmetric selectors, unfold history windows & smoothness

        self.actor_obs_key = actor_obs_key
        self.state_to_obs = state_to_obs
        self.critic_state_key = critic_state_key

        self.obs_history = resolve_history(obs_history)
        self.action_history = resolve_history(action_history, dim = self.dim_action, lag = 1)
        self.goal_window = resolve_history(goal_window, future = True, pad_mode = 'replicate')
        self.obs_normalizer = resolve_normalizer(obs_normalizer)

        self.action_smoothness_weight = action_smoothness_weight

        # optional ema target critic

        self.ema_critic = None

        if use_ema_critic:
            assert exists(critic), '`critic` must be passed when `use_ema_critic = True`'

            ema_kwargs = dict(critic_ema_kwargs)
            ignore_prefixes = set(ema_kwargs.pop('ignore_startswith_names', set())) | {'state_normalizer'}

            self.ema_critic = EMA(
                critic,
                include_online_model = False,
                beta = critic_ema_decay,
                update_after_step = 0,
                update_every = 1,
                ignore_startswith_names = ignore_prefixes,
                **ema_kwargs
            )

            critic_norm = getattr(critic, 'state_normalizer', None)
            if isinstance(critic_norm, WelfordNormalizer):
                self.ema_critic.ema_model.state_normalizer = critic_norm

            freeze(self.ema_critic.ema_model).eval()

            if isinstance(critic_norm, WelfordNormalizer):
                critic_norm.train(critic.training)

        # discount factor, for loss as well as gae (once critic is involved)

        self.assoc_scan = AssocScan(reverse = True)
        self.discount_factor = discount_factor
        self.weight_critic_by_survival = weight_critic_by_survival

        # observation noise for the actor

        self.observation_noise_schema = None
        self.observation_noise_dict = None

        if isinstance(observation_noise_scale, dict):
            leaves, _ = tree_flatten_with_inverse(observation_noise_scale)
            if not (exists(actor_obs_key) or exists(state_to_obs)) and all(is_tensor(t) and t.ndim > 0 for t in leaves):
                observation_noise_scale, inv = flatten_state_with_inverse(observation_noise_scale)
                self.observation_noise_schema = inv.schema
            else:
                self.observation_noise_dict, observation_noise_scale = observation_noise_scale, None
        elif exists(observation_noise_scale):
            observation_noise_scale = as_tensor(observation_noise_scale)

        self.register_buffer('observation_noise_scale', observation_noise_scale)

    def train(self, mode = True):
        super().train(mode)
        if exists(self.ema_critic):
            self.ema_critic.ema_model.eval()
            critic_norm = getattr(self.critic, 'state_normalizer', None)
            if isinstance(critic_norm, WelfordNormalizer):
                critic_norm.train(mode)
        return self

    def extract_obs(self, state: State, sim_context: Tensor | None = None) -> State:
        if exists(self.state_to_obs):
            return self.state_to_obs(state, sim_context)
        if exists(self.actor_obs_key):
            return select_key(state, self.actor_obs_key)
        return state

    @property
    def target_critic(self):
        return self.ema_critic.ema_model if exists(self.ema_critic) else self.critic

    @torch.no_grad()
    def update_ema(self):
        if exists(self.ema_critic):
            self.ema_critic.update()
        return self

    def maybe_one_hot_actions(self, actions, dtype):
        if exists(actions) and self.has_discrete_actions and not actions.is_floating_point():
            return to_one_hot(actions, self.num_discrete_actions, dtype)
        return actions

    def should_median_grad_clip(self, horizon: int, override: bool | None = None) -> bool:
        return default(override, default(self.use_median_grad_clip, horizon > 32))

    def critic_loss(
        self,
        states: State,          # (b t + 1 d)
        rewards,                # (b t)
        pred_terminal = None,   # (b t)
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

        structured_critic = getattr(self.critic, 'structured_input', False)
        states = tree_map_detach(states) if structured_critic else flatten_state(states).detach()
        rewards = rewards.detach()
        pred_terminal = pred_terminal.detach() if exists(pred_terminal) else None

        goals, sim_context, mask = align_batch((goals, sim_context, mask), size(rewards, '[b] t'))

        if exists(goals):
            goals = flatten_state(goals)
            num_states = size(next(t for t in tree_leaves(states) if is_tensor(t)), 'b [t] d')
            if goals.ndim == 3 and size(goals, 'b [t] d') > num_states:
                goals = goals[:, :num_states]

        with torch.no_grad():
            values = self.target_critic(states, goals, sim_context)
            returns = Critic.calc_returns(rewards, values, pred_terminal, discount_factor = discount_factor, lam = lam)

        if weight_by_survival and exists(pred_terminal):
            survival_weights = exclusive_cumprod(1. - pred_terminal, dim = -1)
            mask = mask * survival_weights if exists(mask) else survival_weights

        step_states = tree_map_tensor(lambda t: t[:, :-1], states)
        step_goals = goals[:, :-1] if exists(goals) and goals.ndim == 3 else goals

        critic_norm = getattr(self.critic, 'state_normalizer', None)
        shared_with_real = isinstance(critic_norm, WelfordNormalizer) and (
            critic_norm is getattr(self.world_model, 'state_normalizer', None) or critic_norm is self.obs_normalizer
        )
        norm_kwarg = dict(update_normalizer = False) if shared_with_real else dict()

        return self.critic(step_states, step_goals, sim_context, returns = returns, mask = mask, **norm_kwarg)

    def actor_loss(
        self,
        rewards,                # (b t) or (b * k t)
        pred_terminal = None,   # (b t) or (b * k t)
        bootstrap_value = None, # (b) or (b * k)
        final_state: State | None = None,     # (b d) or (b * k d)
        final_goal = None,      # (b g) or (b * k g)
        sim_context = None,     # (b c) or (b * k c)
        actions: State | None = None,         # (b t d_a) or (b * k t d_a)
        past_actions: State | None = None,    # (b d_a) or (b p d_a)
        past_mask: Tensor | None = None,      # (b p) or (b,)
        action_smoothness_weight: float | None = None,
        discount_factor = None,
        num_explorative_rollouts: int | None = None,
        use_median_grad_clip: bool | None = None,
        median_grad_clip_mult: float | None = None
    ):
        discount_factor = default(discount_factor, self.discount_factor)
        k = default(num_explorative_rollouts, self.num_explorative_rollouts)
        batch_size, horizon = shape(rewards, '[b] [t]')

        sim_context, final_goal, bootstrap_value, past_actions, past_mask = align_batch(
            (sim_context, final_goal, bootstrap_value, past_actions, past_mask),
            batch_size
        )

        if exists(pred_terminal):
            assert size(pred_terminal, 'b [t]') == size(rewards, 'b [t]')
            gates = discount_factor * (1. - pred_terminal)
        else:
            gates = torch.full_like(rewards, discount_factor)

        smoothness_weight = default(action_smoothness_weight, self.action_smoothness_weight)
        if smoothness_weight > 0.:
            assert exists(actions), '`actions` required when `action_smoothness_weight > 0`'
            past_actions = self.maybe_one_hot_actions(past_actions, rewards.dtype)
            rewards = rewards - smoothness_weight * action_smoothness_penalty(actions, past_actions, past_mask = past_mask)

        # bootstrap value from frozen critic, unless passed in

        if not exists(bootstrap_value) and exists(self.critic):
            assert exists(final_state), '`final_state` must be passed when `critic` is set'
            if exists(self.critic_state_key):
                final_state = select_key(final_state, self.critic_state_key)

            if not getattr(self.target_critic, 'structured_input', False):
                final_state = flatten_state(final_state)
            final_goal = flatten_state(final_goal) if exists(final_goal) else None

            with frozen(self.target_critic, eval = True):
                bootstrap_value = self.target_critic(final_state, final_goal, sim_context)

        if exists(bootstrap_value):
            rewards, _ = pack([rewards, bootstrap_value], 'b *')
            gates, _ = pack([gates, torch.ones_like(bootstrap_value)], 'b *')

        losses = select_explorative_loss(-self.assoc_scan(gates, rewards)[:, 0], k)

        if self.should_median_grad_clip(horizon, use_median_grad_clip) and losses.requires_grad:
            assert exists(self._action_probe), 'pass `use_median_grad_clip = True` to `SuperMimic` or `forward`'
            losses = median_clip_rollout_losses(losses, self._action_probe, clip_mult = default(median_grad_clip_mult, self.median_grad_clip_mult))

        return losses.mean()

    def forward(
        self,
        initial_state: State,
        sim_context: Tensor | None = None,
        actor_goals: State | None = None,
        actor_cond: Tensor | None = None,
        references: State | None = None,
        past_obs: State | None = None,
        past_actions: Tensor | None = None,
        past_mask: Tensor | None = None,
        num_rollout_timesteps = 32,
        *,
        num_explorative_rollouts: int | None = None,
        explorative_latent_scale: float | None = None,
        wm_explorative_latent_scale: float | None = None,
        update_normalizer: bool | None = None,
        use_median_grad_clip: bool | None = None
    ):
        actor, world_model = self.actor, self.world_model
        k = default(num_explorative_rollouts, self.num_explorative_rollouts)
        assert k >= 1

        orig_batch = batch_size_of(initial_state)

        wm_norm = getattr(world_model, 'state_normalizer', None)
        if isinstance(wm_norm, WelfordNormalizer):
            wm_norm.init_stats(initial_state)

        if exists(self.obs_normalizer):
            init_obs = self.extract_obs(initial_state, sim_context)
            if default(update_normalizer, self.training):
                self.obs_normalizer.update(init_obs)
            else:
                self.obs_normalizer.init_stats(init_obs)

        initial_state, sim_context, actor_goals, actor_cond, references, past_obs, past_actions, past_mask = repeat_batch(
            (initial_state, sim_context, actor_goals, actor_cond, references, past_obs, past_actions, past_mask),
            k
        )

        state, inverse_state = flatten_state_with_inverse(initial_state, match = self.observation_noise_schema)
        batch_size, device, dtype = size(state, '[b] ...'), state.device, state.dtype

        explorative_latent = None

        if self.dim_explorative_latent > 0:
            explorative_latent = sample_explorative_latent(
                orig_batch,
                k,
                self.dim_explorative_latent,
                antithetic = self.antithetic_explorative_latent,
                scale = default(explorative_latent_scale, self.explorative_latent_scale),
                device = device,
                dtype = dtype
            )

        if exists(actor_goals) and exists(self.goal_window):
            actor_goals = tree_map_tensor(lambda t: t[:, :num_rollout_timesteps], self.goal_window(actor_goals))

        past_actions = self.maybe_one_hot_actions(past_actions, dtype)

        obs_cache = None
        if exists(self.obs_history) and exists(past_obs):
            norm_past_obs = self.obs_normalizer.normalize(past_obs) if exists(self.obs_normalizer) else past_obs
            obs_cache = self.obs_history.push(norm_past_obs, mask = past_mask)

        act_cache = None
        if exists(self.action_history):
            act_cache = self.action_history.init_cache(
                batch_size,
                self.dim_action,
                past = past_actions,
                mask = past_mask,
                device = device,
                dtype = dtype
            )

        # rollout actor within frozen world model

        states, actions, rewards, terminals = [state], [], [], []

        probe_actions = torch.is_grad_enabled() and self.should_median_grad_clip(num_rollout_timesteps, use_median_grad_clip)
        action_probe = None

        obs_noise = default(self.observation_noise_dict, self.observation_noise_scale)
        structured = getattr(actor, 'structured_input', False)
        is_latent_wm = hasattr(world_model, 'encode') and hasattr(world_model, 'forward_from_latent')

        if not structured and exists(explorative_latent):
            actor_cond = safe_cat((actor_cond, explorative_latent), dim = -1)

        explorative_kwarg = dict(explorative_latent = explorative_latent) if exists(explorative_latent) else dict()

        wm_dim_latent = getattr(world_model, 'dim_explorative_latent', 0)
        wm_latent_scale = default(wm_explorative_latent_scale, self.wm_explorative_latent_scale)
        wm_explorative_kwarg = dict()

        if wm_dim_latent > 0 and wm_latent_scale > 0.:
            wm_latent = torch.randn((orig_batch, wm_dim_latent), device = device, dtype = dtype) * (wm_latent_scale * (wm_dim_latent ** -0.5))
            wm_explorative_kwarg = dict(explorative_latent = repeat_batch(wm_latent, k))

        with frozen(world_model, eval = True):
            latent = world_model.encode(state) if is_latent_wm else None

            for t in range(num_rollout_timesteps):
                obs = self.extract_obs(inverse_state(state), sim_context)
                obs = add_observation_noise(obs, obs_noise)

                if exists(self.obs_normalizer):
                    obs = self.obs_normalizer.normalize(obs)

                if exists(self.obs_history):
                    obs, obs_cache = self.obs_history.step(obs, obs_cache)

                past_action = self.action_history.read(act_cache) if exists(self.action_history) else None
                actor_goal = tree_map_tensor(lambda x: x[:, t], actor_goals) if exists(actor_goals) else None

                if structured:
                    action = actor(obs, goal = actor_goal, command = actor_cond, past_action = past_action, **explorative_kwarg)
                else:
                    action = actor(compact(
                        flatten_window(obs),
                        flatten_window(past_action),
                        flatten_window(actor_goal),
                        flatten_window(actor_cond)
                    ))

                if self.has_discrete_actions:
                    if is_tensor(action) and len(self.num_discrete_actions) > 1:
                        action = action.split(self.num_discrete_actions, dim = -1)

                    action = MultiCategorical(action).sample(
                        temperature = self.discrete_action_temperature,
                        differentiable = True
                    )

                if probe_actions and action.requires_grad:
                    if not exists(action_probe):
                        action_probe = action.new_zeros((batch_size, num_rollout_timesteps, *action.shape[1:]), requires_grad = True)
                    action = action + action_probe[:, t]

                if exists(self.action_history):
                    act_cache = self.action_history.push(action, act_cache)

                if is_latent_wm:
                    pred = world_model.forward_from_latent(latent, action, sim_context, **wm_explorative_kwarg)
                    latent = pred.latent
                else:
                    pred = world_model(state, action, sim_context, **wm_explorative_kwarg)

                state = pred.state
                states.append(state)
                actions.append(action)
                rewards.append(pred.reward)
                terminals.append(pred.terminal)

        states = inverse_state(rearrange(states, 't b ... -> b t ...'))
        actions = rearrange(actions, 't b ... -> b t ...')

        if exists(self.reward_fn) or exists(self.termination_fn):
            next_states = tree_map_tensor(lambda x: x[:, 1:], states)

        if exists(self.reward_fn):
            rewards = self.reward_fn(
                next_states,
                references,
                actions = actions,
                past_actions = past_actions,
                past_mask = past_mask,
                sim_context = sim_context
            )
        else:
            rewards = rearrange(rewards, 't b -> b t') if rewards and exists(rewards[0]) else None

        if exists(self.termination_fn):
            terminals = self.termination_fn(next_states, references)
        else:
            terminals = rearrange(terminals, 't b -> b t') if terminals and exists(terminals[0]) else None

        self._action_probe = action_probe

        return Rollout(states, actions, terminals, rewards)
