from __future__ import annotations
from typing import Literal

import torch
from torch import Tensor, cat, is_tensor
from torch.nn import Module, Parameter

import einx
from einops import pack, rearrange, reduce
from torch_einops_utils import (
    pad_left_at_dim,
    pad_right_at_dim,
    shift_left,
    shift_right,
    slice_right_at_dim,
    tree_flatten_with_inverse
)
from torch_einops_utils.shape import shape, size

# constants

PadMode = Literal['zero', 'replicate', 'learned']

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def flatten_tree(tree):
    if is_tensor(tree):
        return tree, identity

    leaves, unflatten_leaves = tree_flatten_with_inverse(tree)
    leaf_dims = [size(leaf, '... d -> d') for leaf in leaves]

    def unflatten(packed: Tensor):
        return unflatten_leaves(packed.split(leaf_dims, dim = -1))

    return cat(leaves, dim = -1), unflatten

def to_sequence_and_mask(inputs, mask: Tensor | None = None):
    sequence, unflatten = flatten_tree(inputs)
    device = sequence.device

    sequence, _ = pack([sequence], 'b * d')
    batch_size, seq_len = shape(sequence, 'b t d -> b t')

    if exists(mask):
        mask, _ = pack([mask], 'b *')
        mask = mask.to(device = device, dtype = torch.bool)
    else:
        mask = torch.ones((batch_size, seq_len), dtype = torch.bool, device = device)

    return sequence, mask, unflatten

def unfold_time(
    tensor: Tensor,
    window_size: int,
    lag: int = 0,
    future: bool = False,
    pad_value = 0.
) -> Tensor:
    seq_len = size(tensor, 'b t ... -> t')

    if lag >= seq_len:
        tensor = torch.full_like(tensor, pad_value)
    elif lag > 0:
        shift_fn = shift_left if future else shift_right
        tensor = shift_fn(tensor, lag, dim = 1, pad_value = pad_value)

    pad_fn = pad_right_at_dim if future else pad_left_at_dim
    padded = pad_fn(tensor, window_size - 1, dim = 1, value = pad_value)
    return padded.unfold(1, window_size, 1)

# action smoothness penalty

def action_smoothness_penalty(
    actions,
    past_actions = None,
    *,
    past_mask: Tensor | None = None,
    mask: Tensor | None = None,
    order: int = 1,
    episode_starts: Tensor | None = None
) -> Tensor:
    actions, _ = flatten_tree(actions)

    if exists(past_actions):
        past_actions, _ = flatten_tree(past_actions)

    history = HistoryWindow(order + 1)

    windows, valid = history(
        actions,
        past = past_actions,
        past_mask = past_mask,
        mask = mask,
        episode_starts = episode_starts,
        return_mask = True
    )

    action_diff = torch.diff(windows, n = order, dim = 2)
    action_diff = rearrange(action_diff, 'b t 1 d -> b t d')

    squared_diff = reduce(action_diff.square(), 'b t d -> b t', 'sum')
    all_valid = valid.all(dim = -1)

    return einx.where('b t, b t, -> b t', all_valid, squared_diff, 0.)

# classes

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

        if isinstance(dim, dict):
            dim = sum(tree_flatten_with_inverse(dim)[0])

        assert window_size >= 1 and lag >= 0 and pad_mode in ('zero', 'replicate', 'learned')
        assert pad_mode != 'learned' or exists(dim), '`dim` required for learned pad embed'

        self.window_size = window_size
        self.dim = dim
        self.lag = lag
        self.future = future
        self.pad_mode = pad_mode
        self.per_slot_pad = per_slot_pad

        pad_shape = (window_size, dim) if per_slot_pad else (dim,)
        self.pad_embed = Parameter(torch.zeros(pad_shape)) if pad_mode == 'learned' else None

    @property
    def dim_out(self) -> int:
        return self.window_size * self.dim

    def fill(self, windows: Tensor, valid: Tensor) -> Tensor:
        device, dtype = windows.device, windows.dtype

        if exists(self.pad_embed):
            pad_embed = self.pad_embed.to(device = device, dtype = dtype)
            pattern = '... w, ... w d, w d -> ... w d' if self.per_slot_pad else '... w, ... w d, d -> ... w d'
            return einx.where(pattern, valid, windows, pad_embed)

        if self.pad_mode == 'replicate':
            if self.future:
                anchor_indices = self.window_size - 1 - valid.flip(-1).int().argmax(dim = -1)
            else:
                anchor_indices = valid.int().argmax(dim = -1)

            anchor = einx.get_at('... [w] d, ... -> ... d', windows, anchor_indices)
            anchor = einx.where('..., ... d, -> ... d', valid.any(dim = -1), anchor, 0.)
            return einx.where('... w, ... w d, ... d -> ... w d', valid, windows, anchor)

        return einx.where('... w, ... w d, -> ... w d', valid, windows, 0.)

    def step(
        self,
        frame = None,
        cache = None,
        *,
        mask: Tensor | None = None,
        reset: Tensor | None = None,
        batch_shape: tuple[int, int] | None = None,
        device = None,
        dtype = None
    ):
        frames, valid, unflatten = cache if exists(cache) else (None, None, identity)

        if exists(reset) and exists(valid):
            keep_mask = ~reset.to(device = valid.device, dtype = torch.bool)
            valid = einx.logical_and('b w, b -> b w', valid, keep_mask)

        if exists(frame):
            flat_frame, frame_mask, unflatten = to_sequence_and_mask(frame, mask)
            device, dtype = flat_frame.device, flat_frame.dtype
            batch_size, dim = shape(flat_frame, 'b t d -> b d')

            frames = default(frames, torch.zeros((batch_size, self.window_size, dim), device = device, dtype = dtype))
            valid = default(valid, torch.zeros((batch_size, self.window_size), device = device, dtype = torch.bool))

            frames, _ = pack([frames, flat_frame], 'b * d')
            valid, _ = pack([valid, frame_mask], 'b *')

            frames = slice_right_at_dim(frames, self.window_size, dim = 1)
            valid = slice_right_at_dim(valid, self.window_size, dim = 1)
        elif not exists(frames):
            batch_size, dim = batch_shape
            frames = torch.zeros((batch_size, self.window_size, dim), device = device, dtype = dtype)
            valid = torch.zeros((batch_size, self.window_size), device = device, dtype = torch.bool)

        window = unflatten(self.fill(frames, valid))
        return window, (frames, valid, unflatten)

    def forward(
        self,
        inputs,
        *,
        past = None,
        past_mask: Tensor | None = None,
        mask: Tensor | None = None,
        episode_starts: Tensor | None = None,
        return_mask: bool = False
    ):
        sequence, seq_mask, unflatten = to_sequence_and_mask(inputs, mask)
        device = sequence.device
        seq_len = size(sequence, 'b t d -> t')

        if exists(past):
            past_seq, past_seq_mask, _ = to_sequence_and_mask(past, past_mask)
            past_len = size(past_seq, 'b p d -> p')

            sequence, _ = pack([past_seq, sequence], 'b * d')
            seq_mask, _ = pack([past_seq_mask, seq_mask], 'b *')

            if exists(episode_starts):
                episode_starts = pad_left_at_dim(episode_starts, past_len, dim = 1, value = False)

        windows = unfold_time(sequence, self.window_size, self.lag, self.future, pad_value = 0.)
        windows = rearrange(windows, 'b t d w -> b t w d')

        valid = unfold_time(seq_mask, self.window_size, self.lag, self.future, pad_value = False)

        if exists(episode_starts):
            episode_ids = episode_starts.to(device = device, dtype = torch.long).cumsum(dim = 1)
            window_episode_ids = unfold_time(episode_ids, self.window_size, self.lag, self.future, pad_value = -1)
            same_episode = einx.equal('b t w, b t -> b t w', window_episode_ids, episode_ids)
            valid = valid & same_episode

        windows = slice_right_at_dim(windows, seq_len, dim = 1)
        valid = slice_right_at_dim(valid, seq_len, dim = 1)

        out = unflatten(self.fill(windows, valid))
        return (out, valid) if return_mask else out

# quick test

if __name__ == '__main__':
    obs_dims = dict(joint_pos = 29, joint_vel = 29, imu = 6)
    obs = {name: torch.randn(2, 6, dim) for name, dim in obs_dims.items()}
    actions = torch.randn(2, 6, 29)

    obs_window = HistoryWindow(4, obs_dims, pad_mode = 'learned')
    action_window = HistoryWindow(2, 29, lag = 1, pad_mode = 'learned')

    assert obs_window(obs)['joint_pos'].shape == (2, 6, 4, 29)
    assert action_window(actions).shape == (2, 6, 2, 29)
    assert action_smoothness_penalty(actions).shape == (2, 6)
