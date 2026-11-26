from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, as_tensor, cat, is_tensor
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def welford_combine(count, mean, m2, other_count, other_mean, other_m2):
    new_count = count + other_count
    safe_count = new_count.clamp_min(1.)
    delta = other_mean - mean

    new_mean = mean + delta * (other_count / safe_count)
    new_m2 = m2 + other_m2 + delta.square() * (count * other_count / safe_count)

    return new_count, new_mean, new_m2

def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def all_gather_welford(count: Tensor, mean: Tensor, m2: Tensor):
    num_stats = mean.numel()
    local_stats = cat((count.reshape(1), mean.reshape(-1), m2.reshape(-1)))

    if dist.get_backend() == 'nccl' and not local_stats.is_cuda:
        local_stats = local_stats.cuda()

    gathered = [torch.empty_like(local_stats) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_stats)

    total_count = count.new_zeros(())
    total_mean = torch.zeros_like(mean)
    total_m2 = torch.zeros_like(m2)

    for stats in gathered:
        other_count = stats[0].to(mean.device)
        other_mean = stats[1:1 + num_stats].reshape_as(mean).to(mean.device)
        other_m2 = stats[1 + num_stats:].reshape_as(m2).to(mean.device)

        total_count, total_mean, total_m2 = welford_combine(
            total_count, total_mean, total_m2,
            other_count, other_mean, other_m2
        )

    return total_count, total_mean, total_m2

# pytree & subset resolvers

def flatten_data(data: Tensor | dict):
    if is_tensor(data):
        return data, identity

    leaves, spec = tree_flatten(data)
    dims = [t.shape[-1] for t in leaves]
    return cat(leaves, dim = -1), lambda t: tree_unflatten(t.split(dims, dim = -1), spec)

def flatten_spec(target, spec, leaf_fn, default_val = None) -> Tensor:
    if is_tensor(target) or isinstance(target, int):
        return leaf_fn(target.shape[-1] if is_tensor(target) else target, spec)

    assert isinstance(target, dict) and len(target) > 0

    if isinstance(spec, str):
        spec = {spec: True}
    elif isinstance(spec, (list, tuple, set)) and all(isinstance(k, str) for k in spec):
        spec = {k: True for k in spec}

    if isinstance(spec, dict):
        assert set(spec) <= set(target), f'unknown keys {sorted(set(spec) - set(target))}'
        children = {k: flatten_spec(v, spec.get(k, default_val), leaf_fn, default_val) for k, v in target.items()}
    else:
        children = {k: flatten_spec(v, spec, leaf_fn, default_val) for k, v in target.items()}

    leaves, _ = tree_flatten(children)
    return cat(leaves, dim = -1)

def leaf_mask(dim_size: int, spec) -> Tensor:
    if not exists(spec) or isinstance(spec, bool):
        return torch.full((dim_size,), spec is not False, dtype = torch.bool)

    if is_tensor(spec) and spec.dtype == torch.bool:
        assert spec.shape == (dim_size,)
        return spec.detach().cpu().clone()

    if isinstance(spec, (list, tuple)) and spec and all(isinstance(x, bool) for x in spec):
        assert len(spec) == dim_size
        return torch.tensor(spec, dtype = torch.bool)

    mask = torch.zeros(dim_size, dtype = torch.bool)
    for item in (spec if isinstance(spec, (list, tuple, set)) else (spec,)):
        mask[item] = True
    return mask

def leaf_min_std(dim_size: int, spec, dtype: torch.dtype) -> Tensor:
    t = as_tensor(default(spec, 0.), dtype = dtype)
    assert (t >= 0.).all(), '`min_std` must be non-negative'
    if t.ndim == 0:
        return t.expand(dim_size).clone()
    assert t.shape == (dim_size,), f'expected min_std of shape ({dim_size},), got {tuple(t.shape)}'
    return t.clone()

# running standardization with welford (flat tensors or nested dicts, optional subset / exclude)
#
# stats shape follows `dim`
#   vectors / sequences (... d) -> per-feature stats, dim = -1 (default)
#   images (b c ...)            -> per-channel stats, dim = 1
#   dim = None                  -> a single global mean / variance

class WelfordNormalizer(Module):
    def __init__(
        self,
        data_shape: int | tuple[int, ...] | dict | None = None,
        dim: int | None = -1,
        eps: float = 1e-5,
        min_std: float | Tensor | dict | None = 1e-2,
        min_count: int = 2,
        max_count: float | None = None,
        subset = None,
        exclude = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        assert min_count >= 1, '`min_count` must be >= 1'
        assert not exists(max_count) or max_count >= min_count, '`max_count` must be >= `min_count`'
        assert exists(dim) or not (exists(subset) or exists(exclude)), '`subset` / `exclude` requires a feature `dim`'

        self.dim = dim
        self.eps = eps
        self.min_count = min_count
        self.max_count = float(max_count) if exists(max_count) else None
        self.subset = subset
        self.exclude = exclude
        self.stats_dtype = dtype
        self.frozen = False

        self._initialized = False

        if isinstance(min_std, dict):
            min_std_buf, self.min_std_spec = torch.zeros(1, dtype = dtype), min_std
        elif exists(min_std):
            t = as_tensor(min_std, dtype = dtype)
            assert t.ndim <= 1 and (t >= 0.).all(), '`min_std` must be a non-negative scalar, 1D tensor, or dict'
            min_std_buf = None if (t.ndim == 0 and t.item() == 0.) else t.clone()
            self.min_std_spec = None
        else:
            min_std_buf, self.min_std_spec = None, None

        # identity until min_count samples have been seen

        has_subset = exists(subset) or exists(exclude)

        self.register_buffer('mean', torch.zeros(1, dtype = dtype))
        self.register_buffer('m2', torch.zeros(1, dtype = dtype))
        self.register_buffer('count', torch.zeros((), dtype = dtype))
        self.register_buffer('subset_mask', torch.ones(1, dtype = torch.bool) if has_subset else None)
        self.register_buffer('min_std', min_std_buf)

        if exists(data_shape):
            self.init_stats(data_shape)

    # shape & lazy init

    def _resolve_dim(self, ndim: int) -> int | None:
        if not exists(self.dim):
            return None

        dim = self.dim + ndim if self.dim < 0 else self.dim
        assert 0 <= dim < ndim, f'dim {self.dim} is out of range for {ndim} dims'
        return dim

    def _broadcast_shape(self, ndim: int) -> tuple[int, ...]:
        dim = self._resolve_dim(ndim)
        return () if not exists(dim) else tuple(self.mean.numel() if i == dim else 1 for i in range(ndim))

    @torch.no_grad()
    def init_stats(self, spec):
        if self._initialized:
            return self

        device = self.mean.device

        if isinstance(spec, dict):
            assert self.dim == -1, 'dict states require dim = -1'
            leaves, _ = tree_flatten(spec)
            device = leaves[0].device if is_tensor(leaves[0]) else device
            num_features = sum(v if isinstance(v, int) else v.shape[-1] for v in leaves)
            target = spec
        else:
            device = spec.device if is_tensor(spec) else device
            shape = spec.shape if is_tensor(spec) else ((1, spec) if isinstance(spec, int) else (1, *tuple(spec)))
            dim = self._resolve_dim(len(shape))
            num_features = target = shape[dim] if exists(dim) else None

        stat_shape = () if not exists(num_features) else (num_features,)
        self.mean = torch.zeros(stat_shape, dtype = self.stats_dtype, device = device)
        self.m2 = torch.zeros(stat_shape, dtype = self.stats_dtype, device = device)
        self.count = torch.zeros((), dtype = self.stats_dtype, device = device)

        if exists(self.subset) or exists(self.exclude):
            mask = flatten_spec(target, self.subset, leaf_mask, default_val = False)
            if exists(self.exclude):
                mask = mask & ~flatten_spec(target, self.exclude, leaf_mask, default_val = False)
            self.subset_mask = mask.to(device = device)

        if exists(self.min_std_spec):
            ms = flatten_spec(target, self.min_std_spec, lambda d, s: leaf_min_std(d, s, self.stats_dtype), default_val = 0.)
            self.min_std = ms.to(device = device)
        elif exists(self.min_std) and self.min_std.ndim == 1:
            assert self.min_std.shape == stat_shape, f'expected min_std of shape {stat_shape}, got {tuple(self.min_std.shape)}'

        self._initialized = True
        return self

    def _prepare(self, data: Tensor | dict):
        if not self._initialized:
            self.init_stats(data)

        if isinstance(data, dict):
            assert self.dim == -1, 'dict states require dim = -1'

        flat, inverse = flatten_data(data)

        if exists(self.dim):
            dim = self._resolve_dim(flat.ndim)
            assert flat.shape[dim] == self.mean.numel(), f'expected size {self.mean.numel()} at dim {self.dim}, got {flat.shape[dim]}'

        return flat, inverse

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        mean_key = prefix + 'mean'
        if mean_key in state_dict and (not self._initialized or self.mean.shape != state_dict[mean_key].shape):
            device = self.mean.device
            for name in ('mean', 'm2', 'subset_mask', 'min_std'):
                if (prefix + name) in state_dict:
                    buf = getattr(self, name)
                    dtype = buf.dtype if exists(buf) else state_dict[prefix + name].dtype
                    setattr(self, name, torch.zeros_like(state_dict[prefix + name], device = device, dtype = dtype))
            self._initialized = True

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def var(self) -> Tensor:
        return self.m2 / self.count.clamp_min(1.)

    @property
    def std(self) -> Tensor:
        _, std = self._stats(self.mean.device, self.mean.dtype)
        return std

    def stats(self) -> tuple[Tensor, Tensor]:
        return self._stats(self.mean.device, self.mean.dtype)

    @torch.no_grad()
    def reset(self):
        self.mean.zero_()
        self.m2.zero_()
        self.count.zero_()
        return self

    # freeze helpers

    def freeze(self):
        self.frozen = True
        return self

    def unfreeze(self):
        self.frozen = False
        return self

    # welford update

    @torch.no_grad()
    def update(self, data: Tensor | dict):
        if self.frozen:
            return self

        flat, _ = self._prepare(data)

        if flat.numel() == 0 or (exists(self.subset_mask) and not self.subset_mask.any()):
            return self

        # half precision stats are not worth the trouble

        if flat.dtype in (torch.float16, torch.bfloat16):
            flat = flat.float()

        resolved_dim = self._resolve_dim(flat.ndim)
        reduce_dims = tuple(i for i in range(flat.ndim) if i != resolved_dim)

        if not reduce_dims:
            batch_mean, batch_m2, batch_count = flat, torch.zeros_like(flat), flat.new_tensor(1.)
        else:
            batch_mean = flat.mean(dim = reduce_dims)
            batch_m2 = (flat - batch_mean.reshape(self._broadcast_shape(flat.ndim))).square().sum(dim = reduce_dims)
            batch_count = flat.new_tensor(flat.numel() / batch_mean.numel())

        if is_distributed():
            batch_count, batch_mean, batch_m2 = all_gather_welford(batch_count, batch_mean, batch_m2)

        count, mean, m2 = welford_combine(
            self.count.to(flat.device, flat.dtype),
            self.mean.to(flat.device, flat.dtype),
            self.m2.to(flat.device, flat.dtype),
            batch_count, batch_mean, batch_m2
        )

        if exists(self.max_count):
            cap = count.clamp_max(self.max_count)
            m2 = m2 * (cap / count.clamp_min(1.))
            count = cap

        if exists(self.subset_mask):
            mask = self.subset_mask.to(flat.device)
            mean = torch.where(mask, mean, torch.zeros_like(mean))
            m2 = torch.where(mask, m2, torch.zeros_like(m2))

        self.count = count.to(self.stats_dtype)
        self.mean = mean.to(self.stats_dtype)
        self.m2 = m2.to(self.stats_dtype)
        return self

    # normalization

    def _stats(self, device, dtype, ndim: int | None = None):
        mean = self.mean.to(device = device, dtype = dtype)
        m2 = self.m2.to(device = device, dtype = dtype)
        count = self.count.to(device = device, dtype = dtype)

        std = (m2 / count.clamp_min(1.) + self.eps).sqrt()

        if exists(self.min_std):
            std = std.maximum(self.min_std.to(device = device, dtype = dtype))

        ready = count >= self.min_count
        if exists(self.subset_mask):
            ready = ready & self.subset_mask.to(device = device)

        mean = torch.where(ready, mean, torch.zeros_like(mean))
        std = torch.where(ready, std, torch.ones_like(std))

        if exists(ndim):
            bcast = self._broadcast_shape(ndim)
            mean, std = mean.reshape(bcast), std.reshape(bcast)

        return mean, std

    def normalize(self, data: Tensor | dict):
        flat, inverse = self._prepare(data)
        mean, std = self._stats(flat.device, flat.dtype, flat.ndim)
        return inverse((flat - mean) / std)

    def denormalize(self, data: Tensor | dict):
        flat, inverse = self._prepare(data)
        mean, std = self._stats(flat.device, flat.dtype, flat.ndim)
        return inverse(flat * std + mean)

    def forward(self, data: Tensor | dict, update: bool | None = None):
        if default(update, self.training):
            self.update(data)

        return self.normalize(data)
