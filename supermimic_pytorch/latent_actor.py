from __future__ import annotations
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, as_tensor, cat
from torch.nn import Module, SiLU, init
import torch.nn.functional as F

from einops import rearrange, repeat
from torch_einops_utils import pad_right_at_dim, safe_cat
from torch_einops_utils.shape import assert_shape, size
from x_mlps_pytorch import MLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from supermimic_pytorch.supermimic import (
    HistoryWindow,
    State,
    default,
    exists,
    flatten_state,
    flatten_window
)

LatentSpace = Literal['box', 'sphere', 'unbounded']

# wang & isola uniformity repulsion + soft box bound

def latent_uniformity_loss(
    latents: Tensor,
    *,
    temperature: float = 2.0,
    max_bound: float | None = None,
    bound_weight: float = 15.0,
    eps: float = 1e-8
) -> Tensor:
    assert latents.ndim == 2, f'expected 2D latents (m, d), got {tuple(latents.shape)}'
    num_motions = size(latents, '[m] d')

    if num_motions < 2:
        loss = latents.new_zeros(())
    else:
        sq_dists = torch.cdist(latents, latents, p = 2).square()
        off_diag = ~torch.eye(num_motions, dtype = torch.bool, device = latents.device)
        loss = torch.log(torch.exp(-temperature * sq_dists[off_diag]).mean().clamp(min = eps))

    if exists(max_bound) and bound_weight > 0.:
        loss = loss + float(bound_weight) * F.relu(latents.abs() - float(max_bound)).square().sum()

    return loss

# motion encoder + latent-conditioned motor policy

class LatentMotionActor(Module):
    structured_input = True

    def __init__(
        self,
        *,
        dim_obs: int,
        dim_action: int,
        dim_motion: int,
        dim_latent: int = 4,
        dim_mask: int = 0,
        mask_dropout: float = 0.0,
        num_obs_frames: int = 1,
        num_action_frames: int = 0,
        dim_explorative_latent: int = 0,
        latent_space: LatentSpace = 'box',
        latent_noise_scale: float = 0.0,
        encoder_hiddens: tuple[int, ...] = (256, 256),
        policy_hiddens: tuple[int, ...] = (256, 256, 256),
        activation: Module | None = None,
        normed_mlp: bool = True,
        squash_action: bool = True,
        action_init_gain: float = 0.01,
        motion_encoder: Module | None = None
    ):
        super().__init__()
        assert dim_obs > 0 and dim_action > 0 and dim_motion > 0 and dim_latent > 0
        assert dim_mask >= 0 and 0. <= mask_dropout <= 1.
        assert num_obs_frames >= 1 and num_action_frames >= 0 and dim_explorative_latent >= 0
        assert latent_space in ('box', 'sphere', 'unbounded')

        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.dim_motion = dim_motion
        self.dim_latent = dim_latent
        self.dim_mask = dim_mask
        self.mask_dropout = float(mask_dropout)
        self.num_obs_frames = num_obs_frames
        self.num_action_frames = num_action_frames
        self.dim_explorative_latent = dim_explorative_latent
        self.latent_space = latent_space
        self.latent_noise_scale = float(latent_noise_scale)
        self.squash_action = bool(squash_action)
        self.encoder_hiddens = tuple(encoder_hiddens)
        self.policy_hiddens = tuple(policy_hiddens)
        self.normed_mlp = bool(normed_mlp)

        act_fn = default(activation, SiLU())
        mlp_klass = NormedMLP if normed_mlp else MLP

        self.motion_encoder = default(
            motion_encoder,
            mlp_klass(dim_motion, *encoder_hiddens, dim_latent, activation = act_fn)
        )

        self.dim_policy_in = num_obs_frames * dim_obs + num_action_frames * dim_action + dim_latent + dim_mask + dim_explorative_latent
        self.policy = mlp_klass(self.dim_policy_in, *policy_hiddens, dim_action, activation = act_fn)

        if exists(action_init_gain):
            last = self.policy.layers[-1]
            linear = last[0] if normed_mlp else last
            init.orthogonal_(linear.weight, gain = action_init_gain)
            init.zeros_(linear.bias)

    def project_latent(self, z: Tensor, *, from_logits: bool = False) -> Tensor:
        if self.latent_space == 'sphere':
            return F.normalize(z, dim = -1, eps = 1e-8)
        if self.latent_space == 'box':
            return z.tanh() if from_logits else z.clamp(-1., 1.)
        return z

    def encode(
        self,
        motion: State,
        *,
        add_noise: bool | None = None,
        noise_scale: float | None = None
    ) -> Tensor:
        flat = flatten_state(motion)
        if flat.ndim > 2:
            flat = rearrange(flat, 'b ... -> b (...)')

        assert_shape(flat, 'b d', d = self.dim_motion)
        z = self.project_latent(self.motion_encoder(flat), from_logits = True)

        scale = default(noise_scale, self.latent_noise_scale)
        if default(add_noise, self.training) and scale > 0.:
            z = self.project_latent(z + torch.randn_like(z) * scale)

        return z

    def uniformity_loss(
        self,
        motions_or_latents: State,
        *,
        is_encoded: bool = False,
        temperature: float = 2.0,
        max_bound: float | None = None,
        bound_weight: float = 15.0
    ) -> Tensor:
        z = motions_or_latents if is_encoded else self.encode(motions_or_latents, add_noise = False)
        bound = 0.78 if (self.latent_space == 'box' and not exists(max_bound)) else max_bound
        return latent_uniformity_loss(z, temperature = temperature, max_bound = bound, bound_weight = bound_weight)

    def act(
        self,
        obs: State,
        latent: Tensor,
        *,
        mask: Tensor | None = None,
        past_action: Tensor | None = None,
        explorative_latent: Tensor | None = None
    ) -> Tensor:
        cond_mask = None
        if self.dim_mask > 0:
            if not exists(mask) and latent.shape[-1] == self.dim_latent + self.dim_mask:
                latent, mask = latent.split((self.dim_latent, self.dim_mask), dim = -1)
            if exists(mask):
                cond_mask = mask.to(latent)
                if self.training and self.mask_dropout > 0.:
                    keep = (torch.rand((*cond_mask.shape[:-1], 1), device = cond_mask.device) >= self.mask_dropout).to(cond_mask)
                    cond_mask = cond_mask * keep
            else:
                cond_mask = latent.new_zeros((*latent.shape[:-1], self.dim_mask))

        z = self.project_latent(latent)
        features = safe_cat((flatten_window(obs), flatten_window(past_action), z, cond_mask), dim = -1)

        if self.dim_explorative_latent > 0:
            features = (
                cat((features, explorative_latent), dim = -1)
                if exists(explorative_latent) else
                pad_right_at_dim(features, self.dim_explorative_latent, dim = -1)
            )

        assert_shape(features, '... d', d = self.dim_policy_in)
        action = self.policy(features)
        return action.tanh() if self.squash_action else action

    def forward(
        self,
        obs: State,
        latent: Tensor | None = None,
        *,
        goal: State | None = None,
        command: Tensor | None = None,
        mask: Tensor | None = None,
        past_action: Tensor | None = None,
        explorative_latent: Tensor | None = None
    ) -> Tensor:
        z = default(latent, command)

        if not exists(z):
            assert exists(goal), 'pass `latent`, `command`, or `goal` to `LatentMotionActor`'
            z = self.encode(goal)

        return self.act(obs, z, mask = mask, past_action = past_action, explorative_latent = explorative_latent)

# portable deployment & stage-2 wrapper around LatentMotionActor

class LatentPolicy(Module):
    structured_input = True

    def __init__(
        self,
        actor: LatentMotionActor,
        *,
        obs_mean: Tensor | None = None,
        obs_std: Tensor | None = None,
        action_offset: Tensor | None = None,
        action_scale: Tensor | None = None,
        clip_latents: Tensor | None = None,
        clip_names: tuple[str, ...] | list[str] = (),
        metadata: dict | None = None,
        normalize_obs: bool = True,
        unscale_action: bool = True
    ):
        super().__init__()
        self.actor = actor
        self.dim_obs = actor.dim_obs
        self.dim_action = actor.dim_action
        self.dim_motion = actor.dim_motion
        self.dim_latent = actor.dim_latent
        self.dim_mask = actor.dim_mask
        self.latent_space = actor.latent_space
        self.obs_window = actor.num_obs_frames
        self.action_window = actor.num_action_frames
        self.clip_names = tuple(clip_names)
        self.metadata = dict(metadata) if exists(metadata) else {}
        self.default_normalize_obs = bool(normalize_obs)
        self.default_unscale_action = bool(unscale_action)

        self.obs_history = HistoryWindow(self.obs_window, self.dim_obs, pad_mode = 'replicate')
        self.action_history = HistoryWindow(self.action_window, self.dim_action, lag = 1, pad_mode = 'zero') if self.action_window > 0 else None

        mean_t = torch.zeros(self.dim_obs) if not exists(obs_mean) else as_tensor(obs_mean, dtype = torch.float32)
        std_t = torch.ones(self.dim_obs) if not exists(obs_std) else as_tensor(obs_std, dtype = torch.float32)
        offset_t = torch.zeros(self.dim_action) if not exists(action_offset) else as_tensor(action_offset, dtype = torch.float32)
        scale_t = torch.ones(self.dim_action) if not exists(action_scale) else as_tensor(action_scale, dtype = torch.float32)

        assert_shape(mean_t, 'd', d = self.dim_obs)
        assert_shape(std_t, 'd', d = self.dim_obs)
        assert_shape(offset_t, 'd', d = self.dim_action)
        assert_shape(scale_t, 'd', d = self.dim_action)

        self.register_buffer('obs_mean', mean_t)
        self.register_buffer('obs_std', std_t.clamp(min = 1e-8))
        self.register_buffer('action_offset', offset_t)
        self.register_buffer('action_scale', scale_t.clamp(min = 1e-8))
        self.register_buffer(
            'clip_latents',
            as_tensor(clip_latents, dtype = torch.float32) if exists(clip_latents) else None
        )

    def normalize_obs(self, obs: Tensor) -> Tensor:
        return (obs - self.obs_mean.to(obs)) / self.obs_std.to(obs)

    def unscale_action(self, action: Tensor) -> Tensor:
        return self.action_offset.to(action) + self.action_scale.to(action) * action

    def scale_action(self, action: Tensor) -> Tensor:
        return (action - self.action_offset.to(action)) / self.action_scale.to(action)

    def project_latent(self, z: Tensor, *, from_logits: bool = False) -> Tensor:
        return self.actor.project_latent(z, from_logits = from_logits)

    def encode(self, motion: State, *, add_noise: bool = False) -> Tensor:
        return self.actor.encode(motion, add_noise = add_noise)

    def sample_latent(self, batch_size: int = 1, *, device = None, dtype: torch.dtype = torch.float32) -> Tensor:
        device = default(device, self.obs_mean.device)
        if self.latent_space == 'box':
            return torch.empty((batch_size, self.dim_latent), device = device, dtype = dtype).uniform_(-1., 1.)
        z = torch.randn((batch_size, self.dim_latent), device = device, dtype = dtype)
        return self.project_latent(z)

    def init_cache(self, batch_size: int = 1, *, device = None, dtype: torch.dtype = torch.float32):
        device = default(device, self.obs_mean.device)
        act_cache = self.action_history.init_cache(batch_size, device = device, dtype = dtype) if exists(self.action_history) else None
        return (None, act_cache)

    def reset_cache(self, cache, done: Tensor):
        if not exists(cache):
            return None
        obs_cache, act_cache = cache
        obs_cache = self.obs_history.reset(obs_cache, done) if exists(obs_cache) else None
        act_cache = self.action_history.reset(act_cache, done) if exists(self.action_history) and exists(act_cache) else None
        return (obs_cache, act_cache)

    def step(
        self,
        obs: Tensor,
        latent: Tensor,
        cache = None,
        *,
        mask: Tensor | None = None,
        reset: Tensor | None = None,
        normalize_obs: bool | None = None,
        unscale_action: bool | None = None
    ):
        squeeze = obs.ndim == 1
        if squeeze:
            obs = rearrange(obs, 'd -> 1 d')
        if latent.ndim == 1:
            latent = rearrange(latent, 'd -> 1 d')
        if exists(mask) and mask.ndim == 1:
            mask = rearrange(mask, 'd -> 1 d')

        if default(normalize_obs, self.default_normalize_obs):
            obs = self.normalize_obs(obs)

        obs_cache, act_cache = default(cache, self.init_cache(obs.shape[0], device = obs.device, dtype = obs.dtype))
        if exists(reset):
            obs_cache, act_cache = self.reset_cache((obs_cache, act_cache), reset)

        obs_w, obs_cache = self.obs_history.step(obs, obs_cache)
        past_act = self.action_history.read(act_cache) if exists(self.action_history) else None
        raw_act = self.actor.act(obs_w, latent, mask = mask, past_action = past_act)

        if exists(self.action_history):
            act_cache = self.action_history.push(raw_act, act_cache)

        out = self.unscale_action(raw_act) if default(unscale_action, self.default_unscale_action) else raw_act
        return (out[0] if squeeze else out), (obs_cache, act_cache)

    def forward(
        self,
        obs: Tensor,
        latent: Tensor | None = None,
        *,
        goal: State | None = None,
        command: Tensor | None = None,
        mask: Tensor | None = None,
        past_action: Tensor | None = None,
        normalize_obs: bool | None = None,
        unscale_action: bool | None = None
    ) -> Tensor:
        use_norm = default(normalize_obs, self.default_normalize_obs)
        use_unscale = default(unscale_action, self.default_unscale_action)

        if use_norm:
            obs = self.normalize_obs(obs)

        if self.obs_window > 1 and obs.ndim == 2:
            obs = repeat(obs, 'b d -> b w d', w = self.obs_window)

        if exists(past_action):
            if use_unscale:
                past_action = self.scale_action(past_action)
        elif self.action_window > 0:
            past_action = obs.new_zeros((obs.shape[0], self.action_window, self.dim_action))

        raw_act = self.actor(obs, latent, goal = goal, command = command, mask = mask, past_action = past_action)
        return self.unscale_action(raw_act) if use_unscale else raw_act

    def wrap_env(
        self,
        env,
        *,
        horizon: int = 8,
        obs_fn = None,
        gamma: float = 0.99,
        clip_lower_actions: bool = True,
        **kwargs
    ):
        try:
            from env_ssl_wrapper import LatentActionWrapper
        except ImportError as err:
            raise ImportError('`wrap_env` requires `env-ssl-wrapper` (`pip install env-ssl-wrapper`)') from err

        return LatentActionWrapper(
            env,
            self,
            horizon = horizon,
            obs_fn = obs_fn,
            obs_history_len = self.obs_window,
            action_history_len = self.action_window,
            obs_pad_mode = 'replicate',
            freeze = True,
            gamma = gamma,
            clip_lower_actions = clip_lower_actions,
            **kwargs
        )

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents = True, exist_ok = True)

        actor_kwargs = dict(
            dim_obs = self.actor.dim_obs,
            dim_action = self.actor.dim_action,
            dim_motion = self.actor.dim_motion,
            dim_latent = self.actor.dim_latent,
            dim_mask = self.actor.dim_mask,
            mask_dropout = self.actor.mask_dropout,
            num_obs_frames = self.actor.num_obs_frames,
            num_action_frames = self.actor.num_action_frames,
            dim_explorative_latent = self.actor.dim_explorative_latent,
            latent_space = self.actor.latent_space,
            latent_noise_scale = self.actor.latent_noise_scale,
            encoder_hiddens = self.actor.encoder_hiddens,
            policy_hiddens = self.actor.policy_hiddens,
            normed_mlp = self.actor.normed_mlp,
            squash_action = self.actor.squash_action,
        )

        payload = dict(
            format_version = 1,
            actor_kwargs = actor_kwargs,
            state_dict = {k: v.detach().cpu() for k, v in self.state_dict().items()},
            clip_names = self.clip_names,
            metadata = self.metadata,
            default_normalize_obs = self.default_normalize_obs,
            default_unscale_action = self.default_unscale_action,
        )
        torch.save(payload, out_path)
        return out_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location = 'cpu',
        normalize_obs: bool | None = None,
        unscale_action: bool | None = None
    ) -> LatentPolicy:
        ckpt = torch.load(Path(path), map_location = map_location, weights_only = False)
        actor = LatentMotionActor(**ckpt['actor_kwargs'])
        state_dict = ckpt['state_dict']

        policy = cls(
            actor,
            clip_latents = state_dict.get('clip_latents'),
            clip_names = ckpt.get('clip_names', ()),
            metadata = ckpt.get('metadata', {}),
            normalize_obs = default(normalize_obs, ckpt.get('default_normalize_obs', True)),
            unscale_action = default(unscale_action, ckpt.get('default_unscale_action', True)),
        )
        policy.load_state_dict(state_dict)
        policy.eval()
        return policy

load_policy = LatentPolicy.load
