from __future__ import annotations
from typing import Literal

import torch
from torch import Tensor, cat
from torch.nn import Module, SiLU, init
import torch.nn.functional as F

from einops import rearrange
from torch_einops_utils import pad_right_at_dim, safe_cat
from torch_einops_utils.shape import assert_shape, size
from x_mlps_pytorch import MLP
from x_mlps_pytorch.normed_mlp import MLP as NormedMLP

from supermimic_pytorch.supermimic import (
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
        assert num_obs_frames >= 1 and num_action_frames >= 0 and dim_explorative_latent >= 0
        assert latent_space in ('box', 'sphere', 'unbounded')

        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.dim_motion = dim_motion
        self.dim_latent = dim_latent
        self.num_obs_frames = num_obs_frames
        self.num_action_frames = num_action_frames
        self.dim_explorative_latent = dim_explorative_latent
        self.latent_space = latent_space
        self.latent_noise_scale = float(latent_noise_scale)
        self.squash_action = bool(squash_action)

        act_fn = default(activation, SiLU())
        mlp_klass = NormedMLP if normed_mlp else MLP

        self.motion_encoder = default(
            motion_encoder,
            mlp_klass(dim_motion, *encoder_hiddens, dim_latent, activation = act_fn)
        )

        self.dim_policy_in = num_obs_frames * dim_obs + num_action_frames * dim_action + dim_latent + dim_explorative_latent
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
        past_action: Tensor | None = None,
        explorative_latent: Tensor | None = None
    ) -> Tensor:
        z = self.project_latent(latent)
        features = safe_cat((flatten_window(obs), flatten_window(past_action), z), dim = -1)

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
        past_action: Tensor | None = None,
        explorative_latent: Tensor | None = None
    ) -> Tensor:
        z = default(latent, command)

        if not exists(z):
            assert exists(goal), 'pass `latent`, `command`, or `goal` to `LatentMotionActor`'
            z = self.encode(goal)

        return self.act(obs, z, past_action = past_action, explorative_latent = explorative_latent)
