import pytest
import torch
from torch import nn

from supermimic_pytorch import (
    Critic,
    HistoryWindow,
    LatentMotionActor,
    SuperMimic,
    WorldModel,
    latent_uniformity_loss,
)

def has_nonzero_grads(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

@pytest.mark.parametrize('latent_space', ('box', 'sphere', 'unbounded'))
@pytest.mark.parametrize('use_history', (False, True))
def test_latent_motion_actor_bptt(latent_space, use_history):
    torch.manual_seed(0)

    b, t, dim_state, dim_action, dim_motion, dim_latent = 4, 5, 12, 6, 24, 2
    obs_win, act_win = (2, 1) if use_history else (1, 0)

    world_model = WorldModel(
        dim_state = dim_state,
        dim_action = dim_action,
        dim_sim_context = 0,
        dim_hiddens = (32, 32),
        pred_residual_state = False,
        pred_terminal = False,
    )
    actor = LatentMotionActor(
        dim_obs = dim_state,
        dim_action = dim_action,
        dim_motion = dim_motion,
        dim_latent = dim_latent,
        num_obs_frames = obs_win,
        num_action_frames = act_win,
        dim_explorative_latent = 2,
        latent_space = latent_space,
        latent_noise_scale = 0.1,
        encoder_hiddens = (32,),
        policy_hiddens = (32, 32),
    )
    critic = Critic(dim_state = dim_state, dim_goal = dim_latent, dim_sim_context = 0, dim_hiddens = (32,))

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        obs_history = HistoryWindow(obs_win, dim_state, pad_mode = 'replicate') if use_history else None,
        action_history = HistoryWindow(act_win, dim_action, lag = 1, pad_mode = 'zero') if use_history else None,
        num_explorative_rollouts = 2,
        dim_explorative_latent = 2,
        explorative_latent_scale = 0.2,
    )

    motions = torch.randn(b, 4, 6)
    z = actor.encode(motions)
    assert z.shape == (b, dim_latent)

    if latent_space == 'box':
        assert (z.abs() <= 1.0 + 1e-6).all()
    elif latent_space == 'sphere':
        assert torch.allclose(z.norm(dim = -1), torch.ones(b), atol = 1e-5)

    rollout = supermimic(
        torch.randn(b, dim_state),
        actor_cond = z,
        past_obs = torch.randn(b, 1, dim_state) if use_history else None,
        past_actions = torch.randn(b, 1, dim_action) if use_history else None,
        num_rollout_timesteps = t,
    )

    assert rollout.states.shape == (b * 2, t + 1, dim_state)
    assert rollout.actions.shape == (b * 2, t, dim_action)
    assert (rollout.actions.abs() <= 1.0 + 1e-6).all()

    rewards = -rollout.states[:, 1:].square().mean(dim = -1)
    loss = supermimic.actor_loss(rewards, final_state = rollout.states[:, -1], final_goal = z) + 0.1 * actor.uniformity_loss(motions)
    loss.backward()

    assert has_nonzero_grads(actor.policy)
    assert has_nonzero_grads(actor.motion_encoder)
    assert not any(p.grad is not None for p in world_model.parameters())

def test_latent_uniformity_loss():
    torch.manual_seed(0)
    raw = nn.Parameter(torch.randn(6, 2) * 0.05)
    opt = torch.optim.Adam([raw], lr = 0.05)

    for _ in range(200):
        opt.zero_grad()
        latent_uniformity_loss(raw.tanh(), temperature = 2.5, max_bound = 0.75, bound_weight = 20.0).backward()
        opt.step()

    with torch.no_grad():
        z = raw.tanh()
        dists = torch.cdist(z, z).fill_diagonal_(float('inf'))
        assert dists.min().item() > 0.55
        assert z.abs().max().item() < 0.88

def test_latent_motion_actor_with_latent_action_wrapper():
    env_ssl = pytest.importorskip('env_ssl_wrapper')
    import gymnasium as gym
    import numpy as np

    class ToyHandEnv(gym.Env):
        observation_space = gym.spaces.Box(-5., 5., shape = (10,), dtype = np.float32)
        action_space = gym.spaces.Box(-1., 1., shape = (4,), dtype = np.float32)

        def reset(self, *, seed = None, options = None):
            super().reset(seed = seed)
            self.t = 0
            return np.zeros(10, dtype = np.float32), {}

        def step(self, action):
            self.t += 1
            obs = np.concatenate((np.asarray(action, dtype = np.float32), np.zeros(6, dtype = np.float32)))
            return obs, float(np.asarray(action).mean()), self.t >= 12, False, {}

    actor = LatentMotionActor(
        dim_obs = 6,
        dim_action = 4,
        dim_motion = 12,
        dim_latent = 2,
        num_obs_frames = 2,
        num_action_frames = 1,
        dim_explorative_latent = 2,
        encoder_hiddens = (16,),
        policy_hiddens = (16,),
    )

    wrapped = env_ssl.LatentActionWrapper(
        [ToyHandEnv() for _ in range(3)],
        actor,
        horizon = 4,
        obs_fn = slice(0, 6),
        obs_history_len = 2,
        action_history_len = 1,
        freeze = True,
        gamma = 0.99,
        clip_lower_actions = True,
    )
    env = env_ssl.StandardizeEnvWrapper(wrapped)

    obs, _ = env.reset(seed = 0)
    assert obs.shape == (3, 10) and env.action_space.shape == (2,)

    obs, rew, _, _, info = env.step(torch.randn(3, 2).clamp(-1., 1.))
    assert obs.shape == (3, 10) and rew.shape == (3,)
    assert info['duration'].tolist() == [4.0, 4.0, 4.0]
