import pytest
import torch
from torch import nn

from x_mlps_pytorch import MLP

from supermimic_pytorch import Critic, SuperMimic, WorldModel
from supermimic_pytorch.adversarial import AdversarialCritic, AdversarialSuperMimic

def has_grads(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

@pytest.mark.parametrize('loss_type', ('r3gan', 'lsgan'))
@pytest.mark.parametrize('detach_motion_reward', (False, True))
def test_adversarial_supermimic_e2e(loss_type, detach_motion_reward):
    batch_size, horizon = 2, 4
    dim_state, dim_action, dim_ctx = 10, 4, 3

    world_model = WorldModel(
        dim_state = dim_state,
        dim_action = dim_action,
        dim_sim_context = dim_ctx,
        dim_hiddens = (24,),
        pred_residual_state = False,
        pred_reward = True,
        pred_terminal = False
    )

    actor = MLP(dim_state, 24, dim_action, activation = nn.GELU())
    value_critic = Critic(dim_state = dim_state, dim_goal = 0, dim_sim_context = dim_ctx, dim_hiddens = (16,))
    supermimic = SuperMimic(actor = actor, world_model = world_model, critic = value_critic)

    motion_critic = AdversarialCritic(
        dim_feature = 6,
        window_size = 2,
        dim_hiddens = (24,),
        feature_fn = slice(0, 6),
        loss_type = loss_type
    )

    dynamics_critic = AdversarialCritic(
        dim_feature = dim_state,
        window_size = 2,
        dim_action = dim_action,
        dim_cond = dim_ctx,
        dim_hiddens = (24,),
        loss_type = loss_type
    )

    adapter = AdversarialSuperMimic(
        supermimic,
        motion_critic = motion_critic,
        dynamics_critic = dynamics_critic,
        motion_reward_weight = 0.5,
        pessimism_weight = 0.1,
        world_model_adv_weight = 0.1,
        detach_motion_reward = detach_motion_reward
    )

    s0 = torch.randn(batch_size, dim_state)
    actions = torch.randn(batch_size, horizon, dim_action)
    next_states = torch.randn(batch_size, horizon, dim_state)
    sim_context = torch.randn(batch_size, dim_ctx)

    # 1. fit world model (+ adversarial realism) and update dynamics critic (real sim vs imagined)

    wm = adapter.world_model_rollout(s0, actions, next_states, sim_context = sim_context, next_rewards = torch.randn(batch_size, horizon))
    wm.loss.backward()
    assert has_grads(world_model) and not has_grads(dynamics_critic)
    world_model.zero_grad()

    adapter.dynamics_critic_loss(wm.real_states, wm, real_actions = actions, real_cond = sim_context).backward()
    assert has_grads(dynamics_critic) and not has_grads(world_model)
    dynamics_critic.zero_grad()

    # 2. imagine rollout (+ AMP style + pessimism rewards) and update motion critic (unpaired mocap vs rollout)

    rollout = adapter(s0, sim_context = sim_context, num_rollout_timesteps = horizon)
    assert rollout.rewards.shape == (batch_size, horizon)
    assert rollout.motion_rewards.requires_grad == (not detach_motion_reward)

    ref_mocap = torch.randn(4, 2, dim_state)
    adapter.motion_critic_loss(ref_mocap, rollout).backward()
    assert has_grads(motion_critic) and not has_grads(actor) and not has_grads(world_model)
    motion_critic.zero_grad()

    # 3. update value critic and actor via BPTT

    adapter.critic_loss(rollout.states, rollout.rewards, sim_context = sim_context).backward()
    assert has_grads(value_critic)
    value_critic.zero_grad()

    adapter.actor_loss(rollout.rewards, final_state = rollout.states[:, -1], sim_context = sim_context).backward()
    assert has_grads(actor)
    assert not has_grads(world_model)
    assert not has_grads(motion_critic)
    assert not has_grads(dynamics_critic)
    assert not has_grads(value_critic)
