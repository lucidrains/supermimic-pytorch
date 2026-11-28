import pytest
import torch
from torch import nn

from x_mlps_pytorch import MLP

from supermimic_pytorch import (
    LatentWorldModel,
    SuperMimic,
    WorldModel,
    state_jacobian_hinge_loss
)

# helpers

def has_nonzero_grads(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

# tests

def test_jacobian_hinge_identity_and_contractive():
    s = torch.randn(8, 12, requires_grad = True)

    loss_id = state_jacobian_hinge_loss(s, s, margin = 1.)
    assert loss_id.shape == (8,)
    assert torch.allclose(loss_id, torch.zeros_like(loss_id))

    loss_contract = state_jacobian_hinge_loss(0.5 * s, s, margin = 1.)
    assert torch.allclose(loss_contract, torch.zeros_like(loss_contract))

@pytest.mark.parametrize('num_probes', (1, 2))
def test_jacobian_hinge_expansive_and_second_order_grad(num_probes):
    scale = nn.Parameter(torch.tensor(2.0))
    s = torch.randn(6, 10, requires_grad = True)
    pred = scale * s

    loss = state_jacobian_hinge_loss(pred, s, margin = 1., num_probes = num_probes)
    assert torch.allclose(loss, torch.full((6,), 3.0))

    loss.mean().backward()
    assert scale.grad is not None
    assert torch.allclose(scale.grad, torch.tensor(4.0))

@pytest.mark.parametrize('dict_state', (False, True))
@pytest.mark.parametrize('use_normalizer', (False, True))
def test_world_model_jacobian_reg_rollout_and_eval(dict_state, use_normalizer):
    dim_state = 10 if dict_state else 8
    dim_action = 3

    def make_state(*leading):
        if dict_state:
            return dict(pos = torch.randn(*leading, 4), vel = torch.randn(*leading, 6))
        return torch.randn(*leading, dim_state)

    wm = WorldModel(
        dim_state = dim_state,
        dim_action = dim_action,
        dim_sim_context = 0,
        dim_hiddens = (16, 16),
        pred_reward = True,
        pred_terminal = True,
        state_normalizer = use_normalizer,
        jacobian_reg_weight = 0.1,
        jacobian_reg_margin = 1.0
    )

    with torch.no_grad():
        wm.to_state_pred.weight.normal_(0., 0.8)

    s0 = make_state(4)
    actions = torch.randn(4, 5, dim_action)
    next_states = make_state(4, 5)
    next_rewards = torch.randn(4, 5)
    next_terminals = torch.randint(0, 2, (4, 5)).float()

    wm.train()
    torch.manual_seed(0)
    loss_off, _, _ = wm.rollout(s0, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards, jacobian_reg_weight = 0., update_normalizer = False)
    torch.manual_seed(0)
    loss_on, _, _ = wm.rollout(s0, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards, update_normalizer = False)
    assert loss_on.item() > loss_off.item()

    loss_on.backward()
    assert has_nonzero_grads(wm)

    # eval mode disables jacobian reg

    wm.eval()
    with torch.no_grad():
        eval_default, _, _ = wm.rollout(s0, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards)
        eval_zero, _, _ = wm.rollout(s0, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards, jacobian_reg_weight = 0.)
    assert torch.allclose(eval_default, eval_zero)

    # imagination inside SuperMimic stays differentiable w.r.t. actor

    actor = MLP(dim_state, 16, dim_action)
    sm = SuperMimic(actor = actor, world_model = wm)
    rollout = sm(s0, num_rollout_timesteps = 3)
    assert rollout.uncertainties is None
    sm.actor_loss(rollout.rewards, rollout.terminals).backward()
    assert has_nonzero_grads(actor)

def test_world_model_jacobian_reg_horizon_invariance():
    wm = WorldModel(
        dim_state = 6,
        dim_action = 2,
        dim_sim_context = 0,
        dim_hiddens = (16,),
        pred_terminal = False,
        rollout_loss_decay = 0.8,
        jacobian_reg_weight = 0.2,
        num_explorative_rollouts = 2,
        dim_explorative_latent = 4
    )
    with torch.no_grad():
        wm.to_state_pred.weight.normal_(0., 0.8)

    s0 = torch.randn(4, 6)
    actions = torch.randn(4, 6, 2)
    next_states = torch.randn(4, 6, 6)

    torch.manual_seed(7)
    l1_on, _, _ = wm.rollout(s0, actions[:, :1], next_states[:, :1], explorative_latent_scale = 0.)
    torch.manual_seed(7)
    l1_off, _, _ = wm.rollout(s0, actions[:, :1], next_states[:, :1], explorative_latent_scale = 0., jacobian_reg_weight = 0.)

    torch.manual_seed(7)
    l6_on, _, _ = wm.rollout(s0, actions, next_states, explorative_latent_scale = 0.)
    torch.manual_seed(7)
    l6_off, _, _ = wm.rollout(s0, actions, next_states, explorative_latent_scale = 0., jacobian_reg_weight = 0.)

    diff_h1 = (l1_on - l1_off) / 1.
    diff_h6 = (l6_on - l6_off) / 6.
    assert diff_h1.item() > 0.
    assert torch.allclose(diff_h1, diff_h6, atol = 1e-4)

    l6_on.backward()
    assert has_nonzero_grads(wm)

@pytest.mark.parametrize('freeze_autoencoder', (False, True))
def test_latent_world_model_jacobian_reg(freeze_autoencoder):
    dim_state, dim_latent, dim_action = 8, 12, 3
    lwm = LatentWorldModel(
        state_encoder = MLP(dim_state, 16, dim_latent),
        state_decoder = MLP(dim_latent, 16, dim_state),
        dim_latent = dim_latent,
        dim_action = dim_action,
        dim_sim_context = 0,
        dim_hiddens = (16,),
        pred_terminal = False,
        freeze_autoencoder = freeze_autoencoder,
        jacobian_reg_weight = 0.2
    )
    with torch.no_grad():
        lwm.to_latent_pred.weight.normal_(0., 0.8)

    s0 = torch.randn(4, dim_state)
    actions = torch.randn(4, 4, dim_action)
    next_states = torch.randn(4, 4, dim_state)

    torch.manual_seed(0)
    loss_off, _, _ = lwm.rollout(s0, actions, next_states, jacobian_reg_weight = 0.)
    torch.manual_seed(0)
    loss_on, _, _ = lwm.rollout(s0, actions, next_states)

    assert loss_on.item() > loss_off.item()

    loss_on.backward()
    assert has_nonzero_grads(lwm.transition_mlp)
    assert has_nonzero_grads(lwm.to_latent_pred)
    if not freeze_autoencoder:
        assert has_nonzero_grads(lwm.state_encoder)
        assert has_nonzero_grads(lwm.state_decoder)
