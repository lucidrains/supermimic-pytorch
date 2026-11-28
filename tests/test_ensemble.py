import pytest
import torch

from x_mlps_pytorch import MLP

from supermimic_pytorch import (
    AdversarialCritic,
    AdversarialSuperMimic,
    Critic,
    LatentWorldModel,
    SuperMimic,
    WorldModel,
    WorldModelEnsemble
)

# helpers

def has_nonzero_grads(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

def make_wm(dim_state = 8, dim_action = 3, dim_sim_context = 0, pred_reward = True, pred_terminal = True, state_normalizer = False):
    return WorldModel(
        dim_state = dim_state,
        dim_action = dim_action,
        dim_sim_context = dim_sim_context,
        dim_hiddens = (16, 16),
        pred_reward = pred_reward,
        pred_terminal = pred_terminal,
        state_normalizer = state_normalizer
    )

def perturb_members(ensemble, scale = 0.2):
    for i, member in enumerate(ensemble.members):
        head = member.to_state_pred if isinstance(member, WorldModel) else member.to_latent_pred
        with torch.no_grad():
            head.weight.normal_(0., scale * (i + 1))

# tests

def test_ensemble_init_and_zero_variance_grads():
    ens = WorldModelEnsemble(lambda: make_wm(8, 3), num_members = 3)
    assert ens.num_members == 3
    assert ens.dim_action == 3

    ens2 = WorldModelEnsemble([make_wm(8, 3), make_wm(8, 3)])
    assert ens2.num_members == 2

    with pytest.raises(AssertionError):
        WorldModelEnsemble([make_wm(8, 3)])

    # zero-init members agree identically -> std grad must stay finite

    s = torch.randn(4, 8, requires_grad = True)
    a = torch.randn(4, 3, requires_grad = True)
    pred = ens(s, a)
    assert pred.epistemic_uncertainty.shape == (4,)
    assert torch.isfinite(pred.epistemic_uncertainty).all()

    pred.epistemic_uncertainty.sum().backward()
    assert torch.isfinite(s.grad).all()
    assert torch.isfinite(a.grad).all()

@pytest.mark.parametrize('dict_state', (False, True))
@pytest.mark.parametrize('use_normalizer', (False, True))
def test_ensemble_rollout_and_epistemic_penalty_bptt(dict_state, use_normalizer):
    dim_state = 10 if dict_state else 8
    dim_action = 3
    dim_sim_context = 4

    def make_state(*leading):
        if dict_state:
            return dict(pos = torch.randn(*leading, 4), vel = torch.randn(*leading, 6))
        return torch.randn(*leading, dim_state)

    ens = WorldModelEnsemble(
        lambda: make_wm(dim_state, dim_action, dim_sim_context, state_normalizer = use_normalizer),
        num_members = 3,
        rollout_mode = 'mean',
        bootstrap_resample = True
    )

    s0 = make_state(4)
    ctx = torch.randn(4, dim_sim_context)
    actions = torch.randn(4, 5, dim_action)
    next_states = make_state(4, 5)
    next_rewards = torch.randn(4, 5)
    next_terminals = torch.randint(0, 2, (4, 5)).float()

    wm_loss, preds, member_losses = ens.rollout(
        s0, actions, next_states,
        sim_context = ctx,
        next_terminals = next_terminals,
        next_rewards = next_rewards
    )
    assert len(preds) == 5
    assert len(member_losses) == 3
    assert preds[0].epistemic_uncertainty.shape == (4,)

    wm_loss.backward()
    for member in ens.members:
        assert has_nonzero_grads(member)

    perturb_members(ens)

    actor = MLP(dim_state + 4, 16, dim_action)
    critic = Critic(dim_state = dim_state, dim_goal = 0, dim_sim_context = dim_sim_context, dim_hiddens = (16,))
    sm = SuperMimic(
        actor = actor,
        world_model = ens,
        critic = critic,
        epistemic_penalty_weight = 2.0,
        num_explorative_rollouts = 2,
        dim_explorative_latent = 4
    )

    rollout = sm(s0, sim_context = ctx, num_rollout_timesteps = 4)
    assert rollout.uncertainties is not None
    assert rollout.uncertainties.shape == (8, 4)
    assert (rollout.uncertainties > 0).all()

    # epistemic penalty alone drives actor gradients via BPTT

    pess_only = sm.actor_loss(
        torch.zeros_like(rollout.rewards),
        rollout.terminals,
        bootstrap_value = torch.zeros(8),
        uncertainties = rollout.uncertainties
    )
    assert pess_only.item() > 0.

    actor.zero_grad()
    pess_only.backward(retain_graph = True)
    assert has_nonzero_grads(actor)

    # per-call override and missing-uncertainties check

    final_state = {k: v[:, -1] for k, v in rollout.states.items()} if dict_state else rollout.states[:, -1]
    loss_beta_0 = sm.actor_loss(rollout.rewards, rollout.terminals, final_state = final_state, sim_context = ctx, uncertainties = rollout.uncertainties, epistemic_penalty_weight = 0.)
    loss_beta_2 = sm.actor_loss(rollout.rewards, rollout.terminals, final_state = final_state, sim_context = ctx, uncertainties = rollout.uncertainties, epistemic_penalty_weight = 2.)
    assert loss_beta_2.item() > loss_beta_0.item()

    with pytest.raises(AssertionError):
        sm.actor_loss(rollout.rewards, rollout.terminals, final_state = final_state, sim_context = ctx)

def test_ensemble_random_mode():
    ens = WorldModelEnsemble(lambda: make_wm(8, 3), num_members = 3, rollout_mode = 'random')
    perturb_members(ens)

    actor = MLP(8, 16, 3)
    sm = SuperMimic(actor = actor, world_model = ens)

    rollout = sm(torch.randn(4, 8), num_rollout_timesteps = 4)
    assert rollout.uncertainties is None

    sm.actor_loss(rollout.rewards, rollout.terminals).backward()
    assert has_nonzero_grads(actor)

def test_ensemble_latent_and_adversarial_compat():
    def make_lwm():
        return LatentWorldModel(
            state_encoder = MLP(8, 16, 12),
            state_decoder = MLP(12, 16, 8),
            dim_latent = 12,
            dim_action = 3,
            dim_sim_context = 0,
            dim_hiddens = (16,),
            pred_reward = True,
            pred_terminal = True,
            freeze_autoencoder = False
        )

    ens = WorldModelEnsemble(make_lwm, num_members = 2, rollout_mode = 'mean')
    perturb_members(ens)

    wm_loss, _, _ = ens.rollout(
        torch.randn(4, 8),
        torch.randn(4, 3, 3),
        torch.randn(4, 3, 8),
        next_terminals = torch.randint(0, 2, (4, 3)).float(),
        next_rewards = torch.randn(4, 3)
    )
    wm_loss.backward()
    ens.update_ema()

    actor = MLP(8, 16, 3)
    sm = SuperMimic(actor = actor, world_model = ens, epistemic_penalty_weight = 1.0)
    adv_sm = AdversarialSuperMimic(sm, motion_critic = AdversarialCritic(dim_feature = 8, dim_hiddens = (16,)), motion_reward_weight = 0.5)

    adv_rollout = adv_sm(torch.randn(4, 8), num_rollout_timesteps = 3)
    assert adv_rollout.uncertainties is not None
    assert adv_rollout.uncertainties.shape == (4, 3)

    sm.actor_loss(adv_rollout.rewards, adv_rollout.terminals, uncertainties = adv_rollout.uncertainties).backward()
    assert has_nonzero_grads(actor)
