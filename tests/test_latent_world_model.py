import pytest
import torch
from torch import nn

from ema_pytorch import EMA
from x_mlps_pytorch import MLP

from supermimic_pytorch.latent_world_model import LatentWorldModel
from supermimic_pytorch.supermimic import SuperMimic

def exists(v):
    return v is not None

def has_grads(module):
    return any(exists(p.grad) for p in module.parameters())

def test_latent_world_model():
    state_encoder = nn.Sequential(
        MLP(71, 16, 8),
        nn.Tanh()
    )

    world_model = LatentWorldModel(
        state_encoder = state_encoder,
        state_decoder = MLP(8, 16, 71),
        dim_latent = 8,
        dim_hiddens = (16,),
        pred_reward = True
    )

    state = torch.randn(3, 71)
    action = torch.randn(3, 29)
    sim_context = torch.randn(3, 33)

    next_state = torch.randn(3, 71)
    next_terminal = torch.randint(0, 2, (3,)).float()
    next_reward = torch.randn(3)

    # frozen autoencoder -> state + latent losses, no recon

    loss, pred, losses = world_model(
        state,
        action,
        sim_context,
        next_state = next_state,
        next_terminal = next_terminal,
        next_reward = next_reward
    )

    assert isinstance(world_model.target_state_encoder, EMA)

    assert pred.state.shape == (3, 71)
    assert pred.latent.shape == (3, 8)
    assert ((0. < pred.terminal) & (pred.terminal < 1.)).all()

    assert not world_model.has_recon_loss
    assert torch.allclose(loss, losses.state + losses.latent + losses.terminal + losses.reward)

    loss.backward()

    assert has_grads(world_model.transition_mlp)
    assert not has_grads(world_model.state_encoder)
    assert not has_grads(world_model.state_decoder)

    # e2e -> recon loss comes online and the autoencoder learns

    world_model.unfreeze_autoencoder()

    loss, _, losses = world_model(state, action, sim_context, next_state)

    assert world_model.has_recon_loss
    assert torch.allclose(loss, losses.state + losses.latent + losses.recon)

    loss.backward()

    assert has_grads(world_model.state_encoder)
    assert has_grads(world_model.state_decoder)

    # actor still trains through frozen world model in latent space

    world_model.zero_grad()

    actor = MLP(71, 16, 29, activation = nn.GELU())
    supermimic = SuperMimic(actor = actor, world_model = world_model)

    initial_state = torch.randn(2, 71)
    sim_context = torch.randn(2, 33)

    rollout = supermimic(initial_state, sim_context, num_rollout_timesteps = 2)

    with torch.no_grad():
        _, preds, _ = world_model.rollout(initial_state, rollout.actions, rollout.states[:, 1:], sim_context)

    assert torch.allclose(rollout.states[:, -1], preds[-1].state)

    actor_loss = supermimic.actor_loss(rollout.rewards, rollout.terminals)
    actor_loss.backward()

    assert has_grads(actor)
    assert not has_grads(world_model)

@pytest.mark.parametrize('num_discrete_actions', (4, (4, 3)))
def test_latent_world_model_discrete_actions(num_discrete_actions):
    action_dims = num_discrete_actions if isinstance(num_discrete_actions, tuple) else (num_discrete_actions,)
    is_multi = len(action_dims) > 1

    world_model = LatentWorldModel(
        state_encoder = MLP(8, 16, 4),
        state_decoder = MLP(4, 16, 8),
        num_discrete_actions = num_discrete_actions,
        dim_sim_context = 0,
        dim_latent = 4,
        dim_hiddens = (16,),
        pred_reward = True,
        pred_terminal = True
    )

    state = torch.randn(3, 8)
    per_head = [torch.randint(0, n, (3,)) for n in action_dims]
    indices = torch.stack(per_head, dim = -1) if is_multi else per_head[0]
    one_hot = torch.cat([torch.nn.functional.one_hot(a, n).float() for a, n in zip(per_head, action_dims)], dim = -1)

    with torch.no_grad():
        pred_idx = world_model(state, indices)
        pred_oh = world_model(state, one_hot)

    assert torch.allclose(pred_idx.state, pred_oh.state)
    assert torch.allclose(pred_idx.latent, pred_oh.latent)

    seq_per_head = [torch.randint(0, n, (3, 4)) for n in action_dims]
    seq_indices = torch.stack(seq_per_head, dim = -1) if is_multi else seq_per_head[0]

    loss, _, _ = world_model.rollout(state, seq_indices, torch.randn(3, 4, 8))
    loss.backward()
    assert has_grads(world_model.transition_mlp)

    world_model.zero_grad()

    actor = MLP(8, 16, sum(action_dims), activation = nn.GELU())
    supermimic = SuperMimic(actor = actor, world_model = world_model)

    rollout = supermimic(state, num_rollout_timesteps = 3)
    actor_loss = supermimic.actor_loss(rollout.rewards, rollout.terminals)
    actor_loss.backward()
    assert has_grads(actor)
    assert not has_grads(world_model)
