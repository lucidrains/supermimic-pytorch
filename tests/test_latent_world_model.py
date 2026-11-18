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

    # actor still trains through the frozen world model

    world_model.zero_grad()

    actor = MLP(71, 16, 29, activation = nn.GELU())
    supermimic = SuperMimic(actor = actor, world_model = world_model)

    rollout = supermimic(torch.randn(2, 71), torch.randn(2, 33), num_rollout_timesteps = 2)

    rollout.rewards.sum().backward()

    assert has_grads(actor)
    assert not has_grads(world_model)
