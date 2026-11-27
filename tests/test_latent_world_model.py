import pytest
import torch
from torch import nn

from ema_pytorch import EMA
from x_mlps_pytorch import MLP
from torch_einops_utils import tree_map_tensor

from supermimic_pytorch.latent_world_model import LatentWorldModel
from supermimic_pytorch.supermimic import SuperMimic, flatten_state

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

def test_latent_world_model_dict_state():
    dim_total = 71 + 93

    def make_state(*prefix):
        return {
            'state': torch.randn(*prefix, 71),
            'observation': {
                'joint_pos': torch.randn(*prefix, 29),
                'joint_vel': torch.randn(*prefix, 29),
                'base_ang_vel': torch.randn(*prefix, 3),
                'projected_gravity': torch.randn(*prefix, 3),
                'prev_action': torch.randn(*prefix, 29)
            }
        }

    world_model = LatentWorldModel(
        state_encoder = nn.Sequential(MLP(dim_total, 16, 8), nn.Tanh()),
        state_decoder = MLP(8, 16, dim_total),
        dim_latent = 8,
        dim_hiddens = (16,),
        pred_reward = True
    )

    state = make_state(3)
    next_state = make_state(3)
    action = torch.randn(3, 29)
    sim_context = torch.randn(3, 33)

    # inference & training with nested dict state

    pred_only = world_model(state, action, sim_context)
    assert isinstance(pred_only.state, dict)
    assert pred_only.state['state'].shape == (3, 71)
    assert pred_only.state['observation']['joint_pos'].shape == (3, 29)

    loss, pred, losses = world_model(state, action, sim_context, next_state)
    assert isinstance(pred.state, dict)
    assert pred.state['observation']['projected_gravity'].shape == (3, 3)

    loss_flat, _, _ = world_model(flatten_state(state), action, sim_context, flatten_state(next_state))
    assert torch.allclose(loss, loss_flat)

    loss.backward()
    assert has_grads(world_model.transition_mlp)
    world_model.zero_grad()

    # direct forward_from_latent with dict next_state reconstitutes pred.state

    latent = world_model.encode(state)
    _, pred_from_latent, _ = world_model.forward_from_latent(latent, action, sim_context, next_state)
    assert isinstance(pred_from_latent.state, dict)
    assert pred_from_latent.state['observation']['joint_vel'].shape == (3, 29)

    # rollout & SuperMimic BPTT with nested dict state

    actor = MLP(dim_total, 16, 29, activation = nn.GELU())
    supermimic = SuperMimic(actor = actor, world_model = world_model)

    initial_state = make_state(2)
    sim_ctx = torch.randn(2, 33)

    rollout = supermimic(initial_state, sim_ctx, num_rollout_timesteps = 3)
    assert isinstance(rollout.states, dict)
    assert rollout.states['state'].shape == (2, 4, 71)
    assert rollout.states['observation']['joint_pos'].shape == (2, 4, 29)

    next_states = tree_map_tensor(lambda t: t[:, 1:], rollout.states)

    with torch.no_grad():
        _, preds, _ = world_model.rollout(initial_state, rollout.actions, next_states, sim_ctx)

    assert isinstance(preds[-1].state, dict)
    assert torch.allclose(rollout.states['state'][:, -1], preds[-1].state['state'])
    assert torch.allclose(
        rollout.states['observation']['joint_pos'][:, -1],
        preds[-1].state['observation']['joint_pos']
    )

    rewards = -rollout.states['observation']['joint_pos'][:, 1:].square().mean(dim = -1)
    actor_loss = supermimic.actor_loss(rewards, rollout.terminals)
    actor_loss.backward()

    assert has_grads(actor)
    assert not has_grads(world_model)

    # mismatched schema raises

    swapped = {'observation': next_state['observation'], 'state': next_state['state']}
    with pytest.raises(AssertionError):
        world_model(state, action, sim_context, swapped)

def test_explorative_latent_world_model():
    world_model = LatentWorldModel(
        state_encoder = nn.Sequential(MLP(8, 16, 6), nn.Tanh()),
        state_decoder = MLP(6, 16, 8),
        dim_action = 2,
        dim_sim_context = 0,
        dim_latent = 6,
        dim_hiddens = (16,),
        pred_reward = True,
        pred_terminal = True,
        dim_explorative_latent = 4,
        explorative_latent_scale = 0.25
    )
    assert world_model.num_explorative_rollouts == 2

    state = torch.randn(3, 8)
    actions = torch.randn(3, 4, 2)
    next_states = torch.randn(3, 4, 8)
    next_terminals = torch.randint(0, 2, (3, 4)).float()
    next_rewards = torch.randn(3, 4)

    # zero-latent default at inference
    pred_default = world_model(state, actions[:, 0])
    pred_zero = world_model(state, actions[:, 0], explorative_latent = torch.zeros(3, 4))
    assert torch.allclose(pred_default.state, pred_zero.state)

    # rollout loss & gradients
    loss, preds, _ = world_model.rollout(state, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards)
    assert preds[-1].state.shape == (6, 8)
    loss.backward()
    assert has_grads(world_model.transition_mlp)
    world_model.zero_grad()

    # scale = 0 matches k = 1
    with torch.no_grad():
        loss_zero, _, _ = world_model.rollout(state, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards, explorative_latent_scale = 0.)
        loss_k1, _, _ = world_model.rollout(state, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards, num_explorative_rollouts = 1)
    assert torch.allclose(loss_zero, loss_k1, atol = 1e-6)

    # SuperMimic BPTT with wm_explorative_latent_scale
    actor = MLP(8, 16, 2, activation = nn.GELU())
    supermimic = SuperMimic(actor = actor, world_model = world_model, wm_explorative_latent_scale = 0.25)
    rollout = supermimic(state, num_rollout_timesteps = 3)
    actor_loss = supermimic.actor_loss(rollout.rewards, rollout.terminals)
    actor_loss.backward()
    assert has_grads(actor)
    assert not has_grads(world_model)

def test_latent_world_model_rollout_decay_and_state_noise():
    torch.manual_seed(0)

    world_model = LatentWorldModel(
        state_encoder = nn.Sequential(MLP(8, 16, 6), nn.Tanh()),
        state_decoder = MLP(6, 16, 8),
        dim_action = 2,
        dim_sim_context = 0,
        dim_latent = 6,
        dim_hiddens = (16,),
        pred_reward = True,
        pred_terminal = True,
        freeze_autoencoder = False,
        rollout_loss_decay = 0.8,
        state_noise_scale = 0.05
    )

    state = torch.randn(4, 8)
    actions = torch.randn(4, 5, 2)
    next_states = torch.randn(4, 5, 8)
    next_rewards = torch.randn(4, 5)
    next_terminals = torch.randint(0, 2, (4, 5)).float()

    # eval mode is deterministic and matches exact rho^t weighted sum

    world_model.eval()
    with torch.no_grad():
        loss1, preds1, step_losses = world_model.rollout(state, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards)
        loss2, preds2, _ = world_model.rollout(state, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards)

    assert torch.allclose(loss1, loss2)
    assert torch.allclose(preds1[-1].state, preds2[-1].state)

    weights = [0.8 ** t for t in range(5)]
    expected_steps = [(l.state + l.latent + l.recon + l.reward + l.terminal) for l in step_losses]
    expected_loss = sum(w * l for w, l in zip(weights, expected_steps)) * (5. / sum(weights))
    assert torch.allclose(loss1, expected_loss, atol = 1e-5)

    # train mode injects state noise and trains end-to-end

    world_model.train()
    loss_t1, preds_t1, _ = world_model.rollout(state, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards)
    _, preds_t2, _ = world_model.rollout(state, actions, next_states, next_terminals = next_terminals, next_rewards = next_rewards)
    assert not torch.allclose(preds_t1[0].state, preds_t2[0].state)

    loss_t1.backward()
    assert has_grads(world_model.transition_mlp)
    assert has_grads(world_model.state_encoder)
    assert has_grads(world_model.state_decoder)
