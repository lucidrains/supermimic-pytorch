import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from torch_einops_utils import tree_map_tensor
from x_mlps_pytorch import MLP

from supermimic_pytorch import (
    Critic,
    HistoryWindow,
    HumanoidActor,
    LatentWorldModel,
    SuperMimic,
    UNITREE_G1_OBS_NOISE,
    WelfordNormalizer,
    WorldModel,
    unitree_g1_state_to_obs
)
from supermimic_pytorch.rewards import UNITREE_G1_STATE_SPEC


def exists(v):
    return v is not None


def has_grads(module):
    return any(exists(p.grad) for p in module.parameters())


def has_nonzero_grads(module):
    return any(exists(p.grad) and p.grad.abs().sum() > 0 for p in module.parameters())


# 1. core Welford math & rank invariance

def test_welford_update_matches_concatenated_batches():
    torch.manual_seed(0)

    # default dim = -1 for RL states / sequences

    batches = [torch.randn(4, 5, 7) * 3. + 2. for _ in range(5)]
    data = torch.cat(batches, dim = 0)

    normalizer = WelfordNormalizer(7)
    for batch in batches:
        normalizer.update(batch)

    expected_mean = data.mean(dim = (0, 1))
    expected_var = data.var(dim = (0, 1), unbiased = False)

    torch.testing.assert_close(normalizer.mean, expected_mean, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(normalizer.m2 / normalizer.count, expected_var, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(normalizer.count, data.new_tensor(data.shape[0] * data.shape[1]))

    # dim = 1 for images (b c h w)

    img_batches = [torch.randn(4, 3, 8, 8) * 2. + 5. for _ in range(4)]
    imgs = torch.cat(img_batches, dim = 0)

    img_norm = WelfordNormalizer((3, 8, 8), dim = 1)
    for batch in img_batches:
        img_norm.update(batch)

    torch.testing.assert_close(img_norm.mean, imgs.mean(dim = (0, 2, 3)), rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(img_norm.m2 / img_norm.count, imgs.var(dim = (0, 2, 3), unbiased = False), rtol = 1e-5, atol = 1e-5)

    normed = img_norm.normalize(imgs)
    assert normed.shape == imgs.shape
    torch.testing.assert_close(img_norm.denormalize(normed), imgs, rtol = 1e-5, atol = 1e-5)


def test_welford_mixed_ranks_and_global_stats():
    torch.manual_seed(1)

    x1 = torch.randn(5) * 4. - 3.
    x2 = torch.randn(6, 5) * 4. - 3.
    x3 = torch.randn(3, 4, 5) * 4. - 3.
    x4 = torch.randn(2, 2, 3, 5) * 4. - 3.

    norm = WelfordNormalizer()
    for x in (x1, x2, x3, x4):
        norm.update(x)

    all_x = torch.cat([x.reshape(-1, 5) for x in (x1, x2, x3, x4)], dim = 0)

    torch.testing.assert_close(norm.mean, all_x.mean(dim = 0), rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(norm.m2 / norm.count, all_x.var(dim = 0, unbiased = False), rtol = 1e-5, atol = 1e-5)

    for x in (x1, x2, x3, x4):
        y = norm.normalize(x)
        assert y.shape == x.shape
        torch.testing.assert_close(norm.denormalize(y), x, rtol = 1e-5, atol = 1e-5)

    global_norm = WelfordNormalizer((4, 5), dim = None)
    global_norm.update(x3)
    torch.testing.assert_close(global_norm.mean, x3.mean(), rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(global_norm.m2 / global_norm.count, x3.var(unbiased = False), rtol = 1e-5, atol = 1e-5)


# 2. subset & exclude normalization (flat tensors, dicts, nested dicts)

@pytest.mark.parametrize('kwargs, expected_active', [
    (dict(subset = slice(0, 6)), [0, 1, 2, 3, 4, 5]),
    (dict(subset = (slice(0, 3), slice(7, 10))), [0, 1, 2, 7, 8, 9]),
    (dict(exclude = slice(3, 7)), [0, 1, 2, 7, 8, 9]),
    (dict(subset = [0, 2, 5, -1]), [0, 2, 5, 9]),
    (dict(subset = torch.tensor([1, 4, 8])), [1, 4, 8]),
    (dict(subset = [True, False, True, False, False, True, False, False, False, True]), [0, 2, 5, 9]),
    (dict(subset = torch.tensor([True, True, False, False, False, False, False, False, True, True])), [0, 1, 8, 9]),
    (dict(subset = slice(0, 8), exclude = (slice(3, 7), 0)), [1, 2, 7]),
    (dict(subset = False), []),
])
def test_tensor_subset_and_exclude(kwargs, expected_active):
    torch.manual_seed(3)

    data = torch.randn(32, 10) * 5. + 12.
    data.requires_grad_(True)

    norm = WelfordNormalizer(10, **kwargs)
    norm.update(data.detach())

    active = torch.zeros(10, dtype = torch.bool)
    if expected_active:
        active[expected_active] = True

    assert torch.equal(norm.subset_mask, active)

    normed = norm.normalize(data)

    if (~active).any():
        assert torch.equal(normed[:, ~active], data[:, ~active])
        assert torch.equal(norm.mean[~active], torch.zeros((~active).sum()))

    if active.any():
        torch.testing.assert_close(normed[:, active].mean(dim = 0), torch.zeros(active.sum()), atol = 1e-5, rtol = 1e-5)
        torch.testing.assert_close(normed[:, active].var(dim = 0, unbiased = False), torch.ones(active.sum()), atol = 1e-4, rtol = 1e-4)

    recovered = norm.denormalize(normed)
    torch.testing.assert_close(recovered, data, atol = 1e-5, rtol = 1e-5)

    normed.sum().backward()
    expected_grad = torch.where(active, 1. / norm.std, torch.ones(10)).expand_as(data)
    torch.testing.assert_close(data.grad, expected_grad, atol = 1e-5, rtol = 1e-5)


def test_dict_and_nested_dict_subset_normalization():
    torch.manual_seed(4)

    data = dict(
        state = torch.randn(40, 12) * 4. + 8.,
        observation = dict(
            joint_pos = torch.randn(40, 6) * 0.5 + 1.,
            joint_vel = torch.randn(40, 6) * 8. - 3.,
            projected_gravity = torch.randn(40, 3) * 0.2 + 5.
        )
    )

    norm_excl = WelfordNormalizer(
        exclude = dict(
            state = slice(3, 7),
            observation = 'projected_gravity'
        )
    )
    norm_incl = WelfordNormalizer(
        subset = dict(
            state = (slice(0, 3), slice(7, 12)),
            observation = ('joint_pos', 'joint_vel')
        )
    )

    out_excl = norm_excl(data)
    out_incl = norm_incl(data)

    assert torch.equal(norm_excl.subset_mask, norm_incl.subset_mask)
    torch.testing.assert_close(out_excl['state'], out_incl['state'])
    for k in data['observation']:
        torch.testing.assert_close(out_excl['observation'][k], out_incl['observation'][k])

    assert torch.equal(out_excl['state'][:, 3:7], data['state'][:, 3:7])
    assert torch.equal(out_excl['observation']['projected_gravity'], data['observation']['projected_gravity'])

    for k in ('joint_pos', 'joint_vel'):
        torch.testing.assert_close(out_excl['observation'][k].mean(dim = 0), torch.zeros(6), atol = 1e-5, rtol = 1e-5)
        torch.testing.assert_close(out_excl['observation'][k].var(dim = 0, unbiased = False), torch.ones(6), atol = 1e-4, rtol = 1e-4)

    rec = norm_excl.denormalize(out_excl)
    torch.testing.assert_close(rec['state'], data['state'], atol = 1e-5, rtol = 1e-5)
    for k in data['observation']:
        torch.testing.assert_close(rec['observation'][k], data['observation'][k], atol = 1e-5, rtol = 1e-5)

    # eager dict spec & typo validation
    g1_norm = WelfordNormalizer(UNITREE_G1_STATE_SPEC, exclude = 'root_quat')
    assert g1_norm.mean.shape == (71,)
    assert int(g1_norm.subset_mask.sum().item()) == 67

    with pytest.raises(AssertionError):
        WelfordNormalizer(UNITREE_G1_STATE_SPEC, exclude = 'nonexistent_key')

    # lazy checkpoint load_state_dict
    restored = WelfordNormalizer(
        exclude = dict(state = slice(3, 7), observation = 'projected_gravity')
    )
    restored.load_state_dict(norm_excl.state_dict())
    torch.testing.assert_close(restored.normalize(data)['state'], out_excl['state'])


def test_freeze_unfreeze_eval_and_reset():
    torch.manual_seed(6)

    norm = WelfordNormalizer(4, exclude = 3)
    norm.update(torch.randn(8, 4))

    count, mean = norm.count.clone(), norm.mean.clone()

    norm.freeze()
    norm(torch.randn(8, 4) * 5. + 10.)
    torch.testing.assert_close(norm.count, count)
    torch.testing.assert_close(norm.mean, mean)

    norm.unfreeze()
    norm(torch.randn(8, 4) * 5. + 10.)
    assert norm.count > count

    count = norm.count.clone()
    norm.eval()
    norm(torch.randn(8, 4) * 3. + 20.)
    torch.testing.assert_close(norm.count, count)

    norm(torch.randn(8, 4) * 3. + 20., update = True)
    assert norm.count > count

    count = norm.count.clone()
    norm.train()
    norm(torch.randn(8, 4) * 3. + 20., update = False)
    torch.testing.assert_close(norm.count, count)

    norm.reset()
    assert norm.count.item() == 0.


def _distributed_worker(rank, world_size, init_file):
    dist.init_process_group(
        backend = 'gloo',
        init_method = f'file://{init_file}',
        rank = rank,
        world_size = world_size
    )

    norm = WelfordNormalizer(3, exclude = 2)

    local_data = torch.tensor([[1., 2., 100.], [3., 4., 200.], [3., 4., 300.]]) if rank == 0 else torch.tensor([[5., 6., -50.]])
    norm.update(local_data)

    torch.testing.assert_close(norm.mean, torch.tensor([3., 4., 0.]), rtol = 1e-6, atol = 1e-6)
    torch.testing.assert_close(norm.m2 / norm.count, torch.tensor([2., 2., 0.]), rtol = 1e-6, atol = 1e-6)
    torch.testing.assert_close(norm.count, torch.tensor(4.))

    dist.destroy_process_group()


def test_welford_distributed_all_reduce():
    if not dist.is_available():
        pytest.skip('torch.distributed is unavailable')

    with tempfile.TemporaryDirectory() as folder:
        init_file = os.path.join(folder, 'init')
        torch.multiprocessing.spawn(
            _distributed_worker,
            args = (2, init_file),
            nprocs = 2,
            join = True
        )


# 3. WorldModel & LatentWorldModel integration

@pytest.mark.parametrize('dict_state', (False, True))
def test_world_model_state_normalizer(dict_state):
    torch.manual_seed(7)
    batch, horizon, dim_action = 4, 5, 3

    if dict_state:
        state_norm = WelfordNormalizer(exclude = 'quat')
        def make_state(*lead):
            return dict(
                pos = torch.randn(*lead, 3) * 0.2 + 1.0,
                quat = F.normalize(torch.randn(*lead, 4), dim = -1),
                vel = torch.randn(*lead, 3) * 20.0 - 5.0
            )
    else:
        state_norm = WelfordNormalizer(10, exclude = slice(3, 7))
        def make_state(*lead):
            return torch.cat((
                torch.randn(*lead, 3) * 0.2 + 1.0,
                F.normalize(torch.randn(*lead, 4), dim = -1),
                torch.randn(*lead, 3) * 20.0 - 5.0
            ), dim = -1)

    wm = WorldModel(
        dim_state = 10,
        dim_action = dim_action,
        dim_sim_context = 0,
        dim_hiddens = (32, 32),
        pred_residual_state = True,
        pred_terminal = False,
        num_explorative_rollouts = 2,
        dim_explorative_latent = 2,
        state_normalizer = state_norm
    )

    s0 = make_state(batch)
    actions = torch.randn(batch, horizon, dim_action)
    next_states = make_state(batch, horizon)

    # pure inference does not update normalizer and preserves zero-init identity dynamics
    pred0 = wm(s0, actions[:, 0])
    assert state_norm.count.item() == 0.
    if dict_state:
        for k in s0:
            torch.testing.assert_close(pred0.state[k], s0[k], atol = 1e-5, rtol = 1e-5)
    else:
        torch.testing.assert_close(pred0.state, s0, atol = 1e-5, rtol = 1e-5)

    # rollout updates stats once on real [s0, next_states], not multiplied by explorative k
    loss, preds, _ = wm.rollout(s0, actions, next_states)
    assert state_norm.count.item() == batch * (horizon + 1)

    # zero-init residual head still predicts exact physical s0 on step 0 after stats update
    # (repeat_batch interleaves (b k), so step-0 preds are [s0[0], s0[0], s0[1], s0[1], ...])
    if dict_state:
        for k in s0:
            torch.testing.assert_close(preds[0].state[k][::2], s0[k], atol = 1e-4, rtol = 1e-4)
    else:
        torch.testing.assert_close(preds[0].state[::2], s0, atol = 1e-4, rtol = 1e-4)

    # MSE state_loss is computed in normalized space
    wm.eval()
    with torch.no_grad():
        s1 = tree_map_tensor(lambda t: t[:, 0], next_states)
        _, _, single_losses = wm(s0, actions[:, 0], next_state = s1)

        norm_s0 = state_norm.normalize(s0)
        norm_s1 = state_norm.normalize(s1)
        if dict_state:
            norm_s0 = torch.cat([norm_s0[k] for k in sorted(norm_s0)], dim = -1)
            norm_s1 = torch.cat([norm_s1[k] for k in sorted(norm_s1)], dim = -1)

        torch.testing.assert_close(single_losses.state, F.mse_loss(norm_s1, norm_s0), atol = 1e-5, rtol = 1e-5)
        assert state_norm.count.item() == batch * (horizon + 1)

    wm.train()
    loss.backward()
    assert has_nonzero_grads(wm)


def test_latent_world_model_state_normalizer():
    torch.manual_seed(8)

    state_norm = WelfordNormalizer(10, exclude = slice(3, 7))
    lwm = LatentWorldModel(
        state_encoder = nn.Sequential(MLP(10, 16, 6), nn.Tanh()),
        state_decoder = MLP(6, 16, 10),
        dim_action = 4,
        dim_sim_context = 0,
        dim_latent = 6,
        dim_hiddens = (16,),
        pred_reward = True,
        state_normalizer = state_norm
    )

    s0 = torch.randn(3, 10) * 5. + 3.
    actions = torch.randn(3, 4, 4)
    next_states = torch.randn(3, 4, 10) * 5. + 3.

    lwm.unfreeze_autoencoder()
    loss, preds, _ = lwm.rollout(s0, actions, next_states)
    assert state_norm.count.item() == 3 * 5
    assert preds[-1].state.shape == (3, 10)

    loss.backward()
    assert has_grads(lwm.transition_mlp)
    assert has_nonzero_grads(lwm.to_latent_pred)
    assert has_nonzero_grads(lwm.state_encoder)
    assert has_nonzero_grads(lwm.state_decoder)
    lwm.zero_grad()

    actor = MLP(10, 16, 4, activation = nn.GELU())
    sm = SuperMimic(actor = actor, world_model = lwm, obs_normalizer = state_norm)

    rollout = sm(s0, num_rollout_timesteps = 3)
    with torch.no_grad():
        lwm.eval()
        _, wm_preds, _ = lwm.rollout(s0, rollout.actions, rollout.states[:, 1:])

    torch.testing.assert_close(rollout.states[:, -1], wm_preds[-1].state, atol = 1e-5, rtol = 1e-5)


# 4. SuperMimic end-to-end: Unitree G1 asymmetric POMDP, EMA critic sharing, shared normalizer & parallel unfold

@pytest.mark.parametrize('use_ema_critic', (False, True))
def test_supermimic_unitree_subset_normalizers_e2e(use_ema_critic):
    torch.manual_seed(9)
    batch, horizon, num_goal_frames = 3, 4, 5

    wm_state_norm = WelfordNormalizer(71, exclude = slice(3, 7))
    obs_norm = WelfordNormalizer(exclude = 'projected_gravity')
    critic_state_norm = WelfordNormalizer(71, exclude = slice(3, 7))

    world_model = WorldModel(
        dim_state = 71,
        dim_action = 29,
        dim_sim_context = 33,
        dim_hiddens = (32,),
        pred_residual_state = True,
        state_normalizer = wm_state_norm
    )

    actor = HumanoidActor(
        dim_obs = 64,
        num_obs_frames = 4,
        num_action_frames = 1,
        num_goal_frames = num_goal_frames,
        dim_hiddens = (16,),
        dim_velocity_hiddens = (16,)
    )

    critic = Critic(
        dim_state = 71,
        dim_goal = 71,
        dim_sim_context = 33,
        dim_hiddens = (16,),
        state_normalizer = critic_state_norm
    )

    s0 = torch.randn(batch, 71) * 3. + 2.
    s0[:, 3:7] = F.normalize(s0[:, 3:7], dim = -1)
    sim_context = torch.randn(batch, 33)
    logged_next = torch.randn(batch, 4, 71) * 3. + 2.
    logged_next[:, :, 3:7] = F.normalize(logged_next[:, :, 3:7], dim = -1)

    wm_loss, _, _ = world_model.rollout(s0, torch.randn(batch, 4, 29), logged_next, sim_context)
    wm_loss.backward()
    world_model.zero_grad()

    assert wm_state_norm.count.item() == batch * 5
    assert torch.equal(wm_state_norm.mean[3:7], torch.zeros(4))

    seen_reward_states = []
    def reward_fn(next_states, references, actions, **kwargs):
        seen_reward_states.append(next_states)
        return -(next_states - references).square().mean(dim = -1)

    supermimic = SuperMimic(
        actor = actor,
        world_model = world_model,
        critic = critic,
        use_ema_critic = use_ema_critic,
        state_to_obs = unitree_g1_state_to_obs,
        obs_history = HistoryWindow(4, 64, pad_mode = 'zero'),
        action_history = HistoryWindow(1, 29, lag = 1, pad_mode = 'zero'),
        goal_window = num_goal_frames,
        obs_normalizer = obs_norm,
        observation_noise_scale = UNITREE_G1_OBS_NOISE,
        reward_fn = reward_fn
    )

    if use_ema_critic:
        assert supermimic.ema_critic.ema_model.state_normalizer is critic_state_norm
        assert critic_state_norm.training

    goals = torch.randn(batch, horizon + num_goal_frames, 71) * 3. + 2.
    refs = goals[:, 1 : horizon + 1]

    rollout = supermimic(
        s0,
        sim_context,
        actor_goals = goals,
        references = refs,
        num_rollout_timesteps = horizon
    )

    assert obs_norm.count.item() == batch
    torch.testing.assert_close(seen_reward_states[0], rollout.states[:, 1:])

    critic_loss = supermimic.critic_loss(
        rollout.states,
        rollout.rewards,
        rollout.terminals,
        goals = goals,
        sim_context = sim_context
    )
    critic_loss.backward()
    assert has_nonzero_grads(critic)
    assert critic_state_norm.count.item() > 0.
    critic.zero_grad()

    # update_ema must succeed cleanly with bool subset_mask and leave shared stats untouched
    prev_count = critic_state_norm.count.clone()
    supermimic.update_ema()
    torch.testing.assert_close(critic_state_norm.count, prev_count)

    supermimic.train()
    assert critic_state_norm.training

    actor_loss = supermimic.actor_loss(
        rollout.rewards,
        rollout.terminals,
        final_state = rollout.states[:, -1],
        final_goal = goals[:, horizon],
        sim_context = sim_context,
        actions = rollout.actions
    )
    actor_loss.backward()
    assert has_nonzero_grads(actor)
    assert not has_grads(world_model)
    assert not has_grads(critic)
    assert not has_grads(supermimic.target_critic)


def test_shared_normalizer_not_polluted_by_critic_and_matches_parallel_unfold():
    torch.manual_seed(10)
    batch, horizon = 2, 4

    shared_norm = WelfordNormalizer(12, exclude = slice(1, 5))

    wm = WorldModel(
        dim_state = 12,
        dim_action = 4,
        dim_sim_context = 0,
        dim_hiddens = (16,),
        pred_residual_state = False,
        pred_reward = True,
        state_normalizer = shared_norm
    )

    actor = MLP(3 * 12 + 4, 16, 4, activation = nn.GELU())
    critic = Critic(dim_state = 12, dim_goal = 0, dim_sim_context = 0, dim_hiddens = (16,), state_normalizer = shared_norm)

    sm = SuperMimic(
        actor = actor,
        world_model = wm,
        critic = critic,
        use_ema_critic = True,
        obs_history = HistoryWindow(3, 12, pad_mode = 'zero'),
        action_history = HistoryWindow(1, 4, lag = 1, pad_mode = 'zero'),
        obs_normalizer = shared_norm
    )

    s0 = torch.randn(batch, 12) * 4. + 2.
    past_obs = torch.randn(batch, 2, 12) * 4. + 2.
    past_actions = torch.randn(batch, 1, 4)

    rollout = sm(s0, past_obs = past_obs, past_actions = past_actions, num_rollout_timesteps = horizon)
    count_after_rollout = shared_norm.count.clone()
    assert count_after_rollout.item() == batch

    # critic_loss must not update shared_norm with imagined states
    sm.critic_loss(rollout.states, rollout.rewards, rollout.terminals).backward()
    torch.testing.assert_close(shared_norm.count, count_after_rollout)

    # streaming rollout with obs_normalizer matches parallel HistoryWindow unfold
    with torch.no_grad():
        obs_win = sm.obs_history(shared_norm.normalize(rollout.states[:, :-1]), past = shared_norm.normalize(past_obs))
        act_win = sm.action_history(rollout.actions, past = past_actions)
        parallel_in = torch.cat((
            rearrange(obs_win, 'b t w d -> b t (w d)'),
            rearrange(act_win, 'b t w d -> b t (w d)')
        ), dim = -1)
        parallel_actions = actor(parallel_in)

    torch.testing.assert_close(parallel_actions, rollout.actions, atol = 1e-5, rtol = 1e-5)


def test_cold_start_dict_world_model_normalizer_in_supermimic():
    torch.manual_seed(11)

    wm = WorldModel(
        dim_state = 10,
        dim_action = 3,
        dim_sim_context = 0,
        dim_hiddens = (16,),
        pred_reward = True,
        state_normalizer = WelfordNormalizer(exclude = 'quat')
    )

    actor = MLP(10, 16, 3, activation = nn.GELU())
    sm = SuperMimic(
        actor = actor,
        world_model = wm,
        obs_normalizer = WelfordNormalizer(subset = ('pos', 'vel'))
    )

    s0 = dict(pos = torch.randn(2, 3), quat = torch.randn(2, 4), vel = torch.randn(2, 3))
    rollout = sm(s0, num_rollout_timesteps = 3)

    assert rollout.states['pos'].shape == (2, 4, 3)
    assert wm.state_normalizer.count.item() == 0.
    assert sm.obs_normalizer.count.item() == 2.
    assert torch.equal(wm.state_normalizer.subset_mask, sm.obs_normalizer.subset_mask)


# 5. low-variance & cold-start robustness

def test_low_variance_min_count_cold_start():
    torch.manual_seed(12)

    # default min_count = 2: single sample has 0-DOF variance -> must stay exact identity (no 1/sqrt(eps) blowup)
    norm = WelfordNormalizer(4)
    x0 = torch.tensor([[5., -3., 0., 10.]])
    norm.update(x0)

    assert norm.count.item() == 1.
    probe = x0 + 1.
    assert torch.equal(norm.normalize(probe), probe)
    assert torch.equal(norm.denormalize(probe), probe)

    # custom min_count warmup accumulates stats silently, then activates cleanly at count >= min_count
    warmup = WelfordNormalizer(4, min_count = 10)
    b1 = torch.randn(4, 4) * 3. + 2.
    b2 = torch.randn(6, 4) * 3. + 2.

    warmup.update(b1)
    assert torch.equal(warmup.normalize(b1), b1)

    warmup.update(b2)
    all_b = torch.cat((b1, b2), dim = 0)
    torch.testing.assert_close(warmup.mean, all_b.mean(dim = 0), atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(warmup.normalize(all_b).mean(dim = 0), torch.zeros(4), atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(warmup.denormalize(warmup.normalize(all_b)), all_b, atol = 1e-5, rtol = 1e-5)


def test_low_variance_min_std_and_dead_features():
    torch.manual_seed(13)

    # mix: dead 0.0, fp32-noisy constant 0.3, stiff low-var joint (std ~ 0.02), and active dim (std ~ 2.0)
    n = 128
    data = torch.stack((
        torch.zeros(n),
        torch.full((n,), 0.3),
        torch.randn(n) * 0.02 + 1.0,
        torch.randn(n) * 2.0 - 4.0
    ), dim = -1)

    # default min_std = 1e-2 caps dead dims smoothly without biasing active dims
    norm_default = WelfordNormalizer(4)
    norm_default.update(data)

    assert torch.allclose(norm_default.std[:2], torch.full((2,), 1e-2))
    assert norm_default.std[2] > 1e-2
    torch.testing.assert_close(norm_default.std[3], data[:, 3].std(unbiased = False), atol = 1e-4, rtol = 1e-4)

    # explicit scalar floor (e.g. robotics min_std = 0.15) bounds amplification & gradients on stiff/dead dims
    norm_floor = WelfordNormalizer(4, min_std = 0.15)
    norm_floor.update(data)

    probe = torch.tensor([[1.0, 1.3, 1.5, -2.0]], requires_grad = True)
    out = norm_floor.normalize(probe)

    # perturbation of +1.0 on dead dim -> 1.0 / 0.15 (not 316.2!)
    torch.testing.assert_close(out[0, 0], torch.tensor(1.0 / 0.15), atol = 1e-4, rtol = 1e-4)
    torch.testing.assert_close(out[0, 1], torch.tensor(1.0 / 0.15), atol = 1e-4, rtol = 1e-4)
    torch.testing.assert_close(norm_floor.denormalize(out), probe, atol = 1e-5, rtol = 1e-5)

    out.sum().backward()
    assert (probe.grad.abs() <= 1.0 / 0.15 + 1e-5).all()

    # per-key dict min_std + excluded key (excluded key must stay exact std = 1.0)
    dict_data = dict(
        qpos = torch.full((32, 3), 0.3),
        quat = torch.full((32, 4), 1.0),
        qvel = torch.randn(32, 3) * 0.05
    )
    dict_norm = WelfordNormalizer(exclude = 'quat', min_std = dict(qpos = 0.15, qvel = 1.0))
    dict_norm.update(dict_data)

    probe_dict = dict(
        qpos = torch.full((2, 3), 0.6),
        quat = torch.full((2, 4), 0.5),
        qvel = torch.full((2, 3), 2.0)
    )
    out_dict = dict_norm.normalize(probe_dict)

    torch.testing.assert_close(out_dict['qpos'], torch.full((2, 3), (0.6 - 0.3) / 0.15), atol = 1e-4, rtol = 1e-4)
    assert torch.equal(out_dict['quat'], probe_dict['quat'])
    torch.testing.assert_close(dict_norm.denormalize(out_dict)['qvel'], probe_dict['qvel'], atol = 1e-5, rtol = 1e-5)

    # lazy checkpoint roundtrip preserves per-key min_std
    restored = WelfordNormalizer(exclude = 'quat', min_std = dict(qpos = 0.15, qvel = 1.0))
    restored.load_state_dict(dict_norm.state_dict())
    torch.testing.assert_close(restored.normalize(probe_dict)['qpos'], out_dict['qpos'])

    # min_std = None disables the floor for legitimately tiny-scale data
    tiny = torch.randn(64, 2) * 1e-3
    norm_tiny = WelfordNormalizer(2, eps = 1e-10, min_std = None)
    norm_tiny.update(tiny)
    torch.testing.assert_close(norm_tiny.normalize(tiny).var(dim = 0, unbiased = False), torch.ones(2), atol = 1e-3, rtol = 1e-3)

    # WorldModel with a dead state dim stays well-conditioned when that dim later moves
    wm = WorldModel(
        dim_state = 4,
        dim_action = 2,
        dim_sim_context = 0,
        dim_hiddens = (16,),
        pred_terminal = False,
        state_normalizer = WelfordNormalizer(4, min_std = 0.15)
    )
    wm.rollout(data[:8], torch.randn(8, 3, 2), data[8:32].reshape(8, 3, 4))

    perturbed_next = data[:4].clone()
    perturbed_next[:, 0] = 0.5  # dead dim moves by 0.5
    loss, _, losses = wm(data[:4], torch.randn(4, 2), next_state = perturbed_next)
    # normalized MSE on dim 0 is (0.5 / 0.15)^2 / 4 ~ 2.78, not (0.5 / 0.00316)^2 / 4 ~ 6250!
    assert losses.state.item() < 20.
    loss.backward()
    assert has_nonzero_grads(wm)

