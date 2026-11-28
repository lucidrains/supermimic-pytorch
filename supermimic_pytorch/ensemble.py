from __future__ import annotations
from typing import Callable, Literal, Sequence

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from einops import reduce
from torch_einops_utils import tree_map_tensor

from supermimic_pytorch.latent_world_model import LatentWorldModel
from supermimic_pytorch.supermimic import (
    EnsemblePredictions,
    State,
    batch_size_of,
    default,
    exists,
    flatten_state_with_inverse
)

# helpers

def ensemble_std(stacked: Tensor, eps: float = 1e-12) -> Tensor:
    return (stacked.var(dim = 0, unbiased = True) + eps).sqrt()

def maybe_bootstrap(enabled: bool, batch: int, device, *trees):
    if not enabled:
        return trees

    idx = torch.randint(0, batch, (batch,), device = device)
    return tuple(tree_map_tensor(lambda t: t[idx], tree) if exists(tree) else None for tree in trees)

# ensemble world model - Yu et al. (2020, MOPO); Li, Krause & Hutter (2026, RWM-U)

class WorldModelEnsemble(Module):
    def __init__(
        self,
        world_models: Sequence[Module] | Callable[[], Module],
        *,
        num_members: int | None = None,
        rollout_mode: Literal['mean', 'random'] = 'mean',
        state_uncertainty_weight: float = 1.,
        reward_uncertainty_weight: float = 1.,
        bootstrap_resample: bool = True,
        std_eps: float = 1e-12
    ):
        super().__init__()

        if callable(world_models):
            assert exists(num_members) and num_members >= 2
            members = [world_models() for _ in range(num_members)]
        else:
            members = list(world_models)
            assert len(members) >= 2
            assert not exists(num_members) or num_members == len(members)

        assert rollout_mode in ('mean', 'random')
        assert state_uncertainty_weight >= 0. and reward_uncertainty_weight >= 0. and std_eps > 0.

        self.members = ModuleList(members)
        self.num_members = len(members)
        self.rollout_mode = rollout_mode
        self.state_uncertainty_weight = state_uncertainty_weight
        self.reward_uncertainty_weight = reward_uncertainty_weight
        self.bootstrap_resample = bootstrap_resample
        self.std_eps = std_eps

        first = self.members[0]
        self.dim_action = first.dim_action
        self.num_discrete_actions = first.num_discrete_actions
        self.has_discrete_actions = first.has_discrete_actions
        self.dim_explorative_latent = first.dim_explorative_latent

    @torch.no_grad()
    def update_ema(self):
        for member in self.members:
            if isinstance(member, LatentWorldModel):
                member.update_ema()
        return self

    def aggregate_predictions(self, preds: Sequence) -> EnsemblePredictions:
        flat_states, inverses = zip(*(flatten_state_with_inverse(p.state) for p in preds))

        stacked_states = torch.stack(flat_states, dim = 0)
        mean_state = inverses[0](stacked_states.mean(dim = 0))
        state_std = reduce(ensemble_std(stacked_states, self.std_eps), 'b d -> b', 'mean')

        mean_terminal = torch.stack([p.terminal for p in preds], dim = 0).mean(dim = 0) if exists(preds[0].terminal) else None

        mean_reward = reward_std = None
        if exists(preds[0].reward):
            stacked_rewards = torch.stack([p.reward for p in preds], dim = 0)
            mean_reward = stacked_rewards.mean(dim = 0)
            reward_std = ensemble_std(stacked_rewards, self.std_eps)

        epistemic = self.state_uncertainty_weight * state_std
        if exists(reward_std) and self.reward_uncertainty_weight > 0.:
            epistemic = epistemic + self.reward_uncertainty_weight * reward_std

        return EnsemblePredictions(mean_state, mean_terminal, mean_reward, state_std, reward_std, epistemic)

    def rollout(
        self,
        state: State,
        actions,
        next_states: State,
        sim_context: Tensor | None = None,
        next_terminals: Tensor | None = None,
        next_rewards: Tensor | None = None,
        *,
        bootstrap_resample: bool | None = None,
        **kwargs
    ):
        use_bootstrap = default(bootstrap_resample, self.bootstrap_resample) and self.training
        batch, device = batch_size_of(state), actions.device

        member_losses, member_preds, member_step_losses = [], [], []

        for member in self.members:
            m_s, m_a, m_ns, m_ctx, m_nt, m_nr = maybe_bootstrap(
                use_bootstrap, batch, device,
                state, actions, next_states, sim_context, next_terminals, next_rewards
            )

            loss, preds, losses = member.rollout(
                m_s, m_a, m_ns,
                sim_context = m_ctx,
                next_terminals = m_nt,
                next_rewards = m_nr,
                **kwargs
            )
            member_losses.append(loss)
            member_preds.append(preds)
            member_step_losses.append(losses)

        mean_loss = torch.stack(member_losses).mean(dim = 0)
        agg_preds = [self.aggregate_predictions(step_preds) for step_preds in zip(*member_preds)]
        return mean_loss, agg_preds, member_step_losses

    def forward(
        self,
        state: State,
        action,
        sim_context: Tensor | None = None,
        next_state: State | None = None,
        next_terminal: Tensor | None = None,
        next_reward: Tensor | None = None,
        *,
        rollout_mode: Literal['mean', 'random'] | None = None,
        bootstrap_resample: bool | None = None,
        **kwargs
    ):
        need_loss = exists(next_state) or exists(next_terminal) or exists(next_reward)

        if need_loss:
            use_bootstrap = default(bootstrap_resample, self.bootstrap_resample) and self.training
            batch, device = batch_size_of(state), action.device

            member_losses, preds, losses_list = [], [], []

            for member in self.members:
                m_s, m_a, m_ctx, m_ns, m_nt, m_nr = maybe_bootstrap(
                    use_bootstrap, batch, device,
                    state, action, sim_context, next_state, next_terminal, next_reward
                )
                loss, pred, losses = member(m_s, m_a, m_ctx, m_ns, m_nt, m_nr, **kwargs)
                member_losses.append(loss)
                preds.append(pred)
                losses_list.append(losses)

            mean_loss = torch.stack(member_losses).mean(dim = 0)
            return mean_loss, self.aggregate_predictions(preds), losses_list

        mode = default(rollout_mode, self.rollout_mode)

        # random member sampling during differentiable rollouts

        if mode == 'random' and torch.is_grad_enabled():
            idx = int(torch.randint(0, self.num_members, ()).item())
            pred = self.members[idx](state, action, sim_context, **kwargs)
            return EnsemblePredictions(pred.state, pred.terminal, pred.reward, None, None, None)

        return self.aggregate_predictions([member(state, action, sim_context, **kwargs) for member in self.members])
