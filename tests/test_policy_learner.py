import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
  cap_weights,
  gae_returns,
)

NUM_ENVS = 2


class _DummyActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(()))

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return obs["actor"] * self.weight


class _DummyRepresentation(nn.Module):
  dt = 1.0


class _TwoStepEnv:
  """Reward 1 per step; the first step terminates with probability 0.5."""

  max_step_reward = float("inf")

  def __init__(self):
    self.dynamics = nn.Linear(1, 1)
    self.representation = _DummyRepresentation()
    self.step_idx = 0

  def reset(self, batch: TensorDict) -> TensorDict:
    del batch
    self.step_idx = 0
    return self._obs()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
    rewards = 1.0 + actions.squeeze(-1)
    term_prob = torch.full((NUM_ENVS,), 0.5 if self.step_idx == 0 else 0.0)
    self.step_idx += 1
    return self._obs(), rewards, term_prob

  def _obs(self) -> TensorDict:
    obs = {"actor": torch.zeros(NUM_ENVS, 1), "critic": torch.zeros(NUM_ENVS, 1)}
    return TensorDict(obs, batch_size=[NUM_ENVS])


class _ConstantCritic(nn.Module):
  def __init__(self, value: float):
    super().__init__()
    self.value = nn.Parameter(torch.tensor(value))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return obs * 0.0 + self.value


def test_survival_masking_counts_current_transition_reward_before_done():
  learner = BpttPolicyLearner(
    _DummyActor(),
    _TwoStepEnv(),
    BpttPolicyLearnerCfg(horizon=2, gamma=1.0, survival_masking=True),
  )

  stats = learner.update([TensorDict({}, batch_size=[NUM_ENVS])])

  # Return 1 + (1 - 0.5) * 1, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx(1.5 / 2)


def test_actor_critic_bootstraps_the_surviving_last_state():
  learner = ActorCriticPolicyLearner(
    _DummyActor(),
    _ConstantCritic(1.0),
    _TwoStepEnv(),
    ActorCriticPolicyLearnerCfg(horizon=2, gamma=0.99, survival_masking=True),
  )

  stats = learner.update([TensorDict({}, batch_size=[NUM_ENVS])])

  # Return 1 + 0.99 * 0.5 * 1, plus the value 1 discounted by 0.99^2 and by
  # survival 0.5, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx((1.495 + 0.49005) / 2)
  assert torch.isfinite(stats["Policy/value_function"])


class _PointEnv:
  """s' = s + a, reward -s'^2: each action moves every later state, so the total
  action gradient differs from the one-step one."""

  max_step_reward = 1.0

  def __init__(self):
    self.dynamics = nn.Linear(1, 1)
    self.representation = _DummyRepresentation()

  def reset(self, batch: TensorDict) -> TensorDict:
    self.s = batch["state"]
    return self._obs()

  def step(self, actions: torch.Tensor):
    self.s = self.s + actions
    reward = -self.s.pow(2).squeeze(-1)
    return self._obs(), reward, torch.zeros_like(reward)

  def _obs(self) -> TensorDict:
    obs = {"actor": self.s, "critic": self.s}
    return TensorDict(obs, batch_size=[self.s.shape[0]])


class _LinearActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Linear(1, 1)
    with torch.no_grad():
      self.net.weight.fill_(-0.3)
      self.net.bias.fill_(0.05)

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return self.net(obs["actor"])


def _capped_update(cap: float, states: list[float]):
  actor = _LinearActor()
  cfg = BpttPolicyLearnerCfg(horizon=3, rollout_cap=cap, grad_clip_value=0.0)
  learner = BpttPolicyLearner(actor, _PointEnv(), cfg)
  start = TensorDict({"state": torch.tensor(states)[:, None]}, batch_size=[len(states)])
  stats = learner.update([start])
  # Compare gradients, not parameters: Adam's first step hides a scale error.
  return torch.cat([p.grad.flatten() for p in actor.parameters()]), stats


def test_rollout_cap_that_never_binds_matches_the_uncapped_update():
  states = [0.04, -0.08, 0.12, 0.02]
  plain, stats_plain = _capped_update(0.0, states)
  capped, stats = _capped_update(1e9, states)
  assert stats_plain["Policy/grad_norm"] < 1.0  # below the global clip
  assert plain.abs().min() > 0
  assert torch.allclose(plain, capped, atol=1e-6)
  assert stats["Policy/cap_frac"] == 0.0


def test_rollout_cap_scales_down_only_the_outlier_rollout():
  _, stats = _capped_update(2.0, [0.1, 0.1, 0.1, 10.0])
  assert stats["Policy/cap_frac"] == pytest.approx(0.25)


def test_cap_weights_scale_outliers_and_skip_a_zero_median():
  w = cap_weights(torch.tensor([1.0, 1.0, 1.0, 50.0]), 10.0)
  assert torch.allclose(w, torch.tensor([1.0, 1.0, 1.0, 0.2]))
  assert torch.equal(
    cap_weights(torch.tensor([0.0, 0.0, 0.0, 5.0]), 10.0), torch.ones(4)
  )


def test_gae_returns_discount_by_soft_done():
  returns = gae_returns(
    values=torch.zeros(2, 1),
    last_value=torch.zeros(1),
    rewards=torch.ones(2, 1),
    dones=torch.tensor([[0.5], [0.0]]),
    gamma=0.99,
    lam=1.0,
  )

  assert torch.allclose(returns, torch.tensor([[1.495], [1.0]]))
