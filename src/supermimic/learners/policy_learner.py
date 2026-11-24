"""Policy learners using a differentiable world model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  build_optimizers,
  graph_state,
)


@dataclass
class BpttPolicyLearnerCfg:
  lr: float = 2e-4
  num_batches: int = 1
  batch_size: int = 4096
  horizon: int = 32
  gamma: float = 1.0
  survival_masking: bool = True
  beta2: float = 0.95  # faster Adam EMA recovery after gradient spikes
  grad_clip_value: float = 0.1  # per-parameter gradient element clipping
  rollout_cap: float = 0.0
  """Cap each rollout's gradient at this many batch-median norms (arXiv
  2611.27648 uses 10). 0 is off. Costs a second, policy-only actor pass."""


@dataclass
class ActorCriticPolicyLearnerCfg(BpttPolicyLearnerCfg):
  gamma: float = 0.99
  lam: float = 0.95


def gae_returns(
  values: torch.Tensor,
  last_value: torch.Tensor,
  rewards: torch.Tensor,
  dones: torch.Tensor,
  gamma: float,
  lam: float,
) -> torch.Tensor:
  """GAE returns ``A_t + V(s_t)`` of a time-major rollout. A soft done ``p_t``
  discounts the next state by ``(1 - p_t) * gamma``."""
  returns = torch.empty_like(values)
  advantage = 0.0
  next_value = last_value
  for t in reversed(range(values.shape[0])):
    not_done = 1.0 - dones[t]
    delta = rewards[t] + not_done * gamma * next_value - values[t]
    advantage = delta + not_done * gamma * lam * advantage
    returns[t] = advantage + values[t]
    next_value = values[t]
  return returns


def cap_weights(
  G: torch.Tensor, k: float, median: torch.Tensor | None = None
) -> torch.Tensor:
  """Per-rollout weights that cap each norm in ``G`` at ``k`` batch medians. A
  zero median (most rollouts without gradient) caps nothing, rather than
  zeroing every rollout that has one."""
  if median is None:
    median = G.median()
  w = (k * median / G.clamp_min(1e-12)).clamp(max=1.0)
  return torch.where(median > 0, w, torch.ones_like(w))


def replay_step_stats(
  real: TensorDict, actions: torch.Tensor, motion_times: torch.Tensor
) -> dict[str, torch.Tensor]:
  """Checks on one replayed step. ``replay_action_gap``: mean squared gap between
  the logged and current actions over real states, the policy drift since
  collection. ``replay_time_mismatch``: share of envs whose clip clock differs
  from the simulator's inside the episode; must be 0 (logged, not asserted: an
  assert would sync the host). ``replay_real_frac``: share of real next states."""
  is_real, next_real = real["is_real"], real["next_is_real"]
  action_gap = (real["action"] - actions.detach()).pow(2).mean(-1)
  mismatch = (motion_times != real["motion_time"]) & next_real
  return {
    "replay_action_gap": (action_gap * is_real).sum() / is_real.sum().clamp(min=1),
    "replay_time_mismatch": mismatch.float().mean(),
    "replay_real_frac": next_real.float().mean(),
  }


class BpttPolicyLearner:
  """Policy Learner using direct Backpropagation Through Time (BPTT)."""

  def __init__(
    self,
    policy: nn.Module,
    wm_env: WorldModelEnv,
    cfg: BpttPolicyLearnerCfg,
    device: str = "cpu",
    *,
    compile_actor: bool = True,
  ):
    self.actor = policy
    self.wm_env = wm_env
    self.cfg = cfg
    self.actor_params = list(self.actor.parameters())
    self.actor_optimizers = build_optimizers(
      [self.actor], self.cfg.lr, betas=(0.9, self.cfg.beta2), device=device
    )
    self._rollout_actor = (
      torch.compile(self.actor)
      if compile_actor and torch.device(device).type == "cuda"
      else self.actor
    )
    self._captured_update = CapturedUpdate(self._update_from, self._graph_state)

  def update(self, starts: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
    self.actor.train()
    for module in (self.wm_env.dynamics, self.wm_env.representation):
      module.eval()
      module.requires_grad_(False)
    return self._captured_update(torch.stack(list(starts)))

  def _update_from(self, starts: TensorDict) -> dict[str, torch.Tensor]:
    totals: dict[str, torch.Tensor] = {}
    for start in starts.unbind(0):
      for k, v in self._train_batch(start).items():
        totals[k] = totals[k] + v if k in totals else v
    n = starts.shape[0]
    reward = totals.pop("return") / self.cfg.horizon / n / self.wm_env.representation.dt
    return {"Policy/reward": reward} | {f"Policy/{k}": v / n for k, v in totals.items()}

  def rollout(self, start: TensorDict) -> tuple:
    """Imagine one rollout from ``start``. Returns ``(ret, aux_loss, taken,
    rollout, last_obs, replay_stats)``: the per-rollout return with its
    bootstrap, the actor's auxiliary loss, each step's ``(obs, action)``, the
    detached steps for the critic, the last observation and the DMO checks."""
    cfg = self.cfg
    obs = self.wm_env.reset(start)
    real = start.get("real", None)  # replayed simulator window (DMO arm)
    ret, aux_loss, survival = 0.0, 0.0, 1.0
    rollout: list[TensorDict] = []
    replay_stats: dict[str, torch.Tensor] = {}
    taken: list[tuple[TensorDict, torch.Tensor]] = []
    for t in range(cfg.horizon):
      with torch.autocast("cuda", torch.bfloat16, enabled=obs.is_cuda):
        actions = self._rollout_actor(obs).float()
      taken.append((obs.detach(), actions))
      if real is None:
        next_obs, reward, term_prob = self.wm_env.step(actions)
      else:
        next_obs, reward, term_prob = self.wm_env.step(actions, real[:, t])
        stats = replay_step_stats(real[:, t], actions, self.wm_env.motion_times)
        for k, v in stats.items():
          replay_stats[k] = replay_stats.get(k, 0.0) + v / cfg.horizon
      step = {"critic": obs["critic"], "reward": reward, "done": term_prob}
      rollout.append(TensorDict(step, batch_size=reward.shape).detach())
      ret = ret + cfg.gamma**t * survival * reward
      aux_loss = aux_loss + getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
      if cfg.survival_masking:
        survival = survival * (1 - term_prob)
      obs = next_obs
    ret = ret + self._bootstrap(obs, survival, term_prob)
    return ret, aux_loss, taken, rollout, obs, replay_stats

  def _train_batch(self, start: TensorDict) -> dict[str, torch.Tensor]:
    """Imagine one rollout from ``start`` and take one actor step on its return."""
    cfg = self.cfg
    ret, aux_loss, taken, rollout, obs, replay_stats = self.rollout(start)
    cap_stats: dict[str, torch.Tensor] = {}
    if cfg.rollout_cap:
      loss, cap_stats = self._capped_loss(ret, taken)
    else:
      loss = (-ret + aux_loss).mean()
    grad_norm = _optimize_step(
      self.actor_optimizers, loss, self.actor_params, 1.0, cfg.grad_clip_value
    )
    return {
      "return": ret.mean().detach(),
      "grad_norm": grad_norm,
      **replay_stats,
      **cap_stats,
      **self._fit_critic(rollout, obs),
      **getattr(self.actor, "auxiliary_stats", lambda: {})(),
    }

  def _capped_loss(
    self, ret: torch.Tensor, taken: list[tuple[TensorDict, torch.Tensor]]
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Actor loss with each rollout's action-gradient norm capped at
    ``rollout_cap`` batch medians (arXiv 2611.27648, Sec. V-A Eq. 2 & 5).
    ``g_t = dret/da_t`` is the total derivative in each action, so
    ``sum_t g_t . pi(o_t)`` with ``o_t`` held fixed has the return's parameter
    gradient; weighting it per rollout caps that rollout."""
    obs, actions = zip(*taken, strict=True)
    g = torch.autograd.grad(ret.sum(), actions)
    G = torch.stack([gt.pow(2).sum(-1) for gt in g]).sum(0).sqrt()
    median = G.median()
    w = cap_weights(G, self.cfg.rollout_cap, median)
    surrogate, aux_loss = 0.0, 0.0
    for gt, o in zip(g, obs, strict=True):
      with torch.autocast("cuda", torch.bfloat16, enabled=o.is_cuda):
        a = self._rollout_actor(o).float()
      surrogate = surrogate + (gt * a).sum(-1)
      aux_loss = aux_loss + getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
    loss = (-w.detach() * surrogate + aux_loss).mean()
    # Sort-indexed p99 avoids torch.quantile's host-scalar transfer under CUDA graphs.
    p99_idx = int(0.99 * (G.numel() - 1))
    g_p99 = G.sort().values[p99_idx]
    return loss, {
      "cap_frac": (w < 1.0).float().mean(),
      "G_median": median.detach(),
      "G_p99": g_p99.detach(),
    }

  def _bootstrap(
    self, obs: TensorDict, survival: torch.Tensor, term_prob: torch.Tensor
  ) -> torch.Tensor | float:
    """Value of the rollout's last state, discounted into its return."""
    del obs, survival, term_prob
    return 0.0

  def _fit_critic(
    self, rollout: list[TensorDict], last_obs: TensorDict
  ) -> dict[str, torch.Tensor]:
    """Fit a critic to the rollout after the actor step."""
    del rollout, last_obs
    return {}

  def _graph_state(self) -> list[torch.Tensor]:
    """What the captured update reads that Python can rebind."""
    wm = self.wm_env
    modules = [self.actor, wm.dynamics, wm.representation]
    return graph_state(modules, self.actor_optimizers)


class ActorCriticPolicyLearner(BpttPolicyLearner):
  """Actor-critic variant that adds a critic bootstrap and GAE critic update."""

  def __init__(
    self,
    policy: nn.Module,
    critic: nn.Module,
    wm_env: WorldModelEnv,
    cfg: ActorCriticPolicyLearnerCfg,
    device: str = "cpu",
    *,
    compile_actor: bool = True,
  ):
    super().__init__(policy, wm_env, cfg, device, compile_actor=compile_actor)
    self.critic = critic
    self.critic_optimizers = build_optimizers([critic], self.cfg.lr, device=device)
    self._v_max = self.wm_env.max_step_reward / (1 - self.cfg.gamma)

  def _bootstrap(
    self, obs: TensorDict, survival: torch.Tensor, term_prob: torch.Tensor
  ) -> torch.Tensor:
    self.critic.requires_grad_(False)
    value = self.critic(obs["critic"]).clamp(max=self._v_max).squeeze(-1)
    alive = survival if self.cfg.survival_masking else 1.0 - term_prob
    return self.cfg.gamma**self.cfg.horizon * value * alive

  def _fit_critic(
    self, rollout: list[TensorDict], last_obs: TensorDict
  ) -> dict[str, torch.Tensor]:
    steps = torch.stack(rollout)
    with torch.no_grad():
      values = self.critic(steps["critic"])
      last_value = self.critic(last_obs["critic"])
    returns = gae_returns(
      values,
      last_value,
      steps["reward"].unsqueeze(-1),
      steps["done"].unsqueeze(-1),
      self.cfg.gamma,
      self.cfg.lam,
    ).clamp(max=self._v_max)
    obs, returns = steps["critic"].flatten(0, 1), returns.flatten(0, 1)
    perm = torch.randperm(len(returns) // 4 * 4, device=returns.device)
    self.critic.requires_grad_(True)
    losses = []
    for _ in range(5):
      for idx in perm.view(4, -1):
        loss = (returns[idx] - self.critic(obs[idx])).pow(2).mean()
        _optimize_step(
          self.critic_optimizers, loss, self.critic.parameters(), max_norm=1.0
        )
        losses.append(loss.detach())
    return {"value_function": torch.stack(losses).mean()}

  def _graph_state(self) -> list[torch.Tensor]:
    return super()._graph_state() + graph_state([self.critic], self.critic_optimizers)
