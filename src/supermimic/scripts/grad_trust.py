"""Gradient trust: does a policy gradient predict the real change in return?

From N simulator start states of a checkpoint's own play, take the action
gradient of the learner's H-step return two ways: imagined with model states
(SuperMimic) and replayed through the simulator's next H states (DMO). Check
each in the simulator: play the policy from the start with actions shifted by
+delta and -delta along the gradient, the largest element of delta being eps.

  trust = (J+ - J-) / (2 * g . delta)

Reports both hard-termination real return J and smooth-survival return J_soft
(matching arXiv 2611.27648, Eq. 1 & Table I), model finite-difference trust
(``model_fd``), sign agreement, fall/replay-real-fraction diagnostics,
symmetric paired model-vs-sim comparisons on identical start states, Step 4
foot contact-phase bins, mixed-age warmup breakdowns, and critic-bootstrap
gradient-share / no-bootstrap diagnostics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mjlab
import mjlab.tasks  # noqa: F401
import torch
import torch.nn.functional as F
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp.commands import resolve_motion_file
from tensordict import TensorDict

import supermimic.tasks  # noqa: F401
from supermimic.envs.world_model_env import WorldModelEnv, replay_start
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
)
from supermimic.runners.supermimic_runner import build_critic, build_world_model
from supermimic.scripts.evaluate import (
  act,
  load_checkpoint,
  load_run,
  pin_randomization,
)
from supermimic.scripts.train_utils import setup_device
from supermimic.utils.sim_clone import restore, snapshot
from supermimic.utils.wandb_artifacts import download_artifact_cached

BAND_EDGES = (0.0, 1.0, 3.0, 10.0, 30.0, math.inf)  # in batch-median norms
FOOT_CONTACT_SENSOR = "feet_ground_contact"
CONTACT_PHASES = ("flight", "single_support", "double_support", "impact")


@dataclass
class GradTrustConfig:
  wandb_path: str
  motion_file: str | None = None
  checkpoint: str = "latest"
  """Model artifact alias or version (``latest``, ``v3``)."""
  num_starts: int = 1024
  """Total valid start states to collect."""
  chunk_size: int = 0
  """Starts per simulator batch (3 envs per start). 0 runs all ``num_starts`` at once."""
  warmup_steps: int = 100
  """Policy steps from random resets before the starts are taken."""
  warmup_schedule: tuple[int, ...] = ()
  """Optional per-chunk warmup ages (e.g. ``0 5 25 100``) to mix RSI-perturbed
  starts and steady-state starts. When non-empty, overrides ``warmup_steps``."""
  eps: float = 0.05
  """Largest element of the action shift (the paper's setting)."""
  eps_values: tuple[float, ...] = ()
  """Optional multi-eps sweep evaluated on identical starts and gradients.
  When non-empty, overrides ``eps``."""
  ls_iterations: int | None = None
  """Optional override for MuJoCo linesearch iterations (e.g. ``100``)."""
  check_noboot: bool = False
  """Also perturb along the no-bootstrap H-step gradient ``dR_{0..H-1}/da`` and
  measure exact no-bootstrap simulator trust."""
  output_file: str | None = None
  """JSON summary path; raw per-start tensors are saved alongside as ``.pt``."""
  device: str | None = None
  seed: int = 0


def add_foot_contact_sensor(env_cfg) -> None:
  """Attach G1 foot-ground contact sensor (if absent) for Step 4 phase binning."""
  sensors = tuple(env_cfg.scene.sensors or ())
  if any(s.name == FOOT_CONTACT_SENSOR for s in sensors):
    return
  feet_ground_cfg = ContactSensorCfg(
    name=FOOT_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  env_cfg.scene.sensors = sensors + (feet_ground_cfg,)


def classify_contact_phase(
  foot_contacts: torch.Tensor, valid_steps: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """Classify each start by contact phase and count horizon foot touchdowns.

  ``foot_contacts`` has shape ``(N, H + 1, F)`` (bool: feet in ground contact
  at ``t = 0..H``). Returns ``(phase, impact_count)``:
  - ``phase`` (int64 ``(N,)``): ``3`` (``impact``) if any foot transitions
    ``0 -> 1`` on ``t = 0 -> 1``; else ``0`` (``flight``), ``1``
    (``single_support``), or ``2`` (``double_support``) from ``t = 0`` contacts.
  - ``impact_count`` (int64 ``(N,)``): total foot touchdown events across
    ``t = 0..H-1`` (masked by ``valid_steps`` of shape ``(N, H)`` if given).
  """
  touchdowns = ~foot_contacts[:, :-1] & foot_contacts[:, 1:]
  if valid_steps is not None:
    touchdowns = touchdowns & valid_steps.unsqueeze(-1)
  touchdown_0 = touchdowns[:, 0].any(dim=-1)
  start_count = foot_contacts[:, 0].sum(dim=-1).clamp(max=2)
  phase = torch.where(touchdown_0, torch.full_like(start_count, 3), start_count).long()
  impact_count = touchdowns.sum(dim=(1, 2)).long()
  return phase, impact_count


def _fmt(v: float | None, spec: str = ".3f") -> str:
  return "-" if v is None else f"{v:{spec}}"


def _nanmedian(x: torch.Tensor) -> float | None:
  valid = x[torch.isfinite(x)]
  return float(valid.median()) if valid.numel() > 0 else None


def _frac(x: torch.Tensor) -> float | None:
  valid = x[torch.isfinite(x.float())]
  return float(valid.float().mean()) if valid.numel() > 0 else None


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float | None:
  union = int((a | b).sum())
  if union == 0:
    return None
  return int((a & b).sum()) / union


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
  """Midranks matching ``argsort().argsort()`` without ties and tie-averaged otherwise."""
  order = torch.argsort(x, stable=True)
  sx = x[order]
  n = sx.numel()
  is_new = torch.ones(n, dtype=torch.bool, device=x.device)
  is_new[1:] = sx[1:] != sx[:-1]
  group = is_new.cumsum(0) - 1
  idx = torch.arange(n, dtype=torch.float64, device=x.device)
  num_groups = int(group[-1]) + 1
  sums = torch.zeros(num_groups, dtype=torch.float64, device=x.device)
  counts = torch.zeros(num_groups, dtype=torch.float64, device=x.device)
  sums.scatter_add_(0, group, idx)
  counts.scatter_add_(0, group, torch.ones_like(idx))
  ranks = torch.empty(n, dtype=torch.float64, device=x.device)
  ranks[order] = (sums / counts)[group]
  return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
  """Exact Spearman rank correlation in pure PyTorch."""
  valid = torch.isfinite(x) & torch.isfinite(y)
  x, y = x[valid], y[valid]
  if x.numel() < 2:
    return None
  rx = _average_ranks(x)
  ry = _average_ranks(y)
  rx = rx - rx.mean()
  ry = ry - ry.mean()
  denom = rx.norm() * ry.norm()
  if float(denom) == 0.0:
    return None
  return float((rx * ry).sum() / denom)


def _norm_distribution_stats(norm: torch.Tensor) -> dict[str, float]:
  """Batch-level norm statistics: median, p99, and top-0.5% squared-norm share."""
  n = norm.numel()
  if n == 0:
    return {"median_norm": 0.0, "p99_norm": 0.0, "top_0p5_sq_share": 0.0}
  sorted_norm = norm.sort().values
  p99 = float(sorted_norm[int(0.99 * (n - 1))])
  sq = sorted_norm.pow(2)
  total_sq = float(sq.sum())
  k_top = max(1, int(math.ceil(0.005 * n)))
  top_share = float(sq[-k_top:].sum()) / total_sq if total_sq > 0.0 else 0.0
  return {
    "median_norm": float(norm.median()),
    "p99_norm": p99,
    "top_0p5_sq_share": top_share,
  }


def _group_row(
  m: torch.Tensor,
  norm: torch.Tensor,
  rel: torch.Tensor,
  trust: torch.Tensor,
  trust_soft: torch.Tensor | None,
  model_fd_trust: torch.Tensor | None,
  sign_agrees: torch.Tensor | None,
  fell_base: torch.Tensor | None,
  fell_any: torch.Tensor | None,
  boot_grad_share: torch.Tensor | None,
  trust_noboot: torch.Tensor | None,
) -> dict[str, int | float | None]:
  any_m = bool(m.any())
  row: dict[str, int | float | None] = {
    "n": int(m.sum()),
    "trust": _nanmedian(trust[m]) if any_m else None,
    "median_norm": _nanmedian(norm[m]) if any_m else None,
    "frac_ge_3x": float((rel[m] >= 3.0).float().mean()) if any_m else None,
    "frac_ge_10x": float((rel[m] >= 10.0).float().mean()) if any_m else None,
  }
  if trust_soft is not None:
    row["trust_soft"] = _nanmedian(trust_soft[m]) if any_m else None
  if model_fd_trust is not None:
    row["model_fd_trust"] = _nanmedian(model_fd_trust[m]) if any_m else None
  if sign_agrees is not None:
    row["sign_agrees"] = _frac(sign_agrees[m]) if any_m else None
  if fell_base is not None:
    row["fell_base_frac"] = _frac(fell_base[m]) if any_m else None
  if fell_any is not None:
    row["fell_any_frac"] = _frac(fell_any[m]) if any_m else None
  if boot_grad_share is not None:
    row["boot_grad_share"] = _nanmedian(boot_grad_share[m]) if any_m else None
  if trust_noboot is not None:
    row["trust_noboot"] = _nanmedian(trust_noboot[m]) if any_m else None
  return row


def summarize(
  norm: torch.Tensor,
  trust: torch.Tensor,
  value: torch.Tensor,
  *,
  trust_soft: torch.Tensor | None = None,
  model_fd_trust: torch.Tensor | None = None,
  sign_agrees: torch.Tensor | None = None,
  fell_base: torch.Tensor | None = None,
  fell_any: torch.Tensor | None = None,
  replay_real_frac: torch.Tensor | None = None,
  boot_grad_share: torch.Tensor | None = None,
  cos_tot_boot: torch.Tensor | None = None,
  trust_noboot: torch.Tensor | None = None,
  trust_noboot_soft: torch.Tensor | None = None,
  sign_agrees_noboot: torch.Tensor | None = None,
  contact_phase: torch.Tensor | None = None,
  impact_count: torch.Tensor | None = None,
  start_warmup: torch.Tensor | None = None,
) -> dict:
  """Median trust per norm band (in batch medians), value tercile, and optional
  contact-phase, impact-count, warmup-age, and critic-bootstrap slices."""
  median = norm.median()
  rel = norm / median if float(median) > 0.0 else torch.zeros_like(norm)
  bands = {}
  for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True):
    m = (rel >= lo) & (rel < hi)
    row: dict[str, int | float | None] = {
      "n": int(m.sum()),
      "trust": _nanmedian(trust[m]) if m.any() else None,
    }
    if trust_soft is not None:
      row["trust_soft"] = _nanmedian(trust_soft[m]) if m.any() else None
    if model_fd_trust is not None:
      row["model_fd_trust"] = _nanmedian(model_fd_trust[m]) if m.any() else None
    if sign_agrees is not None:
      row["sign_agrees"] = _frac(sign_agrees[m]) if m.any() else None
    if fell_base is not None:
      row["fell_base_frac"] = _frac(fell_base[m]) if m.any() else None
    if fell_any is not None:
      row["fell_any_frac"] = _frac(fell_any[m]) if m.any() else None
      surv = m & ~fell_any.bool()
      row["trust_surviving"] = _nanmedian(trust[surv]) if surv.any() else None
    if replay_real_frac is not None:
      row["replay_real_frac"] = float(replay_real_frac[m].mean()) if m.any() else None
    if boot_grad_share is not None:
      row["boot_grad_share"] = _nanmedian(boot_grad_share[m]) if m.any() else None
    if cos_tot_boot is not None:
      row["cos_tot_boot"] = _nanmedian(cos_tot_boot[m]) if m.any() else None
    if trust_noboot is not None:
      row["trust_noboot"] = _nanmedian(trust_noboot[m]) if m.any() else None
    bands[f"{lo:g}-{hi:g}"] = row
  cuts = value.quantile(torch.tensor([1 / 3, 2 / 3], device=value.device))
  tercile = torch.bucketize(value, cuts)
  terciles = {
    name: _nanmedian(trust[tercile == i]) if (tercile == i).any() else None
    for i, name in enumerate(("low", "mid", "high"))
  }
  out: dict = {
    "median_trust": _nanmedian(trust),
    **_norm_distribution_stats(norm),
    "bands": bands,
    "value": terciles,
  }
  if trust_soft is not None:
    out["median_trust_soft"] = _nanmedian(trust_soft)
  if model_fd_trust is not None:
    out["median_model_fd_trust"] = _nanmedian(model_fd_trust)
  if sign_agrees is not None:
    out["sign_agrees"] = _frac(sign_agrees)
  if fell_base is not None:
    out["fell_base_frac"] = _frac(fell_base)
  if fell_any is not None:
    out["fell_any_frac"] = _frac(fell_any)
  if replay_real_frac is not None:
    out["replay_real_frac"] = float(replay_real_frac.mean())
  if boot_grad_share is not None:
    out["median_boot_grad_share"] = _nanmedian(boot_grad_share)
  if cos_tot_boot is not None:
    out["median_cos_tot_boot"] = _nanmedian(cos_tot_boot)
  if trust_noboot is not None:
    out["median_trust_noboot"] = _nanmedian(trust_noboot)
  if trust_noboot_soft is not None:
    out["median_trust_noboot_soft"] = _nanmedian(trust_noboot_soft)
  if sign_agrees_noboot is not None:
    out["sign_agrees_noboot"] = _frac(sign_agrees_noboot)
  if contact_phase is not None:
    out["by_contact_phase"] = {
      name: _group_row(
        contact_phase == i,
        norm,
        rel,
        trust,
        trust_soft,
        model_fd_trust,
        sign_agrees,
        fell_base,
        fell_any,
        boot_grad_share,
        trust_noboot,
      )
      for i, name in enumerate(CONTACT_PHASES)
    }
  if impact_count is not None:
    impact_masks = {
      "0": impact_count == 0,
      "1": impact_count == 1,
      "2+": impact_count >= 2,
    }
    out["by_impact_count"] = {
      label: _group_row(
        m,
        norm,
        rel,
        trust,
        trust_soft,
        model_fd_trust,
        sign_agrees,
        fell_base,
        fell_any,
        boot_grad_share,
        trust_noboot,
      )
      for label, m in impact_masks.items()
    }
  if start_warmup is not None:
    ages = sorted({int(w) for w in start_warmup.tolist()})
    if len(ages) > 1:
      out["by_warmup"] = {
        str(w): {
          **_group_row(
            start_warmup == w,
            norm,
            rel,
            trust,
            trust_soft,
            model_fd_trust,
            sign_agrees,
            fell_base,
            fell_any,
            boot_grad_share,
            trust_noboot,
          ),
          "p99_norm": _norm_distribution_stats(norm[start_warmup == w])["p99_norm"],
        }
        for w in ages
      }
  return out


def summarize_paired(
  model_r: dict[str, torch.Tensor],
  sim_r: dict[str, torch.Tensor],
  cos_model_sim: torch.Tensor,
) -> dict:
  """Symmetrically compare ``model`` and ``sim`` on identical starts, grouped
  by both ``model`` norm bands (``bands``) and ``sim`` norm bands
  (``on_sim_bands``), plus joint band counts, tail Jaccard overlap, Spearman
  rank correlation of gradient norms, and optional contact-phase breakdown."""
  norm_m = model_r["norm"]
  norm_s = sim_r["norm"]
  median_m = norm_m.median()
  rel_m = norm_m / median_m if float(median_m) > 0.0 else torch.zeros_like(norm_m)
  median_s = norm_s.median()
  rel_s = norm_s / median_s if float(median_s) > 0.0 else torch.zeros_like(norm_s)
  nan_m = torch.full_like(norm_m, math.nan)
  nan_s = torch.full_like(norm_s, math.nan)
  norm_ratio = torch.where(norm_m > 1e-12, norm_s / norm_m.clamp_min(1e-12), nan_m)
  norm_ratio_m_over_s = torch.where(
    norm_s > 1e-12, norm_m / norm_s.clamp_min(1e-12), nan_s
  )

  band_labels = [
    f"{lo:g}-{hi:g}" for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True)
  ]
  masks_m = [
    (rel_m >= lo) & (rel_m < hi)
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True)
  ]
  masks_s = [
    (rel_s >= lo) & (rel_s < hi)
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True)
  ]

  bands = {}
  for label, m in zip(band_labels, masks_m, strict=True):
    bands[label] = {
      "n": int(m.sum()),
      "model_trust": _nanmedian(model_r["trust"][m]) if m.any() else None,
      "sim_trust": _nanmedian(sim_r["trust"][m]) if m.any() else None,
      "model_trust_soft": _nanmedian(model_r["trust_soft"][m]) if m.any() else None,
      "sim_trust_soft": _nanmedian(sim_r["trust_soft"][m]) if m.any() else None,
      "model_sign_agrees": _frac(model_r["sign_agrees"][m]) if m.any() else None,
      "sim_sign_agrees": _frac(sim_r["sign_agrees"][m]) if m.any() else None,
      "median_norm_ratio": _nanmedian(norm_ratio[m]) if m.any() else None,
      "median_cos": _nanmedian(cos_model_sim[m]) if m.any() else None,
      "sim_replay_real_frac": (
        float(sim_r["replay_real_frac"][m].mean()) if m.any() else None
      ),
    }

  on_sim_bands = {}
  for label, m in zip(band_labels, masks_s, strict=True):
    on_sim_bands[label] = {
      "n": int(m.sum()),
      "sim_trust": _nanmedian(sim_r["trust"][m]) if m.any() else None,
      "model_trust": _nanmedian(model_r["trust"][m]) if m.any() else None,
      "sim_trust_soft": _nanmedian(sim_r["trust_soft"][m]) if m.any() else None,
      "model_trust_soft": _nanmedian(model_r["trust_soft"][m]) if m.any() else None,
      "sim_sign_agrees": _frac(sim_r["sign_agrees"][m]) if m.any() else None,
      "model_sign_agrees": _frac(model_r["sign_agrees"][m]) if m.any() else None,
      "median_norm_ratio_model_over_sim": (
        _nanmedian(norm_ratio_m_over_s[m]) if m.any() else None
      ),
      "median_cos": _nanmedian(cos_model_sim[m]) if m.any() else None,
      "sim_replay_real_frac": (
        float(sim_r["replay_real_frac"][m].mean()) if m.any() else None
      ),
    }

  joint_band_counts = {
    bm: {bs: int((mm & ms).sum()) for bs, ms in zip(band_labels, masks_s, strict=True)}
    for bm, mm in zip(band_labels, masks_m, strict=True)
  }

  out: dict = {
    "median_norm_ratio": _nanmedian(norm_ratio),
    "median_norm_ratio_model_over_sim": _nanmedian(norm_ratio_m_over_s),
    "median_cos": _nanmedian(cos_model_sim),
    "spearman_norm": _spearman(norm_m, norm_s),
    "jaccard_ge_3x": _jaccard(rel_m >= 3.0, rel_s >= 3.0),
    "jaccard_ge_10x": _jaccard(rel_m >= 10.0, rel_s >= 10.0),
    "bands": bands,
    "on_sim_bands": on_sim_bands,
    "joint_band_counts": joint_band_counts,
  }
  contact_phase = model_r.get("contact_phase")
  if contact_phase is not None:
    out["by_contact_phase"] = {
      name: {
        "n": int((contact_phase == i).sum()),
        "model_trust": (
          _nanmedian(model_r["trust"][contact_phase == i])
          if (contact_phase == i).any()
          else None
        ),
        "sim_trust": (
          _nanmedian(sim_r["trust"][contact_phase == i])
          if (contact_phase == i).any()
          else None
        ),
        "median_norm_ratio": (
          _nanmedian(norm_ratio[contact_phase == i])
          if (contact_phase == i).any()
          else None
        ),
        "median_cos": (
          _nanmedian(cos_model_sim[contact_phase == i])
          if (contact_phase == i).any()
          else None
        ),
        "jaccard_ge_3x": (
          _jaccard(rel_m[contact_phase == i] >= 3.0, rel_s[contact_phase == i] >= 3.0)
          if (contact_phase == i).any()
          else None
        ),
      }
      for i, name in enumerate(CONTACT_PHASES)
    }
  return out


def _rollout_split(
  learner: BpttPolicyLearner, start: TensorDict
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
  """Same rollout math as ``BpttPolicyLearner.rollout``, returning
  ``(ret_steps, boot, actions)`` separately for bootstrap decomposition."""
  cfg = learner.cfg
  obs = learner.wm_env.reset(start)
  real = start.get("real", None)
  batch_size = start.shape[0]
  dev = obs.device
  ret_steps = torch.zeros(batch_size, device=dev)
  survival = torch.ones(batch_size, device=dev)
  term_prob = torch.zeros(batch_size, device=dev)
  actions_list: list[torch.Tensor] = []
  for t in range(cfg.horizon):
    with torch.autocast("cuda", torch.bfloat16, enabled=obs.is_cuda):
      actions = learner._rollout_actor(obs).float()
    actions_list.append(actions)
    if real is None:
      obs, reward, term_prob = learner.wm_env.step(actions)
    else:
      obs, reward, term_prob = learner.wm_env.step(actions, real[:, t])
    ret_steps = ret_steps + cfg.gamma**t * survival * reward
    if cfg.survival_masking:
      survival = survival * (1.0 - term_prob)
  boot = learner._bootstrap(obs, survival, term_prob)
  if not isinstance(boot, torch.Tensor):
    boot = torch.zeros_like(ret_steps)
  return ret_steps, boot, actions_list


def trust_at_starts_multi(
  env: RslRlVecEnvWrapper,
  wm,
  actor,
  critic,
  runner_cfg,
  num_starts: int,
  warmup_steps: int,
  eps_values: Sequence[float],
  *,
  check_noboot: bool = False,
) -> dict[float, dict[str, dict[str, torch.Tensor]]]:
  """Evaluate one or more ``eps`` perturbations on identical starts and gradients.

  Computes ``start``, ``snap``, ``grads_model``, ``window``, ``grads_sim``,
  contact-phase labels (when ``feet_ground_contact`` is present), and
  critic-bootstrap gradient decomposition once, then runs ``play`` and
  ``play_model`` for every ``eps`` in ``eps_values``. Returns
  ``{eps: {"model": ..., "sim": ...}}``.
  """
  eps_list = list(dict.fromkeys(float(e) for e in eps_values))
  if not eps_list:
    raise ValueError("eps_values must be non-empty")
  if any(e < 0.0 for e in eps_list):
    raise ValueError(f"eps_values must be non-negative, got {eps_list}")

  n = num_starts
  dev = env.unwrapped.device
  base, plus, minus = (torch.arange(i * n, (i + 1) * n, device=dev) for i in range(3))
  everyone = torch.arange(3 * n, device=dev)
  proprio = runner_cfg.actor_obs == "proprio"
  am = env.unwrapped.action_manager
  has_foot_sensor = FOOT_CONTACT_SENSOR in env.unwrapped.scene.sensors

  specs = [ts for ts in actor.task_specs if ts.name == "motion"]
  wm_env = WorldModelEnv(
    wm.dynamics, wm.representation, env, specs, 1, runner_cfg.termination_softness
  )
  policy_cfg = runner_cfg.policy
  if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
    learner: BpttPolicyLearner = ActorCriticPolicyLearner(
      actor, critic, wm_env, policy_cfg, dev, compile_actor=False
    )
  else:
    learner = BpttPolicyLearner(actor, wm_env, policy_cfg, dev, compile_actor=False)
  horizon, gamma = policy_cfg.horizon, policy_cfg.gamma
  for module in (wm.dynamics, wm.representation):
    module.eval()
    module.requires_grad_(False)

  def frame(ids, full_state, motion_times, sim_ctx, action=None) -> TensorDict:
    """A stored frame, as the runner's replay buffer holds it."""
    k = len(ids)
    return TensorDict(
      {
        "state": full_state[ids],
        "action": am.action[ids] * 0 if action is None else action[ids],
        "prev_action": am.action[ids].clone(),
        "motion_time": motion_times[ids],
        "sim_ctx": sim_ctx[ids],
        "terminated": torch.zeros(k, dtype=torch.bool, device=dev),
        "truncated": torch.zeros(k, dtype=torch.bool, device=dev),
      },
      batch_size=[k],
    )

  def value(start: TensorDict) -> torch.Tensor:
    return critic(wm_env.reset(start)["critic"]).squeeze(-1)

  def action_gradients_split(
    start: TensorDict,
  ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Total ``d(ret_steps + boot)/da_t`` and no-bootstrap ``d(ret_steps)/da_t``.

    Uses two separate forward+backward passes (no ``retain_graph=True``) so
    compiled representation helpers never hit donated-buffer errors.
    """
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
      torch.cuda.get_rng_state(dev) if torch.device(dev).type == "cuda" else None
    )
    ret_steps, boot, actions_tot = _rollout_split(learner, start)
    g_tot = [
      g.detach() for g in torch.autograd.grad((ret_steps + boot).sum(), actions_tot)
    ]

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
      torch.cuda.set_rng_state(cuda_rng_state, dev)
    ret_steps2, _, actions_steps = _rollout_split(learner, start)
    g_steps = [g.detach() for g in torch.autograd.grad(ret_steps2.sum(), actions_steps)]
    return g_tot, g_steps

  def play_model(
    start: TensorDict, grads: list[torch.Tensor], eps: float
  ) -> torch.Tensor:
    """Closed-loop finite difference inside the world model (Table I's
    ``model, finite difference`` row)."""
    scale = eps / torch.stack(grads).abs().amax(dim=(0, 2)).clamp_min(1e-12)
    rets = []
    # Common random numbers if proprio noise is active inside WorldModelEnv.
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
      torch.cuda.get_rng_state(dev) if torch.device(dev).type == "cuda" else None
    )
    with torch.no_grad():
      for sign in (1.0, -1.0):
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
          torch.cuda.set_rng_state(cuda_rng_state, dev)
        obs = wm_env.reset(start)
        ret = torch.zeros(n, device=dev)
        survival = torch.ones(n, device=dev)
        term_prob = torch.zeros(n, device=dev)
        for t in range(horizon):
          with torch.autocast("cuda", torch.bfloat16, enabled=obs.is_cuda):
            actions = learner._rollout_actor(obs).float()
          actions = actions + sign * scale[:, None] * grads[t]
          obs, reward, term_prob = wm_env.step(actions)
          ret = ret + gamma**t * survival * reward
          if policy_cfg.survival_masking:
            survival = survival * (1.0 - term_prob)
        ret = ret + learner._bootstrap(obs, survival, term_prob)
        rets.append(ret)
    return rets[0] - rets[1]

  def play(
    snap,
    grads,
    eps: float,
    *,
    collect_window: bool = False,
    include_bootstrap: bool = True,
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, TensorDict | None, torch.Tensor | None
  ]:
    """Restore every env to its start, play H steps with the copies shifted
    along ``grads``; return ``(ret_hard, ret_soft, fell, window, contacts)``."""
    restore(env.unwrapped, {k: v.repeat(3, *[1] * (v.dim() - 1)) for k, v in snap.items()},
            everyone)  # fmt: skip
    scale = eps / torch.stack(grads).abs().amax(dim=(0, 2)).clamp_min(1e-12)
    ret = torch.zeros(3 * n, device=dev)
    ret_soft = torch.zeros(3 * n, device=dev)
    alive = torch.ones(3 * n, dtype=torch.bool, device=dev)
    fell = torch.zeros(3 * n, dtype=torch.bool, device=dev)
    survival_soft = torch.ones(3 * n, device=dev)
    term_prob = torch.zeros(3 * n, device=dev)
    frames = []
    contacts: list[torch.Tensor] = []
    record_contacts = collect_window and has_foot_sensor
    if record_contacts:
      foot_sensor = env.unwrapped.scene[FOOT_CONTACT_SENSOR]
      contacts.append((foot_sensor.data.found[base] > 0).clone())
    with torch.no_grad():
      for t in range(horizon):
        actions, full_state, motion_times, sim_ctx = act(env, wm, actor, proprio)
        delta = scale[:, None] * grads[t]
        actions[plus] += delta
        actions[minus] -= delta
        f = (
          frame(base, full_state, motion_times, sim_ctx, actions)
          if collect_window
          else None
        )
        _, reward, dones, extras = env.step(actions)
        dones_b = dones.bool()
        truncated = extras["time_outs"].bool()
        terminated = dones_b & ~truncated
        if f is not None:
          f["terminated"], f["truncated"] = terminated[base], truncated[base]
          frames.append(f)
        if record_contacts:
          contacts.append((foot_sensor.data.found[base] > 0).clone())

        ret += gamma**t * alive * reward
        ret_soft += gamma**t * survival_soft * reward
        fell |= alive & terminated
        alive &= ~dones_b

        next_full = env.get_observations()["full_state"]
        next_aux = wm.representation.extract_aux_state(next_full)
        ref_pos, ref_quat = wm_env.motion.body_state(motion_times)[:2]
        term_prob = wm_env._compute_term_prob(next_aux, ref_pos, ref_quat)
        term_prob = torch.where(dones_b, torch.ones_like(term_prob), term_prob)
        if policy_cfg.survival_masking:
          survival_soft = survival_soft * (1.0 - term_prob)

      if include_bootstrap or collect_window:
        _, full_state, motion_times, sim_ctx = act(env, wm, actor, proprio)
        final = frame(everyone, full_state, motion_times, sim_ctx)
        if include_bootstrap:
          final_obs = wm_env.reset(final)
          ret += learner._bootstrap(final_obs, alive.float(), 1.0 - alive.float())
          ret_soft += learner._bootstrap(final_obs, survival_soft, term_prob)
        if collect_window:
          frames.append(final[base])
    window = torch.stack(frames, dim=1) if collect_window else None
    foot_contacts = torch.stack(contacts, dim=1) if record_contacts else None
    return ret, ret_soft, fell, window, foot_contacts

  env.reset()
  with torch.no_grad():
    for _ in range(warmup_steps):
      env.step(act(env, wm, actor, proprio)[0])
    _, full_state, motion_times, sim_ctx = act(env, wm, actor, proprio)
    start = frame(base, full_state, motion_times, sim_ctx)
    snap = snapshot(env.unwrapped, base)
    start_value = value(start)
    ends = wm_env.motion.motion_ends
    clip_idx = torch.searchsorted(ends, motion_times[base], right=True).clamp(
      max=ends.numel() - 1
    )
    near_clip_end = (motion_times[base] + horizon) >= ends[clip_idx]

  grads_model, grads_model_steps = action_gradients_split(start)
  model_plays: dict[
    float, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
  ] = {}
  model_noboot_plays: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
  window: TensorDict | None = None
  foot_contacts: torch.Tensor | None = None
  for i, eps in enumerate(eps_list):
    ret_m, ret_soft_m, fell_m, w, fc = play(
      snap, grads_model, eps, collect_window=(i == 0)
    )
    if i == 0:
      window = w
      foot_contacts = fc
    d_ret_model_fd = play_model(start, grads_model, eps)
    model_plays[eps] = (ret_m, ret_soft_m, fell_m, d_ret_model_fd)
    if check_noboot:
      ret_nb, ret_nb_soft, _, _, _ = play(
        snap, grads_model_steps, eps, collect_window=False, include_bootstrap=False
      )
      model_noboot_plays[eps] = (ret_nb, ret_nb_soft)

  assert window is not None
  sim_start = replay_start(window)
  grads_sim, grads_sim_steps = action_gradients_split(sim_start)
  sim_plays: dict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
  sim_noboot_plays: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
  for eps in eps_list:
    ret_s, ret_soft_s, fell_s, _, _ = play(snap, grads_sim, eps, collect_window=False)
    sim_plays[eps] = (ret_s, ret_soft_s, fell_s)
    if check_noboot:
      ret_nb, ret_nb_soft, _, _, _ = play(
        snap, grads_sim_steps, eps, collect_window=False, include_bootstrap=False
      )
      sim_noboot_plays[eps] = (ret_nb, ret_nb_soft)

  def _grad_metrics(
    g_tot: list[torch.Tensor], g_steps: list[torch.Tensor]
  ) -> tuple[torch.Tensor, ...]:
    tot_s = torch.stack(g_tot)  # (H, N, A)
    steps_s = torch.stack(g_steps)
    boot_s = tot_s - steps_s
    norm_tot = tot_s.pow(2).sum(dim=(0, 2)).sqrt()
    norm_steps = steps_s.pow(2).sum(dim=(0, 2)).sqrt()
    norm_boot = boot_s.pow(2).sum(dim=(0, 2)).sqrt()
    max_abs_tot = tot_s.abs().amax(dim=(0, 2)).clamp_min(1e-12)
    max_abs_steps = steps_s.abs().amax(dim=(0, 2)).clamp_min(1e-12)
    nan = torch.full_like(norm_tot, math.nan)
    boot_share = torch.where(
      norm_tot > 1e-12, norm_boot / norm_tot.clamp_min(1e-12), nan
    )
    tot_flat = tot_s.permute(1, 0, 2).reshape(n, -1)
    boot_flat = boot_s.permute(1, 0, 2).reshape(n, -1)
    cos_tb = torch.where(
      (norm_tot > 1e-12) & (norm_boot > 1e-12),
      F.cosine_similarity(tot_flat, boot_flat, dim=-1),
      nan,
    )
    return (
      tot_flat,
      norm_tot,
      max_abs_tot,
      norm_steps,
      max_abs_steps,
      boot_share,
      cos_tb,
    )

  (
    m_flat,
    norm_m,
    max_abs_m,
    norm_m_steps,
    max_abs_m_steps,
    boot_share_m,
    cos_tb_m,
  ) = _grad_metrics(grads_model, grads_model_steps)
  (
    s_flat,
    norm_s,
    max_abs_s,
    norm_s_steps,
    max_abs_s_steps,
    boot_share_s,
    cos_tb_s,
  ) = _grad_metrics(grads_sim, grads_sim_steps)

  cos_model_sim = F.cosine_similarity(m_flat, s_flat, dim=-1)
  replay_real_frac_model = torch.ones(n, device=dev)
  next_is_real = sim_start["real"]["next_is_real"]
  replay_real_frac_sim = next_is_real.float().mean(dim=1)
  start_warmup_t = torch.full((n,), int(warmup_steps), dtype=torch.long, device=dev)

  contact_phase_t: torch.Tensor | None = None
  impact_count_t: torch.Tensor | None = None
  if foot_contacts is not None:
    contact_phase_t, impact_count_t = classify_contact_phase(
      foot_contacts, next_is_real
    )

  out_by_eps: dict[float, dict[str, dict[str, torch.Tensor]]] = {}
  for eps in eps_list:
    ret_m, ret_soft_m, fell_m, d_ret_model_fd = model_plays[eps]
    ret_s, ret_soft_s, fell_s = sim_plays[eps]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for (
      source,
      norm,
      max_abs,
      norm_steps,
      max_abs_steps,
      boot_share,
      cos_tb,
      ret,
      ret_soft,
      fell,
      real_frac,
      d_ret_fd,
      nb_play,
    ) in (
      (
        "model",
        norm_m,
        max_abs_m,
        norm_m_steps,
        max_abs_m_steps,
        boot_share_m,
        cos_tb_m,
        ret_m,
        ret_soft_m,
        fell_m,
        replay_real_frac_model,
        d_ret_model_fd,
        model_noboot_plays.get(eps),
      ),
      (
        "sim",
        norm_s,
        max_abs_s,
        norm_s_steps,
        max_abs_s_steps,
        boot_share_s,
        cos_tb_s,
        ret_s,
        ret_soft_s,
        fell_s,
        replay_real_frac_sim,
        None,
        sim_noboot_plays.get(eps),
      ),
    ):
      scale = eps / max_abs
      predicted = scale * norm.pow(2)  # g . delta
      denom = 2.0 * predicted
      valid_pred = denom > 1e-12
      d_ret = ret[plus] - ret[minus]
      d_ret_soft = ret_soft[plus] - ret_soft[minus]
      nan = torch.full_like(d_ret, math.nan)
      trust = torch.where(valid_pred, d_ret / denom.clamp_min(1e-12), nan)
      trust_soft = torch.where(valid_pred, d_ret_soft / denom.clamp_min(1e-12), nan)
      sign_agrees = torch.where(valid_pred, (d_ret > 0).float(), nan)
      entry: dict[str, torch.Tensor] = {
        "norm": norm,
        "norm_steps": norm_steps,
        "boot_grad_share": boot_share,
        "cos_tot_boot": cos_tb,
        "trust": trust,
        "trust_soft": trust_soft,
        "sign_agrees": sign_agrees,
        "value": start_value,
        "d_ret": d_ret,
        "d_ret_soft": d_ret_soft,
        "fell_base": fell[base],
        "fell_plus": fell[plus],
        "fell_minus": fell[minus],
        "fell_any": fell[base] | fell[plus] | fell[minus],
        "replay_real_frac": real_frac,
        "near_clip_end": near_clip_end,
        "cos_model_sim": cos_model_sim,
        "start_warmup": start_warmup_t,
      }
      if d_ret_fd is not None:
        entry["model_fd_trust"] = torch.where(
          valid_pred, d_ret_fd / denom.clamp_min(1e-12), nan
        )
      if nb_play is not None:
        ret_nb, ret_nb_soft = nb_play
        scale_nb = eps / max_abs_steps
        denom_nb = 2.0 * scale_nb * norm_steps.pow(2)
        valid_nb = denom_nb > 1e-12
        d_ret_nb = ret_nb[plus] - ret_nb[minus]
        d_ret_nb_soft = ret_nb_soft[plus] - ret_nb_soft[minus]
        entry["d_ret_noboot"] = d_ret_nb
        entry["trust_noboot"] = torch.where(
          valid_nb, d_ret_nb / denom_nb.clamp_min(1e-12), nan
        )
        entry["trust_noboot_soft"] = torch.where(
          valid_nb, d_ret_nb_soft / denom_nb.clamp_min(1e-12), nan
        )
        entry["sign_agrees_noboot"] = torch.where(valid_nb, (d_ret_nb > 0).float(), nan)
      if contact_phase_t is not None and impact_count_t is not None:
        entry["contact_phase"] = contact_phase_t
        entry["impact_count"] = impact_count_t
        if foot_contacts is not None:
          entry["foot_contacts"] = foot_contacts
      out[source] = entry
    out_by_eps[eps] = out
  return out_by_eps


def trust_at_starts(
  env: RslRlVecEnvWrapper,
  wm,
  actor,
  critic,
  runner_cfg,
  num_starts: int,
  warmup_steps: int,
  eps: float,
  *,
  check_noboot: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
  """Single-``eps`` convenience wrapper around :func:`trust_at_starts_multi`."""
  return trust_at_starts_multi(
    env,
    wm,
    actor,
    critic,
    runner_cfg,
    num_starts,
    warmup_steps,
    (eps,),
    check_noboot=check_noboot,
  )[float(eps)]


def _concat_chunks(
  chunks: list[dict[str, dict[str, torch.Tensor]]], target_n: int
) -> dict[str, dict[str, torch.Tensor]]:
  out: dict[str, dict[str, torch.Tensor]] = {}
  for source in ("model", "sim"):
    keys = chunks[0][source].keys()
    out[source] = {
      k: torch.cat([c[source][k] for c in chunks], dim=0)[:target_n] for k in keys
    }
  return out


def _summarize_eps(
  per_source: dict[str, dict[str, torch.Tensor]], eps: float, actual_starts: int
) -> dict:
  out: dict = {"eps": eps}
  for source, r in per_source.items():
    summary = summarize(
      r["norm"],
      r["trust"],
      r["value"],
      trust_soft=r["trust_soft"],
      model_fd_trust=r.get("model_fd_trust"),
      sign_agrees=r["sign_agrees"],
      fell_base=r["fell_base"],
      fell_any=r["fell_any"],
      replay_real_frac=r["replay_real_frac"],
      boot_grad_share=r.get("boot_grad_share"),
      cos_tot_boot=r.get("cos_tot_boot"),
      trust_noboot=r.get("trust_noboot"),
      trust_noboot_soft=r.get("trust_noboot_soft"),
      sign_agrees_noboot=r.get("sign_agrees_noboot"),
      contact_phase=r.get("contact_phase"),
      impact_count=r.get("impact_count"),
      start_warmup=r.get("start_warmup"),
    )
    # With --eps 0 these measure the noise floor between copies; see trust_at_starts_multi.
    summary["median_abs_d_ret"] = float(r["d_ret"].abs().median())
    summary["p95_abs_d_ret"] = float(
      r["d_ret"].abs().sort().values[int(0.95 * (actual_starts - 1))]
    )
    out[source] = summary
  out["paired"] = summarize_paired(
    per_source["model"], per_source["sim"], per_source["model"]["cos_model_sim"]
  )
  return out


def _print_eps_report(
  eps_summary: dict, eps: float, actual_starts: int, filtered_count: int
) -> None:
  for source in ("model", "sim"):
    s = eps_summary[source]
    if eps == 0.0:
      print(
        f"\n{source} states (eps=0 noise floor, n={actual_starts}): "
        f"median |J+ - J-| {s['median_abs_d_ret']:.2e}, "
        f"p95 |J+ - J-| {s['p95_abs_d_ret']:.2e}"
      )
      continue
    boot_str = (
      f", ||g_V||/||g||={_fmt(s.get('median_boot_grad_share'))}"
      if "median_boot_grad_share" in s
      else ""
    )
    nb_str = (
      f", noboot={_fmt(s.get('median_trust_noboot'))}"
      if "median_trust_noboot" in s
      else ""
    )
    print(
      f"\n{source} states (n={actual_starts}, filtered_end={filtered_count}): "
      f"median trust {_fmt(s['median_trust'])} (soft {_fmt(s['median_trust_soft'])}"
      f"{nb_str}), sign {_fmt(s['sign_agrees'])}, "
      f"G_med={s['median_norm']:.3e}, top0.5%_sq={s['top_0p5_sq_share']:.3f}"
      f"{boot_str}, median |J+ - J-| {s['median_abs_d_ret']:.2e}"
    )
    for band, row in s["bands"].items():
      t = _fmt(row["trust"])
      ts = _fmt(row["trust_soft"])
      sg = _fmt(row["sign_agrees"], ".2f")
      fb = _fmt(row["fell_base_frac"], ".2f")
      fa = _fmt(row["fell_any_frac"], ".2f")
      extra = ""
      if "model_fd_trust" in row:
        extra += f"  model_fd={_fmt(row['model_fd_trust'])}"
      elif row["replay_real_frac"] is not None:
        extra += f"  real_frac={row['replay_real_frac']:.2f}"
      if "boot_grad_share" in row:
        extra += f"  gV={_fmt(row['boot_grad_share'], '.2f')}"
      if "trust_noboot" in row:
        extra += f"  nb={_fmt(row['trust_noboot'])}"
      print(
        f"  {band:>8} medians  n={row['n']:5d}  trust={t}  soft={ts}  "
        f"sign={sg}  fell={fb}/{fa}{extra}"
      )
    if "by_contact_phase" in s:
      print("  by contact phase:")
      for phase, row in s["by_contact_phase"].items():
        print(
          f"    {phase:>14s}  n={row['n']:5d}  trust={_fmt(row['trust'])}  "
          f"G_med={_fmt(row['median_norm'], '.3e')}  >=3x={_fmt(row['frac_ge_3x'], '.2f')}"
        )
    if "by_warmup" in s:
      print("  by warmup steps:")
      for age, row in s["by_warmup"].items():
        print(
          f"    warmup={age:>4s}  n={row['n']:5d}  trust={_fmt(row['trust'])}  "
          f"G_med={_fmt(row['median_norm'], '.3e')}  p99={_fmt(row['p99_norm'], '.3e')}  "
          f">=10x={_fmt(row['frac_ge_10x'], '.3f')}"
        )

  if eps != 0.0:
    p = eps_summary["paired"]
    print(
      f"\npaired (model bands): "
      f"median ||g_sim||/||g_model||={_fmt(p['median_norm_ratio'])}, "
      f"median cos={_fmt(p['median_cos'])}, "
      f"Spearman(G_m, G_s)={_fmt(p['spearman_norm'])}, "
      f"Jaccard >=3x={_fmt(p['jaccard_ge_3x'])}, "
      f">=10x={_fmt(p['jaccard_ge_10x'])}"
    )
    for band, row in p["bands"].items():
      print(
        f"  {band:>8} medians  n={row['n']:5d}  "
        f"model={_fmt(row['model_trust'])}  sim={_fmt(row['sim_trust'])}  "
        f"ratio={_fmt(row['median_norm_ratio'])}  cos={_fmt(row['median_cos'])}"
      )
    print(
      f"\npaired (sim bands): "
      f"median ||g_model||/||g_sim||={_fmt(p['median_norm_ratio_model_over_sim'])}, "
      f"median cos={_fmt(p['median_cos'])}"
    )
    for band, row in p["on_sim_bands"].items():
      print(
        f"  {band:>8} medians  n={row['n']:5d}  "
        f"sim={_fmt(row['sim_trust'])}  model={_fmt(row['model_trust'])}  "
        f"ratio={_fmt(row['median_norm_ratio_model_over_sim'])}  "
        f"cos={_fmt(row['median_cos'])}"
      )
    if "by_contact_phase" in p:
      print("\npaired (contact phase):")
      for phase, row in p["by_contact_phase"].items():
        print(
          f"  {phase:>14s}  n={row['n']:5d}  model={_fmt(row['model_trust'])}  "
          f"sim={_fmt(row['sim_trust'])}  ratio={_fmt(row['median_norm_ratio'])}  "
          f"cos={_fmt(row['median_cos'])}"
        )


def grad_trust(task_id: str, cfg: GradTrustConfig) -> dict:
  eps_list = (
    list(dict.fromkeys(float(e) for e in cfg.eps_values))
    if cfg.eps_values
    else [float(cfg.eps)]
  )
  warmup_list = (
    [int(w) for w in cfg.warmup_schedule]
    if cfg.warmup_schedule
    else [int(cfg.warmup_steps)]
  )
  if any(w < 0 for w in warmup_list):
    raise ValueError(f"warmup steps must be non-negative, got {warmup_list}")

  device = setup_device(cfg.device)
  torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg(task_id, play=False)
  pin_randomization(env_cfg)  # copies share one model; see sim_clone.snapshot
  add_foot_contact_sensor(env_cfg)
  if cfg.ls_iterations is not None:
    env_cfg.sim.mujoco.ls_iterations = cfg.ls_iterations
  api, run, runner_cfg = load_run(task_id, cfg.wandb_path)
  command = env_cfg.commands["motion"]
  if cfg.motion_file is not None:
    command.motion_file = resolve_motion_file(cfg.motion_file)
  else:
    artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
    if not artifacts:
      raise ValueError("No --motion-file given and the run has no motions artifact")
    command.motion_file = [
      str(download_artifact_cached(a) / "motion.npz") for a in artifacts
    ]
  command.sampling_mode = "uniform"
  env_cfg.episode_length_s = int(1e9)
  env_cfg.seed = cfg.seed
  chunk_n = (
    min(cfg.num_starts, cfg.chunk_size) if cfg.chunk_size > 0 else cfg.num_starts
  )
  if len(warmup_list) > 1:
    chunk_n = min(chunk_n, max(1, math.ceil(cfg.num_starts / len(warmup_list))))
  env_cfg.scene.num_envs = 3 * chunk_n
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  wm = build_world_model(runner_cfg, env, device)
  actor, model_dir, artifact = load_checkpoint(
    api, run, cfg.checkpoint, wm, runner_cfg, device
  )
  critic = build_critic(runner_cfg, wm, device)
  critic.load_state_dict(torch.load(model_dir / "critic.pt", map_location=device))

  # Collect non-truncating starts (at least H frames before clip end), chunked
  # across warmup_list so mixed-age schedules balance RSI-perturbed and steady starts.
  valid_chunks: dict[float, list[dict[str, dict[str, torch.Tensor]]]] = {
    e: [] for e in eps_list
  }
  all_chunks: dict[float, list[dict[str, dict[str, torch.Tensor]]]] = {
    e: [] for e in eps_list
  }
  collected = 0
  filtered_count = 0
  max_rounds = math.ceil(cfg.num_starts / chunk_n) + max(3, len(warmup_list))
  first_eps = eps_list[0]
  for round_idx in range(max_rounds):
    w_steps = warmup_list[round_idx % len(warmup_list)]
    raw_by_eps = trust_at_starts_multi(
      env,
      wm,
      actor,
      critic,
      runner_cfg,
      chunk_n,
      w_steps,
      eps_list,
      check_noboot=cfg.check_noboot,
    )
    keep = ~raw_by_eps[first_eps]["model"]["near_clip_end"]
    filtered_count += int((~keep).sum())
    for e in eps_list:
      raw = raw_by_eps[e]
      all_chunks[e].append(
        {s: {k: v.cpu() for k, v in r.items()} for s, r in raw.items()}
      )
      if keep.any():
        valid_chunks[e].append(
          {s: {k: v[keep].cpu() for k, v in r.items()} for s, r in raw.items()}
        )
    if keep.any():
      collected += int(keep.sum())
    if collected >= cfg.num_starts:
      break

  per_eps_tensors: dict[float, dict[str, dict[str, torch.Tensor]]] = {}
  if collected > 0:
    for e in eps_list:
      per_eps_tensors[e] = _concat_chunks(valid_chunks[e], cfg.num_starts)
  else:
    print("[WARN] All sampled starts were within H frames of clip end; keeping all.")
    for e in eps_list:
      per_eps_tensors[e] = _concat_chunks(all_chunks[e], cfg.num_starts)
  actual_starts = int(per_eps_tensors[first_eps]["model"]["norm"].numel())

  base_meta: dict = {
    "run": cfg.wandb_path,
    "checkpoint": artifact.version,
    "num_starts": actual_starts,
    "filtered_near_clip_end": filtered_count,
    "warmup_schedule": warmup_list,
    "check_noboot": cfg.check_noboot,
    "ls_iterations": int(env_cfg.sim.mujoco.ls_iterations),
  }

  if len(eps_list) == 1:
    eps = eps_list[0]
    eps_summary = _summarize_eps(per_eps_tensors[eps], eps, actual_starts)
    results: dict = {**base_meta, **eps_summary}
    _print_eps_report(eps_summary, eps, actual_starts, filtered_count)
    raw_to_save: dict = per_eps_tensors[eps]
  else:
    by_eps: dict[str, dict] = {}
    raw_by_eps_save: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for eps in eps_list:
      key = f"{eps:g}"
      eps_summary = _summarize_eps(per_eps_tensors[eps], eps, actual_starts)
      by_eps[key] = eps_summary
      raw_by_eps_save[key] = per_eps_tensors[eps]
      print(f"\n=== eps={key} ===")
      _print_eps_report(eps_summary, eps, actual_starts, filtered_count)
    results = {**base_meta, "eps_values": eps_list, "by_eps": by_eps}
    raw_to_save = {"by_eps": raw_by_eps_save}

  if cfg.output_file:
    out = Path(cfg.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, allow_nan=False))
    raw_path = out.with_suffix(".pt")
    torch.save(raw_to_save, raw_path)
    print(f"[INFO] Wrote {cfg.output_file} and {raw_path}")
  env.close()
  return results


# Drop UsePythonSyntaxForLiteralCollections and FlagConversionOff so space-separated
# collections (--eps-values / --warmup-schedule) and bare boolean flags (--check-noboot)
# parse cleanly.
CLI_FLAGS = tuple(
  f
  for f in mjlab.TYRO_FLAGS
  if f
  not in (tyro.conf.UsePythonSyntaxForLiteralCollections, tyro.conf.FlagConversionOff)
)


def parse_args(args: Sequence[str] | None = None) -> tuple[str, GradTrustConfig]:
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    args=args,
    add_help=False,
    return_unknown_args=True,
    config=CLI_FLAGS,
  )
  cfg = tyro.cli(GradTrustConfig, args=remaining_args, config=CLI_FLAGS)
  return chosen_task, cfg


def main() -> None:
  chosen_task, cfg = parse_args()
  grad_trust(chosen_task, cfg)


if __name__ == "__main__":
  main()
