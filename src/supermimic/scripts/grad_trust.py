"""Gradient trust: does a policy gradient predict the real change in return?

From N simulator start states of a checkpoint's own play, take the action
gradient of the learner's H-step return two ways: imagined with model states
(SuperMimic) and replayed through the simulator's next H states (DMO). Check
each in the simulator: play the policy from the start with actions shifted by
+delta and -delta along the gradient, the largest element of delta being eps.

  trust = (J+ - J-) / (2 * g . delta)

Reports both hard-termination real return J and smooth-survival return J_soft
(matching arXiv 2611.27648, Eq. 1 & Table I), model finite-difference trust
(``model_fd``), sign agreement, fall/replay-real-fraction diagnostics, and
paired model-vs-sim comparisons on identical start states.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mjlab
import mjlab.tasks  # noqa: F401
import torch
import torch.nn.functional as F
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
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
  eps: float = 0.05
  """Largest element of the action shift (the paper's setting)."""
  output_file: str | None = None
  """JSON summary path; raw per-start tensors are saved alongside as ``.pt``."""
  device: str | None = None
  seed: int = 0


def _nanmedian(x: torch.Tensor) -> float | None:
  valid = x[torch.isfinite(x)]
  return float(valid.median()) if valid.numel() > 0 else None


def _frac(x: torch.Tensor) -> float | None:
  valid = x[torch.isfinite(x.float())]
  return float(valid.float().mean()) if valid.numel() > 0 else None


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
) -> dict:
  """Median trust per norm band (in batch medians) and per critic-value tercile."""
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
      row["replay_real_frac"] = (
        float(replay_real_frac[m].mean()) if m.any() else None
      )
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
  return out


def summarize_paired(
  model_r: dict[str, torch.Tensor],
  sim_r: dict[str, torch.Tensor],
  cos_model_sim: torch.Tensor,
) -> dict:
  """Compare ``model`` and ``sim`` on identical starts, grouped by ``model`` norm bands."""
  norm_m = model_r["norm"]
  norm_s = sim_r["norm"]
  median_m = norm_m.median()
  rel_m = norm_m / median_m if float(median_m) > 0.0 else torch.zeros_like(norm_m)
  norm_ratio = torch.where(
    norm_m > 1e-12, norm_s / norm_m.clamp_min(1e-12), torch.full_like(norm_m, math.nan)
  )
  bands = {}
  for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True):
    m = (rel_m >= lo) & (rel_m < hi)
    bands[f"{lo:g}-{hi:g}"] = {
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
  return {
    "median_norm_ratio": _nanmedian(norm_ratio),
    "median_cos": _nanmedian(cos_model_sim),
    "bands": bands,
  }


def trust_at_starts(
  env: RslRlVecEnvWrapper,
  wm,
  actor,
  critic,
  runner_cfg,
  num_starts: int,
  warmup_steps: int,
  eps: float,
) -> dict[str, dict[str, torch.Tensor]]:
  """Per gradient source (``model``, ``sim``): each start's gradient norm,
  hard/soft simulator trust, model finite-difference trust (for ``model``),
  sign agreement, critic value, fall flags, replay real fraction, and
  ``d_ret = J+ - J-``. The copies sit at different world positions, so float32
  rounding alone makes ``d_ret`` about 1e-5 at ``eps = 0``; real shifts must
  move the return well above that. ``env`` has ``3 * num_starts`` envs: the
  starts, then their +delta and -delta copies."""
  n = num_starts
  dev = env.unwrapped.device
  base, plus, minus = (torch.arange(i * n, (i + 1) * n, device=dev) for i in range(3))
  everyone = torch.arange(3 * n, device=dev)
  proprio = runner_cfg.actor_obs == "proprio"
  am = env.unwrapped.action_manager

  specs = [ts for ts in actor.task_specs if ts.name == "motion"]
  wm_env = WorldModelEnv(
    wm.dynamics, wm.representation, env, specs, 1, runner_cfg.termination_softness
  )
  policy_cfg = runner_cfg.policy
  if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
    learner = ActorCriticPolicyLearner(
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

  def action_gradient(start: TensorDict) -> list[torch.Tensor]:
    """``dret/da_t`` for t < H: the gradient training uses (see ``_capped_loss``)."""
    ret, _, taken, *_ = learner.rollout(start)
    grads = torch.autograd.grad(ret.sum(), [a for _, a in taken])
    return [g.detach() for g in grads]

  def play_model(start: TensorDict, grads: list[torch.Tensor]) -> torch.Tensor:
    """Closed-loop finite difference inside the world model (Table I's
    ``model, finite difference`` row)."""
    scale = eps / torch.stack(grads).abs().amax(dim=(0, 2)).clamp_min(1e-12)
    rets = []
    # Common random numbers if proprio noise is active inside WorldModelEnv.
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(dev) if torch.device(dev).type == "cuda" else None
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
    snap, grads
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TensorDict]:
    """Restore every env to its start, play H steps with the copies shifted
    along ``grads``; return ``(ret_hard, ret_soft, fell, base_window)``."""
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
    with torch.no_grad():
      for t in range(horizon):
        actions, full_state, motion_times, sim_ctx = act(env, wm, actor, proprio)
        delta = scale[:, None] * grads[t]
        actions[plus] += delta
        actions[minus] -= delta
        f = frame(base, full_state, motion_times, sim_ctx, actions)
        _, reward, dones, extras = env.step(actions)
        dones_b = dones.bool()
        truncated = extras["time_outs"].bool()
        terminated = dones_b & ~truncated
        f["terminated"], f["truncated"] = terminated[base], truncated[base]
        frames.append(f)

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

      _, full_state, motion_times, sim_ctx = act(env, wm, actor, proprio)
      final = frame(everyone, full_state, motion_times, sim_ctx)
      final_obs = wm_env.reset(final)
      ret += learner._bootstrap(final_obs, alive.float(), 1.0 - alive.float())
      ret_soft += learner._bootstrap(final_obs, survival_soft, term_prob)
      frames.append(final[base])
    return ret, ret_soft, fell, torch.stack(frames, dim=1)

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

  out: dict[str, dict[str, torch.Tensor]] = {}
  grads_model = action_gradient(start)
  ret_m, ret_soft_m, fell_m, window = play(snap, grads_model)
  d_ret_model_fd = play_model(start, grads_model)

  sim_start = replay_start(window)
  grads_sim = action_gradient(sim_start)
  ret_s, ret_soft_s, fell_s, _ = play(snap, grads_sim)

  g_m_stack = torch.stack(grads_model)  # (H, N, A)
  g_s_stack = torch.stack(grads_sim)
  cos_model_sim = F.cosine_similarity(
    g_m_stack.permute(1, 0, 2).reshape(n, -1),
    g_s_stack.permute(1, 0, 2).reshape(n, -1),
    dim=-1,
  )
  replay_real_frac_sim = sim_start["real"]["next_is_real"].float().mean(dim=1)

  for source, g, ret, ret_soft, fell, real_frac, d_ret_fd in (
    ("model", g_m_stack, ret_m, ret_soft_m, fell_m, torch.ones(n, device=dev), d_ret_model_fd),
    ("sim", g_s_stack, ret_s, ret_soft_s, fell_s, replay_real_frac_sim, None),
  ):
    norm = g.pow(2).sum(dim=(0, 2)).sqrt()
    scale = eps / g.abs().amax(dim=(0, 2)).clamp_min(1e-12)
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
    }
    if d_ret_fd is not None:
      entry["model_fd_trust"] = torch.where(
        valid_pred, d_ret_fd / denom.clamp_min(1e-12), nan
      )
    out[source] = entry
  return out


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


def grad_trust(task_id: str, cfg: GradTrustConfig) -> dict:
  device = setup_device(cfg.device)
  torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg(task_id, play=False)
  pin_randomization(env_cfg)  # copies share one model; see sim_clone.snapshot
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
  env_cfg.scene.num_envs = 3 * chunk_n
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  wm = build_world_model(runner_cfg, env, device)
  actor, model_dir, artifact = load_checkpoint(
    api, run, cfg.checkpoint, wm, runner_cfg, device
  )
  critic = build_critic(runner_cfg, wm, device)
  critic.load_state_dict(torch.load(model_dir / "critic.pt", map_location=device))

  # Collect non-truncating starts (at least H frames before clip end), chunked.
  valid_chunks: list[dict[str, dict[str, torch.Tensor]]] = []
  all_chunks: list[dict[str, dict[str, torch.Tensor]]] = []
  collected = 0
  filtered_count = 0
  max_rounds = math.ceil(cfg.num_starts / chunk_n) + 3
  for _ in range(max_rounds):
    raw = trust_at_starts(
      env, wm, actor, critic, runner_cfg, chunk_n, cfg.warmup_steps, cfg.eps
    )
    all_chunks.append(raw)
    keep = ~raw["model"]["near_clip_end"]
    filtered_count += int((~keep).sum())
    if keep.any():
      valid_chunks.append(
        {
          s: {k: v[keep].cpu() for k, v in r.items()}
          for s, r in raw.items()
        }
      )
      collected += int(keep.sum())
    if collected >= cfg.num_starts:
      break

  if collected > 0:
    per_source = _concat_chunks(valid_chunks, cfg.num_starts)
    actual_starts = int(per_source["model"]["norm"].numel())
  else:
    print("[WARN] All sampled starts were within H frames of clip end; keeping all.")
    cpu_all = [{s: {k: v.cpu() for k, v in r.items()} for s, r in c.items()} for c in all_chunks]
    per_source = _concat_chunks(cpu_all, cfg.num_starts)
    actual_starts = int(per_source["model"]["norm"].numel())

  results: dict = {
    "run": cfg.wandb_path,
    "checkpoint": artifact.version,
    "num_starts": actual_starts,
    "filtered_near_clip_end": filtered_count,
    "eps": cfg.eps,
  }
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
    )
    # With --eps 0 these measure the noise floor between copies; see trust_at_starts.
    summary["median_abs_d_ret"] = float(r["d_ret"].abs().median())
    summary["p95_abs_d_ret"] = float(
      r["d_ret"].abs().sort().values[int(0.95 * (actual_starts - 1))]
    )
    results[source] = summary

  results["paired"] = summarize_paired(
    per_source["model"], per_source["sim"], per_source["model"]["cos_model_sim"]
  )

  for source in ("model", "sim"):
    s = results[source]
    if cfg.eps == 0.0:
      print(
        f"\n{source} states (eps=0 noise floor, n={actual_starts}): "
        f"median |J+ - J-| {s['median_abs_d_ret']:.2e}, "
        f"p95 |J+ - J-| {s['p95_abs_d_ret']:.2e}"
      )
      continue
    med_t = "-" if s["median_trust"] is None else f"{s['median_trust']:.3f}"
    med_ts = "-" if s["median_trust_soft"] is None else f"{s['median_trust_soft']:.3f}"
    print(
      f"\n{source} states (n={actual_starts}, filtered_end={filtered_count}): "
      f"median trust {med_t} (soft {med_ts}), "
      f"sign {s['sign_agrees']:.3f}, "
      f"G_med={s['median_norm']:.3e}, top0.5%_sq={s['top_0p5_sq_share']:.3f}, "
      f"median |J+ - J-| {s['median_abs_d_ret']:.2e}"
    )
    for band, row in s["bands"].items():
      t = "-" if row["trust"] is None else f"{row['trust']:.3f}"
      ts = "-" if row["trust_soft"] is None else f"{row['trust_soft']:.3f}"
      sg = "-" if row["sign_agrees"] is None else f"{row['sign_agrees']:.2f}"
      fb = "-" if row["fell_base_frac"] is None else f"{row['fell_base_frac']:.2f}"
      extra = ""
      if "model_fd_trust" in row:
        mfd = "-" if row["model_fd_trust"] is None else f"{row['model_fd_trust']:.3f}"
        extra = f"  model_fd={mfd}"
      elif row["replay_real_frac"] is not None:
        extra = f"  real_frac={row['replay_real_frac']:.2f}"
      print(
        f"  {band:>8} medians  n={row['n']:5d}  trust={t}  soft={ts}  "
        f"sign={sg}  fell={fb}{extra}"
      )

  if cfg.eps != 0.0:
    p = results["paired"]
    print(
      f"\npaired (model bands): median ||g_sim||/||g_model||={p['median_norm_ratio']:.3f}, "
      f"median cos={p['median_cos']:.3f}"
    )
    for band, row in p["bands"].items():
      mt = "-" if row["model_trust"] is None else f"{row['model_trust']:.3f}"
      st = "-" if row["sim_trust"] is None else f"{row['sim_trust']:.3f}"
      nr = "-" if row["median_norm_ratio"] is None else f"{row['median_norm_ratio']:.3f}"
      cs = "-" if row["median_cos"] is None else f"{row['median_cos']:.3f}"
      print(
        f"  {band:>8} medians  n={row['n']:5d}  model={mt}  sim={st}  "
        f"ratio={nr}  cos={cs}"
      )

  if cfg.output_file:
    out = Path(cfg.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    raw_path = out.with_suffix(".pt")
    torch.save(per_source, raw_path)
    print(f"[INFO] Wrote {cfg.output_file} and {raw_path}")
  env.close()
  return results


def main() -> None:
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  grad_trust(
    chosen_task,
    tyro.cli(GradTrustConfig, args=remaining_args, config=mjlab.TYRO_FLAGS),
  )


if __name__ == "__main__":
  main()
