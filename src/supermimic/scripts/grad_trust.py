"""Gradient trust: does a policy gradient predict the real change in return?

From N simulator start states of a checkpoint's own play, take the action
gradient of the learner's H-step return two ways: imagined with model states
(SuperMimic) and replayed through the simulator's next H states (DMO). Check
each in the simulator: play the policy from the start with actions shifted by
+delta and -delta along the gradient, the largest element of delta being eps.

  trust = (J+ - J-) / (2 * g . delta)

J is the real H-step return, discounted as the learner discounts, plus the
learner's critic bootstrap at the real final state. Trust near 1 means the
gradient predicts the simulator; the paper (arXiv 2611.27648, Table I) reports
the median trust by gradient-norm band in batch medians.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mjlab
import mjlab.tasks  # noqa: F401
import torch
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
  """Start states; the script runs 3 envs per start."""
  warmup_steps: int = 100
  """Policy steps from random resets before the starts are taken."""
  eps: float = 0.05
  """Largest element of the action shift (the paper's setting)."""
  output_file: str | None = None
  device: str | None = None
  seed: int = 0


def summarize(norm: torch.Tensor, trust: torch.Tensor, value: torch.Tensor) -> dict:
  """Median trust per norm band (in batch medians) and per critic-value tercile."""
  rel = norm / norm.median()
  bands = {}
  for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True):
    m = (rel >= lo) & (rel < hi)
    bands[f"{lo:g}-{hi:g}"] = {
      "n": int(m.sum()),
      "trust": float(trust[m].median()) if m.any() else None,
    }
  cuts = value.quantile(torch.tensor([1 / 3, 2 / 3], device=value.device))
  tercile = torch.bucketize(value, cuts)
  terciles = {
    name: float(trust[tercile == i].median()) if (tercile == i).any() else None
    for i, name in enumerate(("low", "mid", "high"))
  }
  return {"median_trust": float(trust.median()), "bands": bands, "value": terciles}


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
  trust, critic value and ``d_ret = J+ - J-``. The copies sit at different
  world positions, so float32 rounding alone makes ``d_ret`` about 1e-5 at
  ``eps = 0``; real shifts must move the return well above that. ``env`` has ``3 * num_starts`` envs: the starts,
  then their +delta and -delta copies."""
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
    learner = ActorCriticPolicyLearner(actor, critic, wm_env, policy_cfg, dev)
  else:
    learner = BpttPolicyLearner(actor, wm_env, policy_cfg, dev)
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

  def play(snap, grads) -> tuple[torch.Tensor, TensorDict]:
    """Restore every env to its start, play H steps with the copies shifted
    along ``grads``; return the real returns and the starts' unshifted window."""
    restore(env.unwrapped, {k: v.repeat(3, *[1] * (v.dim() - 1)) for k, v in snap.items()},
            everyone)  # fmt: skip
    scale = eps / torch.stack(grads).abs().amax(dim=(0, 2)).clamp_min(1e-12)
    ret = torch.zeros(3 * n, device=dev)
    alive = torch.ones(3 * n, dtype=torch.bool, device=dev)
    frames = []
    with torch.no_grad():
      for t in range(horizon):
        actions, full_state, motion_times, sim_ctx = act(env, wm, actor, proprio)
        delta = scale[:, None] * grads[t]
        actions[plus] += delta
        actions[minus] -= delta
        f = frame(base, full_state, motion_times, sim_ctx, actions)
        _, reward, dones, extras = env.step(actions)
        truncated = extras["time_outs"].bool()
        f["terminated"], f["truncated"] = (
          dones.bool()[base] & ~truncated[base],
          truncated[base],
        )
        frames.append(f)
        ret += gamma**t * alive * reward
        alive &= ~dones.bool()
      _, full_state, motion_times, sim_ctx = act(env, wm, actor, proprio)
      final = frame(everyone, full_state, motion_times, sim_ctx)
      ret += learner._bootstrap(wm_env.reset(final), alive.float(), 1.0 - alive.float())
      frames.append(final[base])
    return ret, torch.stack(frames, dim=1)

  env.reset()
  with torch.no_grad():
    for _ in range(warmup_steps):
      env.step(act(env, wm, actor, proprio)[0])
    _, full_state, motion_times, sim_ctx = act(env, wm, actor, proprio)
    start = frame(base, full_state, motion_times, sim_ctx)
    snap = snapshot(env.unwrapped, base)
    start_value = value(start)

  out = {}
  grads = action_gradient(start)
  ret, window = play(snap, grads)
  for source in ("model", "sim"):
    if source == "sim":
      grads = action_gradient(replay_start(window))
      ret, _ = play(snap, grads)
    g = torch.stack(grads)  # (H, N, A)
    norm = g.pow(2).sum(dim=(0, 2)).sqrt()
    scale = eps / g.abs().amax(dim=(0, 2)).clamp_min(1e-12)
    predicted = scale * norm.pow(2)  # g . delta
    d_ret = ret[plus] - ret[minus]
    trust = d_ret / (2 * predicted).clamp_min(1e-12)
    out[source] = {"norm": norm, "trust": trust, "value": start_value, "d_ret": d_ret}
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
    command.motion_file = [
      str(download_artifact_cached(a) / "motion.npz") for a in artifacts
    ]
  command.sampling_mode = "uniform"
  env_cfg.episode_length_s = int(1e9)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = 3 * cfg.num_starts
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  wm = build_world_model(runner_cfg, env, device)
  actor, model_dir, artifact = load_checkpoint(
    api, run, cfg.checkpoint, wm, runner_cfg, device
  )
  critic = build_critic(runner_cfg, wm, device)
  critic.load_state_dict(torch.load(model_dir / "critic.pt", map_location=device))

  per_source = trust_at_starts(
    env, wm, actor, critic, runner_cfg, cfg.num_starts, cfg.warmup_steps, cfg.eps
  )
  results = {
    "run": cfg.wandb_path,
    "checkpoint": artifact.version,
    "num_starts": cfg.num_starts,
    "eps": cfg.eps,
    **{s: summarize(r["norm"], r["trust"], r["value"]) for s, r in per_source.items()},
  }
  for source, r in per_source.items():
    # With --eps 0 this is the noise floor between copies; see trust_at_starts.
    results[source]["median_abs_d_ret"] = float(r["d_ret"].abs().median())
    print(
      f"\n{source} states: median trust {results[source]['median_trust']:.3f}, "
      f"median |J+ - J-| {results[source]['median_abs_d_ret']:.2e}"
    )
    for band, row in results[source]["bands"].items():
      trust = "-" if row["trust"] is None else f"{row['trust']:.3f}"
      print(f"  {band:>8} medians  n={row['n']:5d}  trust={trust}")
  if cfg.output_file:
    out = Path(cfg.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[INFO] Wrote {cfg.output_file}")
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
