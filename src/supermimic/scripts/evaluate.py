"""Score a SuperMimic checkpoint: one whole-clip episode per clip, normalized."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.wandb_artifacts import download_artifact_cached


@dataclass
class EvaluateConfig:
  wandb_path: str
  motion_file: str | None = None
  checkpoint: str = "latest"
  """Model artifact alias or version (``latest``, ``v3``)."""
  output_file: str | None = None
  device: str | None = None
  seed: int = 0


def reward_ceiling(step_dt: float, rewards: Any) -> float:
  """Per-step reward ceiling; same formula as ``WorldModelEnv.max_step_reward``."""
  return step_dt * sum(t.weight for t in rewards.values() if t.weight > 0.0)


def normalized_return(reward_sum: float, max_step: float, num_steps: int) -> float:
  return reward_sum / (max_step * num_steps)


def evaluate(task_id: str, cfg: EvaluateConfig) -> dict:
  device = setup_device(cfg.device)
  env_cfg = load_env_cfg(task_id, play=False)

  import wandb

  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  api = wandb.Api()
  run = api.run(cfg.wandb_path)
  runner_cfg = load_runner_cfg(run.config)
  check_representation_matches(task_id, runner_cfg.representation)
  if runner_cfg.history_length != 1:
    raise ValueError(
      f"history_length={runner_cfg.history_length} is not supported by evaluate.py "
      "(only history_length == 1)"
    )

  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  if cfg.motion_file is not None:
    expanded = os.path.expanduser(cfg.motion_file)
    if any(c in expanded for c in "*?[]"):
      files = sorted(glob.glob(expanded))
    else:
      files = [expanded]
    if not files:
      raise ValueError(f"No motion files matched: {cfg.motion_file}")
  else:
    artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
    if not artifacts:
      raise ValueError("No --motion-file given and the run has no motions artifact")
    files = [str(download_artifact_cached(a) / "motion.npz") for a in artifacts]
  command_term.motion_file = files
  num_clips = len(files)
  print(f"[INFO] Evaluating {num_clips} clip(s)")

  # Whole-clip episodes from frame 0, no reset randomization.
  env_cfg.episode_length_s = int(1e9)
  command_term.pose_range = {}
  command_term.velocity_range = {}
  command_term.joint_position_range = (0.0, 0.0)
  env_cfg.seed = cfg.seed
  command_term.sampling_mode = "external"

  env = create_env(env_cfg, num_clips, device)
  cmd = env.unwrapped.command_manager.get_term("motion")
  motion = cmd.motion
  ends = motion.motion_ends.to(device)
  if len(ends) != num_clips:
    raise RuntimeError(f"expected {num_clips} clips, motion loader has {len(ends)}")
  starts = torch.cat([ends.new_zeros(1), ends[:-1]])
  # ponytail: overrides a private MotionCommand method; breaks if mjlab renames it.
  cmd._external_sampling = lambda env_ids: starts[env_ids]

  wm = build_world_model(runner_cfg, env, device)
  representation = wm.representation

  artifact = api.artifact(f"{run.entity}/{run.project}/model-{run.id}:{cfg.checkpoint}")
  print(f"[INFO] Checkpoint artifact: {artifact.name} (version {artifact.version})")
  model_dir = download_artifact_cached(artifact)
  wm.load_bundle(torch.load(model_dir / "world_model.pt", map_location=device))
  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location=device))
  actor.eval()
  active_specs = [
    ts for ts in actor.task_specs if runner_cfg.slot_mix.get(ts.name, 0.0) > 0.0
  ]
  proprio = runner_cfg.actor_obs == "proprio"

  env.reset()
  base = env.unwrapped
  alive = torch.ones(num_clips, dtype=torch.bool, device=device)
  fell = torch.zeros(num_clips, dtype=torch.bool, device=device)
  reward_sum = torch.zeros(num_clips, device=device)
  with torch.inference_mode():
    while alive.any():
      obs = env.get_observations()
      full_state = obs["full_state"].to(device)
      motion_times = cmd.time_steps.to(device).clone()
      rep_state = representation.represent(full_state)
      aux_state = representation.extract_aux_state(full_state)
      sim_ctx = wm.sim_context.read(base)
      sensor = representation.compute_actor_sensor(
        rep_state,
        sim_ctx,
        aux_state=aux_state,
        sim_actor_obs=obs["actor"].to(device) if proprio else None,
        noisy=proprio,
      )
      goals = {
        ts.name: ts.extract_training_goal(motion, motion_times, aux_state)
        for ts in active_specs
      }
      actions = actor(build_actor_obs_td(goals, sensor.unsqueeze(1)))

      _, rewards, dones, _ = env.step(actions)
      tm = base.termination_manager
      failed = (tm.terminated & ~tm.time_outs).to(device)
      reward_sum += torch.where(alive, rewards.to(device), 0.0)
      ended = alive & dones.to(device).bool()
      fell |= ended & failed
      alive &= ~ended

  max_step = reward_ceiling(base.step_dt, base.cfg.rewards)
  rows = []
  for i in range(num_clips):
    frames = int((ends[i] - starts[i]).item())
    # clip_len steps: terminations and rewards run before the command advances
    # time_steps, so motion_end fires on step clip_len.
    rows.append(
      {
        "index": i,
        "clip": motion.motion_names[i],
        "frames": frames,
        "normalized_return": normalized_return(float(reward_sum[i]), max_step, frames),
        "fell": bool(fell[i]),
      }
    )
    r = rows[-1]
    print(
      f"{i:4d}  {r['clip']}  frames={frames}  "
      f"return={r['normalized_return']:.4f}  fell={r['fell']}"
    )
  mean = sum(r["normalized_return"] for r in rows) / num_clips
  failed_clips = sum(r["fell"] for r in rows)
  print(f"mean normalized return: {mean:.4f}")
  print(f"failed clips: {failed_clips} / {num_clips}")

  results = {
    "run": cfg.wandb_path,
    "seed": cfg.seed,
    "checkpoint": artifact.version,
    "clips": rows,
    "mean_normalized_return": mean,
    "failed_clips": failed_clips,
  }
  if cfg.output_file:
    Path(cfg.output_file).write_text(json.dumps(results, indent=2))
    print(f"[INFO] Wrote {cfg.output_file}")
  env.close()
  return results


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(EvaluateConfig, args=remaining_args, config=mjlab.TYRO_FLAGS)
  evaluate(chosen_task, args)


if __name__ == "__main__":
  main()
