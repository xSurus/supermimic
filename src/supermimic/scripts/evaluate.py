"""Score a SuperMimic checkpoint: one whole-clip episode per clip, normalized."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import resolve_motion_file
from mjlab.utils.wrappers import VideoRecorder

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train_utils import setup_device
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
  video: bool = False
  """Save an MP4 of the first clip to ``videos/``. Pass one clip in
  ``--motion-file`` to choose which. On Colab set ``MUJOCO_GL=egl``."""
  randomize: bool = False
  """Keep the training domain randomization. Off by default: the paper evaluates
  with foot friction 1.0, no CoM offset and no encoder bias (appendix F)."""
  video_length: int = 1000
  """Frames to record (50 per second). Frames are held in memory until saved."""


def reward_ceiling(step_dt: float, rewards: Any) -> float:
  """Per-step reward ceiling; same formula as ``WorldModelEnv.max_step_reward``."""
  return step_dt * sum(t.weight for t in rewards.values() if t.weight > 0.0)


def normalized_return(reward_sum: float, max_step: float, clip_len: int) -> float:
  return reward_sum / (max_step * (clip_len - 1))


def pin_randomization(env_cfg: Any) -> None:
  """Nominal foot friction, no CoM offset, no encoder bias (the paper's
  appendix F). Pinned, not removed: removing an event would change the sim
  context layout the world model was trained on."""
  events = env_cfg.events
  events["foot_friction"].params["ranges"] = (1.0, 1.0)
  events["base_com"].params["ranges"] = {i: (0.0, 0.0) for i in range(3)}
  events["encoder_bias"].params["bias_range"] = (0.0, 0.0)


def load_run(task_id: str, wandb_path: str) -> tuple[Any, Any, Any]:
  """``(api, run, runner_cfg)`` of a W&B training run (history_length 1 only)."""
  import wandb

  print(f"[INFO] Connecting to WandB run: {wandb_path}")
  api = wandb.Api()
  run = api.run(wandb_path)
  runner_cfg = load_runner_cfg(run.config)
  check_representation_matches(task_id, runner_cfg.representation)
  if runner_cfg.history_length != 1:
    raise ValueError(
      f"history_length={runner_cfg.history_length} is not supported "
      "(only history_length == 1)"
    )
  return api, run, runner_cfg


def load_checkpoint(api, run, checkpoint: str, wm, runner_cfg, device: str):
  """Load ``world_model.pt`` into ``wm`` and return ``(actor, model_dir,
  artifact)`` with the actor in eval mode on the motion slot only."""
  artifact = api.artifact(f"{run.entity}/{run.project}/model-{run.id}:{checkpoint}")
  print(f"[INFO] Checkpoint artifact: {artifact.name} (version {artifact.version})")
  model_dir = download_artifact_cached(artifact)
  wm.load_bundle(torch.load(model_dir / "world_model.pt", map_location=device))
  actor = build_actor(runner_cfg, wm, device)
  actor.set_slot_mix({"motion": 1.0})
  actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location=device))
  actor.eval()
  return actor, model_dir, artifact


def act(env, wm, actor, proprio: bool):
  """One deterministic policy step's input and output, as the runner's
  collection step builds them: ``(actions, full_state, motion_times, sim_ctx)``."""
  representation = wm.representation
  cmd = env.unwrapped.command_manager.get_term("motion")
  obs = env.get_observations()  # the env and the models share one device
  full_state = obs["full_state"]
  motion_times = cmd.time_steps.clone()
  rep_state = representation.represent(full_state)
  aux_state = representation.extract_aux_state(full_state)
  sim_ctx = wm.sim_context.read(env.unwrapped)
  sensor = representation.compute_actor_sensor(
    rep_state,
    sim_ctx,
    aux_state=aux_state,
    sim_actor_obs=obs["actor"] if proprio else None,
    noisy=proprio,
  )
  goals = {
    ts.name: ts.extract_training_goal(cmd.motion, motion_times, aux_state)
    for ts in actor.task_specs
    if ts.name == "motion"
  }
  actions = actor(build_actor_obs_td(goals, sensor.unsqueeze(1)))
  return actions, full_state, motion_times, sim_ctx


def evaluate(task_id: str, cfg: EvaluateConfig) -> dict:
  device = setup_device(cfg.device)
  env_cfg = load_env_cfg(task_id, play=False)
  if not cfg.randomize:
    pin_randomization(env_cfg)
  api, run, runner_cfg = load_run(task_id, cfg.wandb_path)

  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  if cfg.motion_file is not None:
    files = resolve_motion_file(cfg.motion_file)
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

  env_cfg.scene.num_envs = num_clips
  base = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if cfg.video else None
  )
  if cfg.video:
    base = VideoRecorder(
      base,
      video_folder="videos",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print(f"[INFO] Recording {cfg.video_length} frames of clip 0 to videos/")
  env = RslRlVecEnvWrapper(base)
  cmd = env.unwrapped.command_manager.get_term("motion")
  motion = cmd.motion
  ends = motion.motion_ends.to(device)
  if len(ends) != num_clips:
    raise RuntimeError(f"expected {num_clips} clips, motion loader has {len(ends)}")
  starts = torch.cat([ends.new_zeros(1), ends[:-1]])

  # ponytail: overrides a private MotionCommand method; breaks if mjlab renames it.
  def start_of_own_clip(env_ids: torch.Tensor, probs: torch.Tensor) -> None:
    del probs  # external mode's bin distribution, unused here
    cmd.time_steps[env_ids] = starts[env_ids]

  cmd._sample_bins = start_of_own_clip

  wm = build_world_model(runner_cfg, env, device)
  actor, _, artifact = load_checkpoint(api, run, cfg.checkpoint, wm, runner_cfg, device)
  proprio = runner_cfg.actor_obs == "proprio"

  env.reset()
  base = env.unwrapped
  alive = torch.ones(num_clips, dtype=torch.bool, device=device)
  fell = torch.zeros(num_clips, dtype=torch.bool, device=device)
  reward_sum = torch.zeros(num_clips, device=device)
  steps = torch.zeros(num_clips, dtype=torch.long, device=device)
  with torch.inference_mode():
    while alive.any():
      actions = act(env, wm, actor, proprio)[0]
      _, rewards, dones, _ = env.step(actions)
      tm = base.termination_manager
      failed = (tm.terminated & ~tm.time_outs).to(device)
      reward_sum += torch.where(alive, rewards.to(device), 0.0)
      steps += alive
      ended = alive & dones.to(device).bool()
      fell |= ended & failed
      alive &= ~ended

  max_step = reward_ceiling(base.step_dt, base.cfg.rewards)
  rows = []
  for i in range(num_clips):
    frames = int((ends[i] - starts[i]).item())
    # frames - 1 steps: env.reset() already advances time_steps by one, so an
    # episode starts at start + 1 and motion_end fires after frames - 1 steps.
    if not fell[i] and int(steps[i]) != frames - 1:
      print(f"[WARN] clip {i}: {int(steps[i])} steps, expected {frames - 1}")
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
    out = Path(cfg.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
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
