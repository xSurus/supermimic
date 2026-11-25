"""Save envs' simulator state mid-episode and put it back, into the same or other envs."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

_DATA_FIELDS = ("qpos", "qvel", "act", "qacc_warmstart")
_ACTION_BUFFERS = ("_action", "_prev_action", "_prev_prev_action")


def snapshot(env: ManagerBasedRlEnv, ids: torch.Tensor) -> dict[str, torch.Tensor]:
  """Everything envs ``ids`` need to continue as they would: physics state, the
  action history the rewards read, the motion clock and the episode length. The
  root position is stored relative to the env origin, since each env's
  reference is offset by its own origin.

  Per-env model fields (domain randomization: friction, CoM, encoder bias) are
  not saved, so pin them to nominal values first, as ``evaluate.py`` does. The
  task has no actuator delay or observation history to save."""
  data, am = env.sim.data, env.action_manager
  snap = {f: getattr(data, f)[ids].clone() for f in _DATA_FIELDS}
  # ponytail: assumes the robot's free joint is the first qpos entry (the only
  # jointed entity in the tracking scene); use the entity's qpos address if not.
  snap["qpos"][:, :3] -= env.scene.env_origins[ids]
  snap |= {b: getattr(am, b)[ids].clone() for b in _ACTION_BUFFERS}
  snap["time_steps"] = env.command_manager.get_term("motion").time_steps[ids].clone()
  snap["episode_length"] = env.episode_length_buf[ids].clone()
  return snap


def restore(
  env: ManagerBasedRlEnv, snap: dict[str, torch.Tensor], ids: torch.Tensor
) -> None:
  """Put ``snap`` (row ``i``) into env ``ids[i]``, which then continues as the
  saved env would have."""
  data, am = env.sim.data, env.action_manager
  for f in _DATA_FIELDS:
    getattr(data, f)[ids] = snap[f]
  data.qpos[ids, :3] += env.scene.env_origins[ids]
  for b in _ACTION_BUFFERS:
    getattr(am, b)[ids] = snap[b]
  cmd = env.command_manager.get_term("motion")
  cmd.set_time_steps(ids, snap["time_steps"])
  env.episode_length_buf[ids] = snap["episode_length"]
  env.sim.forward()
  env.sim.sense()
  cmd.update_relative_body_poses()
  # The observation manager returns its cached buffer unless asked to update.
  # The task keeps no observation history, so updating it is safe.
  env.obs_buf = env.observation_manager.compute(update_history=True)
