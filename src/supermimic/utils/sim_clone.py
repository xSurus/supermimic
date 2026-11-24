"""Copy one env's simulator state into others, mid-episode."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

_DATA_FIELDS = ("qpos", "qvel", "act", "qacc_warmstart")


def copy_env_state(
  env: ManagerBasedRlEnv, src: torch.Tensor, dst: torch.Tensor
) -> None:
  """Make envs ``dst`` continue exactly as envs ``src`` would (``src[i]`` into
  ``dst[i]``): physics state, the action history the rewards read, the motion
  clock and the episode length. The root moves by the difference of the two
  env origins, since each env's reference is offset by its own origin.

  Per-env model fields (domain randomization: friction, CoM, encoder bias) are
  not copied, so pin them to nominal values first, as ``evaluate.py`` does.
  The task has no actuator delay or observation history to copy."""
  data = env.sim.data
  for field in _DATA_FIELDS:
    getattr(data, field)[dst] = getattr(data, field)[src]
  # ponytail: assumes the robot's free joint is the first qpos entry (the only
  # jointed entity in the tracking scene); use the entity's qpos address if not.
  origins = env.scene.env_origins
  data.qpos[dst, :3] += origins[dst] - origins[src]
  am = env.action_manager
  for buf in (am._action, am._prev_action, am._prev_prev_action):
    buf[dst] = buf[src]
  cmd = env.command_manager.get_term("motion")
  cmd.set_time_steps(dst, cmd.time_steps[src])
  env.episode_length_buf[dst] = env.episode_length_buf[src]
  env.sim.forward()
  cmd.update_relative_body_poses()
