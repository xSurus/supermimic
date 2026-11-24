"""Differentiable world model environment for motion tracking.

Replaces the physics simulator with a learned ``Dynamics`` + ``StateRepresentation``
while a reference motion clip supplies per-frame goals. The env holds state only
after ``reset()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from supermimic.tasks.task_spec import TaskSpec, build_actor_obs_td

if TYPE_CHECKING:
  from mjlab.rl import RslRlVecEnvWrapper

  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import StateRepresentation


class WorldModelEnv:
  """Differentiable world model env for motion tracking.

  Observations hold the sensor ``history`` (B, history_length, sensor_dim), the
  privileged ``critic`` input and one ``goal_<name>`` per task. ``task_specs[0]``
  is the motion task, whose goal is the actor half of the motion target.
  """

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: RslRlVecEnvWrapper,
    task_specs: list[TaskSpec],
    history_length: int,
    termination_softness: float,
  ) -> None:
    assert task_specs[0].name == "motion", "task_specs[0] must be the motion task"
    self.dynamics = dynamics
    self.representation = representation
    self.task_specs = task_specs
    self.history_length = history_length

    # Analytic per-step reward ceiling: positive tracking-reward weights, dt-scaled.
    self.max_step_reward = env.unwrapped.step_dt * sum(
      term.weight for term in env.cfg.rewards.values() if term.weight > 0.0
    )

    # Reference-motion buffers.
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    self.motion = motion_cmd.motion
    self._anchor_index = motion_cmd.motion_anchor_body_index

    if termination_softness <= 0.0:
      raise ValueError(
        f"termination_softness must be positive, got {termination_softness}"
      )
    terms = env.cfg.terminations
    self._termination_softness = termination_softness
    self._anchor_pos_threshold = terms["anchor_pos"].params["threshold"]
    self._anchor_ori_threshold = terms["anchor_ori"].params["threshold"]

  def reset(self, batch: TensorDict) -> TensorDict:
    """Reset state, goals and sensor history to ``batch``."""
    full_states = batch["state"]
    self.aux_state = self.representation.extract_aux_state(full_states).detach()
    self.rep_state = self.representation.represent(full_states).detach()
    self.joint_state = self.representation.joint_state(self.rep_state)
    # Raw policy outputs (pre-scale/offset). Dynamics handles the transform;
    # the reward path consumes these directly.
    self.last_actions = batch["prev_action"].detach()
    self.sim_ctx = batch["sim_ctx"]
    self.motion_times = batch["motion_time"]
    self._update_goal()
    if self.history_length > 1:
      self._past_sensors = self._warmstart_past_sensors(batch)
    self._latest_sensor = self._synth_sensor()
    return self.get_observations()

  def step(
    self, actions: torch.Tensor, real: TensorDict | None = None
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
    """Advance one step. Termination and reward score the post-step state against
    the pre-step reference frame, as the simulator does before advancing the clip.

    ``real`` is one step of a replayed simulator window (``sample_replay_window``,
    the DMO arm). Where the current state is real, the logged action replaces the
    policy's value; where the next state is real, it replaces the model's value.
    Both keep the gradient of what they replace, so the slope is the model's and
    the number is the simulator's. After the window's episode ends the rollout
    continues in imagination."""
    if real is not None:
      actions = _swap_value(actions, real["action"], real["is_real"])
    self.aux_state, self.rep_state = self.dynamics(
      self.rep_state, actions, self.aux_state, self.sim_ctx
    )
    if real is not None:
      sim_state, swap = real["state"], real["next_is_real"]
      self.rep_state = _swap_value(
        self.rep_state, self.representation.represent(sim_state), swap
      )
      self.aux_state = _swap_value(
        self.aux_state, self.representation.extract_aux_state(sim_state), swap
      )
    self.joint_state = self.representation.joint_state(self.rep_state)
    term_prob = self._compute_term_prob(self.aux_state, *self._ref_body_state[:2])
    rewards = self.representation.compute_wm_reward_from_reference(
      self.rep_state,
      self.aux_state,
      self.joint_state,
      actions,
      self.last_actions,
      *self._ref_body_state,
    )
    self.last_actions = actions

    self._compute_commands()

    if self.history_length > 1:
      self._past_sensors = torch.cat(
        [self._past_sensors[:, 1:], self._latest_sensor.unsqueeze(1)], dim=1
      )
    self._latest_sensor = self._synth_sensor()

    return self.get_observations(), rewards, term_prob

  def _synth_sensor(self) -> torch.Tensor:
    """Synthesize the actor's sensor from the current WM state."""
    return self.representation.compute_actor_sensor(
      self.rep_state,
      self.sim_ctx,
      aux_state=self.aux_state,
      joint_state=self.joint_state,
      last_action=self.last_actions,
      noisy=True,
    )

  def get_observations(self) -> TensorDict:
    history = self._latest_sensor.unsqueeze(1)
    if self.history_length > 1:
      history = torch.cat([self._past_sensors, history], dim=1)
    critic = torch.cat([self.rep_state, self._wm_goal_obs, self.sim_ctx], dim=-1)
    return build_actor_obs_td(self._task_goal_obs, history, critic)

  def _compute_term_prob(
    self,
    aux_state: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
  ) -> torch.Tensor:
    """Soft termination from the real threshold rules."""
    anchor_pos_w = ref_body_pos_w[:, self._anchor_index]
    anchor_quat_w = ref_body_quat_w[:, self._anchor_index]
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    height_error = (anchor_pos_w[..., 2] - root_pos[..., 2]).abs()
    # Gravity's z in a body frame is 2(x^2 + y^2) - 1 for a unit quaternion.
    ref_gravity_z = 2 * (anchor_quat_w[..., 1:3] ** 2).sum(-1) - 1.0
    root_gravity_z = 2 * (root_quat[..., 1:3] ** 2).sum(-1) - 1.0
    tilt_error = (ref_gravity_z - root_gravity_z).abs()
    normalized_error = torch.maximum(
      height_error / self._anchor_pos_threshold,
      tilt_error / self._anchor_ori_threshold,
    )
    return torch.sigmoid((normalized_error - 1.0) / self._termination_softness)

  def _compute_commands(self) -> None:
    """Advance the reference clip one frame (clamped at clip ends) and refresh."""
    next_motion_times = self.motion_times + 1
    # Membership in the sorted clip ends; torch.isin syncs the host on long lists.
    ends = self.motion.motion_ends
    pos = torch.searchsorted(ends, next_motion_times).clamp(max=ends.numel() - 1)
    at_end = ends[pos] == next_motion_times
    self.motion_times = torch.where(at_end, self.motion_times, next_motion_times)
    self._update_goal()

  def _update_goal(self) -> None:
    """Compute the reference body state at ``motion_times`` and rebuild goals."""
    self._ref_body_state = self.motion.body_state(self.motion_times)
    self._wm_goal_obs, actor_goal = self.representation.compute_wm_goal(
      self.motion, self.motion_times, self.aux_state
    )
    self._task_goal_obs = {"motion": actor_goal}
    for ts in self.task_specs[1:]:
      self._task_goal_obs[ts.name] = ts.extract_training_goal(
        self.motion, self.motion_times, self.aux_state
      )

  def _warmstart_past_sensors(self, batch: TensorDict) -> torch.Tensor:
    """Noise the sampled predecessor window; clamp-padded slots (before the
    episode start) share the oldest real frame's single noise realization, the
    way the real-sim ring's reset padding does."""
    noisy = self.representation.add_proprio_noise(batch["sensor_window"][:, :-1])
    pad = batch["pad_mask"][:, :-1]  # padding is a leading run (sample_causal)
    first_real = (~pad).int().argmax(dim=1)
    shared = noisy[torch.arange(len(noisy), device=noisy.device), first_real]
    return torch.where(pad.unsqueeze(-1), shared.unsqueeze(1), noisy)


def sample_start(
  buffer: ReplayBuffer, history_length: int, batch_size: int
) -> TensorDict:
  """Start states for :meth:`WorldModelEnv.reset`. With history_length > 1 each carries
  ``sensor_window``, the H frames ending at it, and ``pad_mask``, their padded slots."""
  if history_length == 1:
    return buffer.sample(1, batch_size).squeeze(1)
  window, padded = buffer.sample_causal(history_length, batch_size)
  return window[:, -1].set("sensor_window", window["actor_obs"]).set("pad_mask", padded)


def sample_replay_window(
  buffer: ReplayBuffer, horizon: int, batch_size: int, replay_states: bool = True
) -> TensorDict:
  """Start states for the DMO arms (history_length 1): frame 0 of each env's newest
  ``horizon + 1`` frames. With ``replay_states`` (DMO) it carries ``real``
  (B, horizon): step ``k``'s logged action, the stored next state and clip time,
  and two masks. ``is_real[k]``: state ``k`` is the simulator's.
  ``next_is_real[k]``: state ``k + 1`` is too, i.e. no episode ended at frames
  ``0..k``. The buffer stores the post-reset state after a done, so the terminal
  state itself is never available and the step that ends an episode is predicted.
  Without ``replay_states`` (model-fresh) the rollout from that start is imagined."""
  window = buffer.sample_newest(horizon + 1, batch_size)
  return replay_start(window) if replay_states else window[:, 0]


def replay_start(window: TensorDict) -> TensorDict:
  """The DMO start of a (B, horizon + 1) window of stored frames: frame 0 with
  ``real`` attached (see ``sample_replay_window``)."""
  batch_size, horizon = window.shape[0], window.shape[1] - 1
  start = window[:, 0].contiguous()
  ended = (window["terminated"] | window["truncated"])[:, :-1]
  next_is_real = ended.long().cumsum(1) == 0
  is_real = torch.cat([torch.ones_like(next_is_real[:, :1]), next_is_real[:, :-1]], 1)
  real = TensorDict(
    {
      "action": window["action"][:, :-1],
      "state": window["state"][:, 1:],
      "motion_time": window["motion_time"][:, 1:].contiguous(),
      "is_real": is_real,
      "next_is_real": next_is_real,
    },
    batch_size=[batch_size, horizon],
  )
  return start.set("real", real)


def _swap_value(
  x: torch.Tensor, value: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
  """Where ``mask`` (batch-shaped), take ``value``'s number and keep ``x``'s
  gradient (straight-through); elsewhere ``x`` unchanged."""
  swapped = x + (value - x).detach()
  return torch.where(mask.unsqueeze(-1), swapped, x)
