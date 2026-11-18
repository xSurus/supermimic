"""DMO arm: rollouts replay the newest simulator windows with model Jacobians."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.tasks.task_spec import TaskSpec
from tests.conftest import (
  ACTION_DIM,
  MOTION_GOAL_DIM,
  make_mock_command_env,
  make_mock_wm_representation,
)

E, H, FULL, REP = 4, 3, 182, 214


def _buffer(steps: int, done_env0_at: int | None = None) -> ReplayBuffer:
  """Frame k of every env holds stamp k: state, action and clip time all equal k.
  Env 0 ends an episode at frame ``done_env0_at``."""
  schema = {
    "state": FieldSpec(FULL),
    "action": FieldSpec(ACTION_DIM),
    "prev_action": FieldSpec(ACTION_DIM),
    "motion_time": FieldSpec(dtype=torch.long),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
    "sim_ctx": FieldSpec(1),
  }
  buf = ReplayBuffer(schema, buffer_size=5, num_envs=E)
  for k in range(steps):
    done = torch.zeros(E, dtype=torch.bool)
    if k == done_env0_at:
      done[0] = True
    buf.insert(
      TensorDict(
        {
          "state": torch.full((E, FULL), float(k)),
          "action": torch.full((E, ACTION_DIM), float(k)),
          "prev_action": torch.zeros(E, ACTION_DIM),
          "motion_time": torch.full((E,), k, dtype=torch.long),
          "terminated": done,
          "truncated": torch.zeros(E, dtype=torch.bool),
          "sim_ctx": torch.zeros(E, 1),
        },
        batch_size=[E],
      )
    )
  return buf


def test_sample_newest_returns_the_newest_frames_in_order_after_wrap():
  window = _buffer(steps=8).sample_newest(4, E)  # buffer holds stamps 3..7
  assert (window["state"][..., 0] == torch.tensor([4.0, 5.0, 6.0, 7.0])).all()


def _env(buf: ReplayBuffer, weight: torch.Tensor) -> MultiTaskWorldModelEnv:
  rep = make_mock_wm_representation(E)
  # The rep state and aux state are the stored stamp, so swaps are visible.
  rep.represent.side_effect = lambda x: x[..., :1].expand(*x.shape[:-1], REP).clone()
  rep.extract_aux_state.side_effect = lambda x: x[..., :7].clone()
  # Reward = mean rep state, so the gradient flows only through the dynamics.
  rep.compute_wm_reward_from_reference.side_effect = lambda rs, *a, **kw: rs.mean(-1)

  def dynamics(rep_state, action, aux_state, sim_ctx):
    del sim_ctx
    pred = -100.0 + weight * action[:, :1]  # a model that is wrong in value
    return aux_state * 0 + pred, rep_state * 0 + pred

  goal = torch.zeros(E, MOTION_GOAL_DIM)
  spec = TaskSpec("motion", MOTION_GOAL_DIM, lambda m, mt, aux: goal)
  return MultiTaskWorldModelEnv(
    dynamics=dynamics,
    representation=rep,
    env=make_mock_command_env(),
    batch_size=E,
    replay_buffer=buf,
    task_specs=[spec],
    history_length=1,
    termination_softness=0.05,
    device="cpu",
    replay_horizon=H,
  )


def test_replay_takes_values_from_the_sim_and_slope_from_the_model():
  # Stamps 1..4 are the newest 4 frames. Env 0's episode ends at stamp 2, so
  # the stored stamp 3 belongs to the next episode: env 0 steps 1 -> 2 on the
  # sim, predicts 2 -> 3 from logged action 2, then imagines with the policy.
  buf = _buffer(steps=5, done_env0_at=2)
  weight = torch.tensor(2.0, requires_grad=True)
  env = _env(buf, weight)
  start = env.sample_initial_states()
  real = start["real"]
  env.reset(start)
  total = 0.0
  for t in range(H):
    policy_action = torch.zeros(E, ACTION_DIM, requires_grad=True)
    _, reward, _, info = env.step(policy_action, real[:, t])
    total = total + reward.sum()
    assert info["replay_time_mismatch"] == 0.0
  env0 = (real["next_is_real"][:, -1] == 0).nonzero().item()
  others = [i for i in range(E) if i != env0]
  assert (real["next_is_real"][env0] == torch.tensor([True, False, False])).all()
  assert (real["is_real"][env0] == torch.tensor([True, True, False])).all()
  # Real rows end on the stored stamp 4; env 0 ends on the policy's action 0.
  assert (env.rep_state[others, 0] == 4.0).all()
  assert env.rep_state[env0, 0] == -100.0
  total.backward()
  # Each reward's slope in the model weight is the action it was fed: logged
  # stamps 1, 2, 3 for the 3 real envs, logged 1, 2 then 0 for env 0.
  assert weight.grad.item() == pytest.approx(3 * (1 + 2 + 3) + (1 + 2 + 0))


def test_model_fresh_starts_at_the_window_start_without_replay():
  env = _env(_buffer(steps=5), torch.tensor(1.0))
  env.replay_states = False
  start = env.sample_initial_states()
  assert "real" not in start.keys()
  # Same start frame as the DMO arm: the oldest of the newest H + 1 frames.
  assert (start["state"][:, 0] == 5 - (H + 1)).all()


def test_on_policy_blocks_replay_only_actions_taken_since_the_last_block():
  from supermimic.runners.supermimic_runner import on_policy_block

  for horizon in (3, 32):
    last = -1
    for step in range(10 * (horizon + 1)):
      if on_policy_block(step, horizon):
        # The newest horizon + 1 frames end at `step`; their transitions use the
        # actions taken at steps step - horizon .. step - 1.
        assert step - horizon == last + 1
        last = step
    assert last > 0
