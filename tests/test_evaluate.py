from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from supermimic.scripts.evaluate import act, normalized_return, reward_ceiling


def test_reward_ceiling_counts_positive_weights_only():
  rewards = {
    k: SimpleNamespace(weight=w) for k, w in {"a": 1.0, "b": 0.5, "c": -2.0}.items()
  }
  assert reward_ceiling(0.02, rewards) == pytest.approx(0.03)


def test_normalized_return():
  # A 501-frame clip is 500 env steps (reset already advances one frame).
  assert normalized_return(25.0, 0.1, 501) == pytest.approx(0.5)


def test_act_is_noise_free_by_default_even_in_proprio_mode():
  rep = MagicMock()
  rep.represent.return_value = torch.zeros(2, 4)
  rep.extract_aux_state.return_value = torch.zeros(2, 7)
  rep.compute_actor_sensor.return_value = torch.zeros(2, 4)
  wm = SimpleNamespace(
    representation=rep, sim_context=SimpleNamespace(read=lambda _: torch.zeros(2, 1))
  )
  cmd = SimpleNamespace(time_steps=torch.zeros(2, dtype=torch.long), motion=None)
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda _: cmd)
    ),
    get_observations=lambda: {
      "full_state": torch.zeros(2, 8),
      "actor": torch.zeros(2, 4),
    },
  )
  ts = SimpleNamespace(
    name="motion", extract_training_goal=lambda *_: torch.zeros(2, 3)
  )
  actor = MagicMock(return_value=torch.zeros(2, 2))
  actor.task_specs = [ts]

  act(env, wm, actor, proprio=True)
  assert rep.compute_actor_sensor.call_args.kwargs["noisy"] is False
  act(env, wm, actor, proprio=True, noisy=True)
  assert rep.compute_actor_sensor.call_args.kwargs["noisy"] is True
