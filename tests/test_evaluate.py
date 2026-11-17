from types import SimpleNamespace

import pytest

from supermimic.scripts.evaluate import normalized_return, reward_ceiling


def test_reward_ceiling_counts_positive_weights_only():
  rewards = {
    k: SimpleNamespace(weight=w) for k, w in {"a": 1.0, "b": 0.5, "c": -2.0}.items()
  }
  assert reward_ceiling(0.02, rewards) == pytest.approx(0.03)


def test_normalized_return():
  # A 501-frame clip is 501 env steps in mjlab (motion_end fires after len steps).
  assert normalized_return(25.05, 0.1, 501) == pytest.approx(0.5)
