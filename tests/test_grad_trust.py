"""grad_trust: band summary, and the trust path on the CPU simulator."""

import sys
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_critic,
  build_world_model,
)
from supermimic.scripts.evaluate import pin_randomization
from supermimic.scripts.grad_trust import summarize, trust_at_starts

sys.path.insert(0, str(Path(__file__).parents[1] / "colab"))
from build_motions import convert  # noqa: E402


def test_summarize_bins_by_batch_median_norm():
  # Median norm 2: relative norms 0.5 x3, 1, 2.5, 6, 40.
  norm = torch.tensor([1.0, 1.0, 1.0, 2.0, 5.0, 12.0, 80.0])
  trust = torch.tensor([0.9, 0.9, 0.9, 0.8, 0.5, 0.3, 0.0])
  out = summarize(norm, trust, value=torch.arange(7.0))
  assert out["bands"]["0-1"] == {"n": 3, "trust": pytest.approx(0.9)}
  assert out["bands"]["1-3"] == {"n": 2, "trust": pytest.approx(0.5)}  # lower median
  assert out["bands"]["3-10"] == {"n": 1, "trust": pytest.approx(0.3)}
  assert out["bands"]["10-30"] == {"n": 0, "trust": None}
  assert out["bands"]["30-inf"] == {"n": 1, "trust": pytest.approx(0.0)}


@pytest.mark.parametrize("eps", [0.0, 0.05])
def test_trust_runs_and_unshifted_copies_agree(tmp_path, eps):
  csv = tmp_path / "stand.csv"
  np.savetxt(
    csv, np.tile([0, 0, 0.79, 0, 0, 0, 1] + [0.0] * 29, (600, 1)), delimiter=","
  )
  convert(csv, tmp_path / "stand.npz", "cpu")
  env_cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  pin_randomization(env_cfg)
  env_cfg.commands["motion"].motion_file = [str(tmp_path / "stand.npz")]
  env_cfg.commands["motion"].sampling_mode = "uniform"
  env_cfg.episode_length_s = int(1e9)
  env_cfg.seed = 0  # same reset frames every run
  env_cfg.scene.num_envs = 3 * 2
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cpu"))

  torch.manual_seed(0)
  cfg = SuperMimicRunnerCfg(
    dynamics_hidden_dims=(32,),
    actor_hidden_dims=(32, 32),
    critic_hidden_dims=(16,),
    policy=ActorCriticPolicyLearnerCfg(horizon=3),
  )
  wm = build_world_model(cfg, env, "cpu")
  actor = build_actor(cfg, wm, "cpu")
  actor.set_slot_mix({"motion": 1.0})
  actor.eval()
  critic = build_critic(cfg, wm, "cpu")

  out = trust_at_starts(env, wm, actor, critic, cfg, 2, warmup_steps=5, eps=eps)
  for source in ("model", "sim"):
    r = out[source]
    assert r["norm"].shape == (2,) and torch.isfinite(r["norm"]).all()
    assert (r["norm"] > 0).all()
    if eps == 0.0:
      # No shift: the copies replay the start, so J+ and J- differ only by
      # float32 rounding (each copy sits at its own world position).
      assert r["d_ret"].abs().max() < 1e-4
    else:
      # A real shift moves the return well above that rounding.
      assert (r["d_ret"].abs() > 1e-4).all() and torch.isfinite(r["trust"]).all()
  env.close()
