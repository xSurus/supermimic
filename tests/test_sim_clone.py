"""copy_env_state: a copied env steps exactly like its source (CPU simulator)."""

import sys
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.utils.sim_clone import copy_env_state

sys.path.insert(0, str(Path(__file__).parents[1] / "colab"))
from build_motions import convert  # noqa: E402


def test_copied_env_steps_like_its_source(tmp_path):
  csv = tmp_path / "stand.csv"
  np.savetxt(
    csv, np.tile([0, 0, 0.79, 0, 0, 0, 1] + [0.0] * 29, (60, 1)), delimiter=","
  )
  convert(csv, tmp_path / "stand.npz", "cpu")

  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  cfg.commands["motion"].motion_file = [str(tmp_path / "stand.npz")]
  cfg.commands["motion"].sampling_mode = "uniform"
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  cfg.events["base_com"].params["ranges"] = {i: (0.0, 0.0) for i in range(3)}
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  cfg.scene.num_envs = 2
  cfg.episode_length_s = int(1e9)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  dim = env.action_manager.total_action_dim
  torch.manual_seed(0)
  for _ in range(5):  # make the two envs differ
    env.step(torch.randn(2, dim) * 0.3)
  assert not torch.equal(env.sim.data.qpos[0], env.sim.data.qpos[1])

  copy_env_state(env, torch.tensor([0]), torch.tensor([1]))
  origins = env.scene.env_origins
  for _ in range(20):
    _, reward, terminated, truncated, _ = env.step(
      torch.randn(1, dim).repeat(2, 1) * 0.3
    )
    assert not (terminated | truncated).any()
    qpos = env.sim.data.qpos.clone()
    qpos[:, :3] -= origins  # each env's root relative to its own origin
    assert torch.allclose(qpos[0], qpos[1], atol=1e-5)
    assert torch.isclose(reward[0], reward[1], atol=1e-5)
  env.close()
