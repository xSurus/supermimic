"""snapshot and restore: a restored env replays exactly as it ran (CPU simulator)."""

import sys
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.utils.sim_clone import restore, snapshot

sys.path.insert(0, str(Path(__file__).parents[1] / "colab"))
from build_motions import convert  # noqa: E402


def test_restored_envs_replay_the_snapshot_trajectory(tmp_path):
  csv = tmp_path / "stand.csv"
  np.savetxt(
    csv, np.tile([0, 0, 0.79, 0, 0, 0, 1] + [0.0] * 29, (600, 1)), delimiter=","
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
  cfg.seed = 0  # same reset frames every run
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  dim = env.action_manager.total_action_dim
  torch.manual_seed(0)
  for _ in range(5):  # make the two envs differ
    env.step(torch.randn(2, dim) * 0.3)
  assert not torch.equal(env.sim.data.qpos[0], env.sim.data.qpos[1])
  origins = env.scene.env_origins

  def local_qpos() -> torch.Tensor:
    qpos = env.sim.data.qpos.clone()
    qpos[:, :3] -= origins  # each env's root relative to its own origin
    return qpos

  # Record env 0 for 20 steps from a snapshot, then put the snapshot back into
  # env 0 (later, same env) and env 1 (another env) and replay the actions.
  snap = snapshot(env, torch.tensor([0]))
  obs0 = env.observation_manager.compute()["full_state"][0].clone()
  actions = [torch.randn(1, dim).repeat(2, 1) * 0.1 for _ in range(20)]
  recorded = []
  for a in actions:
    _, reward, *_ = env.step(a)
    recorded.append((local_qpos()[0], reward[0].clone()))
  restore(env, {k: v.repeat(2, *[1] * (v.dim() - 1)) for k, v in snap.items()},
          torch.tensor([0, 1]))  # fmt: skip
  # The policy acts on these: they must show the restored state, not a cache.
  obs = env.observation_manager.compute()["full_state"]
  assert torch.allclose(obs, obs0.expand(2, -1), atol=1e-5)
  for k, (a, (qpos0, reward0)) in enumerate(zip(actions, recorded, strict=True)):
    _, reward, terminated, truncated, _ = env.step(a)
    assert torch.allclose(reward, reward0.expand(2), atol=1e-5)
    done = terminated | truncated
    if done.any():  # random actions can topple the robot; both must fall together
      assert done.all() and k >= 10
      break
    assert torch.allclose(local_qpos(), qpos0.expand(2, -1), atol=1e-5)
  env.close()


def test_restore_refreshes_foot_contact_sensor(tmp_path):
  from supermimic.scripts.grad_trust import FOOT_CONTACT_SENSOR, add_foot_contact_sensor

  csv = tmp_path / "stand.csv"
  np.savetxt(
    csv, np.tile([0, 0, 0.79, 0, 0, 0, 1] + [0.0] * 29, (600, 1)), delimiter=","
  )
  convert(csv, tmp_path / "stand.npz", "cpu")
  cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  add_foot_contact_sensor(cfg)
  cfg.commands["motion"].motion_file = [str(tmp_path / "stand.npz")]
  cfg.commands["motion"].sampling_mode = "uniform"
  cfg.scene.num_envs = 2
  cfg.episode_length_s = int(1e9)
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  sensor = env.scene[FOOT_CONTACT_SENSOR]
  # Step with zero actions so the standing robot settles firmly on the ground.
  zeros = torch.zeros(2, env.action_manager.total_action_dim)
  for _ in range(5):
    env.step(zeros)
  grounded = (sensor.data.found[0] > 0).clone()
  assert grounded.any(), "expected standing robot to have ground foot contact"
  snap = snapshot(env, torch.tensor([0]))

  # Lift both envs into the air so contacts drop to zero, then restore grounded snap.
  env.sim.data.qpos[:, 2] += 2.0
  env.sim.forward()
  assert not (sensor.data.found > 0).any()

  restore(
    env,
    {k: v.repeat(2, *[1] * (v.dim() - 1)) for k, v in snap.items()},
    torch.tensor([0, 1]),
  )
  restored = sensor.data.found > 0
  assert torch.equal(restored[0], grounded)
  assert torch.equal(restored[1], grounded)
  env.close()
