"""grad_trust: band summary, and the trust path on the CPU simulator."""

import json
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
from supermimic.scripts.grad_trust import (
  CONTACT_PHASES,
  FOOT_CONTACT_SENSOR,
  _print_eps_report,
  _summarize_eps,
  add_foot_contact_sensor,
  classify_contact_phase,
  parse_args,
  summarize,
  summarize_paired,
  trust_at_starts,
  trust_at_starts_multi,
)

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


def test_summarize_rich_and_paired_metrics():
  norm_m = torch.tensor([1.0, 1.0, 1.0, 2.0, 5.0, 12.0, 80.0])
  norm_s = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.5, 3.0, 4.0])
  trust_m = torch.tensor([0.9, 0.9, 0.9, 0.8, 0.5, 0.2, 0.01])
  trust_s = torch.tensor([0.9, 0.9, 0.9, 0.85, 0.7, 0.6, 0.5])
  sign_m = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
  sign_s = torch.ones(7)
  fell_base = torch.tensor([False, False, False, False, False, True, True])
  real_frac = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.25])
  cos = torch.tensor([1.0, 1.0, 1.0, 0.9, 0.8, 0.4, 0.1])

  rich = summarize(
    norm_m,
    trust_m,
    value=torch.arange(7.0),
    trust_soft=trust_m,
    model_fd_trust=torch.tensor([0.99, 0.99, 0.99, 0.98, 0.95, 0.4, 0.05]),
    sign_agrees=sign_m,
    fell_base=fell_base,
    fell_any=fell_base,
    replay_real_frac=real_frac,
  )
  assert rich["bands"]["30-inf"]["model_fd_trust"] == pytest.approx(0.05)
  assert rich["bands"]["30-inf"]["fell_base_frac"] == pytest.approx(1.0)
  assert rich["bands"]["30-inf"]["trust_surviving"] is None
  assert rich["top_0p5_sq_share"] > 0.95

  paired = summarize_paired(
    {"norm": norm_m, "trust": trust_m, "trust_soft": trust_m, "sign_agrees": sign_m},
    {
      "norm": norm_s,
      "trust": trust_s,
      "trust_soft": trust_s,
      "sign_agrees": sign_s,
      "replay_real_frac": real_frac,
    },
    cos,
  )
  assert paired["bands"]["30-inf"]["model_trust"] == pytest.approx(0.01)
  assert paired["bands"]["30-inf"]["sim_trust"] == pytest.approx(0.5)
  assert paired["bands"]["30-inf"]["median_norm_ratio"] == pytest.approx(0.05)
  assert paired["bands"]["30-inf"]["median_cos"] == pytest.approx(0.1)


def test_summarize_paired_symmetric_overlap_and_spearman():
  # Odd length 9 with exact median 1.0 for both model and sim, and disjoint 10-30x tails:
  # model bands: 0-1={0,1,2,3}, 1-3={4,5}, 3-10={6}, 10-30={7}, 30-inf={8}
  # sim bands:   0-1={0,1,2,7}, 1-3={4,6}, 3-10={5}, 10-30={3}, 30-inf={8}
  norm_m = torch.tensor([0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 5.0, 15.0, 40.0])
  norm_s = torch.tensor([0.4, 0.5, 0.6, 20.0, 1.0, 4.0, 1.2, 0.8, 35.0])
  trust_m = torch.tensor([0.9, 0.9, 0.9, 0.88, 0.8, 0.7, 0.4, 0.05, 0.01])
  trust_s = torch.tensor([0.9, 0.9, 0.9, 0.04, 0.8, 0.5, 0.75, 0.85, 0.02])
  trust_m_soft = trust_m + 0.02
  trust_s_soft = trust_s + 0.03
  sign_m = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
  sign_s = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
  real_frac = torch.tensor([1.0, 1.0, 1.0, 0.5, 1.0, 0.75, 1.0, 1.0, 0.25])
  cos = torch.tensor([1.0, 1.0, 1.0, 0.15, 0.9, 0.6, 0.5, 0.2, 0.05])

  paired = summarize_paired(
    {
      "norm": norm_m,
      "trust": trust_m,
      "trust_soft": trust_m_soft,
      "sign_agrees": sign_m,
    },
    {
      "norm": norm_s,
      "trust": trust_s,
      "trust_soft": trust_s_soft,
      "sign_agrees": sign_s,
      "replay_real_frac": real_frac,
    },
    cos,
  )

  # Mirror-image conditioning on sim's 10-30x band (idx 3 only): sim collapses, model stays high.
  sim_10_30 = paired["on_sim_bands"]["10-30"]
  assert sim_10_30 == {
    "n": 1,
    "sim_trust": pytest.approx(0.04),
    "model_trust": pytest.approx(0.88),
    "sim_trust_soft": pytest.approx(0.07),
    "model_trust_soft": pytest.approx(0.90),
    "sim_sign_agrees": pytest.approx(0.0),
    "model_sign_agrees": pytest.approx(1.0),
    "median_norm_ratio_model_over_sim": pytest.approx(0.8 / 20.0),
    "median_cos": pytest.approx(0.15),
    "sim_replay_real_frac": pytest.approx(0.5),
  }

  # 5x5 joint band counts preserve exact marginal totals.
  jbc = paired["joint_band_counts"]
  band_names = ("0-1", "1-3", "3-10", "10-30", "30-inf")
  assert tuple(jbc.keys()) == band_names
  assert jbc["0-1"]["0-1"] == 3
  assert jbc["0-1"]["10-30"] == 1
  assert jbc["10-30"]["0-1"] == 1
  assert jbc["30-inf"]["30-inf"] == 1
  for b in band_names:
    assert sum(jbc[b].values()) == paired["bands"][b]["n"]
    assert sum(jbc[row][b] for row in band_names) == paired["on_sim_bands"][b]["n"]

  # Jaccard overlap: >=3x is {6,7,8} vs {3,5,8} -> 1/5; >=10x is {7,8} vs {3,8} -> 1/3.
  assert paired["jaccard_ge_3x"] == pytest.approx(1 / 5)
  assert paired["jaccard_ge_10x"] == pytest.approx(1 / 3)

  # Exact Spearman rank correlation: ranks [0..8] vs [0,1,2,7,4,6,5,3,8], sum(d^2)=34 -> 1 - 204/720.
  assert paired["spearman_norm"] == pytest.approx(43 / 60)

  # Empty-union Jaccard returns None and constant norms return Spearman None.
  flat = torch.ones(4)
  paired_flat = summarize_paired(
    {"norm": flat, "trust": flat, "trust_soft": flat, "sign_agrees": flat},
    {
      "norm": flat,
      "trust": flat,
      "trust_soft": flat,
      "sign_agrees": flat,
      "replay_real_frac": flat,
    },
    flat,
  )
  assert paired_flat["jaccard_ge_3x"] is None
  assert paired_flat["jaccard_ge_10x"] is None
  assert paired_flat["spearman_norm"] is None


def test_classify_contact_phase_and_impact_count():
  # Shape (N=5, H+1=4, F=2)
  fc = torch.tensor(
    [
      [[0, 0], [0, 0], [1, 0], [1, 1]],  # flight at t=0, 2 touchdowns later
      [[1, 0], [1, 0], [1, 0], [0, 0]],  # single_support, 0 touchdowns
      [[1, 1], [1, 1], [1, 1], [1, 1]],  # double_support, 0 touchdowns
      [[1, 0], [1, 1], [1, 1], [1, 1]],  # impact on t=0->1 (right foot 0->1)
      [[0, 0], [1, 0], [1, 0], [1, 0]],  # masked out on t=0 (post-reset contact)
    ],
    dtype=torch.bool,
  )
  phase_all, impacts_all = classify_contact_phase(fc)
  assert phase_all.tolist() == [0, 1, 2, 3, 3]
  assert impacts_all.tolist() == [2, 0, 0, 1, 1]

  valid = torch.tensor(
    [
      [1, 0, 0],  # mask out t=1,2 touchdowns on env 0
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
      [0, 0, 0],  # terminated on step 0: must not count post-reset as impact
    ],
    dtype=torch.bool,
  )
  phase_m, impacts_m = classify_contact_phase(fc, valid)
  assert phase_m.tolist() == [0, 1, 2, 3, 0]
  assert impacts_m.tolist() == [0, 0, 0, 1, 0]


def test_summarize_contact_warmup_and_bootstrap_slices():
  norm_m = torch.tensor([1.0, 1.0, 1.0, 2.0, 5.0, 12.0, 80.0])
  norm_s = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.5, 3.0, 4.0])
  trust_m = torch.tensor([0.9, 0.85, 0.8, 0.7, 0.4, 0.1, 0.01])
  trust_s = torch.tensor([0.92, 0.88, 0.85, 0.75, 0.65, 0.5, 0.4])
  sign_m = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
  sign_s = torch.ones(7)
  real_frac = torch.ones(7)
  cos = torch.tensor([0.95, 0.9, 0.85, 0.8, 0.6, 0.3, 0.1])
  boot_share = torch.tensor([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])
  cos_tb = torch.tensor([0.96, 0.96, 0.97, 0.97, 0.98, 0.98, 0.99])
  trust_nb = torch.tensor([0.70, 0.68, 0.65, 0.55, 0.30, 0.05, -0.02])
  contact_phase = torch.tensor([0, 1, 1, 2, 2, 3, 3])
  impact_count = torch.tensor([0, 0, 1, 1, 2, 2, 3])
  warmup = torch.tensor([0, 0, 5, 5, 25, 100, 100])

  out = summarize(
    norm_m,
    trust_m,
    value=torch.arange(7.0),
    trust_soft=trust_m,
    sign_agrees=sign_m,
    boot_grad_share=boot_share,
    cos_tot_boot=cos_tb,
    trust_noboot=trust_nb,
    trust_noboot_soft=trust_nb,
    sign_agrees_noboot=sign_m,
    contact_phase=contact_phase,
    impact_count=impact_count,
    start_warmup=warmup,
  )
  assert out["median_boot_grad_share"] == pytest.approx(0.93)
  assert out["median_cos_tot_boot"] == pytest.approx(0.97)
  assert out["median_trust_noboot"] == pytest.approx(0.55)
  assert out["bands"]["30-inf"]["boot_grad_share"] == pytest.approx(0.96)
  assert out["bands"]["30-inf"]["trust_noboot"] == pytest.approx(-0.02)

  assert tuple(out["by_contact_phase"].keys()) == CONTACT_PHASES
  assert out["by_contact_phase"]["flight"]["n"] == 1
  assert out["by_contact_phase"]["impact"]["n"] == 2
  assert out["by_contact_phase"]["impact"]["frac_ge_10x"] == pytest.approx(0.5)

  assert tuple(out["by_impact_count"].keys()) == ("0", "1", "2+")
  assert out["by_impact_count"]["2+"]["n"] == 3

  assert tuple(out["by_warmup"].keys()) == ("0", "5", "25", "100")
  assert out["by_warmup"]["100"]["n"] == 2
  assert out["by_warmup"]["100"]["p99_norm"] >= 12.0

  # Single warmup age omits by_warmup.
  out_single = summarize(
    norm_m, trust_m, value=torch.arange(7.0), start_warmup=torch.full((7,), 100)
  )
  assert "by_warmup" not in out_single

  paired = summarize_paired(
    {
      "norm": norm_m,
      "trust": trust_m,
      "trust_soft": trust_m,
      "sign_agrees": sign_m,
      "contact_phase": contact_phase,
    },
    {
      "norm": norm_s,
      "trust": trust_s,
      "trust_soft": trust_s,
      "sign_agrees": sign_s,
      "replay_real_frac": real_frac,
    },
    cos,
  )
  assert tuple(paired["by_contact_phase"].keys()) == CONTACT_PHASES
  assert paired["by_contact_phase"]["impact"]["n"] == 2
  assert paired["by_contact_phase"]["flight"]["model_trust"] == pytest.approx(0.9)
  assert paired["by_contact_phase"]["flight"]["sim_trust"] == pytest.approx(0.92)


def test_grad_trust_cli_parses_multi_eps_and_ls_iterations():
  task, cfg = parse_args(
    [
      "SuperMimic-GenCoord-G1",
      "--wandb-path",
      "surus/supermimic/abc123",
      "--num-starts",
      "2048",
      "--chunk-size",
      "512",
      "--warmup-schedule",
      "0",
      "5",
      "25",
      "100",
      "--eps-values",
      "0",
      "0.005",
      "0.015",
      "0.05",
      "--ls-iterations",
      "100",
      "--check-noboot",
    ]
  )
  assert task == "SuperMimic-GenCoord-G1"
  assert cfg.warmup_schedule == (0, 5, 25, 100)
  assert cfg.eps_values == (0.0, 0.005, 0.015, 0.05)
  assert cfg.ls_iterations == 100
  assert cfg.check_noboot is True

  task_legacy, legacy = parse_args(
    [
      "SuperMimic-GenCoord-G1",
      "--wandb-path",
      "surus/supermimic/abc123",
      "--eps",
      "0.015",
    ]
  )
  assert task_legacy == "SuperMimic-GenCoord-G1"
  assert legacy.eps == pytest.approx(0.015)
  assert legacy.eps_values == ()
  assert legacy.warmup_schedule == ()
  assert legacy.check_noboot is False
  assert legacy.ls_iterations is None


def test_trust_at_starts_multi_identical_starts_and_ls_iterations(tmp_path, capsys):
  csv = tmp_path / "stand.csv"
  np.savetxt(
    csv, np.tile([0, 0, 0.79, 0, 0, 0, 1] + [0.0] * 29, (600, 1)), delimiter=","
  )
  convert(csv, tmp_path / "stand.npz", "cpu")
  env_cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
  pin_randomization(env_cfg)
  add_foot_contact_sensor(env_cfg)
  add_foot_contact_sensor(env_cfg)  # idempotent
  assert sum(s.name == FOOT_CONTACT_SENSOR for s in env_cfg.scene.sensors) == 1
  env_cfg.sim.mujoco.ls_iterations = 100
  env_cfg.commands["motion"].motion_file = [str(tmp_path / "stand.npz")]
  env_cfg.commands["motion"].sampling_mode = "uniform"
  env_cfg.episode_length_s = int(1e9)
  env_cfg.seed = 0
  env_cfg.scene.num_envs = 3 * 2
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cpu"))
  assert env.unwrapped.sim.mj_model.opt.ls_iterations == 100
  assert env.unwrapped.sim.wp_model.opt.ls_iterations == 100
  assert FOOT_CONTACT_SENSOR in env.unwrapped.scene.sensors

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

  eps_values = (0.0, 0.005, 0.05)
  by_eps = trust_at_starts_multi(
    env,
    wm,
    actor,
    critic,
    cfg,
    2,
    warmup_steps=5,
    eps_values=eps_values,
    check_noboot=True,
  )
  assert list(by_eps.keys()) == [0.0, 0.005, 0.05]

  for source in ("model", "sim"):
    ref_norm = by_eps[0.0][source]["norm"]
    ref_val = by_eps[0.0][source]["value"]
    ref_cos = by_eps[0.0][source]["cos_model_sim"]
    ref_boot = by_eps[0.0][source]["boot_grad_share"]
    assert (
      ref_norm.shape == (2,) and torch.isfinite(ref_norm).all() and (ref_norm > 0).all()
    )
    assert ref_boot.shape == (2,) and torch.isfinite(ref_boot).all()
    for eps in eps_values:
      r = by_eps[eps][source]
      # Starts and gradients are computed once, so norm/value/cos/boot are identical across eps.
      assert torch.equal(r["norm"], ref_norm)
      assert torch.equal(r["value"], ref_val)
      assert torch.equal(r["cos_model_sim"], ref_cos)
      assert torch.equal(r["boot_grad_share"], ref_boot)
      assert r["foot_contacts"].shape == (2, 4, 2)
      assert r["foot_contacts"].dtype == torch.bool
      assert r["contact_phase"].shape == (2,)
      assert ((r["contact_phase"] >= 0) & (r["contact_phase"] <= 3)).all()
      assert r["impact_count"].shape == (2,) and (r["impact_count"] >= 0).all()
      assert torch.equal(r["start_warmup"], torch.tensor([5, 5]))
      if eps == 0.0:
        assert r["d_ret"].abs().max() < 1e-4
        assert torch.isnan(r["trust"]).all()
        assert torch.isnan(r["trust_soft"]).all()
        assert torch.isnan(r["sign_agrees"]).all()
        assert torch.isnan(r["trust_noboot"]).all()
        assert torch.isnan(r["trust_noboot_soft"]).all()
        assert torch.isnan(r["sign_agrees_noboot"]).all()
        if source == "model":
          assert torch.isnan(r["model_fd_trust"]).all()
      else:
        assert torch.isfinite(r["trust"]).all()
        assert torch.isfinite(r["trust_soft"]).all()
        assert torch.isfinite(r["sign_agrees"]).all()
        assert torch.isfinite(r["trust_noboot"]).all()
        assert torch.isfinite(r["trust_noboot_soft"]).all()
        assert torch.isfinite(r["sign_agrees_noboot"]).all()
        if source == "model":
          assert torch.isfinite(r["model_fd_trust"]).all()
        if eps == 0.05:
          assert (r["d_ret"].abs() > 1e-4).all()

  # Simulate mixed-age warmup tensor to verify JSON + console reporting of all new slices.
  mixed_005 = {
    s: {**r, "start_warmup": torch.tensor([0, 100])} for s, r in by_eps[0.05].items()
  }
  summary_0 = _summarize_eps(by_eps[0.0], 0.0, 2)
  summary_005 = _summarize_eps(mixed_005, 0.05, 2)
  json.dumps({"by_eps": {"0": summary_0, "0.05": summary_005}}, allow_nan=False)
  assert summary_0["model"]["median_trust"] is None
  assert summary_0["model"]["sign_agrees"] is None
  assert "by_contact_phase" in summary_005["model"]
  assert "by_impact_count" in summary_005["model"]
  assert "by_warmup" in summary_005["model"]
  assert "by_contact_phase" in summary_005["paired"]

  _print_eps_report(summary_005, 0.05, 2, 0)
  out_text = capsys.readouterr().out
  assert "||g_V||/||g||=" in out_text
  assert "noboot=" in out_text
  assert "by contact phase:" in out_text
  assert "by warmup steps:" in out_text
  assert "paired (model bands):" in out_text
  assert "paired (sim bands):" in out_text
  assert "paired (contact phase):" in out_text
  assert "Spearman(G_m, G_s)=" in out_text
  assert "Jaccard >=3x=" in out_text
  assert "fell=" in out_text and "/" in out_text.split("fell=")[1].split()[0]
  env.close()


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
    assert r["near_clip_end"].shape == (2,)
    assert r["cos_model_sim"].shape == (2,) and torch.isfinite(r["cos_model_sim"]).all()
    if eps == 0.0:
      # No shift: the copies replay the start, so J+ and J- differ only by
      # float32 rounding (each copy sits at its own world position).
      assert r["d_ret"].abs().max() < 1e-4
      assert torch.isnan(r["trust"]).all()
      assert torch.isnan(r["trust_soft"]).all()
    else:
      # A real shift moves the return well above that rounding.
      assert (r["d_ret"].abs() > 1e-4).all() and torch.isfinite(r["trust"]).all()
      assert torch.isfinite(r["trust_soft"]).all()
      if source == "model":
        assert torch.isfinite(r["model_fd_trust"]).all()
  env.close()
