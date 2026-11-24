# Code changes after "Trust What Repeats"

Source: the Obsidian notes "Plan Adjustments - After Trust What Repeats" and "Paper Notes - Trust What Repeats (Capped WM Policy Gradients)", both 2026-11-24. This doc turns their "Changes to the code" section into work on the current code. Branch: `trust`, cut from `dmo` at `0d0f708`.

## Intent

The thesis question gets sharper. Dang and Edelkamp (arXiv 2611.27648) show that SuperMimic's learned-model policy gradient is untrustworthy on a small tail of rollouts, and they fix it with a per-rollout cap. They do not ask where the tail comes from. The thesis asks: is the tail caused by imagined states drifting from real ones, or by the model's Jacobians through contact? The DMO arm (simulator states, model Jacobians) separates the two.

Two tools are needed. A per-rollout gradient cap, so the SuperMimic baseline is the one a published change already beats, and so the training comparison becomes 2x2. And a gradient trust script that measures, at a fixed checkpoint, how well a gradient predicts the change in real return, binned by gradient norm. The trust comparison does not need DMO to train well.

## What changed since the notes were written

The notes describe `dmo` at `2fcf2a7`. On 2026-11-24 the repo moved onto the new upstream (`dmo` is now `0d0f708`; the old tip is tag `pre-sync-dmo`). Two consequences:

- Upstream removed the collection noise. "The pending noisy-control runs go first" no longer applies: the logged actions DMO replays are the actor's own. The DMO numbers in the Findings note (about 7 points behind on walk1, 90 vs 17 logs with `grad_norm` above 10) were measured with the noise and need re-measuring on the new base before they are quoted.
- The imagined rollout and the dynamics MLP now run in bfloat16 on CUDA. The cap's second pass runs the actor under the same autocast as the rollout.

## Goals

1. `rollout_cap` in the policy learner, off by default, with a CPU test that the capped update equals the current one when the cap never binds.
2. `scripts/grad_trust.py`: simulator trust by norm band for model-state and simulator-state gradients at the same start states.
3. Runs: the 2x2 {base, dmo} x {no cap, cap 10} on walk1, then trust on their checkpoints.

## Non-goals

- The paper's start draw and model-ranked resets (they target the motion library's tail, a different question).
- Their redraw test (a `rep_state` actor sees no sensor noise, so a redraw repeats exactly).
- Exact Jacobians (NablaEngine). Step 3 of the implementation plan, later.
- A policy EMA. Every arm is evaluated with the raw policy, stated in the write-up.

## Anti-goals

- A cap that changes the update when it does not bind. Default runs must be bit-for-bit the code they are today.
- Anything that breaks CUDA graph capture of the policy update (host syncs, data-dependent shapes).

## Constraints

- The policy update is one captured CUDA graph (`learners/learner_utils.py:CapturedUpdate`). The cap must be capture-safe and fixed at capture.
- `_optimize_step` keeps SuperMimic's clipping (global norm 1.0, elements 0.1) after the cap, as the paper's Table VII implies.
- The DMO swap lives in `WorldModelEnv.step`, so the cap works with every `states_from` unchanged.
- The Mac has no CUDA. Capture is verified on Colab.
- Versioning: one commit per working step, each with its test passing; `trust` is pushed after each step. Runs pin a commit hash.

## Approaches

Cap, how to get each rollout's gradient norm:
- A: per-rollout parameter gradients (one backward per rollout, or functorch per-sample grads). Exact, but B backward passes or a large memory cost. Rejected.
- B (chosen, from the notes): two passes. Pass 1 rolls out as today and takes `g_t = dJ/da_t`, the total derivative of the return in each action, with `autograd.grad`. The rollout's norm `G_b` is the norm of its stacked `g_t`. Pass 2 is policy-only: `loss = -mean_b(w_b * sum_t g_bt . pi(o_bt)) + aux`, with `o_bt` held fixed and `w_b = min(1, k * median(G) / G_b)`. When every `w_b` is 1 this has the same parameter gradient as today, because `dJ/dtheta = sum_t (dJ/da_t)^T dpi(o_t)/dtheta` with `o_t` fixed. The cost is a second actor pass (the paper reports 22% slower).
- C: scale the loss of each rollout by `w_b` computed from the return's parameter-gradient norm after one backward. Needs per-rollout norms, so it falls back to A. Rejected.

Trust script, where gradients and returns come from:
- Gradients come from the learner's own rollout code (`_train_batch` up to `g_t`), so the trust test measures the exact gradient training uses, for any `states_from`.
- Returns come from the simulator, played closed-loop from saved start states with actions offset by `+delta` and `-delta` along `g`. The two copies run as parallel envs in one batch.

## Design

### 1. Per-rollout cap (`learners/policy_learner.py`)

- `BpttPolicyLearnerCfg.rollout_cap: float = 0.0`. 0 is off and takes the current code path unchanged.
- In `_train_batch` with the cap on: keep each step's pre-swap action and its detached observation. After the rollout and bootstrap, `g = autograd.grad(ret.sum(), actions)`. Then `G = sqrt(sum_t |g_t|^2)` per rollout, `w = (k * G.median() / G).clamp(max=1).detach()`, and the second pass re-runs the actor on the stored observations (same autocast) to build the surrogate. `aux_loss` is summed in the second pass, since the actor recomputes it on every forward.
- New logs: `Policy/cap_frac` (share with `w < 1`), `Policy/G_median`, `Policy/G_p99`. They are logged whenever the cap is on.
- Test (CPU, `tests/test_policy_learner.py`): with `rollout_cap = 1e9`, one update gives the same actor parameters as with the cap off. With a small cap, the rollout with the largest `G` is the one scaled down.

### 2. Gradient trust (`scripts/grad_trust.py`)

For one checkpoint (W&B run path and version, as `evaluate.py` loads them):
1. Play the policy in the simulator, without the cap or updates, filling a replay buffer as training does. Pick N start frames and keep, for each, its full sim state and the next H-step window.
2. For each start, compute `g` two ways with the learner's rollout code: imagined (`states_from model`, from the start state) and replayed (`states_from sim`, from the window).
3. For each `g`, reset two envs to the start state. Play the policy closed-loop for H steps with `a_t = pi(o_t) +/- delta_t`, `delta = eps * g / max|g|`, `eps = 0.05`. Record each real return.
4. `trust = (J+ - J-) / (2 * g . delta)`. Report the median per norm band (0-1, 1-3, 3-10, 10-30, above 30 batch medians), per contact phase, and per critic-value tercile.

Day-one check: on the base checkpoint, the model-state column should roughly match their Table I (0.94 below 1x, 0.01 above 30x). If it doesn't, find out why before comparing arms.

### 3. Runs

Walk1, seeds 1 to 3, notebook section 8c extended to the 2x2 plus `fresh`: `base`, `base-cap`, `dmo`, `dmo-cap`, `fresh`, named `walk1-v2-<arm>-s<seed>`, pinned to the cap commit. Then `grad_trust.py` on the final and one early checkpoint of `base` and `dmo`. Later: 40 LAFAN1 clips, base and base+cap, 3 seeds, plus one run at the paper's settings (H = 64, B = 1024, lr 4e-4, cap 10).

## Settled while building

- Simulator state for step 3 (2026-11-24): `utils/sim_clone.py:copy_env_state` copies one env into others mid-episode. It copies `qpos`, `qvel`, `act` and `qacc_warmstart`, the action manager's action history, the motion clock (`set_time_steps`) and the episode length, shifts the root by the env-origin difference, then runs `forward()` and `update_relative_body_poses()`. On the CPU simulator a copy matches its source to 1e-5 in `qpos` and reward for 20 steps (`tests/test_sim_clone.py`). The task has no actuator delay or observation history. Per-env randomized model fields are not copied, so trust runs pin randomization to nominal as `evaluate.py` does. Copy-then-play replaces save-and-restore: the start env keeps playing unperturbed and its copies play `+delta` and `-delta`.

## Left to the builder

- Whether the copy stays exact on GPU (warp kernels may not be deterministic there). Check on Colab before trusting trust numbers; small drift only adds noise to `J+ - J-`.
- Contact labels: the task has only a self-collision contact sensor, so foot contact needs a `ContactSensorCfg` on the feet added in the trust script's env cfg.
- Whether N = 5,456 starts (the paper's number) fits one batch on a T4 or needs chunking.
- Whether `torch.median` and `autograd.grad` capture cleanly. Checked on Colab before any cap run.
