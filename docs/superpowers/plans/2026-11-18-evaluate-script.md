# Evaluation script: normalized return per clip

Goal: score a trained SuperMimic checkpoint the way the paper scores LAFAN1 (section 4: "Normalized return ... is the undiscounted sum of the per-step reward over the full episode, divided by the largest value this sum can take. A policy that terminates early therefore scores low."; Figure 4: "Each point on a curve is a saved checkpoint scored on all 40 clips").

Branch: `thesis` in `supermimic_ws/supermimic`. The sibling `../mjlab_research` is read-only.

## Global constraints

- One new file `src/supermimic/scripts/evaluate.py` and one test file `tests/test_evaluate.py`. Do not edit other files under `src/` or `tests/`.
- Use `uv run`. Before committing: `uv run ruff format`, `uv run ruff check`, `MUJOCO_GL=disable uv run pytest -q`. All tests must pass.
- Line length 88, two-space indentation (match the repo).
- Prose rules for comments, docstrings and the commit message: no em dashes, straight quotes, plain words.
- End the commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Keep it small. No new dependencies, no plotting, no W&B logging, no config for values that never change.

## Task 1: `scripts/evaluate.py`

CLI, in the same style as `scripts/play.py` (task id chosen first with `tyro`, then a dataclass):

```
uv run src/supermimic/scripts/evaluate.py SuperMimic-GenCoord-G1 \
  --wandb-path <entity>/<project>/<run_id> \
  --motion-file '/content/store/data/npz/lafan1/*.npz' \
  [--checkpoint latest] [--output-file results.json] [--device cuda:0]
```

Config dataclass `EvaluateConfig`: `wandb_path: str`, `motion_file: str | None = None`, `checkpoint: str = "latest"`, `output_file: str | None = None`, `device: str | None = None`.

Behaviour:

1. Load the run and its runner config exactly as `play.py` does (`wandb.Api().run`, `load_runner_cfg`, `check_representation_matches`). Motions: `--motion-file` (glob allowed, sorted, as in `play.py`), else the run's linked `motions` artifacts, as in `play.py`. Raise if `runner_cfg.history_length != 1` (not supported, say so in the message).
2. Checkpoint: the run logs model checkpoints as artifact `model-{run.id}` (see `utils/wandb_artifacts.py:log_model_artifact`). Fetch `wandb.Api().artifact(f"{run.entity}/{run.project}/model-{run.id}:{cfg.checkpoint}")` (so `latest`, `v3` etc. work) and download it with `download_artifact_cached`. Load `world_model.pt` with `wm.load_bundle` and `actor.pt` into `build_actor(...)`, as `play.py` does. Print which artifact version was used.
3. Env: `load_env_cfg(task_id, play=False)` (keep the training env: same rewards, terminations and domain-randomization events). Then override, for evaluation only:
   - `env_cfg.episode_length_s = int(1e9)`, so an episode is a whole clip.
   - motion command: `pose_range = {}`, `velocity_range = {}` (no reset randomization).
   - one env per clip: `num_envs = number of clips`.
   - every env must start at frame 0 of its own clip. The motion command concatenates clips; clip `i` covers frames `[starts[i], motion_ends[i])` with `starts = [0, *motion_ends[:-1]]` (`MotionLoader.motion_ends`, in `mjlab_research/src/mjlab/tasks/tracking/mdp/commands.py`). `sampling_mode="start"` would put every env on clip 0. Use `sampling_mode="external"` and, after the env is built and before the first reset, override the instance method: `cmd._external_sampling = lambda env_ids: starts[env_ids]`. Mark it with a `# ponytail:` comment (it relies on a private method of mjlab's MotionCommand). Then call `env.reset()`.
   - Build the env with `create_env` from `scripts/train_utils.py` if it fits; it wraps in `RslRlVecEnvWrapper`.
4. Policy step, deterministic (no exploration noise), mirroring `SuperMimicRunner.step` and `_build_sim_obs_td` in `runners/supermimic_runner.py` for `history_length == 1`: read `obs["full_state"]` and `obs["actor"]`, `rep_state = representation.represent(...)`, `aux_state = representation.extract_aux_state(...)`, `sim_ctx = wm.sim_context.read(env.unwrapped)`, the motion times from the command's `time_steps` before stepping, sensor from `representation.compute_actor_sensor(rep_state, sim_ctx, aux_state=aux_state, sim_actor_obs=<actor obs if actor_obs == "proprio" else None>, noisy=<actor_obs == "proprio">)`, goals `{ts.name: ts.extract_training_goal(motion, motion_times, aux_state) for ts in active specs}` where active specs are `actor.task_specs` with `runner_cfg.slot_mix.get(name, 0) > 0`, then `build_actor_obs_td(goals, sensor.unsqueeze(1))` and `actions = actor(td)`. Run under `torch.inference_mode()`.
5. Scoring. Per env, accumulate the reward returned by `env.step` while the env's episode is still running, including the step on which it ends (mjlab computes the reward before the auto-reset). An env's episode ends at its first `done`. Record whether it ended by failure: `termination_manager.terminated & ~termination_manager.time_outs` at that step (`motion_end` is a time-out, so reaching the clip end is not a failure). Stop when every env is done.
   - Per-step reward ceiling: `max_step = env.unwrapped.step_dt * sum(w for w in reward weights if w > 0)`, the same formula as `WorldModelEnv.max_step_reward` in `envs/world_model_env.py`. Put it in a small pure function.
   - Normalized return of clip `i` = reward sum / (`max_step * (clip_len_i - 1)`), with `clip_len_i = motion_ends[i] - starts[i]` frames. Put this in a small pure function too.
6. Output: print one line per clip (index, clip name if the motion loader exposes names, else the file stem, frames, normalized return, `fell`), then `mean normalized return` (mean over clips) and `failed clips: k / n`. With `--output-file`, write JSON with the per-clip rows, the mean, the failure count, the run path and the checkpoint version.

Test (`tests/test_evaluate.py`, runs on CPU without a simulator): test the two pure functions. For the ceiling, a fake reward config with weights `{a: 1.0, b: 0.5, c: -2.0}` and `step_dt=0.02` gives `0.03`. For the normalization, a clip of 501 frames with reward sum 25.0 and `max_step=0.1` gives `0.5`.

Not required: running the script end to end (the Mac has no CUDA). Do check it imports: `uv run python -c "import supermimic.scripts.evaluate"`.
