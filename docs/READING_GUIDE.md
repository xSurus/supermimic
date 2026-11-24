# Reading guide

This guide is for someone who knows RL and has read the SuperMimic paper but has not opened this code. It says what to read, in what order, and which files each thesis idea would touch. References use the form `path:Symbol` (or `path:Class.method`). Paths starting with `src/supermimic/` are in this repo. Paths starting with `mjlab_research/` are in the sibling simulator repo.

Everything here was read from the code on the `thesis` branch. Where the code was unclear, the guide says so.

## How to run

Use the notebook `colab/run.ipynb`. It clones both repos, installs, runs the tests, builds the motion files with `colab/build_motions.py` and starts training. The README shows the plain command: `uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --runner.run_name <name>`.

Two task ids exist, registered in `src/supermimic/tasks/__init__.py:register_mjlab_task`: `SuperMimic-GenCoord-G1` and `SuperMimic-MaxCoord-G1`. Training logs to W&B and needs a login. Motions come from a W&B registry by default (`src/supermimic/scripts/train_utils.py:download_registry_motions`) or from `--motion-file`.

The first learning update compiles with `torch.compile` and can print nothing for many minutes on Colab. That is expected and is not a hang.

## One training iteration

The outer loop is `src/supermimic/runners/supermimic_runner.py:SuperMimicRunner.run`. One loop index is one real environment step for all `num_envs` (4096 by default) plus zero or more learning updates.

1. `src/supermimic/scripts/train.py:main` builds the mjlab env through `src/supermimic/scripts/train_utils.py:create_env` and sets the motion command's `sampling_mode` (the default `return` maps to mjlab's `external` mode). It then constructs `SuperMimicRunner`.
2. Construction runs `src/supermimic/runners/supermimic_runner.py:SuperMimicRunner._init_common`. It calls `build_world_model`, which reads the env's observation sizes, builds the sim context (`src/supermimic/sim_context.py:SimContext.from_env`), the representation (`src/supermimic/representations.py:create_representation`) and the `src/supermimic/models.py:Dynamics` MLP. It also creates the `src/supermimic/replay_buffer.py:ReplayBuffer` and the `src/supermimic/learners/wm_learner.py:WorldModelLearner`.
3. `src/supermimic/runners/supermimic_runner.py:SuperMimicRunner._init_policy` builds the actor (`build_actor`, which returns a `src/supermimic/models.py:MultiEncoderAE`), a critic MLP, and the imagination env `src/supermimic/envs/world_model_env.py:WorldModelEnv`. The default policy learner is `src/supermimic/learners/policy_learner.py:ActorCriticPolicyLearner`.
4. Collection: `src/supermimic/runners/supermimic_runner.py:SuperMimicRunner.step`. It reads `obs["full_state"]`, converts it with `represent`, reads `sim_ctx`, builds the actor input in `_build_sim_obs_td`, and runs the actor. The simulator gets the actor's action as is: upstream removed the collection noise in `7e94b07`. `_step_and_store` steps the real simulator and writes state, action, previous action, motion frame, terminated, truncated and `sim_ctx` into the replay buffer.
5. Reset sampling: still inside `step`, `src/supermimic/runners/reset_sampling.py:ReturnSampler.record` buffers rewards per motion bin. Every 20 steps it returns a new reset distribution, which goes to mjlab through `MotionCommand.set_sampling_distribution`.
6. Update schedule: `src/supermimic/runners/supermimic_runner.py:Ratio` decides how many updates follow this step. With the defaults that is 4096 / (10 * 2048 * 8) * 16 = 0.4 updates per env step, so about one update every 2.5 steps (computed by hand from the config defaults, not measured).
7. World model update: `src/supermimic/learners/wm_learner.py:WorldModelLearner.update` takes 10 windows of 8 steps (batch 2048) from `ReplayBuffer.sample`. For each window `_rollout` runs the dynamics open loop from the first state on the recorded actions, and `_compute_losses` scores the predicted states with per-group weights 1 / nominal_error^2. One optimizer step per window.
8. Policy update: `run` draws start states with `src/supermimic/envs/world_model_env.py:sample_start` and passes them to `src/supermimic/learners/policy_learner.py:BpttPolicyLearner.update`, which freezes the dynamics parameters (`requires_grad_(False)`). `_train_batch` rolls the actor (under bfloat16 autocast on CUDA) for 32 steps through `WorldModelEnv.step`, sums reward weighted by a survival product `(1 - term_prob)`, adds the critic bootstrap (`ActorCriticPolicyLearner._bootstrap`), and takes one actor step on the negated return. Gradients pass through the dynamics into the actions.
9. Critic update: `src/supermimic/learners/policy_learner.py:ActorCriticPolicyLearner._fit_critic` fits the critic on detached imagined steps with `gae_returns` (gamma 0.99, lambda 0.95), 5 epochs of 4 minibatches.
10. One imagined step: `src/supermimic/envs/world_model_env.py:WorldModelEnv.step` calls `Dynamics.forward` (its MLP runs in bfloat16 on CUDA), computes a soft termination in `_compute_term_prob` (a sigmoid of the anchor height and tilt errors against the env's termination thresholds), and the reward in `src/supermimic/representations.py:StateRepresentation.compute_wm_reward_from_reference`, which calls `src/supermimic/utils/tracking_reward.py:tracking_reward`, which scores body velocities relative to the anchor heading (upstream `5fb9229`). `_compute_commands` then advances the reference frame.
11. Checkpoints: every `save_interval` (1000) loop indices, `run` saves `world_model.pt`, `actor.pt` and `critic.pt` and uploads them to W&B. ONNX export (`src/supermimic/runners/exporter.py:export_actor_as_onnx`) only runs in proprio mode and when the motion data has at most 15,000 frames.

Both updates run as CUDA graphs after three warmup calls (`src/supermimic/learners/learner_utils.py:CapturedUpdate`). Anything you add inside `_update_from` or `_train_batch` must avoid host syncs and keep fixed shapes, or the capture breaks. Optimizers are Muon on hidden Linear weights and Adam on the rest (`build_optimizers`).

## Reading order

Line counts are from `wc -l` on this branch. `src/supermimic/` has 6,164 lines in total. The first 14 files below are about 3,600 of them and cover everything the thesis ideas touch. The viewer, exporter and W&B helpers can wait.

| Order | File | Lines | Why |
|---|---|---|---|
| 1 | `src/supermimic/scripts/train.py` | 100 | Entry point, CLI, task selection |
| 2 | `src/supermimic/runners/supermimic_runner.py` | 680 | Config defaults, the loop, collection, checkpoints |
| 3 | `src/supermimic/envs/world_model_env.py` | 194 | Imagined step, termination, reward call, start sampling |
| 4 | `src/supermimic/learners/policy_learner.py` | 202 | BPTT and actor-critic objective |
| 5 | `src/supermimic/learners/wm_learner.py` | 191 | World model loss |
| 6 | `src/supermimic/models.py` | 250 | `Dynamics` and the multi-encoder actor |
| 7 | `src/supermimic/representations.py` | 784 | State layouts, integration, goals, sensor synthesis |
| 8 | `src/supermimic/utils/tracking_reward.py` | 86 | The differentiable reward |
| 9 | `src/supermimic/sim_context.py` | 204 | What physical parameters the world model sees |
| 10 | `src/supermimic/replay_buffer.py` | 121 | Window sampling across episode boundaries |
| 11 | `src/supermimic/tasks/env_cfgs.py` | 238 | How the mjlab G1 tracking task is changed |
| 12 | `src/supermimic/tasks/task_spec.py` | 218 | Motion and velocity goal encoders |
| 13 | `src/supermimic/runners/reset_sampling.py` | 153 | Return-based reset distribution |
| 14 | `src/supermimic/learners/learner_utils.py` | 136 | Optimizers, CUDA graph capture |
| later | `src/supermimic/utils/gencoord_helpers.py`, `utils/maxcoord_helpers.py`, `utils/analytic_ik.py`, `utils/helpers.py` | 113, 254, 155, 214 | Math behind the two representations |
| later | `src/supermimic/scripts/play.py`, `scripts/viewers.py`, `scripts/benchmark.py`, `runners/exporter.py` | 152, 492, 219, 321 | Playback, viewers, speed benchmark, ONNX |

The tests under `tests/` are small and named after the module they cover, for example `tests/test_wm_termination.py` and `tests/test_replay_buffer.py`. Reading a test next to its module is often the fastest way to see the tensor shapes.

The actor's batch is split into contiguous blocks, one per task (`MultiEncoderAE.slot_ranges`). With the default `slot_mix` all slots are motion tracking, so you can ignore the velocity task until idea 4.

## The two representations

Both live in `src/supermimic/representations.py`. Each defines a full state (what the env's `full_state` observation group emits), a rep state (what the dynamics MLP and the critic see), an aux state `[root_pos, root_quat]` kept in the world frame, and a prediction that is integrated analytically into the next state.

`GenCoordRepresentation` uses joint space. The rep state is `[z, g_b, v_b, w_b, qj, vj]`: root height, gravity and root twist in the body frame, joint positions and velocities. The network predicts velocity deltas `[dv_b, dw_b, dvj]` in the current root frame, and `GenCoordRepresentation.apply_prediction` integrates poses semi-implicitly with `substep_coeff` to match the 4 physics substeps. Body poses for the reward come from mjlab's forward kinematics in `world_frame_bodies`.

`MaxCoordRepresentation` uses per-body coordinates, following ControlVAE. The rep state holds every tracked body's position, rot6d, linear and angular velocity in the root frame, plus root height and up direction. The network predicts per-body velocity deltas. Joint angles, which the actor's proprio sensor needs, are recovered by `src/supermimic/utils/analytic_ik.py:AnalyticIK` in `MaxCoordRepresentation.joint_state`.

The world model loss in `WorldModelLearner._compute_losses` scores `loss_blocks`, which differ per representation. The weights are the `nominal_error_*` fields of `WorldModelLearnerCfg`. GenCoord is the default (`SuperMimicRunnerCfg.representation`); `train.py` switches to maxcoord when the task id contains `MaxCoord`.

## What lives in mjlab_research

The simulator side is the fork at `../mjlab_research`, which carries no commits of its own: it follows `HybridRobotics/mjlab_research`. Upstream's `4524fe2` added multi-clip motion loading, forward kinematics for the tracked bodies and the `motion_end` termination, and `2cf7313` the `external` sampling mode. These are not in this repo:

- The G1 tracking task: `mjlab_research/src/mjlab/tasks/tracking/tracking_env_cfg.py:make_tracking_env_cfg` (rewards, events, terminations, 0.005 s timestep, decimation 4) and `mjlab_research/src/mjlab/tasks/tracking/config/g1/env_cfgs.py:unitree_g1_flat_tracking_env_cfg`.
- Motion data and reset sampling: `mjlab_research/src/mjlab/tasks/tracking/mdp/commands.py:MotionLoader` (clip boundaries in `motion_ends`, poses through `fk`) and `MotionCommand` (bins of 0.2 s, `set_sampling_distribution`, `_write_reference_state_to_sim`).
- Forward kinematics: `mjlab_research/src/mjlab/utils/forward_kinematics.py:ForwardKinematics`.
- Termination rules whose thresholds the world model reuses: `mjlab_research/src/mjlab/tasks/tracking/mdp/terminations.py:bad_anchor_pos_z_only` and `bad_anchor_ori`.
- Disturbances that exist but are unused here: `mjlab_research/src/mjlab/envs/mdp/events.py:push_by_setting_velocity`, `mjlab_research/src/mjlab/envs/mdp/dr/body.py:body_mass`, and actuator delay through `delay_min_lag` / `delay_max_lag` in `mjlab_research/src/mjlab/actuator/actuator.py` (backed by `mjlab_research/src/mjlab/utils/buffers/delay_buffer.py:DelayBuffer`).
- Model-free PPO for tracking: `mjlab_research/src/mjlab/scripts/train.py:run_train` with task `Mjlab-Tracking-Flat-Unitree-G1` and `mjlab_research/src/mjlab/tasks/tracking/rl/runner.py:MotionTrackingOnPolicyRunner`.

## Thesis ideas mapped to files

Status words: "exists" means the code is there, "partial" means a piece is there, "missing" means nothing in either repo does it.

### Idea 3: states vs Jacobians, plus the disturbance study

| What | Where | Status |
|---|---|---|
| Learned-model arm (SuperMimic) | `src/supermimic/learners/policy_learner.py:BpttPolicyLearner._train_batch` | exists |
| In-between arm: real-sim states, model Jacobians on fresh 32-step windows | New learner beside `_train_batch`; the window has to come from the real env as in `src/supermimic/runners/supermimic_runner.py:SuperMimicRunner.step`, then each step re-evaluated with `src/supermimic/envs/world_model_env.py:WorldModelEnv.step` and a straight-through state | missing |
| Replaying the sampled action noise | No longer applies: upstream `7e94b07` removed the collection noise, so the buffer's `action` field is the actor's own output | n/a |
| Noisy observation the actor saw | `ReplayBuffer` stores the clean `actor_obs` only when `_stores_actor_obs` is true; the noise from `StateRepresentation.add_proprio_noise` is not stored | missing |
| Exact-gradient arm | Nothing in either repo; would need NablaEngine | missing |
| Gradient agreement by contact phase | No contact labels in the buffer schema | missing |
| Pushes | `src/supermimic/tasks/env_cfgs.py:gencoord_env_cfg` drops every event except `foot_friction`, `base_com` and `encoder_bias`, which removes mjlab's `push_robot` | partial (mjlab has it) |
| Latency | mjlab actuator `delay_min_lag` / `delay_max_lag`; `src/supermimic/models.py:Dynamics.forward` sees only the current action, so action history must be added there, in `WorldModelLearner._rollout` and in `WorldModelEnv.step` | partial (sim side only) |
| Mass and torque-limit randomization as model inputs | Add a `SimContextSpec` subclass and list it in `src/supermimic/sim_context.py:SimContext.from_env`; mjlab has `mjlab_research/src/mjlab/envs/mdp/dr/body.py:body_mass` and `mjlab_research/src/mjlab/envs/mdp/dr/actuator.py:effort_limits` | partial (sim side only) |
| PPO reference | See "PPO baseline status" below | partial |

### Idea 4: reward prompting through the frozen model

| What | Where | Status |
|---|---|---|
| 32-dim tanh latent | `src/supermimic/models.py:MultiEncoderAE.forward` (`z = tanh(encoder(goal))`), size from `SuperMimicRunnerCfg.latent_dim` | exists |
| Decoder on its own | `MultiEncoderAE.decoder` is a plain `nn.Sequential`; you must apply `obs_normalizer` to the history yourself, as `forward` does | partial (no method) |
| Frozen world model rollout | `src/supermimic/envs/world_model_env.py:WorldModelEnv.step` with weights from `world_model.pt` (`src/supermimic/runners/supermimic_runner.py:WorldModel.load_bundle`) | exists |
| New reward | The reward is fixed in `src/supermimic/representations.py:StateRepresentation.compute_wm_reward_from_reference`, called from `WorldModelEnv.step`; a new reward means a subclass or a new hook there | missing |
| Termination without a reference | `WorldModelEnv._compute_term_prob` compares the root with the reference anchor, so it needs a motion | missing |
| Value function for new rewards | The critic input built in `WorldModelEnv.get_observations` includes the reference goal | missing |
| Trained velocity encoder baseline | `src/supermimic/tasks/task_spec.py:make_velocity_task`, enabled by `SuperMimicRunnerCfg.slot_mix` (velocity share is 0.0 by default) | exists |
| Oracle with true simulator rollouts | Needs state cloning, see below | missing |
| CEM, BFM-Zero adapter | Nothing | missing |

### Idea 1: tail failure rates

| What | Where | Status |
|---|---|---|
| Loading a trained actor and world model against the real sim | `src/supermimic/scripts/play.py:play` and `src/supermimic/scripts/benchmark.py:benchmark` both rebuild them with `build_world_model` and `build_actor` | exists |
| Batch Monte Carlo failure counter | Nothing in this repo. `mjlab_research/src/mjlab/tasks/tracking/scripts/evaluate.py:run_evaluate` computes tracking metrics but loads an rsl_rl PPO policy, not the SuperMimic actor | missing |
| Disturbance-sequence API (pushes, delay, payload per episode) | Nothing; mjlab events are random per interval or at startup | missing |
| Model-guided proposal | `WorldModelEnv.step` is differentiable in the action and the start state, so a disturbance written into `rep_state` could be optimized; nothing does this yet | missing |
| Mid-episode cloning for splitting | See state cloning below | missing |
| Matched training randomization | Both the SuperMimic tasks and the model-free PPO task would need the same `events` dict | missing |

### What survives of idea 2: action history in the world model

| What | Where | Status |
|---|---|---|
| Previous action already stored | `prev_action` field in `SuperMimicRunner._buffer_schema`; used only to seed `last_actions` in `WorldModelEnv.reset` | partial |
| Action history as a model input | `Dynamics.__init__` (MLP input size), `Dynamics.forward`, `WorldModelLearner._rollout`, `WorldModelEnv.step` | missing |
| Policy history | `SuperMimicRunnerCfg.history_length` (default 1; above 1 needs `actor_obs="proprio"`, checked in `build_world_model`) | exists |
| Freezing the world model | No flag. `SuperMimicRunner.run` always calls `WorldModelLearner.update`. Warm start exists through `SuperMimicRunnerCfg.wandb_run_path` and `load_wandb_checkpoint` | missing |
| Saving the replay buffer to disk | `ReplayBuffer._storage` is a GPU TensorDict with no save or load | missing |

### Shared engineering gaps

State cloning. The official code has no equivalent of the reproduction's `state_io.py`. The closest pieces are `Entity.write_root_state_to_sim` and `Entity.write_joint_state_to_sim` in `mjlab_research/src/mjlab/entity/entity.py`, which `MotionCommand._write_reference_state_to_sim` uses on reset. Exact cloning would also need the actuator delay buffer, the last action in the action manager, the motion frame, and the obs history ring (`SuperMimicRunner._sim_history`). I did not check whether mjlab restores all of these on a partial write.

PPO baseline status. Upstream removed `train_ppo.py` and `PpoRunner` in `deaf759`. They ran PPO on rollouts inside the world model, so they were never a model-free baseline. A model-free PPO tracker exists only in mjlab (`Mjlab-Tracking-Flat-Unitree-G1`). That task keeps `push_robot`, uses mjlab's own actor observations and does not drop the self-collision reward, so it is not matched to the SuperMimic task out of the box. I did not check whether mjlab's `train.py` runs on the SuperMimic task ids.

## Differences from the reproduction that affect earlier notes

The thesis note was written against `SuperMimic-Reproduced`. These points change when moving to this code.

- The note names `policy/nets.py` for the latent. Here it is `src/supermimic/models.py:MultiEncoderAE`, with one encoder per task and a shared decoder.
- The note names `state_io.py`. Nothing like it exists here; see state cloning above.
- The reproduction's world model state was the raw 71-dim `[qpos, qvel]`. Here it is one of the two representations above, with velocity-delta prediction and analytic integration.
- The reproduction trained a termination head. Here termination is analytic: `WorldModelEnv._compute_term_prob`, a sigmoid with softness `SuperMimicRunnerCfg.termination_softness` (0.05) over the real threshold rules.
- The note says the policy gets 10 frames of history. That was the reproduction's default. Here `history_length` defaults to 1 and `actor_obs` defaults to `rep_state`, so by default the actor sees the world model's privileged state rather than the deployable proprio sensor. ONNX export is skipped in that mode.
- The note says the official repo "ships `train_ppo.py`" as a PPO baseline. It was PPO in imagination, and upstream has removed it.
- Pushes are removed in both codebases. Here the cut is in `src/supermimic/tasks/env_cfgs.py:gencoord_env_cfg`.
- Reset sampling here is return-based (`ReturnSampler`), not mjlab's failure-based adaptive mode.
- The logged `Policy/reward` (`BpttPolicyLearner._update_from`) is the imagined return divided by horizon and `dt`, and in actor-critic mode that return includes the critic bootstrap. It is not the same quantity as the reproduction's returns (0.930, 0.82), so those numbers cannot be compared directly. A comparable number needs a real-sim evaluation with a fixed definition.
