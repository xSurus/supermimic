# Test status

Date: 2026-11-16
Machine: macOS (Darwin 27.2.0), Apple Silicon arm64, no CUDA, Python 3.13.5.
Commits: supermimic 1b5d70e (branch thesis) plus the pin commit 0c300f2, mjlab_research dd51a9f.

## Resolved versions

- torch 2.14.0
- mujoco-warp 3.16.0
- warp-lang 1.18.0
- mjlab 1.6.0 (editable, ../mjlab_research)
- mujoco 3.16.0, mjviser 0.0.16 (no mujoco-core)

pyproject.toml pins torch, warp-lang and mjviser to the versions in ../mjlab_research/uv.lock. A free resolve picks mjviser 0.0.17, which pulls mujoco-core 4.x over mujoco 3.16 and breaks `import mujoco` on macOS.

## Result

Command: `MUJOCO_GL=disable uv run pytest -q -rs`
Summary: `1 failed, 216 passed, 1 warning in 2.35s` (no skips).

## Failure

tests/test_multi_task_world_model_env.py::TestRepresentationApiContract::test_env_update_goals_against_real_representation

Reason: platform. `torch.compile` (used in src/supermimic/representations.py) needs a C++ compiler on CPU. Inductor raises InvalidCxxCompiler because clang++ is unusable until the Xcode license is accepted. To fix it, run `sudo xcodebuild -license` yourself in a terminal. Not a code bug.
