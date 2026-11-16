# Test status

Date: 2026-11-16
Machine: macOS (Darwin 27.2.0), Apple Silicon arm64, no CUDA, Python 3.13.5.
Commits: supermimic 1b5d70e (branch thesis), mjlab_research dd51a9f.

## Resolved versions

- torch 2.15.0
- mujoco-warp 3.16.0
- warp-lang 1.19.0
- mjlab 1.6.0 (editable, ../mjlab_research)
- mujoco 3.16.0, mujoco-core 4.0.1

## Result

`1 failed, 216 passed, 1 warning in 1.97s` (no skips).
Command: `MUJOCO_GL=disable python -m pytest -q -rs`, with the workaround below.

## Failure

tests/test_multi_task_world_model_env.py::TestRepresentationApiContract::test_env_update_goals_against_real_representation

Reason: platform. `torch.compile` (used in src/supermimic/representations.py) needs a C++ compiler on CPU. Inductor raises InvalidCxxCompiler because clang++ is unusable until the Xcode license is accepted (`sudo xcodebuild -license`). Not a code bug.

## Workaround needed to import mujoco

The mujoco 3.16.0 macOS wheel ships plugin/libelasticity.dylib that points at @rpath/mujoco.framework/... and has no LC_RPATH. `import mujoco` fails with OSError, and so does `pytest` as the brief writes it. This passes:

`DYLD_LIBRARY_PATH=<repo>/.venv/lib/python3.13/site-packages/mujoco .venv/bin/python -m pytest`

`uv run` and the `pytest` script drop DYLD_LIBRARY_PATH, so call python directly.
