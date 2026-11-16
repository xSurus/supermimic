# Official-base thesis workspace implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the official SuperMimic code installable, tested and runnable on Colab from Alexander's forks, and document what to read.

**Architecture:** Two sibling forks (`supermimic_ws/supermimic`, `supermimic_ws/mjlab_research`). `main` of each mirrors upstream. All additions live on the `thesis` branch of `supermimic` and touch no file under `src/` or `tests/`.

**Tech stack:** Python 3.13, uv, mjlab (from the sibling path), MuJoCo Warp, PyTorch, W&B, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-official-base-design.md`

## Global constraints

- No edits under `src/` or `tests/` in either repo.
- Nothing imported or copied from `SuperMimic-Reproduced`, including its `data/` npz files.
- `mjlab` resolves from `../mjlab_research` (official `pyproject.toml`, unchanged).
- Workspace root on the Mac: `/Users/alexanderschlieper/Documents/ETH/master/master thesis/supermimic_ws`.
- The Mac has no CUDA. Steps that need a GPU are written into the notebook and verified by Alexander on Colab.
- Prose in every added file follows `~/.claude/CLAUDE.md`: no em dashes, sentence-case headings, plain sentences.
- Commit on `thesis`, never on `main`. End commit messages with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Known facts (checked 2026-11-16)

- `uv lock --check` fails: the official `uv.lock` pins mjlab 1.5.0, torch 2.10.0, mujoco-warp 3.10.0.1, but `mjlab_research` HEAD (`dd51a9f`) is mjlab 1.6.0 and requires torch>=2.14.0, mujoco-warp~=3.16.0. `uv sync` re-resolves.
- Default motions come from the authors' W&B registry (`supermimic/wandb-registry-motions/lafan_run1_subject5`). `train.py --motion-file` accepts a local npz, a comma list or a glob and then skips the registry.
- `mjlab.scripts.csv_to_npz.MotionLoader(motion_file, input_fps, output_fps, line_range, device).arrays()` returns the npz dict. Its `main` always uploads to W&B, so call the class, not `main`.
- Task ids: `SuperMimic-GenCoord-G1`, `SuperMimic-MaxCoord-G1`. Runner flags are `--runner.<field>` (`max_iterations`, `save_interval`, `log_interval`, `run_name`, `wandb_project`).
- The runner always calls `wandb.init`, so `WANDB_API_KEY` must be set.
- LAFAN1 G1 CSVs: Hugging Face dataset `lvhaidong/LAFAN1_Retargeting_Dataset`, folder `g1/`, 30 fps.

## Review focus

1. Stale lock file: a fresh `uv sync` must give the same versions on the Mac and Colab. Task 1 commits the re-resolved `uv.lock`; the notebook runs `uv sync --locked`.
2. Colab NVIDIA driver too old for the resolved torch CUDA build: the notebook prints driver and `torch.cuda.is_available()` and asserts it is true before any run cell (Task 3).
3. `WANDB_API_KEY` secret missing: the notebook asserts it is set with a message naming the Colab secret (Task 3).
4. Motion directory empty or CSV download failed: `build_motions.py` exits non-zero with the missing path when it finds no CSVs (Task 2 test).
5. `torch.compile` warm-up makes a short run look hung on Colab: the smoke cell says so in a comment and the guide repeats it (Tasks 3 and 4).

---

### Task 1: Install and run the official tests on the Mac

**Files:**
- Modify: `uv.lock` (re-resolved, not hand-edited)
- Create: `docs/TEST_STATUS.md`

**Interfaces:**
- Produces: a committed `uv.lock` that `uv sync --locked` accepts; `docs/TEST_STATUS.md` with pass, fail and skip counts and resolved versions of torch, mujoco-warp, warp-lang, mjlab.

- [ ] **Step 1:** Run `uv sync` in `supermimic/`. Expected: completes, `git status --short` shows only `uv.lock` modified.
- [ ] **Step 2:** Run `uv lock --check`. Expected: exit 0.
- [ ] **Step 3:** Run `MUJOCO_GL=disable uv run pytest -q -rs 2>&1 | tail -40`. Record the counts. Expected: no failures. If any test fails, do not edit it. Check whether it fails for a platform reason (no CUDA, macOS) by reading the test, and record the test id and the reason.
- [ ] **Step 4:** Write `docs/TEST_STATUS.md`: date, machine (macOS, no CUDA), commit hashes of both repos, resolved versions (`uv pip list | grep -iE "^(torch|mujoco|warp|mjlab) "`), the pytest summary line, and any failure with its reason. No more than 30 lines.
- [ ] **Step 5:** Commit `uv.lock` and `docs/TEST_STATUS.md` with message "Re-resolve the lock against mjlab_research HEAD and record test status".

### Task 2: Build LAFAN1 motions with the official converter

**Files:**
- Create: `colab/build_motions.py`
- Test: `colab/test_build_motions.py`

**Interfaces:**
- Consumes: `mjlab.scripts.csv_to_npz.MotionLoader` (see Known facts).
- Produces: CLI `uv run python colab/build_motions.py --csv-dir DIR --out-dir DIR [--device cuda:0] [--limit N]`. Writes `<out-dir>/<csv stem>.npz` for every `*.csv`, skipping ones that exist. Function `convert(csv: Path, out: Path, device: str) -> None`.

- [ ] **Step 1: Write the failing tests** in `colab/test_build_motions.py`:

```python
def test_missing_csvs_exits_nonzero(tmp_path):
  r = subprocess.run([sys.executable, "colab/build_motions.py", "--csv-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")], capture_output=True, text=True)
  assert r.returncode != 0 and str(tmp_path) in r.stderr + r.stdout

def test_convert_writes_loader_fields(tmp_path):
  # 36 columns = root pos 3 + quat xyzw 4 + 29 joints, 60 identical frames at 30 fps
  row = [0, 0, 0.8, 0, 0, 0, 1] + [0.0] * 29
  csv = tmp_path / "clip.csv"; np.savetxt(csv, np.tile(row, (60, 1)), delimiter=",")
  convert(csv, tmp_path / "clip.npz", "cpu")
  d = np.load(tmp_path / "clip.npz")
  assert {"fps", "joint_names", "joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"} <= set(d.files)
  assert d["fps"][0] == 50 and d["joint_pos"].shape[1] == 29
```

- [ ] **Step 2:** Run `uv run pytest colab/test_build_motions.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `colab/build_motions.py` with `argparse` (already used by `pack_motions.py`). `convert` is `np.savez(out, **MotionLoader(motion_file=str(csv), input_fps=30.0, output_fps=50.0, line_range=None, device=device).arrays())`. Read `MotionLoader.__init__` first and match its real signature if it differs.
- [ ] **Step 4:** Run the tests. Expected: 2 passed. If the synthetic CSV does not match the column layout the loader expects, fix the test's row to the layout the loader reads, not the loader.
- [ ] **Step 5:** Commit: "Add a local LAFAN1 motion builder on the official converter".

### Task 3: Colab notebook

**Files:**
- Create: `colab/run.ipynb`

**Interfaces:**
- Consumes: `colab/build_motions.py` CLI (Task 2), the committed `uv.lock` (Task 1).
- Produces: cells, in order:
  1. GPU check: print `nvidia-smi` name, driver, memory; assert a GPU exists.
  2. Mount Drive. `STORE=/content/drive/MyDrive/supermimic-official`, `WS=/content/supermimic_ws`. Clone `xSurus/mjlab_research` (branch `main`) and `xSurus/supermimic` (branch `thesis`) as siblings over public HTTPS (no token), or `git pull --ff-only` if present. Symlink `WS/supermimic/data` to `STORE/data`.
  3. `pip install -q uv`, then `uv sync --locked --group dev` in `WS/supermimic`, then print torch version and assert `torch.cuda.is_available()`.
  4. W&B: read `WANDB_API_KEY` from `google.colab.userdata`, assert it is set with the message "Add your W&B API key to Colab Secrets as WANDB_API_KEY", export it to `os.environ`.
  5. Tests: `MUJOCO_GL=disable uv run pytest -q -rs`.
  6. Motions: `uv run --with huggingface_hub hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --include "g1/*" --local-dir data/raw/lafan1`, then `uv run python colab/build_motions.py --csv-dir data/raw/lafan1/g1 --out-dir data/npz/lafan1`.
  7. Smoke run: `uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --motion-file data/npz/lafan1/walk1_subject1.npz --num-envs 256 --runner.max-iterations 300 --runner.run-name colab-smoke`. Comment: the first update compiles and can sit silent for many minutes.
  8. Full run, commented out: same with `--motion-file 'data/npz/lafan1/*.npz'`, `--num-envs 4096`, no iteration cap.
  9. Play, commented out: `uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-G1 --wandb-path <entity>/supermimic/<run id>`.

- [ ] **Step 1:** Confirm the exact flag spellings with `uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --help | grep -E "num.envs|motion.file|max.iterations|run.name"` and `play.py ... --help | grep wandb`. Use what `--help` prints.
- [ ] **Step 2:** Write the notebook as valid nbformat 4 JSON with the cells above and a first markdown cell stating the one-time setup (GPU runtime, `WANDB_API_KEY` secret).
- [ ] **Step 3:** Verify: `uv run python -c "import json,sys; nb=json.load(open('colab/run.ipynb')); assert nb['nbformat']==4; print(len(nb['cells']))"`. Expected: prints 10 or more.
- [ ] **Step 4:** Commit: "Add the Colab runner notebook for the official code".

### Task 4: Reading guide

**Files:**
- Create: `docs/READING_GUIDE.md`

**Interfaces:**
- Consumes: the source tree as is; the four ideas in `/Users/alexanderschlieper/Documents/Obsidian Vault/Efforts/ETH/Masters/Master Thesis/Thesis Proposals - Four Ideas Ranked.md`.

- [ ] **Step 1:** Read every file under `src/supermimic/` and the mjlab_research tracking task (`src/mjlab/tasks/tracking/`). Trace one training iteration from `scripts/train.py` through `runners/supermimic_runner.py` into `learners/` and back.
- [ ] **Step 2:** Write the guide with these sections: how to run (pointer to the notebook), one training iteration as a numbered walk with `file:function` references, a reading order with line counts, what the two representations (gencoord, maxcoord) are, what lives in mjlab_research and not here, a table per thesis idea (idea 3, 4, 1, and the surviving part of 2) naming the files and functions each would change and what is missing (pushes, latency, mass randomization, state cloning, PPO baseline status), and differences from the reproduction that affect earlier notes. Every claim about the code cites a file and symbol that exists. Target 150 to 250 lines.
- [ ] **Step 3:** Verify references: for each `path:symbol` in the guide, `grep -n "symbol" path` finds it. Fix any that do not.
- [ ] **Step 4:** Commit: "Add a reading guide mapped to the thesis ideas".

### After all tasks

- [ ] Push `thesis` to `origin` so the notebook can clone it. Ask Alexander before pushing.
- [ ] Alexander runs the notebook on Colab; record the Colab test result in `docs/TEST_STATUS.md`.
