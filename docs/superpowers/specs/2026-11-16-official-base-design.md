# Thesis codebase on the official SuperMimic code

Approved by Alexander on 2026-11-16.

## Intent

The thesis so far ran on a third-party reproduction (CuboiLeo/SuperMimic-Reproduced). The authors released the official code on 2026-11-10. The thesis moves to the official code so results are not affected by reproduction differences.

## Goals

- A workspace that follows the official layout: `supermimic_ws/supermimic` and `supermimic_ws/mjlab_research` as siblings, both forks under xSurus.
- The official test suite passes locally and on Colab.
- A short training run completes on Colab with local LAFAN1 motions and W&B logging.
- `docs/READING_GUIDE.md`: what to read, in what order, and which files each of the four thesis ideas touches.

## Non-goals

- No thesis experiments yet (disturbance API, in-between estimator, reward prompting).
- No port of the reproduction's push test or video renderer.
- No changes to the official source under `src/` or `tests/`.

## Anti-goals

- Importing or copying code or data from the reproduction. Motions are rebuilt from the raw LAFAN1 G1 CSVs with mjlab_research's own converter.

## Constraints

- Python 3.13. The official `uv.lock` is stale against `mjlab_research` HEAD, so the lock is re-resolved, with torch, warp-lang and mjviser pinned to the versions in `../mjlab_research/uv.lock`.
- `mjlab` resolves from the sibling path `../mjlab_research`.
- The Mac has no CUDA, so anything that needs a GPU is verified on Colab by Alexander.
- Training logs and checkpoints go through W&B. Alexander has an account.

## Approaches

1. Fork both official repos, keep `main` identical to upstream, put additions on a `thesis` branch. Chosen: upstream updates merge cleanly and the diff against the official code is always visible.
2. Replace the contents of the existing `xSurus/supermimic-thesis` repo. Rejected: its history is the reproduction, which is what we are leaving.
3. Vendor the official code into a new repo without fork history. Rejected: loses the upstream link for no gain.

## What gets added

- `colab/run.ipynb`: clone both forks as siblings, `uv sync`, run tests, build motions, launch training.
- `docs/READING_GUIDE.md`.
- This design doc.

## Left to the builder

- How to fetch the LAFAN1 CSVs and which mjlab_research script converts them.
- Whether tests need a GPU, and how `torch.compile` cost is handled for short runs on Colab.
- The exact W&B settings for a smoke run.
