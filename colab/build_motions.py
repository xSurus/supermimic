"""Convert retargeted LAFAN1 CSVs to npz with the official converter, saved locally."""

import argparse
from pathlib import Path

import numpy as np
import torch
from mjlab.scripts.csv_to_npz import MotionLoader


def convert(csv: Path, out: Path, device: str) -> None:
  loader = MotionLoader(str(csv), input_fps=30.0, output_fps=50.0, device=device)
  np.savez(out, **loader.arrays())


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--csv-dir", type=Path, required=True)
  p.add_argument("--out-dir", type=Path, required=True)
  p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  p.add_argument("--limit", type=int, default=None)
  a = p.parse_args()
  csvs = sorted(a.csv_dir.glob("*.csv"))[: a.limit]
  if not csvs:
    raise SystemExit(f"No CSV files found in {a.csv_dir}")
  a.out_dir.mkdir(parents=True, exist_ok=True)
  for csv in csvs:
    out = a.out_dir / f"{csv.stem}.npz"
    if not out.exists():
      convert(csv, out, a.device)
