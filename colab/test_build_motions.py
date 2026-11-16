import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_motions import convert  # noqa: E402


def test_missing_csvs_exits_nonzero(tmp_path):
  r = subprocess.run(
    [sys.executable, "colab/build_motions.py", "--csv-dir", str(tmp_path),
     "--out-dir", str(tmp_path / "o")],
    capture_output=True, text=True,
  )  # fmt: skip
  assert r.returncode != 0 and str(tmp_path) in r.stderr + r.stdout


def test_convert_writes_loader_fields(tmp_path):
  # 36 columns = root pos 3 + quat xyzw 4 + 29 joints, 60 identical frames at 30 fps
  row = [0, 0, 0.8, 0, 0, 0, 1] + [0.0] * 29
  csv = tmp_path / "clip.csv"
  np.savetxt(csv, np.tile(row, (60, 1)), delimiter=",")
  convert(csv, tmp_path / "clip.npz", "cpu")
  d = np.load(tmp_path / "clip.npz")
  assert {"fps", "joint_names", "joint_pos", "joint_vel", "root_pos_w",
          "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"} <= set(d.files)  # fmt: skip
  assert d["fps"][0] == 50 and d["joint_pos"].shape[1] == 29
