"""Add a "progress along the stroke" table to a throw file, so the robot releases by path progress, not by the clock.

    uv run python scripts/release_progress.py exports/<...>/throw.csv [--lead 0.12]

For every row: the disc centre (forward kinematics of the commanded joints) projected on the release direction,
normalised so 0 = the deepest backswing and 1 = the planned release point (peak disc speed). Written into the
.json next to the CSV ("progress", "progress_peak_row", "progress_default"). release_gripper.py --progress P
then sends "open" when the measured arm reaches progress P. Default P: the progress --lead s (the gripper's
command-to-free time, ~0.12 s measured on the robot) before the planned peak.
Prints a table (time, progress, planned disc speed) to pick P from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hack_rl.prepare_references_v3 import Robot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--lead", type=float, default=0.12, help="s between the open command and the frisbee being free")
    a = ap.parse_args()
    rows = np.loadtxt(a.csv, delimiter=",", comments="#")
    t, Q = rows[:, 0], rows[:, 1:9]
    rb = Robot()
    P = np.array([rb.disc(q)[0] for q in Q])
    V = np.gradient(P, t, axis=0)
    sp = np.linalg.norm(V, axis=1)
    k_peak = int(np.argmax(sp[: int(0.8 * len(sp))]))
    d = V[k_peak, :2] / np.linalg.norm(V[k_peak, :2])
    proj = P[:, :2] @ d
    k_back = int(np.argmin(proj[: k_peak + 1]))
    prog = (proj - proj[k_back]) / (proj[k_peak] - proj[k_back])
    prog[:k_back] = 0.0
    t_def = t[k_peak] - a.lead
    k_def = int(np.clip(np.searchsorted(t, t_def), k_back, k_peak))
    meta_p = a.csv.with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.update({"progress": np.round(prog, 4).tolist(), "progress_peak_row": k_peak, "progress_backswing_row": k_back,
                 "progress_default": float(round(prog[k_def], 3)), "progress_lead_s": a.lead})
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"{a.csv.name}: backswing row {k_back} (t={t[k_back]:.2f}s), planned peak row {k_peak} (t={t[k_peak]:.2f}s, "
          f"{sp[k_peak]:.1f} m/s); default --progress {prog[k_def]:.2f} (= {a.lead:.2f}s before the peak)")
    print("   t (s)   progress   planned disc speed (m/s)")
    for k in range(k_back, min(len(t), k_peak + 4)):
        mark = "  <- default" if k == k_def else ("  <- peak" if k == k_peak else "")
        print(f"   {t[k]:5.2f}   {prog[k]:6.2f}     {sp[k]:5.1f}{mark}")


if __name__ == "__main__":
    main()
