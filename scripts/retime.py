"""Play an exported throw slower or faster: the quick aggressiveness knob for robot tests (no retraining).

    uv run python scripts/retime.py exports/<run>/throw_<ref>.csv --speed 0.7     # 70 % speed
    -> exports/<run>/throw_<ref>_speed0.70.csv (+ .json for release_gripper.py)

Only the timestamps change (the joint path is identical), so joint speeds scale by --speed and joint
accelerations by --speed^2. The gripper opens at the same row (release_gripper.py tracks the arm), but its
~0.18 s opening delay is fixed: at lower speed the disc leaves a little earlier along the path. Prints the
--max-joint-vel cartesian_follow.py needs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def spline_peak(q, t):
    """Peak joint speed of the controller's monotone cubic (same as cartesian_follow.py)."""
    d = np.diff(q, axis=0) / np.diff(t)[:, None]
    same = d[:-1] * d[1:] > 0
    v = np.zeros_like(q)
    v[1:-1] = np.where(same, 2 * d[:-1] * d[1:] / (d[:-1] + d[1:] + ~same), 0.0)
    s = np.linspace(0, 1, 21)[:, None, None]
    h = np.diff(t)[None, :, None]
    vv = (6 * s ** 2 - 6 * s) * (q[:-1] - q[1:]) / h + (3 * s ** 2 - 4 * s + 1) * v[:-1] + (3 * s ** 2 - 2 * s) * v[1:]
    return np.abs(vv).max(axis=(0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--speed", type=float, required=True, help="playback speed factor (1 = as exported)")
    a = ap.parse_args()
    rows = np.loadtxt(a.csv, delimiter=",", comments="#")
    t, q = rows[:, 0] / a.speed, rows[:, 1:]
    meta_p = a.csv.with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    peak = spline_peak(q, t)
    need = float(np.ceil(peak.max() * 10) / 10)
    out = a.csv.with_name(f"{a.csv.stem}_speed{a.speed:.2f}.csv")
    k = meta.get("gripper_open_index")
    hdr = (f"{a.csv.name} re-timed to {a.speed:.2f}x speed (same joint path)\n"
           f"t,joint_ra0..joint_ra7 (s, rad): python3 cartesian_follow.py --arm right --joints FILE --max-joint-vel {need}\n"
           f"controller spline peak {peak.max():.2f} rad/s" + (f"; gripper opens at row {k} (t={t[k]:.3f}s)" if k is not None else ""))
    np.savetxt(out, np.c_[t, q], delimiter=",", fmt="%.6f", header=hdr, comments="# ")
    if meta:
        meta.update({"retimed_speed": a.speed, "time_stretch": meta.get("time_stretch", 1.0) / a.speed,
                     "spline_peak_speed_rad_s": peak.round(3).tolist(), "max_joint_vel_needed": need})
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: duration {t[-1]:.2f}s, spline peak {peak.max():.2f} rad/s -> use --max-joint-vel {need}")


if __name__ == "__main__":
    main()
