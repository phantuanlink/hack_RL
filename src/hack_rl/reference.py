"""Reference throws: joint trajectories retargeted from the phone videos (HEADBands retarget.py).

Files are `# comment` lines (one says "release at t=3.341s here") then rows `t, joint_ra0..joint_ra7`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"
# Forward throws, adapted for a flat grip by `python -m hack_rl.prepare_references`. The first video's
# throw ("adapted/joints_trajectory_shoulder_rel.csv") sweeps sideways; add it for variety if wanted.
DEFAULT_FILES = ("adapted/joints_trajectory_shoulder_rel_2.csv", "adapted/joints_trajectory_shoulder_rel_3.csv")


@dataclass
class Reference:
    name: str
    t: np.ndarray          # (N,) seconds, uniform
    q: np.ndarray          # (N, n_joints) rad
    release_t: float       # seconds into the reference


def load_reference(path, dt: float = 0.01) -> Reference:
    path = Path(path)
    text = path.read_text()
    m = re.search(r"release at t=([\d.]+)s", text)
    rows = np.loadtxt(path, delimiter=",", comments="#")
    t_raw, q_raw = rows[:, 0], rows[:, 1:]
    t = np.arange(0.0, t_raw[-1] + 1e-9, dt)
    q = np.stack([np.interp(t, t_raw, q_raw[:, j]) for j in range(q_raw.shape[1])], 1)
    release = float(m.group(1)) if m else 0.8 * t_raw[-1]
    return Reference(path.stem, t, q, release)


def load_references(files=DEFAULT_FILES, data_dir=DATA_DIR, dt: float = 0.01):
    """Stacked, padded arrays for batched lookup: q (R, Nmax, n_joints), length_s (R,), release_s (R,)."""
    refs = [load_reference(Path(data_dir) / f, dt) for f in files]
    n = max(len(r.t) for r in refs)
    q = np.stack([np.concatenate([r.q, np.repeat(r.q[-1:], n - len(r.t), 0)]) for r in refs])
    length = np.array([r.t[-1] for r in refs])
    release = np.array([r.release_t for r in refs])
    return refs, q, length, release, dt
