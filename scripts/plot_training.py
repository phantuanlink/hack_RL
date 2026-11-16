"""Training curves of a run (from its TensorBoard events) -> PNG.

    uv run python scripts/plot_training.py [logs/rsl_rl/panther_frisbee_throw/<run>] [--out curves.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
PANELS = [
    ("Release quality (0-1)", ["Release/quality"]),
    ("Forward speed at release (m/s)", ["Release/v_forward_mps"]),
    ("Spin about disc normal (rad/s)", ["Release/spin_radps"]),
    ("Release components", ["Release/flat", "Release/direction", "Release/no_wobble", "Release/aoa",
                            "Release/elevation"]),
    ("Episode reward", ["Train/mean_reward"]),
    ("Penalties", ["Episode_Reward/body_contact", "Episode_Reward/joint_vel_excess",
                   "Episode_Reward/joint_limits"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.run:
        run = Path(a.run)
    else:
        runs = sorted((ROOT / "logs" / "rsl_rl" / "panther_frisbee_throw").glob("*"))
        run = [r for r in runs if list(r.glob("events.*"))][-1]
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (title, keys) in zip(axes.ravel(), PANELS):
        for k in keys:
            cand = [t for t in tags if t == k or t.endswith("/" + k.split("/", 1)[-1]) and k.split("/")[0] in t]
            tag = k if k in tags else (cand[0] if cand else None)
            if tag is None:
                continue
            ev = ea.Scalars(tag)
            ax.plot([e.step for e in ev], [e.value for e in ev], label=k.split("/")[-1])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        if len(keys) > 1:
            ax.legend(fontsize=7)
    fig.suptitle(f"{run.name}: PPO on Hack-Throw-Frisbee-Panther")
    fig.tight_layout()
    out = Path(a.out) if a.out else run / "training_curves.png"
    fig.savefig(out, dpi=90)
    print("wrote", out, "| available tags:", len(tags))


if __name__ == "__main__":
    main()
