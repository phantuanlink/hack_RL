"""Watch a training run: for every new checkpoint, export the throws and render videos (+ log them to wandb).

    uv run python scripts/watch_checkpoints.py --task Hack-Throw-Frisbee-Panther-Safe \
        --run logs/rsl_rl/panther_frisbee_throw/<run> [--wandb panther-dr2a-videos]

Writes exports/<run>/model_<N>/throw_<ref>.csv/json/png/mp4 for N = 100, 200, ... as they appear, and a
line per checkpoint to exports/<run>/progress.txt (release score, speed, spin, peak joint speed). Stops
--idle s after the last checkpoint (training over).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--wandb", help="wandb run id for the videos (project hack_RL); omit for no wandb")
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--idle", type=float, default=900.0)
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else ROOT / a.run
    out_root = ROOT / "exports" / run.name
    out_root.mkdir(parents=True, exist_ok=True)
    w = None
    if a.wandb:
        import wandb
        w = wandb.init(project="hack_RL", id=a.wandb, name=a.wandb, resume="allow",
                       config={"task": a.task, "run": str(run)})
    done = {p.name for p in out_root.glob("model_*") if (p / "done").exists()}
    last_new = time.time()
    while True:
        ckpts = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        for ck in ckpts:
            if ck.stem in done:
                continue
            time.sleep(5)                                        # let the trainer finish writing it
            out = out_root / ck.stem
            r = subprocess.run([sys.executable, "-m", "hack_rl.export_rollout", "--task", a.task,
                                "--checkpoint", str(ck), "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{ck.stem}: export failed\n{r.stderr[-2000:]}", flush=True)
                continue
            line = [ck.stem]
            videos = {}
            for csv in sorted(out.glob("throw_*.csv")):
                mp4 = csv.with_suffix(".mp4")
                subprocess.run([sys.executable, "scripts/render_throw.py", str(csv), "--out", str(mp4)], cwd=ROOT,
                               capture_output=True, text=True)
                meta = json.loads(csv.with_suffix(".json").read_text())
                c = meta.get("components", {})
                ref = csv.stem.replace("throw_joints_trajectory_shoulder_rel_", "ref")
                line.append(f"{ref}: q {meta.get('quality', float('nan')):.2f} v {c.get('v_forward_mps', float('nan')):.2f} "
                            f"spin {c.get('spin_radps', float('nan')):.1f} flat {c.get('flat', float('nan')):.2f} "
                            f"peak {max(meta['spline_peak_speed_rad_s']):.2f} rad/s "
                            f"{'OK' if meta['passes_cartesian_follow_speed_check'] else 'TOO FAST'}")
                if mp4.exists():
                    videos[ref] = mp4
                    if w is not None:
                        import wandb
                        it = int(ck.stem.split("_")[1])
                        w.log({f"video/{ref}": wandb.Video(str(mp4), format="mp4"),
                               f"export/{ref}/quality": meta.get("quality"),
                               f"export/{ref}/v_forward_mps": c.get("v_forward_mps"),
                               f"export/{ref}/spin_radps": c.get("spin_radps"),
                               "iteration": it}, step=it)
            msg = " | ".join(line)
            print(msg, flush=True)
            with open(out_root / "progress.txt", "a") as f:
                f.write(msg + "\n")
            (out / "done").write_text("")
            done.add(ck.stem)
            last_new = time.time()
        if time.time() - last_new > a.idle:
            break
        time.sleep(a.every)
    if w is not None:
        w.finish()


if __name__ == "__main__":
    main()
