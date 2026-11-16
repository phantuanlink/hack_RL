"""Mirror a TensorBoard run (logs/rsl_rl/...) to Weights & Biases, optionally live while it trains.

    uv run python scripts/tb_to_wandb.py logs/rsl_rl/panther_frisbee_throw/<run> [--follow] [--name NAME]

wandb 0.31 dropped `wandb sync --sync-tensorboard`. This reads the event files and logs every scalar at
its iteration; re-running resumes the same wandb run and only sends iterations it has not sent yet.
--follow keeps polling (every --every s) until no new data arrives for --idle s (the run has ended).
For new runs, `--agent.logger wandb` logs directly instead.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read(run: Path):
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    rows: dict[int, dict] = {}
    for tag in ea.Tags()["scalars"]:
        if tag.endswith("/time"):          # RSL-RL also logs these against wall-clock seconds, not iterations
            continue
        for ev in ea.Scalars(tag):
            rows.setdefault(int(ev.step), {})[tag] = float(ev.value)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--project", default="hack_RL")
    ap.add_argument("--name", help="wandb run name/id (default: the run folder name)")
    ap.add_argument("--id", help="wandb run id (default: from the name)")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--idle", type=float, default=600.0)
    a = ap.parse_args()
    name = a.name or a.run.name
    run_id = a.id or name.replace("_", "-")
    w = wandb.init(project=a.project, id=run_id, name=name, resume="allow",
                   config={"source": str(a.run), "mirrored_from": "tensorboard"})
    last = int(w.summary.get("tb_last_step", -1))
    quiet_since = time.time()
    while True:
        rows = read(a.run)
        new = sorted(s for s in rows if s > last)
        for s in new:
            w.log({**rows[s], "iteration": s}, step=s)
        if new:
            last = new[-1]
            w.summary["tb_last_step"] = last
            quiet_since = time.time()
            print(f"{name}: logged up to iteration {last}", flush=True)
        if not a.follow or time.time() - quiet_since > a.idle:
            break
        time.sleep(a.every)
    w.finish()


if __name__ == "__main__":
    main()
