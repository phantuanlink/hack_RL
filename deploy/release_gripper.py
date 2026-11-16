#!/usr/bin/env python3
"""Open the right gripper at the policy's release point while cartesian_follow.py plays the throw.

cartesian_follow.py (HEADBands) moves the arm through an exported throw_*.csv but does not touch the
gripper, and it adds an automatically timed approach move, so the release can't be triggered by a clock.
This node watches /joint_states, tracks how far along the CSV the right arm is (nearest row, searched
forward only) and publishes "open" once the arm reaches the row the policy opened the gripper at.

On a machine with ROS 2 and unix_msgs (source the controller overlay first, see HEADBands docs/ros2):
    python3 release_gripper.py throw_X.csv --close        # close on the frisbee, confirm, then arm the trigger
    python3 cartesian_follow.py --arm right --joints throw_X.csv          # in another terminal
    python3 release_gripper.py throw_X.csv --dry-run      # only print when it would open

Safety: it only ever sends gripper commands (never arm motion). Stand clear of the throw direction.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

JOINTS = [f"joint_ra{i}" for i in range(8)]


def load(csv: Path):
    text = csv.read_text()
    rows = np.loadtxt(csv, delimiter=",", comments="#")
    meta = csv.with_suffix(".json")
    if meta.exists():
        k_open = int(json.loads(meta.read_text())["gripper_open_index"])
    else:
        m = re.search(r"\(row (\d+),", text)
        if not m:
            sys.exit(f"{csv}: no gripper-open row in the header and no {meta.name}")
        k_open = int(m.group(1))
    return rows[:, 0], rows[:, 1:9], k_open


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--close", action="store_true", help="close the gripper first and wait for Enter")
    ap.add_argument("--dry-run", action="store_true", help="never publish, only report")
    ap.add_argument("--lead-rows", type=int, default=0,
                    help="open this many rows (x20 ms) early to compensate a slow gripper")
    ap.add_argument("--velocity", type=float, default=1.0,
                    help="gripper speed (command.velocity): 1.0 is the fastest (2.0 is no faster); measured 0.18 s to 30 %% open")
    ap.add_argument("--close-position", type=float, default=0.7,
                    help="gripper target when closing on the frisbee (0.7 with the flat pads fitted; 1.0 = fully closed)")
    ap.add_argument("--effort", type=float, help="gripper force (command.effort); default setting reads 0.5")
    ap.add_argument("--start-tol", type=float, default=0.02, help="rad: arm counts as 'at the start pose' (starts the clock)")
    ap.add_argument("--track-tol", type=float, default=0.5,
                    help="rad: max distance (arm joints ra0-ra4) to the nearest row for the tracker to advance")
    ap.add_argument("--track", action="store_true", help="find the release by tracking the arm pose along the file")
    ap.add_argument("--progress", type=float, nargs="?", const=-1.0, default=None,
                    help="release by PATH PROGRESS along the stroke (0 = deepest backswing, 1 = planned release point), "
                         "from the .json written by scripts/release_progress.py; no value = its default (gripper lead "
                         "compensated). Larger = later. Independent of the policy's own release timing")
    ap.add_argument("--time-fallback", type=float, default=3.0,
                    help="open anyway once (planned release time x this) has passed since the start pose (0 = off)")
    a = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    try:
        from unix_msgs.msg import GripperCommand
    except ImportError:
        sys.exit("unix_msgs not found: source the robot controller overlay (HEADBands docs/ros2/README.md)")

    t, Q, k_open = load(a.csv)
    # where the file starts moving (after any initial hold at the start pose)
    moving = np.abs(Q - Q[0]).max(axis=1) > 0.01
    k_on = int(np.argmax(moving)) if moving.any() else len(Q) - 1
    k_trigger = max(0, k_open - a.lead_rows)
    # default: the file's own clock from the start pose (the arm kept the file's timing to ~3 % in every test up to
    # 2 rad/s; tracking poses fails when mostly the wrist moves). --track: follow the arm along the file instead
    # (use if the robot stretches the timing)
    by_time = not a.track and a.progress is None
    meta_all = json.loads(a.csv.with_suffix(".json").read_text()) if a.csv.with_suffix(".json").exists() else {}
    prog = None
    if a.progress is not None:
        if "progress" not in meta_all:
            sys.exit(f"{a.csv.with_suffix('.json').name} has no progress table: run scripts/release_progress.py first")
        prog = np.array(meta_all["progress"])
        p_target = meta_all["progress_default"] if a.progress < 0 else a.progress
        k_peak = int(meta_all["progress_peak_row"])
    print(f"{a.csv.name}: {len(t)} rows, open gripper at row {k_open} (t={t[k_open]:.2f}s), trigger row {k_trigger}; "
          f"file starts moving at row {k_on} -> " + (f"opening at PATH PROGRESS {p_target:.2f} (peak row {k_peak})"
                                                     if prog is not None else
                                                     "opening BY TIME from the start pose" if by_time else "tracking the arm along the file"))

    class Release(Node):
        def __init__(self):
            super().__init__("throw_release_gripper")
            self.pub = self.create_publisher(GripperCommand, "/right_gripper/gripper_command", 10)
            self.sub = self.create_subscription(JointState, "/joint_states", self.on_js, 50)
            self.started = False
            self.k = 0
            self.done = False
            self.t_start = None

        def command(self, value: float):
            msg = GripperCommand()
            msg.mode = "parallel"
            msg.command.name = [""]
            msg.command.position = [float(value)]
            if a.velocity is not None:
                msg.command.velocity = [float(a.velocity)]
            if a.effort is not None:
                msg.command.effort = [float(a.effort)]
            if a.dry_run:
                self.get_logger().info(f"[dry-run] would send gripper {value}")
            else:
                self.pub.publish(msg)

        def on_js(self, js: JointState):
            if self.done:
                return
            idx = {n: i for i, n in enumerate(js.name)}
            if not all(j in idx for j in JOINTS):
                return
            q = np.array([js.position[idx[j]] for j in JOINTS])
            if not self.started:
                d0 = np.abs(q - Q[0]).max()
                if d0 < a.start_tol:
                    self.started = True
                    self.t_start = time.time()
                    self.last_log = 0.0
                    self.get_logger().info("arm at the throw's start pose: tracking")
                elif time.time() - getattr(self, "last_wait_log", 0.0) > 2.0:
                    self.last_wait_log = time.time()
                    self.get_logger().info(f"waiting for the start pose: {d0:.3f} rad away (need < {a.start_tol})")
                return
            elapsed = time.time() - self.t_start
            if prog is not None:                         # path progress, interpolated between rows (arm joints)
                win = Q[self.k:min(len(Q), self.k + 25), :5]
                d_all = np.sqrt(((win - q[:5]) ** 2).mean(axis=1))
                j = int(np.argmin(d_all))
                if d_all[j] < a.track_tol:
                    self.k += j
                k1 = min(self.k + 1, len(Q) - 1)
                seg = Q[k1, :5] - Q[self.k, :5]
                frac = float(np.clip((q[:5] - Q[self.k, :5]) @ seg / max(seg @ seg, 1e-12), 0.0, 1.0))
                p_now = prog[self.k] + frac * (prog[k1] - prog[self.k])
                if elapsed - self.last_log > 0.05:
                    self.last_log = elapsed
                    self.get_logger().info(f"t={elapsed:.2f}s row {self.k} progress {p_now:+.2f} (open at {p_target:.2f})")
                late = a.time_fallback > 0 and elapsed > a.time_fallback * (t[k_peak] - t[0])
                if p_now >= p_target or late:
                    self.command(0.0)
                    why = "TIME FALLBACK" if p_now < p_target else f"progress {p_now:.2f} >= {p_target:.2f}"
                    self.get_logger().info(f"OPEN ({why}): row {self.k}, {elapsed:.2f}s after the start pose")
                    self.done = True
                return
            if by_time:                                  # the file's own clock from the start pose
                self.k = int(np.searchsorted(t - t[0], elapsed))
            else:                                        # nearest row (RMS over all joints), forward only
                win = Q[self.k:min(len(Q), self.k + 25)]
                d_all = np.sqrt(((win - q) ** 2).mean(axis=1))
                j = int(np.argmin(d_all))
                if d_all[j] < a.track_tol:
                    self.k += j
            d = [0.0] if by_time else [float(d_all.min())]
            j = 0
            if elapsed - self.last_log > 0.25:
                self.last_log = elapsed
                self.get_logger().info(f"t={elapsed:.2f}s row {self.k}/{k_trigger} (nearest {d[j]:.3f} rad)")
            late = a.time_fallback > 0 and elapsed > a.time_fallback * (t[k_trigger] - t[0])
            if self.k >= k_trigger or late:
                self.command(0.0)
                why = "TIME FALLBACK (tracking stalled)" if self.k < k_trigger else "at the release row"
                self.get_logger().info(f"OPEN {why}: row {self.k} (target {k_open}), {elapsed:.2f}s after the start pose")
                self.done = True

    rclpy.init()
    node = Release()
    if a.close:
        node.command(a.close_position)
        input("Gripper closed. Check the frisbee is held by its rim, step back, then press Enter to arm the trigger...")
    node.get_logger().info("waiting for the arm to reach the throw's start pose (run cartesian_follow.py now)")
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
