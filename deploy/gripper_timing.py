#!/usr/bin/env python3
"""Measure how fast the right gripper opens (sysid_params.yaml gripper.open_time_s). ROS 2 Humble.

    python3 gripper_timing.py                       # close, wait, open; repeat 3 times; prints the timings
    python3 gripper_timing.py --velocity 1.0        # with an explicit gripper speed

Hold the frisbee in the gripper as for a throw (or an empty gripper, as a lower bound). For each cycle:
the time from the open command until the finger joint first moves (latency), and until it has moved 30 %
of the way to open (the disc is free well before fully open). Also logs to gripper_timing.csv.
"""
import argparse
import csv
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from unix_msgs.msg import GripperCommand


class Timing(Node):
    def __init__(self, velocity=None, effort=None):
        super().__init__("gripper_timing")
        self.velocity, self.effort = velocity, effort
        self.pub = self.create_publisher(GripperCommand, "/right_gripper/gripper_command", 10)
        self.create_subscription(JointState, "/right_gripper/joint_states", self.on_js, 50)
        self.samples = []                       # (wall time, position)

    def on_js(self, js):
        # the gripper sometimes reports -32.767 (int16 min / 1000): an invalid reading, not a position
        if js.position and -0.2 <= js.position[0] <= 1.5:
            self.samples.append((time.monotonic(), float(js.position[0])))

    def command(self, value):
        msg = GripperCommand()
        msg.mode = "parallel"
        msg.command.name = [""]
        msg.command.position = [float(value)]
        if self.velocity is not None:
            msg.command.velocity = [float(self.velocity)]
        if self.effort is not None:
            msg.command.effort = [float(self.effort)]
        self.pub.publish(msg)

    def spin_for(self, s):
        end = time.monotonic() + s
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.005)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--velocity", type=float, help="command.velocity (empty = the gripper's default, slow)")
    ap.add_argument("--effort", type=float, help="command.effort (the default setting reads 0.5)")
    ap.add_argument("--hold", type=float, default=2.0, help="s closed / open per cycle")
    a = ap.parse_args()
    rclpy.init()
    n = Timing(a.velocity, a.effort)
    print(f"velocity {a.velocity if a.velocity is not None else 'default'}, effort {a.effort if a.effort is not None else 'default'}")
    n.spin_for(1.0)
    if not n.samples:
        raise SystemExit("no /right_gripper/joint_states: wrong ROS_DOMAIN_ID or RMW?")
    rows = []
    for c in range(a.cycles):
        n.command(1.0)
        n.spin_for(a.hold)
        closed = n.samples[-1][1]
        start = len(n.samples)
        t0 = time.monotonic()
        n.command(0.0)
        n.spin_for(a.hold)
        after = n.samples[start:]
        opened = after[-1][1] if after else closed
        span = opened - closed
        if abs(span) < 1e-4:
            print(f"cycle {c}: the gripper did not move (closed {closed:.4f}, after {opened:.4f})")
            continue
        frac = [(t - t0, (p - closed) / span) for t, p in after]
        first = next((t for t, f in frac if f > 0.02), float("nan"))
        part = next((t for t, f in frac if f > 0.30), float("nan"))
        full = next((t for t, f in frac if f > 0.95), float("nan"))
        print(f"cycle {c}: closed {closed:.4f} -> open {opened:.4f}; moves after {first * 1000:.0f} ms, "
              f"30 % open {part * 1000:.0f} ms, 95 % open {full * 1000:.0f} ms")
        rows.append((c, closed, opened, first, part, full))
    with open("gripper_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "closed_pos", "open_pos", "t_first_move_s", "t_30pct_s", "t_95pct_s"])
        w.writerows(rows)
    if rows:
        part = sorted(r[4] for r in rows)[len(rows) // 2]
        print(f"-> sysid_params.yaml gripper.open_time_s ~ {part:.3f} (median time to 30 % open)")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
