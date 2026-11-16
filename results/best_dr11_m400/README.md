# BEST working throw (confirmed on the robot, 2026-11-15)
- Policy: dr11 model_400 (task Hack-Throw-Frisbee-Panther-Stroke-Pitch), copy: model_400.pt
- Commands: aggressiveness 1.0, pitch 10 deg, heading -60 (disc leaves ~67 deg right of the robot's front)
- Robot: cartesian_follow --max-joint-vel 12.0; release_gripper --close --close-position 1.0 --progress 0.08
- throw_best.csv/.json = the exact file thrown (with the release-progress table)
- Launch: ~/hack/throw.sh best
