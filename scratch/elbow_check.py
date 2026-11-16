import sys, numpy as np
from hack_rl.prepare_references_v3 import Robot
rb = Robot(); rows = np.loadtxt(sys.argv[1], delimiter=",", comments="#")
print(f"elbow bend min {min(180 - rb.elbow_angle(q) for q in rows[:, 1:]):.0f} deg; ra4 closest to straight {rows[:, 5].max():+.2f} rad")
