import sys, glob, numpy as np, mujoco
from hack_rl.robot import panther as P
m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML))
rng = np.array([m.jnt_range[m.joint(f"joint_ra{i}").id] for i in range(8)])
print("joint ranges (URDF):", [tuple(r.round(2)) for r in rng])
files = sorted(glob.glob("exports/*/throw_*.csv") + glob.glob("exports/*/*/throw_*.csv") + glob.glob("exports/test_*/*.csv"))
worst = []
for f in files:
    q = np.loadtxt(f, delimiter=",", comments="#")[:, 1:9]
    margin = np.minimum(q - rng[:, 0], rng[:, 1] - q).min(0)        # distance to the nearest limit, per joint
    worst.append((margin.min(), f, margin.round(3)))
bad = [w for w in worst if w[0] < 0]
print(f"{len(files)} exported files checked; outside a joint limit: {len(bad)}")
for w in sorted(worst)[:5]:
    print(f"  closest to a limit: {w[0]:+.3f} rad  {w[1].split('exports/')[-1]}  per joint {w[2]}")
