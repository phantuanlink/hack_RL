import numpy as np, mujoco
from hack_rl import enlarge_references as E
from hack_rl.robot import panther as P
m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML)); d = mujoco.MjData(m)
qa = np.array([m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]); arm_ids = {m.body(b).id for b in E.ARM_BODIES}
for r in (2, 3):
    rows = np.loadtxt(f"data/reference/chest/joints_trajectory_shoulder_rel_{r}.csv", delimiter=",", comments="#")
    print(f"ref{r}: min gap arm (ra3..gripper) to torso/head {E.min_gap(m, d, rows[::2, 1:], qa, arm_ids) * 100:.1f} cm")
