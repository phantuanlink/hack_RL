import numpy as np, mujoco, json
from hack_rl.reference import load_reference, DATA_DIR
from hack_rl.robot import panther as P
m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML)); d = mujoco.MjData(m)
adr = np.array([m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]); gb = m.body(P.GRIPPER_BODY).id
sh = None
def disc(q):
    d.qpos[adr] = q; mujoco.mj_kinematics(m, d)
    return d.xpos[gb] + d.xmat[gb].reshape(3, 3) @ np.array(P.GRIP_REL_POS)
d.qpos[:] = 0; mujoco.mj_kinematics(m, d); sh = d.xpos[m.body("link_ra1").id].copy()
def stats(name, t, q, k=None):
    p = np.array([disc(x) for x in q]); seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    k = len(q) - 1 if k is None else k
    L = seg[:k].sum(); v = seg / np.diff(t)
    ext = p[:k+1].max(0) - p[:k+1].min(0); r = np.linalg.norm(p - sh, axis=1)
    rng = q[:k+1].max(0) - q[:k+1].min(0)
    print(f"{name:42s} path to release {L:4.2f} m, box {ext.round(2)} m, hand-shoulder dist {r.min():.2f}-{r.max():.2f} m, "
          f"peak disc speed {v[:k+3].max():.1f} m/s\n{'':42s} joint range used (rad) {rng.round(2)}")
for sub in ("joints_trajectory_shoulder_rel_3.csv", "adapted/joints_trajectory_shoulder_rel_3.csv", "enlarged/joints_trajectory_shoulder_rel_3.csv"):
    r = load_reference(DATA_DIR / sub); k = int(round(r.release_t / (r.t[1] - r.t[0])))
    stats(sub, r.t, r.q, k)
rows = np.loadtxt("exports/test_dr7b/dr7b_a04.csv", delimiter=",", comments="#"); meta = json.load(open("exports/test_dr7b/dr7b_a04.json"))
stats("dr7b model_800 aggr 0.4 (robot file, full speed)", rows[:, 0], rows[:, 1:], meta["gripper_open_index"] + 9)
r = load_reference(DATA_DIR / "enlarged/joints_trajectory_shoulder_rel_3.csv")
print("robot arm reach (shoulder->disc centre, straight arm): see max above; joint limits:", [tuple(np.round(m.jnt_range[m.joint(f'joint_ra{i}').id], 2)) for i in range(8)])
