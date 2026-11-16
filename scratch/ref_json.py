import json, re, numpy as np, mujoco
from hack_rl.prepare_references_v2 import Kin
from hack_rl.reference import DATA_DIR
kin = Kin()
for name in ("joints_trajectory_shoulder_rel_2", "joints_trajectory_shoulder_rel_3"):
    f = DATA_DIR / "human_speed" / f"{name}.csv"
    rows = np.loadtxt(f, delimiter=",", comments="#"); t, q = rows[:, 0], rows[:, 1:]; dt = t[1] - t[0]
    tr = float(re.search(r"release at t=([\d.]+)s", f.read_text()).group(1)); k = int(round(tr / dt))
    p0, R0 = kin.disc(q[k - 1]); p1, R1 = kin.disc(q[k + 1])
    dR = R1 @ R0.T; w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * 2 * dt)
    meta = {"frisbee_leaves_t": tr, "velocity_mps": ((p1 - p0) / (2 * dt)).tolist(), "angular_velocity_radps": w.tolist()}
    f.with_suffix(".json").write_text(json.dumps(meta))
    print(name, "release", tr, "v", np.round(meta["velocity_mps"], 2), "w", np.round(w, 1))
