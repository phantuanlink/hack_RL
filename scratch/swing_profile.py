import numpy as np, math
from hack_rl.prepare_references_v2 import Kin
from hack_rl.reference import DATA_DIR
kin = Kin()
for name in ("joints_trajectory_shoulder_rel_2", "joints_trajectory_shoulder_rel_3"):
    rows = np.loadtxt(DATA_DIR / "human_speed" / f"{name}.csv", delimiter=",", comments="#")
    txt = open(DATA_DIR / "human_speed" / f"{name}.csv").read()
    import re; tr = float(re.search(r"release at t=([\d.]+)s", txt).group(1))
    t, q = rows[:, 0], rows[:, 1:]; dt = t[1] - t[0]
    Pp = np.array([kin.disc(x)[0] for x in q]); V = np.gradient(Pp, dt, axis=0)
    sp = np.linalg.norm(V[:, :2], axis=1); k_rel = int(round(tr / dt)); kmax = int(np.argmax(sp[:k_rel + 1]))
    print(f"== {name}: release (crop end) t={tr:.2f}s; peak horizontal disc speed {sp[kmax]:.1f} m/s at t={t[kmax]:.2f}s")
    for tt in np.arange(0, tr + 0.001, 0.1):
        i = int(round(tt / dt)); v = V[i]
        print(f"   t={tt:4.2f}s  disc pos {Pp[i].round(2)}  horiz speed {sp[i]:4.1f} m/s  heading {math.degrees(math.atan2(v[1], v[0])):+5.0f} deg")
