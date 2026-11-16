import re, numpy as np
from hack_rl.prepare_references_v3 import Robot
rb = Robot()
for r in (2, 3):
    f = f"data/reference/chest/joints_trajectory_shoulder_rel_{r}.csv"
    txt = open(f).read(); rows = np.loadtxt(f, delimiter=",", comments="#"); t, Q = rows[:, 0], rows[:, 1:]; dt = t[1] - t[0]
    k = int(round(float(re.search(r"release at t=([\d.]+)s", txt).group(1)) / dt))
    P = np.array([rb.disc(q)[0] for q in Q]); V = np.gradient(P, dt, axis=0)
    d = V[k, :2] / np.linalg.norm(V[k, :2])                      # release direction (horizontal)
    proj = P[:k + 1, :2] @ d
    b = int(np.argmin(proj))                                      # deepest backswing along the throw direction
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    sp = np.linalg.norm(V, axis=1)
    print(f"ref{r}: whole reference start->release {seg[:k].sum():.2f} m in {t[k]:.2f}s | "
          f"wind-up start->backswing {seg[:b].sum():.2f} m in {t[b]:.2f}s | "
          f"STROKE backswing->release {seg[b:k].sum():.2f} m in {t[k] - t[b]:.2f}s "
          f"(straight-line {np.linalg.norm(P[k] - P[b]):.2f} m, along the throw direction {proj[k] - proj[b]:.2f} m), "
          f"disc speed at backswing {sp[b]:.1f} -> release {sp[k]:.1f} m/s")
