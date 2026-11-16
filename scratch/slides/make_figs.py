import json, math, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import imageio.v3 as iio
BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#14213D", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
d = json.load(open("data/reference/human/trajectory_throw2.json")); t = np.array(d["timestamp"], float)
W, S = np.array(d["wrist"], float), np.array(d["shoulder"], float)
L, R = np.array(d["extra_joints"]["left_shoulder"], float), np.array(d["extra_joints"]["right_shoulder"], float)
left = L - R; yaw = savgol_filter(np.unwrap(np.arctan2(-left[:, 0], left[:, 1])), 11, 2)
t0, t1 = 1.017140, 1.917558; m = (t >= t0) & (t <= t1); idx = np.flatnonzero(m)
rel = savgol_filter(W - S, 7, 2, axis=0)
v = np.gradient(rel, t, axis=0); k = idx[np.argmax(np.linalg.norm(v[m, :2], axis=1))]
c, s = np.cos(-yaw), np.sin(-yaw)
relc = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], 1)
vc = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)
fig, axs = plt.subplots(1, 2, figsize=(14, 6.6), facecolor=SURF)
for ax, P, V, title, chest in ((axs[0], rel[:, :2], v[:, :2], "Camera frame (chest direction at rest)", True),
                               (axs[1], relc, vc, "Chest frame (rotates with the torso)", False)):
    ax.set_facecolor(SURF)
    ax.plot(P[idx, 1], P[idx, 0], color=BLUE, lw=3, solid_capstyle="round", label="hand path")
    ax.plot(P[idx[0], 1], P[idx[0], 0], "o", ms=11, color=BLUE, mec=SURF, mew=2)
    ax.plot(P[k, 1], P[k, 0], "o", ms=13, color=ORANGE, mec=SURF, mew=2)
    u = V[k] / np.linalg.norm(V[k]) * 0.22
    ax.annotate("", xy=(P[k, 1] + u[1], P[k, 0] + u[0]), xytext=(P[k, 1], P[k, 0]),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3, mutation_scale=22))
    ax.text(P[k, 1] + 0.02, P[k, 0] - 0.06, "release", color=INK, fontsize=15, ha="left")
    ax.text(P[idx[0], 1] + 0.03, P[idx[0], 0] - 0.06, "wind-up", color=INK, fontsize=15)
    ax.plot(0, 0, "s", ms=12, color=INK); ax.text(0.03, 0.03, "shoulder", color=INK, fontsize=14)
    if chest:
        for i in idx[::14]:
            f = np.array([math.cos(yaw[i]), math.sin(yaw[i])]) * 0.12
            ax.annotate("", xy=(f[1], f[0]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#9aa0a6", lw=1.6, mutation_scale=14))
        ax.text(0.55, 0.06, "grey: chest direction,\nturns ~50 deg during the throw", color=MUTED, fontsize=14,
                transform=ax.transAxes)
    else:
        ax.annotate("", xy=(0, 0.16), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#9aa0a6", lw=2.2, mutation_scale=16))
        ax.text(0.02, 0.18, "chest forward (fixed)", color=MUTED, fontsize=14)
    ax.set_title(title, color=INK, fontsize=17, loc="left", pad=10)
    ax.set_xlabel("left  <-  y (m)  ->  right"); ax.set_ylabel("x forward (m)")
    ax.invert_xaxis(); ax.set_aspect("equal"); ax.grid(color="#e6e5df", lw=0.8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("scratch/slides/frames.png", dpi=110, facecolor=SURF)
# robot strip from the v4 stroke video (4x slow motion, 50 fps): backswing, mid-stroke, release, follow-through
fr = iio.imread("exports/reference_v2/ref2_stroke_v4_slowmo4x.mp4", index=None)
pick = {"backswing": 0.30, "stroke": 0.44, "release": 0.52, "follow-through": 0.62}
tiles = []
for name, tt in pick.items():
    im = fr[min(len(fr) - 1, int(round(tt * 4 * 50)))]
    tiles.append(im[90:640, 240:900])
strip = np.concatenate([np.pad(x, ((0, 0), (0, 12), (0, 0)), constant_values=252) for x in tiles], 1)
iio.imwrite("scratch/slides/robot_strip.png", strip)
print("ok", strip.shape, "release heading in chest frame",
      round(math.degrees(math.atan2(vc[k, 1], vc[k, 0]))), "camera", round(math.degrees(math.atan2(v[k, 1], v[k, 0]))))
