# hack_RL: teaching the Panther's right arm to throw a frisbee

RL fine-tuning (mjlab / MuJoCo Warp, PPO) of a frisbee throw, starting from the human throws we filmed
with phones (`hack/armmocap` → HEADBands `retarget.py`). Runs natively on Windows with the laptop GPU.

## What is in here
| Path | What |
|---|---|
| `src/hack_rl/robot/sysid_params.yaml` | **System ID results** (masses, inertias, PD gains, effort/velocity limits, gripper timing). Today: the vendor CAD values, assumed identified. Replace with the real identification, then rebuild the model. |
| `src/hack_rl/robot/build_mjcf.py` | URDF + sysid params → `assets/panther_right_arm.xml` (torso, right arm `joint_ra0..7`, 2-finger gripper, capsule collisions, parked left arm) and `assets/frisbee.xml` (175 g, 27.5 cm, Ultimate disc inertia) |
| `src/hack_rl/robot/panther.py` | mjlab entity configs, actuators from the sysid file, the grip weld (frisbee ↔ gripper) |
| `src/hack_rl/robot/franka.py`, `assets/franka/` | Franka Panda (MuJoCo Menagerie, Apache-2.0) as an off-the-shelf 7-DOF comparison arm |
| `src/hack_rl/robot/profile.py` | what the task needs per arm (joints, grip pose, limits, grip friction, references) |
| `src/hack_rl/retarget_arm.py` | Panther throws → another arm: same disc path from the shoulder, best mount by search, weighted IK |
| `data/reference/` | the retargeted human throws from HEADBands; `adapted/` = prepared for a flat grip (below) |
| `src/hack_rl/prepare_references.py` | release at peak forward hand speed + wrist offset that makes the disc flat at release |
| `src/hack_rl/tasks/throw/` | the RL task `Hack-Throw-Frisbee-Panther`: action, observations, reward, PPO config |
| `src/hack_rl/export_rollout.py` | trained policy → joint-target CSV for HEADBands `cartesian_follow.py --joints` + release info |
| `deploy/release_gripper.py` | ROS 2 node for the robot side: closes the gripper, then opens it at the policy's release point |
| `tests/test_throw.py` | reward, model, references, and a GPU test that the frisbee is held then released |

## Setup (Windows, PowerShell or cmd)
```bat
cd C:\Users\linhp\Documents\hack\hack_RL
uv sync
uv run python scripts\check_gpu.py
uv run --group dev pytest -q tests
```
Pins that matter (see `pyproject.toml`): mjlab 1.6.0 needs MuJoCo / MuJoCo Warp 3.11; `mjviser` is held at
0.0.16 because 0.0.17 installs `mujoco-core` 4.x over it. PyTorch comes from the CUDA 13 index (RTX 50-series).
uv's own Python download failed on this machine, so the project uses the installed Python 3.13.

## Train, watch, export
Logging to wandb (account on the CoreWeave-hosted server; `wandb login` once):
```bat
set WANDB_BASE_URL=https://api.forge.coreweave.com
uv run python -m hack_rl.train Hack-Throw-Frisbee-Panther-Heading --env.scene.num-envs 4096 --agent.logger wandb --agent.wandb-project hack_RL --agent.run-name dr3
uv run python scripts\watch_checkpoints.py --task Hack-Throw-Frisbee-Panther-Heading --run logs\rsl_rl\panther_frisbee_throw\<run> --wandb <run>-videos
```
wandb is pinned < 0.30 (0.30+ removed `sync --sync-tensorboard`); `hack_rl.train` drops RSL-RL's
`Settings(start_method=...)`, which current wandb rejects. Older TensorBoard runs: `scripts\tb_to_wandb.py`.

```bat
uv run python -m hack_rl.train Hack-Throw-Frisbee-Panther --env.scene.num-envs 4096 --agent.max-iterations 1500 --agent.logger tensorboard
uv run tensorboard --logdir logs\rsl_rl\panther_frisbee_throw
uv run python -m hack_rl.play Hack-Throw-Frisbee-Panther --checkpoint-file logs\rsl_rl\panther_frisbee_throw\<run>\model_1499.pt
uv run python -m hack_rl.export_rollout            :: latest checkpoint -> exports\<run>_<model>\throw_*.csv/json/png
uv run python scripts\evaluate.py --envs 2048      :: robustness under all randomisation (add --task ... for Franka)
uv run python -m hack_rl.export_rollout --zero     :: baseline: the adapted human throw, no policy
```
About 1.3 s per PPO iteration with 4096 parallel envs on the RTX 5060 laptop GPU (1500 iterations ≈ 35 min).

## The task
**Grip and release.** The frisbee is held by its rim between the fingers: disc normal = the fingers' closing
axis (gripper y), centre one radius beyond the finger tips. While the gripper is closed a weld equality holds
it; when the policy opens the gripper (command 0.0, like `/right_gripper/gripper_command`), the weld is
switched off after a random opening delay (sysid `gripper.open_time_s` ±50 %), and the frisbee leaves with
exactly the hand's velocity and spin. No flight is simulated: only the release state is scored.

**Action (10 per step, 50 Hz).** 8 residual joint offsets on top of the reference throw (±0.3 rad shoulder …
±1.0 rad on `ra7`, whose axis is the disc normal, so a wrist flick there is spin), the reference playback
speed (0.5–3×), and the release timing (±0.6 s around the reference's release). Commanded targets pass
through a second-order limiter: speed ≤ the sysid `velocity_limit` (3 rad/s now, the HEADBands cap) and
acceleration ≤ `accel_limit` (25 rad/s² placeholder), braking in time for the target and for joint limits.
Without the acceleration limit the first policy reversed joints at full speed between 20 ms samples, and the
robot's spline controller would have run them at 1.5× the limit (4.5 rad/s). `export_rollout` reproduces
`cartesian_follow.py`'s spline speed check and prints OK / EXCEEDS for every exported throw.

**Reward.** At release, a score in [0, 1]:
0.30 forward speed × straight-ahead + 0.20 flat (disc normal up) + 0.20 spin about the disc normal
+ 0.10 launch angle (~5°) + 0.10 angle of attack (disc plane along the velocity) + 0.10 no wobble
(spin axis = disc normal). Targets: 4 m/s, 15 rad/s (`ThrowGoal` in `mdp.py`). Before release, potential-based
shaping from "the score if released now" (doesn't change the optimum); imitation of the reference (annealed
0.5 → 0.05); penalties for joint limits, joint speed above the limit, jerky actions, and the arm touching the
torso, head or parked left arm. Domain randomisation: link inertia ±15 %, PD gains ±20 %, frisbee ±6 %,
release delay.

**Why the references are "adapted".** The retargeted throws match the human hand path but not a frisbee
grip: at their recorded release the disc was tilted upside down (normal z ≈ −0.5…−0.65), and that release time
was in the slow-down after the stroke. `prepare_references.py` moves the release to peak forward hand speed
and blends in a wrist offset that makes the disc flat there (normal z 0.95) without moving the hand.

## Domain randomisation (dr1 runs)
Trained policies must survive what system ID gets wrong. Everything below is on in training and in
`scripts/evaluate.py`; play/export use the nominal values.

| What | Range | How |
|---|---|---|
| frisbee mass (+ inertia) | 115–200 g | `dr.pseudo_inertia` on the disc, once per env |
| grip position | ±1 cm deeper/shallower between the pads, ±2 cm along the rim | weld pose per episode (`grip_randomization`) |
| disc tilt in the fingers | ±5° about the radial and the rim axis | same |
| pad friction μ | 0.6–1.5 × nominal (0.6) | friction-grip model, per episode |
| squeeze force | 0.6–1.0 × nominal (Panther 40 N, Franka 70 N) | same |
| pad contact patch | 0.7–1.2 × nominal | same |
| arm command delay | 0–30 ms (6 physics steps), mostly constant, sometimes jumps | mjlab actuator `delay_*_lag` |
| gripper opening delay | sysid `open_time_s` × 0.5–1.5 + 0–30 ms trigger latency | release timer |
| joint calibration offset | ±0.01 rad per joint | real joint = command + offset |
| link inertia | ±15 % (mass and inertia together) | `dr.pseudo_inertia` |
| servo kp, kd | ±20 % | `pd_servo_gains` |
| joint friction / damping | Panther 0–0.1 N m, Franka 0–1 N m / damping 0.5–2 × | `dr.joint_friction`, `dr.joint_damping` |
| start pose | ±0.03 rad | reset |
| release push | ±0.2 m/s and ±2 rad/s per axis on the disc as it leaves (uneven finger opening) | added to the release state, which is what is scored (from dr2) |
| posture | base height ±3 cm (Panther lift / Franka mount), roll and pitch ±2°, yaw ±2° (not facing exactly forward) | `reset_root_state_uniform` per episode (from dr2) |

**Friction grip.** The disc is held by a weld, so MuJoCo contact friction never acts on it. Instead the weld's
constraint wrench (all the fingers must transmit) is compared every physics step with what a two-pad pinch
holds: in-plane force ≤ 2μF, twist about the disc axis ≤ 2μF·r_eff, pull along the closing axis ≤ 2F,
bending ≤ 2F·(pad half size). Over the limit for 10 ms (10 ms low-pass) → the disc slips out: no release
reward, a penalty, episode over. Training starts with a 3× firmer grip and anneals to the real one by
iteration 600; a penalty keeps the load under 70 % of capacity. Measured on the v3 policy: it twisted the
disc with ~1 N m and bent it with 1.5–2.5 N m at the pinch, 2–5× what a 40 N pinch on small pads holds.
Put the measured pad friction, squeeze force and pad size in `sysid_params.yaml` (`grip:`).

**Open-loop actor.** The robot plays an open-loop joint CSV, so the actor sees no sensors (phase, reference,
its own commanded targets); the critic gets the measured state and the randomised parameters (asymmetric
actor-critic). Domain randomisation then makes the one exported trajectory robust, and `evaluate.py` measures
exactly the CSV's robustness. (v3 had a closed-loop actor, but only its nominal rollout was exported.)

**Servo model.** Arm joints use `BuiltinPdActuatorCfg`: PD on commanded position *and* velocity, the 50 Hz
targets interpolated in between, torque-limited. A plain position actuator damps toward zero speed
(Franka: kd 450 × 2 rad/s = 10× its 87 N m limit) and stepping targets every 20 ms kicked the wrist.

**Franka Panda.** Menagerie model with measured inertias, datasheet limits (2.175/2.61 rad/s, 7.5–20 rad/s²,
87/12 N m, 70 N grip). `python -m hack_rl.retarget_arm --robot franka` puts its shoulder at the Panther's and
searches 24 mount orientations: hanging from above (ceiling mount) tracks the human disc path best
(1.4 cm, ~1° at release). Train with `Hack-Throw-Frisbee-Franka`; export with `--task Hack-Throw-Frisbee-Franka`.

## Results before domain randomisation (`results/v3/`, policy `policy_v3_model_1499.pt`)
v3 used plain position servos, no grip-friction model and a closed-loop actor; not comparable to dr1.

Trained 1500 PPO iterations (≈35 min, 4096 envs), converged by ≈400. Release score, nominal dynamics:
**0.81 / 0.86** for the two throws (adapted human throw without RL: 0.39 / 0.29). Under randomised dynamics
(2048 envs, `robustness_eval.json`): mean **0.83** (p10 0.72, p90 0.90) vs 0.34 for the human throw.

| at release | human throw (adapted) | RL policy v3 |
|---|---|---|
| forward speed | 0.9–1.7 m/s | **3.7 m/s** (p10–p90 3.3–4.1) |
| spin about the disc axis | 1–2.5 rad/s | **7.9 rad/s** (≈1.3 rev/s) |
| flat (normal z⁴) / no wobble / angle of attack | 0.8 / 0.2 / 0.07 | **0.996 / 0.98 / 0.985** |
| straight ahead | 0.97 | 0.82 (10 % veer ~20–30° right) |
| arm touches torso / left arm | — | 3 % of episodes |

`throw_*.csv` are ready for `cartesian_follow.py --joints` (time stretched ×1.04–1.05 so the controller's
spline stays ≤ 3 rad/s; checked with the same formula), `throw_*.json` say where to open the gripper, `*.mp4`
show the throw (kinematic replay + ballistic frisbee), `training_curves.png` the learning curves. Spin and
speed are capped by the 3 rad/s joint limit; raise it in `sysid_params.yaml` if the robot allows more.

## Testing on the robot
1. **System ID first:** put the identified values into `sysid_params.yaml` (masses/inertias per link, kp/kd,
   effort and **velocity limits**, gripper `open_time_s`), then `uv run python -m hack_rl.robot.build_mjcf`
   and retrain. The velocity limit decides how much speed and spin are possible.
2. Export: `uv run python -m hack_rl.export_rollout`. Copy `exports\...\throw_*.csv` and `.json` to the
   HEADBands machine.
3. Dry run (no motion): `python3 cartesian_follow.py --arm right --joints throw_X.csv --dry-run`, look at the
   planned path in RViz. Then the same in the HEADBands mock sim.
4. Move the **left arm out of the way**: reference throws 2 and 3 swing the right hand across where the left
   arm hangs (contact in our collision check; the policy is penalised for it, but check).
5. On the robot: `python3 deploy/release_gripper.py throw_X.csv --close` (closes the gripper on the frisbee,
   waits for Enter), then in a second terminal `python3 cartesian_follow.py --arm right --joints throw_X.csv`.
   The node opens the gripper when the arm reaches the release row. Stand clear of the throw direction (+x).

## Known gaps
- Masses, gains and limits are CAD/guesses until system ID; the policy is trained with randomisation around
  them, but large errors (especially joint speed limits and servo tracking) need a retrain.
- Gripper opening speed is unmeasured (0.08 s assumed); `release_gripper.py --lead-rows` can compensate.
- Collision shapes are capsules/boxes, not the real meshes (those are on the robot).
- No aerodynamics: flight quality is judged only from the release state (flat, forward, spinning).
- `release_gripper.py` is untested (no ROS on Windows, no robot tonight).
