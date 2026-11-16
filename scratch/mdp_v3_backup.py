"""MDP terms for the frisbee throw.

Action (10 numbers per step, all roughly in [-1, 1]):
  0..7  residual joint offsets added to the reference throw at the current phase (x residual_scale rad)
  8     playback speed of the reference: 1 + a * speed_scale, clamped to speed_range
  9     release timing: release when the reference phase passes its recorded release time
        + a * release_shift_s. Releasing = gripper command 0 (open); after a random opening delay the
        frisbee's weld is switched off and it leaves with exactly the hand's velocity and spin.

Reward at release (one-shot), the user's spec: the disc is flat, moves straight forward, and spins fast
about its own axis. A weighted sum of components in [0, 1]:
  0.30 forward speed x straightness + 0.20 flat + 0.20 spin + 0.10 launch angle + 0.10 angle of attack
  + 0.10 spin axis along the disc normal (no wobble)
Before release a potential-based shaping term gamma*Phi(s') - Phi(s), with Phi = quality of the hand's
current state, gives a dense signal without changing the optimal policy. No flight is simulated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from hack_rl.reference import DEFAULT_FILES, load_references
from hack_rl.robot.panther import (ARM_JOINTS, GRIP_REL_POS, GRIP_REL_QUAT, GRIPPER_BODY, GRIPPER_JOINT,
                                   WELD_NAME, joint_velocity_limits, load_params)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ----------------------------------------------------------------------------- throw quality
@dataclass
class ThrowGoal:
    forward: tuple = (1.0, 0.0, 0.0)  # robot forward (world x; the torso is not rotated)
    speed: float = 4.0                # m/s forward at release for full credit
    spin: float = 15.0                # rad/s about the disc normal for full credit (~2.4 rev/s); the
                                      # placeholder joint velocity limit (6 rad/s) caps the wrist flick
    launch_deg: float = 5.0           # preferred launch angle above horizontal
    # weights of the release score (sum = 1). A weighted sum, not a product: every signal pays on its
    # own, which PPO can climb; a product is zero until everything is right at once.
    weights: dict = field(default_factory=lambda: {
        "speed_dir": 0.30, "flat": 0.20, "spin": 0.20, "elevation": 0.10, "aoa": 0.10, "no_wobble": 0.10})


def throw_quality(v: torch.Tensor, w: torch.Tensor, n: torch.Tensor, goal: ThrowGoal):
    """v, w: frisbee linear/angular velocity (world, B x 3); n: disc normal (world, B x 3).
    Returns (quality (B,), dict of components) with every component in [0, 1]."""
    fwd = torch.tensor(goal.forward, device=v.device, dtype=v.dtype)
    eps = 1e-6
    vh = v[:, :2]
    vh_norm = vh.norm(dim=1) + eps
    v_fwd = (v * fwd).sum(1)
    speed = (v_fwd / goal.speed).clamp(0.0, 1.0)
    cos_dir = (vh * fwd[:2]).sum(1) / vh_norm
    direction = ((cos_dir - 0.7) / 0.3).clamp(0.0, 1.0) ** 2            # 1 straight ahead, 0 past ~45 deg
    elev = torch.atan2(v[:, 2], vh_norm)
    elevation = torch.exp(-((elev - math.radians(goal.launch_deg)) / 0.35) ** 2)
    flat = n[:, 2].clamp(0.0, 1.0) ** 4                                # upright and level
    aoa = torch.exp(-(((n * v).sum(1) / (v.norm(dim=1) + eps)) / 0.35) ** 2)   # disc plane along v
    w_n = (w * n).sum(1)
    spin = (w_n.abs() / goal.spin).clamp(0.0, 1.0)
    wobble = (w - w_n[:, None] * n).norm(dim=1) / (w.norm(dim=1) + eps)
    no_wobble = torch.exp(-(wobble / 0.5) ** 2)
    w8 = goal.weights
    q = (w8["speed_dir"] * speed * direction + w8["flat"] * flat + w8["spin"] * spin
         + w8["elevation"] * elevation + w8["aoa"] * aoa + w8["no_wobble"] * no_wobble)
    comps = {"speed": speed, "direction": direction, "elevation": elevation, "flat": flat, "aoa": aoa,
             "spin": spin, "no_wobble": no_wobble, "v_forward_mps": v_fwd, "spin_radps": w_n.abs()}
    return q, comps


# ----------------------------------------------------------------------------- action
@dataclass(kw_only=True)
class ThrowActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    frisbee_name: str = "frisbee"
    reference_files: tuple = DEFAULT_FILES
    reference_ids: tuple | None = None          # restrict to some references (None = all)
    # rad per unit action, per joint (ra0..ra7): small on the shoulder (keep the human path), larger on
    # the wrist (grip orientation) and largest on ra7, whose axis is the disc normal (spin = ra7 flick)
    residual_scale: tuple = (0.3, 0.3, 0.3, 0.3, 0.5, 0.6, 0.6, 1.0)
    speed_scale: float = 0.5
    speed_range: tuple = (0.5, 3.0)
    release_shift_s: float = 0.6                 # reference seconds per unit action
    min_release_frac: float = 0.3                # ignore release commands earlier than this
    release_delay_range: tuple | None = None     # s; None -> sysid_params gripper.open_time_s +-50 %
    start_noise_rad: float = 0.03
    shaping_gamma: float = 0.99
    accel_limit: float = 25.0                    # rad/s^2 for commanded targets, unless sysid_params has one
    deterministic: bool = False                  # play/export: env i -> reference i % R, mean delay, no noise
    goal: ThrowGoal = field(default_factory=ThrowGoal)

    def build(self, env: ManagerBasedRlEnv) -> ThrowAction:
        return ThrowAction(self, env)


class ThrowAction(ActionTerm):
    cfg: ThrowActionCfg

    def __init__(self, cfg: ThrowActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        dev = self.device
        n = self.num_envs
        self._frisbee = env.scene[cfg.frisbee_name]
        ids, _ = self._entity.find_joints(list(ARM_JOINTS), preserve_order=True)
        self._arm_ids = torch.tensor(ids, device=dev, dtype=torch.long)
        gids, _ = self._entity.find_joints(["finger_l_joint", "finger_r_joint"], preserve_order=True)
        self._finger_ids = torch.tensor(gids, device=dev, dtype=torch.long)
        self._grip_act_ids = torch.tensor(self._entity.find_joints([GRIPPER_JOINT])[0], device=dev)
        self._gripper_body = self._entity.find_bodies([GRIPPER_BODY])[0][0]
        params = load_params()
        self._stroke = float(params["gripper"]["stroke_m"])
        if cfg.release_delay_range is None:
            t = float(params["gripper"]["open_time_s"])
            self._delay_range = (0.5 * t, 1.5 * t)
        else:
            self._delay_range = cfg.release_delay_range

        refs, q, length, release, dt = load_references(cfg.reference_files)
        sel = list(cfg.reference_ids) if cfg.reference_ids is not None else list(range(len(refs)))
        self.ref_names = [refs[i].name for i in sel]
        self._ref_q = torch.tensor(q[sel], device=dev, dtype=torch.float32)          # (R, N, 8)
        self._ref_len = torch.tensor(length[sel], device=dev, dtype=torch.float32)   # (R,)
        self._ref_rel = torch.tensor(release[sel], device=dev, dtype=torch.float32)
        self._ref_dt = dt

        lim = self._entity.data.soft_joint_pos_limits[0, self._arm_ids]            # (8, 2)
        self._qmin, self._qmax = lim[:, 0].clone(), lim[:, 1].clone()

        model = env.sim.mj_model
        self._weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME)
        if self._weld < 0:
            raise RuntimeError(f"equality '{WELD_NAME}' not in the scene (SceneCfg.spec_fn missing?)")
        self._rel_pos = torch.tensor(GRIP_REL_POS, device=dev).expand(n, 3)
        self._rel_quat = torch.tensor(GRIP_REL_QUAT, device=dev).expand(n, 4)

        self._res_scale = torch.tensor(cfg.residual_scale, device=dev, dtype=torch.float32)
        # commanded targets never move faster than the joint velocity limits (sysid_params.yaml): the
        # exported trajectory must pass cartesian_follow.py's --max-joint-vel check on the robot
        vl = joint_velocity_limits(params)
        self._vmax = torch.tensor([vl[j] for j in ARM_JOINTS], device=dev, dtype=torch.float32)
        self._amax = torch.tensor([float(params["joints"][j].get("accel_limit", cfg.accel_limit)) for j in ARM_JOINTS],
                                  device=dev, dtype=torch.float32)
        self._raw = torch.zeros(n, 10, device=dev)
        self.phase = torch.zeros(n, device=dev)                 # seconds along the reference
        self.speed = torch.ones(n, device=dev)
        self.ref_id = torch.zeros(n, device=dev, dtype=torch.long)
        self.q_target = torch.zeros(n, 8, device=dev)
        self.q_ref = torch.zeros(n, 8, device=dev)
        self.v_cmd = torch.zeros(n, 8, device=dev)
        self.held = torch.ones(n, device=dev, dtype=torch.bool)       # weld on
        self.opening = torch.zeros(n, device=dev, dtype=torch.bool)   # open commanded, weld still on
        self.timer = torch.zeros(n, device=dev)
        self.just_released = torch.zeros(n, device=dev, dtype=torch.bool)
        self.release_time = torch.full((n,), float("inf"), device=dev)
        self.release_state = torch.zeros(n, 13, device=dev)          # pos, quat(wxyz), lin vel, ang vel (world)
        self.release_phase = torch.zeros(n, device=dev)
        self.open_cmd_time = torch.full((n,), float("inf"), device=dev)   # when the gripper was told to open
        self.phi_prev = torch.zeros(n, device=dev)

    # ------------------------------------------------------------------ properties
    @property
    def action_dim(self) -> int:
        return 10

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    @property
    def ref_length(self) -> torch.Tensor:
        return self._ref_len[self.ref_id]

    @property
    def ref_release(self) -> torch.Tensor:
        return self._ref_rel[self.ref_id]

    def reference_at(self, phase: torch.Tensor, ref_id: torch.Tensor | None = None) -> torch.Tensor:
        """Linear interpolation of the reference joint positions, (B, 8)."""
        rid = self.ref_id if ref_id is None else ref_id
        x = (phase / self._ref_dt).clamp(min=0.0)
        i0 = x.floor().long().clamp(max=self._ref_q.shape[1] - 1)
        i1 = (i0 + 1).clamp(max=self._ref_q.shape[1] - 1)
        a = (x - i0.float()).clamp(0.0, 1.0)[:, None]
        return (1 - a) * self._ref_q[rid, i0] + a * self._ref_q[rid, i1]

    def env_time(self) -> torch.Tensor:
        return self._env.episode_length_buf.float() * self._env.step_dt

    # ------------------------------------------------------------------ action processing
    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        c = self.cfg
        self.speed = (1.0 + c.speed_scale * a[:, 8]).clamp(*c.speed_range)
        self.phase = torch.minimum(self.phase + self.speed * self._env.step_dt, self.ref_length)
        self.q_ref = self.reference_at(self.phase)
        desired = torch.clamp(self.q_ref + self._res_scale * a[:, :8], self._qmin, self._qmax)
        # second-order limiter: commanded targets respect both the velocity and the acceleration limit,
        # braking in time (v <= sqrt(2 a |err|)) so they never overshoot. Without the acceleration limit a
        # policy can reverse a joint at full speed between two 20 ms samples, and the robot's spline
        # trajectory controller turns that into 1.5x the speed limit.
        dt = self._env.step_dt
        a_dt = self._amax * dt

        def brake(dist):        # fastest v that can still stop within dist: v*dt + v^2/(2a) <= dist
            return -a_dt + torch.sqrt(a_dt * a_dt + 2 * self._amax * dist.clamp(min=0.0))

        err = desired - self.q_target
        v_des = torch.sign(err) * torch.minimum(self._vmax, brake(err.abs()))
        # never head into a joint limit faster than it can be stopped
        v_des = torch.clamp(v_des, -brake(self.q_target - self._qmin), brake(self._qmax - self.q_target))
        self.v_cmd = torch.clamp(self.v_cmd + torch.clamp(v_des - self.v_cmd, -a_dt, a_dt), -self._vmax, self._vmax)
        self.q_target = torch.clamp(self.q_target + self.v_cmd * dt, self._qmin, self._qmax)
        # release command (latched); forced at the end of the reference so every episode throws
        when = self.ref_release + c.release_shift_s * a[:, 9]
        frac = self.phase / self.ref_length
        cmd = ((self.phase >= when) & (frac >= c.min_release_frac)) | (self.phase >= self.ref_length - 1e-6)
        start = cmd & self.held & ~self.opening
        if start.any():
            lo, hi = self._delay_range
            u = 0.5 if c.deterministic else torch.rand(int(start.sum()), device=self.device)
            self.timer[start] = lo + (hi - lo) * u
            self.open_cmd_time[start] = self.env_time()[start]
            self.opening |= start
        self.just_released[:] = False

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self.q_target, joint_ids=self._arm_ids)
        grip = torch.where(self.opening | ~self.held, 0.0, self._stroke)
        self._entity.set_joint_position_target(grip[:, None], joint_ids=self._grip_act_ids)
        if self.opening.any():
            self.timer -= self._env.physics_dt * self.opening.float()
            fire = self.opening & (self.timer <= 0.0)
            if fire.any():
                ids = fire.nonzero(as_tuple=False).squeeze(-1)
                self.release_state[ids] = self._frisbee_state(ids)      # exact state at the instant
                self._env.sim.data.eq_active[ids, self._weld] = False
                self.held[ids] = False
                self.opening[ids] = False
                self.just_released[ids] = True
                self.release_time[ids] = self.env_time()[ids]
                self.release_phase[ids] = self.phase[ids]

    def _frisbee_state(self, ids: torch.Tensor) -> torch.Tensor:
        idx = self._frisbee.data.indexing
        qpos = self._env.sim.data.qpos[ids][:, idx.free_joint_q_adr]       # pos(3) quat(4)
        qvel = self._env.sim.data.qvel[ids][:, idx.free_joint_v_adr]       # lin (world), ang (body)
        ang_w = quat_apply(qpos[:, 3:7], qvel[:, 3:6])
        return torch.cat([qpos, qvel[:, :3], ang_w], dim=1)

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        k = len(env_ids)
        if k == 0:
            return
        self._raw[env_ids] = 0.0
        if self.cfg.deterministic:
            self.ref_id[env_ids] = env_ids % self._ref_q.shape[0]
        else:
            self.ref_id[env_ids] = torch.randint(0, self._ref_q.shape[0], (k,), device=self.device)
        self.phase[env_ids] = 0.0
        self.speed[env_ids] = 1.0
        self.held[env_ids] = True
        self.opening[env_ids] = False
        self.timer[env_ids] = 0.0
        self.just_released[env_ids] = False
        self.release_time[env_ids] = float("inf")
        self.release_state[env_ids] = 0.0
        self.release_phase[env_ids] = 0.0
        self.open_cmd_time[env_ids] = float("inf")
        self._env.sim.data.eq_active[env_ids, self._weld] = True

        q0 = self._ref_q[self.ref_id[env_ids], 0]
        if not self.cfg.deterministic:
            q0 = q0 + self.cfg.start_noise_rad * (2 * torch.rand_like(q0) - 1)
        q0 = torch.clamp(q0, self._qmin, self._qmax)
        self.q_ref[env_ids] = q0
        self.q_target[env_ids] = q0
        self.v_cmd[env_ids] = 0.0
        self._entity.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=self._arm_ids, env_ids=env_ids)
        closed = torch.full((k, 2), self._stroke, device=self.device)
        self._entity.write_joint_state_to_sim(closed, torch.zeros_like(closed), joint_ids=self._finger_ids,
                                              env_ids=env_ids)
        # put the frisbee in the grip: forward kinematics, then the weld's relative pose
        self._env.sim.forward()
        g = self._entity.data.body_link_pose_w[env_ids, self._gripper_body]     # (k, 7)
        pos = g[:, :3] + quat_apply(g[:, 3:7], self._rel_pos[:k])
        quat = quat_mul(g[:, 3:7], self._rel_quat[:k])
        state = torch.cat([pos, quat, torch.zeros(k, 6, device=self.device)], dim=1)
        self._frisbee.write_root_state_to_sim(state, env_ids=env_ids)
        self._env.sim.forward()
        self.phi_prev[env_ids] = hand_quality(self._env, self)[0][env_ids]


def _term(env: ManagerBasedRlEnv) -> ThrowAction:
    return env.action_manager.get_term("throw")  # type: ignore[return-value]


def _frisbee_now(env: ManagerBasedRlEnv, term: ThrowAction):
    d = term._frisbee.data
    quat = d.root_link_quat_w
    n = quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(quat.shape[0], 3))
    return d.root_link_lin_vel_w, d.root_link_ang_vel_w, n


def hand_quality(env: ManagerBasedRlEnv, term: ThrowAction | None = None):
    """Quality of the throw if the frisbee were released now (it moves rigidly with the hand)."""
    term = term or _term(env)
    v, w, n = _frisbee_now(env, term)
    return throw_quality(v, w, n, term.cfg.goal)


# ----------------------------------------------------------------------------- observations
def throw_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    t = _term(env)
    frac = t.phase / t.ref_length
    to_release = (t.ref_release - t.phase)
    return torch.stack([frac, torch.sin(math.pi * frac), torch.cos(math.pi * frac), to_release,
                        t.speed, t.held.float(), t.opening.float()], dim=1)


def reference_preview(env: ManagerBasedRlEnv, horizons: tuple = (0.0, 0.1, 0.25)) -> torch.Tensor:
    """Reference joint positions at the current phase and a little ahead, relative to the robot's."""
    t = _term(env)
    q = t._entity.data.joint_pos[:, t._arm_ids]
    return torch.cat([t.reference_at(t.phase + h) - q for h in horizons], dim=1)


def arm_joint_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    t = _term(env)
    return t._entity.data.joint_pos[:, t._arm_ids]


def arm_joint_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
    t = _term(env)
    return t._entity.data.joint_vel[:, t._arm_ids]


def frisbee_motion(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Disc normal, linear and angular velocity (world): what would be thrown right now."""
    t = _term(env)
    v, w, n = _frisbee_now(env, t)
    return torch.cat([n, v, w * 0.1], dim=1)


# ----------------------------------------------------------------------------- rewards
def release_quality(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-shot reward on the step the frisbee leaves the hand, scored on its exact release state."""
    t = _term(env)
    s = t.release_state
    pos, quat, v, w = s[:, :3], s[:, 3:7], s[:, 7:10], s[:, 10:13]
    n = quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(quat.shape[0], 3))
    q, comps = throw_quality(v, w, n, t.cfg.goal)
    jr = t.just_released.float()
    if jr.any():
        log = env.extras.setdefault("log", {})
        m = t.just_released
        for k, c in comps.items():
            log[f"Release/{k}"] = c[m].mean()
        log["Release/quality"] = q[m].mean()
        log["Release/phase_frac"] = (t.release_phase / t.ref_length)[m].mean()
    # the reward manager multiplies by dt; undo that for a one-shot bonus so its weight reads plainly
    return q * jr / env.step_dt


def release_shaping(env: ManagerBasedRlEnv) -> torch.Tensor:
    """gamma*Phi(s') - Phi(s), Phi = quality of the hand state while held, 0 after release."""
    t = _term(env)
    phi, _ = hand_quality(env, t)
    phi = phi * t.held.float()
    r = t.cfg.shaping_gamma * phi - t.phi_prev
    t.phi_prev = phi.detach().clone()
    return r / env.step_dt


def track_reference(env: ManagerBasedRlEnv, std: float = 0.3) -> torch.Tensor:
    """Stay close to the human throw (annealed by the curriculum)."""
    t = _term(env)
    q = t._entity.data.joint_pos[:, t._arm_ids]
    err = ((q - t.q_ref) ** 2).sum(1)
    return torch.exp(-err / std ** 2) * t.held.float()


def arm_velocity_excess(env: ManagerBasedRlEnv, limits: dict) -> torch.Tensor:
    t = _term(env)
    v = t._entity.data.joint_vel[:, t._arm_ids].abs()
    lim = torch.tensor([limits[j] for j in ARM_JOINTS], device=env.device)
    return ((v - lim).clamp_min(0.0) ** 2).sum(1)


def body_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    data = env.scene[sensor_name].data
    return data.found.reshape(data.found.shape[0], -1).any(dim=1).float()


# ----------------------------------------------------------------------------- terminations
def thrown(env: ManagerBasedRlEnv, after_s: float = 0.1) -> torch.Tensor:
    """End the episode shortly after release: only the release state is scored."""
    t = _term(env)
    return (t.env_time() - t.release_time) >= after_s
