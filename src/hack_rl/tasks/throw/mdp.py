"""MDP terms for the frisbee throw.

Action (J + 2 numbers per step, J = arm joints, all roughly in [-1, 1]):
  0..J-1  residual joint offsets added to the reference throw at the current phase (x residual_scale rad)
  J       playback speed of the reference: 1 + a * speed_scale, clamped to speed_range
  J+1     release timing: release when the reference phase passes its recorded release time
        + a * release_shift_s. Releasing = gripper command 0 (open); after a random opening delay the
        frisbee's weld is switched off and it leaves with exactly the hand's velocity and spin.

Reward at release (one-shot), the user's spec: the disc is flat, moves straight forward, and spins fast
about its own axis. A weighted sum of components in [0, 1]:
  0.30 forward speed x straightness + 0.20 flat + 0.20 spin + 0.10 launch angle + 0.10 angle of attack
  + 0.10 spin axis along the disc normal (no wobble)
Before release a potential-based shaping term gamma*Phi(s') - Phi(s), with Phi = quality of the hand's
current state, gives a dense signal without changing the optimal policy. No flight is simulated.

Friction grip. The frisbee is held by a weld, so MuJoCo's contact friction never acts on it. Instead the
weld's constraint wrench (exactly what the fingers must transmit: the disc has no other contacts) is
compared, every physics step, with what a two-pad friction pinch can hold (friction mu, squeeze force F per
finger, pad patch half sizes a_t x a_r):
  in-plane force <= 2 mu F, twist about the disc normal <= 2 mu F r_eff   (friction limit surface)
  pull along the closing axis <= 2 F, bending moments <= 2 F a_t, 2 F a_r   (a pad lifts off)
If the load exceeds it for 10 ms the disc slips out: the throw fails (no release reward, penalty).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_from_angle_axis, quat_mul

from hack_rl.reference import load_references
from hack_rl.robot.panther import WELD_NAME
from hack_rl.robot.profile import ArmProfile, get_profile

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


def throw_quality(v: torch.Tensor, w: torch.Tensor, n: torch.Tensor, goal: ThrowGoal,
                  forward: torch.Tensor | None = None, v_target: torch.Tensor | None = None,
                  w_target: torch.Tensor | None = None, n_target: torch.Tensor | None = None):
    """v, w: frisbee linear/angular velocity (world, B x 3); n: disc normal (world, B x 3).
    forward: per-env target direction (B x 3, horizontal unit vectors); default goal.forward.
    Returns (quality (B,), dict of components) with every component in [0, 1]."""
    if forward is None:
        forward = torch.tensor(goal.forward, device=v.device, dtype=v.dtype).expand(v.shape[0], 3)
    fwd = forward
    eps = 1e-6
    vh = v[:, :2]
    vh_norm = vh.norm(dim=1) + eps
    v_fwd = (v * fwd).sum(1)
    if v_target is None:
        speed = (v_fwd / goal.speed).clamp(0.0, 1.0)
    else:                                   # aggressiveness: hit the requested speed (+-40 % -> 0)
        speed = (1.0 - (v_fwd - v_target).abs() / (0.4 * v_target)).clamp(0.0, 1.0)
    cos_dir = (vh * fwd[:, :2]).sum(1) / vh_norm
    direction = ((cos_dir - 0.7) / 0.3).clamp(0.0, 1.0) ** 2            # 1 straight ahead, 0 past ~45 deg
    elev = torch.atan2(v[:, 2], vh_norm)
    elevation = torch.exp(-((elev - math.radians(goal.launch_deg)) / 0.35) ** 2)
    flat = n[:, 2].clamp(0.0, 1.0) ** 4                                # upright and level
    aoa = torch.exp(-(((n * v).sum(1) / (v.norm(dim=1) + eps)) / 0.35) ** 2)   # disc plane along v
    if n_target is not None:     # commanded attitude (pitch): disc normal vs the commanded normal, ~7 deg width
        ang = torch.acos((n * n_target).sum(1).clamp(-1.0, 1.0))
        attitude = torch.exp(-(ang / 0.12) ** 2)
        flat, aoa = attitude, attitude
    w_n = (w * n).sum(1)
    if w_target is None:
        spin = (w_n.abs() / goal.spin).clamp(0.0, 1.0)
    else:                                   # at least the requested spin (more is fine)
        spin = (w_n.abs() / w_target).clamp(0.0, 1.0)
    wobble = (w - w_n[:, None] * n).norm(dim=1) / (w.norm(dim=1) + eps)
    no_wobble = torch.exp(-(wobble / 0.5) ** 2)
    w8 = goal.weights
    # everything else only counts for a throw that goes forward: without this gate a policy free enough
    # learned to fling the disc BACKWARDS (-7 m/s) for easy spin and still scored 0.62 (dr7)
    forward_gate = direction * (v_fwd > 0).to(v.dtype)
    q = (w8["speed_dir"] * speed * direction
         + forward_gate * (w8["flat"] * flat + w8["spin"] * spin + w8["elevation"] * elevation
                           + w8["aoa"] * aoa + w8["no_wobble"] * no_wobble))
    comps = {"speed": speed, "direction": direction, "elevation": elevation, "flat": flat, "aoa": aoa,
             "spin": spin, "no_wobble": no_wobble, "v_forward_mps": v_fwd, "spin_radps": w_n.abs()}
    return q, comps


# ----------------------------------------------------------------------------- action
@dataclass(kw_only=True)
class ThrowActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    frisbee_name: str = "frisbee"
    robot: str = "panther"                      # ArmProfile name (hack_rl.robot.profile)
    reference_files: tuple | None = None        # None -> the profile's
    reference_ids: tuple | None = None          # restrict to some references (None = all)
    # rad per unit action, per arm joint; None -> the profile's (Panther: small on the shoulder to keep the
    # human path, largest on ra7 whose axis is the disc normal, i.e. a wrist flick there is spin)
    residual_scale: tuple | None = None
    speed_scale: float = 0.5
    speed_range: tuple = (0.5, 3.0)
    release_shift_s: float = 0.6                 # reference seconds per unit action
    min_release_frac: float = 0.3                # ignore release commands earlier than this
    release_delay_range: tuple | None = None     # s; None -> profile gripper_open_time_s x (0.5, 1.5)
    release_latency_range: tuple = (0.0, 0.03)   # s added: trigger + command latency (joint_states, DDS)
    start_noise_rad: float = 0.03
    shaping_gamma: float = 0.99
    deterministic: bool = False                  # play/export: env i -> reference i % R, nominal everything
    joint_max_override: dict | None = None       # {arm joint index: max commanded position (rad)}, e.g. keep the
                                                 # elbow bent: {4: -0.6} (ra4 = 0 is a straight arm, a singularity)
    vel_cap: float | tuple | None = None         # rad/s: commanded speed <= min(profile limit, cap); per joint ok
    accel_cap: float | tuple | None = None       # rad/s^2: same for acceleration
    # target heading (deg, + = left of the robot's forward x). Training: drawn per episode from heading_range
    # and observed by the policy; on the robot the base is turned so that heading points at the target.
    heading_range: tuple | None = None           # None -> always heading_fixed
    heading_fixed: float = 0.0                   # play / export / no range
    # aggressiveness a: requested release speed a*goal.speed and spin a*goal.spin. Training: drawn per episode
    # from aggression_range and observed by the policy; export: --aggression. None -> maximise (old reward).
    aggression_range: tuple | None = None
    aggression_fixed: float = 1.0
    # pitch (deg, nose up): the disc's leading edge (towards the target) above its trailing edge at release.
    # Training: drawn per episode from pitch_range and observed; export: --pitch. None -> "flat" reward (old).
    pitch_range: tuple | None = None
    pitch_fixed: float | None = None
    grip_strength: float = 1.0                   # multiplies the measured grip (squeeze force -> every limit);
                                                 # user: the gripper closed at position 1.0 holds firmly
    # --- friction grip (see module doc); the slip check runs in training and in play
    slip_model: bool = True
    slip_substeps: int = 2                       # physics steps over the limit before the disc slips out
    load_filter_s: float = 0.01                  # low-pass time constant of the grip load
    # --- domain randomisation done by this term at reset (ignored when deterministic)
    joint_offset_rad: float = 0.01               # calibration error: real joint = commanded + offset
    # the fingers never open perfectly evenly: a random push on the disc as it leaves (world, per axis)
    release_push_mps: float = 0.2
    release_push_radps: float = 2.0
    goal: ThrowGoal = field(default_factory=ThrowGoal)

    def build(self, env: ManagerBasedRlEnv) -> ThrowAction:
        return ThrowAction(self, env)


class ThrowAction(ActionTerm):
    cfg: ThrowActionCfg

    def __init__(self, cfg: ThrowActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        dev = self.device
        n = self.num_envs
        prof: ArmProfile = get_profile(cfg.robot)
        self.profile = prof
        self.arm_joints = prof.arm_joints
        J = self.nj = len(prof.arm_joints)
        self._frisbee = env.scene[cfg.frisbee_name]
        ids, _ = self._entity.find_joints(list(prof.arm_joints), preserve_order=True)
        self._arm_ids = torch.tensor(ids, device=dev, dtype=torch.long)
        gids, _ = self._entity.find_joints(list(prof.finger_joints), preserve_order=True)
        self._finger_ids = torch.tensor(gids, device=dev, dtype=torch.long)
        self._grip_act_ids = torch.tensor(self._entity.find_joints(list(prof.finger_actuated))[0], device=dev)
        self._gripper_body = self._entity.find_bodies([prof.gripper_body])[0][0]
        self._closed, self._open = prof.finger_closed, prof.finger_open
        t_open = prof.gripper_open_time_s
        self._delay_range = cfg.release_delay_range or (0.5 * t_open, 1.5 * t_open)

        refs, q, length, release, dt = load_references(cfg.reference_files or prof.reference_files)
        if q.shape[2] != J:
            raise ValueError(f"references have {q.shape[2]} joints, {prof.name} has {J}")
        sel = list(cfg.reference_ids) if cfg.reference_ids is not None else list(range(len(refs)))
        self.ref_names = [refs[i].name for i in sel]
        self._ref_q = torch.tensor(q[sel], device=dev, dtype=torch.float32)          # (R, N, J)
        self._ref_len = torch.tensor(length[sel], device=dev, dtype=torch.float32)   # (R,)
        self._ref_rel = torch.tensor(release[sel], device=dev, dtype=torch.float32)
        self._ref_dt = dt

        lim = self._entity.data.soft_joint_pos_limits[0, self._arm_ids]            # (J, 2)
        self._qmin, self._qmax = lim[:, 0].clone(), lim[:, 1].clone()
        for j, v in (cfg.joint_max_override or {}).items():
            self._qmax[j] = min(float(self._qmax[j]), v)

        model = env.sim.mj_model
        self._weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD_NAME)
        if self._weld < 0:
            raise RuntimeError(f"equality '{WELD_NAME}' not in the scene (SceneCfg.spec_fn missing?)")
        # grip geometry, per env (randomised by the grip_randomization event, nominal otherwise)
        self._rel_quat0 = torch.tensor(prof.grip_rel_quat, device=dev, dtype=torch.float32)
        self._pinch_g = torch.tensor(prof.pinch_point, device=dev, dtype=torch.float32)
        rel_pos0 = torch.tensor(prof.grip_rel_pos, device=dev, dtype=torch.float32)
        self._pinch_d0 = quat_apply_inverse(self._rel_quat0[None], (self._pinch_g - rel_pos0)[None])[0]
        self.rel_pos = rel_pos0.repeat(n, 1)
        self.rel_quat = self._rel_quat0.repeat(n, 1)
        self.pinch_d = self._pinch_d0.repeat(n, 1)                 # pinch point in the disc frame
        g = dict(prof.grip)
        g["squeeze_force_n"] = g["squeeze_force_n"] * cfg.grip_strength   # scales every grip limit
        self.grip_nominal = g
        self.mu = torch.full((n,), g["mu"], device=dev)
        self.squeeze = torch.full((n,), g["squeeze_force_n"], device=dev)
        self.pad_t = torch.full((n,), g["pad_half_tangential_m"], device=dev)
        self.pad_r = torch.full((n,), g["pad_half_radial_m"], device=dev)
        self.capacity_scale = 1.0                                  # curriculum: generous early, 1.0 later
        self.joint_offset = torch.zeros(n, J, device=dev)

        self._res_scale = torch.tensor(cfg.residual_scale or prof.residual_scale, device=dev, dtype=torch.float32)
        # commanded targets never move faster than the joint velocity limits: the exported trajectory must
        # pass the robot controller's speed check (HEADBands cartesian_follow.py --max-joint-vel)
        self._vmax = torch.tensor(prof.vel_limits, device=dev, dtype=torch.float32)
        self._amax = torch.tensor(prof.accel_limits, device=dev, dtype=torch.float32)
        if cfg.vel_cap is not None:
            self._vmax = torch.minimum(self._vmax, torch.as_tensor(cfg.vel_cap, device=dev, dtype=torch.float32))
        if cfg.accel_cap is not None:
            self._amax = torch.minimum(self._amax, torch.as_tensor(cfg.accel_cap, device=dev, dtype=torch.float32))
        self.aggression = torch.full((n,), cfg.aggression_fixed, device=dev)
        self.pitch = torch.full((n,), math.radians(cfg.pitch_fixed or 0.0), device=dev)
        self.heading = torch.full((n,), math.radians(cfg.heading_fixed), device=dev)
        self.forward = torch.zeros(n, 3, device=dev)
        self._set_heading(torch.arange(n, device=dev))
        self._raw = torch.zeros(n, J + 2, device=dev)
        self.phase = torch.zeros(n, device=dev)                 # seconds along the reference
        self.speed = torch.ones(n, device=dev)
        self.ref_id = torch.zeros(n, device=dev, dtype=torch.long)
        self.q_target = torch.zeros(n, J, device=dev)
        self.q_ref = torch.zeros(n, J, device=dev)
        self.v_cmd = torch.zeros(n, J, device=dev)
        self.held = torch.ones(n, device=dev, dtype=torch.bool)       # weld on
        self.opening = torch.zeros(n, device=dev, dtype=torch.bool)   # open commanded, weld still on
        self.timer = torch.zeros(n, device=dev)
        self.just_released = torch.zeros(n, device=dev, dtype=torch.bool)
        self.just_slipped = torch.zeros(n, device=dev, dtype=torch.bool)
        self.slipped = torch.zeros(n, device=dev, dtype=torch.bool)
        self.release_time = torch.full((n,), float("inf"), device=dev)   # release or slip
        self.release_state = torch.zeros(n, 13, device=dev)          # pos, quat(wxyz), lin vel, ang vel (world)
        self.release_phase = torch.zeros(n, device=dev)
        self.open_cmd_time = torch.full((n,), float("inf"), device=dev)   # when the gripper was told to open
        self.phi_prev = torch.zeros(n, device=dev)
        self.grip_u = torch.zeros(n, device=dev)                     # grip load / capacity, now
        self.grip_u_max = torch.zeros(n, device=dev)                 # ... max this episode
        self._over = torch.zeros(n, device=dev, dtype=torch.int32)
        self._u_filt = torch.zeros(n, device=dev)
        self._grace = torch.zeros(n, device=dev, dtype=torch.int32)
        self._q_prev = torch.zeros(n, J, device=dev)
        self._v_prev = torch.zeros(n, J, device=dev)
        self._sub = 0

    # ------------------------------------------------------------------ properties
    @property
    def action_dim(self) -> int:
        return self.nj + 2

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
        """Linear interpolation of the reference joint positions, (B, J)."""
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
        J = self.nj
        self.speed = (1.0 + c.speed_scale * a[:, J]).clamp(*c.speed_range)
        self.phase = torch.minimum(self.phase + self.speed * self._env.step_dt, self.ref_length)
        self.q_ref = self.reference_at(self.phase)
        desired = torch.clamp(self.q_ref + self._res_scale * a[:, :J], self._qmin, self._qmax)
        # second-order limiter: commanded targets respect both the velocity and the acceleration limit,
        # braking in time (v <= sqrt(2 a |err|)) so they never overshoot. Without the acceleration limit a
        # policy can reverse a joint at full speed between two 20 ms samples, and the robot's spline
        # trajectory controller turns that into 1.5x the speed limit.
        dt = self._env.step_dt
        a_dt = self._amax * dt
        self._q_prev = self.q_target.clone()
        self._v_prev = self.v_cmd.clone()
        self._sub = 0

        def brake(dist):        # fastest v that can still stop within dist: v*dt + v^2/(2a) <= dist
            return -a_dt + torch.sqrt(a_dt * a_dt + 2 * self._amax * dist.clamp(min=0.0))

        err = desired - self.q_target
        v_des = torch.sign(err) * torch.minimum(self._vmax, brake(err.abs()))
        # never head into a joint limit faster than it can be stopped
        v_des = torch.clamp(v_des, -brake(self.q_target - self._qmin), brake(self._qmax - self.q_target))
        self.v_cmd = torch.clamp(self.v_cmd + torch.clamp(v_des - self.v_cmd, -a_dt, a_dt), -self._vmax, self._vmax)
        self.q_target = torch.clamp(self.q_target + self.v_cmd * dt, self._qmin, self._qmax)
        # release command (latched); forced at the end of the reference so every episode throws
        when = self.ref_release + c.release_shift_s * a[:, J + 1]
        frac = self.phase / self.ref_length
        cmd = ((self.phase >= when) & (frac >= c.min_release_frac)) | (self.phase >= self.ref_length - 1e-6)
        start = cmd & self.held & ~self.opening
        if start.any():
            lo, hi = self._delay_range
            k = int(start.sum())
            if c.deterministic:
                delay = torch.full((k,), 0.5 * (lo + hi) + 0.5 * sum(c.release_latency_range), device=self.device)
            else:
                ll, lh = c.release_latency_range
                delay = lo + (hi - lo) * torch.rand(k, device=self.device) + ll + (lh - ll) * torch.rand(k, device=self.device)
            self.timer[start] = delay
            self.open_cmd_time[start] = self.env_time()[start]
            self.opening |= start
        self.just_released[:] = False
        self.just_slipped[:] = False

    def apply_actions(self) -> None:
        # the 50 Hz targets are interpolated in between (like the robot's trajectory controller), with the
        # commanded velocity as feed-forward; the joints end up at command + calibration offset
        self._sub += 1
        frac = min(self._sub / self._env.cfg.decimation, 1.0)
        q_cmd = self._q_prev + frac * (self.q_target - self._q_prev)
        v_cmd = self._v_prev + frac * (self.v_cmd - self._v_prev)   # a step in the velocity target is a kick
        self._entity.set_joint_position_target(q_cmd + self.joint_offset, joint_ids=self._arm_ids)
        self._entity.set_joint_velocity_target(v_cmd, joint_ids=self._arm_ids)
        grip = torch.where(self.opening | ~self.held, self._open, self._closed)
        self._entity.set_joint_position_target(grip[:, None].expand(-1, len(self._grip_act_ids)),
                                               joint_ids=self._grip_act_ids)
        if self.cfg.slip_model:
            self._check_slip()
        if self.opening.any():
            self.timer -= self._env.physics_dt * self.opening.float()
            fire = self.opening & (self.timer <= 0.0)
            if fire.any():
                ids = fire.nonzero(as_tuple=False).squeeze(-1)
                state = self._frisbee_state(ids)                         # exact state at the instant
                if not self.cfg.deterministic:
                    k = len(ids)
                    push = torch.cat([self.cfg.release_push_mps * (2 * torch.rand(k, 3, device=self.device) - 1),
                                      self.cfg.release_push_radps * (2 * torch.rand(k, 3, device=self.device) - 1)], 1)
                    state[:, 7:13] += push
                    self._frisbee.write_root_link_velocity_to_sim(state[:, 7:13], env_ids=ids)
                self.release_state[ids] = state
                self._let_go(ids)
                self.just_released[ids] = True

    def _let_go(self, ids: torch.Tensor) -> None:
        self._env.sim.data.eq_active[ids, self._weld] = False
        self.held[ids] = False
        self.opening[ids] = False
        self.release_time[ids] = self.env_time()[ids]
        self.release_phase[ids] = self.phase[ids]

    def uses_pitch(self) -> bool:
        return self.cfg.pitch_range is not None or self.cfg.pitch_fixed is not None

    def targets(self):
        """(speed target, spin target, disc-normal target) per env; None where not commanded."""
        n_t = None
        if self.uses_pitch():
            p = self.pitch[:, None]
            z = torch.zeros_like(self.forward)
            z[:, 2] = 1.0
            n_t = torch.cos(p) * z - torch.sin(p) * self.forward    # leaning back from the target = nose up
        if self.cfg.aggression_range is None and self.cfg.aggression_fixed >= 1.0:
            return None, None, n_t
        g = self.cfg.goal
        return self.aggression * g.speed, self.aggression * g.spin, n_t

    def _set_heading(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        if c.aggression_range is not None and not c.deterministic:
            lo_a, hi_a = c.aggression_range
            self.aggression[env_ids] = lo_a + (hi_a - lo_a) * torch.rand(len(env_ids), device=self.device)
        else:
            self.aggression[env_ids] = c.aggression_fixed
        if c.pitch_range is not None and not c.deterministic:
            lo_p, hi_p = (math.radians(x) for x in c.pitch_range)
            self.pitch[env_ids] = lo_p + (hi_p - lo_p) * torch.rand(len(env_ids), device=self.device)
        else:
            self.pitch[env_ids] = math.radians(c.pitch_fixed or 0.0)
        if c.heading_range is not None and not c.deterministic:
            lo, hi = (math.radians(x) for x in c.heading_range)
            self.heading[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
        else:
            self.heading[env_ids] = math.radians(c.heading_fixed)
        h = self.heading[env_ids]
        self.forward[env_ids] = torch.stack([torch.cos(h), torch.sin(h), torch.zeros_like(h)], 1)

    # ------------------------------------------------------------------ friction grip
    def grip_wrench(self):
        """Force and moment the fingers apply to the disc, at the pinch point, in the disc frame.
        From the weld's generalized constraint force on the frisbee's free joint (world force at the COM,
        torque in the body frame), i.e. what the last physics step needed to carry the disc."""
        idx = self._frisbee.data.indexing
        f = self._env.sim.data.qfrc_constraint[:, idx.free_joint_v_adr]
        quat = self._env.sim.data.qpos[:, idx.free_joint_q_adr][:, 3:7]
        F = quat_apply_inverse(quat, f[:, :3])
        tau = f[:, 3:6] - torch.cross(self.pinch_d, F, dim=1)
        return F, tau

    def grip_utilization(self) -> torch.Tensor:
        F, tau = self.grip_wrench()
        r = self.pinch_d.clone()
        r[:, 2] = 0.0
        r = r / (r.norm(dim=1, keepdim=True) + 1e-9)                   # disc centre -> pinch, in plane
        t = torch.stack([-r[:, 1], r[:, 0], torch.zeros_like(r[:, 0])], dim=1)   # along the rim
        Fs = self.squeeze * self.capacity_scale
        r_eff = 0.54 * torch.sqrt(self.pad_t ** 2 + self.pad_r ** 2)   # mean lever of a rectangular patch
        u_fric = torch.sqrt((F[:, :2].norm(dim=1) / (2 * self.mu * Fs)) ** 2
                            + (tau[:, 2] / (2 * self.mu * Fs * r_eff)) ** 2)
        u_lift = (F[:, 2].abs() / (2 * Fs) + (tau * r).sum(1).abs() / (2 * Fs * self.pad_t)
                  + (tau * t).sum(1).abs() / (2 * Fs * self.pad_r))
        return torch.maximum(u_fric, u_lift)

    def _check_slip(self) -> None:
        watch = self.held & ~self.opening & (self._grace <= 0)
        self._grace -= 1
        # 10 ms low-pass: pads are compliant and slipping takes a sustained load; it also removes the
        # step-to-step ringing of the weld force that a stiff 5 ms simulation has
        a = self._env.physics_dt / (self._env.physics_dt + self.cfg.load_filter_s)
        self._u_filt = torch.where(watch, self._u_filt + a * (self.grip_utilization() - self._u_filt),
                                   torch.zeros_like(self._u_filt))
        u = self._u_filt
        self.grip_u = u
        self.grip_u_max = torch.maximum(self.grip_u_max, u)
        self._over = torch.where(watch & (u > 1.0), self._over + 1, torch.zeros_like(self._over))
        slip = self._over >= self.cfg.slip_substeps
        if slip.any():
            ids = slip.nonzero(as_tuple=False).squeeze(-1)
            self.release_state[ids] = self._frisbee_state(ids)
            self._let_go(ids)
            self.slipped[ids] = True
            self.just_slipped[ids] = True
            self._over[ids] = 0

    def _frisbee_state(self, ids: torch.Tensor) -> torch.Tensor:
        idx = self._frisbee.data.indexing
        qpos = self._env.sim.data.qpos[ids][:, idx.free_joint_q_adr]       # pos(3) quat(4)
        qvel = self._env.sim.data.qvel[ids][:, idx.free_joint_v_adr]       # lin (world), ang (body)
        ang_w = quat_apply(qpos[:, 3:7], qvel[:, 3:6])
        return torch.cat([qpos, qvel[:, :3], ang_w], dim=1)

    # ------------------------------------------------------------------ domain randomisation of the grip
    def randomize_grip(self, env_ids: torch.Tensor, depth_m, along_rim_m, tilt_deg, mu_scale, squeeze_scale,
                       pad_scale) -> None:
        """Where and how firmly the disc is held this episode. Writes the weld pose for these envs."""
        k = len(env_ids)
        dev = self.device

        def u(rng, shape=(k,)):
            return rng[0] + (rng[1] - rng[0]) * torch.rand(shape, device=dev)

        r = self._pinch_d0.clone()
        r[2] = 0.0
        r = r / r.norm()
        t = torch.stack([-r[1], r[0], torch.zeros((), device=dev)])
        # deeper (+) / shallower (-) between the pads, and along the rim: the pinch moves on the disc
        p_d = self._pinch_d0[None] - u(depth_m)[:, None] * r[None] + u(along_rim_m)[:, None] * t[None]
        # the disc sits tilted in the fingers (about the radial and the rim axis)
        ang = torch.deg2rad(torch.tensor(tilt_deg, device=dev, dtype=torch.float32))
        rot = u((-1.0, 1.0), (k, 2)) * ang
        q_r = quat_from_angle_axis(rot[:, 0], r[None].expand(k, 3))
        q_t = quat_from_angle_axis(rot[:, 1], t[None].expand(k, 3))
        rel_q = quat_mul(self._rel_quat0[None].expand(k, 4), quat_mul(q_r, q_t))
        rel_p = self._pinch_g[None] - quat_apply(rel_q, p_d)     # the pinch point stays between the pads
        self.pinch_d[env_ids] = p_d
        self.rel_pos[env_ids] = rel_p
        self.rel_quat[env_ids] = rel_q
        eq = self._env.sim.model.eq_data
        eq[env_ids, self._weld, 3:6] = rel_p
        eq[env_ids, self._weld, 6:10] = rel_q
        g = self.grip_nominal
        self.mu[env_ids] = g["mu"] * u(mu_scale)
        self.squeeze[env_ids] = g["squeeze_force_n"] * u(squeeze_scale)
        ps = u(pad_scale)
        self.pad_t[env_ids] = g["pad_half_tangential_m"] * ps
        self.pad_r[env_ids] = g["pad_half_radial_m"] * ps

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        k = len(env_ids)
        if k == 0:
            return
        self._raw[env_ids] = 0.0
        self._set_heading(env_ids)
        if self.cfg.deterministic:
            self.ref_id[env_ids] = env_ids % self._ref_q.shape[0]
        else:
            self.ref_id[env_ids] = torch.randint(0, self._ref_q.shape[0], (k,), device=self.device)
            self.joint_offset[env_ids] = self.cfg.joint_offset_rad * (2 * torch.rand(k, self.nj, device=self.device) - 1)
        self.phase[env_ids] = 0.0
        self.speed[env_ids] = 1.0
        self.held[env_ids] = True
        self.opening[env_ids] = False
        self.timer[env_ids] = 0.0
        self.just_released[env_ids] = False
        self.just_slipped[env_ids] = False
        self.slipped[env_ids] = False
        self.release_time[env_ids] = float("inf")
        self.release_state[env_ids] = 0.0
        self.release_phase[env_ids] = 0.0
        self.open_cmd_time[env_ids] = float("inf")
        self.grip_u[env_ids] = 0.0
        self.grip_u_max[env_ids] = 0.0
        self._over[env_ids] = 0
        self._u_filt[env_ids] = 0.0
        self._grace[env_ids] = 2                 # the constraint force of the step before reset is stale
        self._env.sim.data.eq_active[env_ids, self._weld] = True

        q0 = self._ref_q[self.ref_id[env_ids], 0]
        if not self.cfg.deterministic:
            q0 = q0 + self.cfg.start_noise_rad * (2 * torch.rand_like(q0) - 1)
        q0 = torch.clamp(q0, self._qmin, self._qmax)
        self.q_ref[env_ids] = q0
        self.q_target[env_ids] = q0
        self._q_prev[env_ids] = q0
        self._v_prev[env_ids] = 0.0
        self.v_cmd[env_ids] = 0.0
        q_real = q0 + self.joint_offset[env_ids]
        self._entity.write_joint_state_to_sim(q_real, torch.zeros_like(q0), joint_ids=self._arm_ids, env_ids=env_ids)
        closed = torch.full((k, len(self._finger_ids)), self._closed, device=self.device)
        self._entity.write_joint_state_to_sim(closed, torch.zeros_like(closed), joint_ids=self._finger_ids,
                                              env_ids=env_ids)
        # put the frisbee in the grip: forward kinematics, then the weld's relative pose
        self._env.sim.forward()
        g = self._entity.data.body_link_pose_w[env_ids, self._gripper_body]     # (k, 7)
        pos = g[:, :3] + quat_apply(g[:, 3:7], self.rel_pos[env_ids])
        quat = quat_mul(g[:, 3:7], self.rel_quat[env_ids])
        state = torch.cat([pos, quat, torch.zeros(k, 6, device=self.device)], dim=1)
        self._frisbee.write_root_state_to_sim(state, env_ids=env_ids)
        self._env.sim.forward()
        self.phi_prev[env_ids] = hand_quality(self._env, self)[0][env_ids]


@requires_model_fields("eq_data")
def grip_randomization(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, depth_m=(-0.01, 0.01),
                       along_rim_m=(-0.02, 0.02), tilt_deg=5.0, mu_scale=(0.6, 1.5), squeeze_scale=(0.6, 1.0),
                       pad_scale=(0.7, 1.2)) -> None:
    """Reset event: grip position on the rim, tilt in the fingers, friction, squeeze force, pad size."""
    t = _term(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    t.randomize_grip(env_ids.to(env.device, dtype=torch.long), depth_m, along_rim_m, tilt_deg, mu_scale,
                     squeeze_scale, pad_scale)


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
    return throw_quality(v, w, n, term.cfg.goal, term.forward, *term.targets())  # noqa: targets = (v, w, n)


# ----------------------------------------------------------------------------- observations
# Open-loop actor: the exported throw is played open loop on the robot (a joint CSV), so the actor only
# sees what is known without sensors (phase, the reference, its own commanded targets). The critic also
# gets the measured state and the randomised parameters (asymmetric actor-critic): domain randomisation
# then makes ONE trajectory robust, the one that is exported, instead of a feedback policy we can't deploy.
def throw_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    t = _term(env)
    frac = t.phase / t.ref_length
    to_release = (t.ref_release - t.phase)
    return torch.stack([frac, torch.sin(math.pi * frac), torch.cos(math.pi * frac), to_release,
                        t.speed, t.held.float(), t.opening.float()], dim=1)


def target_pitch(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Commanded nose-up pitch at release, in units of 15 deg."""
    return (_term(env).pitch / math.radians(15.0))[:, None]


def target_aggression(env: ManagerBasedRlEnv) -> torch.Tensor:
    """How hard to throw this episode (0..1 of the max speed / spin)."""
    return _term(env).aggression[:, None]


def target_heading(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Where the throw should go this episode (cos, sin of the heading)."""
    t = _term(env)
    return torch.stack([torch.cos(t.heading), torch.sin(t.heading)], dim=1)


def reference_preview(env: ManagerBasedRlEnv, horizons: tuple = (0.0, 0.1, 0.25)) -> torch.Tensor:
    """Reference joint positions at the current phase and a little ahead, relative to the measured joints."""
    t = _term(env)
    q = t._entity.data.joint_pos[:, t._arm_ids]
    return torch.cat([t.reference_at(t.phase + h) - q for h in horizons], dim=1)


def reference_preview_cmd(env: ManagerBasedRlEnv, horizons: tuple = (0.0, 0.1, 0.25)) -> torch.Tensor:
    """Same, relative to the commanded targets (no sensing: open loop)."""
    t = _term(env)
    return torch.cat([t.reference_at(t.phase + h) - t.q_target for h in horizons], dim=1)


def commanded_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    t = _term(env)
    return torch.cat([t.q_target, t.v_cmd], dim=1)


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


def privileged_grip(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic only: this episode's grip (friction, force, pads, where the disc sits), its load, the
    frisbee mass and the joint calibration offsets."""
    t = _term(env)
    g = t.grip_nominal
    fb = int(t._frisbee.indexing.body_ids[0])
    mass = (env.sim.model.body_mass[:, fb] / 0.175).expand(t.num_envs)
    return torch.cat([(t.mu / g["mu"])[:, None], (t.squeeze / g["squeeze_force_n"])[:, None],
                      (t.pad_t / g["pad_half_tangential_m"])[:, None], (t.pinch_d - t._pinch_d0) * 20.0,
                      t.rel_quat, t.grip_u[:, None], mass[:, None], t.joint_offset * 50.0], dim=1)


# ----------------------------------------------------------------------------- rewards
def release_quality(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-shot reward on the step the frisbee leaves the hand, scored on its exact release state."""
    t = _term(env)
    s = t.release_state
    pos, quat, v, w = s[:, :3], s[:, 3:7], s[:, 7:10], s[:, 10:13]
    n = quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(quat.shape[0], 3))
    q, comps = throw_quality(v, w, n, t.cfg.goal, t.forward, *t.targets())
    jr = t.just_released.float()
    ended = t.just_released | t.just_slipped
    if ended.any():
        log = env.extras.setdefault("log", {})
        log["Grip/slip_rate"] = t.just_slipped[ended].float().mean()
        log["Grip/u_max"] = t.grip_u_max[ended].mean()
        log["Grip/capacity_scale"] = torch.tensor(float(t.capacity_scale))
    if jr.any():
        log = env.extras.setdefault("log", {})
        m = t.just_released
        for k, c in comps.items():
            log[f"Release/{k}"] = c[m].mean()
        log["Release/quality"] = q[m].mean()
        log["Release/phase_frac"] = (t.release_phase / t.ref_length)[m].mean()
        if t.uses_pitch():                                   # achieved nose-up pitch vs the command, in degrees
            pitch = torch.rad2deg(torch.asin((-(n * t.forward).sum(1)).clamp(-1.0, 1.0)))
            log["Release/pitch_deg"] = pitch[m].mean()
            log["Release/pitch_err_deg"] = (pitch - torch.rad2deg(t.pitch))[m].abs().mean()
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


def track_reference(env: ManagerBasedRlEnv, std: float = 0.3, n_joints: int | None = None) -> torch.Tensor:
    """Stay close to the human throw (annealed by the curriculum). n_joints: only the first n arm joints
    (the phones track the wrist position, which ra0-ra3 set; the wrist joints carry no human information)."""
    t = _term(env)
    n = n_joints or t.nj
    q = t._entity.data.joint_pos[:, t._arm_ids[:n]]
    err = ((q - t.q_ref[:, :n]) ** 2).sum(1)
    return torch.exp(-err / std ** 2) * t.held.float()


def arm_velocity_excess(env: ManagerBasedRlEnv) -> torch.Tensor:
    t = _term(env)
    v = t._entity.data.joint_vel[:, t._arm_ids].abs()
    return ((v - t._vmax).clamp_min(0.0) ** 2).sum(1)


def body_contact(env: ManagerBasedRlEnv, sensor_names: tuple) -> torch.Tensor:
    hit = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name in sensor_names:
        data = env.scene[name].data
        hit |= data.found.reshape(data.found.shape[0], -1).any(dim=1)
    return hit.float()


def grip_margin(env: ManagerBasedRlEnv, threshold: float = 0.7) -> torch.Tensor:
    """Keep the grip load below `threshold` of what friction can hold (margin for an unknown mu)."""
    t = _term(env)
    return (t.grip_u - threshold).clamp_min(0.0) ** 2 * t.held.float()


def grip_slip(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-shot on the step the disc slipped out of the fingers."""
    t = _term(env)
    return t.just_slipped.float() / env.step_dt


# ----------------------------------------------------------------------------- curriculum
def grip_capacity_curriculum(env: ManagerBasedRlEnv, env_ids, stages: list) -> float:
    """Start with a firmer grip than the real one, so the policy can find a throw before it must also
    respect friction; end at the real capacity (scale 1)."""
    t = _term(env)
    for st in stages:
        if env.common_step_counter >= st["step"]:
            t.capacity_scale = float(st["scale"])
    return t.capacity_scale


# ----------------------------------------------------------------------------- terminations
def thrown(env: ManagerBasedRlEnv, after_s: float = 0.1) -> torch.Tensor:
    """End the episode shortly after release (or slip): only the release state is scored."""
    t = _term(env)
    return (t.env_time() - t.release_time) >= after_s


# ----------------------------------------------------------------------------- servo gain randomisation
@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_servo_gains(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, kp_range=(0.8, 1.2), kd_range=(0.8, 1.2),
                   entity: str = "robot") -> None:
    """Scale kp and kd of every BuiltinPdActuator (position element: kp, velocity element: kd).
    (mjlab's dr.pd_gains indexes its actuator list by MuJoCo element id, which breaks for PD pairs.)"""
    from mjlab.actuator.builtin_actuator import BuiltinPdActuator
    asset = env.scene[entity]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, dtype=torch.long)
    gain0 = env.sim.get_default_field("actuator_gainprm")
    bias0 = env.sim.get_default_field("actuator_biasprm")
    m = env.sim.model
    for act in asset.actuators:
        if not isinstance(act, BuiltinPdActuator):
            continue
        ids = act.global_ctrl_ids
        n = act.num_targets
        pos, vel = ids[:n], ids[n:]
        kp = kp_range[0] + (kp_range[1] - kp_range[0]) * torch.rand(len(env_ids), n, device=env.device)
        kd = kd_range[0] + (kd_range[1] - kd_range[0]) * torch.rand(len(env_ids), n, device=env.device)
        e = env_ids[:, None]
        m.actuator_gainprm[e, pos, 0] = gain0[pos, 0] * kp
        m.actuator_biasprm[e, pos, 1] = bias0[pos, 1] * kp
        m.actuator_gainprm[e, vel, 0] = gain0[vel, 0] * kd
        m.actuator_biasprm[e, vel, 2] = bias0[vel, 2] * kd


# ----------------------------------------------------------------------------- what shakes the real robot
def joint_torque_excess(env: ManagerBasedRlEnv, limits: tuple, threshold: float = 0.7) -> torch.Tensor:
    """Servo torque above `threshold` x its limit (N m, per arm joint in order; only the first len(limits)
    joints). The real shoulder saturated at ~31 N m in the 2 rad/s system-ID run and the base shook."""
    t = _term(env)
    n = len(limits)
    ids = t._entity.indexing.joint_v_adr[t._arm_ids[:n]]
    tau = env.sim.data.qfrc_actuator[:, ids].abs()
    lim = torch.tensor(limits, device=env.device, dtype=tau.dtype)
    return ((tau / lim - threshold).clamp_min(0.0) ** 2).sum(1)


class ArmReaction:
    """Force the arm pushes into the torso beyond gravity: M * (acceleration of the arm's centre of mass),
    by finite differences at the policy rate. The simulated torso is fixed, the real base shakes."""
    def __init__(self):
        self.prev_c = self.prev_v = None
        self.bodies = None

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        t = _term(env)
        if self.bodies is None:
            ids, _ = t._entity.find_bodies([r"link_ra.*", t.profile.gripper_body, r"finger_.*"])
            self.bodies = t._entity.indexing.body_ids[torch.tensor(ids, device=env.device)].long()
            self.mass = env.sim.model.body_mass[:, self.bodies]
        m = self.mass
        c = (env.sim.data.xipos[:, self.bodies] * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        dt = env.step_dt
        if self.prev_c is None or self.prev_c.shape != c.shape:
            self.prev_c, self.prev_v = c.clone(), torch.zeros_like(c)
        v = (c - self.prev_c) / dt
        a = (v - self.prev_v) / dt
        fresh = env.episode_length_buf <= 1                      # just reset: no valid history
        a[fresh] = 0.0
        self.prev_c, self.prev_v = c.clone(), v.clone()
        f = m.sum(1) * a.norm(dim=1)                              # N
        return (f / (m.sum(1) * 9.81)) ** 2                      # in units of the arm's weight


arm_reaction = ArmReaction()
