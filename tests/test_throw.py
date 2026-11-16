"""hack_rl: release score, robot model, references, and (GPU) the throw environment."""
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from hack_rl.reference import DEFAULT_FILES, load_references
from hack_rl.robot.panther import FRISBEE_XML, GRIP_REL_POS, ROBOT_XML
from hack_rl.tasks.throw.mdp import ThrowGoal, throw_quality

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU (MuJoCo Warp)")


def _q(v, w, n):
    t = lambda x: torch.tensor([x], dtype=torch.float32)  # noqa: E731
    return throw_quality(t(v), t(w), t(n), ThrowGoal())


def test_ideal_throw_scores_about_one():
    el = math.radians(5)
    v = [4.0 * math.cos(el), 0.0, 4.0 * math.sin(el)]
    q, c = _q(v, [0, 0, 15.0], [0, 0, 1.0])
    assert q.item() > 0.97 and c["flat"].item() == pytest.approx(1.0)


def test_each_signal_matters():
    good, _ = _q([4, 0, 0.35], [0, 0, 15], [0, 0, 1])
    vertical_disc, c = _q([4, 0, 0.35], [15, 0, 0], [1, 0, 0])
    backwards, _ = _q([-4, 0, 0.35], [0, 0, 15], [0, 0, 1])
    no_spin, _ = _q([4, 0, 0.35], [0, 0, 0], [0, 0, 1])
    upside_down, _ = _q([4, 0, 0.35], [0, 0, 15], [0, 0, -1])
    for worse in (vertical_disc, backwards, no_spin, upside_down):
        assert worse.item() < good.item() - 0.15
    assert c["flat"].item() == 0.0


def test_robot_model_loads_and_is_collision_free_at_start():
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    d = mujoco.MjData(m)
    assert [m.joint(i).name for i in range(8)] == [f"joint_ra{i}" for i in range(8)]
    assert m.site("grip_site").id >= 0 and m.body("gripper").id >= 0
    _, Q, *_ = load_references(DEFAULT_FILES)
    for r in range(Q.shape[0]):
        d.qpos[:8] = Q[r, 0]
        mujoco.mj_forward(m, d)
        assert d.ncon == 0
    f = mujoco.MjModel.from_xml_path(str(FRISBEE_XML))
    assert f.body_mass[1] == pytest.approx(0.175)


def test_adapted_references_release_flat_and_forward():
    refs, Q, length, release, dt = load_references(DEFAULT_FILES)
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    d = mujoco.MjData(m)
    g = m.body("gripper").id
    for r, ref in enumerate(refs):
        assert 0 < release[r] < length[r]
        k = int(round(release[r] / dt))
        d.qpos[:8] = ref.q[k]
        mujoco.mj_kinematics(m, d)
        normal = d.xmat[g].reshape(3, 3)[:, 1]
        assert normal[2] > 0.9                                   # disc flat and upright at release
        assert np.all(ref.q >= m.jnt_range[:8, 0] - 1e-6) and np.all(ref.q <= m.jnt_range[:8, 1] + 1e-6)
    assert np.linalg.norm(GRIP_REL_POS) > 0.25                    # disc centre beyond the finger tips


@gpu
@pytest.mark.parametrize("task", ["Hack-Throw-Frisbee-Panther", "Hack-Throw-Frisbee-Franka"])
def test_env_holds_then_releases_the_frisbee(task):
    import hack_rl.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    term = env.action_manager.get_term("throw")
    fr = env.scene["frisbee"]
    grip = lambda: term._entity.data.body_link_pose_w[:, term._gripper_body, :3]  # noqa: E731
    d0 = (fr.data.root_link_pos_w - grip()).norm(dim=1)
    released = torch.zeros(2, dtype=torch.bool, device="cuda:0")
    for _ in range(400):
        held_before = term.held.clone()
        env.step(torch.zeros(2, term.action_dim, device="cuda:0"))
        d = (fr.data.root_link_pos_w - grip()).norm(dim=1)
        still = held_before & term.held
        assert torch.allclose(d[still], d0[still], atol=0.01)     # rigidly held while the weld is on
        released |= term.just_released
        if released.all():
            break
    assert released.all()                                          # every episode throws
    env.close()


def test_spline_check_matches_controller_behaviour():
    from hack_rl.export_rollout import spline_peak_speed
    t = np.arange(0, 1.0001, 0.02)
    smooth = np.outer(0.5 * (1 - np.cos(np.pi * t)), np.ones(8))     # rest to rest, peak speed pi/2
    assert spline_peak_speed(smooth, t).max() == pytest.approx(np.pi / 2, rel=0.02)
    zigzag = smooth.copy()
    zigzag[1::2] += 0.03                                       # reversals at every sample
    assert spline_peak_speed(zigzag, t).max() > 3.0            # what a chattering policy would cause


@gpu
def test_commanded_targets_respect_speed_and_acceleration_limits():
    import hack_rl.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Hack-Throw-Frisbee-Panther", play=True)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    term = env.action_manager.get_term("throw")
    dt = env.step_dt
    q = [term.q_target.clone()]
    resets = []
    for _ in range(60):                                        # random, aggressive actions
        env.step(torch.sign(torch.randn(8, term.action_dim, device="cuda:0")))
        q.append(term.q_target.clone())
        resets.append(env.episode_length_buf == 0)            # this step ended in a reset: a jump, not motion
    Q = torch.stack(q)                                         # (T, envs, J)
    v = (Q[1:] - Q[:-1]) / dt
    acc = (v[1:] - v[:-1]) / dt
    jump = torch.stack(resets)                                 # (T-1, envs)
    ok_v = (v.abs() <= term._vmax * 1.001)[~jump]
    assert ok_v.all()
    j2 = jump[1:] | jump[:-1]
    if len(jump) > 2:
        j2 = j2 | torch.cat([torch.zeros_like(jump[:1]), jump[:-2]])   # the step after a reset starts from rest
    assert (acc.abs() <= term._amax * 1.001)[~j2].all()
    env.close()


def _train_env(task="Hack-Throw-Frisbee-Panther", n=16):
    import hack_rl.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task)                 # training config: every randomisation on
    cfg.scene.num_envs = n
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    return env, env.action_manager.get_term("throw")


@gpu
def test_grip_randomization_keeps_the_pinch_between_the_pads():
    from mjlab.utils.lab_api.math import quat_apply
    env, t = _train_env()
    eq = env.sim.model.eq_data[:, t._weld]
    assert eq[:, 3:6].std(0).sum() > 1e-3                          # different grip per env
    assert torch.allclose(eq[:, 3:6], t.rel_pos) and torch.allclose(eq[:, 6:10], t.rel_quat)
    pinch = t.rel_pos + quat_apply(t.rel_quat, t.pinch_d)          # pinch point back in the gripper frame
    assert torch.allclose(pinch, t._pinch_g.expand_as(pinch), atol=1e-5)
    assert (t.mu.std() > 0) and (t.squeeze.std() > 0)
    env.close()


@gpu
def test_grip_load_scales_with_squeeze_force_and_a_weak_grip_slips():
    env, t = _train_env()
    zero = torch.zeros(16, t.action_dim, device="cuda:0")
    t.capacity_scale = 1.0
    for _ in range(3):
        env.step(zero)
    u1 = t.grip_utilization().clone()
    t.squeeze *= 0.5
    u2 = t.grip_utilization()
    assert torch.allclose(u2, 2 * u1, rtol=1e-4)                    # load / capacity, capacity ~ squeeze force
    t.capacity_scale = 1e-3                                           # a grip that holds nothing
    holding = t.held.clone()
    slipped = torch.zeros(16, dtype=torch.bool, device="cuda:0")
    for _ in range(3):
        env.step(zero)
        slipped |= t.just_slipped
        assert not t.just_released[t.just_slipped].any()              # a slip is not a release
    assert holding.sum() >= 4 and slipped[holding].all()              # every disc still held slips out
    env.close()


@gpu
@pytest.mark.parametrize("task", ["Hack-Throw-Frisbee-Panther", "Hack-Throw-Frisbee-Franka"])
def test_posture_randomization_tilts_and_moves_the_base(task):
    env, t = _train_env(task, n=64)
    root = t._entity.data.root_link_pose_w
    z = root[:, 2] - env.scene.env_origins[:, 2]
    assert 0.01 < float(z.max() - z.min()) <= 0.0601                # lift / mount height +-3 cm
    up = torch.zeros(64, 3, device="cuda:0")
    up[:, 2] = 1.0
    from mjlab.utils.lab_api.math import quat_apply
    q0 = torch.tensor(t.profile.robot_cfg().init_state.rot, device="cuda:0").expand(64, 4)
    tilt = torch.rad2deg(torch.acos((quat_apply(root[:, 3:7], up) * quat_apply(q0, up)).sum(1).clamp(-1, 1)))
    assert 0.3 < float(tilt.max()) < 3.0                             # roll/pitch up to 2 deg each
    env.close()


@gpu
def test_release_push_is_what_the_frisbee_leaves_with():
    env, t = _train_env(n=32)
    t.capacity_scale = 100.0                                          # no slips: every env releases
    zero = torch.zeros(32, t.action_dim, device="cuda:0")
    pushes = []
    for _ in range(400):
        env.step(zero)
        ids = t.just_released.nonzero().squeeze(-1)
        if len(ids):
            # right after the release step the disc flies freely: its velocity is the scored (pushed) one
            v = t._frisbee.data.root_link_lin_vel_w[ids]
            assert torch.allclose(v[:, :2], t.release_state[ids, 7:9], atol=0.05)
            pushes.append(ids)
        if sum(len(p) for p in pushes) >= 16:
            break
    assert pushes
    env.close()


def test_backward_or_sideways_throw_scores_nothing():
    goal = ThrowGoal()
    n = torch.tensor([[0.0, 0.0, 1.0]])
    w = torch.tensor([[0.0, 0.0, 20.0]])
    back, _ = throw_quality(torch.tensor([[-6.0, 0.0, 0.3]]), w, n, goal)
    side, _ = throw_quality(torch.tensor([[0.0, 6.0, 0.3]]), w, n, goal)
    fwd, _ = throw_quality(torch.tensor([[4.0, 0.0, 0.35]]), w, n, goal)
    assert back.item() < 0.01 and side.item() < 0.01 and fwd.item() > 0.9


def test_pitch_command_rewards_the_commanded_attitude():
    goal = ThrowGoal()
    f = torch.tensor([[1.0, 0.0, 0.0]])
    v = torch.tensor([[5.0, 0.0, 0.4]])
    w = torch.tensor([[0.0, 0.0, 15.0]])
    p = math.radians(10.0)
    n_cmd = torch.tensor([[-math.sin(p), 0.0, math.cos(p)]])          # nose up 10 deg: normal leans back
    q_ok, c_ok = throw_quality(v, w, n_cmd, goal, f, None, None, n_cmd)
    q_flat, c_flat = throw_quality(v, w, torch.tensor([[0.0, 0.0, 1.0]]), goal, f, None, None, n_cmd)
    q_down, _ = throw_quality(v, w, torch.tensor([[math.sin(p), 0.0, math.cos(p)]]), goal, f, None, None, n_cmd)
    assert c_ok["flat"].item() > 0.99 and q_ok > q_flat > q_down
