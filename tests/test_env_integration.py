"""Integration characterization tests for MyoLegsIm.

Instantiates the real env headless against the real MuJoCo model, but with a
tiny synthetic IK reference (a few frames) instead of the 500 MB dataset.
Pins observation sizes, reset state, heading math, motion bookkeeping and a
deterministic reward/step snapshot. np.random is seeded and heading
randomization disabled so runs are reproducible.
"""
import numpy as np
import joblib
import mujoco
import pytest
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as sRot

import helpers as H
from myohuman.env.myolegs_im import MyoLegsIm

DT = 5 / 150  # control_frequency_inv / sim_timestep_inv == 1/30


def _make_cfg(pose_file, **overrides):
    run = dict(
        headless=True, fast_forward=True, control_mode="direct",
        xml_path=str(H.XML_PATH), initial_pose_file=str(pose_file),
        motion_id=0, im_eval=False, test=True, num_motions=1,
        random_sample=False, random_start=False, recording_biomechanics=False,
        motion_ids_file=None,
        proprioceptive_inputs=[
            "root_height", "root_tilt", "local_body_pos", "local_body_rot",
            "local_body_vel", "local_body_ang_vel", "feet_contacts",
        ],
        task_inputs=["diff_local_body_pos", "diff_local_vel", "local_ref_body_pos"],
    )
    run.update(overrides)
    env = dict(
        sim_timestep_inv=150, control_frequency_inv=5, kp_scale=1.0, kd_scale=1.0,
        clip_actions=True, resampling_interval=100, termination_distance=0.15,
        body_termination_distance=None, body_reward_weights=None,
        reward_specs=dict(
            k_pos=200, k_vel=5, k_muscle_len=2000, k_muscle_vel=20,
            w_pos=0.7, w_vel=0.3, w_energy=0.002, w_muscle_len=0, w_muscle_vel=0,
        ),
    )
    return OmegaConf.create({"run": run, "env": env})


@pytest.fixture(scope="module")
def nq():
    model = mujoco.MjModel.from_xml_path(str(H.XML_PATH))
    return model.nq


@pytest.fixture(scope="module")
def pose_file(tmp_path_factory, nq):
    """Tiny synthetic IK reference in the new {frames, metadata} schema."""
    qpos0 = H.standing_qpos(nq)
    # three frames of one motion, slightly varying the root x-position
    frames = {}
    for i in range(3):
        q = qpos0.copy()
        q[0] = 0.01 * i
        frames[round(i * DT, 6)] = q
    num_frames = 3
    length = (num_frames - 1) * DT
    data = {
        "frames": {0: frames},
        "metadata": {0: {"length": length, "dt": DT, "fps": 30, "num_frames": num_frames}},
    }
    path = tmp_path_factory.mktemp("ik") / "ik_tiny.pkl"
    joblib.dump(data, path)
    return path


@pytest.fixture
def env(pose_file):
    np.random.seed(0)
    e = MyoLegsIm(_make_cfg(pose_file))
    e._randomize_heading = False
    e._heading_rot = None
    return e


# ─────────────────────────── construction / sizes ──────────────────────────
def test_reference_data_loaded(env):
    assert env._num_total_motions == 1
    np.testing.assert_array_equal(env._all_motion_ids, np.array([0]))
    assert env.get_motion_length(0) == pytest.approx(2 * DT)


def test_tracked_bodies(env):
    assert len(env.tracked_bodies) == 14
    assert env.tracked_bodies[0] == "root"
    assert all(b in env.body_names for b in env.tracked_bodies)


def test_obs_size_matches_computation(env):
    # self obs + task obs, computed from the model, must equal the concat length
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == env.get_obs_size()
    assert obs.shape[0] == env.get_self_obs_size() + env.get_task_obs_size()
    assert obs.dtype == np.float32


def test_task_obs_size_formula(env):
    # 3 task inputs, each 3 * num_tracked_bodies
    assert env.get_task_obs_size() == 3 * 3 * len(env.tracked_bodies)


def test_action_size_is_model_nu(env):
    assert env.get_action_size() == env.mj_model.nu


# ─────────────────────────── reset / initial state ─────────────────────────
def test_reset_sets_reference_qpos(env, nq):
    env.reset(seed=0, options={"start_time": 0.0})
    qpos0 = H.standing_qpos(nq)
    # with heading disabled and start_time 0, qpos equals the stored frame
    np.testing.assert_allclose(env.mj_data.qpos[:nq], qpos0, atol=1e-12)


def test_reset_zeroes_velocity_at_start(env):
    env.reset(seed=0, options={"start_time": 0.0})
    np.testing.assert_allclose(env.mj_data.qvel, 0.0, atol=1e-12)


# ─────────────────────────── heading math ──────────────────────────────────
def test_apply_heading_identity_when_disabled(env, nq):
    env._heading_rot = None
    q = H.standing_qpos(nq)
    out = env._apply_heading(q.copy())
    np.testing.assert_array_equal(out, q)


def test_apply_heading_rotates_root(env, nq):
    angle = 0.7
    env._heading_rot = sRot.from_euler("z", angle)
    q = H.standing_qpos(nq)
    q[0], q[1] = 0.3, 0.4  # nonzero planar translation
    out = env._apply_heading(q.copy())
    # translation rotated about z, height preserved
    expected_xy = sRot.from_euler("z", angle).apply([0.3, 0.4, q[2]])
    np.testing.assert_allclose(out[:3], expected_xy, atol=1e-10)
    np.testing.assert_allclose(out[2], q[2], atol=1e-12)
    # quaternion stays unit
    np.testing.assert_allclose(np.linalg.norm(out[3:7]), 1.0, atol=1e-10)
    # joints untouched
    np.testing.assert_array_equal(out[7:], q[7:])


# ─────────────────────────── motion bookkeeping ────────────────────────────
def test_lookup_reference_qpos_exact_and_nearest(env, nq):
    q_exact = env._lookup_reference_qpos(0, 0.0)
    np.testing.assert_allclose(q_exact, H.standing_qpos(nq), atol=1e-12)
    # a time between frames snaps to nearest stored key
    q_near = env._lookup_reference_qpos(0, DT * 0.4)
    q_frame0 = env._lookup_reference_qpos(0, 0.0)
    np.testing.assert_array_equal(q_near, q_frame0)


def test_sample_motions_contiguous_wrap(env):
    env.sample_motions()
    assert env._sampled_motion_id == 0


def test_get_motion_length_metadata_branch(env):
    assert env.get_motion_length(0) == pytest.approx(2 * DT)
    # unknown motion → 0.0
    assert env.get_motion_length(999) == 0.0


# ─────────────────────────── deterministic step snapshot ───────────────────
def test_step_reward_is_finite_and_reproducible(env):
    env.reset(seed=0, options={"start_time": 0.0})
    action = H.fixed_action(env.get_action_size())
    obs, reward, terminated, truncated, info = env.step(action.copy())
    assert np.isfinite(reward)
    assert obs.shape[0] == env.get_obs_size()
    assert "r_body_pos" in info and "energy_cost" in info

    # replay from same seed/state → identical reward (determinism)
    np.random.seed(0)
    env2 = MyoLegsIm(_make_cfg(env.initial_pose_file))
    env2._randomize_heading = False
    env2._heading_rot = None
    env2.reset(seed=0, options={"start_time": 0.0})
    _, reward2, _, _, _ = env2.step(action.copy())
    assert reward == pytest.approx(reward2)


def test_compute_energy_reward_available_on_base_env():
    # compute_energy_reward now lives on MyoLegsEnv, so base physics_step works.
    from myohuman.env.myolegs_env import MyoLegsEnv
    from myohuman.env.myolegs_im import MyoLegsIm
    assert hasattr(MyoLegsEnv, "compute_energy_reward")
    # MyoLegsIm inherits the same implementation (no override)
    assert MyoLegsIm.compute_energy_reward is MyoLegsEnv.compute_energy_reward
    assert MyoLegsEnv.compute_energy_reward(None, np.array([3.0, 4.0])) == pytest.approx(12.0)


def test_start_end_eval_toggle_flags(env):
    env.start_eval(im_eval=True)
    assert env.test is True
    assert env.im_eval is True
    assert env._randomize_heading is False
    assert env.random_start is False
    # the dead termination_distance save/restore was removed
    assert not hasattr(env, "_temp_termination_distance")
    td = env.termination_distance
    env.end_eval()
    assert env.test is False
    assert env.im_eval is False
    assert env._randomize_heading is True
    assert env.termination_distance == td  # unchanged by eval cycle


def test_biomechanics_records_sampled_motion_id(pose_file):
    np.random.seed(0)
    e = MyoLegsIm(_make_cfg(pose_file, recording_biomechanics=True))
    e._randomize_heading = False
    e._heading_rot = None
    e.reset(seed=0, options={"start_time": 0.0})
    e._sampled_motion_id = 0
    e.step(H.fixed_action(e.get_action_size()))
    # motion_id log holds the actually-playing motion, not the config motion_id const
    assert e.motion_id[-1] == e._sampled_motion_id


def test_reward_at_reference_pose_is_high(env):
    # At reset the sim pose equals the reference frame, so tracking reward
    # (position/velocity) should be near its maximum before any physics drift.
    env.reset(seed=0, options={"start_time": 0.0})
    ref = env.get_state_from_motionlib_cache(env._motion_start_time)
    body_pos = env.get_body_xpos()[None,][..., env.tracked_bodies_id, :]
    ref_pos = ref.xpos[..., env.tracked_bodies_id, :]
    np.testing.assert_allclose(body_pos, ref_pos, atol=1e-6)
