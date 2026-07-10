"""Characterization tests for the pure reward/observation math in the env.

These functions decide the reward signal, the imitation observations and the
early-termination condition — the behavioral core of training. They take plain
numpy arrays, so no mujoco model or env instance is needed.
"""

import numpy as np
import pytest

import helpers as H
from myohuman.env.myohuman_env import compute_self_observations
from myohuman.env.myohuman_im import (
    compute_imitation_observations,
    compute_imitation_reward,
    compute_humanoid_im_reset,
    MyoHumanIm,
)

REWARD_GOLDEN = np.load(H.GOLDEN_DIR / "imitation_reward.npz")
OBS_GOLDEN = np.load(H.GOLDEN_DIR / "imitation_obs.npz")
RESET_GOLDEN = np.load(H.GOLDEN_DIR / "im_reset.npz")
SELF_GOLDEN = np.load(H.GOLDEN_DIR / "self_obs.npz")


# ───────────────────────────── reward ──────────────────────────────
def test_imitation_reward_uniform_golden():
    body_pos, body_vel, ref_pos, ref_vel = H.imitation_reward_inputs()
    reward, raw = compute_imitation_reward(
        body_pos,
        body_vel,
        ref_pos,
        ref_vel,
        H.REWARD_SPECS,
        body_weights=None,
    )
    np.testing.assert_allclose(
        reward, REWARD_GOLDEN["reward_uniform"], rtol=1e-10
    )
    np.testing.assert_allclose(
        raw["r_body_pos"], REWARD_GOLDEN["r_body_pos_uniform"], rtol=1e-10
    )
    np.testing.assert_allclose(
        raw["r_vel"], REWARD_GOLDEN["r_vel_uniform"], rtol=1e-10
    )


def test_imitation_reward_weighted_golden():
    body_pos, body_vel, ref_pos, ref_vel = H.imitation_reward_inputs()
    reward, raw = compute_imitation_reward(
        body_pos,
        body_vel,
        ref_pos,
        ref_vel,
        H.REWARD_SPECS,
        body_weights=REWARD_GOLDEN["body_weights"],
    )
    np.testing.assert_allclose(
        reward, REWARD_GOLDEN["reward_weighted"], rtol=1e-10
    )
    np.testing.assert_allclose(
        raw["r_body_pos"], REWARD_GOLDEN["r_body_pos_weighted"], rtol=1e-10
    )


def test_imitation_reward_perfect_match_is_max():
    body_pos = H.rng(1).normal(size=(1, H.NUM_TRACKED_BODIES, 3))
    body_vel = H.rng(2).normal(size=(1, H.NUM_TRACKED_BODIES, 3))
    reward, raw = compute_imitation_reward(
        body_pos,
        body_vel,
        body_pos.copy(),
        body_vel.copy(),
        H.REWARD_SPECS,
    )
    np.testing.assert_allclose(raw["r_body_pos"], 1.0, atol=1e-12)
    np.testing.assert_allclose(raw["r_vel"], 1.0, atol=1e-12)
    np.testing.assert_allclose(
        reward, H.REWARD_SPECS["w_pos"] + H.REWARD_SPECS["w_vel"], atol=1e-12
    )


def test_imitation_reward_matches_closed_form():
    body_pos, body_vel, ref_pos, ref_vel = H.imitation_reward_inputs()
    reward, raw = compute_imitation_reward(
        body_pos,
        body_vel,
        ref_pos,
        ref_vel,
        H.REWARD_SPECS,
    )
    pos_mse = ((ref_pos - body_pos) ** 2).mean(axis=-1).mean(axis=-1)
    vel_mse = ((ref_vel - body_vel) ** 2).mean(axis=-1).mean(axis=-1)
    exp_pos = np.exp(-H.REWARD_SPECS["k_pos"] * pos_mse)[0]
    exp_vel = np.exp(-H.REWARD_SPECS["k_vel"] * vel_mse)[0]
    np.testing.assert_allclose(raw["r_body_pos"], exp_pos, rtol=1e-10)
    np.testing.assert_allclose(raw["r_vel"], exp_vel, rtol=1e-10)
    np.testing.assert_allclose(
        reward,
        H.REWARD_SPECS["w_pos"] * exp_pos + H.REWARD_SPECS["w_vel"] * exp_vel,
        rtol=1e-10,
    )


def test_imitation_reward_uniform_weights_equals_none():
    body_pos, body_vel, ref_pos, ref_vel = H.imitation_reward_inputs()
    r_none, _ = compute_imitation_reward(
        body_pos, body_vel, ref_pos, ref_vel, H.REWARD_SPECS
    )
    w = np.full(H.NUM_TRACKED_BODIES, 1.0 / H.NUM_TRACKED_BODIES)
    r_w, _ = compute_imitation_reward(
        body_pos, body_vel, ref_pos, ref_vel, H.REWARD_SPECS, body_weights=w
    )
    np.testing.assert_allclose(r_none, r_w, rtol=1e-10)


# ─────────────────────────── observations ──────────────────────────
def test_imitation_observations_golden():
    inp = H.imitation_obs_inputs()
    obs = compute_imitation_observations(*inp, time_steps=1)
    for key in (
        "diff_local_body_pos",
        "diff_local_vel",
        "local_ref_body_pos",
        "diff_muscle_len",
        "diff_muscle_vel",
    ):
        np.testing.assert_allclose(
            obs[key], OBS_GOLDEN[key], rtol=1e-10, atol=1e-12
        )


def test_imitation_observations_zero_when_matched():
    (
        root_pos,
        root_rot,
        body_pos,
        body_vel,
        ref_pos,
        ref_vel,
        mlen,
        mvel,
        rmlen,
        rmvel,
    ) = H.imitation_obs_inputs()
    obs = compute_imitation_observations(
        root_pos,
        root_rot,
        body_pos,
        body_vel,
        body_pos.copy(),
        body_vel.copy(),
        mlen,
        mvel,
        mlen.copy(),
        mvel.copy(),
        time_steps=1,
    )
    np.testing.assert_allclose(obs["diff_local_body_pos"], 0.0, atol=1e-12)
    np.testing.assert_allclose(obs["diff_local_vel"], 0.0, atol=1e-12)
    np.testing.assert_allclose(obs["diff_muscle_len"], 0.0, atol=1e-12)
    np.testing.assert_allclose(obs["diff_muscle_vel"], 0.0, atol=1e-12)


def test_imitation_observations_muscle_diff_sign():
    inp = list(H.imitation_obs_inputs())
    obs = compute_imitation_observations(*inp, time_steps=1)
    # diff = ref - current
    np.testing.assert_allclose(
        obs["diff_muscle_len"], inp[8] - inp[6], rtol=1e-12
    )
    np.testing.assert_allclose(
        obs["diff_muscle_vel"], inp[9] - inp[7], rtol=1e-12
    )


# ─────────────────────────── reset / termination ───────────────────
def test_im_reset_golden():
    body, close, far = (
        RESET_GOLDEN["body"],
        RESET_GOLDEN["ref_close"],
        RESET_GOLDEN["ref_far"],
    )
    thr = RESET_GOLDEN["per_body_thresh"]
    np.testing.assert_array_equal(
        compute_humanoid_im_reset(body, close, 0.15, False),
        RESET_GOLDEN["close_max"],
    )
    np.testing.assert_array_equal(
        compute_humanoid_im_reset(body, close, 0.15, True),
        RESET_GOLDEN["close_mean"],
    )
    np.testing.assert_array_equal(
        compute_humanoid_im_reset(body, far, 0.15, False),
        RESET_GOLDEN["far_max"],
    )
    np.testing.assert_array_equal(
        compute_humanoid_im_reset(body, far, 0.15, True),
        RESET_GOLDEN["far_mean"],
    )
    np.testing.assert_array_equal(
        compute_humanoid_im_reset(body, far, thr, False),
        RESET_GOLDEN["far_max_perbody"],
    )


def test_im_reset_close_not_fallen():
    body, close = RESET_GOLDEN["body"], RESET_GOLDEN["ref_close"]
    assert not compute_humanoid_im_reset(body, close, 0.15, False)[0]


def test_im_reset_far_body_triggers_max_not_necessarily_mean():
    body, far = RESET_GOLDEN["body"], RESET_GOLDEN["ref_far"]
    assert compute_humanoid_im_reset(body, far, 0.15, False)[
        0
    ]  # one body over threshold


def test_im_reset_boundary_is_strict_gt():
    # single body exactly at threshold distance → not fallen (strict >)
    body = np.zeros((1, 1, 3))
    ref = np.zeros((1, 1, 3))
    ref[0, 0, 0] = 0.15
    assert not compute_humanoid_im_reset(body, ref, 0.15, False)[0]
    ref[0, 0, 0] = 0.1500001
    assert compute_humanoid_im_reset(body, ref, 0.15, False)[0]


# ───────────────────────── self observations ───────────────────────
def test_self_observations_golden():
    body_pos, body_rot, body_vel, body_ang_vel = H.self_obs_inputs()
    obs = compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)
    for key in (
        "root_h_obs",
        "local_body_pos",
        "local_body_rot_obs",
        "local_body_vel",
        "local_body_ang_vel",
    ):
        np.testing.assert_allclose(
            obs[key], SELF_GOLDEN[key], rtol=1e-10, atol=1e-12
        )


def test_self_observations_shapes():
    n = 20
    body_pos, body_rot, body_vel, body_ang_vel = H.self_obs_inputs(num_bodies=n)
    obs = compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)
    assert obs["root_h_obs"].shape == (1, 1)
    assert obs["local_body_pos"].shape == (1, 3 * n - 3)  # root pos removed
    assert obs["local_body_rot_obs"].shape == (1, 6 * n)
    assert obs["local_body_vel"].shape == (1, 3 * n)
    assert obs["local_body_ang_vel"].shape == (1, 3 * n)


def test_self_observations_root_height_is_root_z():
    body_pos, body_rot, body_vel, body_ang_vel = H.self_obs_inputs()
    obs = compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)
    np.testing.assert_allclose(
        obs["root_h_obs"][0, 0], body_pos[0, 0, 2], rtol=1e-12
    )


# ─────────────────────── energy reward (pure) ──────────────────────
def test_energy_reward_l1_plus_l2():
    fn = MyoHumanIm.compute_energy_reward
    action = np.array([3.0, 4.0])
    # bound method not needed — self is unused; call unbound with a dummy self
    val = fn(None, action)
    assert val == pytest.approx(3 + 4 + 5.0)  # |a|_1=7, |a|_2=5


def test_energy_reward_zero_action():
    assert MyoHumanIm.compute_energy_reward(None, np.zeros(10)) == 0.0
