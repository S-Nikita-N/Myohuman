"""Characterization + property tests for numpy quaternion/heading utils.

These functions feed every observation computed by the env, so their exact
numeric behavior is pinned against golden snapshots. Property tests add
mathematical invariants that any correct implementation must satisfy.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

import helpers as H
import myohuman.utils.np_transform_utils as npt

GOLDEN = np.load(H.GOLDEN_DIR / "np_transform.npz")


# ─────────────────────────── golden snapshots ───────────────────────────
@pytest.mark.parametrize(
    "key,fn",
    [
        ("quat_rotate", lambda: npt.quat_rotate(GOLDEN["q"], GOLDEN["v"])),
        ("quat_mul", lambda: npt.quat_mul(GOLDEN["q"], GOLDEN["q2"])),
        ("quat_conjugate", lambda: npt.quat_conjugate(GOLDEN["q"])),
        ("calc_heading", lambda: npt.calc_heading(GOLDEN["q"])),
        ("calc_heading_quat", lambda: npt.calc_heading_quat(GOLDEN["q"])),
        ("calc_heading_quat_inv", lambda: npt.calc_heading_quat_inv(GOLDEN["q"])),
        ("quat_to_tan_norm", lambda: npt.quat_to_tan_norm(GOLDEN["q"])),
        ("quat_from_angle_axis",
         lambda: npt.quat_from_angle_axis(GOLDEN["angle"], GOLDEN["axis"])),
        ("normalize", lambda: npt.normalize(GOLDEN["v"])),
        ("quat_to_exp_map", lambda: npt.quat_to_exp_map(GOLDEN["q"])),
    ],
)
def test_golden(key, fn):
    np.testing.assert_allclose(fn(), GOLDEN[key], rtol=1e-10, atol=1e-12)


def test_quat_to_angle_axis_golden():
    angle, axis = npt.quat_to_angle_axis(GOLDEN["q"])
    np.testing.assert_allclose(angle, GOLDEN["quat_to_angle_axis_angle"], rtol=1e-10)
    np.testing.assert_allclose(axis, GOLDEN["quat_to_angle_axis_axis"], rtol=1e-10)


def test_normalize_angle_golden():
    x = H.rng(606).uniform(-10, 10, size=12)
    np.testing.assert_allclose(npt.normalize_angle(x), GOLDEN["normalize_angle"], rtol=1e-10)


# ─────────────────────────── property checks ────────────────────────────
def test_quat_mul_matches_scipy():
    # project uses wxyz; scipy uses xyzw
    q1 = H.random_unit_quats_wxyz(6, seed=11)
    q2 = H.random_unit_quats_wxyz(6, seed=22)
    got = npt.quat_mul(q1, q2)
    expected_xyzw = (
        sRot.from_quat(q1[:, [1, 2, 3, 0]]) * sRot.from_quat(q2[:, [1, 2, 3, 0]])
    ).as_quat()
    expected_wxyz = expected_xyzw[:, [3, 0, 1, 2]]
    # quaternion sign ambiguity: compare up to sign
    same = np.allclose(got, expected_wxyz, atol=1e-8)
    flipped = np.allclose(got, -expected_wxyz, atol=1e-8)
    assert same or flipped or np.allclose(np.abs(got), np.abs(expected_wxyz), atol=1e-8)


def test_quat_rotate_matches_scipy():
    q = H.random_unit_quats_wxyz(6, seed=33)
    v = H.random_vectors(6, seed=44)
    got = npt.quat_rotate(q, v)
    expected = sRot.from_quat(q[:, [1, 2, 3, 0]]).apply(v)
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_heading_inv_cancels_heading():
    q = H.random_unit_quats_wxyz(5, seed=55)
    h = npt.calc_heading_quat(q)
    h_inv = npt.calc_heading_quat_inv(q)
    prod = npt.quat_mul(h, h_inv)
    identity = np.zeros_like(prod)
    identity[:, 0] = 1.0
    # up to sign
    assert np.allclose(prod, identity, atol=1e-8) or np.allclose(prod, -identity, atol=1e-8)


def test_quat_conjugate_inverts_rotation():
    q = H.random_unit_quats_wxyz(5, seed=66)
    v = H.random_vectors(5, seed=77)
    rotated = npt.quat_rotate(q, v)
    back = npt.quat_rotate(npt.quat_conjugate(q), rotated)
    np.testing.assert_allclose(back, v, atol=1e-8)


def test_wxyz_xyzw_roundtrip():
    q = H.random_unit_quats_wxyz(4, seed=88)
    np.testing.assert_array_equal(npt.xyzw_to_wxyz(npt.wxyz_to_xyzw(q)), q)


def test_calc_heading_is_yaw_only():
    # pure yaw rotation of given angle → heading equals that angle
    angles = np.array([0.0, 0.5, -1.2, 3.0])
    # pure-Z unit quaternion in wxyz: [cos(a/2), 0, 0, sin(a/2)]
    quats_wxyz = np.zeros((len(angles), 4))
    quats_wxyz[:, 0] = np.cos(angles / 2)
    quats_wxyz[:, 3] = np.sin(angles / 2)
    heading = npt.calc_heading(quats_wxyz)
    np.testing.assert_allclose(npt.normalize_angle(heading), angles, atol=1e-8)


def test_remove_base_rot_unknown_type_raises():
    q = H.random_unit_quats_wxyz(3, seed=1)
    with pytest.raises(ValueError):
        npt.remove_base_rot(q, humanoid_type="mujoco")


def test_quat_to_tan_norm_shape_and_unit():
    q = H.random_unit_quats_wxyz(7, seed=99)
    tn = npt.quat_to_tan_norm(q)
    assert tn.shape == (7, 6)
    np.testing.assert_allclose(np.linalg.norm(tn[:, :3], axis=-1), 1.0, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(tn[:, 3:], axis=-1), 1.0, atol=1e-8)
