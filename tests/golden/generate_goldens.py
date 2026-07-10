"""Regenerate golden snapshots for characterization tests.

Run ONLY when a behavior change is intentional:

    uv run python tests/golden/generate_goldens.py

The tests compare live outputs against these files; regenerating them
after an unintended change would mask a regression. Review the git diff
of the .npz files before committing.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ on path

import helpers as H  # noqa: E402
import myohuman.utils.np_transform_utils as npt  # noqa: E402
from myohuman.env.myohuman_env import compute_self_observations  # noqa: E402
from myohuman.env.myohuman_im import (  # noqa: E402
    compute_imitation_observations,
    compute_imitation_reward,
    compute_humanoid_im_reset,
)

OUT = H.GOLDEN_DIR


def save(name, **arrays):
    np.savez(
        OUT / f"{name}.npz", **{k: np.asarray(v) for k, v in arrays.items()}
    )
    print(f"wrote {name}.npz  ({', '.join(arrays)})")


def gen_np_transform():
    q = H.random_unit_quats_wxyz(8, seed=101)
    q2 = H.random_unit_quats_wxyz(8, seed=202)
    v = H.random_vectors(8, seed=303)
    angle = H.rng(404).uniform(-np.pi, np.pi, size=6)
    axis = H.random_vectors(6, seed=505)
    save(
        "np_transform",
        q=q,
        q2=q2,
        v=v,
        angle=angle,
        axis=axis,
        quat_rotate=npt.quat_rotate(q, v),
        quat_mul=npt.quat_mul(q, q2),
        quat_conjugate=npt.quat_conjugate(q),
        calc_heading=npt.calc_heading(q),
        calc_heading_quat=npt.calc_heading_quat(q),
        calc_heading_quat_inv=npt.calc_heading_quat_inv(q),
        quat_to_tan_norm=npt.quat_to_tan_norm(q),
        quat_from_angle_axis=npt.quat_from_angle_axis(angle, axis),
        normalize=npt.normalize(v),
        quat_to_angle_axis_angle=npt.quat_to_angle_axis(q)[0],
        quat_to_angle_axis_axis=npt.quat_to_angle_axis(q)[1],
        quat_to_exp_map=npt.quat_to_exp_map(q),
        normalize_angle=npt.normalize_angle(
            H.rng(606).uniform(-10, 10, size=12)
        ),
    )


def gen_imitation_reward():
    body_pos, body_vel, ref_pos, ref_vel = H.imitation_reward_inputs()
    r_uniform, raw_u = compute_imitation_reward(
        body_pos,
        body_vel,
        ref_pos,
        ref_vel,
        H.REWARD_SPECS,
        body_weights=None,
    )
    w = H.rng(1234).uniform(0.5, 2.0, size=H.NUM_TRACKED_BODIES)
    w = w / w.sum()
    r_weighted, raw_w = compute_imitation_reward(
        body_pos,
        body_vel,
        ref_pos,
        ref_vel,
        H.REWARD_SPECS,
        body_weights=w,
    )
    save(
        "imitation_reward",
        body_pos=body_pos,
        body_vel=body_vel,
        ref_pos=ref_pos,
        ref_vel=ref_vel,
        body_weights=w,
        reward_uniform=r_uniform,
        r_body_pos_uniform=raw_u["r_body_pos"],
        r_vel_uniform=raw_u["r_vel"],
        reward_weighted=r_weighted,
        r_body_pos_weighted=raw_w["r_body_pos"],
        r_vel_weighted=raw_w["r_vel"],
    )


def gen_imitation_obs():
    inp = H.imitation_obs_inputs()
    obs = compute_imitation_observations(*inp, time_steps=1)
    save(
        "imitation_obs",
        diff_local_body_pos=obs["diff_local_body_pos"],
        diff_local_vel=obs["diff_local_vel"],
        local_ref_body_pos=obs["local_ref_body_pos"],
        diff_muscle_len=obs["diff_muscle_len"],
        diff_muscle_vel=obs["diff_muscle_vel"],
    )


def gen_reset():
    g = H.rng(2468)
    body = g.normal(size=(1, H.NUM_TRACKED_BODIES, 3))
    ref_close = body + 0.01 * g.normal(size=body.shape)
    ref_far = body.copy()
    ref_far[0, 3] += 5.0  # one body way off
    per_body = np.full(H.NUM_TRACKED_BODIES, 0.15)
    save(
        "im_reset",
        body=body,
        ref_close=ref_close,
        ref_far=ref_far,
        per_body_thresh=per_body,
        close_max=compute_humanoid_im_reset(body, ref_close, 0.15, False),
        close_mean=compute_humanoid_im_reset(body, ref_close, 0.15, True),
        far_max=compute_humanoid_im_reset(body, ref_far, 0.15, False),
        far_mean=compute_humanoid_im_reset(body, ref_far, 0.15, True),
        far_max_perbody=compute_humanoid_im_reset(
            body, ref_far, per_body, False
        ),
    )


def gen_self_obs():
    body_pos, body_rot, body_vel, body_ang_vel = H.self_obs_inputs()
    obs = compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)
    save(
        "self_obs",
        body_pos=body_pos,
        body_rot=body_rot,
        body_vel=body_vel,
        body_ang_vel=body_ang_vel,
        root_h_obs=obs["root_h_obs"],
        local_body_pos=obs["local_body_pos"],
        local_body_rot_obs=obs["local_body_rot_obs"],
        local_body_vel=obs["local_body_vel"],
        local_body_ang_vel=obs["local_body_ang_vel"],
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gen_np_transform()
    gen_imitation_reward()
    gen_imitation_obs()
    gen_reset()
    gen_self_obs()
    print("done")
