"""Deterministic input builders shared by tests and the golden generator.

These inputs are frozen: changing them invalidates the golden files in
tests/golden/. Regenerate goldens with `uv run python tests/golden/generate_goldens.py`
only when a behavior change is intentional.
"""

import numpy as np

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
XML_PATH = REPO_ROOT / "xml" / "myohuman.xml"

# Matches cfg/env/env_im.yaml
REWARD_SPECS = {
    "k_pos": 200,
    "k_vel": 5,
    "k_muscle_len": 2000,
    "k_muscle_vel": 20,
    "w_pos": 0.7,
    "w_vel": 0.3,
    "w_energy": 0.002,
    "w_muscle_len": 0,
    "w_muscle_vel": 0,
}

NUM_TRACKED_BODIES = 14  # len(MYOHUMAN_TRACKED_BODIES)


def rng(seed=12345):
    return np.random.default_rng(seed)


def random_unit_quats_wxyz(n, seed=12345):
    """Random unit quaternions in wxyz order (project convention)."""
    g = rng(seed)
    q = g.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def random_vectors(n, seed=54321):
    return rng(seed).normal(size=(n, 3))


def imitation_reward_inputs():
    """Inputs for compute_imitation_reward: (1, J, 3) arrays with small ref offset."""
    g = rng(777)
    body_pos = g.normal(size=(1, NUM_TRACKED_BODIES, 3))
    body_vel = g.normal(size=(1, NUM_TRACKED_BODIES, 3))
    ref_body_pos = body_pos + 0.05 * g.normal(size=body_pos.shape)
    ref_body_vel = body_vel + 0.1 * g.normal(size=body_vel.shape)
    return body_pos, body_vel, ref_body_pos, ref_body_vel


def imitation_obs_inputs(num_muscles=10):
    """Inputs for compute_imitation_observations (B=1, time_steps=1)."""
    g = rng(888)
    body_pos = g.normal(size=(1, NUM_TRACKED_BODIES, 3))
    body_vel = g.normal(size=(1, NUM_TRACKED_BODIES, 3))
    ref_body_pos = body_pos + 0.05 * g.normal(size=body_pos.shape)
    ref_body_vel = body_vel + 0.1 * g.normal(size=body_vel.shape)
    root_pos = body_pos[:, 0].copy()
    root_rot = random_unit_quats_wxyz(1, seed=999)
    muscle_len = g.uniform(0.05, 0.4, size=num_muscles)
    muscle_vel = g.normal(size=num_muscles)
    ref_muscle_len = muscle_len + 0.01 * g.normal(size=num_muscles)
    ref_muscle_vel = muscle_vel + 0.1 * g.normal(size=num_muscles)
    return (
        root_pos,
        root_rot,
        body_pos,
        body_vel,
        ref_body_pos,
        ref_body_vel,
        muscle_len,
        muscle_vel,
        ref_muscle_len,
        ref_muscle_vel,
    )


def self_obs_inputs(num_bodies=20):
    """Inputs for compute_self_observations: (1, J, ...) arrays."""
    g = rng(4242)
    body_pos = g.normal(size=(1, num_bodies, 3))
    body_pos[:, :, 2] += 1.0
    body_rot = random_unit_quats_wxyz(num_bodies, seed=2024)[None,]
    body_vel = g.normal(size=(1, num_bodies, 3))
    body_ang_vel = g.normal(size=(1, num_bodies, 3))
    return body_pos, body_rot, body_vel, body_ang_vel


def standing_qpos(nq):
    """Standing pose matching MyoHumanEnv.init_myohuman()."""
    qpos = np.zeros(nq)
    qpos[2] = 0.94
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    return qpos


def fixed_action(nu, seed=31337):
    """Deterministic action in [-1, 1] for the muscle-control chain."""
    return rng(seed).uniform(-1.0, 1.0, size=nu)
