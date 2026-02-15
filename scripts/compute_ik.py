#!/usr/bin/env python
"""
compute_ik.py — Unified script that:
  1. Loads raw KIT motion data (.npz files)
  2. Runs SMPL Forward Kinematics to get global body positions
  3. Solves Inverse Kinematics for the myohuman Mujoco model on every frame
  4. Saves a single reference file for the training environment

Output format:
    {
        "frames":   { motion_id: { time_float: qpos_array, ... }, ... },
        "metadata": { motion_id: { "length", "dt", "fps", "num_frames" }, ... },
    }

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/compute_ik.py --split train --workers 9
"""

import os
import sys
import glob
import torch
import joblib
import logging
import argparse
import mujoco
import scipy.optimize
import numpy as np

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.transform import Rotation as sRot

from myohuman.utils.forward_kinematics import ForwardKinematics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_KIT_DIR = str(BASE_DIR / "data" / "KIT")
DEFAULT_SMPL_DIR = str(BASE_DIR / "data" / "smpl")
DEFAULT_XML_PATH = str(BASE_DIR / "xml" / "myohuman.xml")
DEFAULT_TRAIN_KEYS = str(BASE_DIR / "data" / "dataset" / "kit_train_keys.txt")
DEFAULT_TEST_KEYS = str(BASE_DIR / "data" / "dataset" / "kit_test_keys.txt")
DEFAULT_TRAIN_OUTPUT = str(BASE_DIR / "data" / "inverse_kinematics" / "ik_train.pkl")
DEFAULT_TEST_OUTPUT = str(BASE_DIR / "data" / "inverse_kinematics" / "ik_test.pkl")

CHECKPOINT_EVERY = 50   # save intermediate progress every N completed motions
TARGET_FPS = 30
INITIAL_ROT = sRot.from_euler("XYZ", [-np.pi / 2, 0, -np.pi / 2])

SMPL_TRACKED_IDS = [
    0,   # Pelvis
    2,   # L_Knee
    6,   # R_Knee
    3,   # L_Ankle
    7,   # R_Ankle
    4,   # L_Toe
    8,   # R_Toe
    13,  # Head
    20,  # R_Shoulder
    21,  # R_Elbow
    22,  # R_Wrist
    15,  # L_Shoulder
    16,  # L_Elbow
    17,  # L_Wrist
]

MYOLEG_TRACKED_BODIES = [
    "root",      # Таз
    "tibia_l",   # Левая голень
    "tibia_r",   # Правая голень
    "talus_l",   # Левая лодыжка
    "talus_r",   # Правая лодыжка
    "toes_l",    # Левые пальцы стопы
    "toes_r",    # Правые пальцы стопы
    "head",       # Голова
    "humerus_r",    # Правая плечевая кость
    "radius_r",     # Правое предплечье
    "lunate_r",     # Правое запятье
    'humerus_l',  # Левая плечевая кость
    'radius_l',   # Левое предплечье
    'lunate_l',   # Левое запястье
]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Compute IK reference data for all frames")
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="Which data split to process")
    p.add_argument("--output", type=str, default=None,
                   help="Output .pkl path (defaults based on --split)")
    p.add_argument("--kit-dir", type=str, default=DEFAULT_KIT_DIR)
    p.add_argument("--smpl-dir", type=str, default=DEFAULT_SMPL_DIR)
    p.add_argument("--xml-path", type=str, default=DEFAULT_XML_PATH)
    p.add_argument("--train-keys", type=str, default=DEFAULT_TRAIN_KEYS)
    p.add_argument("--test-keys", type=str, default=DEFAULT_TEST_KEYS)
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel workers (default: 1 on macOS, cpu_count elsewhere)")
    p.add_argument("--checkpoint-dir", type=str, default=str(BASE_DIR / "data" / "tmp"))
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_keys(path: str) -> list:
    """Load motion key names from a text file (one key per line)."""
    with open(path) as f:
        return [k.strip() for k in f.readlines() if k.strip()]


def discover_motions(kit_dir: str, keys: list) -> dict:
    """
    Scan the KIT directory for .npz files and match them to keys.
    Returns {key_name: file_path} for every key that has a matching file.
    """
    all_npz = glob.glob(os.path.join(kit_dir, "**", "*.npz"), recursive=True)
    key_set = set(keys)
    result = {}
    for data_path in all_npz:
        splits = data_path.split("/")[-2:]
        key_name = "0-KIT_" + "_".join(splits).replace(".npz", "")
        if key_name in key_set:
            result[key_name] = data_path
    return result


def load_npz_motion(data_path: str):
    """
    Load a single KIT .npz file, downsample to TARGET_FPS.
    Returns (pose_aa, root_trans, fps)  or  None on failure.

    pose_aa:    (N, 72)  axis-angle for 24 SMPL joints (original SMPL order)
    root_trans: (N, 3)   root translation
    """
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if "mocap_framerate" not in entry_data:
        return None

    framerate = float(entry_data["mocap_framerate"])
    skip = max(1, int(framerate / TARGET_FPS))

    root_trans = entry_data["trans"][::skip, :].astype(np.float32)

    # First 66 values = 22 SMPL joints * 3 axis-angle;  pad 2 hand joints with zeros
    raw_poses = entry_data["poses"][::skip, :66].astype(np.float32)
    pose_aa = np.concatenate(
        [raw_poses, np.zeros((root_trans.shape[0], 6), dtype=np.float32)],
        axis=-1,
    )  # (N, 72)

    return pose_aa, root_trans, TARGET_FPS


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline: FK + IK for a single motion
# ──────────────────────────────────────────────────────────────────────────────
def process_motion(motion_id: int, data_path: str, xml_path: str, smpl_dir: str):
    """
    Full pipeline for one motion:
        load .npz  →  SMPL FK  →  fix height  →  IK every frame  →  return result

    Returns dict {"frames": {time: qpos}, "metadata": {...}}  or  None.
    """
    try:
        loaded = load_npz_motion(data_path)
        if loaded is None:
            return None
        pose_aa, root_trans, fps = loaded
        dt = 1.0 / fps
        N = pose_aa.shape[0]

        # ── Forward Kinematics ──────────────────────────────────────────────
        fk_model = ForwardKinematics(smpl_dir)
        trans_t = torch.from_numpy(root_trans).float()
        pose_aa_t = torch.from_numpy(pose_aa).float().reshape(-1, 24, 3)

        # Fix height so that feet touch the ground
        with torch.no_grad():
            n_check = min(30, N)
            verts, _ = fk_model.smpl_parser.get_joints_verts(
                pose_aa_t[:n_check], th_trans=trans_t[:n_check]
            )
            height_fix = verts[:n_check, ..., -1].min(dim=-1).values.min()
            trans_t[..., -1] -= height_fix

        fk_model.update_model(betas=torch.zeros((1, 10)), dt=dt)
        fk_out = fk_model.fk_batch(pose_aa_t[None,], trans_t[None,])

        # (N, 24, 3)  global body positions from SMPL FK
        global_pos = fk_out.global_translation[0].numpy()
        # (N, qpos_dim)  = [trans(3), root_quat_wxyz(4), dof_euler(J*3)]
        fk_qpos = fk_out.qpos[0].numpy()

        motion_length = dt * (N - 1)

        # ── Mujoco model for IK ────────────────────────────────────────────
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)

        body_names = [mj_model.body(i).name for i in range(mj_model.nbody)][1:]
        track_ids = [body_names.index(b) for b in MYOLEG_TRACKED_BODIES]
        idx_start = 1        # skip "world" body in xpos
        idx_end = mj_model.nbody

        joint_bounds = [tuple(b) for b in mj_model.jnt_range[1:]]

        # ── Solve IK for every frame ────────────────────────────────────────
        frames = {}
        prev_qpos = None

        for fi in range(N):
            t = round(fi * dt, 6)
            fq = fk_qpos[fi]

            # Set root position / orientation in Mujoco frame
            mj_data.qpos[:] = 0
            mj_data.qvel[:] = 0
            mj_data.qpos[2] = 0.94
            mj_data.qpos[3:7] = [1, 0, 0, 0]

            mj_data.qpos[:3] = fq[:3]                                # translation
            rq_wxyz = fq[3:7]                                        # pytorch3d w,x,y,z
            rq_xyzw = rq_wxyz[[1, 2, 3, 0]]                         # → scipy x,y,z,w
            rq_rot = (sRot.from_quat(rq_xyzw) * INITIAL_ROT).as_quat()
            mj_data.qpos[3:7] = np.roll(rq_rot, 1)                  # → mujoco w,x,y,z

            mujoco.mj_kinematics(mj_model, mj_data)
            init_qpos = mj_data.qpos.copy()

            # IK target: SMPL FK body positions at tracked joints (skip root/pelvis)
            ref_pos = global_pos[fi][SMPL_TRACKED_IDS[1:]]           # (13, 3)

            # --- optimisation closures (capture mj_data, mj_model, etc.) ---
            def objective(q):
                return np.linalg.norm(q - init_qpos[7:]) * 5

            def constraint(q):
                mj_data.qpos[7:] = q
                mujoco.mj_kinematics(mj_model, mj_data)
                bp = mj_data.xpos[idx_start:idx_end]
                return np.linalg.norm(bp[track_ids[1:]] - ref_pos, axis=-1).sum()

            x0 = prev_qpos[7:] if prev_qpos is not None else init_qpos[7:]

            sol = scipy.optimize.fmin_slsqp(
                func=objective,
                x0=x0,
                eqcons=[constraint],
                bounds=joint_bounds,
                iprint=0,
                iter=200,
                acc=0.02,
            )

            full_qpos = np.concatenate([init_qpos[:7], sol])
            frames[t] = full_qpos
            prev_qpos = full_qpos

        return {
            "frames": frames,
            "metadata": {
                "length": motion_length,
                "dt": dt,
                "fps": fps,
                "num_frames": N,
            },
        }
    except Exception as e:
        logger.error(f"Motion {motion_id}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Resolve paths
    keys_path = args.train_keys if args.split == "train" else args.test_keys
    output_path = args.output or (
        DEFAULT_TRAIN_OUTPUT if args.split == "train" else DEFAULT_TEST_OUTPUT
    )
    ckpt_path = Path(args.checkpoint_dir) / f"ik_{args.split}_ckpt.pkl"

    # Discover motions
    keys = load_keys(keys_path)
    logger.info(f"Loaded {len(keys)} keys for split '{args.split}'")

    key_to_path = discover_motions(args.kit_dir, keys)
    logger.info(f"Found {len(key_to_path)}/{len(keys)} matching .npz files")

    # Assign sequential integer IDs (same order as in the keys file)
    ordered_keys = [k for k in keys if k in key_to_path]
    motion_id_map = {mid: key for mid, key in enumerate(ordered_keys)}

    # Load checkpoint (keyed by motion key name for stability across key-file changes)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists():
        checkpoint = joblib.load(ckpt_path)
        logger.info(f"Resumed checkpoint: {len(checkpoint)} motions already done")
    else:
        checkpoint = {}

    remaining = [(mid, key) for mid, key in motion_id_map.items()
                 if key not in checkpoint]
    logger.info(f"{len(remaining)} motions remaining to process")

    # Process
    if remaining:
        num_workers = args.workers
        if num_workers is None:
            num_workers = 1 if sys.platform == "darwin" else min(os.cpu_count() or 1, 64)
        logger.info(f"Using {num_workers} worker(s)")

        since_save = 0

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(
                    process_motion, mid,
                    key_to_path[key], args.xml_path, args.smpl_dir,
                ): mid
                for mid, key in remaining
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="IK"):
                mid = futures[future]
                key = motion_id_map[mid]
                result = future.result()
                if result is not None:
                    checkpoint[key] = result
                    since_save += 1
                else:
                    logger.warning(f"Motion {mid} ({key}) returned None (skipped)")

                if since_save >= CHECKPOINT_EVERY:
                    joblib.dump(checkpoint, ckpt_path)
                    since_save = 0

        # Final checkpoint
        if since_save > 0:
            joblib.dump(checkpoint, ckpt_path)

    # ── Assemble final output (sequential integer IDs in key-file order) ────
    output = {"frames": {}, "metadata": {}}
    for mid in sorted(motion_id_map.keys()):
        key = motion_id_map[mid]
        if key in checkpoint:
            output["frames"][mid] = checkpoint[key]["frames"]
            output["metadata"][mid] = checkpoint[key]["metadata"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, output_path)
    logger.info(f"Saved {len(output['frames'])} motions → {output_path}")


if __name__ == "__main__":
    main()
