#!/usr/bin/env python
"""
compute_ik.py — Unified script that:
  1. Loads raw KIT motion data (.npz files)
  2. Runs SMPL Forward Kinematics to get global body positions
  3. Solves Inverse Kinematics for the myohuman Mujoco model on every frame
  4. Saves reference .pkl (one file per split, or two files for --split both)

Output format:
    {
        "frames":   { motion_id: { time_float: qpos_array, ... }, ... },
        "metadata": { motion_id: { "length", "dt", "fps", "num_frames" }, ... },
    }

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python scripts/compute_ik.py --split train --workers 190
    # train + test в одном прогоне → два файла (ik_train.pkl, ik_test.pkl):
    ... --split both --workers 64
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

MYOHUMAN_TRACKED_BODIES = [
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
    p.add_argument("--split", choices=["train", "test", "both"], default="train",
                   help="train | test | both (both: один пул воркеров, два выходных .pkl)")
    p.add_argument("--output", type=str, default=None,
                   help="Output .pkl (train default path; при --split both задаёт только train-файл, test — по умолчанию)")
    p.add_argument("--kit-dir", type=str, default=DEFAULT_KIT_DIR)
    p.add_argument("--smpl-dir", type=str, default=DEFAULT_SMPL_DIR)
    p.add_argument("--xml-path", type=str, default=DEFAULT_XML_PATH)
    p.add_argument("--train-keys", type=str, default=DEFAULT_TRAIN_KEYS)
    p.add_argument("--test-keys", type=str, default=DEFAULT_TEST_KEYS)
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel workers (default: 1 on macOS, cpu_count elsewhere)")
    p.add_argument("--checkpoint-dir", type=str, default=str(BASE_DIR / "data" / "tmp"))
    p.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Только загрузить чекпоинт и записать выходные .pkl; IK не считать. "
            "Имеет смысл после полного прогона, если упало на joblib.dump (например, нет места на диске). "
            "Сейчас поддерживается только при --split both (ik_both_ckpt.pkl)."
        ),
    )
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
        track_ids = [body_names.index(b) for b in MYOHUMAN_TRACKED_BODIES]
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
def assemble_output(ordered_keys: list, checkpoint: dict) -> dict:
    """Собрать {"frames": {mid: ...}, "metadata": {mid: ...}} в порядке ordered_keys."""
    out = {"frames": {}, "metadata": {}}
    for mid, key in enumerate(ordered_keys):
        if key in checkpoint:
            out["frames"][mid] = checkpoint[key]["frames"]
            out["metadata"][mid] = checkpoint[key]["metadata"]
    return out


def main():
    args = parse_args()

    if args.finalize_only and args.split != "both":
        raise SystemExit("--finalize-only поддерживается только с --split both")

    if args.split == "both":
        train_keys = load_keys(args.train_keys)
        test_keys = load_keys(args.test_keys)
        combined = list(dict.fromkeys(train_keys + test_keys))
        keys_for_discover = combined
        ckpt_path = Path(args.checkpoint_dir) / "ik_both_ckpt.pkl"
        output_train = args.output or DEFAULT_TRAIN_OUTPUT
        output_test = DEFAULT_TEST_OUTPUT
        logger.info(
            "Split 'both': %d train keys, %d test keys → %d unique keys",
            len(train_keys), len(test_keys), len(combined),
        )
    elif args.split == "train":
        train_keys = load_keys(args.train_keys)
        combined = train_keys
        keys_for_discover = combined
        ckpt_path = Path(args.checkpoint_dir) / "ik_train_ckpt.pkl"
        output_train = args.output or DEFAULT_TRAIN_OUTPUT
        logger.info(f"Loaded {len(combined)} keys for split 'train'")
    else:
        test_keys = load_keys(args.test_keys)
        combined = test_keys
        keys_for_discover = combined
        ckpt_path = Path(args.checkpoint_dir) / "ik_test_ckpt.pkl"
        output_train = args.output or DEFAULT_TEST_OUTPUT
        logger.info(f"Loaded {len(combined)} keys for split 'test'")

    key_to_path = discover_motions(args.kit_dir, keys_for_discover)
    logger.info(f"Found {len(key_to_path)}/{len(keys_for_discover)} matching .npz files")

    ordered_keys = [k for k in combined if k in key_to_path]
    motion_id_map = {mid: key for mid, key in enumerate(ordered_keys)}

    if args.split == "both":
        ordered_train_out = [k for k in train_keys if k in key_to_path]
        ordered_test_out = [k for k in test_keys if k in key_to_path]

    # Load checkpoint (keyed by motion key name for stability across key-file changes)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        if not ckpt_path.exists():
            raise SystemExit(f"--finalize-only: нет файла чекпоинта {ckpt_path}")
        checkpoint = joblib.load(ckpt_path)
        logger.info(
            "Finalize-only: загружен чекпоинт (%d записей), IK пропускается",
            len(checkpoint),
        )
        remaining = []
    elif ckpt_path.exists():
        checkpoint = joblib.load(ckpt_path)
        logger.info(f"Resumed checkpoint: {len(checkpoint)} motions already done")
        remaining = [(mid, key) for mid, key in motion_id_map.items()
                     if key not in checkpoint]
    else:
        checkpoint = {}
        remaining = [(mid, key) for mid, key in motion_id_map.items()
                     if key not in checkpoint]

    if not args.finalize_only:
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

    if args.split == "both":
        out_tr = assemble_output(ordered_train_out, checkpoint)
        out_te = assemble_output(ordered_test_out, checkpoint)
        Path(output_train).parent.mkdir(parents=True, exist_ok=True)
        Path(output_test).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(out_tr, output_train)
        joblib.dump(out_te, output_test)
        logger.info(
            "Saved %d train → %s; %d test → %s",
            len(out_tr["frames"]), output_train,
            len(out_te["frames"]), output_test,
        )
    else:
        output_path = output_train
        ordered = ordered_keys
        output = assemble_output(ordered, checkpoint)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(output, output_path)
        logger.info(f"Saved {len(output['frames'])} motions → {output_path}")


if __name__ == "__main__":
    main()
