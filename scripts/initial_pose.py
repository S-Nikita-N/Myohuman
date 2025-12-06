import sys
import os
import hydra
import joblib
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from pathlib import Path
import logging

from myohuman.env.myolegs_im import MyoLegsIm

sys.path.append(os.getcwd())

# CHECKPOINT_PATH = Path("/Users/nikita/Projects/diploma/fullbody/data/dataset/initial_pose_checkpoint.pkl")
CHECKPOINT_PATH = Path("/Users/nikita/Projects/diploma/fullbody/data/dataset/initial_pose_checkpoint_eval.pkl")
CHECKPOINT_EVERY = 100  # сохраняем прогресс каждые N motions


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.is_file():
        try:
            data = joblib.load(CHECKPOINT_PATH)
            if isinstance(data, dict):
                logging.info(f"Loaded checkpoint with {len(data)} motions from {CHECKPOINT_PATH}")
                return data
        except Exception as e:
            logging.warning(f"Failed to load checkpoint {CHECKPOINT_PATH}: {e}")
    return {}


def save_checkpoint(data: dict) -> None:
    _ensure_parent_dir(CHECKPOINT_PATH)
    joblib.dump(data, CHECKPOINT_PATH)
    logging.info(f"Saved checkpoint with {len(data)} motions to {CHECKPOINT_PATH}")


def process_motion(motion_step, cfg_dict):
    """
    Обрабатывает одну анимацию для создания словаря начальных поз.
    """
    # В subprocess нет зарегистрированного hydра resolver, поэтому используем уже резолвленный dict.
    cfg = OmegaConf.create(cfg_dict)
    env = MyoLegsIm(cfg)
    env.initial_pos_data = {}
    
    env.motion_lib.load_motions(
        env.motion_lib_cfg,
        shape_params=env.gender_betas,
        random_sample=False,
        start_idx=motion_step,
    )
    
    motion_id = env.motion_lib._curr_motion_ids[0]
    motion_length = env.motion_lib._motion_lengths[0]
    
    print(f'Processing motion {motion_id}: {motion_length} frames')
    
    initial_pose_dict_single_motion = {}
    for start_time in np.arange(0, motion_length, 0.2):
        print(f'Start time: {start_time}')
        env.reset(options={'start_time': start_time})
        initial_pose_dict_single_motion[start_time] = env.initial_pose
        env.initial_pose = None
        
    return motion_id, initial_pose_dict_single_motion


@hydra.main(
    version_base=None,
    config_path="/Users/nikita/Projects/diploma/fullbody/cfg",
    # config_name="config",
    config_name="eval_config",
)
def main(cfg):

    # Раскрываем интерполяции один раз в главном процессе.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Читаем чекпоинт и определяем, что осталось.
    initial_pose_dict = load_checkpoint()
    remaining_ids = [i for i in range(cfg.run.num_motions) if i not in initial_pose_dict]

    if not remaining_ids:
        logging.info("Nothing to process; checkpoint is complete.")
    else:
        for start in tqdm(range(0, len(remaining_ids), CHECKPOINT_EVERY), desc="Batches"):
            batch = remaining_ids[start: start + CHECKPOINT_EVERY]
            results = Parallel(n_jobs=-1)(
                delayed(process_motion)(motion_step, cfg_dict) for motion_step in batch
            )
            for motion_id, poses in results:
                initial_pose_dict[motion_id] = poses
            save_checkpoint(initial_pose_dict)

    _ensure_parent_dir(Path(cfg.run.initial_pose_file))

    new_data = {}
    for motion_key in tqdm(initial_pose_dict.keys()):
        new_data[motion_key] = {}
        for frame_key in initial_pose_dict[motion_key].keys():
            new_key = np.round(frame_key, 1)
            new_data[motion_key][new_key] = initial_pose_dict[motion_key][frame_key]
            print(f'Old key: {frame_key}, New key: {new_key}')

    joblib.dump(new_data, cfg.run.initial_pose_file)
    logging.info(f"Saved final data to {cfg.run.initial_pose_file}")


if __name__ == "__main__":
    main()
