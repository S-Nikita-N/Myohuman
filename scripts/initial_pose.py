import sys
import os
import hydra
import joblib
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path

from myohuman.env.myolegs_im import MyoLegsIm

sys.path.append(os.getcwd())

# CHECKPOINT_PATH = Path("/Users/nikita/Projects/diploma/fullbody/data/dataset/initial_pose_checkpoint.pkl")
CHECKPOINT_PATH = Path("/workspace/Myohuman/data/tmp/ik_train_ckpt.pkl")
CHECKPOINT_EVERY = 500  # сохраняем прогресс каждые N выполненных motions

# Путь к файлу логов (можно вынести в конфиг или оставить здесь)
LOG_FILE_PATH = Path("/workspace/Myohuman/data/tmp/processing.log") 


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def setup_file_logger(log_path: Path):
    """Настраивает дополнительный вывод логов в файл."""
    # Создаем форматтер
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    
    # Создаем хендлер для файла
    file_handler = logging.FileHandler(log_path, mode='a') # 'a' - append (дописывать), 'w' - перезаписывать
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Добавляем хендлер к корневому логгеру
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


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
    # joblib.dump атомарен в большинстве случаев, но для надежности сохраняем
    joblib.dump(data, CHECKPOINT_PATH)
    logging.info(f"Saved checkpoint with {len(data)} motions to {CHECKPOINT_PATH}")


def process_motion(motion_step, cfg_dict):
    """
    Обрабатывает одну анимацию для создания словаря начальных поз.
    """
    try:
        # В subprocess нет зарегистрированного hydra resolver, поэтому используем уже резолвленный dict.
        cfg = OmegaConf.create(cfg_dict)
        env = MyoLegsIm(cfg)
        env.initial_pos_data = {}
        
        env.motion_lib.load_motions(
            env.motion_lib_cfg,
            shape_params=env.gender_betas,
            random_sample=False,
            start_idx=motion_step,
        )
        
        # Проверка, есть ли загруженные движения
        if not env.motion_lib._curr_motion_ids:
            return None, None

        motion_id = env.motion_lib._curr_motion_ids[0]
        motion_length = env.motion_lib._motion_lengths[0]
        dt = env.motion_lib._motion_dt[0]
        
        initial_pose_dict_single_motion = {}
        
        # Используем numpy.arange с небольшим запасом, чтобы включить последний кадр, если нужно
        for start_time in np.arange(0, motion_length, dt):
            env.reset(options={'start_time': start_time})
            initial_pose_dict_single_motion[start_time] = env.initial_pose
            env.initial_pose = None
            
        return motion_id, initial_pose_dict_single_motion
    except Exception as e:
        # Логируем ошибку. Чтобы она попала в общий файл из подпроцесса, 
        # может потребоваться доп. настройка, но Hydra обычно перехватывает stderr.
        logging.error(f"Error processing motion step {motion_step}: {e}")
        return None, None


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config",
)
def main(cfg):
    # 1. НАСТРОЙКА ЛОГГЕРА
    # Мы вызываем это внутри main, чтобы Hydra уже инициализировала свои логгеры,
    # и мы просто добавляем к ним еще один вывод в наш файл.
    setup_file_logger(LOG_FILE_PATH)
    
    logging.info(f"Script started. Logging to {LOG_FILE_PATH.absolute()}")

    # Раскрываем интерполяции один раз в главном процессе.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Читаем чекпоинт и определяем, что осталось.
    initial_pose_dict = load_checkpoint()
    remaining_ids = [i for i in range(cfg.run.num_motions) if i not in initial_pose_dict]

    if not remaining_ids:
        logging.info("Nothing to process; checkpoint is complete.")
    else:
        logging.info(f"Starting processing for {len(remaining_ids)} motions...")
        
        # Используем ProcessPoolExecutor для асинхронного выполнения
        with ProcessPoolExecutor(max_workers=None) as executor:
            # Словарь future -> motion_step (для отладки, если нужно)
            futures = {
                executor.submit(process_motion, m_step, cfg_dict): m_step 
                for m_step in remaining_ids
            }
            
            since_last_save = 0
            
            # as_completed возвращает результаты по мере их готовности (в любом порядке)
            for future in tqdm(as_completed(futures), total=len(remaining_ids), desc="Processing"):
                motion_id, poses = future.result()
                
                if motion_id is not None:
                    initial_pose_dict[motion_id] = poses
                    since_last_save += 1
                
                # Если накопили достаточно изменений — сохраняем
                if since_last_save >= CHECKPOINT_EVERY:
                    save_checkpoint(initial_pose_dict)
                    since_last_save = 0
            
            # Финальное сохранение чекпоинта после цикла
            if since_last_save > 0:
                save_checkpoint(initial_pose_dict)

    _ensure_parent_dir(Path(cfg.run.initial_pose_file))

    logging.info("Post-processing data (rounding keys)...")
    new_data = {}
    for motion_key in tqdm(initial_pose_dict.keys(), desc="Rounding keys"):
        new_data[motion_key] = {}
        for frame_key in initial_pose_dict[motion_key].keys():
            new_key = np.round(frame_key, 1)
            new_data[motion_key][new_key] = initial_pose_dict[motion_key][frame_key]

    joblib.dump(new_data, cfg.run.initial_pose_file)
    logging.info(f"Saved final data to {cfg.run.initial_pose_file}")


if __name__ == "__main__":
    main()
# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
# python scripts/initial_pose.py \
#     run.initial_pose_file="/workspace/Myohuman/data/tmp/ik_train.pkl"
