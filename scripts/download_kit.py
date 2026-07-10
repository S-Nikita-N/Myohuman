#!/usr/bin/env python3
"""
Скачивает датасет KIT с Hugging Face (N1ksw1r/KIT) в папку data/KIT.
"""

import argparse

from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "N1ksw1r/KIT"


########################################
#             Download KIT             #
########################################
def main():
    parser = argparse.ArgumentParser(
        description="Скачать KIT датасет с Hugging Face в data/KIT",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Папка data (по умолчанию: Myohuman/data)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = args.data_dir or (project_root / "data")
    local_dir = data_dir / "KIT"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Скачиваю {REPO_ID} в {local_dir} ")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print(f"Готово: {local_dir}")


# uv run python scripts/download_kit.py --data-dir data/
if __name__ == "__main__":
    main()
