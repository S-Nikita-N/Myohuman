Quick setup (with existing LFS data/models)
```bash
brew install git-lfs
git lfs install
git clone <repo>
cd <repo>
source install.sh
```

Prepare dataset from scratch
1) Download SMPL (neutral) from https://smpl.is.tue.mpg.de, rename to `SMPL_NEUTRAL.pkl`, place in `data/smpl/` or get via ```git lfs```.
2) Download KIT (AMASS, SMPL-H):
```bash
uv run python scripts/download_kit.py --data-dir data/
```
3) Run the notebook `notebooks/dataset.ipynb` (keys selection steps).
4) In repo root run:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python scripts/compute_ik.py --split train --workers 9
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python scripts/compute_ik.py --split test --workers 9
```
5) Run the second part of `notebooks/dataset.ipynb` to finish processing.


