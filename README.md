Quick setup (with existing LFS data/models)
```bash
brew install git-lfs
git lfs install
git clone <repo>
cd <repo>
source install.sh
```

Prepare dataset from scratch
1) Download SMPL (neutral) from https://smpl.is.tue.mpg.de, rename to `SMPL_NEUTRAL.pkl`, place in `data/smpl/`.
2) Download KIT (AMASS, SMPL-H) from https://amass.is.tue.mpg.de, unpack to `data/KIT/`.
3) Run the first part of `notebooks/dataset.ipynb` (keys selection steps).
4) In repo root:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 poetry run python scripts/compute_ik.py --split train --workers 9
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 poetry run python scripts/compute_ik.py --split test --workers 9
```
5) Run the second part of `notebooks/dataset.ipynb` to finish processing.
