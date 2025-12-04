1) Download SMPL (neutral) from https://smpl.is.tue.mpg.de/index.html, rename the neutral model file to `SMPL_NEUTRAL.pkl`, and place it in `data/smpl/`.
2) Download KIT (AMASS, SMPL-H) from https://amass.is.tue.mpg.de and unpack it into `data/KIT/`.
3) In the repo root, run:
```bash
source install.sh
poetry run python scripts/convert_kit.py
poetry run python scripts/initial_pose.py
```
