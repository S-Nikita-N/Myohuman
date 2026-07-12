import os
import sys

import torch
import pytest

from pathlib import Path

# Make tests/helpers.py importable as `helpers` from any test module.
sys.path.insert(0, str(Path(__file__).resolve().parent))


########################################
#           Two-stage gating           #
########################################

# Stage 1 (default): fast, CPU-only, deterministic — runs everywhere.
# Stage 2: heavy / accelerator tests marked `@pytest.mark.stage2`; run only
# when a GPU/MPS accelerator is present, or when forced with RUN_STAGE2=1.


def _accelerator_available() -> bool:
    if torch.cuda.is_available():
        return True
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _stage2_enabled() -> bool:
    return os.environ.get("RUN_STAGE2") == "1" or _accelerator_available()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "stage2: heavy/accelerator test; runs only when a GPU/MPS device is "
        "present or RUN_STAGE2=1",
    )


def pytest_collection_modifyitems(config, items):
    if _stage2_enabled():
        return
    skip = pytest.mark.skip(
        reason="stage-2 test (needs a GPU/MPS device or RUN_STAGE2=1)",
    )
    for item in items:
        if "stage2" in item.keywords:
            item.add_marker(skip)
