import sys
from pathlib import Path

# Make tests/helpers.py importable as `helpers` from any test module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
