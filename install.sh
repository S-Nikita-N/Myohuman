#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing poetry..."
pip install poetry

echo "Installing project dependencies..."
poetry install

echo "Installing chumpy (PEP517 isolation off)..."
PIP_NO_BUILD_ISOLATION=1 poetry run pip install "chumpy==0.70"

echo "Patching chumpy for Python >=3.11 (getargspec -> getfullargspec)..."
CH_FILE=$(
poetry run python <<'PY'
import site
import pathlib

site_packages = next(p for p in site.getsitepackages() if "site-packages" in p)
print(pathlib.Path(site_packages) / "chumpy" / "ch.py")
PY
)
if [ -n "$CH_FILE" ] && [ -f "$CH_FILE" ]; then
    # BSD/macOS sed needs an explicit backup suffix; '' keeps no backup file.
    sed -i '' 's/inspect.getargspec/inspect.getfullargspec/g' "$CH_FILE"
    echo "Patched $CH_FILE"
else
    echo "chumpy/ch.py not found; skip patch"
fi

echo "Patching chumpy __init__.py for NumPy>=2.0 imports..."
CH_INIT=$(
poetry run python <<'PY'
import site
import pathlib

site_packages = next(p for p in site.getsitepackages() if "site-packages" in p)
f = pathlib.Path(site_packages) / "chumpy" / "__init__.py"
print(f)
PY
)
if [ -n "$CH_INIT" ] && [ -f "$CH_INIT" ]; then
    sed -i '' 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/g' "$CH_INIT"
    echo "Patched $CH_INIT"
else
    echo "chumpy/__init__.py not found; skip patch"
fi

echo "Done. To enter the venv run: poetry shell"
