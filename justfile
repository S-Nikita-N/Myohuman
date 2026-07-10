# Common dev tasks. Run `just` (no args) to list them.
#
# Thin wrappers over the real entrypoints (uv, pre-commit, pytest) — a
# convenience layer, not a source of truth. The pre-commit gate stays the
# single definition of what lint/format/types run (see AGENTS.md).

# list available recipes
default:
    @just --list

# sync the dev environment from the lockfile
install:
    uv sync

# install the pre-commit git hook so the gate runs on every commit
hooks:
    uv run pre-commit install

# run the whole pre-commit gate (lint, format, types, style checks)
lint:
    uv run pre-commit run --all-files

# fast, CPU-only, deterministic test suite
test *args:
    uv run pytest {{args}}

# tests with branch coverage (informational, no threshold)
cov:
    uv run pytest --cov

# compute IK reference datasets (see README for splits/workers)
ik *args:
    uv run python scripts/compute_ik.py {{args}}
