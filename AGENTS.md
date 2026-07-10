# AGENTS.md — how this project is built

The engineering workflow: how the code is written, checked and run. Read it
first to be up to speed. Code-style rules live in the `code-style` rule
(`.claude/rules/code-style.md`), auto-loaded when a Python file is read.
Repo-specific commands (how to run things) live in the README.

## Toolchain

| Concern                  | Tool                                       |
| ------------------------ | ------------------------------------------ |
| dependencies & packaging | uv (src layout)                            |
| lint + format            | Ruff (`ruff check`, `ruff format`)         |
| type checking            | mypy (typed `tools/`; src typing TBD)          |
| tests                    | pytest (+ golden characterization tests)   |
| config                   | Hydra structured configs                   |
| commit gate              | pre-commit                                 |

## Quick commands (`just`)

A `justfile` provides shortcuts — run `just` to list them:

| Recipe          | Runs                                |
| --------------- | ----------------------------------- |
| `just install`  | `uv sync`                           |
| `just hooks`    | `uv run pre-commit install`         |
| `just lint`     | `uv run pre-commit run --all-files` |
| `just test`     | `uv run pytest`                     |
| `just cov`      | `uv run pytest --cov`               |
| `just ik`       | `uv run python scripts/compute_ik.py` |

It is a thin convenience layer that **calls** these entrypoints — the
`pre-commit` gate stays the single source of truth, so `just` is never required.

## Code style

The full rules live in `.claude/rules/code-style.md` — a path-scoped rule that
auto-loads when a Python file is read. In short: banners, length ladders,
four-block absolute imports, trailing commas, 80-column lines. The mechanical
rules are **enforced by pre-commit** through the `check-*` hooks; the rest is
human/agent judgment. Each violation message cites the `code-style §N` rule it
breaks. Vendored math (`poselib/`, `utils/transformation.py`,
`utils/pytorch3d_transforms.py`) is third-party and excluded from the style.

## pre-commit is the gate

Everything runs before each commit (`pre-commit install`), and CI runs the same
set — `.pre-commit-config.yaml` is the single source of truth, never duplicated.

Hooks, in order:

- **hygiene** — trailing-whitespace, end-of-file, check-yaml, typos
- **format** — `add-trailing-comma` → `ruff-format` (explode multi-line
  constructs one element per line)
- **lint** — `ruff check`
- **types** — `mypy`
- **style as code** — `check-imports`, `check-banners`, `check-device`: project
  rules encoded as executable checks (what ruff / mypy cannot express), each
  citing the `code-style` rule it enforces.

**`pre-commit` only sees git-tracked files** — a new, unstaged file is silently
skipped by `pre-commit run --all-files`. `git add` new files before trusting the
gate.

Run all: `pre-commit run --all-files`. Run one: `pre-commit run ruff-check`.

## Linting & type checking

- **ruff** — `ruff check` (rules `E, F, UP, B, SIM, C4, PT`; isort `I` is off,
  imports are ordered by length — code-style §6) and `ruff format`.
- **mypy** — runs over the typed `tools/`. Annotating the legacy `src/`
  (hundreds of implicit-Optional / attr-defined findings) is a follow-up.

## Tests

CPU-only, deterministic (seeded, tiny tensors), no downloads — golden
characterization tests over the math, env and reward logic. Run on every commit
and in CI. Regenerate goldens only on an intentional behavior change
(`uv run python tests/golden/generate_goldens.py`).

Run: `pytest` (or `just test`).

## CI

Every push / PR runs `pre-commit run --all-files` (lint, format, types, style
checks) + `pytest`, on cheap CPU runners. Docs-only pushes are skipped
(`paths-ignore`). CI invokes `pre-commit`, not the tools directly — one source
of truth shared with local commits.

## Commits

- Run pre-commit before committing; fix what the `check-*` hooks flag.
- `git add` new files before trusting the gate — pre-commit only sees tracked
  files, so untracked ones are silently skipped.
- Conventional prefixes (`feat:`, `fix:`, `docs:`, `style:`, `chore:`, `test:`),
  one logical change per commit.

For code review, use the built-in `/code-review`. For test coverage + mutation
quality of a change, delegate the read-only `test-coverage-auditor` subagent.
