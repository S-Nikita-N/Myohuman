---
name: explore-codebase
description: >
  Understand an unfamiliar repository before working in
  it — reading, running, testing, or editing. Runs a map script that
  recursively lists every file (minus dot-prefixed names
  and .gitignored paths) and, under a single line budget on the whole
  output, prints the richest detail that fits — degrading from full content, to
  signatures with docstrings, to module docstrings, to a bare tree. Trigger
  before any task in a codebase you do not already hold in context, or when
  asked to "get up to speed" / "map" a repo.
license: Apache-2.0
metadata:
  author: Nikita Sviridov
---

# Explore a codebase before working in it

The goal is an accurate mental model at the lowest context cost. The map script
sizes the repo and picks the detail level for you; your job is to run it, then
read deliberately into whatever it summarized — and into the especially
valuable, central parts of the code.

## When to use this skill

- You are about to implement, debug, run, or test in a repo you have not read.
- You are asked to do any task involving a repo whose contents you do not know.
- You are asked to "map", "summarize", or "get up to speed on" a codebase.
- A subsystem is too large to hold whole and you need a navigable index.

## Procedure

### 1. Always run the map script

```bash
uv run python .claude/skills/explore-codebase/scripts/repo_map.py
```

It lists every file by its full path (excluding dot-prefixed names and
`.gitignore`d paths) and prints, under a single **line budget on the whole
output**, the richest detail that fits — degrading automatically:

1. full content (source / text files only — lockfiles, data, binaries are
   path-only),
2. module docstring / header comment + function·class·method signatures with
   their docstrings (`.py`),
3. module docstring / header comment only (`.py`),
4. the tree alone.

Its header line says which level was used (`# N files · detail: … · X lines`).
You did not choose the level — the budget did. Small repo → you get the actual
code; large repo → you get a navigable index.

**Consuming the output.** Run it as-is — never pipe it through `head`/`tail`.

If the harness truncates it you will see a message like:

> `Output too large (NN KB). Full output saved to: <path>. Preview (first 2KB): …`

When that happens, the 2KB preview is **NOT** the map — it is the first scrap of
it. You **MUST** `Read` that saved `<path>` to the **END**, in 1000–2000-line
chunks (the Read tool caps near 25k tokens per call), **before** answering or
acting — no matter how narrow the question looks, and no matter how small the
preview seems sufficient. Answering from the preview alone is a failure of this
skill. Never truncate the map yourself, and never stop at the preview.

### 2. Read the parts the level summarized

If the script printed full content, you already hold the code — done. If it fell
back to signatures / docstrings / tree, use that index to:

- locate the **entry points** (CLI / `main` / `__init__` re-exports);
- find **where a concept lives** (grep the output for a name, open that file);
- read **in full** with `Read` every file needed to solve the task, plus any
  especially valuable / central code the level only summarized.

### 3. Budget exceeded — narrow

If the script **raises** (more paths than the budget allows — too large to list),
it is fine for this skill to end on that error. Rerun on a subdirectory and map
one part at a time: `… repo_map.py src`.

## Output

Produce a short structured map for yourself / the user:

- **Entry points** — how a run starts, file by file.
- **Module responsibilities** — one line each.
- **Key types / public API** — the names a caller touches.
- **Where to look for X** — pointers for the task at hand.

## Anti-patterns

- Skipping the map and grepping blind — you miss where things actually live.
- Editing before you have the map — you reimplement what already exists.
- Trusting the map's signatures for behaviour — open the file before changing it.
