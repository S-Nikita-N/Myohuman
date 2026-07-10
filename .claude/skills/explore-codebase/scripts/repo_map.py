"""Print a budgeted map of a repository, degrading detail to fit the budget.

The output lists every file by its full path (excluding dot-prefixed names and
anything in ``.gitignore``). Under each *source / text* file (lockfiles, data,
and binaries are path-only) it shows as much content as fits a single
line budget on the *whole* output. Detail degrades until it fits:

    1. full file content
    2. module docstring / header comment + signatures with docstrings  (.py)
    3. module docstring / header comment only                          (.py)
    4. the path list alone (no content)

If even the bare path list exceeds the budget, it raises —
the repo is too large to map whole; narrow to a subdirectory instead.

    python repo_map.py [ROOT]      # default: .
"""

from __future__ import annotations

import os
import ast
import argparse
import subprocess

from pathlib import Path

BUDGET = 10_000
DIVIDER = "─" * 60

# Suffixes whose content is worth showing; everything else (lockfiles, data,
# binaries) appears as a path only.
CONTENT_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".pyi",
    ".rst",
    ".txt",
    ".cfg",
    ".ini",
    ".yml",
    ".toml",
    ".yaml",
}

# Detail levels, richest first.
FULL, SIGNATURES, DOCSTRINGS, TREE = range(4)
LEVELS = (
    (FULL, "full content"),
    (SIGNATURES, "signatures + docstrings"),
    (DOCSTRINGS, "module docstrings"),
    (TREE, "tree only"),
)

Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef

########################################
#              Discovery               #
########################################


def candidate_files(root: Path) -> list[Path]:
    # ── Every file under root, skipping dot-prefixed dirs and files ──
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if not name.startswith("."):
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                files.append(Path(rel))
    return files


def gitignored(files: list[Path], root: Path) -> set[Path]:
    # ── Paths git would ignore (empty when there is no .gitignore) ──
    if not (root / ".gitignore").exists():
        return set()
    proc = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(str(f) for f in files),
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    return {Path(line) for line in proc.stdout.splitlines()}


########################################
#              Rendering               #
########################################


def _first_doc(node: Definition) -> str:
    doc = ast.get_docstring(node)
    return doc.splitlines()[0].strip() if doc else ""


def module_header(source: str, tree: ast.Module) -> list[str]:
    # ── Full module docstring, or the leading #-comment block if none ──
    doc = ast.get_docstring(tree)
    if doc:
        return doc.splitlines()
    header: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            header.append(stripped)
        elif not stripped and not header:
            continue
        else:
            break
    return header


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{node.name}({args}){ret}"


def _render_def(node: Definition) -> list[str]:
    # ── A class lists its methods; a function is a single line ──
    doc = _first_doc(node)
    tail = f"  — {doc!r}" if doc else ""
    if isinstance(node, ast.ClassDef):
        lines = [f"class {node.name}{tail}"]
        for child in node.body:
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                lines.append(f"  {_signature(child)}")
        return lines
    return [f"{_signature(node)}{tail}"]


def file_block(root: Path, rel: Path, level: int) -> list[str]:
    # ── The content lines for one file at the given detail level ──
    if level == TREE:
        return []

    # ── Full content, but only for source / text files ──
    if level == FULL:
        if rel.suffix not in CONTENT_SUFFIXES:
            return []
        try:
            return (root / rel).read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            return ["(binary or unreadable)"]

    # ── Signatures / docstrings — Python only ──
    if rel.suffix != ".py":
        return []
    try:
        source = (root / rel).read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []

    out = module_header(source, tree)
    if level == DOCSTRINGS:
        return out
    for node in tree.body:
        if isinstance(node, Definition):
            out.extend(_render_def(node))
    return out


def render(root: Path, files: list[Path], level: int) -> list[str] | None:
    # ── Each file is a block headed by its full path; one with content is
    #    set off by a blank line and a horizontal rule. ──
    lines: list[str] = []
    for rel in files:
        content = file_block(root, rel, level)
        if content and lines:
            lines.append("")
            lines.append(DIVIDER)
        marker = "▸ " if content else ""
        lines.append(f"{marker}{rel}")
        for body in content:
            lines.append("  " + body)
        if len(lines) > BUDGET:
            return None
    return lines


########################################
#             Entry point              #
########################################


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Budgeted repository map.")
    p.add_argument("root", nargs="?", default=".", type=Path)
    args = p.parse_args(argv)
    root: Path = args.root

    candidates = candidate_files(root)
    files = sorted(set(candidates) - gitignored(candidates, root))

    for level, label in LEVELS:
        lines = render(root, files, level)
        if lines is not None:
            print(
                f"# {len(files)} files · detail: {label} · {len(lines)} lines",
            )
            print("\n".join(lines))
            return 0

    raise RuntimeError(
        f"{len(files)} files exceed the {BUDGET}-line budget even as a bare "
        f"tree — narrow to a subdirectory and map that instead",
    )


if __name__ == "__main__":
    raise SystemExit(main())
