"""Check structural comments line up (code-style §1-§2).

Tier-1 — boxed `#` banners (border / `# centered label #` / border): every
banner in a file shares one width, the middle lines up, the label is centred,
and the banner is indented to the code it heads. A module that defines a
function or class must carry at least one such banner; empty and pure
re-export modules (no def/class) are exempt.

Tier-2 — `# ──` block intros: a blank line above (separating it from the
previous block) and code directly below (the block hugs its intro). A suite
header (`def …:`, `for …:`) or a docstring close above needs no blank — the
intro opens the first block of that suite.

    python tools/pre_commit/check_banners.py FILE ...
"""

from __future__ import annotations

import ast
import argparse

from pathlib import Path

########################################
#              Detection               #
########################################


def is_border(text: str) -> bool:
    return len(text) >= 10 and set(text) == {"#"}


def is_middle(text: str) -> bool:
    inner = set(text) - {"#"}
    return (
        len(text) >= 10
        and text.startswith("#")
        and text.endswith("#")
        and bool(inner)
    )


def is_intro(text: str) -> bool:
    return text.lstrip().startswith("# ──")


def continues_structure(text: str) -> bool:
    # ── A suite header, docstring close, or an open/continuing collection
    #    above: the intro needs no blank line before it. ──
    return text.rstrip().endswith((":", '"""', "'''", "[", "{", "(", ","))


def banner_tops(lines: list[str]) -> list[int]:
    # ── Index of each banner's top border (border / middle / border) ──
    tops: list[int] = []
    for i in range(len(lines) - 2):
        top = lines[i].strip()
        mid = lines[i + 1].strip()
        bot = lines[i + 2].strip()
        if is_border(top) and is_border(bot) and is_middle(mid):
            tops.append(i)
    return tops


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def headed_indent(lines: list[str], start: int) -> int | None:
    # ── Indentation of the first non-blank line a banner heads ──
    for line in lines[start:]:
        if line.strip():
            return indent_of(line)
    return None


def defines_symbol(source: str) -> bool:
    # ── True when the module has a top-level function or class — a file with
    #    real structure, which §1 asks to head with a section banner. Empty
    #    and pure re-export modules (no def/class) are exempt. ──
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False  # leave syntax errors to ruff / python
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    return any(isinstance(node, kinds) for node in tree.body)


########################################
#            Tier-1 banners            #
########################################


def check_banners(path: Path, lines: list[str]) -> bool:
    tops = banner_tops(lines)
    if not tops:
        return False
    bad = False

    # ── Every banner in the file shares one width ──
    widths = sorted({len(lines[i].strip()) for i in tops})
    if len(widths) > 1:
        print(f"{path}: banners have mixed widths {widths} (code-style §1)")
        bad = True

    # ── Middle lines up, label centred, indent matches the headed code ──
    for i in tops:
        number = i + 2
        width = len(lines[i].strip())
        mid = lines[i + 1].strip()
        bottom = len(lines[i + 2].strip())

        if len(mid) != width:
            print(
                f"{path}:{number}: banner middle width "
                f"{len(mid)} != border {width} (code-style §1)",
            )
            bad = True

        if bottom != width:
            print(
                f"{path}:{number + 1}: banner bottom border width "
                f"{bottom} != top border {width} (code-style §1)",
            )
            bad = True
        else:
            content = mid[1:-1]
            left = len(content) - len(content.lstrip())
            right = len(content) - len(content.rstrip())
            if right - left not in (0, 1):
                print(
                    f"{path}:{number}: banner label not centred "
                    f"(pad {left}/{right}) (code-style §1)",
                )
                bad = True

        indent = indent_of(lines[i])
        headed = headed_indent(lines, i + 3)
        if headed is not None and headed != indent:
            print(
                f"{path}:{number}: banner indent {indent} != "
                f"headed code indent {headed} (code-style §1)",
            )
            bad = True

    return bad


def check_has_banner(path: Path, lines: list[str]) -> bool:
    # ── A structured module (defines a function/class) must carry at least
    #    one Tier-1 banner; trivial modules are exempt (see defines_symbol). ──
    if not defines_symbol("\n".join(lines)):
        return False
    if banner_tops(lines):
        return False
    print(
        f"{path}:1: file defines a function/class but has no Tier-1 "
        f"banner — add at least one section banner (code-style §1)",
    )
    return True


########################################
#            Tier-2 intros             #
########################################


def check_intros(path: Path, lines: list[str]) -> bool:
    bad = False

    # ── `# ──` intros: blank above (unless a suite header), code below ──
    for j, line in enumerate(lines):
        if not is_intro(line):
            continue
        number = j + 1

        below = lines[j + 1] if j + 1 < len(lines) else ""
        if not below.strip():
            print(
                f"{path}:{number}: block intro detached — "
                f"blank line below it (code-style §2)",
            )
            bad = True

        if (
            j > 0
            and lines[j - 1].strip()
            and not continues_structure(lines[j - 1])
        ):
            print(
                f"{path}:{number}: block intro needs a "
                f"blank line above it (code-style §2)",
            )
            bad = True

    return bad


########################################
#             Entry point              #
########################################


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Check Tier-1 / Tier-2 structural comments.",
    )
    p.add_argument("files", nargs="*", type=Path)
    args = p.parse_args(argv)

    bad = False
    for path in args.files:
        lines = path.read_text().splitlines()
        bad = check_banners(path, lines) or bad
        bad = check_intros(path, lines) or bad
        bad = check_has_banner(path, lines) or bad
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
