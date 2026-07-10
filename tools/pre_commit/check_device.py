"""Forbid hard-coded CUDA so code stays device-agnostic (CPU / CUDA / MPS).

Training code should route tensors through a single `device` variable chosen
once (e.g. cuda -> mps -> cpu), not pin `cuda` at every call site. This flags
the common hard-codes; append `# device-ok` to a line to allow a deliberate
exception. Device *selection* (`torch.cuda.is_available()`, `torch.device(...)`)
is intentionally not flagged.

    python tools/pre_commit/check_device.py FILE ...
"""

from __future__ import annotations

import re
import argparse

from pathlib import Path

########################################
#               Patterns               #
########################################

# ── Each (pattern, message) is a hard-coded-device smell. `cuda` may carry
#    an index (`cuda:0`), so an optional `:N` is part of every quoted match. ──
SMELLS = [
    (re.compile(r"\.cuda\("), "hard-coded .cuda() call"),
    (re.compile(r"""\.to\(\s*["']cuda(:\d+)?["']"""), 'hard-coded .to("cuda")'),
    (
        re.compile(r"""device\s*=\s*["']cuda(:\d+)?["']"""),
        'hard-coded device="cuda"',
    ),
]
ALLOW = "# device-ok"


########################################
#               Checking               #
########################################


def check_file(path: Path) -> bool:
    bad = False

    # ── Flag each smell unless the line opts out with `# device-ok` ──
    for number, text in enumerate(path.read_text().splitlines(), start=1):
        if ALLOW in text:
            continue
        for pattern, message in SMELLS:
            if pattern.search(text):
                print(
                    f"{path}:{number}: {message} "
                    f"(route through a device variable)",
                )
                bad = True

    return bad


########################################
#             Entry point              #
########################################


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Forbid hard-coded CUDA devices.")
    p.add_argument("files", nargs="*", type=Path)
    args = p.parse_args(argv)

    bad = False
    for path in args.files:
        bad = check_file(path) or bad
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
