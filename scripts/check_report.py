"""Check report files for common unfinished placeholders."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PATTERNS = [
    "TODO",
    "TBD",
    "Fill in",
    "Replace the placeholders",
    "final matrix will be",
    "should be generated",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=[Path("lab5_report.md"), Path("lab5_answers_draft.md")],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    found = False
    for path in args.files:
        if not path.exists():
            print(f"missing: {path}")
            found = True
            continue

        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern in DEFAULT_PATTERNS:
                if pattern.lower() in line.lower():
                    print(f"{path}:{line_number}: {line}")
                    found = True
                    break

    if found:
        raise SystemExit(1)
    print("No unfinished placeholders found.")


if __name__ == "__main__":
    main()
