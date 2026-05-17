"""Format a cross-lingual accuracy CSV as a Markdown table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_csv", type=Path)
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of decimals to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.matrix_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise SystemExit(f"No rows found in {args.matrix_csv}")

    languages = [field for field in rows[0] if field != "train_language"]
    header = ["Train \\\\ Test", *languages]
    alignment = ["---", *["---:" for _language in languages]]

    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(alignment) + " |")
    for row in rows:
        values = [row["train_language"]]
        for language in languages:
            values.append(f"{float(row[language]):.{args.digits}f}")
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
