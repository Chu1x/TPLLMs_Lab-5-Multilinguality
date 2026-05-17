"""Generate Markdown snippets from Lab 5 output files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lab5"),
        help="Experiment output directory.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of decimals to print.",
    )
    return parser.parse_args()


def read_matrix(output_dir: Path) -> list[dict[str, str]]:
    matrix_path = output_dir / "cross_lingual_accuracy_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing matrix: {matrix_path}")
    with matrix_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def markdown_matrix(rows: list[dict[str, str]], digits: int) -> str:
    languages = [field for field in rows[0] if field != "train_language"]
    lines = [
        "| Train \\\\ Test | " + " | ".join(languages) + " |",
        "|---|" + "|".join("---:" for _language in languages) + "|",
    ]
    for row in rows:
        values = [
            f"{float(row[language]):.{digits}f}"
            for language in languages
        ]
        lines.append(f"| {row['train_language']} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def diagonal_and_best_transfer(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    languages = [field for field in rows[0] if field != "train_language"]
    diagonal_notes: list[str] = []
    transfer_notes: list[str] = []

    for row in rows:
        train_language = row["train_language"]
        if train_language in languages:
            diagonal_notes.append(
                f"- `{train_language}` in-language accuracy: "
                f"{float(row[train_language]):.3f}"
            )

        transfer_scores = [
            (language, float(row[language]))
            for language in languages
            if language != train_language
        ]
        if transfer_scores:
            best_language, best_score = max(transfer_scores, key=lambda item: item[1])
            worst_language, worst_score = min(transfer_scores, key=lambda item: item[1])
            transfer_notes.append(
                f"- Model trained on `{train_language}` transfers best to "
                f"`{best_language}` ({best_score:.3f}) and worst to "
                f"`{worst_language}` ({worst_score:.3f})."
            )

    return diagonal_notes, transfer_notes


def config_summary(output_dir: Path, rows: list[dict[str, str]]) -> str:
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists():
        return "No `experiment_config.json` found."

    config = json.loads(config_path.read_text(encoding="utf-8"))
    completed_train_languages = [row["train_language"] for row in rows]
    return "\n".join(
        [
            f"- Model: `{config['model_checkpoint']}`",
            f"- Languages: {', '.join(config['languages'])}",
            f"- Completed matrix rows: {', '.join(completed_train_languages)}",
            f"- Last run training languages: {', '.join(config['train_languages'])}",
            f"- Epochs: {config['epochs']}",
            f"- Batch size: {config['batch_size']}",
            f"- Learning rate: {config['learning_rate']}",
            f"- Weight decay: {config['weight_decay']}",
            f"- Seed: {config['seed']}",
            f"- CPU only: {config['use_cpu']}",
            f"- Label count: {config['label_count']}",
        ]
    )


def main() -> None:
    args = parse_args()
    rows = read_matrix(args.output_dir)
    diagonal_notes, transfer_notes = diagonal_and_best_transfer(rows)

    print("## Experiment Configuration")
    print()
    print(config_summary(args.output_dir, rows))
    print()
    print("## Accuracy Matrix")
    print()
    print(markdown_matrix(rows, args.digits))
    print()
    print("## Automatic Notes")
    print()
    print("In-language results:")
    print("\n".join(diagonal_notes))
    print()
    print("Transfer directions:")
    print("\n".join(transfer_notes))


if __name__ == "__main__":
    main()
