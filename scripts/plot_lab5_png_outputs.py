"""Create PNG plots used by the Lab 5 PDF report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lab5"))
    parser.add_argument("--warmup-dir", type=Path, default=Path("outputs/lab5/warmup"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/lab5/plots"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def save_label_distribution(path: Path, rows: list[dict[str, str]]) -> None:
    labels = [row["label"] for row in rows]
    counts = [int(row["count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(labels[::-1], counts[::-1], color="#2f6f9f")
    ax.set_title("Sequoia Test Label Distribution")
    ax.set_xlabel("Normalized token count")
    ax.grid(axis="x", color="#d9dee3", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_corpus_sizes(path: Path, rows: list[dict[str, str]]) -> None:
    languages = []
    for row in rows:
        if row["language"] not in languages:
            languages.append(row["language"])
    splits = ["train", "dev", "test"]
    counts = {
        (row["language"], row["split"]): int(row["tokens"])
        for row in rows
    }
    x_positions = range(len(languages))
    width = 0.24
    colors = {"train": "#2f6f9f", "dev": "#d58b2a", "test": "#4f8f5f"}

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, split in enumerate(splits):
        offset = (idx - 1) * width
        ax.bar(
            [x + offset for x in x_positions],
            [counts[(language, split)] for language in languages],
            width,
            label=split,
            color=colors[split],
        )
    ax.set_title("Corpus Size by Language")
    ax.set_ylabel("Normalized token count")
    ax.set_xticks(list(x_positions), languages)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d9dee3", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_accuracy_heatmap(path: Path, rows: list[dict[str, str]]) -> None:
    languages = [field for field in rows[0] if field != "train_language"]
    train_languages = [row["train_language"] for row in rows]
    values = [[float(row[language]) for language in languages] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(values, vmin=0.35, vmax=1.0, cmap="YlGnBu")
    ax.set_title("Cross-Lingual Accuracy Matrix")
    ax.set_xlabel("Test language")
    ax.set_ylabel("Training language")
    ax.set_xticks(range(len(languages)), languages)
    ax.set_yticks(range(len(train_languages)), train_languages)
    for row_idx, row in enumerate(values):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    label_path = args.warmup_dir / "label_distribution.csv"
    if label_path.exists():
        save_label_distribution(
            args.plots_dir / "sequoia_label_distribution.png",
            read_csv(label_path),
        )

    corpus_path = args.output_dir / "corpus_stats.csv"
    if corpus_path.exists():
        save_corpus_sizes(
            args.plots_dir / "corpus_size_tokens.png",
            read_csv(corpus_path),
        )

    matrix_path = args.output_dir / "cross_lingual_accuracy_matrix.csv"
    if matrix_path.exists():
        save_accuracy_heatmap(
            args.plots_dir / "cross_lingual_accuracy_heatmap.png",
            read_csv(matrix_path),
        )

    print(f"Wrote PNG plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
