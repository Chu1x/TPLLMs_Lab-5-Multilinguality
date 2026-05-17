"""Create lightweight SVG plots for Lab 5 outputs.

This script intentionally avoids plotting dependencies. It writes simple SVG
bar charts for:

- Sequoia warm-up label distribution
- corpus sizes by language/split
- optional cross-lingual accuracy matrix heatmap
"""

from __future__ import annotations

import argparse
import csv
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
        "--warmup-dir",
        type=Path,
        default=Path("outputs/lab5/warmup"),
        help="Warm-up output directory.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("outputs/lab5/plots"),
        help="Directory where SVG plots are written.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_horizontal_bar_chart(
    path: Path,
    title: str,
    rows: list[tuple[str, float]],
    value_label: str,
    width: int = 900,
) -> None:
    left = 150
    right = 40
    top = 58
    row_height = 28
    bar_height = 18
    chart_width = width - left - right
    height = top + row_height * len(rows) + 34
    max_value = max((value for _label, value in rows), default=1.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700">{esc(title)}</text>',
        f'<text x="{left}" y="50" font-family="Arial, sans-serif" font-size="12" fill="#555">{esc(value_label)}</text>',
    ]

    for idx, (label, value) in enumerate(rows):
        y = top + idx * row_height
        bar_width = 0 if max_value == 0 else (value / max_value) * chart_width
        lines.extend(
            [
                f'<text x="{left - 12}" y="{y + 14}" text-anchor="end" font-family="Arial, sans-serif" font-size="13">{esc(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{chart_width}" height="{bar_height}" fill="#edf1f5"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="#2f6f9f"/>',
                f'<text x="{left + bar_width + 6:.1f}" y="{y + 14}" font-family="Arial, sans-serif" font-size="12" fill="#333">{value:g}</text>',
            ]
        )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_grouped_corpus_chart(
    path: Path,
    rows: list[dict[str, str]],
    width: int = 950,
) -> None:
    languages = []
    for row in rows:
        if row["language"] not in languages:
            languages.append(row["language"])
    splits = ["train", "dev", "test"]
    colors = {"train": "#2f6f9f", "dev": "#d58b2a", "test": "#4f8f5f"}
    counts = {
        (row["language"], row["split"]): int(row["tokens"])
        for row in rows
    }
    max_value = max(counts.values(), default=1)

    left = 76
    right = 40
    top = 64
    bottom = 70
    chart_width = width - left - right
    chart_height = 360
    height = top + chart_height + bottom
    group_width = chart_width / len(languages)
    bar_width = 24

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700">Corpus Size by Language</text>',
        f'<text x="{left}" y="50" font-family="Arial, sans-serif" font-size="12" fill="#555">Token counts in normalized UD corpora</text>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{width - right}" y2="{top + chart_height}" stroke="#333"/>',
    ]

    for tick in range(0, 6):
        value = max_value * tick / 5
        y = top + chart_height - (value / max_value) * chart_height
        lines.extend(
            [
                f'<line x1="{left - 5}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e3e7eb"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#555">{int(value):,}</text>',
            ]
        )

    for lang_idx, language in enumerate(languages):
        group_x = left + lang_idx * group_width + group_width / 2
        for split_idx, split in enumerate(splits):
            value = counts[(language, split)]
            bar_height = (value / max_value) * chart_height
            x = group_x + (split_idx - 1) * (bar_width + 6) - bar_width / 2
            y = top + chart_height - bar_height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[split]}"/>'
            )
        lines.append(
            f'<text x="{group_x:.1f}" y="{top + chart_height + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(language)}</text>'
        )

    legend_x = left
    legend_y = height - 24
    for idx, split in enumerate(splits):
        x = legend_x + idx * 90
        lines.extend(
            [
                f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{colors[split]}"/>',
                f'<text x="{x + 20}" y="{legend_y}" font-family="Arial, sans-serif" font-size="12">{split}</text>',
            ]
        )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def color_for_score(score: float) -> str:
    score = max(0.0, min(1.0, score))
    red = int(245 - score * 150)
    green = int(247 - score * 75)
    blue = int(250 - score * 145)
    return f"#{red:02x}{green:02x}{blue:02x}"


def write_matrix_heatmap(path: Path, rows: list[dict[str, str]], width: int = 760) -> None:
    languages = [field for field in rows[0] if field != "train_language"]
    cell = 82
    left = 112
    top = 82
    height = top + cell * len(rows) + 50
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="112" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700">Cross-Lingual Accuracy Matrix</text>',
        '<text x="112" y="50" font-family="Arial, sans-serif" font-size="12" fill="#555">Rows = training language, columns = test language</text>',
    ]
    for col_idx, language in enumerate(languages):
        x = left + col_idx * cell + cell / 2
        lines.append(
            f'<text x="{x:.1f}" y="{top - 16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(language)}</text>'
        )

    for row_idx, row in enumerate(rows):
        y = top + row_idx * cell
        lines.append(
            f'<text x="{left - 14}" y="{y + cell / 2 + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13">{esc(row["train_language"])}</text>'
        )
        for col_idx, language in enumerate(languages):
            x = left + col_idx * cell
            value = float(row[language])
            lines.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" fill="{color_for_score(value)}" stroke="#ffffff" stroke-width="2"/>',
                    f'<text x="{x + cell / 2 - 2:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#1f2933">{value:.3f}</text>',
                ]
            )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    label_path = args.warmup_dir / "label_distribution.csv"
    if label_path.exists():
        label_rows = read_csv(label_path)
        write_horizontal_bar_chart(
            args.plots_dir / "sequoia_label_distribution.svg",
            "Sequoia Test Label Distribution",
            [(row["label"], float(row["count"])) for row in label_rows],
            "Normalized token count",
        )

    corpus_path = args.output_dir / "corpus_stats.csv"
    if corpus_path.exists():
        write_grouped_corpus_chart(
            args.plots_dir / "corpus_size_tokens.svg",
            read_csv(corpus_path),
        )

    matrix_path = args.output_dir / "cross_lingual_accuracy_matrix.csv"
    if matrix_path.exists():
        write_matrix_heatmap(
            args.plots_dir / "cross_lingual_accuracy_heatmap.svg",
            read_csv(matrix_path),
        )

    print(f"Wrote plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
