"""Inspect a CoNLL-U file for the warm-up questions in Lab 5."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import lab5_multilinguality as lab5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conllu_file", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lab5/warmup"),
    )
    return parser.parse_args()


def multiword_rows(sentences: list[list[lab5.ConlluToken]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sent_idx, sentence in enumerate(sentences, start=1):
        by_id = {token.token_id: token for token in sentence}
        for token in sentence:
            if "-" not in token.token_id:
                continue

            start_text, end_text = token.token_id.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            child_tokens = [
                by_id[str(child_id)]
                for child_id in range(start, end + 1)
                if str(child_id) in by_id
            ]
            rows.append(
                {
                    "sentence": str(sent_idx),
                    "token_id": token.token_id,
                    "form": token.form,
                    "children": " ".join(child.form for child in child_tokens),
                    "child_labels": "+".join(child.upos for child in child_tokens),
                }
            )
    return rows


def space_token_rows(sentences: list[list[lab5.ConlluToken]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sent_idx, sentence in enumerate(sentences, start=1):
        for token in sentence:
            if " " in token.form:
                rows.append(
                    {
                        "sentence": str(sent_idx),
                        "token_id": token.token_id,
                        "form": token.form,
                        "form_without_spaces": token.form.replace(" ", ""),
                        "upos": token.upos,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_sentences = lab5.load_conllu(args.conllu_file)
    normalized = [lab5.normalize_ud_sentence(sentence) for sentence in raw_sentences]
    labels = lab5.label_distribution(normalized)
    mwt_rows = multiword_rows(raw_sentences)
    space_rows = space_token_rows(raw_sentences)

    write_csv(args.output_dir / "label_distribution.csv", [
        {"label": label, "count": str(count)}
        for label, count in labels.most_common()
    ])
    write_csv(args.output_dir / "multiword_tokens.csv", mwt_rows)
    write_csv(args.output_dir / "tokens_with_spaces.csv", space_rows)

    print(f"sentences: {len(raw_sentences)}")
    print(f"normalized tokens: {sum(len(words) for words, _labels in normalized)}")
    print(f"labels: {dict(labels.most_common())}")
    print(f"multiword tokens: {len(mwt_rows)}")
    print(f"tokens containing spaces: {len(space_rows)}")
    if mwt_rows[:5]:
        print(f"first multiword examples: {mwt_rows[:5]}")
    if space_rows[:5]:
        print(f"first space-token examples: {space_rows[:5]}")


if __name__ == "__main__":
    main()
