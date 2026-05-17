"""Run corpus statistics, mBERT fine-tuning, and cross-lingual evaluation.

Expected data layout:

data/
  fr/
    train.conllu
    dev.conllu
    test.conllu
  es/
    train.conllu
    dev.conllu
    test.conllu

Example:

python3 scripts/run_lab5_experiments.py \
    --languages fr es en ar zh \
    --data-dir data \
    --output-dir outputs/lab5 \
    --stats-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import lab5_multilinguality as lab5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="Language folder names under --data-dir, e.g. fr es en ar zh.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing one subdirectory per language.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lab5"),
        help="Directory where models, stats, and matrices are written.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute corpus statistics and truncation reports.",
    )
    parser.add_argument(
        "--skip-truncation",
        action="store_true",
        help="Skip mBERT tokenizer loading when only corpus counts are needed.",
    )
    parser.add_argument(
        "--train-languages",
        nargs="+",
        default=None,
        help="Optional subset of --languages to train. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training examples per language.",
    )
    parser.add_argument(
        "--max-dev-samples",
        type=int,
        default=None,
        help="Optional cap on development examples per language.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on test examples per language.",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU training/evaluation. Useful if MPS runs out of memory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in cross_lingual_accuracy_matrix.csv.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional tokenizer max_length. Defaults to the model maximum.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def maybe_select(dataset, max_samples: int | None):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return dataset.select(range(max_samples))


def conllu_path(data_dir: Path, language: str, split: str) -> Path:
    path = data_dir / language / f"{split}.conllu"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split} file for {language}: {path}. "
            "Expected layout is data/<language>/<split>.conllu."
        )
    return path


def load_all_corpora(
    data_dir: Path,
    languages: list[str],
) -> dict[str, dict[str, list[tuple[list[str], list[str]]]]]:
    corpora: dict[str, dict[str, list[tuple[list[str], list[str]]]]] = {}
    for language in languages:
        corpora[language] = {}
        for split in ("train", "dev", "test"):
            corpora[language][split] = lab5.load_normalized_conllu(
                conllu_path(data_dir, language, split)
            )
    return corpora


def write_corpus_stats(
    output_dir: Path,
    corpora: dict[str, dict[str, list[tuple[list[str], list[str]]]]],
    tokenizer=None,
    max_length: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "corpus_stats.csv"
    label_path = output_dir / "label_distribution_test.json"
    truncation_path = output_dir / "truncation_report.json"

    with stats_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["language", "split", "sentences", "tokens"],
        )
        writer.writeheader()
        for language, split_map in corpora.items():
            for split, corpus in split_map.items():
                sentences, tokens = lab5.count_sentences_and_tokens(corpus)
                writer.writerow(
                    {
                        "language": language,
                        "split": split,
                        "sentences": sentences,
                        "tokens": tokens,
                    }
                )

    label_distributions = {
        language: dict(lab5.label_distribution(split_map["test"]))
        for language, split_map in corpora.items()
    }
    label_path.write_text(
        json.dumps(label_distributions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if tokenizer is None:
        return

    truncation_reports = {
        language: {
            split: lab5.truncation_report(tokenizer, corpus, max_length=max_length)
            for split, corpus in split_map.items()
        }
        for language, split_map in corpora.items()
    }
    truncation_path.write_text(
        json.dumps(truncation_reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_shared_label_maps(
    corpora: dict[str, dict[str, list[tuple[list[str], list[str]]]]],
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Build one global observed label schema for consistent label IDs."""

    all_corpora = [
        corpus
        for split_map in corpora.values()
        for corpus in split_map.values()
    ]
    return lab5.build_label_maps(all_corpora)


def make_training_args(args: argparse.Namespace, train_language: str):
    from transformers import TrainingArguments

    train_output_dir = args.output_dir / "models" / train_language
    common_kwargs = {
        "output_dir": str(train_output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_dir": str(args.output_dir / "logs" / train_language),
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "seed": args.seed,
        "report_to": "none",
        "use_cpu": args.use_cpu,
    }

    try:
        return TrainingArguments(evaluation_strategy="epoch", **common_kwargs)
    except TypeError:
        return TrainingArguments(eval_strategy="epoch", **common_kwargs)


def train_one_language(
    args: argparse.Namespace,
    train_language: str,
    tokenizer,
    label_list: list[str],
    label2id: dict[str, int],
    id2label: dict[int, str],
):
    from transformers import Trainer

    train_ds = lab5.build_dataset_from_conllu(
        conllu_path(args.data_dir, train_language, "train"),
        tokenizer,
        label2id,
        max_length=args.max_length,
    )
    train_ds = maybe_select(train_ds, args.max_train_samples)
    dev_ds = lab5.build_dataset_from_conllu(
        conllu_path(args.data_dir, train_language, "dev"),
        tokenizer,
        label2id,
        max_length=args.max_length,
    )
    dev_ds = maybe_select(dev_ds, args.max_dev_samples)

    model = lab5.make_model(len(label_list), id2label, label2id)
    trainer = Trainer(
        model=model,
        args=make_training_args(args, train_language),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=lab5.compute_accuracy_from_logits,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "models" / train_language / "final"))
    return trainer


def build_test_datasets(
    args: argparse.Namespace,
    tokenizer,
    label2id: dict[str, int],
) -> dict[str, object]:
    return {
        language: maybe_select(
            lab5.build_dataset_from_conllu(
                conllu_path(args.data_dir, language, "test"),
                tokenizer,
                label2id,
                max_length=args.max_length,
            ),
            args.max_test_samples,
        )
        for language in args.languages
    }


def evaluate_one_row(train_language: str, trainer, test_datasets) -> dict[str, str | float]:
    row: dict[str, str | float] = {"train_language": train_language}
    for test_language, test_ds in test_datasets.items():
        metrics = trainer.evaluate(eval_dataset=test_ds)
        row[test_language] = metrics["eval_accuracy"]
    return row


def write_matrix(output_dir: Path, languages: list[str], matrix_rows) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cross_lingual_accuracy_matrix.csv"
    json_path = output_dir / "cross_lingual_accuracy_matrix.json"

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["train_language", *languages],
        )
        writer.writeheader()
        writer.writerows(matrix_rows)

    json_path.write_text(
        json.dumps(matrix_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_experiment_config(args: argparse.Namespace, label_list: list[str]) -> None:
    config = {
        "signature": experiment_signature(args),
        "languages": args.languages,
        "train_languages": args.train_languages or args.languages,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "model_checkpoint": lab5.MODEL_CHECKPOINT,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "use_cpu": args.use_cpu,
        "max_train_samples": args.max_train_samples,
        "max_dev_samples": args.max_dev_samples,
        "max_test_samples": args.max_test_samples,
        "max_length": args.max_length,
        "label_count": len(label_list),
        "labels": label_list,
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def experiment_signature(args: argparse.Namespace) -> dict[str, object]:
    return {
        "languages": args.languages,
        "data_dir": str(args.data_dir),
        "model_checkpoint": lab5.MODEL_CHECKPOINT,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_dev_samples": args.max_dev_samples,
        "max_test_samples": args.max_test_samples,
        "max_length": args.max_length,
    }


def read_saved_signature(output_dir: Path) -> dict[str, object] | None:
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("signature")


def read_existing_matrix(
    output_dir: Path,
    expected_signature: dict[str, object] | None = None,
) -> list[dict[str, str | float]]:
    csv_path = output_dir / "cross_lingual_accuracy_matrix.csv"
    if not csv_path.exists():
        return []

    saved_signature = read_saved_signature(output_dir)
    if expected_signature is not None and saved_signature != expected_signature:
        raise ValueError(
            "Existing matrix was produced with a different experiment config. "
            "Use a different --output-dir or remove the old matrix/config."
        )

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    parsed_rows: list[dict[str, str | float]] = []
    for row in rows:
        parsed_row: dict[str, str | float] = {"train_language": row["train_language"]}
        for key, value in row.items():
            if key == "train_language":
                continue
            parsed_row[key] = float(value)
        parsed_rows.append(parsed_row)
    return parsed_rows


def cleanup_after_trainer(trainer) -> None:
    model = getattr(trainer, "model", None)
    del trainer
    if model is not None:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpora = load_all_corpora(args.data_dir, args.languages)
    label_list, label2id, id2label = build_shared_label_maps(corpora)
    (args.output_dir / "label_list.json").write_text(
        json.dumps(label_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tokenizer = None if args.skip_truncation else lab5.make_tokenizer()
    write_corpus_stats(args.output_dir, corpora, tokenizer, max_length=args.max_length)

    if args.stats_only:
        write_experiment_config(args, label_list)
        print(f"Wrote corpus statistics to {args.output_dir}")
        return

    if tokenizer is None:
        tokenizer = lab5.make_tokenizer()

    train_languages = args.train_languages or args.languages
    unknown_languages = sorted(set(train_languages) - set(args.languages))
    if unknown_languages:
        raise ValueError(f"--train-languages not present in --languages: {unknown_languages}")

    matrix_rows = (
        read_existing_matrix(args.output_dir, experiment_signature(args))
        if args.resume
        else []
    )
    write_experiment_config(args, label_list)
    test_datasets = build_test_datasets(args, tokenizer, label2id)
    completed_languages = {
        str(row["train_language"])
        for row in matrix_rows
    }

    for language in train_languages:
        if language in completed_languages:
            print(f"Skipping completed row for {language}")
            continue

        trainer = train_one_language(
            args,
            language,
            tokenizer,
            label_list,
            label2id,
            id2label,
        )
        matrix_rows.append(evaluate_one_row(language, trainer, test_datasets))
        write_matrix(args.output_dir, args.languages, matrix_rows)
        cleanup_after_trainer(trainer)

    write_matrix(args.output_dir, args.languages, matrix_rows)
    print(f"Wrote cross-lingual matrix to {args.output_dir}")


if __name__ == "__main__":
    main()
