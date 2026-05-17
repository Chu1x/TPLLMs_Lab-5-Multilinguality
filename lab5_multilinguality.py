"""Utilities for Lab 5: multilingual PoS tagging with mBERT.

The functions in this file cover the parts that are easiest to get subtly
wrong: CoNLL-U reading, UD multiword-token normalization, mBERT subtoken/label
alignment, HuggingFace Dataset creation, and accuracy computation with -100
ignored labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

IGNORE_LABEL_ID = -100
PAD_LABEL = "<pad>"
MODEL_CHECKPOINT = "bert-base-multilingual-cased"


@dataclass(frozen=True)
class ConlluToken:
    """A single line from a CoNLL-U sentence."""

    token_id: str
    form: str
    upos: str


def _remove_token_spaces(form: str) -> str:
    """UD allows spaces inside tokens; mBERT input words should not keep them."""

    return form.replace(" ", "")


def _parse_conllu_sentence(block: str) -> list[ConlluToken]:
    """Parse one CoNLL-U sentence block.

    We only keep the fields needed for this lab. Empty nodes such as ``3.1`` are
    kept in the raw parse but ignored later during normalization.
    """

    tokens: list[ConlluToken] = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")
        if len(fields) != 10:
            raise ValueError(f"Invalid CoNLL-U line with {len(fields)} fields: {line}")

        token_id, form, upos = fields[0], fields[1], fields[3]
        tokens.append(ConlluToken(token_id=token_id, form=form, upos=upos))
    return tokens


def load_conllu(filename: str | Path) -> list[list[ConlluToken]]:
    """Load a CoNLL-U file as raw sentence/token records."""

    text = Path(filename).read_text(encoding="utf-8")
    return [
        _parse_conllu_sentence(block)
        for block in text.split("\n\n")
        if block.strip()
    ]


def normalize_ud_sentence(sentence: list[ConlluToken]) -> tuple[list[str], list[str]]:
    """Apply the lab's UD-side normalization.

    Multiword tokens are kept as one surface token and receive a concatenated
    label, e.g. ``au`` -> ``ADP+DET``. Their syntactic child tokens are skipped.
    Empty nodes with decimal IDs are ignored. Spaces inside token forms are
    removed.
    """

    words: list[str] = []
    labels: list[str] = []
    index = 0

    while index < len(sentence):
        token = sentence[index]

        if "." in token.token_id:
            index += 1
            continue

        if "-" in token.token_id:
            start_text, end_text = token.token_id.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            child_labels: list[str] = []
            index += 1

            while index < len(sentence):
                child = sentence[index]
                if "." in child.token_id or "-" in child.token_id:
                    break

                child_id = int(child.token_id)
                if child_id < start:
                    index += 1
                    continue
                if child_id > end:
                    break

                child_labels.append(child.upos)
                index += 1

            if not child_labels:
                raise ValueError(f"Multiword token {token.form!r} has no child labels")

            words.append(_remove_token_spaces(token.form))
            labels.append("+".join(child_labels))
            continue

        words.append(_remove_token_spaces(token.form))
        labels.append(token.upos)
        index += 1

    return words, labels


def load_normalized_conllu(filename: str | Path) -> list[tuple[list[str], list[str]]]:
    """Load a CoNLL-U file directly into normalized ``(words, labels)`` pairs."""

    return [normalize_ud_sentence(sentence) for sentence in load_conllu(filename)]


def label_distribution(corpus: Iterable[tuple[list[str], list[str]]]) -> Counter[str]:
    """Count PoS labels in a normalized corpus."""

    counts: Counter[str] = Counter()
    for _words, labels in corpus:
        counts.update(labels)
    return counts


def build_label_maps(
    corpora: Iterable[Iterable[tuple[list[str], list[str]]]],
    include_pad: bool = True,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Create stable label/id mappings from one or more corpora."""

    label_set: set[str] = set()
    for corpus in corpora:
        for _words, labels in corpus:
            label_set.update(labels)

    label_list = sorted(label_set)
    if include_pad and PAD_LABEL not in label_list:
        label_list.append(PAD_LABEL)

    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def tokenize_and_align_labels(
    tokenizer: Any,
    sentences: list[list[str]],
    labels: list[list[str]],
    label2id: dict[str, int],
    padding: bool | str = True,
    truncation: bool = True,
    max_length: int | None = None,
) -> dict[str, list[list[int]]]:
    """Tokenize UD words with mBERT and align one label per resulting token.

    The first subtoken of each original word receives the word's label. Special
    tokens, padding tokens, and continuation subtokens receive ``-100`` so both
    loss and evaluation can ignore them.
    """

    encoded = tokenizer(
        sentences,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )

    aligned_labels: list[list[int]] = []
    offset_mappings = encoded.pop("offset_mapping")

    for sentence_labels, offsets in zip(labels, offset_mappings):
        label_index = 0
        current_labels: list[int] = []

        for start, end in offsets:
            if start == 0 and end == 0:
                current_labels.append(IGNORE_LABEL_ID)
            elif start == 0:
                if label_index >= len(sentence_labels):
                    current_labels.append(IGNORE_LABEL_ID)
                else:
                    current_labels.append(label2id[sentence_labels[label_index]])
                    label_index += 1
            else:
                current_labels.append(IGNORE_LABEL_ID)

        aligned_labels.append(current_labels)

    encoded["labels"] = aligned_labels
    return dict(encoded)


def build_dataset_from_conllu(
    filename: str | Path,
    tokenizer: Any,
    label2id: dict[str, int],
    max_length: int | None = None,
):
    """Create a HuggingFace Dataset from one CoNLL-U file."""

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "Install datasets first, e.g. `python3 -m pip install datasets`."
        ) from exc

    corpus = load_normalized_conllu(filename)
    sentences = [words for words, _labels in corpus]
    labels = [sentence_labels for _words, sentence_labels in corpus]
    encoded = tokenize_and_align_labels(
        tokenizer,
        sentences,
        labels,
        label2id,
        max_length=max_length,
    )

    rows = [
        {key: encoded[key][idx] for key in encoded}
        for idx in range(len(sentences))
    ]
    return Dataset.from_list(rows)


def count_sentences_and_tokens(corpus: Iterable[tuple[list[str], list[str]]]) -> tuple[int, int]:
    """Return sentence and normalized-token counts for a corpus."""

    sentence_count = 0
    token_count = 0
    for words, _labels in corpus:
        sentence_count += 1
        token_count += len(words)
    return sentence_count, token_count


def truncation_report(
    tokenizer: Any,
    corpus: Iterable[tuple[list[str], list[str]]],
    max_length: int | None = None,
) -> dict[str, float | int]:
    """Estimate how often tokenized sentences exceed mBERT's max length."""

    limit = max_length or getattr(tokenizer, "model_max_length", 512)
    total = 0
    truncated = 0
    longest = 0

    for words, _labels in corpus:
        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=False,
        )
        length = len(tokenized["input_ids"])
        total += 1
        longest = max(longest, length)
        truncated += int(length > limit)

    return {
        "sentences": total,
        "truncated": truncated,
        "truncation_rate": truncated / total if total else 0.0,
        "longest_tokenized_length": longest,
        "max_length": limit,
    }


def compute_accuracy_from_logits(eval_pred: Any) -> dict[str, float]:
    """Metric function suitable for HuggingFace ``Trainer(compute_metrics=...)``."""

    import numpy as np

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != IGNORE_LABEL_ID

    if not mask.any():
        return {"accuracy": 0.0}

    correct = (predictions == labels) & mask
    return {"accuracy": float(correct.sum() / mask.sum())}


def make_tokenizer():
    """Load the mBERT tokenizer lazily."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install transformers first, e.g. `python3 -m pip install transformers`."
        ) from exc

    return AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def make_model(num_labels: int, id2label: dict[int, str], label2id: dict[str, int]):
    """Load an mBERT token-classification model lazily."""

    try:
        from transformers import AutoModelForTokenClassification
    except ImportError as exc:
        raise ImportError(
            "Install transformers first, e.g. `python3 -m pip install transformers`."
        ) from exc

    return AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
