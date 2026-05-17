# Lab 5 Multilinguality Workflow

This folder now contains a reusable implementation for the data-processing
steps of the lab:

- `lab5_report.md`: primary report-style write-up. This is the file to read
  and convert to PDF for submission.
- `lab5_report.pdf`: generated PDF version of the primary report.
- `lab5_answers.md`: question-by-question answer sheet kept as supporting
  material for checking Q1-Q21 individually.
- `lab5_answers_draft.md`: legacy copy of `lab5_answers.md` from the drafting
  phase; it is not a separate report.
- `outputs/old_reports/`: earlier numbered PDF exports, kept only as archive.
- `lab5_multilinguality.py`: CoNLL-U loading, UD normalization, mBERT label
  alignment, Dataset creation, truncation checks, and accuracy computation.
- `scripts/run_lab5_experiments.py`: command-line entry point for corpus stats,
  training, and the final `5 x 5` evaluation matrix.
- `scripts/download_ud_data.py`: helper script for downloading a recommended
  5-language UD setup.
- `test_lab5_multilinguality.py`: small unit tests for the core logic.
- `requirements.txt`: dependencies for training and testing.

## 1. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

The same workflow is also available through `make`:

```bash
make install
make data
make warmup
make stats
make test
make smoke
make full
make controlled
make snippets
make controlled-snippets
make plots
make controlled-plots
make check-report
make report-pdf
```

## 2. Choose UD Languages

Pick 5 Universal Dependencies treebanks and keep the train/dev/test files for
each language. A convenient folder layout is:

```text
data/
  fr/
    train.conllu
    dev.conllu
    test.conllu
  es/
    train.conllu
    dev.conllu
    test.conllu
```

The lab asks you to motivate the choice linguistically. A good set usually
varies along language family, morphology, and script.

The default choice in this project is:

```text
fr = UD_French-GSD
es = UD_Spanish-GSD
en = UD_English-EWT
ar = UD_Arabic-PADT
zh = UD_Chinese-GSD
```

You can download these treebanks with:

```bash
python3 scripts/download_ud_data.py
```

The lab warm-up questions mention the Sequoia test set specifically. Download it
as well with:

```bash
python3 scripts/download_ud_data.py --include-sequoia-warmup
python3 scripts/inspect_ud_warmup.py data/sequoia/test.conllu
```

## 3. Inspect a Corpus

```python
import lab5_multilinguality as lab5

train = lab5.load_normalized_conllu("data/fr/train.conllu")
dev = lab5.load_normalized_conllu("data/fr/dev.conllu")
test = lab5.load_normalized_conllu("data/fr/test.conllu")

print(lab5.count_sentences_and_tokens(train))
print(lab5.label_distribution(test))
```

This answers the corpus-size and label-distribution questions.

## 4. Build Label Maps

For fair cross-lingual evaluation, build one shared label inventory from all
train/dev/test corpora used in the 5-language experiment.

```python
all_corpora = [train, dev, test]  # extend with the other 4 languages
label_list, label2id, id2label = lab5.build_label_maps(all_corpora)
```

## 5. Create HuggingFace Datasets

```python
tokenizer = lab5.make_tokenizer()

train_ds = lab5.build_dataset_from_conllu("data/fr/train.conllu", tokenizer, label2id)
dev_ds = lab5.build_dataset_from_conllu("data/fr/dev.conllu", tokenizer, label2id)
test_ds = lab5.build_dataset_from_conllu("data/fr/test.conllu", tokenizer, label2id)

print(lab5.truncation_report(tokenizer, train))
```

The truncation report helps answer whether `truncation=True` may bias the
evaluation.

## 6. Trainer Metric

Use the provided metric function in the HuggingFace Trainer:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=lab5.compute_accuracy_from_logits,
)
```

The metric ignores labels equal to `-100`, so padding, special tokens, and
subword continuations are not counted.

## 7. Cross-Lingual Matrix

For each of the 5 languages:

1. Train one model on that language's train set.
2. Select/check it using that language's dev set.
3. Evaluate the trained model on all 5 test sets.
4. Write a `5 x 5` matrix where rows are training languages and columns are test
   languages.

Use the same hyperparameters for comparability, but discuss whether this is
fully fair when corpus sizes differ substantially.

You can run the whole workflow from the command line once the data is in place.
First check corpus sizes and truncation:

```bash
python3 scripts/run_lab5_experiments.py \
  --languages fr es en ar zh \
  --data-dir data \
  --output-dir outputs/lab5 \
  --stats-only
```

If you have not installed `transformers` yet, or only want sentence/token counts
and label distributions, add `--skip-truncation`.

Then run the full training/evaluation matrix:

```bash
python3 scripts/run_lab5_experiments.py \
  --languages fr es en ar zh \
  --data-dir data \
  --output-dir outputs/lab5
```

Before launching the full run, it is useful to do a tiny end-to-end smoke test:

```bash
python3 scripts/run_lab5_experiments.py \
  --languages fr es en ar zh \
  --train-languages fr \
  --data-dir data \
  --output-dir outputs/lab5_smoke \
  --max-train-samples 8 \
  --max-dev-samples 8 \
  --max-test-samples 8 \
  --epochs 1 \
  --batch-size 2
```

This does not produce meaningful accuracy, but it checks that the tokenizer,
Dataset conversion, Trainer, metrics, and matrix writing all work.

On Apple Silicon, long Arabic batches may trigger MPS memory warnings. For a
more conservative run, lower the batch size, cap sequence length, or force CPU:

```bash
python3 scripts/run_lab5_experiments.py \
  --languages fr es en ar zh \
  --data-dir data \
  --output-dir outputs/lab5 \
  --epochs 3 \
  --batch-size 4 \
  --max-length 256 \
  --use-cpu \
  --resume
```

For a faster controlled experiment with equal sentence caps across languages:

```bash
make controlled
```

This uses 1000 train, 300 dev, and 300 test sentences per language, `max_length
= 256`, 2 epochs, and resume-safe matrix writing.

You can also run the controlled experiment language by language:

```bash
make controlled-fr
make controlled-es
make controlled-en
make controlled-zh
make controlled-ar
```

The Arabic target uses CPU by default, because Arabic batches were the ones most
likely to trigger MPS memory warnings during pilot runs. All controlled targets
write to the same `outputs/lab5_controlled` matrix and use `--resume`.

The script writes:

- `outputs/lab5/corpus_stats.csv`
- `outputs/lab5/label_distribution_test.json`
- `outputs/lab5/truncation_report.json`
- `outputs/lab5/cross_lingual_accuracy_matrix.csv`
- `outputs/lab5/cross_lingual_accuracy_matrix.json`

The controlled run writes the same files under `outputs/lab5_controlled/`.

Format a completed matrix for the report with:

```bash
python3 scripts/format_matrix.py outputs/lab5/cross_lingual_accuracy_matrix.csv
python3 scripts/format_matrix.py outputs/lab5_controlled/cross_lingual_accuracy_matrix.csv
```

Generate a fuller report snippet, including the saved experiment configuration
and automatic transfer notes, with:

```bash
python3 scripts/report_snippets.py --output-dir outputs/lab5
python3 scripts/report_snippets.py --output-dir outputs/lab5_controlled
```

Generate SVG plots for the warm-up label distribution, corpus sizes, and the
matrix heatmap after training:

```bash
python3 scripts/plot_lab5_outputs.py --output-dir outputs/lab5
python3 scripts/plot_lab5_outputs.py --output-dir outputs/lab5_controlled \
  --warmup-dir outputs/lab5/warmup \
  --plots-dir outputs/lab5_controlled/plots
```
