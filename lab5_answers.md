# Lab 5 Answers

This answer includes the controlled 5 x 5 experiment results from
`outputs/lab5_controlled/cross_lingual_accuracy_matrix.csv`.

## Language Choice

I use five UD treebanks:

| Code | Treebank | Linguistic motivation |
|---|---|---|
| fr | UD_French-GSD | Romance language, Latin script, useful baseline for Sequoia-style French data |
| es | UD_Spanish-GSD | Romance language, Latin script, close to French |
| en | UD_English-EWT | Germanic language, Latin script, less morphologically rich than Romance |
| ar | UD_Arabic-PADT | Semitic language, Arabic script, richer morphology and different writing system |
| zh | UD_Chinese-GSD | Sino-Tibetan language, Han script, no whitespace word segmentation in raw text |

This set makes it possible to compare near transfer, such as French to Spanish,
with more distant transfer across language family, morphology, and script.

## 1. Consistent Annotation

Consistent annotation is necessary because the experiment compares models across
languages. If the same syntactic category were annotated differently in different
treebanks, a lower cross-lingual score could reflect annotation mismatch rather
than a real limitation of mBERT.

## 2. `yield` and `list`

`yield` makes `load_conllu` a generator: it returns one sentence at a time
instead of building the whole corpus immediately. Calling `list(...)` consumes
the generator and stores all yielded `(tokens, tags)` pairs in memory, which is
convenient for analysis and dataset construction.

## 3. Sequoia Label Distribution

The Sequoia test set contains 456 sentences and 9734 normalized tokens. Its
label distribution is highly imbalanced:

| Label | Count |
|---|---:|
| NOUN | 2125 |
| ADP | 1320 |
| DET | 1176 |
| PUNCT | 1084 |
| VERB | 781 |
| ADJ | 636 |
| PROPN | 480 |
| ADV | 417 |
| PRON | 398 |
| AUX | 345 |
| ADP+DET | 310 |
| NUM | 257 |
| CCONJ | 227 |
| SCONJ | 106 |
| X | 38 |
| SYM | 34 |

The visualization shows that common categories such as `NOUN`, `ADP`, `DET`,
and `PUNCT` dominate the corpus, while `X` and `SYM` are very rare. This
means raw accuracy can hide poor performance on rare labels.

The corresponding SVG plot is saved at
`outputs/lab5/plots/sequoia_label_distribution.svg`.

## 4. Multiword Tokens

In UD, multiword tokens are surface forms that correspond to several syntactic
words. For example, French contractions such as `au` can correspond to `à` plus
`le`, with labels `ADP` and `DET`. The surface token is represented with a range
ID such as `2-3`, followed by the syntactic words with integer IDs.

In the Sequoia test set there are 310 multiword tokens. The first examples
found by the inspection script are:

| Surface form | Syntactic words | Labels |
|---|---|---|
| des | de les | ADP+DET |
| du | de le | ADP+DET |
| Aux | À les | ADP+DET |
| au | à le | ADP+DET |
| du | de le | ADP+DET |

## 5. Why Split Multiword Tokens?

UD splits multiword tokens because syntactic annotation is defined over
syntactic words, not necessarily over orthographic tokens. This makes dependency
relations and PoS labels more comparable across languages. For this lab,
however, the model sees surface tokens, so we keep the surface token and combine
the labels, such as `ADP+DET`.

## 6. Tokens Containing Spaces

UD token forms may contain spaces. For mBERT input, these spaces should be
removed because the tokenizer receives a list of already-tokenized words through
`is_split_into_words=True`. The implementation in `lab5_multilinguality.py`
removes spaces from token forms during normalization.

In the Sequoia test set there are 13 token forms containing spaces. They are
mostly numbers, for example `500 000`, `800 000`, `80 000`, `3 862`, and
`3 852`. The implementation removes these internal spaces, yielding forms such
as `500000`.

## 7. Why Subword Tokenization?

mBERT uses subword tokenization to handle rare words, unseen words, productive
morphology, spelling variation, and many languages with one shared vocabulary.
Instead of mapping every unknown word to `[UNK]`, it can compose words from
smaller pieces.

## 8. UD vs mBERT Tokenization

For the sentence:

> Pouvez-vous donner les mêmes garanties au sein de l’Union Européenne

UD tokenization follows linguistic-word conventions, so contractions and
multiword tokens may be represented differently from the surface string. mBERT
WordPiece tokenization may further split words into subtokens, especially
accented, infrequent, or morphologically complex forms.

For the sentence, the UD-style token list is:

```text
Pouvez -vous donner les mêmes garanties à le sein de l’ Union Européenne
```

Here `Pouvez-vous` is split into `Pouvez` and `-vous`, `au` corresponds to
`à le`, and `l’Union` is split at the elided determiner. The spelling is
`Européenne` throughout.

I computed the exact mBERT tokenization with:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
print(tokenizer.tokenize("Pouvez-vous donner les mêmes garanties au sein de l’Union Européenne"))
```

The resulting mBERT WordPiece tokenization is:

```text
Po ##uve ##z - vous donner les mêmes gara ##nties au sein de l [UNK] Union Euro ##pée ##nne
```

The token `l’Union` is especially illustrative: the apostrophe character in the
input leads to an `[UNK]` token in this tokenizer output, and `Européenne` is
split into several WordPiece units.

## 9. Why Tokenization Mismatch Is a Problem

The gold PoS labels are attached to UD tokens, but the model predicts one label
per mBERT input token. If a UD token is split into several subtokens, we need a
rule for assigning labels. Otherwise the training labels and the model outputs
will have incompatible lengths.

## 10-11 and 13. Implemented Functions

The implementation is in `lab5_multilinguality.py`:

- `normalize_ud_sentence`: keeps UD multiword tokens and concatenates child
  labels. This answers question 10.
- `tokenize_and_align_labels`: aligns UD labels to mBERT subtokens.
- `build_dataset_from_conllu`: creates a HuggingFace `Dataset` with
  `input_ids`, `attention_mask`, and `labels`. These two functions answer
  question 11.
- `build_label_maps`: encodes string labels as integers. This answers question
  13.
- `IGNORE_LABEL_ID = -100`: marks padding, special tokens, and subword
  continuations for loss/evaluation masking.

All cross-lingual runs use one global `label2id` / `id2label` mapping. It is
built once from the normalized labels in the five selected languages and all
splits, then shared by every training and test dataset. This keeps label IDs
comparable across models and languages.

Core pseudocode:

```python
if token is multiword range:
    emit(surface_form, "+".join(child_upos_labels))
elif token is not empty_node:
    emit(form_without_spaces, upos)

if offset == (0, 0): label = -100      # special/padding
elif offset.start == 0: label = label2id[next_ud_label]
else: label = -100                    # continuation subtoken

mask = labels != -100
accuracy = mean(argmax(logits)[mask] == labels[mask])
```

## 12. Effect of `truncation=True`

`truncation=True` silently drops tokens beyond mBERT's maximum sequence length,
usually 512 subtokens including special tokens. Their labels are also dropped.
If truncation is frequent, evaluation may be biased toward shorter sentences or
against languages whose tokenization produces more subtokens. The controlled
experiment uses `max_length=256`, so the relevant truncation counts are:

In the selected corpora, truncation is extremely rare:

| Language | Split | Truncated / Sentences | Rate | Longest tokenized length |
|---|---|---:|---:|---:|
| fr | train | 1 / 14450 | 0.0069% | 562 |
| fr | dev | 0 / 1476 | 0% | 147 |
| fr | test | 0 / 416 | 0% | 122 |
| es | train | 0 / 14186 | 0% | 204 |
| es | dev | 0 / 1400 | 0% | 175 |
| es | test | 0 / 427 | 0% | 157 |
| en | train | 0 / 12544 | 0% | 190 |
| en | dev | 0 / 2001 | 0% | 98 |
| en | test | 1 / 2077 | 0.0481% | 334 |
| ar | train | 32 / 6075 | 0.5267% | 654 |
| ar | dev | 0 / 909 | 0% | 202 |
| ar | test | 12 / 680 | 1.7647% | 467 |
| zh | train | 0 / 3997 | 0% | 168 |
| zh | dev | 0 / 500 | 0% | 134 |
| zh | test | 0 / 500 | 0% | 155 |

At this limit, Arabic test data has the highest truncation rate, so Arabic
scores may be slightly biased toward shorter examples. Other languages are
nearly unaffected.

## 14-16. Dataset and Training

The HuggingFace Dataset contains `input_ids`, `attention_mask`, and `labels`.
Training uses `Trainer` with `compute_accuracy_from_logits`, which ignores
positions labelled `-100`. Reporting accuracy during training is important
because loss alone is hard to interpret for PoS tagging and may not reflect the
actual tagging quality on development data.

Controlled experiment configuration:

| Setting | Value |
|---|---|
| Checkpoint | `bert-base-multilingual-cased` |
| Initialization | each model starts from the same mBERT checkpoint |
| Train/dev/test caps | 1000 / 300 / 300 sentences |
| `max_length` | 256 |
| Epochs | 2 |
| Batch size | 4 |
| Learning rate | `2e-5` |
| Random seed | 42 |
| Metric | token accuracy, masking labels equal to `-100` |
| Label map | one global 138-label vocabulary shared by all languages |

## 17. Pires et al. Language Choice

Pires et al. (2019) used two main sequence-labelling settings. For NER, they
used CoNLL data for Dutch, Spanish, English, and German, plus an internal
16-language dataset: Arabic, Bengali, Czech, German, English, Spanish, French,
Hindi, Indonesian, Italian, Japanese, Korean, Portuguese, Russian, Turkish, and
Chinese. For POS tagging, they used UD data for 41 languages: Arabic,
Bulgarian, Catalan, Czech, Danish, German, Greek, English, Spanish, Estonian,
Basque, Persian, Finnish, French, Galician, Hebrew, Hindi, Croatian, Hungarian,
Indonesian, Italian, Japanese, Korean, Latvian, Marathi, Dutch, Norwegian
Bokmaal and Nynorsk, Polish, European and Brazilian Portuguese, Romanian,
Russian, Slovak, Slovenian, Swedish, Tamil, Telugu, Turkish, Urdu, and Chinese.

This gives broad coverage, but it is not a controlled linguistic sample.
Language family, script, morphology, corpus size, domain, pretraining resource
availability, and annotation quality vary simultaneously. As a result, it is
difficult to know whether strong transfer reflects genuinely multilingual
representations or easier factors such as shared script, lexical overlap,
similar word order, relatedness within Indo-European, or larger training data.

## 18. Corpus Size

Corpus sizes for the selected five treebanks:

| Language | Split | Sentences | Tokens |
|---|---|---:|---:|
| fr | train | 14450 | 344961 |
| fr | dev | 1476 | 34664 |
| fr | test | 416 | 9738 |
| es | train | 14186 | 375031 |
| es | dev | 1400 | 36461 |
| es | test | 427 | 11733 |
| en | train | 12544 | 201963 |
| en | dev | 2001 | 24788 |
| en | test | 2077 | 24740 |
| ar | train | 6075 | 191871 |
| ar | dev | 909 | 25987 |
| ar | test | 680 | 24201 |
| zh | train | 3997 | 98614 |
| zh | dev | 500 | 12665 |
| zh | test | 500 | 12010 |

It is not fully fair to compare languages with very different training sizes
because a model may perform better simply due to more supervised examples.
French and Spanish have much larger training sets than Chinese, and Arabic has
many fewer sentences but long/token-dense sentences. Corpus size can therefore
confound conclusions about multilinguality.

The corpus-size SVG plot is saved at
`outputs/lab5/plots/corpus_size_tokens.svg`.

## 19. Results Matrix

The controlled matrix is saved at:

```text
outputs/lab5_controlled/cross_lingual_accuracy_matrix.csv
```

Rows are training languages and columns are test languages.

Pilot matrix, using only 200 training sentences, 100 dev sentences, 100 test
sentences, and 1 epoch:

| Train \\ Test | fr | es | en | ar | zh |
|---|---:|---:|---:|---:|---:|
| fr | 0.265 | 0.192 | 0.154 | 0.343 | 0.280 |
| es | 0.261 | 0.319 | 0.212 | 0.342 | 0.280 |
| en | 0.206 | 0.254 | 0.180 | 0.375 | 0.347 |
| ar | 0.004 | 0.006 | 0.012 | 0.010 | 0.017 |
| zh | 0.184 | 0.195 | 0.167 | 0.332 | 0.314 |

This pilot matrix is only a technical check, not the controlled experimental result:
the training sets are tiny, the models are undertrained, and the Arabic run
triggered MPS out-of-memory warnings.

As a more controlled laptop-friendly alternative, the command `make controlled`
uses equal sentence caps across languages: 1000 train, 300 dev, 300 test,
`max_length=256`, and 2 epochs. This controls corpus size more directly than the
full-corpus setting, but any conclusions must mention that it is a capped-data
experiment.

Controlled 5 x 5 matrix:

| Train \\ Test | fr | es | en | ar | zh |
|---|---:|---:|---:|---:|---:|
| fr | 0.964 | 0.916 | 0.853 | 0.553 | 0.647 |
| es | 0.929 | 0.953 | 0.862 | 0.549 | 0.634 |
| en | 0.839 | 0.835 | 0.947 | 0.446 | 0.628 |
| ar | 0.549 | 0.568 | 0.465 | 0.901 | 0.497 |
| zh | 0.634 | 0.607 | 0.625 | 0.398 | 0.931 |

Mean in-language accuracy is 0.939, compared with 0.652 mean cross-lingual
accuracy. The best transfer directions are `es -> fr` at 0.929, `fr -> es` at
0.916, and `en -> fr` at 0.839. Arabic and Chinese both have strong
in-language performance but weaker transfer: Arabic drops from 0.901
in-language to 0.568 best transfer, while Chinese drops from 0.931 to 0.634.

## 20. Hyperparameters

Using the same hyperparameters makes the comparison more controlled, but it may
not be optimal for every language or corpus size. A fair protocol should fix the
main hyperparameters in advance, use comparable training budgets, and tune only
on development data without looking at test results.

## 21. Analysis

The actual matrix matches the expected pattern. French and Spanish are the
strongest pair: `fr -> es` reaches 0.916 and `es -> fr` reaches 0.929. Arabic is
the hardest target: averaging cross-lingual scores by test language gives `fr`
0.738, `es` 0.732, `en` 0.701, `zh` 0.602, and `ar` only 0.486. Chinese shows
moderate asymmetric transfer, suggesting that script and segmentation matter,
but interact with label distribution, tokenization, and pretraining coverage.

## References Used

- Pires, Schlinger, and Garrette (2019), *How multilingual is Multilingual
  BERT?*, ACL Anthology: https://aclanthology.org/P19-1493/
- Google Research publication page:
  https://research.google/pubs/how-multilingual-is-multilingual-bert/
