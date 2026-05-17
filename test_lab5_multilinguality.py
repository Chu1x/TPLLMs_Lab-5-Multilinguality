from pathlib import Path

import lab5_multilinguality as lab5


def test_normalize_ud_sentence_merges_multiword_tokens_and_removes_spaces():
    sentence = [
        lab5.ConlluToken("1", "Revenons", "VERB"),
        lab5.ConlluToken("2-3", "aux", "_"),
        lab5.ConlluToken("2", "à", "ADP"),
        lab5.ConlluToken("3", "les", "DET"),
        lab5.ConlluToken("4", "choses", "NOUN"),
        lab5.ConlluToken("5", "essentiel les", "ADJ"),
        lab5.ConlluToken("6", ".", "PUNCT"),
    ]

    words, labels = lab5.normalize_ud_sentence(sentence)

    assert words == ["Revenons", "aux", "choses", "essentielles", "."]
    assert labels == ["VERB", "ADP+DET", "NOUN", "ADJ", "PUNCT"]


def test_load_normalized_conllu(tmp_path: Path):
    conllu_file = tmp_path / "toy.conllu"
    conllu_file.write_text(
        "\n".join(
            [
                "# sent_id = toy",
                "1\tJe\tje\tPRON\t_\t_\t0\troot\t_\t_",
                "2-3\tau\t_\t_\t_\t_\t_\t_\t_\t_",
                "2\tà\tà\tADP\t_\t_\t1\tcase\t_\t_",
                "3\tle\tle\tDET\t_\t_\t4\tdet\t_\t_",
                "4\tcinéma\tcinéma\tNOUN\t_\t_\t1\tobl\t_\t_",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert lab5.load_normalized_conllu(conllu_file) == [
        (["Je", "au", "cinéma"], ["PRON", "ADP+DET", "NOUN"])
    ]


def test_build_label_maps_adds_pad_last():
    corpus = [(["a", "b"], ["NOUN", "VERB"]), (["c"], ["ADP+DET"])]
    other_language_corpus = [(["x"], ["PROPN"])]

    label_list, label2id, id2label = lab5.build_label_maps(
        [corpus, other_language_corpus]
    )

    assert label_list[-1] == lab5.PAD_LABEL
    assert id2label[label2id["NOUN"]] == "NOUN"
    assert id2label[label2id["PROPN"]] == "PROPN"


def test_count_sentences_and_tokens():
    corpus = [(["a", "b"], ["X", "Y"]), (["c"], ["Z"])]

    assert lab5.count_sentences_and_tokens(corpus) == (2, 3)


def test_compute_accuracy_from_logits_ignores_minus_100():
    import numpy as np

    logits = np.array(
        [
            [
                [0.1, 0.9],
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        ]
    )
    labels = np.array([[1, lab5.IGNORE_LABEL_ID, 0]])

    assert lab5.compute_accuracy_from_logits((logits, labels)) == {"accuracy": 0.5}


def test_tokenize_and_align_labels_uses_first_subtoken_only():
    class FakeTokenizer:
        def __call__(self, sentences, **kwargs):
            assert kwargs["is_split_into_words"] is True
            assert kwargs["return_offsets_mapping"] is True
            assert kwargs["max_length"] == 8
            return {
                "input_ids": [[101, 10, 11, 12, 102, 0]],
                "attention_mask": [[1, 1, 1, 1, 1, 0]],
                "offset_mapping": [[(0, 0), (0, 3), (3, 5), (0, 1), (0, 0), (0, 0)]],
            }

    encoded = lab5.tokenize_and_align_labels(
        FakeTokenizer(),
        [["first", "."]],
        [["NOUN", "PUNCT"]],
        {"NOUN": 0, "PUNCT": 1},
        max_length=8,
    )

    assert encoded["labels"] == [[-100, 0, -100, 1, -100, -100]]
