"""Download a small 5-language Universal Dependencies setup for Lab 5.

The default treebanks are chosen to make the transfer analysis linguistically
interesting:

- fr: French-GSD, Romance, Latin script
- es: Spanish-GSD, Romance, Latin script
- en: English-EWT, Germanic, Latin script
- ar: Arabic-PADT, Semitic, Arabic script
- zh: Chinese-GSD, Sino-Tibetan, Han script, no whitespace word segmentation

The files are normalized into the layout expected by run_lab5_experiments.py:

data/<language>/train.conllu
data/<language>/dev.conllu
data/<language>/test.conllu
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_UD_BRANCH = "master"


@dataclass(frozen=True)
class Treebank:
    language: str
    repository: str
    file_prefix: str


DEFAULT_TREEBANKS = {
    "fr": Treebank("fr", "UD_French-GSD", "fr_gsd"),
    "es": Treebank("es", "UD_Spanish-GSD", "es_gsd"),
    "en": Treebank("en", "UD_English-EWT", "en_ewt"),
    "ar": Treebank("ar", "UD_Arabic-PADT", "ar_padt"),
    "zh": Treebank("zh", "UD_Chinese-GSD", "zh_gsd"),
}

SEQUOIA_TREEBANK = Treebank("fr_sequoia", "UD_French-Sequoia", "fr_sequoia")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_TREEBANKS),
        choices=sorted(DEFAULT_TREEBANKS),
        help="Languages to download. Defaults to the recommended 5-language set.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Output directory.",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_UD_BRANCH,
        help="UniversalDependencies branch/tag to download from.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--include-sequoia-warmup",
        action="store_true",
        help="Also download French-Sequoia test data for questions 3-6.",
    )
    return parser.parse_args()


def raw_url(treebank: Treebank, split: str, branch: str) -> str:
    filename = f"{treebank.file_prefix}-ud-{split}.conllu"
    return (
        "https://raw.githubusercontent.com/"
        f"UniversalDependencies/{treebank.repository}/{branch}/{filename}"
    )


def download_file(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"skip existing {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url} -> {destination}")
    with urllib.request.urlopen(url) as response:
        content = response.read()
    destination.write_bytes(content)


def main() -> None:
    args = parse_args()
    for language in args.languages:
        treebank = DEFAULT_TREEBANKS[language]
        for split in ("train", "dev", "test"):
            download_file(
                raw_url(treebank, split, args.branch),
                args.data_dir / language / f"{split}.conllu",
                args.force,
            )

    if args.include_sequoia_warmup:
        download_file(
            raw_url(SEQUOIA_TREEBANK, "test", args.branch),
            args.data_dir / "sequoia" / "test.conllu",
            args.force,
        )


if __name__ == "__main__":
    main()
