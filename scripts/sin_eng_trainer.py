"""
scripts/sin_eng_trainer.py

Trains the sin_eng BPE vocabulary from downloaded Sinhala corpora and saves
it to akuru_token/vocabs/sin_eng.json.

Usage:
    python scripts/train_sin_eng.py
    python scripts/train_sin_eng.py --data-dir /path/to/data
    python scripts/train_sin_eng.py --vocab-size 16000 --min-frequency 2
    python scripts/train_sin_eng.py --output /path/to/custom/vocab.json
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterator

from akuru_token import BPETrainer
from akuru_token.pretokenizer import GraphemePreTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "clean"
DEFAULT_VOCAB_OUT = (
    Path(__file__).parent.parent / "akuru_token" / "vocabs" / "sin_eng.json"
)

# Datasets to include, in order of inclusion
CORPUS_FILES = [
    "cc100/si.txt",
    "wikipedia/si.txt",
    "culturax/si.txt",
]

SINHALA_CODEPOINTS = sorted(
    {chr(0x200D)} # zero width joiner
    | set(chr(cp) for cp in range(0x0D82, 0x0D84))  # Various signs 0D82-0D83
    | set(chr(cp) for cp in range(0x0D85, 0x0D97))  # Independent vowels  0D85-0D96
    | (
        set(chr(cp) for cp in range(0x0D9A, 0x0DC7))
        - {chr(0x0DB2), chr(0x0DBC), chr(0x0DBE), chr(0x0DBF)}  # Consonants 0D9A-0DC6
    )
    | {chr(0x0DCA)}  # Al-lakuna 0DCA
    | (
        set(chr(cp) for cp in range(0x0DCF, 0x0DE0)) - {chr(0x0DD5), chr(0x0DD7)}
    )  # Vowel signs 0DCF-0DDF
    | set(chr(cp) for cp in range(0x0DE6, 0x0DF0))  # Sinhala Lith digits 0DE6-0DEF
    | set(chr(cp) for cp in range(0x0DF2, 0x0DF4))  # Vowel signs (cont.) 0DF2-0DF3
    | {chr(0x0DF4)}  # Kundaliya 0DF4
    | set(chr(cp) for cp in range(0x111E1, 0x111F5))  # Archaic numbers 111E1-111F4
)


def iter_corpus(data_dir: Path) -> Iterator[str]:
    """
    Yield lines from all corpus files in order.
    Skips missing files with a warning so training can proceed with
    whatever data is available.
    """
    for rel_path in CORPUS_FILES:
        path = data_dir / rel_path
        if not path.exists():
            logger.warning("Corpus file not found, skipping: %s", path)
            continue
        logger.info("Reading %s...", path)
        with path.open(encoding="utf-8") as f:
            yield from f


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sin_eng BPE vocabulary.")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing downloaded corpus files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_VOCAB_OUT,
        help=f"Output path for the vocab JSON (default: {DEFAULT_VOCAB_OUT})",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    parser.add_argument(
        "--vocab-size",
        "-vs",
        type=int,
        default=12_000,
        help="Target vocabulary size (default: 16000)",
    )
    parser.add_argument(
        "--min-frequency",
        "-mf",
        type=int,
        default=2,
        help="Minimum pair frequency to merge (default: 2)",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=500,
        help="Log a progress message every N merges (default: 500, 0 = silent)",
    )
    args = parser.parse_args()

    # Verify at least one corpus file exists before starting
    available = [
        args.data_dir / p for p in CORPUS_FILES if (args.data_dir / p).exists()
    ]
    if not available:
        raise FileNotFoundError(
            f"No corpus files found in {args.data_dir}. "
            f"Run scripts/download_data.py first."
        )
    logger.info("Corpus files found: %s", [str(p) for p in available])

    # Abort if output exists unless explicitly allowed to overwrite
    if args.output.exists():
        if not args.force:
            raise FileExistsError(
                f"Output file already exists: {args.output}. "
                "Use --force to overwrite."
            )
        logger.warning("Overwriting existing output file: %s", args.output)
        args.output.unlink()

    logger.info(
        "Initialising trainer: vocab_size=%d, min_frequency=%d",
        args.vocab_size,
        args.min_frequency,
    )

    trainer = BPETrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        pre_tokenizer=GraphemePreTokenizer(),
        show_progress=args.progress,
        guaranteed_tokens=SINHALA_CODEPOINTS,
    )

    logger.info("Starting training...")
    t0 = time.time()
    vocab = trainer.train(iter_corpus(args.data_dir))
    elapsed = time.time() - t0

    logger.info(
        "Training complete in %.1fs. Vocab size: %d, Merges: %d",
        elapsed,
        len(vocab),
        len(vocab.merges),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(args.output)
    logger.info("Vocab saved to %s", args.output)

    vocab.describe()


if __name__ == "__main__":
    main()
