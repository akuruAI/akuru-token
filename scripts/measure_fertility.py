"""
measure_fertility.py

Trains BPE vocabularies at multiple sizes and reports:
  - Fertility (tokens per word)
  - Whole-word token count (tokens >= 4 graphemes)
  - Token length distribution (mean, median, p75, p90, p99)

Samples up to --lines lines from each corpus file in order.
Saves each trained vocab as a JSON file in --output-dir.

Usage:
    python measure_fertility.py
    python measure_fertility.py --lines 2000000
    python measure_fertility.py --data-dir /path/to/data/clean --output-dir ./vocabs_out
    python measure_fertility.py --sizes 4000 8000 12000 16000
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Iterator, List

from akuru_token import BPETokenizer, BPETrainer
from akuru_token.pretokenizer import GraphemePreTokenizer, split_graphemes
from akuru_token.vocab import Vocab


DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "clean"
DEFAULT_OUTPUT_DIR = Path("./fertility_vocabs")
DEFAULT_LINES = 2_000_000
DEFAULT_SIZES = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
DEFAULT_MIN_FREQ = 2

CORPUS_FILES = [
    "cc100/si.txt",
    "wikipedia/si.txt",
    "culturax/si.txt",
]

WHOLE_WORD_GRAPHEME_THRESHOLD = 5



def iter_corpus(data_dir: Path, max_lines: int) -> Iterator[str]:
    """
    Yield up to max_lines non-empty lines from corpus files in order.
    Skips missing files with a warning.
    """
    count = 0
    for rel in CORPUS_FILES:
        path = data_dir / rel
        if not path.exists():
            print(f"  [warning] corpus file not found, skipping: {path}")
            continue
        print(f"  reading {path} …")
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield line
                    count += 1
                    if count >= max_lines:
                        return


def load_sample(data_dir: Path, max_lines: int) -> List[str]:
    print(f"Loading up to {max_lines:,} lines from {data_dir} …")
    sample = list(iter_corpus(data_dir, max_lines))
    print(f"Loaded {len(sample):,} lines total.\n")
    return sample



def measure_fertility(tokenizer: BPETokenizer, texts: List[str]) -> float:
    total_tokens = 0
    total_words = 0
    for text in texts:
        words = tokenizer.pre_tokenizer.pre_tokenize(text)
        for word in words:
            tokens = tokenizer._bpe(word)
            total_tokens += len(tokens)
            total_words += 1
    return total_tokens / total_words if total_words else 0.0


def _grapheme_len(token: str) -> int:
    """Length of a token in grapheme clusters, ignoring the leading Ġ marker."""
    text = token.lstrip("\u0120")
    return len(split_graphemes(text)) if text else 0


def measure_whole_words(vocab: Vocab, threshold: int = WHOLE_WORD_GRAPHEME_THRESHOLD) -> int:
    return sum(
        1 for t, _ in vocab.tokens()
        if t.startswith("\u0120") and _grapheme_len(t) >= threshold
    )


def measure_length_distribution(vocab: Vocab) -> dict:
    """
    Grapheme-length stats over all merged tokens (length >= 2 graphemes).
    Single-grapheme base tokens are excluded - they are the starting alphabet,
    not the result of merges, so they would skew the distribution downward.
    """
    special = {Vocab.SPECIAL_UNK, Vocab.SPECIAL_PAD, Vocab.SPECIAL_BOS, Vocab.SPECIAL_EOS}
    lengths = []
    for token, _ in vocab.tokens():
        if token in special:
            continue
        gl = _grapheme_len(token)
        if gl >= 2:
            lengths.append(gl)

    if not lengths:
        return {}

    s = sorted(lengths)
    n = len(s)

    def pct(p: float) -> int:
        return s[min(int(n * p / 100), n - 1)]

    return {
        "count": n,
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "p75": pct(75),
        "p90": pct(90),
        "p99": pct(99),
        "max": s[-1],
    }



def print_length_distribution(ld: dict, indent: str = "    ") -> None:
    if not ld:
        print(f"{indent}(no merged tokens)")
        return
    print(f"{indent}merged tokens : {ld['count']:,}")
    print(f"{indent}mean          : {ld['mean']:.2f}")
    print(f"{indent}median        : {ld['median']:.1f}")
    print(f"{indent}p75           : {ld['p75']}")
    print(f"{indent}p90           : {ld['p90']}")
    print(f"{indent}p99           : {ld['p99']}")
    print(f"{indent}max           : {ld['max']}")


def print_summary_table(results: dict) -> None:
    print("\n" + "─" * 72)
    print(f"{'vocab':>8}  {'fertility':>9}  {'whole_words':>12}  "
          f"{'mean_gl':>8}  {'p90_gl':>7}  {'max_gl':>7}")
    print("─" * 72)
    for size, r in sorted(results.items()):
        ld = r["length_dist"]
        mean_gl = f"{ld['mean']:.2f}" if ld else "-"
        p90_gl  = str(ld["p90"]) if ld else "-"
        max_gl  = str(ld["max"]) if ld else "-"
        print(
            f"{size:>8}  {r['fertility']:>9.3f}  {r['whole_word_tokens']:>12,}  "
            f"{mean_gl:>8}  {p90_gl:>7}  {max_gl:>7}"
        )
    print("─" * 72)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fertility sweep with token length distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing cleaned corpus sub-folders.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save trained vocab JSON files.",
    )
    parser.add_argument(
        "--lines", "-n",
        type=int,
        default=DEFAULT_LINES,
        help="Maximum number of lines to load from the corpus.",
    )
    parser.add_argument(
        "--sizes", "-s",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
        metavar="N",
        help="Vocab sizes to train and evaluate.",
    )
    parser.add_argument(
        "--min-frequency", "-mf",
        type=int,
        default=DEFAULT_MIN_FREQ,
        help="Minimum pair frequency for BPE merges.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(args.data_dir, args.lines)
    if not sample:
        raise FileNotFoundError(
            f"No lines loaded from {args.data_dir}. "
            "Run data_cleaner.py first."
        )

    results = {}

    for size in sorted(args.sizes):
        print(f"\n{'═' * 50}")
        print(f"  vocab_size = {size:,}")
        print(f"{'═' * 50}")

        trainer = BPETrainer(
            vocab_size=size,
            min_frequency=args.min_frequency,
            pre_tokenizer=GraphemePreTokenizer(),
            show_progress=0,
        )
        vocab = trainer.train(iter(sample))

        # Save vocab
        out_path = args.output_dir / f"{size}.json"
        vocab.save(out_path)
        print(f"  saved -> {out_path}")

        tok = BPETokenizer(vocab)

        fertility   = measure_fertility(tok, sample)
        whole_words = measure_whole_words(vocab)
        ld          = measure_length_distribution(vocab)

        results[size] = {
            "fertility": fertility,
            "whole_word_tokens": whole_words,
            "length_dist": ld,
        }

        print(f"  fertility     : {fertility:.3f}")
        print(f"  whole_words   : {whole_words:,}")
        print("  token length distribution (graphemes):")
        print_length_distribution(ld)

    print_summary_table(results)


if __name__ == "__main__":
    main()