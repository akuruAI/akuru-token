"""
scripts/data_cleaner.py

Cleans downloaded corpus files before BPE training.

Cleaning pipeline (per line):
  1. Line-level filter  - drop lines containing characters outside the
                          allowed script ranges (Sinhala, Latin, Common).
  2. Character-level fixes - applied to lines that pass the filter:
       - Strip malformed ZWJ sequences (ZWJ not preceded by virama)
       - Collapse repeated combining marks (vowel signs, viramas)
       - Strip null bytes and non-printable control characters
  3. Drop syntactically incorrect lines even after the fix
  4. Post-fix filter   - drop lines that are empty or too short after fixing.

Note: This process does not support touching letters and they will be dropped
by either step 2 or 3. The pretokenizer grapheme splitter does not support
touching letters either.

Allowed Unicode ranges
----------------------
  Sinhala          U+0D80  - U+0DFF
  Latin + Extended U+0000  - U+024F   (English, accented Latin)
  Common           U+0250  - U+02FF   (IPA, spacing modifiers)
                   U+2000  - U+206F   (general punctuation)
                   U+20A0  - U+20CF   (currency symbols)
  Whitespace       (space, tab, newline - always allowed)

Input/output
------------
  Reads from <data-dir>/<dataset>/si.txt
  Writes to <data-dir>/clean/<dataset>/si.txt

Usage:
    python scripts/data_cleaner.py
    python scripts/data_cleaner.py --data-dir /path/to/data
    python scripts/data_cleaner.py --min-length 10
    python scripts/data_cleaner.py --skip wikipedia
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from sinhala_validator import find_invalid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_MIN_LENGTH = 10  # min line length after cleaning

# Datasets to clean, must match download_data.py output structure
DATASETS = ["cc100", "wikipedia", "culturax"]


def _in_allowed_range(cp: int) -> bool:
    """
    Return True if codepoint cp is in an allowed Unicode range.

    Allowed ranges:
      U+0000-U+024F   Latin, Basic Latin, Latin Extended A/B
      U+0250-U+02FF   IPA extensions, spacing modifier letters
      U+2000-U+206F   General punctuation
      U+20A0-U+20CF   Currency symbols
      U+0D80-U+0DFF   Sinhala
    Whitespace is always allowed (checked separately).
    """
    return (
        0x0000 <= cp <= 0x024F  # Latin + extensions
        or 0x0250 <= cp <= 0x02FF  # IPA + spacing modifiers
        or 0x2000 <= cp <= 0x206F  # General punctuation
        or 0x20A0 <= cp <= 0x20CF  # Currency symbols
        or 0x0D80 <= cp <= 0x0DFF  # Sinhala
    )


def line_is_allowed(line: str) -> bool:
    """
    Return True if every non-whitespace character in the line is within
    the allowed Unicode ranges.
    """
    for ch in line:
        if ch.isspace():
            continue
        if not _in_allowed_range(ord(ch)):
            return False
    return True


# Sinhala vowel signs: U+0DCF-U+0DDF  (also U+0DD8, U+0DF2, U+0DF3)
_SINHALA_VOWEL_SIGN = re.compile(r"([\u0dcf-\u0ddf\u0df2\u0df3])\1+")

# Al lakuna: U+0DCA
_AL_LAKUNA = re.compile(r"\u0dca\u0dca+")

# ZWJ not preceded by virama - strip the ZWJ
# Lookbehind: not preceded by U+0DCA
_INVALID_ZWJ = re.compile(r"(?<!\u0dca)\u200d")

# Multiple consecutive ZWJs
_MULTI_ZWJ = re.compile(r"\u200d{2,}")

# Control characters (except tab, newline, carriage return)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def fix_line(line: str) -> str:
    """
    Apply character-level fixes to a line that has passed the script filter.

      1. Strip control characters
      2. Strip multiple consecutive ZWJs
      3. Strip ZWJ not preceded by virama
      4. Collapse repeated Sinhala vowel signs
      5. Collapse repeated Sinhala viramas
      6. Strip leading/trailing whitespace
    """
    line = _CONTROL_CHARS.sub("", line)
    line = _MULTI_ZWJ.sub("\u200d", line)
    line = _INVALID_ZWJ.sub("", line)
    line = _SINHALA_VOWEL_SIGN.sub(r"\1", line)
    line = _AL_LAKUNA.sub("\u0dca", line)
    return line.strip()


def clean_file(in_path: Path, out_path: Path, min_length: int) -> dict:
    """
    Clean a single corpus file. Returns a stats dict.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "dropped_script": 0,
        "dropped_short": 0,
        "dropped_syntax": 0,
        "kept": 0,
    }

    with (
        in_path.open(encoding="utf-8") as src,
        out_path.open("w", encoding="utf-8") as dst,
    ):
        for row in src:
            # U+2028 (Line Separator) and U+2029 (Paragraph Separator)
            # are replaced here, before the line is split.
            row = row.rstrip("\n").replace("\u2028", "\n").replace("\u2029", "\n")
            for line in row.split("\n"):
                stats["total"] += 1

                # 1. Script filter
                if not line_is_allowed(line):
                    stats["dropped_script"] += 1
                    continue

                # 2. Character-level fixes
                line = fix_line(line)
                
                # 3. Syntax filter
                invalid_idx = find_invalid(line)
                if invalid_idx:
                    stats["dropped_syntax"] += 1
                    continue

                # 4. Post-fix length filter
                if len(line) < min_length:
                    stats["dropped_short"] += 1
                    continue

                dst.write(line + "\n")
                stats["kept"] += 1

                if stats["total"] % 500_000 == 0:
                    logger.info(
                        "  %s: processed %d lines, kept %d (%.1f%%)",
                        in_path.stem,
                        stats["total"],
                        stats["kept"],
                        100 * stats["kept"] / stats["total"],
                    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean sin_eng corpus files.")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--min-length",
        "-ml",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help=f"Minimum line length after cleaning (default: {DEFAULT_MIN_LENGTH})",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=DATASETS,
        default=[],
        metavar="DATASET",
        help="Datasets to skip. Choices: cc100, wikipedia, culturax.",
    )
    args = parser.parse_args()

    to_clean = [d for d in DATASETS if d not in args.skip]

    total_stats = {
        "total": 0,
        "dropped_script": 0,
        "dropped_short": 0,
        "kept": 0,
        "dropped_syntax": 0,
    }

    for dataset in to_clean:
        in_path = args.data_dir / dataset / "si.txt"
        out_path = args.data_dir / "clean" / dataset / "si.txt"

        if not in_path.exists():
            logger.warning("Input file not found, skipping: %s", in_path)
            continue

        logger.info("--- Cleaning %s ---", dataset)
        logger.info("  Input:  %s", in_path)
        logger.info("  Output: %s", out_path)

        stats = clean_file(in_path, out_path, args.min_length)

        drop_pct = (
            100
            * (
                stats["dropped_script"]
                + stats["dropped_short"]
                + total_stats["dropped_syntax"]
            )
            / max(stats["total"], 1)
        )
        logger.info(
            "  Done: %d total, %d kept (%.1f%% dropped) "
            "[script: %d, short: %d, syntax: %d]",
            stats["total"],
            stats["kept"],
            drop_pct,
            stats["dropped_script"],
            stats["dropped_short"],
            stats["dropped_syntax"],
        )

        out_size = out_path.stat().st_size / 1024 / 1024
        logger.info("  Output size: %.1f MB", out_size)

        for k in total_stats:
            total_stats[k] += stats[k]

    logger.info("-" * 40)
    logger.info(
        "Total: %d lines processed, %d kept (%.1f%% dropped)",
        total_stats["total"],
        total_stats["kept"],
        100
        * (
            total_stats["dropped_script"]
            + total_stats["dropped_short"]
            + total_stats["dropped_syntax"]
        )
        / max(total_stats["total"], 1),
    )


if __name__ == "__main__":
    main()
