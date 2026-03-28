"""
scripts/diff_sample.py

Samples lines from a raw corpus file and writes a unified diff file
after applying the same cleaning as in `data_cleaner.py`. This diff
can be used to visualize the cleanup process.

Lines are dropped if they contain a foreign script, invalid Sinhala syntax,
or are too short after cleaning. Lines containing Unicode paragraph/line
separators (U+2028, U+2029) are split into fragments; short fragments are
dropped, and each surviving fragment is shown as a separate addition in the
diff so the paragraph break is visible as a natural split.

Usage:
    python scripts/diff_sample.py
    python scripts/diff_sample.py --dataset wikipedia
    python scripts/diff_sample.py --n 200 --every-n 500
    python scripts/diff_sample.py --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from data_cleaner import line_is_allowed, fix_line
from sinhala_validator import find_invalid

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_N = 300
DEFAULT_MIN_LENGTH = 10


def build_samples(
    in_path: Path,
    n: int,
    min_length: int,
    every_n: int,
) -> tuple[list[str], list[str], dict]:
    stats = {"total_sampled": 0, "changed": 0, "dropped": 0, "kept": 0}
    before_lines = []
    after_lines = []

    with in_path.open(encoding="utf-8") as f:
        for i, raw_line in enumerate(f):
            if stats["total_sampled"] >= n:
                break
            if i % every_n != 0:
                continue

            line = raw_line.rstrip("\n")
            stats["total_sampled"] += 1

            if not line_is_allowed(line):
                before_lines.append(line)
                after_lines.append("[DROPPED — foreign script]")
                stats["dropped"] += 1
                continue

            fixed = fix_line(line)

            invalid_idx = find_invalid(fixed)
            if invalid_idx is not None:
                before_lines.append(line)
                start = max(0, invalid_idx-6)
                end = min(len(fixed), invalid_idx+6)
                after_lines.append(
                    f"[DROPPED — invalid Sinhala syntax at {invalid_idx} after fixing. Error is around ({fixed[start:end]})]"
                )
                stats["dropped"] += 1
                continue

            # Split on Unicode paragraph/line separators and real newlines.
            # Each fragment is checked independently against min_length.
            fragments = (
                fixed.rstrip("\n")
                .replace("\u2028", "\n")
                .replace("\u2029", "\n")
                .split("\n")
            )
            fragments = [f for f in fragments if len(f) >= min_length]

            if not fragments:
                before_lines.append(line)
                after_lines.append("[DROPPED — too short after cleaning]")
                stats["dropped"] += 1
                continue

            before_lines.append(line)
            after_lines.append(fragments[0])

            for fragment in fragments[1:]:
                before_lines.append("")
                after_lines.append(fragment)

            after = fragments[0] if len(fragments) == 1 else None
            if after != line:
                stats["changed"] += 1
            else:
                stats["kept"] += 1
    print(before_lines, after_lines, stats)
    return before_lines, after_lines, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a unified diff file of daa cleaning sample for VS Code rendering."
    )
    parser.add_argument("--data-dir", "-d", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output base directory (default: <data-dir>/diff)",
    )
    parser.add_argument(
        "--dataset",
        "-ds",
        choices=["cc100", "wikipedia", "culturax"],
        default="cc100",
    )
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of lines to sample (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--every-n",
        "-en",
        type=int,
        default=1000,
        help="Sample every N-th line (default: 1000)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help=f"Minimum line length after cleaning (default: {DEFAULT_MIN_LENGTH})",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.data_dir / "diff"

    in_path = args.data_dir / args.dataset / "si.txt"
    if not in_path.exists():
        print(f"File not found: {in_path}")
        print("Run scripts/download_data.py first.")
        sys.exit(1)

    out_path = args.output_dir / args.dataset / "sample.diff"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size_mb = in_path.stat().st_size / 1024 / 1024
    print(f"Input : {in_path}  ({size_mb:.1f} MB)")
    print(f"Sampling every {args.every_n}th line, target {args.n} samples")

    before_lines, after_lines, stats = build_samples(
        in_path, args.n, args.min_length, args.every_n
    )

    # Generate unified diff
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile="before (raw)",
        tofile="after (cleaned)",
        lineterm="",
    )
    print(diff)

    out_path.write_text("\n".join(diff), encoding="utf-8")

    print(f"\nSampled : {stats['total_sampled']} lines")
    print(f"  OK      : {stats['kept']}")
    print(f"  Changed : {stats['changed']}")
    print(f"  Dropped : {stats['dropped']}")
    print(f"\nDiff file: {out_path}")
    print("Open in VS Code — it renders .diff files natively.")


if __name__ == "__main__":
    main()
