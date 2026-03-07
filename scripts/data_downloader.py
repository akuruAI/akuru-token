"""
scripts/data_downloader.py

Downloads and caches all datasets needed for sin_eng BPE training.

Datasets:
  - CC-100 (Sinhala)       : ~3-4GB
  - Sinhala Wikipedia      : ~100MB
  - CulturaX (Sinhala)     : ~6-7GB

Output directory structure:
  data/
  ├── cc100/
  │   └── si.txt
  ├── wikipedia/
  │   └── si.txt
  └── culturax/
      └── si.txt

Usage:
    python scripts/data_downloader.py
    python scripts/data_downloader.py --output-dir /path/to/data
    python scripts/data_downloader.py --skip cc100 wikipedia
"""

from __future__ import annotations

import argparse
import logging
import lzma
import unicodedata
import urllib.request
from pathlib import Path

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data"

CC100_URL = "https://data.statmt.org/cc-100/si.txt.xz"


def normalize(text: str) -> str:
    """NFC-normalize and strip leading/trailing whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def write_dataset(rows, text_field: str, out_path: Path, dataset_name: str) -> int:
    """
    Write text rows to a plain UTF-8 text file, one document per line.
    Returns the number of lines written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            text = normalize(row[text_field])
            if text:
                f.write(text + "\n")
                count += 1
            if count % 100_000 == 0 and count > 0:
                logger.info("  %s: %d lines written…", dataset_name, count)
    return count


def download_cc100(out_dir: Path) -> None:
    out_path = out_dir / "cc100" / "si.txt"
    if out_path.exists():
        logger.info("CC-100 already exists at %s - skipping.", out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    xz_path = out_path.parent / "si.txt.xz"

    logger.info("Downloading CC-100 (Sinhala) from %s...", CC100_URL)
    urllib.request.urlretrieve(CC100_URL, xz_path)
    logger.info("Download complete. Decompressing...")

    count = 0
    with (
        lzma.open(xz_path, "rt", encoding="utf-8") as src,
        out_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            text = normalize(line)
            if text:
                dst.write(text + "\n")
                count += 1
            if count % 100_000 == 0 and count > 0:
                logger.info("  CC-100: %d lines written...", count)

    xz_path.unlink()
    logger.info("CC-100 done: %d lines -> %s", count, out_path)


def download_wikipedia(out_dir: Path) -> None:
    out_path = out_dir / "wikipedia" / "si.txt"
    if out_path.exists():
        logger.info("Wikipedia already exists at %s — skipping.", out_path)
        return

    logger.info("Downloading Sinhala Wikipedia…")

    ds = load_dataset("wikimedia/wikipedia", "20231101.si", split="train")
    count = write_dataset(
        ds, text_field="text", out_path=out_path, dataset_name="Wikipedia"
    )
    logger.info("Wikipedia done: %d lines -> %s", count, out_path)


def download_culturax(out_dir: Path, token: str | None = None) -> None:
    out_path = out_dir / "culturax" / "si.txt"
    if out_path.exists():
        logger.info("CulturaX already exists at %s - skipping.", out_path)
        return

    logger.info("Downloading CulturaX (Sinhala)...")
    if not token:
        logger.warning(
            "CulturaX is a gated dataset. Without a token download might fail."
        )

    try:
        ds = load_dataset("uonlp/CulturaX", "si", split="train", token=token)
    except DatasetNotFoundError as e:
        logger.error(e)
        logger.info(
            "CulturaX is a gated dataset. Try passing your HuggingFace token via --hf-token. Skipping CulturaX"
        )
        return

    count = write_dataset(
        ds, text_field="text", out_path=out_path, dataset_name="CulturaX"
    )
    logger.info("CulturaX done: %d lines -> %s", count, out_path)


DATASETS = {
    "cc100": download_cc100,
    "wikipedia": download_wikipedia,
    "culturax": download_culturax,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sin_eng training data.")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root directory to write data into (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--hf-token",
        "-t",
        type=str,
        default=None,
        help="HuggingFace token for gated datasets (required for CulturaX).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=list(DATASETS),
        default=[],
        metavar="DATASET",
        help="Datasets to skip. Choices: cc100, wikipedia, culturax.",
    )
    args = parser.parse_args()

    to_download = {k: v for k, v in DATASETS.items() if k not in args.skip}

    if not to_download:
        logger.warning("All datasets skipped — nothing to do.")
        return

    logger.info("Output directory: %s", args.output_dir)
    logger.info("Datasets to download: %s", list(to_download))

    for name, fn in to_download.items():
        logger.info("--- %s ---", name)
        try:
            if name == "culturax":
                fn(args.output_dir, token=args.hf_token)
            else:
                fn(args.output_dir)
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)
            raise

    logger.info("All downloads complete.")
    logger.info("Data directory contents:")
    for p in sorted(args.output_dir.rglob("*.txt")):
        size_mb = p.stat().st_size / 1024 / 1024
        logger.info("  %s  (%.1f MB)", p, size_mb)


if __name__ == "__main__":
    main()
