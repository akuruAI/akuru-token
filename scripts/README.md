# Training the sin_eng vocabulary

This directory contains the full data pipeline for training the built-in `sin_eng` BPE vocabulary. The pipeline has four steps: download -> inspect -> clean -> train.

## Prerequisites

Install the training dependencies from the project root:

```bash
pip install -e ".[train]"
```

## Step 1 - Download corpora

`data_downloader.py` fetches three Sinhala corpora and writes them to `data/` as plain UTF-8 text files. Each line is NFC-normalized (Unicode canonical decomposition followed by canonical composition) and whitespace-stripped at download time, so all subsequent steps work from a consistent encoding baseline.

```bash
python scripts/data_downloader.py
```

| Dataset | Source | Approx. size |
|---------|--------|-------------|
| CC-100 | [data.statmt.org](https://data.statmt.org/cc-100/) | ~3–4 GB |
| Wikipedia | wikimedia/wikipedia (HuggingFace) | ~100 MB |
| CulturaX | uonlp/CulturaX (HuggingFace, gated) | ~6–7 GB |

CulturaX requires accepting the dataset licence on HuggingFace and passing your token:

```bash
python scripts/data_downloader.py --hf-token hf_...
```

Datasets already present on disk are skipped automatically. To skip a dataset entirely:

```bash
python scripts/data_downloader.py --skip culturax
```

Output structure:

```
data/
├── cc100/si.txt
├── wikipedia/si.txt
└── culturax/si.txt
```

## Step 2 - Inspect raw data (optional)

`diff_sample.py` samples lines from a raw corpus and writes a unified diff showing exactly what the cleaner would do to each line - useful for understanding the data and sanity-checking the cleaning rules before committing a full run.

```bash
python scripts/diff_sample.py --dataset wikipedia
```

The output is a `.diff` file that VS Code renders natively with colour highlighting:

```
data/diff/wikipedia/sample.diff
```

```bash
# Sample 500 lines, every 200th line from cc100
python scripts/diff_sample.py --dataset cc100 --n 500 --every-n 200
```

## Step 3 - Clean corpora

`data_cleaner.py` filters and repairs each corpus file, writing cleaned output to `data/clean/`.

```bash
python scripts/data_cleaner.py
```

Each line goes through the following pipeline:

1. **Script filter** - drops lines containing characters outside Sinhala, Latin, or common Unicode ranges (e.g. Tamil, Arabic, CJK).
2. **Character fixes** - strips ZWNJ, invalid ZWJ sequences, repeated combining marks, and control characters.
3. **Syntax filter** - drops lines with invalid Sinhala combining sequences per SLS 1134:2011 (validated by `sinhala_validator.py`).
4. **Length filter** - drops lines shorter than `--min-length` characters after cleaning.

```bash
# Adjust minimum line length
python scripts/data_cleaner.py --min-length 20

# Skip a dataset
python scripts/data_cleaner.py --skip wikipedia
```

Output structure:

```
data/clean/
├── cc100/si.txt
├── wikipedia/si.txt
└── culturax/si.txt
```

## Step 4 - Train

`sin_eng_trainer.py` reads the cleaned corpora, trains a BPE vocabulary with `GraphemePreTokenizer`, and saves it to `akuru_token/vocabs/sin_eng.json`.

```bash
python scripts/sin_eng_trainer.py
```

Key options:

```bash
python scripts/sin_eng_trainer.py \
    --vocab-size 16000 \
    --min-frequency 2 \
    --progress 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-size` | `16000` | Target vocabulary size |
| `--min-frequency` | `2` | Minimum pair frequency to merge |
| `--data-dir` | `data/clean` | Directory containing cleaned corpora |
| `--output` | `akuru_token/vocabs/sin_eng.json` | Output path |
| `--force` | off | Overwrite output file if it already exists |
| `--progress` | `500` | Log a progress line every N merges |

Training logs to stdout via Python's `logging` module. A full run across all three corpora takes roughly 30–60 minutes depending on hardware.

### Guaranteed Sinhala codepoints

Before merge training begins, every valid Sinhala codepoint is injected directly into the initial vocabulary via the `guaranteed_tokens` parameter of `BPETrainer`. This covers the full set defined in [SLS 1134:2011](https://www.language.lk/download/sls1134/):

| Range | Description |
|-------|-------------|
| U+0D82–U+0D83 | Various signs (anusvara, visarga) |
| U+0D85–U+0D96 | Independent vowels |
| U+0D9A–U+0DC6 | Consonants (excluding unassigned gaps) |
| U+0DCA | Al-lakuna (virama) |
| U+0DCF–U+0DDF | Vowel signs (excluding unassigned U+0DD5, U+0DD7) |
| U+0DE6–U+0DEF | Sinhala Lith digits |
| U+0DF2–U+0DF3 | Vowel signs (cont.) |
| U+0DF4 | Kundaliya (punctuation) |
| U+111E1–U+111F4 | Sinhala archaic numbers |

Without this step, a codepoint absent from the training corpus would be missing from the initial symbol set entirely, causing `<unk>` at inference time for valid Sinhala characters that happened to be rare or absent in the data. Guaranteeing the full codepoint set upfront means the tokenizer can always represent any conforming Sinhala text, regardless of corpus coverage.

## Sinhala validator

`sinhala_validator.py` is used internally by the cleaner but can also be used standalone to inspect specific lines:

```python
from sinhala_validator import find_invalid

find_invalid("කිරිබත්")   # None  - valid
find_invalid("කිරි‍බත්")  # 4     - invalid ZWJ sequence at index 4
```