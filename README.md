# akuru-token

A grapheme-aware BPE tokenizer library with support for Sinhala script and mixed Sinhala–English text.

## About

akuru-token is a small, focused tokenizer library for Sinhala and mixed Sinhala–English text. It prioritizes grapheme clusters over raw codepoints so the base units match how Sinhala is read and written.

### Why grapheme-aware?

Most tokenizers split text into Unicode codepoints before applying BPE. In Sinhala, this shreds syllables: `කිරිබත්` becomes `['ක', 'ි', 'ර', 'ි', 'බ', 'ත', '්']` - seven atoms, none of which is a valid linguistic unit. akuru-token splits first into *grapheme clusters*, so the same word becomes `['කි', 'රි', 'බ', 'ත්']` - four syllabic units that a native reader would recognise as the base alphabet. BPE then learns higher-level patterns (syllables, morphemes, common word fragments) rather than spending merges reconstructing what should have been atomic from the start.

The current implementation focuses on Sinhala. In future versions, we aim to extend support to other abugida scripts such as Tamil, Devanagari, and Malayalam, where the same grapheme-cluster problem applies.

**Note: Sinhala touching letters are not supported yet**.

### Citation

If you use akuru-token in your research, please cite:

```bibtex
@software{akuru_token,
  author  = {Ayantha Randika},
  title   = {akuru-token: A Grapheme-Aware BPE Tokenizer},
  year    = {2025},
  url     = {https://github.com/akuruAI/akuru-token},
}
```

## Installation

> **Note:** akuru-token is not yet on PyPI. Install from source for now.

```bash
git clone https://github.com/your-username/akuru-token.git
cd akuru-token
pip install -e .
```

If you want to run the tests:

```bash
pip install -e ".[dev]"
```

Once published:
```bash
pip install akuru-token
```

## Built-in vocabularies

| Name | Script | Vocab size | Details |
|------|--------|------------|---------|
| `sin_eng` | Sinhala + English | <!-- TODO --> | See below |

### sin_eng

<!-- TODO: fill in once training is complete

Training corpus:
- Dataset 1 - description, size (e.g. Sinhala Wikipedia dump, ~X M tokens)
- Dataset 2 - description, size
- ...

Total: ~X M tokens
Vocab size: X
min_frequency: X
-->

```python
from akuru_token import Vocab

Vocab.list_vocabs()           # ['sin_eng']
vocab = Vocab.load("sin_eng")
vocab.describe()
```

## Pre-tokenizers

| Class | Splitting | Symbols | Use case |
|-------|-----------|---------|----------|
| `GraphemePreTokenizer` | Whitespace | Grapheme clusters | Sinhala, mixed Sinhala–English |
| `GPT2PreTokenizer` | GPT-2 regex | Codepoints | English, Latin-script |
| `WhitespacePreTokenizer` | Whitespace | Codepoints | Simple / reference use |

The pre-tokenizer is recorded in the vocab JSON and resolved automatically on load. You never need to specify it manually.

## Inference

```python
from akuru_token import BPETokenizer

tok = BPETokenizer.from_file("sin_eng")

ids    = tok.encode("දළදා මාළිගාව")                 # [42, 7, 31, ...]
tokens = tok.encode("දළදා මාළිගාව", as_ids=False)   # ['දළ', 'දා', ...]
text   = tok.decode(ids)                             # "දළදා මාළිගාව"

# Batch
tok.encode_batch(["ශ්‍රී ලංකාව", "hello world"])
tok.decode_batch([ids1, ids2])
```

### BOS / EOS tokens

```python
tok = BPETokenizer.from_file("sin_eng", add_bos=True, add_eos=True)
tok.encode("ආයුබෝවන්", as_ids=False)
# ['<bos>', 'ආයු', 'බෝ', 'වන්', '<eos>']
```

## Training a custom vocabulary

```python
from akuru_token import BPETrainer, GraphemePreTokenizer

corpus = [
    "ශ්‍රී ලංකාව is a beautiful island.",
    "මහනුවර නගරයේ දළදා මාළිගාව පිහිටා ඇත.",
    "Kandy is the cultural capital of Sri Lanka.",
]

trainer = BPETrainer(
    vocab_size=4000,
    min_frequency=2,
    pre_tokenizer=GraphemePreTokenizer(),
)
vocab = trainer.train(corpus)
vocab.save("/path/to/my_vocab.json")
```

Training progress is logged via Python's standard `logging` module. To see it:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Vocab format

Vocabularies are stored as JSON:

```json
{
  "pretokenizer": { "name": "GraphemePreTokenizer", "attributes": {} },
  "vocab": { "<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3, "කි": 4, ... },
  "merges": [["කි", "රි"], ["කිරි", "බ"], ...]
}
```

Saving new files to the `vocabs/` directory is allowed. Overwriting an existing file is blocked. Delete it manually first to overwrite.

## Running tests

```bash
pytest
pytest -v                                           # verbose
pytest tests/test_tokenizer.py::TestBPETokenizer   # single class
```

## Project layout

```
akuru_token/
├── __init__.py
├── pretokenizer.py   # GraphemePreTokenizer, GPT2PreTokenizer, split_graphemes
├── trainer.py        # BPETrainer
├── tokenizer.py      # BPETokenizer
├── vocab.py          # Vocab
└── vocabs/
    └── sin_eng.json
scripts/
├── data_cleaner.py
├── data_downloader.py
├── diff_sample.py
├── sinhala_validator.py
└── sin_eng_trainer.py
test/
├── test_tokenizer.py
└── test_trainer.py
```

## Scripts

- `data_downloader.py` - download and cache Sinhala corpora (CC-100, Wikipedia, CulturaX).
- `data_cleaner.py` - clean corpus lines and filter invalid Sinhala sequences.
- `diff_sample.py` - sample raw vs cleaned lines into a unified diff for inspection.
- `sinhala_validator.py` - validate Sinhala combining sequences (SLS 1134:2011).
- `sin_eng_trainer.py` - train the `sin_eng` BPE vocabulary from cleaned corpora.
