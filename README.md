# akuru-token

A grapheme-aware BPE tokenizer for Sinhala and mixed Sinhala–English text.

Most tokenizers split text into Unicode codepoints before applying BPE. In Sinhala, this shreds syllables: `කිරිබත්` becomes `['ක', 'ි', 'ර', 'ි', 'බ', 'ත', '්']` - seven atoms, none of which is a valid linguistic unit. akuru-token splits first into *grapheme clusters*, so the same word becomes `['කි', 'රි', 'බ', 'ත්']` - four syllabic units a native reader would recognise. BPE then learns syllables, morphemes, and word fragments rather than spending its merge budget reconstructing what should have been atomic from the start. 

As [UAX #29](https://unicode.org/reports/tr29/) (Unicode grapheme clustering spec), excludes Sinhala ZWJ conjuncts, a post-processing step re-joins valid conjunct sequences, covering rakaransaya, yansaya, repaya, and chained conjuncts. Training data is filtered against [SLS 1134:2011](https://www.language.lk/download/sls1134/) to remove lines with invalid Sinhala syntax.

The current implementation focuses on Sinhala. Future versions aim to extend support to other abugida scripts (Tamil, Devanagari, Malayalam), where the same grapheme-cluster problem applies.

**Note:** Sinhala touching letters are not yet supported.

## Citation

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

> akuru-token is not yet on PyPI. Install from source for now.

```bash
git clone https://github.com/akuruAI/akuru-token.git
cd akuru-token
pip install -e .
```

## Inference

```python
from akuru_token import BPETokenizer

tok = BPETokenizer.from_file("sin_eng")

ids    = tok.encode("දළදා මාළිගාව")                # [42, 7, 31, ...]
tokens = tok.encode("දළදා මාළිගාව", as_ids=False)  # ['දළ', 'දා', ...]
text   = tok.decode(ids)                            # "දළදා මාළිගාව"

tok.encode_batch(["ශ්‍රී ලංකාව", "hello world"])
tok.decode_batch([ids1, ids2])
```

Adding BOS / EOS tokens:

```python
tok = BPETokenizer.from_file("sin_eng", add_bos=True, add_eos=True)
tok.encode("ආයුබෝවන්", as_ids=False)
# ['<bos>', 'ආයු', 'බෝ', 'වන්', '<eos>']
```

## Training a custom vocabulary

Use `BPETrainer` with any pre-tokenizer and an iterable of strings. See [`scripts/README.md`](scripts/README.md) for the full training pipeline used to produce the built-in `sin_eng` vocabulary.

## Pre-tokenizers

| Class | Word splitting | Base symbols | Use case |
|-------|---------------|--------------|----------|
| `GraphemePreTokenizer` | Whitespace + digit boundary (`\p{N}{1,3}`) | Grapheme clusters | Sinhala, mixed Sinhala–English |
| `GPT2PreTokenizer` | GPT-2 regex | Codepoints | English, Latin-script |
| `WhitespacePreTokenizer` | Whitespace | Codepoints | Simple / reference use |

The pre-tokenizer is recorded in the vocab JSON and resolved automatically on load - you never need to specify it manually.

### Whitespace boundaries

`GraphemePreTokenizer` splits on all whitespace, with different treatment for spaces and non-space whitespace:

- **Spaces** attach as `Ġ` (U+0120) prefix to the following word. Multiple consecutive spaces produce standalone `Ġ` tokens for all but the last, which attaches to the next word.
- **Non-space whitespace** (`\n`, `\t`, `\r`, etc.) acts as a hard boundary. Consecutive runs are grouped into a single standalone token so BPE can learn merges like `\n\n` within the run, but never across into adjacent text.

```
"a \n real"  → ['a', 'Ġ', '\n', 'Ġreal']
"a\nreal"    → ['a', '\n', 'real']
"a\n\nreal"  → ['a', '\n\n', 'real']
"a\t real"   → ['a', '\t', 'Ġreal']
```

This ensures whitespace characters that carry semantic meaning (paragraph breaks, indentation) are preserved faithfully as standalone tokens, following the same principle as LLaMA/SentencePiece (`\n` and `\n\n` as distinct tokens) and tiktoken (explicit `\s*[\r\n]` boundary in the regex).

### Digit boundary splitting

The digit-boundary split caps numeric runs to 3 digits, following the [`\p{N}{1,3}` split pattern](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py) in tiktoken's `cl100k_base` encoding. Without this, a number like `2024` is a single pre-tokenization unit, and if it is frequent enough in the corpus BPE will eventually merge its digits into a single token - one that is useless for encoding any other number. With the cap, `2024` pre-splits as `["202", "4"]` before BPE runs, so no digit token longer than 3 digits can ever form regardless of how often it appears in training data.

## Vocab format

```json
{
  "pretokenizer": { "name": "GraphemePreTokenizer", "attributes": {} },
  "vocab": { "<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3, "කි": 4, ... },
  "merges": [["කි", "රි"], ["කිරි", "බ"], ...]
}
```

Saving to the `vocabs/` directory is allowed; overwriting an existing file is blocked. Delete the file manually first to replace it.