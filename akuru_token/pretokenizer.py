"""
Pre-tokenization: splits raw text into words before BPE is applied.

Pre-tokenizers handle two concerns:
  1. Word splitting  - how to break a sentence into word-level chunks
  2. Symbol splitting - how to break each word into its base symbols (the
                        atoms that BPE will merge)

pre-tokenizers
--------------
WhitespacePreTokenizer   Simple whitespace split; codepoint symbols.
GPT2PreTokenizer         GPT-2 regex split; codepoint symbols.
GraphemePreTokenizer     Whitespace split; grapheme cluster symbols.
                         Handles Sinhala + English mixed text correctly.

Grapheme segmentation
---------------------
UAX #29 (via ugrapheme) is used as the base segmentation algorithm.
Sinhala ZWJ conjuncts are not covered by UAX #29 default rules - the spec
deliberately excludes them. A post-processing step re-joins clusters that
form valid Sinhala conjuncts:

    consonant + ් (virama, U+0DCA) + ZWJ (U+200D) + consonant

This covers rakaransaya (‍ර), yansaya (‍ය), repaya (ර්‍), and chained
conjuncts. No other ZWJ sequences are re-joined - malformed sequences from
noisy web text (ZWJ after vowel signs, multiple ZWJs, ZWJ + punctuation)
are correctly left split by UAX #29.

Note: This implementation does not support touching letters
"""

from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod
from typing import List

import regex


def _is_sinhala_consonant(ch: str) -> bool:
    return "\u0d9a" <= ch <= "\u0dc6"


def split_graphemes(text: str) -> List[str]:
    """
    Split *text* into a list of Unicode extended grapheme clusters.

    Uses regex ``\\X`` (UAX #29) as the base, then re-joins Sinhala ZWJ conjunct
    sequences that UAX #29 splits but Sinhala rendering requires to be atomic:

        consonant + ් + ZWJ + consonant  (rakaransaya, yansaya, repaya, etc.)

    Malformed sequences from noisy web text are intentionally left split:
        - ZWJ after a vowel sign  (e.g. රා + ZWJ + ම)
        - Multiple consecutive ZWJs
        - ZWJ followed by punctuation or non-Sinhala characters
    Note: This function does not support touching letters
    """
    clusters: List[str] = regex.findall(r'\X', text)
    return _rejoin_sinhala_conjuncts(clusters)


def _rejoin_sinhala_conjuncts(clusters: List[str]) -> List[str]:
    """
    Re-join adjacent clusters that form a valid Sinhala ZWJ conjunct.

    A cluster qualifies for re-joining when:
      - It ends with Al lakuna + ZWJ  (U+0DCA + U+200D)
      - The following cluster starts with a Sinhala consonant (U+0D9A-U+0DC6)

    This is applied repeatedly so chained conjuncts like ක්‍ෂ්‍ර merge fully.
    """
    if not clusters:
        return clusters

    result: List[str] = []
    i = 0
    while i < len(clusters):
        current = clusters[i]
        # Check if current cluster ends with virama + ZWJ
        while (
            i + 1 < len(clusters)
            and current.endswith("\u0dca\u200d")
            and _is_sinhala_consonant(clusters[i + 1][0])
        ):
            i += 1
            current += clusters[i]
        result.append(current)
        i += 1

    return result

class BasePreTokenizer(ABC):
    """
    Parameters
    ----------
    normalize:
        If True (default), apply Unicode NFC normalization to the input text.
        This ensures consistent vocab keys. The flag is persisted in the vocab
        JSON so training and inference always agree.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def _normalize(self, text: str) -> str:
        """Return NFC-normalized text when ``self.normalize`` is True."""
        return unicodedata.normalize("NFC", text) if self.normalize else text

    @abstractmethod
    def pre_tokenize(self, text: str) -> List[str]:
        """
        Split text into word-level chunks.

        Each returned string is a word - BPE will not merge across word
        boundaries. Implementations must call ``self._normalize(text)``
        as their first step.
        """

    def word_to_symbols(self, word: str) -> List[str]:
        """
        Break a single pre-tokenized word into its base symbols.

        By default it splits by Unicode codepoint.
        Override this to change the atomic unit (e.g. grapheme clusters).
        """
        return list(word)


class WhitespacePreTokenizer(BasePreTokenizer):
    """Splits on whitespace; codepoint-level symbols."""

    def __init__(self, normalize: bool = True) -> None:
        super().__init__(normalize=normalize)

    def pre_tokenize(self, text: str) -> List[str]:
        return self._normalize(text).split()


class GPT2PreTokenizer(BasePreTokenizer):
    """
    GPT-2 regex word splitter; codepoint-level symbols.

    Handles English contractions, punctuation, and digits.
    Leading space is encoded as Ġ (U+0120) so the model is space-aware.
    """

    _PATTERN = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
        regex.UNICODE,
    )

    def __init__(self, normalize: bool = True) -> None:
        super().__init__(normalize=normalize)

    def pre_tokenize(self, text: str) -> List[str]:
        tokens = []
        for match in self._PATTERN.finditer(self._normalize(text)):
            word = match.group()
            if word.startswith(" "):
                word = "\u0120" + word[1:]
            tokens.append(word)
        return tokens


class GraphemePreTokenizer(BasePreTokenizer):
    """
    Pre-tokenizer for mixed Sinhala + English text.

    Word splitting
    ~~~~~~~~~~~~~~
    Splits on whitespace runs, attaching a leading Ġ (U+0120) space marker to
    each non-first word so the model is space-aware.

    This is intentionally simpler than the GPT-2 regex. The GPT-2 regex uses
    \\w which does NOT match Sinhala combining marks (Unicode category Mn/Mc),
    so vowel signs like ි, ා, ු would be split off as separate tokens before
    BPE even runs - shredding grapheme clusters apart. Whitespace splitting
    keeps every Sinhala word intact, including its vowel signs, virama, and
    ZWJ conjuncts, ready for grapheme segmentation.

    Punctuation is kept attached to its word (e.g. කිරිබත්,). BPE will
    learn to split punctuation through merges, which is standard practice.

    Symbol splitting
    ~~~~~~~~~~~~~~~~
    Each word is broken into grapheme clusters rather than codepoints:

    - කිරිබත්	->  [කි, රි, බ, ත්]   (4 graphemes, not 7 codepoints)
    - ශ්‍රී	->  [ශ්‍රී]             (1 conjunct grapheme via ZWJ)
    - hello	->  [h, e, l, l, o]   (unchanged for Latin)

    Every grapheme cluster is a linguistically valid atom, so BPE starts from
    meaningful units and spends its merge budget on higher-level patterns
    (syllables, morphemes, word fragments) rather than reconstructing graphemes
    from raw codepoints.
    """

    _WHITESPACE = regex.compile(r" +")

    def __init__(self, normalize: bool = True) -> None:
        super().__init__(normalize=normalize)

    def pre_tokenize(self, text: str) -> List[str]:
        parts = self._WHITESPACE.split(self._normalize(text))
        tokens = []
        for i, part in enumerate(parts):
            if not part:
                continue
            tokens.append(("\u0120" + part) if i > 0 else part)
        return tokens

    def word_to_symbols(self, word: str) -> List[str]:
        """Return grapheme clusters, preserving the leading Ġ marker if present."""
        if word.startswith("\u0120"):
            return ["\u0120"] + split_graphemes(word[1:])
        return split_graphemes(word)


DefaultPreTokenizer = GraphemePreTokenizer
