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
GraphemePreTokenizer      Whitespace split; grapheme cluster symbols.
                         Handles Sinhala + English mixed text correctly.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List


def split_graphemes(text: str) -> List[str]:
    """
    Split *text* into a list of Unicode extended grapheme clusters.

    Rules applied (sufficient for Sinhala + Latin scripts):
      - Combining marks (Mn, Mc, Me) are attached to the preceding base char.
      - A Zero Width Joiner (U+200D) and the character following it are absorbed
        into the current cluster, forming conjunct consonants (e.g. ශ්‍ර).

    For scripts beyond Sinhala/Latin the full UAX #29 algorithm would be needed,
    but this covers all practical cases in the target domain.
    """
    graphemes: List[str] = []
    current = ""
    i = 0
    while i < len(text):
        ch = text[i]
        cat = unicodedata.category(ch)

        if not current:
            current = ch
        elif ch == "\u200d":
            # Zero Width Joiner: pull it and the next char into the current cluster
            current += ch
            if i + 1 < len(text):
                i += 1
                current += text[i]
        elif cat in ("Mn", "Mc", "Me"):
            # Combining mark: attach to current cluster
            current += ch
        else:
            graphemes.append(current)
            current = ch

        i += 1

    if current:
        graphemes.append(current)

    return graphemes


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

    _PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
        re.UNICODE,
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

    _WHITESPACE = re.compile(r" +")

    def __init__(self, normalize: bool = True) -> None:
        super().__init__(normalize=normalize)

    def pre_tokenize(self, text: str) -> List[str]:
        parts = self._WHITESPACE.split(self._normalize(text))
        tokens = []
        for i, part in enumerate(parts):
            if not part:
                continue
            # Prepend Ġ to mark inter-word spaces
            tokens.append(("\u0120" + part) if i > 0 else part)
        return tokens

    def word_to_symbols(self, word: str) -> List[str]:
        """Return grapheme clusters, preserving the leading Ġ marker if present."""
        if word.startswith("\u0120"):
            return ["\u0120"] + split_graphemes(word[1:])
        return split_graphemes(word)


DefaultPreTokenizer = GraphemePreTokenizer
