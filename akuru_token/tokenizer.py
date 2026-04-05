"""
BPETokenizer: encodes and decodes text using a trained BPE :class:`Vocab`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

from .pretokenizer import (
    BasePreTokenizer,
    GraphemePreTokenizer,
    GPT2PreTokenizer,
    WhitespacePreTokenizer,
)
from .vocab import Vocab


# Registry mapping pretokenizer names (as stored in the vocab JSON) to classes.
_PRETOKENIZER_REGISTRY: Dict[str, Type[BasePreTokenizer]] = {
    "GraphemePreTokenizer": GraphemePreTokenizer,
    "GPT2PreTokenizer": GPT2PreTokenizer,
    "WhitespacePreTokenizer": WhitespacePreTokenizer,
}


def _resolve_pretokenizer(vocab: Vocab) -> BasePreTokenizer:
    """Instantiate the pre-tokenizer recorded in *vocab*, including its attributes."""
    name = vocab.pretokenizer_name
    cls = _PRETOKENIZER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown pretokenizer {name!r}. " f"Known: {list(_PRETOKENIZER_REGISTRY)}."
        )
    return cls(**vocab.pretokenizer_attributes)


class BPETokenizer:
    """
    Inference-time tokenizer.  Requires a trained :class:`Vocab`.

    The pre-tokenizer is resolved automatically from the vocab's
    ``pretokenizer_name`` attribute. Override it by passing *pre_tokenizer*
    explicitly, but this should rarely be necessary.

    Parameters
    ----------
    vocab:
        A :class:`Vocab` produced by :class:`BPETrainer` (or loaded from disk).
    pre_tokenizer:
        Override the pre-tokenizer recorded in the vocab. If omitted, the
        correct pre-tokenizer is resolved automatically.
    add_bos / add_eos:
        Automatically prepend / append BOS or EOS token ids.
    unk_fallback:
        When True (default), unknown grapheme clusters are decomposed into
        codepoints before BPE so that existing merges can still apply,
        avoiding unnecessary ``<unk>`` tokens.  Set to False to disable
        this fallback (unknown clusters map straight to ``<unk>``).
    """

    def __init__(
        self,
        vocab: Vocab,
        pre_tokenizer: Optional[BasePreTokenizer] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        unk_fallback: bool = True,
    ) -> None:
        self.vocab = vocab
        self.pre_tokenizer: BasePreTokenizer = (
            pre_tokenizer if pre_tokenizer is not None else _resolve_pretokenizer(vocab)
        )
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.unk_fallback = unk_fallback

        # Build a fast lookup: (a, b) -> merge rank
        self._merge_rank: Dict[Tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(vocab.merges)
        }

    def encode(self, text: str, as_ids: bool = True) -> List[int] | List[str]:
        """
        Encode *text* into a list of token ids (default) or token strings.

        Parameters
        ----------
        text:
            Raw input string.
        as_ids:
            If True (default) return integer ids; otherwise return token strings.
        """
        tokens: List[str] = []

        if self.add_bos:
            tokens.append(Vocab.SPECIAL_BOS)

        for word in self.pre_tokenizer.pre_tokenize(text):
            tokens.extend(self._bpe(word))

        if self.add_eos:
            tokens.append(Vocab.SPECIAL_EOS)

        if as_ids:
            unk_id = self.vocab.unk_id
            return [self.vocab.token_to_id(t) or unk_id for t in tokens]
        return [t if t in self.vocab else self.vocab.SPECIAL_UNK for t in tokens]

    def encode_batch(self, texts: List[str], as_ids: bool = True) -> List[List]:
        """Encode a list of strings."""
        return [self.encode(t, as_ids=as_ids) for t in texts]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token ids back to a string.

        Parameters
        ----------
        ids:
            List of integer token ids.
        skip_special_tokens:
            If True, strip BOS/EOS/PAD/UNK from the output.
        """
        special = {
            Vocab.SPECIAL_UNK,
            Vocab.SPECIAL_PAD,
            Vocab.SPECIAL_BOS,
            Vocab.SPECIAL_EOS,
        }
        pieces: List[str] = []
        for idx in ids:
            token = self.vocab.id_to_token(idx)
            if token is None:
                continue
            if skip_special_tokens and token in special:
                continue
            pieces.append(token)

        # If any piece contains the GPT-2 space marker (Ġ), use it for spacing.
        # Otherwise, join with spaces.
        if any("\u0120" in p for p in pieces):
            text = "".join(pieces).replace("\u0120", " ").strip()
        else:
            text = "".join(pieces)
        return text

    def decode_batch(
        self, batch: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of id sequences."""
        return [self.decode(ids, skip_special_tokens) for ids in batch]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        add_bos: bool = False,
        add_eos: bool = False,
        unk_fallback: bool = True,
    ) -> "BPETokenizer":
        """
        Load a tokenizer from a vocab JSON file.

        The pre-tokenizer is resolved automatically from the vocab file.
        """
        vocab = Vocab.load(path)
        return cls(vocab, add_bos=add_bos, add_eos=add_eos, unk_fallback=unk_fallback)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _bpe(self, word: str) -> List[str]:
        """Apply the learned merge rules to a single pre-tokenized word."""
        if not word:
            return []

        symbols: List[str] = self.pre_tokenizer.word_to_symbols(word)

        if self.unk_fallback:
            # Decompose unknown grapheme clusters into codepoints so BPE
            # can merge what it knows instead of producing <unk>.
            expanded: List[str] = []
            for sym in symbols:
                if sym in self.vocab:
                    expanded.append(sym)
                else:
                    expanded.extend(list(sym))
            symbols = expanded

        while len(symbols) > 1:
            # Find the lowest rank adjacent pair
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1 or best_rank == float("inf"):
                break  # no more known merges

            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2 :]

        return symbols
