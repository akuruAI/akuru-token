"""
BPETrainer: learns BPE merge rules from a text corpus.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .pretokenizer import BasePreTokenizer, DefaultPreTokenizer
from .vocab import Vocab

logger = logging.getLogger(__name__)

Word = Tuple[str, ...]
WordFreqs = Dict[Word, int]
Pair = Tuple[str, str]


class BPETrainer:
    """
    Trains a BPE vocabulary from raw text.

    Parameters
    ----------
    vocab_size:
        Target vocabulary size (including special tokens and byte-level chars).
    min_frequency:
        Pairs that appear fewer than this many times are not merged.
    special_tokens:
        Extra special tokens to add before byte characters.
    guaranteed_tokens:
        Tokens to force into the initial vocabulary before merge training.
    pre_tokenizer:
        How to split text into words before BPE. Defaults to GPT-2 style.
    show_progress:
        Log progress every *show_progress* merges (0 = silent).
    """

    def __init__(
        self,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        guaranteed_tokens: Optional[List[int]] = None,
        pre_tokenizer: Optional[BasePreTokenizer] = None,
        show_progress: int = 500,
    ) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or [
            Vocab.SPECIAL_UNK,
            Vocab.SPECIAL_PAD,
            Vocab.SPECIAL_BOS,
            Vocab.SPECIAL_EOS,
        ]
        self.guaranteed_tokens = guaranteed_tokens
        self.pre_tokenizer: BasePreTokenizer = pre_tokenizer or DefaultPreTokenizer()
        self.show_progress = show_progress

    def train(self, texts: Iterable[str]) -> Vocab:
        """
        Train BPE on an iterable of strings and return a fitted :class:`Vocab`.
        """
        logger.info("Counting word frequencies…")
        word_freqs = self._count_word_frequencies(texts)

        vocab = self._initialize_vocab(word_freqs)
        vocab.pretokenizer_name = type(self.pre_tokenizer).__name__
        vocab.pretokenizer_attributes = {"normalize": self.pre_tokenizer.normalize}
        num_merges = self.vocab_size - len(vocab)

        if num_merges <= 0:
            logger.warning(
                "Initial vocab (%d tokens) already meets vocab_size=%d; no merges performed.",
                len(vocab),
                self.vocab_size,
            )
            return vocab

        logger.info(
            "Starting BPE training: %d initial tokens, %d merges to perform.",
            len(vocab),
            num_merges,
        )

        # word: list-of-symbols
        word_symbols: Dict[Word, List[str]] = {word: list(word) for word in word_freqs}

        # Build pair_freqs and pair_to_words once. Updated incrementally each merge.
        pair_freqs, pair_to_words = self._build_index(word_symbols, word_freqs)

        for merge_idx in range(num_merges):
            if not pair_freqs:
                logger.info("No more pairs to merge at step %d.", merge_idx)
                break

            best_pair, best_freq = max(
                pair_freqs.items(), key=lambda kv: (kv[1], kv[0])
            )

            if best_freq < self.min_frequency:
                logger.info(
                    "Best pair frequency %d < min_frequency %d; stopping.",
                    best_freq,
                    self.min_frequency,
                )
                break

            merged = best_pair[0] + best_pair[1]
            vocab.merges.append(best_pair)
            vocab.add_token(merged)

            self._apply_merge(
                word_symbols, word_freqs, pair_freqs, pair_to_words, best_pair, merged
            )

            if self.show_progress and (merge_idx + 1) % self.show_progress == 0:
                logger.info(
                    "Merge %d/%d: %r + %r → %r (freq=%d)",
                    merge_idx + 1,
                    num_merges,
                    *best_pair,
                    merged,
                    best_freq,
                )

        logger.info("Training complete. Final vocab size: %d", len(vocab))
        return vocab

    def train_from_files(self, *paths: str) -> Vocab:
        """Convenience wrapper: read files line-by-line and train."""

        def _lines():
            for p in paths:
                with open(p, encoding="utf-8") as f:
                    yield from f

        return self.train(_lines())

    def _count_word_frequencies(self, texts: Iterable[str]) -> WordFreqs:
        counter: Counter = Counter()
        for text in texts:
            for word in self.pre_tokenizer.pre_tokenize(text):
                if word:
                    symbols = self.pre_tokenizer.word_to_symbols(word)
                    counter[tuple(symbols)] += 1
        return dict(counter)

    def _initialize_vocab(self, word_freqs: WordFreqs) -> Vocab:
        
        vocab = Vocab()
        for tok in self.special_tokens:
            vocab.add_token(tok)
        
        if self.guaranteed_tokens is None:
            symbols: set = set()
        else:
             symbols: set = set(self.guaranteed_tokens)
        
        for word in word_freqs:
            symbols.update(word)
        for sym in sorted(symbols):
            vocab.add_token(sym)
        return vocab

    def _build_index(
        self,
        word_symbols: Dict[Word, List[str]],
        word_freqs: WordFreqs,
    ) -> Tuple[Dict[Pair, int], Dict[Pair, Set[Word]]]:
        """
        Build pair_freqs and pair_to_words.

        pair_freqs     : frequency of each adjacent pair.
        pair_to_words  : set of words that contain each pair.

        Both structures are maintained incrementally by _apply_merge.
        """
        pair_freqs: Dict[Pair, int] = defaultdict(int)
        pair_to_words: Dict[Pair, Set[Word]] = defaultdict(set)

        for word, symbols in word_symbols.items():
            freq = word_freqs[word]
            for a, b in zip(symbols, symbols[1:]):
                pair_freqs[(a, b)] += freq
                pair_to_words[(a, b)].add(word)

        return pair_freqs, pair_to_words

    def _apply_merge(
        self,
        word_symbols: Dict[Word, List[str]],
        word_freqs: WordFreqs,
        pair_freqs: Dict[Pair, int],
        pair_to_words: Dict[Pair, Set[Word]],
        pair: Pair,
        merged: str,
    ) -> None:
        """
        Apply one BPE merge and update pair_freqs / pair_to_words incrementally.

        Only the words that actually contain *pair* are visited (via pair_to_words),
        and only the neighbouring pairs that changed are updated in pair_freqs.
        """
        a, b = pair
        affected_words = list(pair_to_words.get(pair, set()))

        for word in affected_words:
            symbols = word_symbols[word]
            freq = word_freqs[word]
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    # remove broken left neighbour pair (x, a)
                    if i > 0:
                        left_pair = (symbols[i - 1], a)
                        pair_freqs[left_pair] -= freq
                        if pair_freqs[left_pair] <= 0:
                            del pair_freqs[left_pair]
                        pair_to_words[left_pair].discard(word)

                    # remove broken right neighbour pair (b, y)
                    if i + 2 < len(symbols):
                        right_pair = (b, symbols[i + 2])
                        pair_freqs[right_pair] -= freq
                        if pair_freqs[right_pair] <= 0:
                            del pair_freqs[right_pair]
                        pair_to_words[right_pair].discard(word)

                    # apply the merge
                    symbols[i] = merged
                    del symbols[i + 1]

                    # add new left neighbour pair (x, merged)
                    if i > 0:
                        new_left = (symbols[i - 1], merged)
                        pair_freqs[new_left] += freq
                        pair_to_words[new_left].add(word)

                    # add new right neighbour pair (merged, y)
                    if i + 1 < len(symbols):
                        new_right = (merged, symbols[i + 1])
                        pair_freqs[new_right] += freq
                        pair_to_words[new_right].add(word)
                else:
                    i += 1

        # The pair is merged such pair no longer exists
        pair_freqs.pop(pair, None)
        pair_to_words.pop(pair, None)
        
