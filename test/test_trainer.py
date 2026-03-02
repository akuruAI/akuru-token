"""
Extended tests for BPETrainer - focused on training correctness.

  Layer 1 - Merge validity
      Merges are well-formed, ordered by frequency, respect min_frequency.

  Layer 2 - Pair frequency consistency
      After each merge step, pair counts in the resulting vocab are consistent
      with what a fresh count of the corpus would produce.

  Layer 3 - Edge case corpora
      Single words, repeated pairs, empty lines, very small/large vocab sizes.
"""

from collections import Counter, defaultdict

import pytest

from akuru_token import BPETrainer, BPETokenizer, Vocab
from akuru_token.pretokenizer import GraphemePreTokenizer

SMALL_CORPUS = [
    "ශ්‍රී දළදා මාලිගාව යනු බුදුරජාණන් වහන්සේගේ",
    "ශ්‍රී ලංකාව is a beautiful island.",
    "the cat sat on the mat",
    "low lower lowest",
    "new newer newest",
]

MIXED_CORPUS = SMALL_CORPUS + [
    "Kandy මහනුවර is the cultural capital.",
    "වර්තමානයේ තැන්පත් කර ඇති මාළිගාවයි.",
]


def make_trainer(**kwargs) -> BPETrainer:
    defaults = dict(
        vocab_size=170,
        min_frequency=1,
        pre_tokenizer=GraphemePreTokenizer(),
        show_progress=0,
    )
    defaults.update(kwargs)
    return BPETrainer(**defaults)


@pytest.fixture(scope="module")
def trained_vocab():
    return make_trainer(vocab_size=200).train(MIXED_CORPUS)


@pytest.fixture(scope="module")
def trained_tokenizer(trained_vocab):
    return BPETokenizer(trained_vocab)



class TestMergeValidity:

    def test_merges_are_pairs(self, trained_vocab):
        for pair in trained_vocab.merges:
            assert len(pair) == 2, f"Merge {pair!r} is not a 2-tuple"

    def test_merged_token_exists_in_vocab(self, trained_vocab):
        # Every merge (a, b) must produce a+b that is in the vocab
        for a, b in trained_vocab.merges:
            merged = a + b
            assert (
                merged in trained_vocab
            ), f"Merged token {merged!r} from merge ({a!r}, {b!r}) missing from vocab"

    def test_merge_components_exist_in_vocab(self, trained_vocab):
        # Both sides of every merge must themselves be in the vocab
        for a, b in trained_vocab.merges:
            assert a in trained_vocab, f"Left component {a!r} missing from vocab"
            assert b in trained_vocab, f"Right component {b!r} missing from vocab"

    def test_merges_respect_min_frequency(self):
        min_freq = 5
        corpus = ["ab cd ef"] * 3 + ["ab cd"] * 10
        vocab = make_trainer(vocab_size=100, min_frequency=min_freq).train(corpus)
        merged_tokens = {a + b for a, b in vocab.merges}

        # ("a","b") appears 13 times: above threshold, must be merged
        assert "ab" in merged_tokens

        # ("c","d") appears 13 times: above threshold, must be merged with Ġ
        assert "Ġcd" in merged_tokens

        # ("e","f") appears 3 times: below threshold, must NOT be merged with or without Ġ
        assert "ef" not in merged_tokens
        assert "Ġef" not in merged_tokens

    def test_vocab_size_does_not_exceed_target(self):
        vocab = make_trainer(vocab_size=100).train(MIXED_CORPUS)
        assert len(vocab) <= 100

    def test_special_tokens_have_lowest_ids(self, trained_vocab):
        assert trained_vocab.token_to_id(Vocab.SPECIAL_UNK) == 0
        assert trained_vocab.token_to_id(Vocab.SPECIAL_PAD) == 1
        assert trained_vocab.token_to_id(Vocab.SPECIAL_BOS) == 2
        assert trained_vocab.token_to_id(Vocab.SPECIAL_EOS) == 3

    def test_merge_count_equals_vocab_growth(self, trained_vocab):
        # Each merge adds exactly one new token - so merge count must equal
        # (final vocab size) - (initial base symbol count)
        # We can infer base count: special tokens + all unique base symbols
        pt = GraphemePreTokenizer()
        base_symbols = set()
        for text in MIXED_CORPUS:
            for word in pt.pre_tokenize(text):
                base_symbols.update(pt.word_to_symbols(word))
        base_vocab_size = 4 + len(base_symbols)  # 4 special tokens
        assert len(trained_vocab.merges) == len(trained_vocab) - base_vocab_size

    def test_deterministic_across_runs(self):
        # Same corpus + same settings must produce identical merges every time
        v1 = make_trainer(vocab_size=100).train(SMALL_CORPUS)
        v2 = make_trainer(vocab_size=100).train(SMALL_CORPUS)
        assert v1.merges == v2.merges


class TestPairFrequencyConsistency:

    def _count_pair_freqs(self, corpus):
        """Recount all pair frequencies from scratch on the raw corpus."""
        pt = GraphemePreTokenizer()
        word_freqs = Counter()
        for text in corpus:
            for word in pt.pre_tokenize(text):
                if word:
                    word_freqs[tuple(pt.word_to_symbols(word))] += 1

        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            for a, b in zip(word, word[1:]):
                pair_freqs[(a, b)] += freq
        return dict(pair_freqs)

    def test_first_merge_is_most_frequent_pair(self):
        corpus = ["ab cd ab cd ab cd", "ab xy"]
        vocab = make_trainer(vocab_size=50, min_frequency=1).train(corpus)
        pair_freqs = self._count_pair_freqs(corpus)
        first_merge = vocab.merges[0]
        first_freq = pair_freqs.get(first_merge, 0)
        max_freq = max(pair_freqs.values())
        assert first_freq == max_freq, (
            f"First merge {first_merge!r} has freq {first_freq}, "
            f"but max pair freq is {max_freq}"
        )

    def test_merges_are_non_increasing_frequency(self):
        """
        Each merge should have frequency >= the next merge at the time it was chosen.
        We approximate this on the original corpus - not a perfect check since
        pair freqs shift as merges are applied, but catches gross violations.
        """
        corpus = ["ab"] * 20 + ["cd"] * 10 + ["ef"] * 3
        vocab = make_trainer(vocab_size=50, min_frequency=1).train(corpus)
        pair_freqs = self._count_pair_freqs(corpus)
        freqs = [pair_freqs.get(pair, 0) for pair in vocab.merges]
        for i in range(len(freqs) - 1):
            assert freqs[i] >= freqs[i + 1], (
                f"Merge {i} has lower initial freq ({freqs[i]}) "
                f"than merge {i+1} ({freqs[i+1]})"
            )

    def test_high_frequency_pairs_are_merged_before_low(self):
        # In a corpus where one pair dominates, it must appear early in merges
        corpus = ["ab"] * 50 + ["cd"] * 5
        vocab = make_trainer(vocab_size=50, min_frequency=1).train(corpus)
        merged_tokens = [a + b for a, b in vocab.merges]
        assert "ab" in merged_tokens
        if "cd" in merged_tokens:
            assert merged_tokens.index("ab") < merged_tokens.index(
                "cd"
            ), "High-frequency 'ab' should be merged before low-frequency 'cd'"

    def test_equal_frequency_pairs_both_eventually_merged(self):
        # Two equally frequent pairs - both should end up merged given enough budget
        corpus = ["ab"] * 10 + ["cd"] * 10
        vocab = make_trainer(vocab_size=50, min_frequency=1).train(corpus)
        merged_tokens = {a + b for a, b in vocab.merges}
        assert "ab" in merged_tokens
        assert "cd" in merged_tokens

    def test_min_frequency_two_blocks_rare_merges(self):
        corpus = ["ab"] * 1 + ["cd"] * 10
        vocab = make_trainer(vocab_size=50, min_frequency=2).train(corpus)
        merged_tokens = {a + b for a, b in vocab.merges}
        # "ab" appears once - must NOT be merged
        assert "ab" not in merged_tokens
        # "cd" appears 10 times - must be merged
        assert "cd" in merged_tokens

    def test_sinhala_frequent_grapheme_pairs_merged(self):
        # ශ්‍රී appears multiple times - its grapheme pairs should be merged
        corpus = ["ශ්‍රී ලංකාව"] * 20
        vocab = make_trainer(vocab_size=50, min_frequency=1).train(corpus)
        assert len(vocab.merges) > 0
        # At least one Sinhala multi-grapheme token must exist
        sinhala_merges = [
            a + b
            for a, b in vocab.merges
            if any("\u0d80" <= ch <= "\u0dff" for ch in a + b)
        ]
        assert len(sinhala_merges) > 0

class TestEdgeCaseCorpora:

    def test_single_word_corpus(self):
        vocab = make_trainer(vocab_size=20).train(["hello"])
        assert len(vocab) > 0

    def test_repeated_pair_in_word(self):
        # Word containing same pair twice: "abab" → should merge (a,b) twice
        vocab = make_trainer(vocab_size=20, min_frequency=1).train(["abab"] * 5)
        merged_tokens = {a + b for a, b in vocab.merges}
        assert "ab" in merged_tokens
        tok = BPETokenizer(vocab)
        ids = tok.encode("abab")
        decoded = tok.decode(ids)
        assert decoded == "abab"

    def test_pair_only_at_start_of_word(self):
        # Pair (a,b) only at position 0 - no left neighbour
        vocab = make_trainer(vocab_size=20, min_frequency=1).train(["abcd"] * 5)
        tok = BPETokenizer(vocab)
        assert tok.decode(tok.encode("abcd")) == "abcd"

    def test_pair_only_at_end_of_word(self):
        # Pair (c,d) only at last position - no right neighbour
        vocab = make_trainer(vocab_size=20, min_frequency=1).train(["abcd"] * 5)
        tok = BPETokenizer(vocab)
        assert tok.decode(tok.encode("abcd")) == "abcd"

    def test_two_symbol_word_merges_to_one(self):
        vocab = make_trainer(vocab_size=10, min_frequency=1).train(["ab"] * 10)
        merged_tokens = {a + b for a, b in vocab.merges}
        assert "ab" in merged_tokens

    def test_empty_lines_ignored(self):
        corpus = ["hello world", "", "   ", "ශ්‍රී ලංකාව"]
        vocab = make_trainer(vocab_size=50).train(corpus)
        assert len(vocab) > 0

    def test_vocab_size_cap_when_pairs_exhausted(self):
        # Requesting far more merges than possible - should stop gracefully
        vocab = make_trainer(vocab_size=10_000, min_frequency=1).train(["ab cd"])
        assert len(vocab) < 10_000

    def test_large_repetition_does_not_crash(self):
        corpus = ["ශ්‍රී ලංකාව is beautiful"] * 500
        vocab = make_trainer(vocab_size=300).train(corpus)
        tok = BPETokenizer(vocab)
        assert tok.decode(tok.encode("ශ්‍රී ලංකාව")) == "ශ්‍රී ලංකාව"
