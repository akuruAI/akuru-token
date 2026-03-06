"""
Tests for akuru-token.
"""

import json
import pytest

from akuru_token import BPETokenizer, BPETrainer, Vocab
from akuru_token.pretokenizer import (
    GraphemePreTokenizer,
    GPT2PreTokenizer,
    WhitespacePreTokenizer,
    split_graphemes,
)

# Sinhala Wikipedia
SINHALA_SENTENCE = (
    "ශ්‍රී දළදා මාලිගාව යනු බුදුරජාණන් වහන්සේගේ වම් දන්තධාතූන් වහන්සේ "
    "වර්තමානයේ තැන්පත් කර ඇති මාළිගාවයි. වර්තමාන දළදා මාළිගාව ශ්‍රී "
    "ලංකාවේ මහනුවර නගරයේ පිහිටා ඇත."
)

SINHALA_WORDS = SINHALA_SENTENCE.split()

MIXED_CORPUS = [
    SINHALA_SENTENCE,
    "The Temple of the Tooth is in Kandy.",
    "ශ්‍රී ලංකාව is a beautiful island.",
    "Kandy මහනුවර is the cultural capital.",
]


@pytest.fixture(scope="module")
def grapheme_vocab():
    trainer = BPETrainer(
        vocab_size=300,
        min_frequency=1,
        pre_tokenizer=GraphemePreTokenizer(normalize=False),
        show_progress=0,
    )
    return trainer.train(MIXED_CORPUS)


@pytest.fixture(scope="module")
def grapheme_tokenizer(grapheme_vocab):
    return BPETokenizer(grapheme_vocab)


class TestGraphemeSegmentation:
    """Unit tests for the grapheme cluster splitter."""

    def test_sinhala_vowel_sign_attached(self):
        # කි is one grapheme (consonant + vowel sign), not two codepoints
        assert split_graphemes("කි") == ["කි"]

    def test_sinhala_virama_attached(self):
        # ත් is one grapheme (consonant + virama)
        assert split_graphemes("ත්") == ["ත්"]

    def test_repaya(self):
        # ර්‍ම is one grapheme (consonant + virama)
        assert split_graphemes("ර්‍ම") == ["ර්‍ම"]

    def test_zwj_conjunct(self):
        # Conjuncts formed with ZWJ must be one cluster
        # Tests rakaranshaya and yansaya
        assert split_graphemes("ශ්‍රක්‍ය") == ["ශ්‍ර", "ක්‍ය"]

    def test_chained_zwj_conjunct(self):
        # Chained conjunct: k + virama + ZWJ + sh + virama + ZWJ + r
        assert split_graphemes("ක්‍ෂ්‍ර") == ["ක්‍ෂ්‍ර"]

    def test_word_splits_correctly(self):
        # කිරිබත් → [කි, රි, බ, ත්]
        assert split_graphemes("කිරිබත්") == ["කි", "රි", "බ", "ත්"]

    def test_latin_unchanged(self):
        assert split_graphemes("hello") == ["h", "e", "l", "l", "o"]

    def test_mixed_script(self):
        result = split_graphemes("hi කි")
        assert result == ["h", "i", " ", "කි"]

    def test_sinhala_digits(self):
        result = split_graphemes("෧෨")
        assert result == ["෧", "෨"]

    def test_empty_string(self):
        assert split_graphemes("") == []

    def test_grapheme_count_less_than_codepoint_count(self):
        # Every Sinhala word with vowel signs will have fewer graphemes than codepoints
        word = "මාලිගාව"
        graphemes = split_graphemes(word)
        assert len(graphemes) < len(word)


class TestGraphemePreTokenizer:
    """Tests for word splitting and symbol extraction."""

    def setup_method(self):
        self.pt = GraphemePreTokenizer()

    def test_space_marker_on_non_first_word(self):
        words = self.pt.pre_tokenize("ශ්‍රී ලංකාව")
        assert words[0] == "ශ්‍රී"
        assert words[1].startswith("\u0120")  # Ġ marker

    def test_word_to_symbols_returns_graphemes(self):
        syms = self.pt.word_to_symbols("කිරිබත්")
        assert syms == ["කි", "රි", "බ", "ත්"]

    def test_space_marker_preserved_in_symbols(self):
        syms = self.pt.word_to_symbols("\u0120ලංකාව")
        assert syms[0] == "\u0120"

    def test_single_word_no_marker(self):
        words = self.pt.pre_tokenize("ආයුබෝවන්")
        assert len(words) == 1
        assert not words[0].startswith("\u0120")

    def test_mixed_script_words(self):
        words = self.pt.pre_tokenize("hello ලංකාව")
        assert any("\u0120" in w for w in words[1:])


class TestNFCNormalization:
    """
    Tests for Unicode NFC normalization in pre-tokenizers.

    NFD input (decomposed) must produce identical tokens/ids to NFC input
    (precomposed). The flag must round-trip through the vocab JSON so
    training and inference always agree.
    """

    # é as a single precomposed codepoint (NFC, U+00E9)
    NFC_TEXT = "caf\u00e9 ශ්‍රී"
    # é as base e + combining acute (NFD, U+0065 + U+0301)
    NFD_TEXT = "cafe\u0301 ශ්‍රී"

    def test_nfc_and_nfd_same_words_grapheme(self):
        pt = GraphemePreTokenizer(normalize=True)
        assert pt.pre_tokenize(self.NFC_TEXT) == pt.pre_tokenize(self.NFD_TEXT)

    def test_nfc_and_nfd_same_words_gpt2(self):
        pt = GPT2PreTokenizer(normalize=True)
        assert pt.pre_tokenize(self.NFC_TEXT) == pt.pre_tokenize(self.NFD_TEXT)

    def test_nfc_and_nfd_same_words_whitespace(self):
        pt = WhitespacePreTokenizer(normalize=True)
        assert pt.pre_tokenize(self.NFC_TEXT) == pt.pre_tokenize(self.NFD_TEXT)

    def test_normalize_false_preserves_nfd(self):
        # With normalize=False the NFD combining acute stays as a separate codepoint,
        # so the two inputs must produce different results.
        pt = GraphemePreTokenizer(normalize=False)
        assert pt.pre_tokenize(self.NFC_TEXT) != pt.pre_tokenize(self.NFD_TEXT)

    def test_normalize_default_is_true(self):
        assert GraphemePreTokenizer().normalize is True
        assert GPT2PreTokenizer().normalize is True
        assert WhitespacePreTokenizer().normalize is True

    def test_normalize_persisted_in_vocab(self):
        trainer = BPETrainer(
            vocab_size=200,
            min_frequency=1,
            pre_tokenizer=GraphemePreTokenizer(normalize=True),
            show_progress=0,
        )
        vocab = trainer.train(["caf\u00e9 ශ්‍රී ලංකාව"])
        assert vocab.pretokenizer_attributes["normalize"] is True

    def test_normalize_false_persisted_in_vocab(self):
        trainer = BPETrainer(
            vocab_size=200,
            min_frequency=1,
            pre_tokenizer=GraphemePreTokenizer(normalize=False),
            show_progress=0,
        )
        vocab = trainer.train(["caf\u00e9 ශ්‍රී ලංකාව"])
        assert vocab.pretokenizer_attributes["normalize"] is False

    def test_normalize_roundtrips_via_json(self, tmp_path):
        trainer = BPETrainer(
            vocab_size=200,
            min_frequency=1,
            pre_tokenizer=GraphemePreTokenizer(normalize=False),
            show_progress=0,
        )
        vocab = trainer.train(["hello world"])
        p = tmp_path / "vocab.json"
        vocab.save(p)
        loaded = Vocab.load(p)
        assert loaded.pretokenizer_attributes["normalize"] is False

    def test_resolved_pretokenizer_inherits_normalize(self, tmp_path):
        # After save/load the resolved pre-tokenizer must carry the correct flag.
        trainer = BPETrainer(
            vocab_size=200,
            min_frequency=1,
            pre_tokenizer=GraphemePreTokenizer(normalize=False),
            show_progress=0,
        )
        vocab = trainer.train(["hello world"])
        p = tmp_path / "vocab.json"
        vocab.save(p)
        tok = BPETokenizer.from_file(p)
        assert tok.pre_tokenizer.normalize is False

    def test_nfd_input_encodes_same_ids_as_nfc(self):
        # End-to-end: train on NFC, encode NFD — must produce identical ids.
        corpus = ["caf\u00e9 au lait", "caf\u00e9 noir"]
        trainer = BPETrainer(
            vocab_size=200,
            min_frequency=1,
            pre_tokenizer=GraphemePreTokenizer(normalize=True),
            show_progress=0,
        )
        vocab = trainer.train(corpus)
        tok = BPETokenizer(vocab)
        nfc_ids = tok.encode("caf\u00e9 au lait")
        nfd_ids = tok.encode("cafe\u0301 au lait")
        assert nfc_ids == nfd_ids


class TestVocab:

    def test_add_and_lookup(self):
        v = Vocab()
        idx = v.add_token("කි")
        assert v.token_to_id("කි") == idx
        assert v.id_to_token(idx) == "කි"

    def test_contains(self):
        v = Vocab()
        v.add_token("ත්")
        assert "ත්" in v
        assert "xyz" not in v

    def test_special_tokens_have_lowest_ids(self, grapheme_vocab):
        assert grapheme_vocab.token_to_id(Vocab.SPECIAL_UNK) == 0
        assert grapheme_vocab.token_to_id(Vocab.SPECIAL_PAD) == 1
        assert grapheme_vocab.token_to_id(Vocab.SPECIAL_BOS) == 2
        assert grapheme_vocab.token_to_id(Vocab.SPECIAL_EOS) == 3

    def test_pretokenizer_name_set_after_training(self, grapheme_vocab):
        assert grapheme_vocab.pretokenizer_name == "GraphemePreTokenizer"

    def test_pretokenizer_attributes_empty(self, grapheme_vocab):
        assert grapheme_vocab.pretokenizer_attributes == {"normalize": False}

    def test_repr(self, grapheme_vocab):
        r = repr(grapheme_vocab)
        assert r.startswith("Vocab(")
        assert "size=" in r
        assert "merges=" in r

    def test_tokens_slice(self, grapheme_vocab):
        pairs = grapheme_vocab.tokens(0, 4)
        assert pairs[0] == (Vocab.SPECIAL_UNK, 0)
        assert len(pairs) == 4

    def test_tokens_default_full(self, grapheme_vocab):
        all_tokens = grapheme_vocab.tokens()
        assert len(all_tokens) == len(grapheme_vocab)

    def test_save_load_roundtrip(self, grapheme_vocab, tmp_path):
        p = tmp_path / "vocab.json"
        grapheme_vocab.save(p)
        loaded = Vocab.load(p)
        assert len(loaded) == len(grapheme_vocab)
        assert loaded.merges == grapheme_vocab.merges
        assert loaded.pretokenizer_name == grapheme_vocab.pretokenizer_name

    def test_save_includes_pretokenizer_block(self, grapheme_vocab, tmp_path):
        p = tmp_path / "vocab.json"
        grapheme_vocab.save(p)
        raw = json.loads(p.read_text(encoding="utf-8"))
        assert "pretokenizer" in raw
        assert raw["pretokenizer"]["name"] == "GraphemePreTokenizer"
        assert "attributes" in raw["pretokenizer"]

    def test_load_missing_pretokenizer_defaults(self, tmp_path):
        # Old-format files without a pretokenizer block default to GraphemePreTokenizer
        p = tmp_path / "old.json"
        p.write_text(json.dumps({"vocab": {}, "merges": []}))
        v = Vocab.load(p)
        assert v.pretokenizer_name == "GraphemePreTokenizer"

    def test_save_blocks_overwrite_in_vocabs_dir(self, grapheme_vocab, tmp_path):
        # Saving a new name to vocabs/ is fine; overwriting it is blocked
        grapheme_vocab.save("_test_tmp")
        with pytest.raises(FileExistsError):
            grapheme_vocab.save("_test_tmp")
        # Cleanup
        from akuru_token.vocab import _VOCABS_DIR

        (_VOCABS_DIR / "_test_tmp.json").unlink()

    def test_list_builtin_returns_list(self):
        builtins = Vocab.list_vocabs()
        assert isinstance(builtins, list)
        assert "sin_eng" in builtins


class TestBPETrainer:

    def test_vocab_larger_than_base_symbols(self, grapheme_vocab):
        assert len(grapheme_vocab) > 20

    def test_merges_recorded(self, grapheme_vocab):
        assert len(grapheme_vocab.merges) > 0

    def test_merges_are_pairs(self, grapheme_vocab):
        for pair in grapheme_vocab.merges:
            assert len(pair) == 2

    def test_sinhala_graphemes_in_vocab(self, grapheme_vocab):
        # Base graphemes from the corpus must all be in the vocab
        for word in SINHALA_WORDS[:5]:
            for grapheme in split_graphemes(word):
                assert grapheme in grapheme_vocab, f"{grapheme!r} missing from vocab"

    def test_common_sinhala_merges_learned(self, grapheme_vocab):
        # ශ්‍රී appears multiple times. Its graphemes should have been merged
        merged_tokens = [a + b for a, b in grapheme_vocab.merges]
        # At least some Sinhala multi-grapheme tokens should exist
        sinhala_merges = [
            t for t in merged_tokens if any("\u0d80" <= ch <= "\u0dff" for ch in t)
        ]
        assert len(sinhala_merges) > 0


class TestBPETokenizer:

    def test_pretokenizer_auto_resolved(self, grapheme_tokenizer):
        assert isinstance(grapheme_tokenizer.pre_tokenizer, GraphemePreTokenizer)

    def test_encode_returns_ids(self, grapheme_tokenizer):
        ids = grapheme_tokenizer.encode("ශ්‍රී ලංකාව")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_returns_tokens(self, grapheme_tokenizer):
        tokens = grapheme_tokenizer.encode("ශ්‍රී ලංකාව", as_ids=False)
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_sinhala_roundtrip(self, grapheme_tokenizer):
        # Every word in the training corpus must roundtrip perfectly
        for word in SINHALA_WORDS[:10]:
            ids = grapheme_tokenizer.encode(word)
            decoded = grapheme_tokenizer.decode(ids)
            assert decoded == word, f"Roundtrip failed for {word!r}: got {decoded!r}"

    def test_mixed_script_roundtrip(self, grapheme_tokenizer):
        text = "ශ්‍රී ලංකාව is beautiful"
        ids = grapheme_tokenizer.encode(text)
        decoded = grapheme_tokenizer.decode(ids)
        assert decoded == text

    def test_full_sentence_roundtrip(self, grapheme_tokenizer):
        ids = grapheme_tokenizer.encode(SINHALA_SENTENCE)
        decoded = grapheme_tokenizer.decode(ids)
        assert decoded == SINHALA_SENTENCE

    def test_bos_eos(self, grapheme_vocab):
        tok = BPETokenizer(grapheme_vocab, add_bos=True, add_eos=True)
        tokens = tok.encode("ශ්‍රී", as_ids=False)
        assert tokens[0] == Vocab.SPECIAL_BOS
        assert tokens[-1] == Vocab.SPECIAL_EOS

    def test_encode_batch(self, grapheme_tokenizer):
        texts = ["ශ්‍රී ලංකාව", "මහනුවර", "hello"]
        batch = grapheme_tokenizer.encode_batch(texts)
        assert len(batch) == 3
        assert all(isinstance(ids, list) for ids in batch)

    def test_decode_batch(self, grapheme_tokenizer):
        texts = ["ශ්‍රී ලංකාව", "මහනුවර"]
        batch_ids = grapheme_tokenizer.encode_batch(texts)
        decoded = grapheme_tokenizer.decode_batch(batch_ids)
        assert decoded == texts

    def test_skip_special_tokens(self, grapheme_vocab):
        tok = BPETokenizer(grapheme_vocab, add_bos=True, add_eos=True)
        ids = tok.encode("ශ්‍රී")
        decoded_skip = tok.decode(ids, skip_special_tokens=True)
        decoded_keep = tok.decode(ids, skip_special_tokens=False)
        assert Vocab.SPECIAL_BOS not in decoded_skip
        assert Vocab.SPECIAL_BOS in decoded_keep

    def test_from_file_auto_resolves_pretokenizer(self, grapheme_vocab, tmp_path):
        p = tmp_path / "vocab.json"
        grapheme_vocab.save(p)
        tok = BPETokenizer.from_file(p)
        assert isinstance(tok.pre_tokenizer, GraphemePreTokenizer)

    def test_from_file_encode_matches_original(self, grapheme_tokenizer, tmp_path):
        p = tmp_path / "vocab.json"
        grapheme_tokenizer.vocab.save(p)
        tok2 = BPETokenizer.from_file(p)
        assert tok2.encode("ශ්‍රී ලංකාව") == grapheme_tokenizer.encode("ශ්‍රී ලංකාව")

    def test_unknown_pretokenizer_raises(self):
        v = Vocab()
        v.pretokenizer_name = "DoesNotExist"
        with pytest.raises(ValueError, match="Unknown pretokenizer"):
            BPETokenizer(v)
            