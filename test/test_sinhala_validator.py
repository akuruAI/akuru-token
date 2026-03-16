"""
tests/test_sinhala_validator.py

Sanity checks for sinhala_validator against SLS 1134:2011.
Run with: pytest tests/test_sinhala_validator.py
"""

from scripts.sinhala_validator import find_invalid


# Sanity check — should return None (all valid)
# Covers: vowels (§5.1), vowels+marks (§6.3b), semi-consonants (§5.7, §3.3),
# all 17 vocalic strokes on a consonant (Table 2, precomposed and decomposed),
# pure consonant (§5.3), yansaya combinations (Table 3), rakaaraansaya combinations
# (Table 3), repaya (§5.9), repaya+ya-yansaya (§5.9), conjunct letters (§5.10),
# non-standard letters (§5.5),
# Sinhala Lith Illakkam digits (§3.5.4), kundaliya (§4.1).
_SANITY_CHECK = (
    # §5.1 — all 18 vowels
    "අආඇඈඉඊඋඌඍඎඏඐඑඒඓඔඕඖ "
    # §6.3b — vowels with allowed marks
    "අා අැ අෑ උෟ ඍෘ එ් ඔ් ඔෟ "
    # §5.7 — semi-consonants after vowels
    "අං අඃ "
    # Table 2 — consonant with all vocalic strokes (precomposed and decomposed)
    "ක කා කැ කෑ කි කී කු කූ කෘ කෘෘ කෲ කෟ කෳ "
    "කෙ කේ \u0d9a\u0dd9\u0dca "                      # bare, ේ pre, ේ dec
    "කෛ "
    "කො \u0d9a\u0dd9\u0dcf "                          # ො pre, ො dec
    "කෝ \u0d9a\u0dd9\u0dcf\u0dca "                    # ෝ pre, ෝ dec
    "කෞ \u0d9a\u0dd9\u0ddf "                          # ෞ pre, ෞ dec
    # §5.3 — pure consonant; §5.7 — semi-consonants after consonant forms
    "ක් කං කඃ කාං කාඃ කිං "
    # Table 3 — yansaya (precomposed and decomposed kombuva forms)
    "ක්‍ය ක්‍යා ක්‍යු ක්‍යූ "
    "ක්‍යෙ ක්‍යේ \u0d9a\u0dca\u200d\u0dba\u0dd9\u0dca "
    "ක්‍යො \u0d9a\u0dca\u200d\u0dba\u0dd9\u0dcf ක්‍යෝ "
    # Table 3 — rakaaraansaya (precomposed and decomposed kombuva forms)
    "ක්‍ර ක්‍රා ක්‍රැ ක්‍රෑ ක්‍රි ක්‍රී "
    "ක්‍රෙ ක්‍රේ \u0d9a\u0dca\u200d\u0dbb\u0dd9\u0dca "
    "ක්‍රෛ ක්‍රො \u0d9a\u0dca\u200d\u0dbb\u0dd9\u0dcf ක්‍රෝ ක්‍රෞ "
    # §5.9 — repaya; repaya + ya-yansaya
    "ර්‍ක ර්‍කා ර්‍කේ "
    "ර්‍ය්‍ය "
    # §5.10 — conjunct letters
    "ක්‍ෂ න්‍ද ක්‍ෂා ක්‍ෂේ ක්‍ෂ්‍ර "
    # §5.5 — non-standard letters
    "රැ රෑ රු රූ ළු ළූ "
    # §3.5.4 — Sinhala Lith Illakkam digits; §4.1 — kundaliya
    "෦෧෨෩෪෫෬෭෮෯ ෴"
)

# Invalid combinations — each should return a non-None index.
# One example per failure mode, mirroring the structure of _SANITY_CHECK.
_INVALID_CASES = [
    # Stray marks at cluster start (§5.1, §5.4)
    ("්ක",   "stray al-lakuna at start"),
    ("ාක",   "stray vowel sign at start"),
    ("‍ක",   "stray ZWJ at start"),           # bare ZWJ before consonant

    # Vowel + disallowed mark (§6.3b — only specific vowels accept marks)
    ("ඉා",   "ඉ does not take ා"),
    ("ආා",   "ආ does not take any mark"),
    ("අි",   "අ does not take ි"),

    # Vowel + semi-consonant then another mark (§5.7 — semi-consonant is always last)
    ("අංා",  "mark after semi-consonant"),
    ("අඃං",  "semi-consonant after semi-consonant"),

    # Consonant + double vowel sign (Table 2 — one tail mark only)
    ("කාැ",  "two vowel signs on consonant"),
    ("කිී",  "two is-pillas"),

    # Vowel sign followed by al-lakuna (Tables 2–3)
    ("කා්",  "vowel sign + ්"),
    ("කි්",  "is-pilla + ්"),

    # Kombuva followed by invalid mark (Table 2 rows 12–17)
    ("කෙි",  "kombuva + is-pilla"),

    # Pure consonant followed by semi-consonant (§3.3, §3.5)
    ("ක්ං",  "pure consonant + anusvaraya"),
    ("ක්ඃ",  "pure consonant + visargaya"),

    # Yansaya / rakaaraansaya violations (Tables 2–3)
    ("ක්‍ය්", "yansaya + ්"),
    ("ක්‍ර්", "rakaaraansaya + ්"),
    ("ක්‍රා්","rakaaraansaya + vowel sign + ්"),
    ("ක්‍ය්‍ය","double yansaya"),
    ("ක්‍ර්‍ර","double rakaaraansaya"),

    # Touching letters — not supported
    ("\u0dc3\u200d\u0dca\u0dc3", "touching letters (§5.11, unsupported)"),

    # Reserved / unassigned Sinhala codepoints (§4.3)
    ("\u0d97",  "reserved 0D97"),
    ("\u0dc7",  "reserved 0DC7"),
    ("\u0dd5",  "reserved 0DD5"),
]


def test_sanity_check_all_valid():
    """_SANITY_CHECK covers all valid cluster forms from SLS 1134:2011."""
    assert find_invalid(_SANITY_CHECK) is None


def test_invalid_cases():
    """Every entry in _INVALID_CASES must be rejected."""
    failures = []
    for text, label in _INVALID_CASES:
        if find_invalid(text) is None:
            failures.append(label)
    assert not failures, "Wrongly accepted: " + ", ".join(failures)


def test_valid_pure_ya_ra():
    """Bare ය + ් and ර + ් are valid pure consonants (not yansaya/rakaaraansaya)."""
    assert find_invalid("ය්") is None
    assert find_invalid("ර්") is None