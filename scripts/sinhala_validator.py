"""
sinhala_validator.py

Validates Sinhala combining character sequences against SLS 1134:2011.
Returns the index of the first invalid codepoint, or None if valid.

Public API
----------
find_invalid(text, strict_conjunct=True)
    Return the index of the first invalid codepoint, or None if the text
    is fully valid.
    If strict_conjunct is True, only conjuncts (except for yansaya and
    rakaranshaya) that are in the mapping are allowed.

line_has_invalid_clusters(line)
    Convenience wrapper; returns True when find_invalid() is not None.
"""

from __future__ import annotations
from typing import Optional
import unicodedata

_C_START = 0x0D9A
_C_END = 0x0DC6
_V_START = 0x0D85
_V_END = 0x0D96

_AL_LAKUNA = 0x0DCA
_ZWJ = 0x200D
_KOMBUVA = 0x0DD9
_ANUSVARAYA = 0x0D82
_VISARGAYA = 0x0D83
_RAYANNA = 0x0DBB

# Valid single tail marks after a consonant cluster
_SINGLE_VOWEL_SIGNS = frozenset(
    {
        0x0DCF,  # ා
        0x0DD0,  # ැ
        0x0DD1,  # ෑ
        0x0DD2,  # ි
        0x0DD3,  # ී
        0x0DD4,  # ු
        0x0DD6,  # ූ
        0x0DD8,  # ෘ  (may double for ෲ)
        0x0DDF,  # ෟ
        0x0DF2,  # ෲ  (precomposed diga gaetta-pilla)
        0x0DF3,  # ෳ
        0x0DDB,  # ෛ  (precomposed kombu deka)
        0x0DDA,  # ේ  (precomposed diga kombuva)
        0x0DDC,  # ො  (precomposed)
        0x0DDD,  # ෝ  (precomposed)
        0x0DDE,  # ෞ  (precomposed)
    }
)

# Valid vowel marks per vowel letter (§6.3b, §5.1).
# Vowels absent from this map accept no combining marks.
_VOWEL_MARK_MAP: dict[int, frozenset[int]] = {
    0x0D85: frozenset({0x0DCF, 0x0DD0, 0x0DD1}),  # අ  → ආ ඇ ඈ
    0x0D8B: frozenset({0x0DDF}),  # උ  → ඌ
    0x0D8D: frozenset({0x0DD8}),  # ඍ  → ඎ  (§6.3b: ඍ + ෘ)
    0x0D91: frozenset({0x0DCA}),  # එ  → ඒ
    0x0D94: frozenset({0x0DCA, 0x0DDF}),  # ඔ  → ඕ ඖ
}

# Sinhala codepoints that are valid as standalone base characters but are
# neither vowel letters, consonants, nor combining marks.  Anything in the
# Sinhala block (0D80–0DFF) that falls outside every other recognised
# category must be in this set, otherwise find_invalid() will flag it as
# an unassigned / reserved codepoint.
_KNOWN_STANDALONE_SINHALA: frozenset[int] = frozenset(
    {
        # Sinhala Lith Illakkam digits (§3.5.4, Table 4)
        *range(0x0DE6, 0x0DF0),  # ෦ – ෯  (0DE6–0DEF)
        # Punctuation
        0x0DF4,  # ෴  kundaliya
    }
)

# All combining Sinhala marks (anything that cannot start a cluster).
# In Unicode logical/storage order (SLS 1134:2011 §5.4) kombuva (0DD9) is a
# *trailing* dependent vowel sign - it follows its consonant in the codepoint
# stream, exactly like ා or ි.  The keyboard "type kombuva before the consonant"
# convention is an input-method concern; a conforming encoder always stores
# cons + 0DD9.  A bare kombuva at cluster-start therefore indicates invalid
# logical order and is rejected here alongside the other stray marks.
_ALL_MARKS = _SINGLE_VOWEL_SIGNS | frozenset(
    {
        _AL_LAKUNA,
        _ANUSVARAYA,
        _VISARGAYA,
        _KOMBUVA,  # trailing dependent vowel sign in logical order (§5.4)
        0x0D81,  # candrabindu
    }
)

# All the valid Sinhala conjunct pairs are mapped here for strict
# conjunct evaluation. In the default strict mode any conjunct in the form
# <consonant 1> + + ් + ZWJ + <consonant 2> where consonant 1 is not
# mapped to consonant 2 in  the mapping, is flagged invalid.
CONJUNCTS_MAP: dict[int, frozenset[int]] = {
    0x0D9A: frozenset({0x0DC0, 0x0DC2}),  # ක  :  ['ව', 'ෂ']
    0x0DAD: frozenset({0x0DC0, 0x0DAE}),  # ත  :  ['ව', 'ථ']
    0x0DB1: frozenset({0x0DC0, 0x0DAE, 0x0DB0, 0x0DAF}),  # න  :  ['ව', 'ථ', 'ධ', 'ද']
    0x0DAF: frozenset({0x0DC0, 0x0DB0}),  # ද  :  ['ව', 'ධ']
    0x0DA7: frozenset({0x0DA8}),  # ට  :  ['ඨ']
    0x0DA4: frozenset({0x0DA0, 0x0DA1, 0x0DA2}),  # ඤ  :  ['ච', 'ඡ', 'ජ']
}

for key, val in CONJUNCTS_MAP.items():
    print(chr(key), " : ", [chr(x) for x in val])


def _is_consonant(cp: int) -> bool:
    return _C_START <= cp <= _C_END and unicodedata.category(chr(cp)) == "Lo"


def _is_vowel_letter(cp: int) -> bool:
    return _V_START <= cp <= _V_END and unicodedata.category(chr(cp)) == "Lo"


def _is_sinhala(cp: int) -> bool:
    return 0x0D80 <= cp <= 0x0DFF or cp == _ZWJ


_YANSAYA_CONS = 0x0DBA  # ය
_RAKAARAANSAYA_CONS = 0x0DBB  # ර


def _parse_consonant_cluster(
    cps: list[int], i: int, n: int, strict_conjunct=True
) -> tuple[int, bool]:
    """
    Parse a consonant cluster in the standard conjunct encoding:

    Conjunct (§5.8-5.10) — repaya, yansaya, rakaaraansaya, and other bandi forms:
        [ ර + ් + ZWJ ] base_consonant [ ් + ZWJ + consonant ]

    Returns (new_i, is_valid).
    """

    # Consume repaya prefix: ර + ් + ZWJ + <base consonant>  (§5.9)
    if (
        i + 3 < n
        and cps[i] == _RAYANNA
        and cps[i + 1] == _AL_LAKUNA
        and cps[i + 2] == _ZWJ
        and _is_consonant(cps[i + 3])
    ):
        i += 4
    else:
        if i >= n or not _is_consonant(cps[i]):
            return i, False
        i += 1

    # Zero or more conjuncts (් + ZWJ + cons) (§5.8, 5.10).
    # Yansaya and rakaaraansaya may appear at most once and
    # only as the final consonant in the chain.
    prev_conjunct_is_terminal = False
    while True:
        if (
            i + 2 < n
            and cps[i] == _AL_LAKUNA
            and cps[i + 1] == _ZWJ
            and _is_consonant(cps[i + 2])
        ):
            # If prev conjunct was ය/ර, no further conjunct is allowed.
            if prev_conjunct_is_terminal:
                return i, False

            consumed_cons = cps[i + 2]
            if consumed_cons in (_YANSAYA_CONS, _RAKAARAANSAYA_CONS):
                prev_conjunct_is_terminal = True
            else:
                prev_conjunct_is_terminal = False

                # Check whether a given consonant is valid by mapping
                if strict_conjunct and (
                    cps[i - 1] not in CONJUNCTS_MAP
                    or cps[i + 2] not in CONJUNCTS_MAP[cps[i - 1]]
                ):
                    print(hex(cps[i]), hex(cps[i + 2]))
                    return i, False

            i += 3
        else:
            break

    return i, True


def _parse_tail_mark(
    cps: list[int],
    i: int,
    n: int,
) -> tuple[int, bool, bool]:
    """
    Parse the optional tail mark after a consonant cluster.
    Returns (new_i, is_valid, is_pure_consonant).

    is_pure_consonant is True when the tail mark is solely ් (hal kirima),
    meaning the cluster is a pure consonant and may not be followed by a
    semi-consonant (SLS 1134:2011 §3.3, §3.5).
    """
    if i >= n:
        return i, True, False

    cp = cps[i]

    # Semi-consonants are not tail marks - leave for _parse_semi_consonant
    if cp in (_ANUSVARAYA, _VISARGAYA):
        return i, True, False

    # ZWJ here is invalid (stray joiner after a complete cluster)
    if cp == _ZWJ:
        return i, False, False

    # Trailing kombuva (0DD9) is valid after a consonant in storage order.
    # e.g. කෙ = 0D9A 0DD9, rakaaraansaya+kombuva = C + ් + ZWJ + ර + 0DD9.
    # It is rejected as a *cluster-starter* (via _ALL_MARKS) but consumed here
    # when it legitimately trails its consonant.
    # Kombuva may be followed by (Table 2, rows 13–17, decomposed keyboard forms):
    #   ්      -diga kombuva ේ  (0DD9 + 0DCA)
    #   ා      -kombuva haa aela-pilla ො  (0DD9 + 0DCF)
    #   ා + ්    -kombuva haa diga aela-pilla ෝ  (0DD9 + 0DCF + 0DCA)
    #   ෟ      -kombuva haa gayanukitta ෞ  (0DD9 + 0DDF)
    # Anything else after kombuva is invalid.
    if cp == _KOMBUVA:
        i += 1
        if i < n and cps[i] == _AL_LAKUNA:
            # ේ decomposed
            i += 1
        elif i < n and cps[i] == 0x0DCF:
            # ො or ෝ decomposed
            i += 1
            if i < n and cps[i] == _AL_LAKUNA:
                i += 1
        elif i < n and cps[i] == 0x0DDF:
            # ෞ decomposed
            i += 1
        elif i < n and cps[i] in _ALL_MARKS:
            # any other mark after kombuva is invalid
            return i, False, False
        return i, True, False

    if cp not in (_ALL_MARKS):
        return i, True, False

    if cp in _SINGLE_VOWEL_SIGNS:
        i += 1
        # ෘ + ෘ = diga gaetta-pilla (keyboard decomposed form, §6.3b)
        if cp == 0x0DD8 and i < n and cps[i] == 0x0DD8:
            i += 1
        # Any further mark after a vowel sign is invalid (Table 2: no such combinations).
        # Semi-consonants are handled separately by _parse_semi_consonant.
        if i < n and cps[i] in _ALL_MARKS and cps[i] not in (_ANUSVARAYA, _VISARGAYA):
            return i, False, False
        return i, True, False

    if cp == _AL_LAKUNA:
        # hal kirima - pure consonant; semi-consonant not allowed after (§3.5)
        i += 1
        return i, True, True

    return i, False, False


def _parse_semi_consonant(cps: list[int], i: int, n: int) -> tuple[int, bool]:
    """
    Consume optional anusvaraya/visargaya. Verify nothing follows in same cluster.
    Returns (new_i, is_valid).
    """
    if i < n and cps[i] in (_ANUSVARAYA, _VISARGAYA):
        i += 1
    # After semi-consonant, no more marks allowed
    if i < n and (cps[i] in _ALL_MARKS or cps[i] == _ZWJ):
        return i, False
    return i, True


def find_invalid(text: str, strict_conjunct=True) -> Optional[int]:
    """
    Return index of first invalid codepoint, or None if text is valid.

    Parameters
    ----------
    text:
        The string to validate.
    """
    cps = [ord(ch) for ch in text]
    i = 0
    n = len(cps)

    while i < n:
        cp = cps[i]

        # Non-Sinhala: pass through
        if not _is_sinhala(cp):
            i += 1
            continue

        # Stray mark at cluster start
        if cp in _ALL_MARKS:
            return i

        # Stray ZWJ at cluster start (not preceded by ් in a conjunct)
        if cp == _ZWJ:
            return i

        #  Vowel letter cluster
        if _is_vowel_letter(cp):
            vowel_cp = cp
            i += 1
            # optional vowel mark (semi-consonants are not vowel marks)
            if (
                i < n
                and cps[i] in _ALL_MARKS
                and cps[i] not in (_ANUSVARAYA, _VISARGAYA)
            ):
                mark_cp = cps[i]
                allowed = _VOWEL_MARK_MAP.get(vowel_cp, frozenset())
                if mark_cp not in allowed:
                    return i
                i += 1
                # no second mark after vowel mark except semi consonant
                if i < n and (
                    (cps[i] in _ALL_MARKS and cps[i] not in (_ANUSVARAYA, _VISARGAYA))
                    or cps[i] == _ZWJ
                ):
                    return i
            # semi-consonant
            i, ok = _parse_semi_consonant(cps, i, n)
            if not ok:
                return i
            continue

        #  Consonant cluster
        if _is_consonant(cp):
            i, ok = _parse_consonant_cluster(cps, i, n, strict_conjunct)
            if not ok:
                return i

            # Lookback: detect conjunct yansaya/rakaaraansaya by checking the
            # last 3 codepoints.a bare ් tail is illegal afterwards and only
            # vowel signs are permitted.
            ends_with_ya_ra = (
                i >= 3
                and cps[i - 3] == _AL_LAKUNA
                and cps[i - 2] == _ZWJ
                and cps[i - 1] in (_YANSAYA_CONS, _RAKAARAANSAYA_CONS)
            )

            i, ok, is_pure = _parse_tail_mark(cps, i, n)

            if ok and is_pure and ends_with_ya_ra:
                return i - 1  # ් after yansaya/rakaaraansaya is invalid (Tables 2–3).
            if not ok:
                return i
            if is_pure and i < n and cps[i] in (_ANUSVARAYA, _VISARGAYA):
                return i  # semi-consonant invalid after pure consonant (§3.5)
            if not is_pure:
                i, ok = _parse_semi_consonant(cps, i, n)
            if not ok:
                return i
            continue

        #  Unknown / reserved Sinhala codepoint
        # The Sinhala block has reserved gaps (e.g. 0D97–0D99, 0DC7–0DC9,
        # 0DD5, 0DD7).  Only explicitly assigned standalone codepoints are
        # allowed here; anything else is flagged.
        if cp in _KNOWN_STANDALONE_SINHALA:
            i += 1
            continue

        return i


def line_has_invalid_clusters(line: str) -> bool:
    return find_invalid(line) is not None
