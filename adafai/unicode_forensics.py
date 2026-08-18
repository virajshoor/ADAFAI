"""Character-level provenance forensics. Zero dependencies.

Unlike every other signal in ADAFAI, this one is not stylistic - it
looks for artifacts of *how the text was produced and transported*.
That makes it low-recall but high-precision: most AI text carries none
of these marks, but human keyboards very rarely produce them either, so
a hit is strong evidence rather than a hint.

Two distinct things get reported, and they should not be conflated:

  1. INVISIBLE CHARACTERS (zero-width space/joiner, BOM, narrow no-break
     space, soft hyphen). Researchers have repeatedly observed these in
     output from recent hosted models; providers describe them as
     training/serving artifacts rather than deliberate marks. They are
     also the classic way a *third party* tags text. Either way their
     presence is anomalous and worth surfacing verbatim.

  2. TYPOGRAPHIC POLISH (curly quotes, em dashes, ellipsis character).
     Hosted assistants emit these far more consistently than people
     typing on a keyboard, where straight quotes dominate. This is
     suggestive, NOT probative: word processors and publishing tools
     autocorrect straight quotes into curly ones, so edited human prose
     trips it constantly. Weighted accordingly.

Deliberately NOT a watermark detector. Real watermarks (Google SynthID,
Anthropic's) live in token *choice*, not in stray codepoints - see
adafai/watermark/. Anything found here is an artifact, not a decoded mark.
"""
import re
import unicodedata

# Codepoints that carry no visible glyph. Presence in ordinary prose is anomalous.
INVISIBLE = {
    "​": "ZERO WIDTH SPACE",
    "‌": "ZERO WIDTH NON-JOINER",
    "‍": "ZERO WIDTH JOINER",
    "⁠": "WORD JOINER",
    "﻿": "ZERO WIDTH NO-BREAK SPACE (BOM)",
    "­": "SOFT HYPHEN",
    "᠎": "MONGOLIAN VOWEL SEPARATOR",
    "؜": "ARABIC LETTER MARK",
    "‎": "LEFT-TO-RIGHT MARK",
    "‏": "RIGHT-TO-LEFT MARK",
}

# Unusual-but-visible spaces. Rarely typed by hand; common in model output.
EXOTIC_SPACES = {
    " ": "NO-BREAK SPACE",
    " ": "NARROW NO-BREAK SPACE",
    " ": "THIN SPACE",
    " ": "EN SPACE",
    " ": "EM SPACE",
    " ": "FIGURE SPACE",
}

# Latin lookalikes from other scripts - the standard evasion trick for
# defeating statistical detectors, so finding them suggests tampering.
HOMOGLYPHS = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "а": "a", "е": "e", "о": "o", "р": "p",
    "с": "c", "х": "x", "у": "y", "Α": "A", "Β": "B",
    "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _find(text: str, table: dict) -> dict:
    hits = {}
    for ch, name in table.items():
        n = text.count(ch)
        if n:
            hits[name] = {"codepoint": f"U+{ord(ch):04X}", "count": n}
    return hits


def analyze_unicode(text: str) -> dict:
    """Character-level provenance report.

    `score` is an AI-likelihood in [0, 1] for ensemble use, but read
    `invisible_characters` and `homoglyphs` directly - a single hit
    there means far more than the blended number does.
    """
    words = _WORD_RE.findall(text)
    word_count = len(words)
    if not text:
        return {
            "word_count": 0, "invisible_characters": {}, "exotic_spaces": {},
            "homoglyphs": {}, "typography": {}, "flags": [], "score": 0.0,
        }

    invisible = _find(text, INVISIBLE)
    spaces = _find(text, EXOTIC_SPACES)
    homoglyphs = _find(text, HOMOGLYPHS)

    curly = sum(text.count(c) for c in "‘’“”")
    straight = text.count("'") + text.count('"')
    em_dash = text.count("—")
    en_dash = text.count("–")
    ellipsis = text.count("…")
    total_quotes = curly + straight

    typography = {
        "curly_quotes": curly,
        "straight_quotes": straight,
        "curly_quote_ratio": round(curly / total_quotes, 4) if total_quotes else None,
        "em_dashes": em_dash,
        "en_dashes": en_dash,
        "ellipsis_char": ellipsis,
        "em_dashes_per_1000w": round(em_dash * 1000 / word_count, 2) if word_count else 0.0,
    }

    flags = []
    if invisible:
        flags.append("invisible characters present - anomalous in typed text")
    if homoglyphs:
        flags.append("mixed-script homoglyphs - possible detector evasion")
    if spaces:
        flags.append("exotic space characters present")
    if not text.isprintable() and "\n" not in text and "\t" not in text:
        flags.append("non-printable characters present")

    # High-precision terms dominate; typography contributes only a little
    # because word processors autocorrect human text into the same shape.
    invisible_count = sum(v["count"] for v in invisible.values())
    invisible_term = min(invisible_count / 3.0, 1.0) if invisible else 0.0
    homoglyph_term = 1.0 if homoglyphs else 0.0
    space_term = min(sum(v["count"] for v in spaces.values()) / 5.0, 1.0) if spaces else 0.0

    typo_term = 0.0
    if total_quotes >= 4 and curly / total_quotes > 0.9:
        typo_term += 0.5
    if word_count >= 100 and em_dash * 1000 / word_count > 8:
        typo_term += 0.5
    typo_term = min(typo_term, 1.0)

    score = min(
        0.45 * invisible_term + 0.30 * homoglyph_term + 0.10 * space_term + 0.15 * typo_term,
        1.0,
    )

    return {
        "word_count": word_count,
        "invisible_characters": invisible,
        "exotic_spaces": spaces,
        "homoglyphs": homoglyphs,
        "typography": typography,
        "scripts": sorted({unicodedata.name(c, "?").split()[0] for c in text if c.isalpha()}),
        "flags": flags,
        "score": round(score, 4),
    }
