"""Zero-dependency discourse-level signals: the rhetorical scaffold that
survives rewriting.

adafai.stylometry measures word- and sentence-level surface statistics. A
"humanizer" pass - manual or automated - defeats those by varying sentence
lengths, swapping vocabulary, and deleting known tell-phrases, while leaving
the rhetorical scaffold untouched, because changing *that* means rethinking
the argument rather than rewording it. This module scores the scaffold:

  - negation pivots ("not merely X", "X isn't A, but B", "not X; it is Y",
    "but rather"): the best-documented rhetorical habit of LLM essays
    (Wikipedia:Signs of AI writing calls the family "negative parallelisms")
  - tricolons ("X, Y, and Z"): the rule-of-three list tic
  - self-answered questions ("Slightly creepy? Sure."): the writer asks a
    question and immediately answers it in a few words - fake
    conversational voice
  - paragraph-final punch lines: the share of multi-sentence paragraphs that
    end on a short aphoristic sentence. Humanized AI keeps the
    zinger-per-paragraph cadence; human writers end paragraphs unevenly
  - paragraph-length uniformity: the sentence-length-CV tell one level up,
    exactly where stylistic rewrites don't reach

Every threshold is a documented heuristic in the same style as detector.py;
nothing here is fitted on labeled data. Each sub-signal is weak on its own -
human writers use all of these devices. The signal is the *density* of
several at once, which is why the terms are weighted so that no single habit
can dominate the score. Paragraph-level features need >= MIN_PARAGRAPHS
paragraphs and are reported as None (neutral, zero-weighted) below that -
the conservative direction, matching the rest of the package.
"""
import math
import re

from adafai.stylometry import _sentences, _words

PUNCH_LINE_MAX_WORDS = 10   # a paragraph-final sentence this short reads as a zinger
SELF_QA_MAX_ANSWER_WORDS = 5  # an answer this short to one's own question is the tic
MIN_PARAGRAPHS = 3          # below this, paragraph-level statistics are noise

_NEGATION = r"(?:[a-z']*n't|not)"
NEGATION_PIVOT_RES = [
    re.compile(r"\bnot\s+(?:just|merely|simply|only)\b"),
    re.compile(rf"\b{_NEGATION}\b[^.!?]{{0,60}}?\bbut\b"),
    re.compile(r"\b(?:isn't|is not|aren't|are not|wasn't|was not)\s+whether\b"),
    re.compile(r"\bbut\s+rather\b"),
    re.compile(rf"\b{_NEGATION}\b[^.!?]{{0,60}}?;\s*(?:it|they|this|that|we|he|she)\b"),
]
TRICOLON_RE = re.compile(r"\b\w+[^.!?;:()]{1,60},[^.!?;:()]{1,60},\s*(?:and|or)\b")


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def negation_pivots(text: str) -> list[str]:
    """Matched pivot constructions. Hits are per pattern, not per sentence:
    "not merely a matter of convenience; it has become" matches twice, once
    for each construction it contains."""
    low = text.lower()
    return [m.group(0) for r in NEGATION_PIVOT_RES for m in r.finditer(low)]


def self_qa_count(sentences: list[str]) -> int:
    return sum(
        1 for a, b in zip(sentences, sentences[1:])
        if a.rstrip().endswith("?") and 0 < len(_words(b)) <= SELF_QA_MAX_ANSWER_WORDS
    )


def punch_line_ratio(paragraphs: list[str]) -> float | None:
    """Share of multi-sentence paragraphs ending on a short punch-line."""
    multi = [p for p in paragraphs if len(_sentences(p)) >= 2]
    if len(multi) < MIN_PARAGRAPHS:
        return None
    punched = sum(
        1 for p in multi if len(_words(_sentences(p)[-1])) <= PUNCH_LINE_MAX_WORDS
    )
    return punched / len(multi)


def paragraph_length_cv(paragraphs: list[str]) -> float | None:
    """Coefficient of variation of paragraph word counts - uniformity one
    level above sentence-length CV."""
    if len(paragraphs) < MIN_PARAGRAPHS:
        return None
    lengths = [len(_words(p)) for p in paragraphs]
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return None
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return math.sqrt(var) / mean


def analyze_discourse(text: str) -> dict:
    words = _words(text)
    word_count = len(words)
    if word_count == 0:
        return {
            "word_count": 0, "paragraph_count": 0,
            "negation_pivots_per_1000w": 0.0, "tricolons_per_1000w": 0.0,
            "self_qa_per_1000w": 0.0, "punch_line_ratio": None,
            "paragraph_length_cv": None, "negation_pivot_hits": [],
            "score": 0.0,
        }

    sentences = _sentences(text)
    paragraphs = _paragraphs(text)
    pivots = negation_pivots(text)
    tricolons = TRICOLON_RE.findall(text.lower())
    qas = self_qa_count(sentences)
    punch = punch_line_ratio(paragraphs)
    para_cv = paragraph_length_cv(paragraphs)

    pivots_per_1000 = len(pivots) * 1000 / word_count
    tris_per_1000 = len(tricolons) * 1000 / word_count
    qa_per_1000 = qas * 1000 / word_count

    # Each term clamped to [0, 1] before weighting; higher = more AI-like.
    pivot_term = min(pivots_per_1000 / 5.0, 1.0)   # >=5 pivots/1000w = max
    tri_term = min(tris_per_1000 / 8.0, 1.0)       # >=8 tricolons/1000w = max
    qa_term = min(qa_per_1000 / 3.0, 1.0)          # >=3 self-Q&As/1000w = max
    # <=40% of paragraphs ending on a zinger is unremarkable; >=80% is a cadence
    punch_term = 0.0 if punch is None else min(max((punch - 0.4) / 0.4, 0.0), 1.0)
    # human paragraph lengths are lumpy (CV ~0.5+); model essays are even
    para_cv_term = 0.0 if para_cv is None else max(0.0, (0.5 - para_cv) / 0.5)

    score = (
        0.30 * pivot_term
        + 0.20 * tri_term
        + 0.15 * qa_term
        + 0.15 * punch_term
        + 0.20 * para_cv_term
    )

    return {
        "word_count": word_count,
        "paragraph_count": len(paragraphs),
        "negation_pivots_per_1000w": round(pivots_per_1000, 2),
        "tricolons_per_1000w": round(tris_per_1000, 2),
        "self_qa_per_1000w": round(qa_per_1000, 2),
        "punch_line_ratio": None if punch is None else round(punch, 4),
        "paragraph_length_cv": None if para_cv is None else round(para_cv, 4),
        "negation_pivot_hits": sorted(set(pivots))[:6],
        "score": round(min(max(score, 0.0), 1.0), 4),
    }
