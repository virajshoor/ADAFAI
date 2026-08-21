"""Sentence-level localization: *where* a document looks AI-generated, and why.

A single document-level score hides mixed authorship - a human introduction
wrapped around pasted AI paragraphs averages out to "uncertain" and says
nothing about which part is which. This module scores each sentence so mixed
documents can be read span by span.

Signal directions are the same published ones used by the document-level
ensemble (adafai.stylometry, adafai.unicode_forensics), computed locally:

  - AI-tell phrase hits inside the sentence itself
  - invisible characters / homoglyphs / exotic spaces inside the sentence
    (provenance artifacts - one hit outweighs every statistical term)
  - lexical diversity and trigram repetition over a word window centered on
    the sentence, because a lone sentence is too short for those measures
  - the sentence-visible slice of the discourse signal (adafai.discourse):
    negation pivots, rule-of-three lists, self-answered questions
  - typographic polish (em dash / curly quotes) - weak, as everywhere else

Every threshold is a documented heuristic in the same style as detector.py;
nothing here is fitted on labeled data. Span scores answer "where should a
human look closer", never "this sentence is AI". Sentences under
MIN_WORDS_LOCAL words carry almost no statistical signal and are returned
with low_confidence=True unless a provenance artifact fired.
"""
from bisect import bisect_left

from adafai.discourse import (
    NEGATION_PIVOT_RES,
    SELF_QA_MAX_ANSWER_WORDS,
    TRICOLON_RE,
)
from adafai.stylometry import (
    AI_TELL_PHRASES,
    _WORD_RE,
    _sentence_spans,
    mattr,
    repeated_trigram_ratio,
)
from adafai.unicode_forensics import EXOTIC_SPACES, HOMOGLYPHS, INVISIBLE

MIN_WORDS_LOCAL = 8
CONTEXT_WORDS = 100  # word window centered on the sentence for diversity/repetition

_CURLY_QUOTES = "‘’“”"


def _present(text: str, table: dict) -> list[tuple[str, int]]:
    return [(name, text.count(ch)) for ch, name in table.items() if ch in text]


def analyze_spans(text: str) -> dict:
    """Score each sentence of `text` for AI-likelihood.

    Returns {"sentences": [{start, end, text, word_count, score,
    low_confidence, reasons}], "sentence_count", "flagged", "note"}.
    `start`/`end` index into the original text so callers can highlight
    in place without re-tokenizing.
    """
    spans = _sentence_spans(text)
    word_matches = list(_WORD_RE.finditer(text.lower()))
    words = [m.group(0) for m in word_matches]
    word_starts = [m.start() for m in word_matches]
    half = CONTEXT_WORDS // 2

    sentences = []
    for i, (start, end, sent) in enumerate(spans):
        low = sent.lower()
        n_words = len(_WORD_RE.findall(low))
        reasons = []

        phrase_hits = sorted({p for p in AI_TELL_PHRASES if p in low})
        hits = sum(low.count(p) for p in AI_TELL_PHRASES)

        i0 = bisect_left(word_starts, start)
        i1 = bisect_left(word_starts, end)
        ctx = words[max(0, i0 - half): i1 + half]
        diversity = mattr(ctx)
        rep = repeated_trigram_ratio(ctx)

        invisible = _present(sent, INVISIBLE)
        exotic = _present(sent, EXOTIC_SPACES)
        homoglyphs = _present(sent, HOMOGLYPHS)
        typo = sum(sent.count(c) for c in _CURLY_QUOTES) + sent.count("—")

        # Sentence-visible slice of the discourse signal (adafai.discourse):
        # the rhetorical habits a humanizer pass leaves behind.
        pivots = [m.group(0) for r in NEGATION_PIVOT_RES for m in r.finditer(low)]
        tricolon = bool(TRICOLON_RE.search(low))
        self_qa = (
            i > 0
            and spans[i - 1][2].rstrip().endswith("?")
            and 0 < n_words <= SELF_QA_MAX_ANSWER_WORDS
        )

        phrase_term = min(hits / 2.0, 1.0)          # 2+ tells in one sentence = max
        diversity_term = max(0.0, (0.75 - diversity) / 0.75)   # as document-level
        rep_term = min(rep / 0.15, 1.0)                        # as document-level
        typo_term = 1.0 if typo else 0.0

        score = (
            0.40 * phrase_term
            + 0.20 * diversity_term
            + 0.20 * rep_term
            + 0.10 * typo_term
            + 0.10 * (1.0 if pivots else 0.0)
            + 0.10 * (1.0 if tricolon else 0.0)
            + 0.10 * (1.0 if self_qa else 0.0)
        )
        artifact = False
        if invisible:
            score = max(score, 0.85)
            artifact = True
        if homoglyphs:
            score = max(score, 0.90)
            artifact = True
        if exotic:
            score = min(1.0, score + 0.15)
            artifact = True

        if phrase_hits:
            shown = ", ".join(f"'{p}'" for p in phrase_hits[:4])
            more = f" (+{len(phrase_hits) - 4} more)" if len(phrase_hits) > 4 else ""
            reasons.append(f"AI-tell phrase{'s' if len(phrase_hits) > 1 else ''}: {shown}{more}")
        for name, c in invisible:
            reasons.append(f"invisible character {name} x{c} - anomalous in typed text")
        for name, c in homoglyphs:
            reasons.append(f"lookalike character {name} x{c} - possible detector evasion")
        for name, c in exotic:
            reasons.append(f"exotic space character {name} x{c}")
        if diversity_term > 0 and len(ctx) >= 30:
            reasons.append(
                f"low local lexical diversity (MATTR {diversity:.2f}; human baseline ~0.75+)"
            )
        if rep > 0:
            reasons.append(f"{rep:.1%} of nearby word trigrams repeat")
        if pivots:
            shown = ", ".join(f"'{p}'" for p in sorted(set(pivots))[:2])
            reasons.append(
                f"negation pivot {shown} - 'not X, but Y' rhetoric, a discourse-level AI tell"
            )
        if tricolon:
            reasons.append("rule-of-three list ('X, Y, and Z') - a discourse-level AI tell")
        if self_qa:
            reasons.append("self-answered question - fake conversational voice")
        if typo:
            reasons.append("typographic polish (em dash / curly quotes) - weak signal")
        if not reasons:
            reasons.append("no AI signals in this sentence")

        sentences.append({
            "start": start,
            "end": end,
            "text": sent,
            "word_count": n_words,
            "score": round(min(max(score, 0.0), 1.0), 4),
            "low_confidence": n_words < MIN_WORDS_LOCAL and not artifact,
            "reasons": reasons,
        })

    flagged = sum(1 for s in sentences if s["score"] >= 0.5 and not s["low_confidence"])
    note = None
    if len(words) < 150:
        note = (
            f"{len(words)} words is below the 150-word floor: sentence scores are "
            "shown for navigation only and are not meaningful statistically."
        )

    return {"sentences": sentences, "sentence_count": len(sentences),
            "flagged": flagged, "note": note}
