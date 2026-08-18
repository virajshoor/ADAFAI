"""Zero-dependency stylometric signals.

No model download required, so this runs even when torch/transformers
aren't installed. Directionality of each feature is drawn from published
findings (see README): AI text tends toward lower lexical diversity, more
uniform sentence lengths, more self-repetition, and a cluster of
overused "AI-tell" words (delve, underscore, tapestry, ...).
"""
import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
_PUNCT_CHARS = ".,;:!?—–-\"'()"

# Words/phrases disproportionately overused by GPT-family and Claude-family
# models relative to human baseline corpora (Wikipedia:Signs of AI writing,
# Reuters Institute, Grammarly, Forbes AI-writing-signs coverage, 2024-2026).
AI_TELL_PHRASES = [
    "delve into", "delve", "boasts", "boast", "bolster", "bolstered",
    "underscore", "underscores", "underscoring", "tapestry", "testament",
    "meticulous", "meticulously", "intricate", "intricacies", "interplay",
    "pivotal", "landscape", "realm", "beacon", "cacophony", "myriad",
    "plethora", "holistic", "paradigm shift", "cutting-edge", "game-changer",
    "leverage", "leveraging", "harness", "illuminate", "facilitate",
    "navigate the complexities", "unlock the potential", "seamlessly",
    "in today's world", "in the ever-evolving", "when it comes to",
    "it is important to note", "it's important to note", "in conclusion",
    "in summary", "in essence", "on the other hand", "furthermore",
    "moreover", "notably", "commendable", "robust", "at its core",
    "plays a pivotal role", "stands as a testament", "rich tapestry",
    "as an ai language model", "vibrant",
]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def mattr(words: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio: lexical diversity, length-invariant."""
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(words)) / n
    ratios = []
    for i in range(n - window + 1):
        chunk = words[i:i + window]
        ratios.append(len(set(chunk)) / window)
    return sum(ratios) / len(ratios)


def sentence_length_cv(sentences: list[str]) -> float:
    """Coefficient of variation of sentence lengths (burstiness proxy)."""
    lengths = [len(_words(s)) for s in sentences if _words(s)]
    if len(lengths) < 2:
        return 1.0  # not enough data -> assume human-like (no penalty)
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 1.0
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return math.sqrt(var) / mean


def punctuation_entropy(text: str) -> float:
    counts = Counter(c for c in text if c in _PUNCT_CHARS)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def repeated_trigram_ratio(words: list[str]) -> float:
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    counts = Counter(trigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(trigrams)


def ai_phrase_density(text: str, word_count: int) -> float:
    """AI-tell phrase hits per 1000 words."""
    if word_count == 0:
        return 0.0
    low = text.lower()
    hits = sum(low.count(p) for p in AI_TELL_PHRASES)
    return hits * 1000 / word_count


def analyze_stylometry(text: str) -> dict:
    words = _words(text)
    sentences = _sentences(text)
    word_count = len(words)

    if word_count == 0:
        return {
            "word_count": 0, "sentence_count": 0, "lexical_diversity_mattr": 0.0,
            "sentence_length_cv": 0.0, "repeated_trigram_ratio": 0.0,
            "ai_phrase_density_per_1000w": 0.0, "punctuation_entropy": 0.0,
            "score": 0.0,
        }

    diversity = mattr(words)
    cv = sentence_length_cv(sentences)
    rep_tri = repeated_trigram_ratio(words)
    phrase_density = ai_phrase_density(text, word_count)
    punct_ent = punctuation_entropy(text)

    # Each term clamped to [0, 1] before weighting; higher = more AI-like.
    phrase_term = min(phrase_density / 5.0, 1.0)          # >=5 hits/1000w = max
    cv_term = max(0.0, (0.6 - cv) / 0.6)                   # human CV ~0.6+
    diversity_term = max(0.0, (0.75 - diversity) / 0.75)   # human MATTR ~0.75+
    rep_term = min(rep_tri / 0.15, 1.0)                    # >=15% repeat = max

    score = (
        0.35 * phrase_term
        + 0.25 * cv_term
        + 0.20 * diversity_term
        + 0.20 * rep_term
    )

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "lexical_diversity_mattr": round(diversity, 4),
        "sentence_length_cv": round(cv, 4),
        "repeated_trigram_ratio": round(rep_tri, 4),
        "ai_phrase_density_per_1000w": round(phrase_density, 2),
        "punctuation_entropy": round(punct_ent, 4),
        "score": round(min(max(score, 0.0), 1.0), 4),
    }
