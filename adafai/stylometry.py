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

# Sentence-boundary protection. A naive [.!?] split corrupts the burstiness
# signal (CV) on ordinary prose: "Dr." and "3.14" are not boundaries. The
# periods below are swapped for a placeholder before splitting and restored
# after. Rule-based only - no trained tokenizer, no corpus assumptions.
# Trade-off: a sentence-final abbreviation followed by a lowercase word
# ("...and so on etc. the next day") is merged into one sentence. That is the
# conservative direction for this tool: fewer false boundaries.
_PLACEHOLDER = "\uE000"  # private-use area, 1:1 length-preserving
_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|St|vs|Jr|Sr|Prof|cf|etc|Inc|Ltd|No|Fig|Eq|al|e\.g|i\.e)\."
)
_INITIAL_RE = re.compile(r"\b[A-Z]\.")          # "J. R. R. Tolkien"
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")    # "3.14"

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


def _protect(text: str) -> str:
    text = _DECIMAL_RE.sub(_PLACEHOLDER, text)
    text = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)
    return _INITIAL_RE.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, sentence) tuples. Offsets index into the original text -
    the placeholder swap is 1:1, so protection never moves a character."""
    protected = _protect(text)
    return [
        (m.start(), m.end(), m.group(0).replace(_PLACEHOLDER, ".").strip())
        for m in _SENT_RE.finditer(protected)
        if m.group(0).strip()
    ]


def _sentences(text: str) -> list[str]:
    return [s for _, _, s in _sentence_spans(text)]


def mattr(words: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio: lexical diversity, length-invariant.

    Sliding distinct-count, O(n): results are identical to recomputing
    len(set(chunk)) per window, but a book-length input no longer costs
    O(n * window).
    """
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(words)) / n
    counts: dict[str, int] = {}
    for w in words[:window]:
        counts[w] = counts.get(w, 0) + 1
    distinct = len(counts)
    total = distinct
    for i in range(window, n):
        old = words[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            distinct -= 1
        new = words[i]
        if counts.get(new, 0) == 0:
            distinct += 1
        counts[new] = counts.get(new, 0) + 1
        total += distinct
    return total / ((n - window + 1) * window)


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
