"""assert-based self-check, no test framework required: `python tests/test_discourse.py`

The two fixtures under tests/fixtures/ are a real false-negative report: a
machine-written essay and the same essay after a "humanizer" rewrite. Both
were cleared as "likely human" by the surface signals alone; the discourse
signal exists so the rhetorical scaffold the rewrite left behind is scored.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adafai import analyze, analyze_discourse
from adafai.discourse import (
    negation_pivots,
    paragraph_length_cv,
    punch_line_ratio,
    self_qa_count,
)
from adafai.stylometry import _sentences

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ORIGINAL_ESSAY = (FIXTURES / "ai_essay_original.txt").read_text(encoding="utf-8")
HUMANIZED_ESSAY = (FIXTURES / "ai_essay_humanized.txt").read_text(encoding="utf-8")

# Casual multi-paragraph human prose. Uses two of the tics on purpose (a
# self-answered question, short paragraph endings): one habit alone must not
# move the verdict.
HUMAN_MULTI = (
    "update on the shed roof: it's leaking again. of course it is.\n\n"
    "I climbed up there saturday with the tar patch stuff and honestly I think "
    "I made it worse. There's a whole section near the gutter where the plywood "
    "feels spongy, which dan says means the rot goes deeper than the shingles. "
    "He offered to come look next weekend but he's said that before.\n\n"
    "So now I'm watching youtube videos about roof repair at 1am like that's "
    "going to help. The quotes I got last year were all around 4k which we "
    "absolutely do not have. Might just let it leak into the bucket for another "
    "winter and deal with it in spring. that's a problem for future me.\n\n"
    "Anyway. The tomatoes are doing great at least. Deb came by sunday with her "
    "ladder and we just sat on the porch instead, drinking her homemade lemonade "
    "and pretending the roof doesn't exist. honestly? best afternoon in weeks."
)


def test_empty_text_does_not_crash():
    result = analyze_discourse("")
    assert result["score"] == 0.0
    assert result["punch_line_ratio"] is None
    assert result["paragraph_length_cv"] is None


def test_negation_pivot_patterns():
    assert negation_pivots("This is not merely good. It is great.") == ["not merely"]
    assert negation_pivots("The question isn't whether it works.")
    assert negation_pivots("It isn't follow-through, but it is a start.")
    assert negation_pivots("It is not noise; it is signal.")
    assert negation_pivots("The goal is not speed but rather correctness.")
    # a lone human contrast is fine - presence is not the signal, density is
    assert negation_pivots("I don't like olives. He does.") == []


def test_self_qa_count():
    assert self_qa_count(_sentences("Slightly creepy? Sure. Moving on now.")) == 1
    assert self_qa_count(_sentences("Was it worth it? Absolutely.")) == 1
    long_answer = "Was it worth it? " + " ".join(["word"] * 20) + "."
    assert self_qa_count(_sentences(long_answer)) == 0
    assert self_qa_count(_sentences("No question here. Just statements.")) == 0


def test_paragraph_features_need_three_paragraphs():
    two = "First paragraph here. With two sentences.\n\nSecond one. Also two."
    assert punch_line_ratio([p for p in two.split("\n\n")]) is None
    assert paragraph_length_cv([p for p in two.split("\n\n")]) is None


def test_punch_line_ratio_ignores_single_sentence_paragraphs():
    paras = [
        "A long enough opening sentence to stand alone as its own paragraph.",
        "Two sentences in this one. Short zinger.",
        "Again two sentences here. Another zinger.",
        "And once more two sentences. This final sentence is far too long to be a punch-line.",
    ]
    assert punch_line_ratio(paras) == 2 / 3


def test_original_essay_is_not_cleared_as_human():
    result = analyze(ORIGINAL_ESSAY)
    d = result["signals"]["discourse"]
    assert result["verdict"] != "likely human", result["score"]
    assert d["negation_pivots_per_1000w"] >= 5
    assert d["tricolons_per_1000w"] >= 8


def test_humanized_rewrite_is_not_cleared_as_human():
    """Regression: the rewrite passed every surface signal (MATTR 0.87,
    sentence-CV 0.67, zero tell-phrases) and scored 0.02 - 'likely human'."""
    result = analyze(HUMANIZED_ESSAY)
    d = result["signals"]["discourse"]
    assert result["verdict"] != "likely human", result["score"]
    assert d["score"] > result["signals"]["stylometry"]["score"]
    assert d["self_qa_per_1000w"] > 0
    assert d["negation_pivots_per_1000w"] > 0
    assert d["punch_line_ratio"] is not None and d["punch_line_ratio"] >= 0.5


def test_human_control_stays_likely_human():
    result = analyze(HUMAN_MULTI)
    assert result["verdict"] == "likely human", result["score"]


if __name__ == "__main__":
    test_empty_text_does_not_crash()
    test_negation_pivot_patterns()
    test_self_qa_count()
    test_paragraph_features_need_three_paragraphs()
    test_punch_line_ratio_ignores_single_sentence_paragraphs()
    test_original_essay_is_not_cleared_as_human()
    test_humanized_rewrite_is_not_cleared_as_human()
    test_human_control_stays_likely_human()
    print("ok")
