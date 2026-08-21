"""assert-based self-check, no test framework required: `python tests/test_spans.py`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adafai.spans import analyze_spans

AI_PARA = (
    "In today's world, it is important to note that technology plays a pivotal role. "
    "Furthermore, this cutting-edge approach will delve into the intricate landscape. "
    "Moreover, we must underscore the robust, holistic paradigm shift."
)

HUMAN_PARA = (
    "so I tried fixing the sink again lol. didn't work. of course. "
    "Called a plumber, guy showed up 3 hrs late but fixed it in like 5 min?? "
    "cost me $180 which is insane tbh but whatever, it's done."
)


def test_mixed_document_localizes_ai_section():
    result = analyze_spans(HUMAN_PARA + "\n\n" + AI_PARA + "\n\n" + HUMAN_PARA)
    sents = result["sentences"]
    ai_scores = [s["score"] for s in sents if any(
        p in s["text"].lower() for p in ("pivotal", "delve", "underscore"))]
    human_scores = [s["score"] for s in sents
                    if "plumber" in s["text"] or "pasta" in s["text"] or "sink" in s["text"]]
    assert ai_scores and human_scores
    assert min(ai_scores) > max(human_scores), (ai_scores, human_scores)
    assert result["flagged"] >= 2


def test_reasons_name_the_actual_phrase():
    result = analyze_spans(AI_PARA)
    delve = next(s for s in result["sentences"] if "delve" in s["text"])
    assert any("delve" in r for r in delve["reasons"])


def test_invisible_character_overrides_statistical_score():
    result = analyze_spans("Perfectly ordinary short line.\u200b With a stowaway.")
    hit = next(s for s in result["sentences"]
               if any("ZERO WIDTH SPACE" in r for r in s["reasons"]))
    assert hit["score"] >= 0.85
    assert not hit["low_confidence"]


def test_short_sentence_is_low_confidence():
    result = analyze_spans("Yes. No.")
    assert all(s["low_confidence"] for s in result["sentences"])


def test_offsets_index_original_text():
    text = HUMAN_PARA + "\n\n" + AI_PARA
    for s in analyze_spans(text)["sentences"]:
        assert text[s["start"]:s["end"]].strip().startswith(s["text"][:20])


def test_empty_text_does_not_crash():
    result = analyze_spans("")
    assert result["sentences"] == []
    assert result["flagged"] == 0


if __name__ == "__main__":
    test_mixed_document_localizes_ai_section()
    test_reasons_name_the_actual_phrase()
    test_invisible_character_overrides_statistical_score()
    test_short_sentence_is_low_confidence()
    test_offsets_index_original_text()
    test_empty_text_does_not_crash()
    print("ok")
