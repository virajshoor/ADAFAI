"""assert-based self-check, no test framework required: `python tests/test_stylometry.py`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adafai.stylometry import analyze_stylometry

AI_LIKE = (
    "In today's world, it is important to note that technology plays a pivotal role. "
    "Furthermore, this cutting-edge approach will delve into the intricate landscape. "
    "Moreover, we must underscore the robust, holistic paradigm shift. "
    "In conclusion, this rich tapestry stands as a testament to innovation."
)

HUMAN_LIKE = (
    "so I tried fixing the sink again lol. didn't work. of course. "
    "Called a plumber, guy showed up 3 hrs late but fixed it in like 5 min?? "
    "cost me $180 which is insane tbh but whatever, it's done. "
    "anyway dinner was good, made the pasta thing from last week again."
)


def test_directionality():
    ai_result = analyze_stylometry(AI_LIKE)
    human_result = analyze_stylometry(HUMAN_LIKE)
    assert ai_result["score"] > human_result["score"], (ai_result, human_result)
    assert ai_result["ai_phrase_density_per_1000w"] > 0


def test_empty_text_does_not_crash():
    result = analyze_stylometry("")
    assert result["score"] == 0.0
    assert result["word_count"] == 0


if __name__ == "__main__":
    test_directionality()
    test_empty_text_does_not_crash()
    print("ok")
