"""Ensemble: combines every available signal into one verdict.

Always-on signals (stylometry, unicode forensics) are pure standard
library. Perplexity and Binoculars need torch+transformers and run only
with use_models=True.

The weights below are a documented heuristic, NOT fit on labeled data.
Treat the score as triage, not proof - see README "Limitations". The
evaluation harness needed to replace these with fitted weights is the
next milestone on the roadmap.

Watermark checks live in adafai.watermark and are deliberately separate:
a green-list z-test is a cryptographic result, not a statistical hint,
so blending it into an average would destroy the thing that makes it
valuable.
"""
from adafai.stylometry import analyze_stylometry
from adafai.unicode_forensics import analyze_unicode

# Below this, statistical signals are noise. The literature is
# consistent that short texts are not reliably classifiable.
MIN_WORDS_FOR_VERDICT = 150


def analyze(
    text: str,
    use_models: bool = False,
    perplexity_model: str = "distilgpt2",
    binoculars_observer: str = "distilgpt2",
    binoculars_performer: str = "gpt2",
) -> dict:
    signals = {
        "stylometry": analyze_stylometry(text),
        "unicode": analyze_unicode(text),
    }
    weights = {"stylometry": 0.30, "unicode": 0.10}

    if use_models:
        from adafai.binoculars import analyze_binoculars
        from adafai.perplexity import analyze_perplexity

        signals["perplexity"] = analyze_perplexity(text, model_name=perplexity_model)
        weights["perplexity"] = 0.15

        signals["binoculars"] = analyze_binoculars(
            text, observer_name=binoculars_observer, performer_name=binoculars_performer
        )
        weights["binoculars"] = 0.45

    total = sum(weights.values())
    score = sum(signals[k]["score"] * w for k, w in weights.items()) / total

    word_count = signals["stylometry"]["word_count"]
    if word_count < MIN_WORDS_FOR_VERDICT:
        verdict = "insufficient evidence"
    elif score < 0.30:
        verdict = "likely human"
    elif score < 0.70:
        verdict = "uncertain"
    else:
        verdict = "likely AI-generated"

    result = {
        "score": round(score, 4),
        "verdict": verdict,
        "word_count": word_count,
        "signals": signals,
    }

    if verdict == "insufficient evidence":
        result["note"] = (
            f"{word_count} words is below the {MIN_WORDS_FOR_VERDICT}-word floor "
            "for a statistical verdict; the score is reported but not meaningful."
        )
    else:
        result["note"] = (
            "Heuristic ensemble with unfitted weights - triage signal, not proof. "
            "Never use as sole evidence in an academic-integrity or disciplinary decision."
        )

    # Provenance findings are qualitatively stronger than the blended score.
    if signals["unicode"]["flags"]:
        result["provenance_flags"] = signals["unicode"]["flags"]

    return result
