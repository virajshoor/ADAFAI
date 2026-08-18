"""Google DeepMind SynthID-Text detector, via Hugging Face `transformers`.

SynthID-Text (Dathathri et al., Nature 2024) biases sampling with
"Tournament Sampling": each position's token competes across k rounds
of pseudorandom scoring functions seeded by the secret keys + preceding
n-gram. Detection is a Bayesian classifier over how often the observed
tokens win their tournaments versus chance.

This wraps transformers>=4.46's official SynthIDTextWatermarkDetector.
It requires:
  1. the same `keys` / `ngram_len` used at generation time, and
  2. a BayesianDetectorModel calibrated on watermarked vs. unwatermarked
     samples from that same model+keys (see transformers'
     examples/research_projects/synthid_text/detector_training.py).

Usable for self-hosted open-weight models you watermark yourself (e.g.
Gemma via SynthIDTextWatermarkingConfig). Google does not publish the
keys used in production Gemini, so this cannot detect Gemini output
directly - see watermark/providers.py.
"""


def detect_synthid(
    text: str,
    tokenizer_name: str,
    detector_model_path: str,
    keys: list[int],
    ngram_len: int = 5,
) -> dict:
    try:
        from transformers import (
            AutoTokenizer,
            BayesianDetectorModel,
            SynthIDTextWatermarkDetector,
            SynthIDTextWatermarkingConfig,
        )
    except ImportError as e:
        raise ImportError(
            "SynthID detection needs a recent transformers: "
            "pip install -U transformers"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    detector_model = BayesianDetectorModel.from_pretrained(detector_model_path)
    watermarking_config = SynthIDTextWatermarkingConfig(keys=keys, ngram_len=ngram_len)

    detector = SynthIDTextWatermarkDetector(
        detector_module=detector_model,
        logits_processor=None,
        tokenizer=tokenizer,
    )

    ids = tokenizer([text], return_tensors="pt", padding=True)["input_ids"]
    result = detector(ids)
    prob = float(result[0]) if hasattr(result, "__len__") else float(result)

    return {
        "watermark_probability": round(prob, 4),
        "is_watermarked": prob > 0.5,
    }
