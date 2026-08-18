"""Perplexity + burstiness (GPTZero's original method, pre-autumn-2023).

Requires torch + transformers. Perplexity alone is a weak, model- and
domain-dependent signal (GPTZero itself dropped it for a fine-tuned
classifier), so the ensemble in detector.py gives it modest weight and
Binoculars (binoculars.py) does the heavy lifting.
"""
import math
import re

_SENT_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")

_model_cache: dict[str, tuple] = {}


def _load(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "perplexity scoring needs torch + transformers: "
            "pip install torch transformers"
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    _model_cache[model_name] = (tokenizer, model, torch)
    return _model_cache[model_name]


def _text_perplexity(text: str, tokenizer, model, torch) -> float:
    if not text.strip():
        return float("nan")
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = ids["input_ids"]
    if input_ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return math.exp(out.loss.item())


def analyze_perplexity(text: str, model_name: str = "distilgpt2") -> dict:
    """Returns whole-text perplexity and per-sentence burstiness.

    Lower perplexity + lower burstiness (std dev of per-sentence
    perplexity) is the historical AI-text signature. Thresholds are
    domain-dependent; calibrate `ai_likelihood` against your own corpus
    before trusting it in isolation.
    """
    tokenizer, model, torch = _load(model_name)

    whole_ppl = _text_perplexity(text, tokenizer, model, torch)

    sentences = [s.strip() for s in _SENT_RE.findall(text) if s.strip()]
    sent_ppls = [_text_perplexity(s, tokenizer, model, torch) for s in sentences]
    sent_ppls = [p for p in sent_ppls if not math.isnan(p)]

    if len(sent_ppls) >= 2:
        mean = sum(sent_ppls) / len(sent_ppls)
        var = sum((p - mean) ** 2 for p in sent_ppls) / len(sent_ppls)
        burstiness = math.sqrt(var)
    else:
        burstiness = float("nan")

    # Heuristic mapping into [0, 1]; low perplexity/burstiness -> AI-like.
    # Centered on typical distilgpt2 ranges for natural English prose.
    ppl_term = 1.0 - min(whole_ppl, 100.0) / 100.0 if not math.isnan(whole_ppl) else 0.5
    burst_term = 1.0 - min(burstiness, 60.0) / 60.0 if not math.isnan(burstiness) else 0.5
    score = 0.5 * ppl_term + 0.5 * burst_term

    return {
        "model": model_name,
        "perplexity": round(whole_ppl, 2) if not math.isnan(whole_ppl) else None,
        "burstiness": round(burstiness, 2) if not math.isnan(burstiness) else None,
        "score": round(min(max(score, 0.0), 1.0), 4),
    }
