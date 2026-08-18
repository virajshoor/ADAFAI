"""Binoculars (Hans et al., ICML 2024): the current zero-shot SOTA.

>90% detection at a 0.01% false-positive rate on ChatGPT-family text in
the paper, without training on any target-model data. Idea: an
"observer" LM's perplexity on the text, divided by the cross-perplexity
between an "performer" LM's predicted next-token distribution and the
observer's, over the same text. Human text scores high (near 1);
machine text from a source close to the performer model scores low.

Paper defaults to Falcon-7B / Falcon-7B-Instruct with threshold 0.9015.
Lighter model pairs (default here) trade some accuracy for speed and
need their own threshold — see calibrate_threshold().

Reference: https://github.com/ahans30/Binoculars
"""
import math

_pair_cache: dict[tuple, tuple] = {}

DEFAULT_THRESHOLD = 0.9015  # from the paper; recalibrate for other model pairs


def _load_pair(observer_name: str, performer_name: str):
    key = (observer_name, performer_name)
    if key in _pair_cache:
        return _pair_cache[key]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Binoculars needs torch + transformers: pip install torch transformers"
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(observer_name)
    observer = AutoModelForCausalLM.from_pretrained(observer_name)
    performer = AutoModelForCausalLM.from_pretrained(performer_name)
    observer.eval()
    performer.eval()
    _pair_cache[key] = (tokenizer, observer, performer, torch)
    return _pair_cache[key]


def binoculars_score(
    text: str,
    observer_name: str = "distilgpt2",
    performer_name: str = "gpt2",
) -> float | None:
    """Lower = more machine-like. None if text is too short to score."""
    tokenizer, observer, performer, torch = _load_pair(observer_name, performer_name)

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        observer_logits = observer(input_ids).logits[:, :-1, :]
        performer_logits = performer(input_ids).logits[:, :-1, :]
        labels = input_ids[:, 1:]

        observer_logp = torch.log_softmax(observer_logits, dim=-1)
        performer_p = torch.softmax(performer_logits, dim=-1)

        # perplexity(x, observer): observer's surprise at the actual tokens
        token_logp = observer_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ppl = torch.exp(-token_logp.mean())

        # cross-perplexity: how surprised is the observer by the
        # performer's full predicted next-token distribution
        cross_ent_per_pos = -(performer_p * observer_logp).sum(dim=-1)
        x_ppl = torch.exp(cross_ent_per_pos.mean())

    score = (ppl / x_ppl).item()
    return score if not math.isnan(score) else None


def analyze_binoculars(
    text: str,
    observer_name: str = "distilgpt2",
    performer_name: str = "gpt2",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    raw = binoculars_score(text, observer_name, performer_name)
    if raw is None:
        return {"raw_score": None, "threshold": threshold, "score": 0.5}

    # Map raw score to [0, 1] AI-likelihood: raw << threshold -> ~1, raw >> threshold -> ~0.
    # Slope tuned so raw values 0.15 away from threshold saturate.
    score = 1.0 / (1.0 + math.exp((raw - threshold) / 0.05))
    return {
        "observer": observer_name,
        "performer": performer_name,
        "raw_score": round(raw, 4),
        "threshold": threshold,
        "score": round(min(max(score, 0.0), 1.0), 4),
    }
