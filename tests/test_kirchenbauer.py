"""Watermark interop self-check: `python tests/test_kirchenbauer.py`

The important test is test_matches_reference_implementation: it builds a
watermarked token sequence using *transformers' own*
WatermarkLogitsProcessor green lists, then checks ADAFAI's independent
detector flags it. That proves PRNG interop, which a self-referential
test cannot.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from adafai.watermark.kirchenbauer import DEFAULT_HASH_KEY, GreenList, detect_green_red

VOCAB = 5000
GAMMA = 0.25


def _reference_processor(seeding_scheme="lefthash", context_width=1):
    from transformers.generation.logits_process import WatermarkLogitsProcessor

    return WatermarkLogitsProcessor(
        vocab_size=VOCAB,
        device="cpu",
        greenlist_ratio=GAMMA,
        hashing_key=DEFAULT_HASH_KEY,
        seeding_scheme=seeding_scheme,
        context_width=context_width,
    )


def test_matches_reference_implementation():
    """ADAFAI's green list must be identical to transformers' green list."""
    ref = _reference_processor()
    ours = GreenList(VOCAB, gamma=GAMMA, hash_key=DEFAULT_HASH_KEY)

    for prev_token in [1, 7, 42, 999, 4321]:
        context = torch.tensor([prev_token], dtype=torch.long)
        ref_ids = set(ref._get_greenlist_ids(context).tolist())
        our_ids = set(ours.ids(context).tolist())
        assert ref_ids == our_ids, f"green list mismatch at prev_token={prev_token}"


def test_selfhash_matches_reference():
    ref = _reference_processor(seeding_scheme="selfhash", context_width=2)
    ours = GreenList(
        VOCAB, gamma=GAMMA, hash_key=DEFAULT_HASH_KEY,
        seeding_scheme="selfhash", context_width=2,
    )
    for ctx in ([5, 9], [100, 2000], [3, 4321]):
        context = torch.tensor(ctx, dtype=torch.long)
        assert set(ref._get_greenlist_ids(context).tolist()) == set(ours.ids(context).tolist())


def _make_watermarked_sequence(n=300, seed=0):
    """Emit tokens that are always green under the REFERENCE green list."""
    ref = _reference_processor()
    g = torch.Generator().manual_seed(seed)
    ids = [11]
    for _ in range(n):
        context = torch.tensor([ids[-1]], dtype=torch.long)
        greens = ref._get_greenlist_ids(context)
        ids.append(int(greens[torch.randint(len(greens), (1,), generator=g)].item()))
    return ids


def test_detects_reference_watermarked_text():
    ids = _make_watermarked_sequence()
    result = detect_green_red(ids, hash_key=DEFAULT_HASH_KEY, vocab_size=VOCAB, gamma=GAMMA)
    assert result["is_watermarked"] is True, result
    assert result["z_score"] > 4.0, result
    assert result["p_value"] < 1e-4, result


def test_wrong_key_does_not_fire():
    ids = _make_watermarked_sequence()
    result = detect_green_red(ids, hash_key=DEFAULT_HASH_KEY + 1, vocab_size=VOCAB, gamma=GAMMA)
    assert result["is_watermarked"] is False, result


def test_unwatermarked_text_does_not_fire():
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, VOCAB, (400,), generator=g).tolist()
    result = detect_green_red(ids, hash_key=DEFAULT_HASH_KEY, vocab_size=VOCAB, gamma=GAMMA)
    assert result["is_watermarked"] is False, result


def test_short_text_abstains():
    result = detect_green_red([5], vocab_size=VOCAB)
    assert result["is_watermarked"] is False
    assert result["num_tokens_scored"] == 0


if __name__ == "__main__":
    for fn in [
        test_matches_reference_implementation,
        test_selfhash_matches_reference,
        test_detects_reference_watermarked_text,
        test_wrong_key_does_not_fire,
        test_unwatermarked_text_does_not_fire,
        test_short_text_abstains,
    ]:
        fn()
        print(f"  ok  {fn.__name__}")
    print("ok")
