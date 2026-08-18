"""Kirchenbauer et al. green/red-list watermark detector.

"A Watermark for Large Language Models" (Kirchenbauer et al., 2023).
At generation time the vocabulary is split per-position into a "green"
list (gamma fraction) and a "red" list, seeded from the preceding
token(s) and a secret key; green logits get a bias +delta, so
watermarked text ends up measurably "too green". Detection needs no
model - only the same tokenizer, key, gamma and seeding scheme.

INTEROP: the green list is defined by a specific PRNG, so a detector
only works if it reproduces the generator's permutation *exactly*. This
implementation matches the reference
(https://github.com/jwkirchenbauer/lm-watermarking, and its port in
`transformers.generation.logits_process.WatermarkLogitsProcessor`):

    seed = hash_key * prev_token          # "lefthash"
    rng.manual_seed(seed % (2**64 - 1))
    greenlist = torch.randperm(vocab_size, generator=rng)[:gamma * vocab_size]

which is why torch is required here. Verified against transformers'
implementation by `tests/test_kirchenbauer.py`.

This only works when you control the key. It cannot detect a closed
provider's watermark - see adafai/watermark/providers.py.
"""
import math

# Reference defaults (transformers WatermarkLogitsProcessor / lm-watermarking).
DEFAULT_HASH_KEY = 15485863
DEFAULT_GAMMA = 0.25
DEFAULT_Z_THRESHOLD = 4.0
_TABLE_SIZE = 1_000_003


def _normal_sf(z: float) -> float:
    """P(Z > z). math.erfc is exact enough; avoids a scipy dependency."""
    return 0.5 * math.erfc(z / math.sqrt(2))


class GreenList:
    """Reproduces the reference green-list PRNG.

    seeding_scheme: "lefthash" (seed from the previous token) or
    "selfhash" (seed also mixes the candidate token, making the
    watermark harder to reverse-engineer). context_width h uses the last
    h tokens as context.
    """

    def __init__(
        self,
        vocab_size: int,
        gamma: float = DEFAULT_GAMMA,
        hash_key: int = DEFAULT_HASH_KEY,
        seeding_scheme: str = "lefthash",
        context_width: int = 1,
        device: str = "cpu",
    ):
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "Watermark detection must reproduce torch's RNG exactly to "
                "interoperate with the reference implementation: pip install torch"
            ) from e
        if seeding_scheme not in ("lefthash", "selfhash"):
            raise ValueError(f"seeding_scheme must be lefthash or selfhash, got {seeding_scheme}")

        self.torch = torch
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.greenlist_size = int(vocab_size * gamma)
        self.hash_key = hash_key
        self.seeding_scheme = seeding_scheme
        self.context_width = context_width
        self.device = device

        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(hash_key)
        self.fixed_table = torch.randperm(_TABLE_SIZE, generator=self.rng, device=device)

    def _set_seed(self, context) -> None:
        context = context[-self.context_width:]
        if self.seeding_scheme == "selfhash":
            a = self.fixed_table[context % _TABLE_SIZE] + 1
            b = self.fixed_table[context[-1] % _TABLE_SIZE] + 1
            seed = (self.hash_key * a * b).min().item()
        else:
            seed = self.hash_key * context[-1].item()
        self.rng.manual_seed(seed % (2**64 - 1))

    def ids(self, context):
        """Green-list token ids for the position following `context`."""
        self._set_seed(context)
        perm = self.torch.randperm(self.vocab_size, generator=self.rng, device=self.device)
        return perm[: self.greenlist_size]

    def contains(self, context, token_id: int) -> bool:
        return bool((self.ids(context) == token_id).any().item())


def detect_green_red(
    token_ids,
    hash_key: int = DEFAULT_HASH_KEY,
    vocab_size: int | None = None,
    gamma: float = DEFAULT_GAMMA,
    seeding_scheme: str = "lefthash",
    context_width: int = 1,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    ignore_repeated_ngrams: bool = True,
) -> dict:
    """One-proportion z-test on the green-token rate.

    token_ids: text tokenized with the SAME tokenizer used at generation.
    hash_key / gamma / seeding_scheme / context_width: must match generation.

    Under the null (no watermark) each token is green with probability
    gamma, so z = (green - gamma*T) / sqrt(T*gamma*(1-gamma)). The paper
    uses z > 4 (p < 3.2e-5).

    ignore_repeated_ngrams de-duplicates repeated context->token pairs,
    which otherwise inflate z on repetitive text (the reference detector
    offers the same option).
    """
    import torch

    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(list(token_ids), dtype=torch.long)
    if vocab_size is None:
        vocab_size = int(token_ids.max().item()) + 1

    if token_ids.numel() <= context_width:
        return {
            "z_score": None, "p_value": None, "is_watermarked": False,
            "num_tokens_scored": 0, "green_hits": 0, "gamma": gamma,
            "reason": "text too short to score",
        }

    green = GreenList(vocab_size, gamma, hash_key, seeding_scheme, context_width)

    seen: set[tuple] = set()
    green_hits = 0
    scored = 0
    for i in range(context_width, token_ids.numel()):
        context = token_ids[i - context_width: i]
        token = int(token_ids[i].item())
        if ignore_repeated_ngrams:
            key = (tuple(context.tolist()), token)
            if key in seen:
                continue
            seen.add(key)
        if green.contains(context, token):
            green_hits += 1
        scored += 1

    if scored == 0:
        return {
            "z_score": None, "p_value": None, "is_watermarked": False,
            "num_tokens_scored": 0, "green_hits": 0, "gamma": gamma,
            "reason": "no scoreable positions",
        }

    denom = math.sqrt(scored * gamma * (1 - gamma))
    z = (green_hits - gamma * scored) / denom if denom > 0 else 0.0

    return {
        "green_hits": green_hits,
        "num_tokens_scored": scored,
        "green_fraction": round(green_hits / scored, 4),
        "gamma": gamma,
        "seeding_scheme": seeding_scheme,
        "z_score": round(z, 4),
        "p_value": _normal_sf(z),
        "is_watermarked": z > z_threshold,
    }
