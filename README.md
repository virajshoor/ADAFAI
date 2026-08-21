<div align="center">

# ADAFAI

### Automated Detection Algorithm For Artificial Intelligence

**An open-source, multi-signal toolkit for detecting AI-generated text — and for reading the watermarks model providers have started shipping.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#honest-limitations)

</div>

---

## Why another detector

Most AI detectors are a black box that returns a percentage. That percentage gets
used to accuse students of cheating, and nobody — including the vendor — can
explain where it came from.

ADAFAI takes the opposite stance:

- **Every signal is separable and inspectable.** You get each component score, not just a verdict.
- **Watermark detection is kept separate from statistical guessing.** A green-list z-test is a cryptographic result with a real p-value. Blending it into a vibes-based average would throw away the only part that can actually be *proven*.
- **It refuses to answer when it can't.** Under 150 words, you get `insufficient evidence`, not a confident number.
- **The limitations are in the README, not buried.** [Read them](#honest-limitations) before you use this on a real person.

---

## Quick start

```bash
git clone https://github.com/virajshoor/ADAFAI.git
cd ADAFAI
pip install -e .
```

The core signals are **standard library only** — no dependencies, no model downloads:

```bash
adafai detect --file essay.txt
```

```
verdict : likely AI-generated
score   : 0.78  (0 = human, 1 = AI)
words   : 412

signals:
  stylometry   0.81
  unicode      0.45

Heuristic ensemble with unfitted weights - triage signal, not proof.
```

Add the language-model signals (perplexity, Binoculars) when you want more accuracy:

```bash
pip install -e ".[models]"
adafai detect --file essay.txt --models
```

As a library:

```python
from adafai import analyze

result = analyze(open("essay.txt").read(), use_models=True)
print(result["verdict"], result["score"])
print(result["signals"]["binoculars"])
```

---

## Web UI

**[https://virajshoor.github.io/ADAFAI/](https://virajshoor.github.io/ADAFAI/)** — paste any amount
of text and get the verdict, the signal breakdown, and a sentence-by-sentence
heatmap where clicking a sentence shows exactly *why* it scored what it did
(named tell-phrases, invisible characters, local diversity, repetition).

The UI is a zero-dependency static site in `docs/`: the core signals are ported
to JavaScript and run entirely in your browser, so **nothing is uploaded and
there is no length limit**. The torch-based signals (perplexity, Binoculars,
watermarks) can't run in a browser and remain Python-only.

It deploys via `.github/workflows/pages.yml` on every push to `main` that
touches `docs/`. One-time setup: **Settings → Pages → Source → GitHub Actions**.

---

## What it detects

| Signal | What it measures | Deps | Status |
|---|---|:--:|:--:|
| **Stylometry** | Lexical diversity (MATTR), sentence-length variance, punctuation entropy, self-repetition, and density of overused AI-tell phrases (*delve, underscore, tapestry, pivotal…*) | none | ✅ |
| **Unicode forensics** | Zero-width characters, exotic spaces, mixed-script homoglyphs, typographic polish. *Provenance*, not style — low recall, high precision | none | ✅ |
| **Perplexity + burstiness** | GPTZero's original method: how predictable the text is, and how much that predictability varies between sentences | torch | ✅ |
| **Binoculars** | Ratio of an observer model's perplexity to observer/performer cross-perplexity. Best-performing zero-shot method published ([Hans et al., ICML 2024](https://arxiv.org/abs/2401.12070)) | torch | ✅ |
| **Green-list watermark** | Kirchenbauer et al. z-test — **verified interoperable** with the reference implementation | torch | ✅ |
| **Fast-DetectGPT** | Conditional probability curvature ([Bao et al.](https://arxiv.org/abs/2310.05130)) | torch | 🚧 |
| **GLTR rank buckets** | Token-rank histogram ([Gehrmann et al.](https://arxiv.org/abs/1906.04043)) | torch | 🚧 |

---

## Watermarking: what each provider actually does

This is the part of the landscape that changed fastest, and it's where detection
can be *certain* rather than statistical — but only if you hold the key.

| Provider | Status | Can outsiders detect it? |
|---|---|---|
| **Google** (Gemini) | [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) has run in production since 2024 — the first LLM watermark deployed at scale. Biases sampling via *Tournament Sampling*: pseudorandom scoring functions seeded by a secret key and the preceding n-gram decide which token wins each position. | ❌ Only via Google's gated SynthID Detector. Production keys are not published. |
| **Anthropic** (Claude) | Began embedding an invisible text watermark on **2 August 2026**, biasing the randomness behind word choice. Driven by EU AI Act Article 50; no opt-out. A detection API has been announced but not yet shipped publicly. | ⏳ Not yet — no public endpoint. Won't catch heavily rewritten, mixed, or very short text even once it lands. |
| **OpenAI** (ChatGPT) | Built a working cryptographic watermarking prototype years ago and **deliberately never shipped it**, citing robustness, false positives, and user backlash. | ➖ Nothing to detect. Reported honestly rather than faked. |

### What ADAFAI can genuinely do today

Watermarks you generate yourself, where you hold the key:

```python
from adafai.watermark import detect_green_red
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok(suspect_text, return_tensors="pt")["input_ids"][0]

print(detect_green_red(ids, hash_key=15485863, vocab_size=len(tok)))
# {'z_score': 11.4, 'p_value': 2.3e-30, 'is_watermarked': True, ...}
```

```bash
adafai watermark --file output.txt --tokenizer gpt2 --key 15485863
```

This detector reproduces the reference PRNG exactly — `torch.randperm` seeded
`hash_key * prev_token`, both `lefthash` and `selfhash` schemes. The test suite
asserts green-list equality against `transformers`' own
`WatermarkLogitsProcessor` rather than against itself, so interoperability is
proven, not assumed.

---

## Honest limitations

> **Never use ADAFAI as sole evidence in an academic-integrity or disciplinary decision.**
> No detector on the market — this one included — is accurate enough to justify that.

- **No benchmark numbers yet.** The ensemble weights are a documented heuristic, not fitted on labeled data. Building the evaluation harness is the [next milestone](#roadmap). Until then, treat the score as triage. Anyone publishing accuracy claims without measurement is guessing, and we'd rather say so than invent a number.
- **Paraphrasing defeats it.** Running AI text through a rewrite degrades every signal here, watermarks included. This is a documented property of the whole field, not a bug specific to ADAFAI.
- **Short text is unscoreable.** Below 150 words the tool abstains by design.
- **Non-native English writers get false positives.** Stylometric and perplexity signals systematically misfire on ESL writing across every tool in this space. If you deploy this in an educational setting without accounting for that, you will disproportionately harm the students least able to contest it.
- **Edited human text trips the typography signal.** Word processors autocorrect straight quotes into curly ones, which is why that signal is weighted low.
- **Per-sentence scores are navigation hints, not mini-verdicts.** They exist to show *where* the document-level signals concentrate. A flagged sentence is a reason to look closer, not evidence about that sentence.

---

## Roadmap

- [ ] **Evaluation harness** — AUROC and TPR@1%/0.01% FPR against [RAID](https://arxiv.org/abs/2405.07940), HC3, and M4
- [ ] **Fitted ensemble** — replace hand-picked weights with logistic regression on labeled data
- [ ] **Calibrated thresholds** — per-model-pair Binoculars thresholds instead of the paper's Falcon-specific constant
- [ ] Fast-DetectGPT and GLTR over a shared forward pass
- [x] Per-sentence scoring for mixed human/AI documents (`adafai.spans`, powers the web UI heatmap)
- [ ] Anthropic watermark API once the endpoint ships

---

## Project layout

```
adafai/
├── stylometry.py          # lexical + statistical features (no deps)
├── unicode_forensics.py   # character-level provenance (no deps)
├── spans.py               # per-sentence localization + reasons (no deps)
├── perplexity.py          # perplexity + burstiness
├── binoculars.py          # cross-model perplexity ratio
├── detector.py            # ensemble → score + verdict
├── cli.py                 # adafai detect [--spans] / adafai watermark
└── watermark/
    ├── kirchenbauer.py    # green/red-list z-test (reference-interoperable)
    ├── synthid.py         # SynthID-Text detector wrapper
    └── providers.py       # per-provider watermark status
docs/                      # web UI (zero-dependency JS port of the core signals)
tests/                     # assert-based, no framework required
```

Run the checks:

```bash
python tests/test_stylometry.py && python tests/test_unicode_forensics.py && python tests/test_kirchenbauer.py
python tests/test_spans.py
node tests/test_web_ui.mjs   # browser-port parity behaviors
```

`adafai detect --spans` additionally lists the most AI-like sentences and the
reason for each — useful for mixed human/AI documents.

---

## Contributing

Issues and PRs welcome — particularly labeled evaluation data, which is the
single thing most blocking real accuracy numbers.

Please don't submit PRs that add accuracy claims without measurement to back them.

## License

[MIT](LICENSE) © 2026 Viraj Shoor
