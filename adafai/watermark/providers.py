"""Status of first-party watermarking across the major providers.

Detecting a closed provider's production watermark requires their
secret key or their hosted detection API - neither this repo nor
anyone outside the provider can reconstruct it from text alone. These
functions call the provider's official detector when one exists, and
raise clearly when it doesn't, instead of guessing.

Last checked: August 2026.

- Google (Gemini): SynthID-Text has been live in production since 2024,
  the first LLM watermark deployed at scale. Detection is available to
  verified users/enterprises through Google's SynthID Detector, not as
  an open public API. See adafai.watermark.synthid for the open-source
  detector code path (only usable against models *you* watermark).
- Anthropic (Claude): started embedding an invisible text watermark in
  Claude output on 2026-08-02, driven by EU AI Act Article 50, with no
  opt-out. Anthropic has announced a detection API but has not yet
  published its endpoint/spec as of this writing - `check_claude`
  below raises NotImplementedError until that ships. It also will not
  catch heavily rewritten, mixed, or very short text.
- OpenAI (ChatGPT/GPT): built a cryptographic watermarking prototype
  but has never deployed it in production; there is no OpenAI text
  watermark to detect as of this writing.
"""


class ProviderWatermarkUnavailable(NotImplementedError):
    pass


def check_openai(text: str) -> dict:
    return {
        "provider": "openai",
        "supported": False,
        "reason": "OpenAI has not deployed a production text watermark.",
    }


def check_claude(text: str, api_endpoint: str | None = None) -> dict:
    if api_endpoint is None:
        raise ProviderWatermarkUnavailable(
            "Anthropic has announced but not yet published its watermark "
            "detection API. Pass api_endpoint once Anthropic ships one, "
            "or check https://anthropic.com for availability."
        )
    import urllib.request
    import json

    req = urllib.request.Request(
        api_endpoint,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def check_gemini(text: str) -> dict:
    return {
        "provider": "gemini",
        "supported": False,
        "reason": (
            "SynthID-Text is live in Gemini but detection is gated behind "
            "Google's SynthID Detector for verified users, not a public API. "
            "See https://deepmind.google/technologies/synthid/"
        ),
    }
