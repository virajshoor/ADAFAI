import argparse
import json
import sys

from adafai.detector import analyze


def _read(args) -> str:
    if args.text:
        return args.text
    if args.file == "-":
        return sys.stdin.read()
    with open(args.file, encoding="utf-8") as fh:
        return fh.read()


def _cmd_detect(args) -> None:
    text = _read(args)
    if not text.strip():
        sys.exit("no text to analyze")
    result = analyze(text, use_models=args.models)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"verdict : {result['verdict']}")
    print(f"score   : {result['score']}  (0 = human, 1 = AI)")
    print(f"words   : {result['word_count']}")
    for flag in result.get("provenance_flags", []):
        print(f"  ! {flag}")
    print("\nsignals:")
    for name, sig in result["signals"].items():
        print(f"  {name:12} {sig['score']}")
    print(f"\n{result['note']}")


def _cmd_watermark(args) -> None:
    from transformers import AutoTokenizer

    from adafai.watermark import detect_green_red

    text = _read(args)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    result = detect_green_red(
        ids,
        hash_key=args.key,
        vocab_size=len(tok),
        gamma=args.gamma,
        seeding_scheme=args.scheme,
        context_width=args.context_width,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adafai",
        description="Detect AI-generated text via stylometry, LM statistics, and watermarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="score text for AI authorship")
    src = d.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="path to a text file, or - for stdin")
    src.add_argument("--text", help="text to analyze")
    d.add_argument("--models", action="store_true",
                   help="also run perplexity + Binoculars (needs torch, transformers)")
    d.add_argument("--json", action="store_true", help="emit raw JSON")
    d.set_defaults(func=_cmd_detect)

    w = sub.add_parser("watermark", help="test for a Kirchenbauer green-list watermark")
    wsrc = w.add_mutually_exclusive_group(required=True)
    wsrc.add_argument("--file", help="path to a text file, or - for stdin")
    wsrc.add_argument("--text", help="text to analyze")
    w.add_argument("--tokenizer", default="gpt2", help="tokenizer used at generation")
    w.add_argument("--key", type=int, default=15485863, help="generation-time hash key")
    w.add_argument("--gamma", type=float, default=0.25, help="green-list fraction")
    w.add_argument("--scheme", default="lefthash", choices=["lefthash", "selfhash"])
    w.add_argument("--context-width", type=int, default=1)
    w.set_defaults(func=_cmd_watermark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
