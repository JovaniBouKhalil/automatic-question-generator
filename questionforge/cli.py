"""Small command line entry point; use --help on a command for reproducible options."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main():
    # Windows redirected stdout can default to cp1252, which cannot encode Arabic.
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="questionforge")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download pinned public benchmarks and audit splits")
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    prepare.add_argument("--train-size", type=int, default=600)
    prepare.add_argument("--dev-size", type=int, default=15)
    prepare.add_argument("--test-size", type=int, default=40)
    prepare.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="Run/resume a paired generation experiment")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--methods", nargs="+", default=["baseline", "greedy", "beam4"])
    evaluate.add_argument("--languages", nargs="+", choices=["en", "fr", "ar"], default=["en", "fr", "ar"])
    evaluate.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    evaluate.add_argument("--no-verify", action="store_true")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--seed", type=int, default=42)
    generate = commands.add_parser("generate", help="Generate one source-grounded question")
    generate.add_argument("--text", required=True)
    generate.add_argument("--answer", required=True)
    generate.add_argument("--language", choices=["en", "fr", "ar"], default="en")
    generate.add_argument("--method", choices=["baseline", "transformer"], default="transformer")
    generate.add_argument("--beams", type=int, default=4)
    generate.add_argument("--verify", action="store_true")
    generate.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare_data
        if min(args.train_size, args.dev_size, args.test_size) < 1:
            parser.error("Dataset sizes must be positive.")
        print(json.dumps(prepare_data(args.output, args.train_size, args.dev_size, args.test_size, args.seed), indent=2))
    elif args.command == "evaluate":
        from .experiment import run_experiment
        result = run_experiment(args.data, args.output, methods=args.methods, languages=args.languages,
                                device=args.device, verify=not args.no_verify, limit=args.limit,
                                checkpoint=args.checkpoint, seed=args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        from .baseline import template_question
        from .models import generate_question, verify_question
        if args.method == "baseline":
            result = {"question": template_question(args.text, args.answer, args.language), "answer": args.answer}
        else:
            result = asdict(generate_question(args.text, args.answer, args.language, beams=args.beams, device=args.device))
        if args.verify:
            result.update(verify_question(result["question"], args.text, args.answer, args.language, args.device))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
