"""Run/resume expanded native-language training, then evaluate every frozen checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(name, arguments):
    directory = ROOT / "outputs/pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    print("Running:", name, flush=True)
    with (directory / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect outputs/pipeline/{name}.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    args = parser.parse_args()
    if args.stage in {"train", "all"}:
        if not all((ROOT / f"data/processed/{split}.jsonl").exists() for split in ["train", "dev", "test"]):
            run("expanded_original_prepare", ["-m", "questionforge.cli", "prepare"])
        run("expanded_prepare", ["-m", "scripts.prepare_expanded_data"])
        for language in ["en", "fr", "ar"]:
            run(f"expanded_resume_{language}", ["-m", "scripts.train_expanded_lora", "--language", language, "--resume"])
    if args.stage in {"evaluate", "all"}:
        # All languages must complete before test generation; no test-based checkpoint selection.
        for language in ["en", "fr", "ar"]:
            path = ROOT / f"artifacts/expanded/{language}/training_manifest.json"
            if not path.exists() or not json.loads(path.read_text(encoding="utf-8"))["training_complete"]:
                raise ValueError(f"{language} training is not complete; do not evaluate partial checkpoints.")
        for language in ["en", "fr", "ar"]:
            for name, source in [("public", "data/processed"), ("classroom", "data/novel")]:
                run(f"expanded_{language}_{name}_evaluation", [
                    "-m", "questionforge.cli", "evaluate", "--data", f"{source}/test.jsonl",
                    "--output", f"results/expanded_{language}_{name}_test", "--languages", language,
                    "--methods", "beam4", "lora_beam4", "--checkpoint", f"artifacts/expanded/{language}",
                    "--seed", "42"])
        run("expanded_summary", ["-m", "scripts.analyze_expanded_training"])
        run("expanded_blind_review", ["-m", "scripts.prepare_expanded_review"])
        run("expanded_human_status", ["-m", "scripts.analyze_human_review", "--directory", "results/expanded_human_evaluation"])
        run("expanded_qualitative_pairs", ["-m", "scripts.compare_expanded_cases"])
        run("expanded_findings", ["-m", "scripts.build_expanded_training_report"])


if __name__ == "__main__":
    main()
