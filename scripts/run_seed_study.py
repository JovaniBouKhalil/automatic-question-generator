"""Repeat the prespecified 160-update LoRA experiment without overwriting old artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from questionforge.experiment import fingerprint_file

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 42, 73)


def run(arguments):
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    args = parser.parse_args()
    for seed in SEEDS:
        adapter = Path(f"artifacts/seed_study/seed{seed}")
        if args.stage in {"train", "all"}:
            manifest_path = ROOT / adapter / "training_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["seed"] != seed or manifest["steps"] != 160:
                    raise ValueError("Existing training manifest differs from the fixed protocol.")
                if manifest["data_sha256"] != {split: fingerprint_file(ROOT / f"data/processed/{split}.jsonl")
                                               for split in ["train", "dev", "test"]}:
                    raise ValueError("Training inputs differ from the saved adapter manifest.")
            else:
                run(["-m", "scripts.train_lora", "--steps", "160", "--seed", str(seed),
                     "--output", adapter.as_posix()])
        if args.stage in {"evaluate", "all"}:
            for dataset, source in [("public", "data/processed"), ("classroom", "data/novel")]:
                output = f"results/seed{seed}_{dataset}_test"
                run(["-m", "questionforge.cli", "evaluate", "--data", f"{source}/test.jsonl",
                     "--output", output, "--languages", "en", "--methods", "beam4", "lora_beam4",
                     "--checkpoint", adapter.as_posix(), "--seed", str(seed)])


if __name__ == "__main__":
    main()
