"""Train one prespecified adapter with article-isolated English development data."""

import argparse
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

from questionforge.data import load_jsonl, write_jsonl
from questionforge.experiment import fingerprint_file


def title(row):
    return " ".join(unicodedata.normalize("NFKC", row.get("title", "")).casefold().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate", "all"], default="all")
    args = parser.parse_args()
    directory = Path("data/processed/article_sensitivity")
    output = Path("results/article_sensitivity")
    directory.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    if args.stage in {"prepare", "train", "all"}:
        data = {split: [r for r in load_jsonl(f"data/processed/{split}.jsonl") if r["language"] == "en"]
                for split in ["train", "dev", "test"]}
        if not all(title(r) for rows in data.values() for r in rows):
            raise ValueError("Article isolation requires populated source titles.")
        held_titles = {title(r) for split in ["dev", "test"] for r in data[split]}
        excluded = [r for r in data["train"] if title(r) in held_titles]
        data["train"] = [r for r in data["train"] if title(r) not in held_titles]
        assert not ({title(r) for r in data["train"]} & held_titles)
        for split, rows in data.items():
            write_jsonl(directory / f"{split}.jsonl", rows)
        manifest = {"scope": "English article-isolation sensitivity, not a replacement of frozen original experiments.",
                    "seed": 42, "updates": 160, "counts": {split: len(rows) for split, rows in data.items()},
                    "removed_training_ids": [r["id"] for r in excluded], "removed_training_rows": len(excluded),
                    "training_heldout_article_intersection": [],
                    "data_sha256": {split: fingerprint_file(directory / f"{split}.jsonl") for split in data}}
        (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in manifest.items() if k != "removed_training_ids"}, indent=2), flush=True)
    adapter = "artifacts/article_isolated_lora"
    if args.stage in {"train", "all"}:
        saved = Path(adapter) / "training_manifest.json"
        if saved.exists():
            previous = json.loads(saved.read_text(encoding="utf-8"))
            expected = {split: fingerprint_file(directory / f"{split}.jsonl") for split in ["train", "dev", "test"]}
            if previous["data_sha256"] != expected or previous["steps"] != 160 or previous["seed"] != 42:
                raise ValueError("Saved article-isolated adapter differs from the fixed run.")
        else:
            subprocess.run([sys.executable, "-m", "scripts.train_lora", "--data-dir", str(directory),
                            "--output", adapter, "--seed", "42", "--steps", "160"], check=True)
    if args.stage in {"evaluate", "all"}:
        for dataset, source in [("public", "data/processed"), ("classroom", "data/novel")]:
            subprocess.run([sys.executable, "-m", "questionforge.cli", "evaluate", "--data", f"{source}/test.jsonl",
                            "--output", f"results/article_{dataset}_test", "--languages", "en", "--methods",
                            "beam4", "lora_beam4", "--checkpoint", adapter, "--seed", "42"], check=True)


if __name__ == "__main__":
    main()
