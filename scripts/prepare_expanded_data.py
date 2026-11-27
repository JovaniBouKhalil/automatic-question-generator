"""Prepare larger, source-train-only language datasets without changing frozen inputs."""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from questionforge.data import (
    DATASETS,
    assert_disjoint,
    context_hash,
    download_dataset,
    load_jsonl,
    normalize_records,
    validate_record,
)
from questionforge.experiment import fingerprint_file

TYDI = {"id": "google-research-datasets/tydiqa", "revision": "da78f23f9119363459acbaf46bf89426ff26c259",
        "license": "Apache-2.0; underlying Wikipedia attribution also applies",
        "train": "secondary_task/train-00000-of-00001.parquet",
        "test": "secondary_task/validation-00000-of-00001.parquet"}
BUDGETS = {"en": (5000, 200), "fr": (1000, 60), "ar": (2000, 150)}


def write_expanded_jsonl(path, rows):
    """Match the frozen Windows UTF-8/CRLF inputs byte-for-byte on every platform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\r\n" for row in rows).encode("utf-8"))


def title(row):
    return " ".join(unicodedata.normalize("NFKC", row.get("title", "")).casefold().split())


def partition_key(row):
    return "article:" + title(row) if title(row) else "paragraph:" + row["group"]


def sample_diverse(rows, limit, seed=42):
    """Use one QA per paragraph before adding further distinct questions from those paragraphs."""
    rng = random.Random(seed)
    grouped = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        grouped.setdefault(row["group"], []).append(row)
    groups = sorted(grouped)
    rng.shuffle(groups)
    for values in grouped.values():
        rng.shuffle(values)
    result = []
    depth = 0
    while len(result) < limit:
        layer = [grouped[group][depth] for group in groups if depth < len(grouped[group])]
        if not layer:
            break
        result.extend(layer[:limit - len(result)])
        depth += 1
    return result


def exclude_published_sources(training, protected):
    """Protect source text even if the published evaluation answer annotation is invalid."""
    groups = {r["group"] for r in protected}
    titles = {title(r) for r in protected if title(r)}
    removed = [r for r in training if r["group"] in groups or (title(r) and title(r) in titles)]
    removed_ids = {r["id"] for r in removed}
    return [r for r in training if r["id"] not in removed_ids], removed


def split_expanded(train, evaluation, original_dev, train_limit, dev_limit):
    reserved_groups = {r["group"] for r in evaluation + original_dev}
    reserved_titles = {title(r) for r in evaluation + original_dev if title(r)}
    clean = [r for r in train if r["group"] not in reserved_groups and
             (not title(r) or title(r) not in reserved_titles)]
    keys = sorted({partition_key(r) for r in clean})
    random.Random(42).shuffle(keys)
    # Reserve enough *different paragraphs*, not many questions from one article.
    groups_by_key = {}
    for row in clean:
        groups_by_key.setdefault(partition_key(row), set()).add(row["group"])
    development_keys = set()
    development_groups = set()
    for key in keys:
        if len(development_groups) >= max(0, dev_limit - len(original_dev)):
            break
        development_keys.add(key)
        development_groups.update(groups_by_key[key])
    dev_candidates = [r for r in clean if partition_key(r) in development_keys]
    development = original_dev + sample_diverse(dev_candidates, max(0, dev_limit - len(original_dev)))
    training = sample_diverse([r for r in clean if partition_key(r) not in development_keys], train_limit)
    assert_disjoint({"train": training, "dev": development, "test": evaluation})
    held_titles = {title(r) for r in development + evaluation if title(r)}
    assert not ({title(r) for r in training if title(r)} & held_titles)
    return training, development, {"source_training_rows": len(train),
                                   "excluded_for_original_dev_or_published_evaluation": len(train) - len(clean),
                                   "training_paragraphs": len({r["group"] for r in training}),
                                   "development_paragraphs": len({r["group"] for r in development}),
                                   "exact_paragraph_intersection": [], "known_article_title_intersection": []}


def download_tydi():
    import pandas as pd

    root = Path("data/downloads/tydiqa_goldp")
    root.mkdir(exist_ok=True)
    output, sources, full_evaluation = {}, {}, []
    for split in ["train", "test"]:
        target = root / f"{split}.parquet"
        if not target.exists():
            url = f"https://huggingface.co/datasets/{TYDI['id']}/resolve/{TYDI['revision']}/{TYDI[split]}?download=true"
            partial = target.with_suffix(".partial")
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
                        length = int(response.headers.get("Content-Length", 0))
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                    if length and partial.stat().st_size != length:
                        raise IOError("Incomplete TyDiQA file")
                    pd.read_parquet(partial, columns=["id"])
                    partial.replace(target)
                    break
                except Exception:
                    if attempt == 4:
                        raise
        sources[split] = {"path": TYDI[split], "sha256": fingerprint_file(target), "bytes": target.stat().st_size}
        rows, rejected = [], Counter()
        for raw in pd.read_parquet(target).to_dict(orient="records"):
            if not str(raw["id"]).casefold().startswith("arabic"):
                continue
            if split == "test":
                # Invalid answer offsets do not make a published validation article
                # available for training. Reserve its source even when QA is unusable.
                full_evaluation.append({"group": context_hash(raw["context"]), "title": raw.get("title", "")})
            answers = raw["answers"]
            if len(answers["text"]) == 0:
                rejected["empty_answer"] += 1
                continue
            row = {"id": "ar:tydiqa:" + str(raw["id"]), "context": raw["context"],
                   "question": raw["question"], "answer": str(answers["text"][0]),
                   "answer_start": int(answers["answer_start"][0]), "language": "ar",
                   "group": context_hash(raw["context"]), "title": raw.get("title", ""), "source": TYDI["id"]}
            try:
                validate_record(row)
            except ValueError:
                rejected["invalid_record_or_span"] += 1
                continue
            rows.append(row)
        if not rows:
            raise ValueError("No Arabic TyDiQA rows found; inspect the schema before training.")
        output[split] = rows
        sources[split]["valid_arabic_records"] = len(rows)
        sources[split]["excluded"] = dict(rejected)
    output["test_protection"] = full_evaluation
    return output, {**TYDI, "files": sources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Rebuild unused preparation outputs before any expanded adapter exists.")
    args = parser.parse_args()
    output = Path("data/processed/expanded")
    if args.refresh and any((Path("artifacts/expanded") / lang / "adapter_config.json").exists() for lang in BUDGETS):
        raise ValueError("Adapters already exist; preserve their frozen training inputs.")
    if (output / "manifest.json").exists() and not args.refresh:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for language in BUDGETS:
            for split, expected in manifest["languages"][language]["sha256"].items():
                assert fingerprint_file(output / language / f"{split}.jsonl") == expected
        print("Existing expanded datasets have matching hashes; preserved.")
        return
    tydi, tydi_source = download_tydi()
    old_dev = load_jsonl("data/processed/dev.jsonl")
    old_test = load_jsonl("data/processed/test.jsonl")
    manifest = {"seed": 42, "protocol_sha256": fingerprint_file(Path("docs/EXPANDED_TRAINING_PROTOCOL.md")),
                "languages": {}, "additional_source": tydi_source,
                "note": "Source training only. Entire published evaluation pools are protected, not just sampled test rows."}
    for language, (train_limit, dev_limit) in BUDGETS.items():
        raw = download_dataset(language, Path("data/downloads"))
        train, exclusions = normalize_records(raw["train"], language)
        evaluation, _ = normalize_records(raw["test"], language)
        if language == "ar":
            train += tydi["train"]
            evaluation += tydi["test"]
        # Cross-source duplicates can otherwise give the same passage artificial weight.
        unique, seen = [], set()
        for row in train:
            signature = (row["group"], row["question"].strip(), row["answer"])
            if signature not in seen:
                unique.append(row)
                seen.add(signature)
        original_dev = [r for r in old_dev if r["language"] == language]
        training, development, audit = split_expanded(unique, evaluation, original_dev, train_limit, dev_limit)
        if language == "ar":
            protected = tydi["test_protection"]
            training, removed = exclude_published_sources(training, protected)
            audit["training_paragraphs"] = len({r["group"] for r in training})
            audit["additional_full_validation_article_filter_removed_ids"] = [r["id"] for r in removed]
            audit["all_published_arabic_validation_records_checked"] = len(protected)
        partitions = {"train": training, "dev": development,
                      "test": [r for r in old_test if r["language"] == language]}
        for split, rows in partitions.items():
            write_expanded_jsonl(output / language / f"{split}.jsonl", rows)
        info = {"source": DATASETS[language], "source_exclusions": dict(exclusions),
                "counts": {split: len(rows) for split, rows in partitions.items()}, "audit": audit,
                "sha256": {split: fingerprint_file(output / language / f"{split}.jsonl") for split in partitions},
                "record_ids": {split: [r["id"] for r in rows] for split, rows in partitions.items()}}
        manifest["languages"][language] = info
        print(language, info["counts"], audit, flush=True)
    output.mkdir(exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = Path("results/expanded_training")
    receipt.mkdir(exist_ok=True)
    (receipt / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
