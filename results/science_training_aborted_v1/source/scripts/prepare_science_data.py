"""Prepare pinned SciQ science passages without crossing published split boundaries."""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

from questionforge.data import assert_disjoint, context_hash, validate_record, write_jsonl

DATASET = "allenai/sciq"
REVISION = "2c94ad3e1aafab77146f384e23536f97a4849815"


def convert(rows, split):
    retained, excluded, seen = [], Counter(), set()
    for index, item in enumerate(rows):
        context, answer, question = (str(item.get(k, "")).strip() for k in ["support", "correct_answer", "question"])
        if not context or not answer or not question:
            excluded["missing_support_answer_or_question"] += 1
            continue
        matches = list(re.finditer(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", context, re.IGNORECASE))
        if len(matches) != 1:
            excluded["answer_missing_or_ambiguous"] += 1
            continue
        if "?" not in question or len(context) < 35 or len(answer) > 120:
            excluded["malformed_or_out_of_budget"] += 1
            continue
        group = context_hash(context)
        if group in seen:
            excluded["duplicate_support_group"] += 1
            continue
        seen.add(group)
        match = matches[0]
        row = {"id": f"sciq:{split}:{index}", "context": context, "question": question,
               "answer": context[match.start():match.end()], "answer_start": match.start(),
               "language": "en", "group": group, "source": DATASET,
               "source_split": split, "answer_alignment": "unique_case_insensitive_word_bounded_exact_span"}
        validate_record(row)
        retained.append(row)
    return retained, dict(excluded)


def main():
    import pandas as pd

    target = Path("data/processed/science")
    if (target / "manifest.json").exists():
        raise ValueError("Prepared science study exists; preserve it rather than overwrite.")
    cache = Path(".cache/science-source")
    cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}?blobs=true", timeout=60) as r:
        metadata = json.load(r)
    assert metadata["sha"] == REVISION
    raw_counts, exclusions, partitions, files = {}, {}, {}, {}
    for split in ["train", "validation", "test"]:
        name = f"data/{split}-00000-of-00001.parquet"
        entry = next(x for x in metadata["siblings"] if x["rfilename"] == name)
        path = cache / Path(name).name
        if not path.exists():
            with urllib.request.urlopen(f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}?download=true", timeout=120) as r:
                content = r.read()
            path.write_bytes(content)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["lfs"]["sha256"], name
        files[name] = digest
        raw = pd.read_parquet(path).to_dict(orient="records")
        raw_counts[split] = len(raw)
        partitions[split], exclusions[split] = convert(raw, split)
    test_groups = {r["group"] for r in partitions["test"]}
    dev_groups = {r["group"] for r in partitions["validation"]}
    for split, forbidden in [("train", dev_groups | test_groups), ("validation", test_groups)]:
        before = len(partitions[split])
        partitions[split] = [r for r in partitions[split] if r["group"] not in forbidden]
        exclusions[split]["cross_split_support_group"] = before - len(partitions[split])
    limits = {"train": 2048, "dev": 128, "test": 200}
    result = {}
    for split, published in [("train", "train"), ("dev", "validation"), ("test", "test")]:
        records = sorted(partitions[published], key=lambda r: r["id"])
        random.Random(42).shuffle(records)
        result[split] = records[:limits[split]]
        write_jsonl(target / f"{split}.jsonl", result[split])
    assert_disjoint(result)
    manifest = {"dataset": DATASET, "revision": REVISION, "license": "CC-BY-NC-3.0", "seed": 42,
                "published_counts": raw_counts, "exclusions": exclusions,
                "prepared_counts": {s: len(r) for s, r in result.items()}, "source_sha256": files,
                "prepared_sha256": {s: hashlib.sha256((target / f"{s}.jsonl").read_bytes()).hexdigest() for s in result},
                "protocol_sha256": hashlib.sha256(Path("docs/SCIENCE_TRAINING_PROTOCOL.md").read_bytes()).hexdigest(),
                "test_used_for_training_or_selection": False}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (cache / "hub_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
