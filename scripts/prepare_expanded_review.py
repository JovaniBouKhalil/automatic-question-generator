"""Prepare a separate blinded comparison of original and expanded native adapters."""

import hashlib
import json
import random
from pathlib import Path

from questionforge.experiment import fingerprint_file
from questionforge.review import GUIDE, ratings_csv
from scripts.prepare_human_review import BOOKLET


def main():
    rng = random.Random(42)
    items, mapping, sources = [], [], {}
    selected_topics = None
    for dataset in ["public", "classroom"]:
        for language in ["en", "fr", "ar"]:
            path = Path(f"results/expanded_{language}_{dataset}_test/predictions.jsonl")
            sources[path.as_posix()] = fingerprint_file(path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            baseline = sorted([r for r in rows if r["method"] == "beam4"], key=lambda r: r["id"])
            if dataset == "classroom":
                if selected_topics is None:
                    selected_topics = sorted(rng.sample(sorted({r["group"] for r in baseline}), 3))
                selected = [next(r for r in baseline if r["group"] == topic) for topic in selected_topics]
            else:
                selected = rng.sample(baseline, 3)
            lookup = {(r["id"], r["method"]): r for r in rows}
            for source in selected:
                for method in ["beam4", "lora_beam4"]:
                    row = lookup[(source["id"], method)]
                    identifier = hashlib.sha256(f"qf-expanded-review-v1:{dataset}:{row['id']}:{method}".encode()).hexdigest()[:12]
                    items.append({"item_id": identifier, "language": language, "context": row["context"],
                                  "question": row["question"], "answer": row["answer"]})
                    mapping.append({"item_id": identifier, "language": language, "dataset": dataset,
                                    "source_id": row["id"], "group": row["group"], "method": method})
    rng.shuffle(items)
    directory = Path("results/expanded_human_evaluation")
    directory.mkdir(exist_ok=True)
    digest = hashlib.sha256(json.dumps(items, ensure_ascii=False, indent=2).encode()).hexdigest()
    manifest = {"items": len(items), "sampling_seed": 42, "items_sha256": digest, "source_sha256": sources,
                "blinding": "Original/expanded checkpoint identity, references and QA scores are hidden in the offline booklet.",
                "status": "prepared; genuine human ratings pending"}
    for name, value in [("items.json", items), ("researcher_mapping.json", mapping), ("manifest.json", manifest)]:
        path = directory / name
        content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Preserve the existing review sample before changing it.")
        path.write_text(content, encoding="utf-8")
    html = BOOKLET.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("<", "\\u003c"))
    html = html.replace("__GUIDE__", json.dumps(GUIDE)).replace("__HASH__", json.dumps(digest[:16]))
    (directory / "review_booklet.html").write_text(html, encoding="utf-8")
    (directory / "ratings_template.csv").write_bytes(ratings_csv([{"item_id": r["item_id"]} for r in items]).encode("utf-8-sig"))
    print(f"Prepared {len(items)} blinded original/expanded comparisons; no human ratings supplied.")


if __name__ == "__main__":
    main()
