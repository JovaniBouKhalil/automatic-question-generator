"""Validate saved learning studies without mistaking mechanical checks for usefulness."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from scripts.run_learning_focus_study import source_hash, write_json


def audit(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    sources = config.get("sources", config.get("source_sha256_normalized_utf8"))
    for name, expected in sources.items():
        snapshot = directory / "source" / name
        assert source_hash(snapshot) == expected, (directory.name, "source snapshot", name)
    data = Path("data/authoring/learning_passages.json") if "educational" in config["split"] else Path("data/novel/dev.jsonl")
    assert source_hash(data) == config["data_sha256"], (directory.name, "development data changed")
    if data.suffix == ".json":
        scenarios = json.loads(data.read_text(encoding="utf-8"))
        known = {(lang, scenario[lang]["context"]) for scenario in scenarios for lang in ["en", "fr", "ar"]}
    else:
        known = {(r["language"], r["context"]) for r in map(json.loads, data.read_text(encoding="utf-8").splitlines())}
    cases = []
    for path in directory.glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(item, dict) and "case_id" in item:
            cases.append(item)
    assert len({c["case_id"] for c in cases}) == len(cases)
    counts = {}
    item_ids = set()
    for case in cases:
        context = case["context"]
        assert (case["language"], context) in known, (directory.name, case["case_id"])
        for method, rows in case["outputs"].items():
            counts[method] = counts.get(method, 0) + len(rows)
            for index, row in enumerate(rows):
                offset, answer = row["answer_start"], row["answer"]
                assert offset >= 0 and context[offset:offset + len(answer)] == answer
                assert row["evidence"] in context
                assert row["question"].strip() and 0 <= row["answer_f1"] <= 1
                item_ids.add(f"{case['case_id']}:{method}:{index}")
        report = case.get("planning_report")
        if report:
            assert report["planned"] == len(report["facts"])
            for fact in report["facts"]:
                start = fact["answer_start"]
                assert context[start:start + len(fact["answer"])] == fact["answer"]
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    if summary:
        assert summary["passage_language_pairs"] == len(cases)
        recorded = summary.get("returned", {k: v["returned_questions"] for k, v in summary.get("methods", {}).items()})
        for method, count in recorded.items():
            assert counts.get(method, 0) == count, (directory.name, method)
    template = directory / "learning_review_template.csv"
    rating_rows = []
    if template.exists():
        rows = list(csv.DictReader(io.StringIO(template.read_text(encoding="utf-8-sig"), newline="")))
        assert {r["item_id"] for r in rows} == item_ids
        rating_rows = [r for r in rows if r["reviewer_id"] or any(r[k] for k in
                       ["factual_support_0_2", "clarity_0_2", "learning_value_0_2"])]
    return {"study": directory.name, "completed_case_records": len(cases), "source_snapshots_checked": len(sources),
            "returned_questions": counts, "execution_error_cases": sum(bool(c.get("error")) for c in cases),
            "review_rows_with_any_ratings": len(rating_rows),
            "interpretation": "Exact spans, source snapshots and counts checked. Logical support and educational value are not established by this audit."}


def main():
    directories = [p for p in Path("results").glob("learning*") if p.is_dir() and (p / "config.json").exists()]
    results = [audit(directory) for directory in sorted(directories)]
    target = Path("results/quality/learning_focus_validation.json")
    write_json(target, {"studies": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
