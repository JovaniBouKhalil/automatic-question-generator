"""Exploratory paired verifier probe: append an unsupported location to real questions.

This post-hoc diagnostic was motivated by a source-inspected factual error with perfect
QA agreement. It does not tune generation models or the fixed acceptance thresholds.
Labels mean 'unsupported by the supplied passage', not proven false in the real world.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from questionforge.data import load_jsonl, write_jsonl
from questionforge.experiment import environment_info, fingerprint_file
from questionforge.metrics import leaks_answer, quality_decision
from questionforge.models import verify_question
from questionforge.registry import VERIFIER

QUALIFIER = {"en": " on Mars?", "fr": " sur Mars ?", "ar": " على المريخ؟"}
LOCATION = {"en": "mars", "fr": "mars", "ar": "المريخ"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/novel/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/verifier_stress"))
    args = parser.parse_args()
    records = load_jsonl(args.data)
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"scope": "Exploratory post-hoc unsupported-qualifier probe; no model/threshold tuning",
              "dataset_sha256": fingerprint_file(args.data), "verifier": asdict(VERIFIER),
              "qualifiers": QUALIFIER, "min_answer_f1": .5, "min_span_score": .1,
              "script_sha256": fingerprint_file(Path(__file__)), "environment": environment_info()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    for index, record in enumerate(records, 1):
        language = record["language"]
        if LOCATION[language] in record["context"].casefold():
            raise ValueError("The added location already occurs in this passage; the probe is invalid.")
        modified = record["question"].rstrip(" ?؟") + QUALIFIER[language]
        for condition, question in [("reference", record["question"]), ("unsupported_qualifier", modified)]:
            row = {**record, "id": record["id"] + ":" + condition, "source_id": record["id"],
                   "condition": condition, "reference": record["question"], "question": question,
                   "answer_leak": leaks_answer(question, record["answer"], language)}
            row.update(verify_question(question, record["context"], record["answer"], language))
            row["accepted"], row["review_reasons"] = quality_decision(row)
            rows.append(row)
        print(f"Verifier probe {index}/{len(records)}: {language}", flush=True)
    write_jsonl(args.output / "predictions.jsonl", rows)
    summary = []
    for language in ["en", "fr", "ar"]:
        for condition in ["reference", "unsupported_qualifier"]:
            selected = [r for r in rows if r["language"] == language and r["condition"] == condition]
            if selected:
                summary.append({"language": language, "condition": condition, "n": len(selected),
                                "target_answer_f1": 100 * sum(r["answer_f1"] for r in selected) / len(selected),
                                "automatic_acceptance_pct": 100 * sum(r["accepted"] for r in selected) / len(selected)})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
