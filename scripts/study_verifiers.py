"""Fixed paired verifier study on synthetic controls, not human-rated generation quality."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from questionforge.data import load_jsonl
from questionforge.experiment import bootstrap_mean_ci, environment_info, fingerprint_file
from questionforge.metrics import leaks_answer, quality_decision
from questionforge.registry import VERIFIER
from questionforge.text import normalize_answer
from questionforge.verification import MULTILINGUAL_QA, check_answer

QUALIFIER = {"en": " on Mars?", "fr": " sur Mars ?", "ar": " على المريخ؟"}
LOCATION = {"en": "mars", "fr": "mars", "ar": "المريخ"}
UNRELATED = {"en": "What is the serial number of the red submarine?",
             "fr": "Quel est le numéro de série du sous-marin rouge ?",
             "ar": "ما الرقم التسلسلي للغواصة الحمراء؟"}


def build_controls(records):
    controls = []
    for record in records:
        language = record["language"]
        if LOCATION[language] in record["context"].casefold():
            raise ValueError("Unsupported qualifier already appears in the source.")
        alternatives = [r for r in records if r["group"] == record["group"] and r["language"] == language
                        and normalize_answer(r["answer"], language) != normalize_answer(record["answer"], language)]
        if not alternatives:
            raise ValueError("Each control topic must contain two distinct target answers.")
        mismatch = sorted(alternatives, key=lambda r: r["id"])[0]["answer"]
        conditions = [("reference", record["question"], record["answer"], True),
                      ("unrelated", UNRELATED[language], record["answer"], False),
                      ("unsupported_location", record["question"].rstrip(" ?؟") + QUALIFIER[language], record["answer"], False),
                      ("target_mismatch", record["question"], mismatch, False)]
        for condition, question, answer, supported in conditions:
            controls.append({**record, "source_id": record["id"], "condition": condition,
                             "question": question, "answer": answer,
                             "target_consistent_by_construction": supported})
    return controls


def summarize(rows):
    if any(row.get("error") for row in rows):
        raise ValueError("Verifier failures are saved, but must be resolved before claiming control accuracy.")
    output, paired = [], []
    for split in ["dev", "test"]:
        for language in ["en", "fr", "ar"]:
            for verifier in ["original", "multilingual"]:
                selected = [r for r in rows if r["partition"] == split and r["language"] == language
                            and r["verifier"] == verifier]
                if not selected:
                    continue
                positive = [r for r in selected if r["condition"] == "reference"]
                negative = [r for r in selected if r["condition"] != "reference"]
                true_accept = sum(r["accepted"] for r in positive) / len(positive)
                false_accept = sum(r["accepted"] for r in negative) / len(negative)
                conditions = []
                for condition in sorted({r["condition"] for r in selected}):
                    values = [r for r in selected if r["condition"] == condition]
                    conditions.append({"condition": condition, "n": len(values),
                                       "acceptance_pct": 100 * sum(r["accepted"] for r in values) / len(values),
                                       "target_f1": 100 * sum(r["answer_f1"] for r in values) / len(values)})
                output.append({"partition": split, "language": language, "verifier": verifier,
                               "reference_f1": 100 * sum(r["answer_f1"] for r in positive) / len(positive),
                               "reference_em": 100 * sum(r["answer_em"] for r in positive) / len(positive),
                               "reference_acceptance_pct": 100 * true_accept,
                               "negative_false_acceptance_pct": 100 * false_accept,
                               "constructed_control_balanced_accuracy_pct": 50 * (true_accept + 1 - false_accept),
                               "errors": sum(bool(r.get("error")) for r in selected), "conditions": conditions})
            original = {(r["source_id"], r["condition"]): r for r in rows
                        if r["partition"] == split and r["language"] == language and r["verifier"] == "original"}
            other = {(r["source_id"], r["condition"]): r for r in rows
                     if r["partition"] == split and r["language"] == language and r["verifier"] == "multilingual"}
            if set(original) != set(other):
                raise ValueError("Verifier comparisons must have identical control IDs.")
            if not original:
                continue
            # Equal class weighting: one positive vs three different negative constructions.
            differences = []
            for key, base in original.items():
                alternative = other[key]
                positive = base["target_consistent_by_construction"]
                weight = 2.0 if positive else 2.0 / 3.0
                correctness = lambda r: float(r["accepted"] == positive)  # noqa: E731
                differences.append({"group": base["group"], "answer_f1": weight * (correctness(alternative) - correctness(base))})
            paired.append({"partition": split, "language": language,
                           "multilingual_minus_original_balanced_accuracy_pp": 100 * sum(r["answer_f1"] for r in differences) / len(differences),
                           "topic_bootstrap_ci95_pp": bootstrap_mean_ci(differences, seed=42),
                           "independent_topics": len({r["group"] for r in differences})})
    return {"scope": "Follow-up constructed-control study. Not human-rated generated-question accuracy; no threshold tuning.",
            "models": output, "paired_differences": paired}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/verification_study"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"models": {"original": asdict(VERIFIER), "multilingual": asdict(MULTILINGUAL_QA)},
              "thresholds": {"answer_f1": .5, "span_score": .1}, "seed": 42,
              "data_sha256": {split: fingerprint_file(Path(f"data/novel/{split}.jsonl")) for split in ["dev", "test"]},
              "protocol_sha256": fingerprint_file(Path("docs/EXTENSION_PROTOCOL.md")),
              "source_sha256": {name: fingerprint_file(Path(name)) for name in
                                ["scripts/study_verifiers.py", "questionforge/verification.py"]}}
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Study configuration changed; choose a new output directory.")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "environment.json").write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")
    path = args.output / "predictions.jsonl"
    rows = load_jsonl_raw(path)
    completed = {(r["partition"], r["source_id"], r["condition"], r["verifier"]) for r in rows}
    with path.open("a", encoding="utf-8") as stream:
        for split in ["dev", "test"]:
            controls = build_controls(load_jsonl(f"data/novel/{split}.jsonl"))
            for verifier in ["original", "multilingual"]:
                for index, control in enumerate(controls, 1):
                    key = (split, control["source_id"], control["condition"], verifier)
                    if key in completed:
                        continue
                    started = time.perf_counter()
                    row = {**control, "partition": split, "verifier": verifier, "error": None}
                    try:
                        row.update(check_answer(row["question"], row["context"], row["answer"], row["language"], verifier))
                        row["answer_leak"] = leaks_answer(row["question"], row["answer"], row["language"])
                        row["accepted"], row["review_reasons"] = quality_decision(row)
                    except Exception as exc:
                        row.update(error=f"{type(exc).__name__}: {exc}", accepted=False, answer_f1=0., answer_em=0., qa_score=0.)
                    row["seconds"] = time.perf_counter() - started
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    print(f"{split}/{verifier}: {index}/{len(controls)}", flush=True)
    report = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def load_jsonl_raw(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


if __name__ == "__main__":
    main()
