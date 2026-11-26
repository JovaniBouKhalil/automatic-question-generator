"""Resumable, paired comparisons. All generated examples (including failures) are scored."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .baseline import template_question
from .data import load_jsonl
from .metrics import aggregate, leaks_answer, lexical_relevance, quality_decision
from .models import generate_question, load_generator, load_verifier, verify_question
from .registry import GENERATORS, VERIFIER
from .text import evidence_for

METHODS = {"baseline": None, "greedy": 1, "beam4": 4, "lora_beam4": 4}


def environment_info():
    packages = {}
    for name in ["torch", "transformers", "numpy", "scikit-learn", "sacrebleu", "peft"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    root = Path(__file__).resolve().parents[1]
    source_files = list((root / "questionforge").glob("*.py")) + list((root / "scripts").glob("*.py"))
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
              for p in sorted(source_files)}
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "git_commit": commit, "source_sha256": hashes}


def fingerprint_file(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(data_path: Path, output: Path, *, methods=("baseline", "greedy", "beam4"),
                   languages=("en", "fr", "ar"), device="cpu", verify=True, limit=None,
                   checkpoint: str | None = None, seed=42):
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if method == "lora_beam4" and not checkpoint:
            raise ValueError("lora_beam4 requires --checkpoint.")
    records = [r for r in load_jsonl(data_path) if r["language"] in languages]
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive.")
        records = [r for lang in languages for r in [x for x in records if x["language"] == lang][:limit]]
    if not records:
        raise ValueError("No records match the requested languages.")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_fingerprint = None
    if checkpoint:
        checkpoint_fingerprint = {p.name: fingerprint_file(p) for p in Path(checkpoint).glob("*") if p.is_file()}
    config = {"dataset_sha256": fingerprint_file(data_path), "methods": list(methods), "languages": list(languages),
              "device": device, "verify": verify, "limit": limit, "seed": seed, "input_tokens": 384,
              "max_new_tokens": 64, "no_repeat_ngram_size": 3, "do_sample": False,
              "checkpoint": checkpoint, "checkpoint_files": checkpoint_fingerprint,
              "generators": {k: asdict(v) for k, v in GENERATORS.items()}, "verifier": asdict(VERIFIER),
              "quality_thresholds": {"min_answer_f1": 0.5, "min_span_score": 0.1},
              "version": 1}
    config["code_sha256"] = {path.name: hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                              for path in Path(__file__).parent.glob("*.py")}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("This output folder contains a different experiment. Use a new folder.")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")
    predictions_path = output / "predictions.jsonl"
    existing = []
    if predictions_path.exists():
        existing = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line]
    complete = {(r["id"], r["method"]) for r in existing}
    rows = existing[:]
    load_times = {}
    if verify:
        begin = time.perf_counter()
        load_verifier(device)
        load_times["verifier_seconds"] = time.perf_counter() - begin
    with predictions_path.open("a", encoding="utf-8") as stream:
        for language in languages:
            selected = [r for r in records if r["language"] == language]
            for method in (["reference_control", "unrelated_control"] if verify else []) + list(methods):
                pending = [r for r in selected if (r["id"], method) not in complete]
                if pending and method in METHODS and METHODS[method] is not None:
                    begin = time.perf_counter()
                    load_generator(language, device, checkpoint if method == "lora_beam4" else None)
                    load_times[f"{language}_{method}_load_seconds"] = time.perf_counter() - begin
                for index, record in enumerate(pending):
                    start = time.perf_counter()
                    row = {**record, "reference": record["question"], "method": method,
                           "cropped": False, "error": None}
                    try:
                        if method == "reference_control":
                            question = record["question"]
                        elif method == "unrelated_control":
                            question = {"en": "What is the serial number of the red submarine?",
                                        "fr": "Quel est le numéro de série du sous-marin rouge ?",
                                        "ar": "ما الرقم التسلسلي للغواصة الحمراء؟"}[language]
                        elif method == "baseline":
                            question = template_question(record["context"], record["answer"], language, record["answer_start"])
                        else:
                            generated = generate_question(record["context"], record["answer"], language,
                                answer_start=record["answer_start"], beams=METHODS[method], device=device,
                                checkpoint=checkpoint if method == "lora_beam4" else None)
                            question = generated.question
                            row.update({"cropped": generated.cropped, "input_tokens": generated.input_tokens,
                                        "model_id": generated.model_id})
                        row["question"] = question
                        row["generation_seconds"] = time.perf_counter() - start
                        row["evidence"] = evidence_for(record["context"], record["answer"], record["answer_start"]).text
                        row["lexical_relevance"] = lexical_relevance(question, row["evidence"], language)
                        row["answer_leak"] = leaks_answer(question, record["answer"], language)
                        if verify:
                            qa_started = time.perf_counter()
                            row.update(verify_question(question, record["context"], record["answer"], language, device))
                            row["verification_seconds"] = time.perf_counter() - qa_started
                            row["accepted"], row["review_reasons"] = quality_decision(row)
                    except Exception as exc:
                        # Errors remain in the denominator; do not cherry-pick successful generations.
                        row.update({"question": row.get("question", "") if verify else "", "error": f"{type(exc).__name__}: {exc}",
                                    "generation_seconds": time.perf_counter() - start, "answer_f1": 0.0,
                                    "answer_em": 0.0, "qa_score": 0.0, "qa_answer": "",
                                    "accepted": False, "lexical_relevance": 0.0, "answer_leak": False})
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    print(f"{language} {method} {index + 1}/{len(pending)}: {row['question'][:90]}"
                          + (f" ERROR {row['error']}" if row["error"] else ""), flush=True)
    (output / "load_times.json").write_text(json.dumps(load_times, indent=2), encoding="utf-8")
    return summarize(rows, output, seed)


def bootstrap_mean_ci(rows: list[dict], key="answer_f1", seed=42, repeats=1000):
    """Resample paragraph/topic groups, not correlated questions from the same passage."""
    import numpy as np

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(float(row.get(key, 0)))
    values = [grouped[k] for k in sorted(grouped)]
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    for _ in range(repeats):
        sample = [value for _ in values for value in rng.choice(values)]
        means.append(100 * sum(sample) / len(sample))
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def summarize(rows: list[dict], output: Path, seed=42):
    import pandas as pd

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["language"], row["method"]].append(row)
    summary = []
    for (language, method), values in sorted(grouped.items()):
        metrics = aggregate(values)
        metrics.update({"language": language, "method": method,
                        "errors": sum(bool(r.get("error")) for r in values),
                        "paragraph_groups": len({r["group"] for r in values})})
        if all("answer_f1" in r for r in values):
            metrics["answer_f1_ci95"] = bootstrap_mean_ci(values, seed=seed)
        summary.append(metrics)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    # Human annotation template: blanks are intentional. Automatic checks are NOT human ratings.
    human_rows = [{k: r.get(k, "") for k in ["id", "language", "method", "context", "question", "answer", "qa_answer"]}
                  | {"human_relevance_0_2": "", "human_answerability_0_2": "", "human_fluency_0_2": "",
                     "error_category": "", "reviewer_notes": ""}
                  for r in rows if r["method"] not in {"reference_control", "unrelated_control"}]
    if not (output / "human_review.csv").exists():
        pd.DataFrame(human_rows).to_csv(output / "human_review.csv", index=False, encoding="utf-8-sig")
    # Identical record IDs make a paired improvement estimate possible.
    paired = []
    for language in sorted({r["language"] for r in rows}):
        baseline = {r["id"]: r for r in rows if r["language"] == language and r["method"] == "baseline"}
        for method in ["greedy", "beam4", "lora_beam4"]:
            others = [r for r in rows if r["language"] == language and r["method"] == method]
            differences = [{"group": r["group"], "answer_f1": r.get("answer_f1", 0) - baseline[r["id"]].get("answer_f1", 0)}
                           for r in others if r["id"] in baseline and "answer_f1" in r]
            if differences:
                paired.append({"language": language, "method_minus_baseline": method, "n": len(differences),
                               "mean_answer_f1_difference_pp": 100 * sum(r["answer_f1"] for r in differences) / len(differences),
                               "group_bootstrap_ci95_pp": bootstrap_mean_ci(differences, seed=seed)})
    (output / "paired_comparisons.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    return summary
