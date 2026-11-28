"""Seal unused human-authored evaluation rows before running the frozen audit."""

from __future__ import annotations

import hashlib
import json
import random
import textwrap
import unicodedata
from pathlib import Path

from questionforge.data import DATASETS, context_hash, flatten_squad, normalize_records
from scripts.prepare_expanded_data import TYDI
from scripts.prepare_science_data import convert
from scripts.run_learning_focus_study import source_hash, write_json

SEED = 20261128
TARGET = Path("data/processed/generalization_v1")
RESULTS = Path("results/generalization_v1")


def key(text):
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def protection():
    groups, questions, titles, inputs = set(), set(), set(), {}

    def visit(value, language=None):
        if isinstance(value, list):
            for item in value:
                visit(item, language)
        elif isinstance(value, dict):
            if isinstance(value.get("language"), str):
                language = value["language"]
            for name in ["context", "source_context", "original_context"]:
                if isinstance(value.get(name), str) and value[name].strip():
                    groups.add(context_hash(value[name]))
            for name in ["question", "reference", "reference_question"]:
                if isinstance(value.get(name), str) and value[name].strip():
                    questions.add(key(value[name]))
            if language in {"en", "fr", "ar"} and isinstance(value.get("title"), str) and value["title"].strip():
                titles.add((language, key(value["title"])))
            for item in value.values():
                if isinstance(item, (list, dict)):
                    visit(item, language)

    paths = sorted([*Path("data/processed").rglob("*.jsonl"), *Path("results").rglob("*.jsonl"),
                    *Path("results").rglob("*.json")])
    for path in paths:
        if TARGET in path.parents or RESULTS in path.parents:
            continue
        inputs[path.as_posix()] = source_hash(path)
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for line in content.splitlines():
                if line.strip():
                    visit(json.loads(line))
        else:
            visit(json.loads(content))
    private = Path(".cache/user_reported_passage_fr.txt")
    if private.exists():
        groups.add(context_hash(private.read_text(encoding="utf-8")))
    return groups, questions, titles, inputs


def sample_unseen(rows, language, groups, questions, titles, count, seed):
    eligible = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        if (row["group"] in groups or key(row["question"]) in questions
                or (language, key(row.get("title", ""))) in titles
                or not 100 <= len(row["context"]) <= 1500):
            continue
        eligible.setdefault(row["group"], []).append(row)
    order = sorted(eligible)
    rng = random.Random(seed)
    rng.shuffle(order)
    selected, used_titles = [], set()
    for group in order:
        options = eligible[group]
        title = key(options[0].get("title", ""))
        if title and title in used_titles:
            continue
        selected.append(rng.choice(options))
        if title:
            used_titles.add(title)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible distinct-title {language} cases; preserve the protocol and investigate.")
    return selected, len(eligible)


def main():
    import pandas as pd

    if TARGET.exists() or RESULTS.exists():
        raise ValueError("Holdout already exists. Preserve it; do not resample after inspecting results.")
    groups, questions, titles, inputs = protection()
    sources = {"en": Path("data/downloads/en/validation-00000-of-00001.parquet"),
               "fr": Path("data/downloads/fr/valid.json"),
               "ar": Path("data/downloads/tydiqa_goldp/test.parquet")}
    cases, metadata = [], {}
    headers = {"en": "CHAPTER\nNOTES", "fr": "CHAPITRE\nNOTES", "ar": "الفصل\nملاحظات"}
    for language, path in sources.items():
        if path.suffix == ".parquet":
            raw = pd.read_parquet(path).to_dict(orient="records")
        else:
            text = path.read_text(encoding="utf-8")
            try:
                raw = flatten_squad(json.loads(text))
            except json.JSONDecodeError:
                raw = [json.loads(line) for line in text.splitlines() if line.strip()]
        if language == "ar":
            raw = [row for row in raw if str(row["id"]).startswith("arabic-")]
        rows, excluded = normalize_records(raw, language)
        spec = TYDI if language == "ar" else DATASETS[language]
        for row in rows:
            row["source"] = spec["id"]
        selected, available = sample_unseen(rows, language, groups, questions, titles, 4, SEED)
        for row in selected:
            cases.append({**row, "slide_text": headers[language] + "\n\n" +
                          textwrap.fill(row["context"], width=68, break_long_words=False, break_on_hyphens=False)})
        metadata[language] = {"source": spec, "cached_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "valid_rows": len(rows), "source_exclusions": dict(excluded),
                              "eligible_unseen_paragraphs": available, "selected_ids": [r["id"] for r in selected],
                              "known_titles_available": all(bool(key(r.get("title", ""))) for r in selected)}
    science = Path(".cache/science-source/test-00000-of-00001.parquet")
    science_manifest = json.loads(Path("results/science_training/data_manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(science.read_bytes()).hexdigest()
    assert digest == science_manifest["source_sha256"]["data/test-00000-of-00001.parquet"]
    pool, exclusions = convert(pd.read_parquet(science).to_dict(orient="records"), "test")
    fresh = [r for r in sorted(pool, key=lambda r: r["id"]) if r["group"] not in groups and key(r["question"]) not in questions]
    random.Random(SEED).shuffle(fresh)
    if len(fresh) < 50:
        raise ValueError("Fewer than 50 unused SciQ test cases remain; do not silently relax exclusions.")
    fresh = fresh[:50]
    TARGET.mkdir(parents=True)
    RESULTS.mkdir(parents=True)
    for name, rows in [("automatic", cases), ("science", fresh)]:
        (TARGET / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
        (RESULTS / f"{name}_data.jsonl").write_bytes((TARGET / f"{name}.jsonl").read_bytes())
    write_json(RESULTS / "sealed_protection.json", {
        "paragraph_hashes": sorted(groups),
        "reference_question_hashes": sorted(hashlib.sha256(q.encode()).hexdigest() for q in questions),
        "known_titles": sorted(titles)})
    manifest = {"seed": SEED, "automatic_cases": len(cases), "science_fixed_answer_cases": len(fresh),
                "datasets": metadata, "science_source_sha256": digest, "science_source_exclusions": exclusions,
                "science_selected_ids": [r["id"] for r in fresh],
                "protection": {"paragraph_hashes": len(groups), "question_keys": len(questions), "known_titles": len(titles)},
                "protection_input_sha256_normalized_utf8": inputs,
                "protocol_sha256": source_hash("docs/GENERALIZATION_PROTOCOL.md"),
                "data_sha256": {name: source_hash(TARGET / f"{name}.jsonl") for name in ["automatic", "science"]},
                "our_training_or_prior_evaluation_overlap": False,
                "upstream_pretrained_exposure": "Unknown; public datasets may have appeared in upstream training.",
                "test_used_for_tuning": False}
    write_json(TARGET / "manifest.json", manifest)
    write_json(RESULTS / "data_manifest.json", manifest)
    print(json.dumps({"automatic_cases": len(cases), "science_fixed_answer_cases": len(fresh),
                      "eligible_paragraphs": {lang: v["eligible_unseen_paragraphs"] for lang, v in metadata.items()},
                      "test_used_for_tuning": False}, indent=2))


if __name__ == "__main__":
    main()
