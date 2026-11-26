"""Pinned datasets, exact answer-span validation and paragraph-grouped partitions."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from .text import validate_language

DATASETS = {
    "en": {"id": "rajpurkar/squad", "revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
           "license": "CC-BY-SA-4.0", "format": "parquet", "train": "plain_text/train-00000-of-00001.parquet",
           "test": "plain_text/validation-00000-of-00001.parquet"},
    "fr": {"id": "lincoln/newsquadfr", "revision": "6aca57928d2edaffa6f9a29bdecaa789c28d0391",
           "license": "CC-BY-NC-SA-4.0", "format": "json", "train": "train.json", "test": "valid.json"},
    "ar": {"id": "hsseinmz/arcd", "revision": "cc6906b6eda547e4ffc63b8d88ccca7e0515187a",
           "license": "MIT (Hub metadata; underlying Wikipedia attribution also applies)", "format": "parquet",
           "train": "plain_text/train-00000-of-00001.parquet", "test": "plain_text/validation-00000-of-00001.parquet"},
}


def context_hash(context: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", context).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_record(row: dict) -> None:
    for key in ["id", "context", "question", "answer", "answer_start", "language", "group", "source"]:
        if key not in row:
            raise ValueError(f"Missing {key} in record.")
    validate_language(row["language"])
    answer, start = row["answer"], row["answer_start"]
    if not isinstance(start, int) or start < 0 or not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Invalid answer span in {row['id']}.")
    if row["context"][start:start + len(answer)] != answer:
        raise ValueError(f"Answer offset does not match the original passage in {row['id']}.")
    if not row["question"].strip():
        raise ValueError(f"Empty reference in {row['id']}.")


def load_jsonl(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        validate_record(row)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record IDs.")
    return rows


def write_jsonl(path: str | Path, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def flatten_squad(data: dict | list) -> list[dict]:
    if isinstance(data, list):
        # Some public files are already flattened; preserve their answer lists.
        return data
    rows = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa.get("is_impossible") or not qa.get("answers"):
                    continue
                rows.append({"id": str(qa["id"]), "title": article.get("title", ""),
                             "context": paragraph["context"], "question": qa["question"],
                             "answers": {"text": [a["text"] for a in qa["answers"]],
                                         "answer_start": [a["answer_start"] for a in qa["answers"]]}})
    return rows


def download_dataset(language: str, cache: Path) -> dict[str, list[dict]]:
    import pandas as pd

    spec = DATASETS[language]
    result = {}
    for split in ["train", "test"]:
        filename = spec[split]
        target = cache / language / Path(filename).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            url = f"https://huggingface.co/datasets/{spec['id']}/resolve/{spec['revision']}/{filename}?download=true"
            # Download to a temporary file; an interrupted transfer never becomes a cached dataset.
            partial = target.with_suffix(target.suffix + ".partial")
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                        expected = int(response.headers.get("Content-Length", 0))
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                    if expected and partial.stat().st_size != expected:
                        raise IOError("Incomplete dataset download")
                    partial.replace(target)
                    break
                except Exception:
                    if attempt == 3:
                        raise
        if spec["format"] == "parquet":
            result[split] = pd.read_parquet(target).to_dict(orient="records")
        else:
            text = target.read_text(encoding="utf-8")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
            result[split] = flatten_squad(parsed)
    return result


def normalize_records(raw: list[dict], language: str) -> tuple[list[dict], Counter]:
    records, excluded = [], Counter()
    seen = set()
    for item in raw:
        answers = item.get("answers", {})
        if isinstance(answers, list):
            texts = [a["text"] for a in answers]
            starts = [a["answer_start"] for a in answers]
        else:
            texts, starts = answers.get("text", []), answers.get("answer_start", [])
        if len(texts) == 0:
            excluded["unanswerable"] += 1
            continue
        row = {"id": f"{language}:{item['id']}", "context": item["context"],
               "question": item["question"], "answer": str(texts[0]), "answer_start": int(starts[0]),
               "language": language, "group": context_hash(item["context"]),
               "title": item.get("title", ""), "source": DATASETS[language]["id"]}
        try:
            validate_record(row)
        except ValueError:
            # Never repair offsets by guessing: translated/noisy data can point to the wrong occurrence.
            excluded["invalid_span_or_record"] += 1
            continue
        signature = (row["group"], row["question"].strip(), row["answer"])
        if signature in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(signature)
        records.append(row)
    return records, excluded


def grouped_split(train: list[dict], test: list[dict], seed: int = 42, dev_fraction: float = 0.15):
    """Keep the published evaluation pool intact and exclude its paragraphs from training.

    All questions from one normalized paragraph stay in a single partition. This
    prevents a common QA leak: random row splits of different questions on one text.
    """
    if not 0 < dev_fraction < 1:
        raise ValueError("Development fraction must lie between 0 and 1.")
    test_groups = {row["group"] for row in test}
    clean_train = [row for row in train if row["group"] not in test_groups]
    groups = sorted({row["group"] for row in clean_train})
    random.Random(seed).shuffle(groups)
    dev_count = max(1, int(len(groups) * dev_fraction)) if len(groups) > 1 else 0
    dev_groups = set(groups[:dev_count])
    return {"train": [row for row in clean_train if row["group"] not in dev_groups],
            "dev": [row for row in clean_train if row["group"] in dev_groups], "test": test}, {
                "removed_training_rows_overlapping_test": len(train) - len(clean_train)}


def sample_groups(rows: list[dict], count: int, seed: int) -> list[dict]:
    """One randomly selected question per paragraph; deterministic representative subset."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)
    rng = random.Random(seed)
    groups = sorted(grouped)
    rng.shuffle(groups)
    return [rng.choice(grouped[group]) for group in groups[:count]]


def assert_disjoint(partitions: dict[str, list[dict]]):
    keys = list(partitions)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            overlap = {r["group"] for r in partitions[left]} & {r["group"] for r in partitions[right]}
            if overlap:
                raise ValueError(f"Paragraph leakage between {left} and {right}: {len(overlap)} groups")


def prepare_data(output: Path, train_size: int = 600, dev_size: int = 15, test_size: int = 40, seed: int = 42):
    manifest = {"seed": seed, "sampling": "one question per paragraph; paragraph-disjoint partitions",
                "pretrained_exposure": "These partitions prevent leakage in OUR experiments. Upstream model exposure cannot be ruled out.",
                "datasets": {}}
    all_rows = {split: [] for split in ["train", "dev", "test"]}
    for language, spec in DATASETS.items():
        print(f"Preparing {language}: {spec['id']}", flush=True)
        raw = download_dataset(language, output.parent / "downloads")
        train, rejected_train = normalize_records(raw["train"], language)
        test, rejected_test = normalize_records(raw["test"], language)
        partitions, audit = grouped_split(train, test, seed)
        assert_disjoint(partitions)
        info = {**spec, **audit, "raw_counts": {k: len(v) for k, v in raw.items()},
                "excluded_train": dict(rejected_train), "excluded_test": dict(rejected_test),
                "available_clean_counts": {k: len(v) for k, v in partitions.items()}, "selected_counts": {}}
        for split, count in [("train", train_size), ("dev", dev_size), ("test", test_size)]:
            selected = sample_groups(partitions[split], count, seed)
            all_rows[split].extend(selected)
            info["selected_counts"][split] = len(selected)
        manifest["datasets"][language] = info
    assert_disjoint(all_rows)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in all_rows.items():
        write_jsonl(output / f"{split}.jsonl", rows)
        manifest[f"{split}_sha256"] = hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
