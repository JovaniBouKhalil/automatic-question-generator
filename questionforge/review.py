"""Blinded annotation data validation and ordinal agreement; never invent ratings."""

from __future__ import annotations

import csv
import io
import itertools
import math
from collections import defaultdict

SCORES = ("relevance", "answerability", "fluency")
FIELDS = ("reviewer_id", "item_id", *SCORES, "comment")
GUIDE = {
    "relevance": "0: unrelated/wrong fact; 1: partly relevant or vague; 2: asks clearly about the intended fact.",
    "answerability": "0: unsupported/wrong premise or target; 1: ambiguous/partly supported; 2: passage clearly supports the intended answer.",
    "fluency": "0: malformed or wrong language; 1: understandable with errors; 2: natural, well-formed question in the passage language.",
}


def ratings_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def validate_ratings(text, items):
    """Allow unfinished rows, but reject unknown items, duplicate votes and invalid scores."""
    known = {item["item_id"] for item in items}
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not set(FIELDS[:-1]).issubset(reader.fieldnames or []):
        raise ValueError("CSV needs reviewer_id, item_id, relevance, answerability and fluency columns.")
    rows, seen = [], set()
    for number, original in enumerate(reader, 2):
        if None in original or any(value is None for value in original.values()):
            raise ValueError(f"Malformed CSV row {number}.")
        row = {field: original.get(field, "").strip() for field in FIELDS}
        if not row["reviewer_id"] or row["item_id"] not in known:
            raise ValueError(f"Row {number}: use a nonempty reviewer ID and a known item ID.")
        key = (row["reviewer_id"], row["item_id"])
        if key in seen:
            raise ValueError(f"Duplicate reviewer/item pair at row {number}.")
        seen.add(key)
        for score in SCORES:
            value = row[score]
            if value not in {"", "0", "1", "2"}:
                raise ValueError(f"Row {number}: {score} must be 0, 1, 2 or blank.")
            row[score] = int(value) if value else None
        rows.append(row)
    return rows


def summarize_ratings(rows, items, mapping):
    """Compute method scores from complete paired blocks, not convenient partial ratings.

    One block contains every sampled generation method for the same source input and
    reviewer. Missing ratings cannot give one method a selectively easier set of items.
    """
    from sklearn.metrics import cohen_kappa_score

    lookup = {row["item_id"]: row for row in mapping}
    if len(lookup) != len(mapping) or set(lookup) != {item["item_id"] for item in items}:
        raise ValueError("Blinded items and researcher mapping differ.")
    complete = [row for row in rows if all(row[s] is not None for s in SCORES)]
    methods = sorted({m["method"] for m in mapping})
    blocks = defaultdict(list)
    for row in complete:
        meta = lookup[row["item_id"]]
        blocks[(row["reviewer_id"], meta["dataset"], meta["source_id"])].append({**meta, **row})
    paired = [r for block in blocks.values() if sorted(r["method"] for r in block) == methods for r in block]
    aggregates = defaultdict(list)
    for row in paired:
        aggregates[(row["reviewer_id"], row["language"], row["method"])].append(row)
    means = [{"reviewer_id": reviewer, "language": language, "method": method,
              "n_complete_paired_items": len(selected),
              **{s: sum(r[s] for r in selected) / len(selected) for s in SCORES}}
             for (reviewer, language, method), selected in sorted(aggregates.items())]
    ratings_by_reviewer = defaultdict(dict)
    for row in complete:
        ratings_by_reviewer[row["reviewer_id"]][row["item_id"]] = row
    agreements = []
    for first, second in itertools.combinations(sorted(ratings_by_reviewer), 2):
        a, b = ratings_by_reviewer[first], ratings_by_reviewer[second]
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        for score in SCORES:
            left, right = [a[i][score] for i in shared], [b[i][score] for i in shared]
            # Constant identical labels have zero expected disagreement, so kappa is undefined.
            value = None
            if len(shared) >= 2 and len(set(left + right)) > 1:
                candidate = cohen_kappa_score(left, right, labels=[0, 1, 2], weights="quadratic")
                if math.isfinite(candidate):
                    value = float(candidate)
            agreements.append({"reviewers": [first, second], "dimension": score, "shared_items": len(shared),
                               "quadratic_weighted_kappa": value,
                               "exact_agreement": sum(x == y for x, y in zip(left, right)) / len(shared)})
    return {"status": "ratings_received" if complete else "pending_human_annotation",
            "prepared_items": len(items), "received_rows": len(rows), "complete_rating_rows": len(complete),
            "complete_paired_blocks": len(paired) // len(methods) if methods else 0,
            "method_means_0_to_2": means, "inter_rater_agreement": agreements,
            "interpretation": "Only genuine supplied ratings are summarized. Method means use complete paired input/reviewer blocks; small samples are descriptive."}
