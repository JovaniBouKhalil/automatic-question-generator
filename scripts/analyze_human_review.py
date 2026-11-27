"""Analyze supplied human CSVs, or explicitly report that annotation is still pending."""

import argparse
import json
from pathlib import Path

from questionforge.review import ratings_csv, summarize_ratings, validate_ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings", nargs="*", type=Path)
    parser.add_argument("--directory", type=Path, default=Path("results/human_evaluation"))
    args = parser.parse_args()
    items = json.loads((args.directory / "items.json").read_text(encoding="utf-8"))
    mapping = json.loads((args.directory / "researcher_mapping.json").read_text(encoding="utf-8"))
    rows = []
    inputs = args.ratings or ([args.directory / "ratings.csv"] if (args.directory / "ratings.csv").exists() else [])
    for path in inputs:
        rows.extend(validate_ratings(path.read_text(encoding="utf-8-sig"), items))
    # Validate again across files to prevent accidentally counting a reviewer twice.
    rows = validate_ratings(ratings_csv(rows), items)
    summary = summarize_ratings(rows, items, mapping)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        (args.directory / "ratings.csv").write_text(ratings_csv(rows), encoding="utf-8-sig")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
