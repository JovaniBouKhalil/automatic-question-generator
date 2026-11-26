"""Print low-agreement examples by language for source-based qualitative inspection."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--method", default="beam4")
    parser.add_argument("--per-language", type=int, default=2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.results / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for language in ["en", "fr", "ar"]:
        selected = sorted([r for r in rows if r["method"] == args.method and r["language"] == language],
                          key=lambda r: (r.get("answer_f1", 0), r["id"]))[:args.per_language]
        for row in selected:
            fields = ["id", "language", "question", "reference", "answer", "qa_answer", "answer_f1", "evidence", "context"]
            print(json.dumps({k: row.get(k) for k in fields}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
