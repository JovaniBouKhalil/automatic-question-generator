"""Freeze original parallel classroom scenarios before running the models.

These small, AI-assisted scenarios are a controlled supplement, not a replacement
for human-annotated benchmarks. Fictional classroom events are facts *within the
passage*. Translations share a topic group and never cross development/test splits.
"""

import json
from pathlib import Path

from questionforge.data import assert_disjoint, validate_record, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main():
    scenarios = json.loads((ROOT / "data/authoring/scenarios.json").read_text(encoding="utf-8"))
    partitions = {"dev": [], "test": []}
    for scenario in scenarios:
        for language in ["en", "fr", "ar"]:
            context, questions = scenario[language]
            for index, (question, answer) in enumerate(questions):
                row = {"id": f"novel:{scenario['topic']}:{language}:{index}", "context": context,
                       "question": question, "answer": answer, "answer_start": context.find(answer),
                       "language": language, "group": f"novel:{scenario['topic']}",
                       "source": "original_parallel_classroom_scenarios", "split": scenario["split"]}
                validate_record(row)
                partitions[scenario["split"]].append(row)
    assert_disjoint(partitions)
    for split, rows in partitions.items():
        write_jsonl(ROOT / f"data/novel/{split}.jsonl", rows)
        print(f"{split}: {len(rows)} records, {len({r['group'] for r in rows})} scenario groups")


if __name__ == "__main__":
    main()
