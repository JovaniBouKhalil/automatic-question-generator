"""Revisit all prior beam-search qualitative cases without relabeling them as human ratings."""

import json
from pathlib import Path


def main():
    originals = json.loads(Path("results/qualitative_examples.json").read_text(encoding="utf-8"))["examples"]
    cache, comparisons = {}, []
    for original in originals:
        if original["method"] != "beam4":
            continue
        dataset = "public" if original["experiment"] == "benchmark_test" else "classroom"
        key = (original["language"], dataset)
        if key not in cache:
            path = Path(f"results/expanded_{key[0]}_{key[1]}_test/predictions.jsonl")
            cache[key] = {(r["id"], r["method"]): r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        before = cache[key][(original["id"], "beam4")]
        after = cache[key][(original["id"], "lora_beam4")]
        assert before["question"] == original["question"], "Original beam control must reproduce the frozen case"
        assert before["context"] == after["context"] == original["context"]
        assert before["answer"] == after["answer"] == original["answer"]
        comparisons.append({"id": original["id"], "language": original["language"], "dataset": dataset,
                            "context": original["context"], "answer": original["answer"], "reference": original["reference"],
                            "original_case_category": original["category"], "original_case_note": original["review"],
                            "before_question": before["question"], "after_question": after["question"],
                            "before_qa_answer": before["qa_answer"], "after_qa_answer": after["qa_answer"],
                            "before_answer_f1": before["answer_f1"], "after_answer_f1": after["answer_f1"],
                            "question_changed": before["question"] != after["question"]})
    output = {"scope": "Revisit every prior beam-4 qualitative case. Original labels describe the OLD question only; no new human labels or prevalence estimates are claimed.",
              "interpretation": "Read the source and both questions. Changed wording or higher QA agreement alone does not establish a factual repair.",
              "cases": comparisons}
    Path("results/expanded_training/before_after_cases.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(comparisons)} paired source-linked cases; original categories are not assigned to the new outputs.")


if __name__ == "__main__":
    main()
