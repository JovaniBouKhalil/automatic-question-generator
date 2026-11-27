"""Report the article-isolation sensitivity without concealing changes in training data."""

import json
from pathlib import Path

from questionforge.experiment import bootstrap_mean_ci


def main():
    results = []
    for dataset in ["public", "classroom"]:
        directory = Path(f"results/article_{dataset}_test")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        if any(r.get("error") for r in rows):
            raise ValueError("Do not summarize failed sensitivity inference as valid results.")
        original = {r["id"]: r for r in rows if r["method"] == "beam4"}
        adapted = {r["id"]: r for r in rows if r["method"] == "lora_beam4"}
        assert set(original) == set(adapted)
        differences = [{"group": r["group"], "answer_f1": r["answer_f1"] - original[i]["answer_f1"]}
                       for i, r in adapted.items()]
        results.append({"dataset": dataset, "n": len(original),
                        "mean_qa_f1_delta_pp": 100 * sum(r["answer_f1"] for r in differences) / len(differences),
                        "paired_topic_ci95_pp": bootstrap_mean_ci(differences, seed=42),
                        "method_metrics": [{k: r[k] for k in ["method", "answer_f1", "rouge_l", "chrf_pp"]}
                                           for r in summary if r["method"] in ["beam4", "lora_beam4"]]})
    output = Path("results/article_sensitivity")
    training = json.loads(Path("artifacts/article_isolated_lora/training_manifest.json").read_text(encoding="utf-8"))
    report = {"scope": "Follow-up article-isolated English training; fewer training examples, identical held-out inputs.",
              "n_training": training["counts"]["train"], "best_dev_loss": training["best_dev_loss"],
              "best_step": training["best_step"], "results": results,
              "interpretation": "This sensitivity changes training coverage as well as isolation. A single seed does not isolate a causal effect. Test scores do not choose the app adapter."}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
