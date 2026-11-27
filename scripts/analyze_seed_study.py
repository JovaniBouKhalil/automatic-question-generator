"""Report every planned seed and paired uncertainty without selecting on test scores."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from questionforge.experiment import bootstrap_mean_ci  # noqa: E402

SEEDS = (13, 42, 73)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    output = Path("results/seed_study")
    output.mkdir(exist_ok=True)
    training = {seed: read(f"artifacts/seed_study/seed{seed}/training_manifest.json") for seed in SEEDS}
    selected = min(SEEDS, key=lambda seed: (training[seed]["best_dev_loss"], seed))
    seed_rows, paired_summary = [], []
    for dataset in ["public", "classroom"]:
        paired = {}
        for seed in SEEDS:
            directory = Path(f"results/seed{seed}_{dataset}_test")
            summary = read(directory / "summary.json")
            base = next(r for r in summary if r["method"] == "beam4")
            adapted = next(r for r in summary if r["method"] == "lora_beam4")
            rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            if any(r.get("error") for r in rows):
                raise ValueError("Resolve experiment failures before reporting seed robustness.")
            originals = {r["id"]: r for r in rows if r["method"] == "beam4"}
            updated = {r["id"]: r for r in rows if r["method"] == "lora_beam4"}
            if set(originals) != set(updated):
                raise ValueError("Seed comparison does not have matching input IDs.")
            differences = []
            for identifier, row in updated.items():
                difference = row["answer_f1"] - originals[identifier]["answer_f1"]
                differences.append({"group": row["group"], "answer_f1": difference})
                if identifier not in paired:
                    paired[identifier] = {"group": row["group"], "values": []}
                paired[identifier]["values"].append(difference)
            seed_rows.append({"seed": seed, "dataset": dataset, "n_unique_inputs": len(originals),
                              "selected_for_optional_app_by_dev_loss": seed == selected,
                              "best_dev_loss": training[seed]["best_dev_loss"], "best_step": training[seed]["best_step"],
                              "unadapted_qa_f1": base["answer_f1"], "adapted_qa_f1": adapted["answer_f1"],
                              "qa_f1_delta_pp": adapted["answer_f1"] - base["answer_f1"],
                              "paired_topic_ci95_pp": bootstrap_mean_ci(differences, seed=42),
                              "unadapted_rouge_l": base["rouge_l"], "adapted_rouge_l": adapted["rouge_l"],
                              "qa_f1_improved_inputs": sum(r["answer_f1"] > 0 for r in differences),
                              "qa_f1_worsened_inputs": sum(r["answer_f1"] < 0 for r in differences)})
        if any(len(r["values"]) != len(SEEDS) for r in paired.values()):
            raise ValueError("All seeds must evaluate identical inputs.")
        mean_differences = [{"group": r["group"], "answer_f1": statistics.mean(r["values"])} for r in paired.values()]
        scores = [r["qa_f1_delta_pp"] for r in seed_rows if r["dataset"] == dataset]
        paired_summary.append({"dataset": dataset, "unique_test_inputs": len(paired), "training_seeds": len(SEEDS),
                               "mean_qa_f1_delta_pp": statistics.mean(scores),
                               "between_seed_sample_sd_pp": statistics.stdev(scores),
                               "topic_bootstrap_ci95_of_seed_mean_pp": bootstrap_mean_ci(mean_differences, seed=42),
                               "interpretation": "Topic bootstrap holds the three trained seeds fixed. Between-seed SD is descriptive with only three runs. Repeated IDs are not independent extra test data."})
    report = {"scope": "Prespecified follow-up: all three 160-update training runs, with development-only checkpoint and optional application selection.",
              "seeds": list(SEEDS), "steps": 160, "selected_seed_by_development_loss": selected,
              "optional_app_adapter": f"artifacts/seed_study/seed{selected}",
              "runs": seed_rows, "paired_seed_means": paired_summary,
              "training_seconds_total": sum(t["elapsed_seconds"] for t in training.values())}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, dataset in zip(axes, ["public", "classroom"]):
        rows = [r for r in seed_rows if r["dataset"] == dataset]
        values = [r["qa_f1_delta_pp"] for r in rows]
        ax.bar([str(r["seed"]) for r in rows], values, color="#205c4a", width=.6)
        intervals = [r["paired_topic_ci95_pp"] for r in rows]
        ax.errorbar(range(len(rows)), values,
                    yerr=[[max(0, v - interval[0]) for v, interval in zip(values, intervals)],
                          [max(0, interval[1] - v) for v, interval in zip(values, intervals)]],
                    fmt="none", ecolor="#173c3b", capsize=4)
        ax.axhline(0, color="#8b938c", linewidth=1)
        ax.set(title=f"{dataset.title()} test: adaptation vs pretrained", xlabel="Training seed", ylabel="QA F1 difference (percentage points)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "seed_stability.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    for seed in SEEDS:
        history = read(f"artifacts/seed_study/seed{seed}/history.json")
        events = [h for h in history if "dev_loss" in h]
        ax.plot([0, *[h["step"] for h in events]],
                [training[seed]["initial_dev_loss"], *[h["dev_loss"] for h in events]], label=f"seed {seed}")
    ax.set(xlabel="Optimizer updates", ylabel="Development token cross-entropy", title="Checkpoint selection uses development loss only")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.savefig(output / "development_loss.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
