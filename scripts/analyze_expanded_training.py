"""Keep validation-only app selection separate from measured held-out generation changes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from questionforge.experiment import bootstrap_mean_ci, fingerprint_file  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    languages, measurements = {}, []
    for language in ["en", "fr", "ar"]:
        checkpoint = f"artifacts/expanded/{language}"
        training = read(Path(checkpoint) / "training_manifest.json")
        if not training["training_complete"]:
            raise ValueError("Do not report unfinished training.")
        options = [(training["initial_dev_loss"], None), (training["best_dev_loss"], checkpoint)]
        if language == "en":
            options.append((training["previous_adapter_dev_loss"], "artifacts/seed_study/seed42"))
        # Stable ties prefer the unadapted generator, avoiding unnecessary adapter use.
        selected_loss, selected_path = min(options, key=lambda option: option[0])
        history = read(Path(checkpoint) / "history.json")
        saved_examples = sum(r["examples"] for r in history if r["step"] <= training["best_step"])
        languages[language] = {"adapter": checkpoint, "selected_app_adapter": selected_path,
                               "selected_by": "minimum token loss on the SAME expanded development rows; test scores unused",
                               "selected_app_dev_loss": selected_loss,
                               "training_examples": training["counts"]["train"], "development_examples": training["counts"]["dev"],
                               "steps": training["steps"], "best_step": training["best_step"],
                               "initial_dev_loss": training["initial_dev_loss"], "best_dev_loss": training["best_dev_loss"],
                               "previous_english_adapter_dev_loss": training["previous_adapter_dev_loss"],
                               "unique_training_ids_seen": training["unique_training_ids_seen"],
                               "examples_seen": training["examples_seen"], "training_loop_seconds": training["elapsed_seconds"],
                               "prepared_training_examples": training["prepared_counts"]["train"],
                               "saved_checkpoint_examples_seen": saved_examples,
                               "saved_checkpoint_unique_examples_seen": min(training["counts"]["train"], saved_examples),
                               "checkpoint_files": {p.name: fingerprint_file(p) for p in Path(checkpoint).iterdir() if p.is_file()}}
        for dataset in ["public", "classroom"]:
            directory = Path(f"results/expanded_{language}_{dataset}_test")
            summary = read(directory / "summary.json")
            rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            if any(r.get("error") for r in rows):
                raise ValueError("Inference errors must be resolved before interpreting improvements.")
            original = {r["id"]: r for r in rows if r["method"] == "beam4"}
            adapted = {r["id"]: r for r in rows if r["method"] == "lora_beam4"}
            if set(original) != set(adapted):
                raise ValueError("Paired comparisons need identical input IDs.")
            differences = [{"group": r["group"], "answer_f1": r["answer_f1"] - original[i]["answer_f1"]}
                           for i, r in adapted.items()]
            baseline = next(r for r in summary if r["method"] == "beam4")
            fine_tuned = next(r for r in summary if r["method"] == "lora_beam4")
            measurement = {"language": language, "dataset": dataset, "n_unique_inputs": len(original),
                                 "independent_groups": len({r["group"] for r in original.values()}),
                                 "base_qa_f1": baseline["answer_f1"], "adapted_qa_f1": fine_tuned["answer_f1"],
                                 "qa_f1_delta_pp": fine_tuned["answer_f1"] - baseline["answer_f1"],
                                 "paired_topic_ci95_pp": bootstrap_mean_ci(differences, seed=42),
                                 "base_rouge_l": baseline["rouge_l"], "adapted_rouge_l": fine_tuned["rouge_l"],
                                 "base_chrf_pp": baseline["chrf_pp"], "adapted_chrf_pp": fine_tuned["chrf_pp"],
                                 "base_answer_leak_pct": baseline["answer_leak"], "adapted_answer_leak_pct": fine_tuned["answer_leak"],
                                 "base_automatic_acceptance_pct": baseline["accepted"], "adapted_automatic_acceptance_pct": fine_tuned["accepted"],
                                 "improved_inputs": sum(r["answer_f1"] > 0 for r in differences),
                           "worsened_inputs": sum(r["answer_f1"] < 0 for r in differences)}
            if language == "en":
                previous_dir = Path(f"results/seed42_{dataset}_test")
                previous = {r["id"]: r for r in map(json.loads, (previous_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines())
                            if r["method"] == "lora_beam4"}
                if set(previous) != set(adapted):
                    raise ValueError("Previous and expanded English adapters need matching held-out inputs.")
                relative = [{"group": r["group"], "answer_f1": r["answer_f1"] - previous[i]["answer_f1"]}
                            for i, r in adapted.items()]
                measurement.update(previous_adapter_qa_f1=100 * sum(r["answer_f1"] for r in previous.values()) / len(previous),
                                   expanded_minus_previous_qa_f1_pp=100 * sum(r["answer_f1"] for r in relative) / len(relative),
                                   expanded_minus_previous_ci95_pp=bootstrap_mean_ci(relative, seed=42))
            measurements.append(measurement)
    report = {"scope": "Expanded native-language LoRA training; previously inspected held-out sets; no test-based selection.",
              "languages": languages, "evaluations": measurements,
              "interpretation": "One seed per language. Larger datasets, validation sets and training budgets change together. QA agreement and overlap do not establish factuality or fluency; human review is pending."}
    output = Path("results/expanded_training")
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for ax, dataset in zip(axes, ["public", "classroom"]):
        selected = [r for r in measurements if r["dataset"] == dataset]
        values = [r["qa_f1_delta_pp"] for r in selected]
        intervals = [r["paired_topic_ci95_pp"] for r in selected]
        ax.bar([r["language"].upper() for r in selected], values, width=.58, color="#205c4a")
        ax.errorbar(range(3), values,
                    yerr=[[max(0, v - bound[0]) for v, bound in zip(values, intervals)],
                          [max(0, bound[1] - v) for v, bound in zip(values, intervals)]],
                    fmt="none", capsize=4, ecolor="#173c3b")
        ax.axhline(0, linewidth=1, color="#8b938c")
        ax.set(title=f"{dataset.title()}: expanded adapter vs pretrained", ylabel="QA F1 difference (percentage points)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    for ax, (language, info) in zip(axes, languages.items()):
        history = read(Path(info["adapter"]) / "history.json")
        events = [r for r in history if "dev_loss" in r]
        ax.plot([0, *[r["step"] for r in events]], [info["initial_dev_loss"], *[r["dev_loss"] for r in events]], color="#205c4a")
        ax.set(title=language.upper(), xlabel="Optimizer updates", ylabel="Development cross-entropy")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "development_loss.png", dpi=180)
    plt.close(fig)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "summary.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
