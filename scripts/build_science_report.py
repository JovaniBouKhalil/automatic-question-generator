"""Build educational training figures and reports exclusively from measured outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.run_learning_focus_study import render_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=Path("results/science_training"))
    parser.add_argument("--evaluation-prefix", type=Path, default=Path("results/science_evaluation"))
    parser.add_argument("--output", type=Path, default=Path("docs/SCIENCE_TRAINING_RESULTS.md"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.training_dir
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    assert selection["complete"] and selection["trials"] == 8
    trials = sorted([json.loads(p.read_text(encoding="utf-8")) for p in root.glob("lr*.json")], key=lambda r: r["trial"])
    assert len(trials) == 8 and all(r["complete"] for r in trials)
    labels = [f"{r['learning_rate']:.0e}\nr={r['rank']}\nd={r['dropout']:.2f}" for r in trials]
    colors = ["#205c4a" if r["trial"] == selection["selected_trial"] else "#aac2ac" for r in trials]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    axes[0].bar(range(8), [r["best_dev_loss"] for r in trials], color=colors)
    axes[0].axhline(selection["baseline_dev_loss"], color="#8d673a", linestyle="--", label="Unadapted T5")
    axes[0].set_xticks(range(8), labels, fontsize=8)
    axes[0].set_ylabel("Validation token cross-entropy (lower is better)")
    axes[0].set_title("All eight equal-budget trials")
    axes[0].set_ylim(0, max(selection["baseline_dev_loss"] * 1.12, 1.3))
    axes[0].legend(fontsize=9)
    for trial in trials:
        checks = [r for r in trial["history"] if "dev_loss" in r]
        axes[1].plot([r["step"] for r in checks], [r["dev_loss"] for r in checks], marker="o", markersize=3,
                     linewidth=2 if trial["trial"] == selection["selected_trial"] else 1, alpha=.9,
                     label=f"lr={trial['learning_rate']:.0e}, r={trial['rank']}, d={trial['dropout']:.2f}")
    axes[1].set_xlabel("Optimizer updates")
    axes[1].set_ylabel("Validation token cross-entropy")
    axes[1].set_title("Checkpoint selection uses validation only")
    axes[1].legend(fontsize=7)
    fig.suptitle("English SciQ adaptation: lower loss is not proof of more useful questions", fontsize=12)
    fig.savefig(root / "validation_search.png", dpi=180)
    fig.savefig(root / "validation_search.pdf")
    plt.close(fig)
    lines = ["# Educational science T5 training", "",
             "This experiment changes the English training domain to human-authored science questions and",
             "systematically searches three adapter hyperparameters. It does not train French/Arabic or",
             "establish human-rated usefulness. The separate Qwen planner/editor is inference-only.", "",
             "Source: [AllenAI SciQ](https://huggingface.co/datasets/allenai/sciq), Welbl, Liu and Gardner (2017),",
             "CC BY-NC 3.0, revision `2c94ad3e1aafab77146f384e23536f97a4849815`. See",
             "[the frozen protocol](SCIENCE_TRAINING_PROTOCOL.md). The interrupted preflight attempt is preserved",
             "in `results/science_training_aborted_v1`; no weights from it enter this study.", "",
             "## Training and selection", "",
             "Prepared: 2,048 training / 128 validation / 200 reserved test examples, one per support passage.",
             f"Usable token-budget examples: {trials[0]['counts']['train']:,} train / {trials[0]['counts']['dev']} validation.",
             "Full raw published validation/test support passages and question strings remain protected,",
             "even when an answer annotation is unusable. Exact duplicate support/questions are removed.",
             "Public checkpoint pretraining exposure and semantic near-duplicates cannot be excluded.", "",
             "All eight trials: seed 42, 256 updates, nominal effective batch 8, FP32 CPU, q/v LoRA,",
             "alpha = 2 * rank, AdamW, 16 warmup updates, validation at steps 0/128/256. Same example order.",
             "An overlong question is excluded, never silently truncated; the input window retains the answer.", "",
             "| Learning rate | Rank | Dropout | Saved step | Validation loss | Distinct training rows: whole run |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in trials:
        lines.append(f"| {r['learning_rate']:.0e} | {r['rank']} | {r['dropout']:.2f} | {r['best_step']} | {r['best_dev_loss']:.4f} | {r['unique_training_ids_seen']:,} |")
    lines += ["", f"Selected `{selection['selected_trial']}` at update {selection['selected_step']}; validation loss",
              f"{selection['baseline_dev_loss']:.4f} → {selection['selected_dev_loss']:.4f}.",
              "Selection includes the unadapted step-zero model. One seed and one budget per setting do not",
              "establish a globally optimal configuration or isolate each hyperparameter's causal effect.", "",
              "![Measured validation search](" + os.path.relpath(root / "validation_search.png", args.output.parent).replace("\\", "/") + ")", "",
              "## Generation after selection", "",
              "Original and selected T5 receive identical exact answers/passages and beam-4 decoding.",
              "Question/answer pairs differ from the original project tests. These are public published",
              "SciQ splits, not guaranteed-unseen checkpoint pretraining data. No test result reselects weights.", "",
              "| Split / method | Inputs | BLEU | chrF++ | ROUGE-L | QA answer F1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for split in ["dev", "test"]:
        folder = Path(str(args.evaluation_prefix) + "_" + split)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        for method, score in summary["methods"].items():
            lines.append(f"| {split} / {method} | {score['n']} | {score['bleu']:.2f} | {score['chrf_pp']:.2f} | {score['rouge_l']:.2f} | {score['answer_f1']:.2f} |")
        diff = summary["paired_qa_answer_f1_difference_pp"]
        lines += ["", f"{split}: selected minus original QA answer F1 {diff['mean']:+.2f} percentage points;",
                  f"paired paragraph-bootstrap 95% interval [{diff['ci95'][0]:.2f}, {diff['ci95'][1]:.2f}].", ""]
        if split == "dev":
            predictions = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            ids = list(dict.fromkeys(r["id"] for r in predictions))
            cases = [{"case_id": key, "language": "en", "context": next(r["context"] for r in predictions if r["id"] == key),
                      "outputs": {method: [r for r in predictions if r["id"] == key and r["method"] == method]
                                  for method in ["original_t5", "selected_t5"]}} for key in ids]
            render_report(folder, cases, write_template=True, methods=["original_t5", "selected_t5"])
    review_path = root / "qualitative_development_review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        lines += ["## Source-linked development inspection", "", review["scope"], ""]
        lines += [f"- `{case['id']}`: {case['observation']}" for case in review["cases"]]
        lines += [""]
    lines += ["QA F1 measures answer recovery, not question accuracy or educational value. Reference controls",
              "show the checker's performance on human questions from this domain. Overlap metrics depend on",
              "one reference wording. Inspect `results/science_evaluation_dev/comparison.html` for every saved",
              "development example; the review template is intentionally empty. Human ratings are pending.", "",
              "## Reproduce on your computer", "", "```powershell", "python -m scripts.prepare_science_data",
              "python -m scripts.train_science_grid --output outputs/science_reproduction --adapters outputs/science_adapters",
              "python -m scripts.evaluate_science --split dev --training-dir outputs/science_reproduction --output outputs/science_eval_dev",
              "python -m scripts.evaluate_science --split test --training-dir outputs/science_reproduction --output outputs/science_eval_test",
              "python -m scripts.build_science_report --training-dir outputs/science_reproduction --evaluation-prefix outputs/science_eval --output outputs/science_report.md", "```", "",
              "Do not run a second training process concurrently. Completed compatible trials are reused;",
              "mid-trial optimizer resume is not implemented. Preserve an interrupted adapter rather than",
              "silently overwrite it. Weights and exact source snapshots accompany all completed trials.", "",
              "In the app, enable **Use adapted English generator**, then choose **Educational science**",
              "under **English adaptation data**. It is optional; original and general-reading adapters remain",
              "available. Compare actual questions rather than assume that the validation winner is best for",
              "every subject or learning goal.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("Science report, validation figures and development comparison written.")


if __name__ == "__main__":
    main()
