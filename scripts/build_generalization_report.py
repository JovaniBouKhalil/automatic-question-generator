"""Build held-out findings from sealed predictions and source-linked inspections."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_learning_focus_study import source_hash

ROOT = Path("results/generalization_v1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def figure(science):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    training = read("results/science_training/lr3e-04_r16_d0.05.json")
    losses = [r["train_loss"] for r in training["history"] if "train_loss" in r]
    checks = [r for r in training["history"] if "dev_loss" in r]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot(range(32, len(losses) + 1), np.convolve(losses, np.ones(32) / 32, mode="valid"),
                 color="#8d673a", label="Training minibatches, 32-update mean")
    axes[0].plot([r["step"] for r in checks], [r["dev_loss"] for r in checks], "o-",
                 color="#205c4a", label="Fixed validation token loss")
    axes[0].set(xlabel="Optimizer update", ylabel="Token cross-entropy", title="Validation selects weights; it does not rate questions")
    axes[0].legend(fontsize=8)
    baseline, adapted = (science["methods"][m] for m in ["original_t5", "selected_t5"])
    x = np.arange(2)
    axes[1].bar(x - .18, [baseline["bleu"], baseline["answer_f1"]], .36, color="#aac2ac", label="Original T5")
    axes[1].bar(x + .18, [adapted["bleu"], adapted["answer_f1"]], .36, color="#205c4a", label="Selected adapter")
    axes[1].set_xticks(x, ["BLEU", "QA answer F1"])
    axes[1].set(ylim=(0, 100), ylabel="Diagnostic score (0–100)", title="50 additional unused SciQ test examples")
    axes[1].legend(fontsize=8)
    delta = science["paired_qa_answer_f1_difference_pp"]
    fig.suptitle(f"QA recovery change {delta['mean']:+.2f} pp; paired 95% interval [{delta['ci95'][0]:.2f}, {delta['ci95'][1]:.2f}]", fontsize=11)
    fig.savefig(ROOT / "validation_and_holdout.png", dpi=180)
    fig.savefig(ROOT / "validation_and_holdout.pdf")
    plt.close(fig)


def main():
    science, automatic, review = (read(ROOT / name) for name in
                                  ["science_summary.json", "automatic_summary.json", "qualitative_review.json"])
    assert automatic["complete"] and len(review["automatic_cases"]) == 12
    for item in review["automatic_cases"]:
        assert source_hash(item["source_file"]) == item["source_sha256"] and item["observation"].strip()
    figure(science)
    lines = ["# Generalization audit: unused data and unchanged models", "",
             "This evaluates a frozen system after fixing a development example. The user's lecture",
             "is not a test set and no model is trained on it. No weights, prompts, candidate rules",
             "or thresholds are changed in response to this audit. See [the frozen protocol](GENERALIZATION_PROTOCOL.md).", "",
             "## Week 10 connection", "",
             "Physical slides 7/11 cover dropout and early stopping; slide 19 separates training,",
             "validation and final testing; slides 20/41/42 cover search, error analysis and fair",
             "comparison. The earlier eight-setting grid already varies learning rate, LoRA rank",
             "and dropout and saves validation-selected weights. This new audit checks unused data",
             "instead of treating a lower training/validation loss or a repaired example as proof of generalization.", "",
             "## Fixed-answer T5 generalization", "",
             "Fifty additional human-authored SciQ test pairs were excluded from every previous",
             "prepared/evaluated paragraph and reference-question pool. Both models receive identical",
             "passages, exact target answers and beam-4 decoding. The adapter was selected before",
             "this test; its weights and selection are unchanged. Original and adapted results are both retained.", "",
             "| Model | Inputs | BLEU | chrF++ | ROUGE-L | QA answer F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, score in science["methods"].items():
        lines.append(f"| {method} | {score['n']} | {score['bleu']:.2f} | {score['chrf_pp']:.2f} | {score['rouge_l']:.2f} | {score['answer_f1']:.2f} |")
    d = science["paired_qa_answer_f1_difference_pp"]
    lines += ["", f"Selected minus original QA recovery: **{d['mean']:+.2f} percentage points**, paired",
              f"paragraph-bootstrap 95% interval **[{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]**.",
              "BLEU improves while QA recovery falls descriptively; the interval includes zero.",
              "There is no established general question-quality improvement. This does not by",
              "itself diagnose overfitting: sample variation, reference style and training-objective",
              "mismatch also matter. The reference QA control shows that the judge is imperfect.", "",
              "![Validation history and unused-test diagnostics](../results/generalization_v1/validation_and_holdout.png)", "",
              "The training curve is a moving mean over changing minibatches with training-time",
              "dropout; validation is a fixed evaluation pool. Do not treat their exact difference",
              "as a calibrated generalization gap. Lower token loss is not a usefulness rating.", "",
              "### Source-linked test examples", "",
              "Purposive examples explaining successes/regressions, not a representative error-rate sample.", ""]
    for item in review["science_examples"]:
        lines.append(f"- `{item['id']}`: {item['observation']}")
    lines += ["", "## Automatic multilingual transfer", "",
              "Four previously unused human-authored paragraphs per language were sampled from",
              "pinned SQuAD/NewsQuAD-fr/TyDiQA-GoldP evaluation pools before any inference. Their",
              "paragraphs/reference questions are disjoint from all prior prepared/evaluated material.",
              "Known article titles are also excluded and distinct; French article identity is absent",
              "so article-level isolation cannot be established there. Generic headings and 68-character",
              "line wrapping test layout handling without inventing the factual content.", "",
              "No human reference question, reference answer or hand-written objective is supplied",
              "to automatic selection. Native drafts and revised questions share the selected targets.",
              "The human reference may ask about a different fact, so reference-overlap scores are",
              "not calculated for this end-to-end arm.", "",
              f"Recorded **{automatic['case_records']} cases**, **{automatic['execution_errors']} execution-error cases**; "
              f"maximum requested **{automatic['requested_maximum_questions']} questions**.", "",
              "| Stage | Returned questions | Conditional QA answer F1 |",
              "|---|---:|---:|"]
    for method, score in automatic["methods"].items():
        value = score["mean_qa_answer_f1_diagnostic"]
        lines.append(f"| {method} | {score['returned_questions']} | {100 * value:.2f} |" if value is not None else
                     f"| {method} | {score['returned_questions']} | unavailable |")
    lines += ["", "Coverage and conditional QA averages are not percentages of correct or useful questions.",
              "Omissions and failures stay in the record. Read [every source-linked comparison](../results/generalization_v1/automatic/comparison.html).", ""]
    for item in review["automatic_cases"]:
        case = read(item["source_file"])
        lines.append(f"- `{case['case_id']}` ({case['language']}): {item['observation']}")
    lines += ["", "## What the evidence supports", "",
              "The classroom fix addresses generic layout/answer-scope failures and gives a clearer",
              "choice of T5-only versus Qwen-assisted generation. It is not a topic-specific answer",
              "lookup. The frozen test records show where it transfers and where coverage/wording",
              "still fail. The adapter experiment does not establish that fine-tuning is uniformly better.", "",
              "Twelve automatic paragraphs and 50 SciQ pairs are limited evidence. Public checkpoints",
              "may have seen these corpora upstream; our split audit cannot rule that out. Semantic",
              "near-duplicates may remain. Human correctness/usefulness ratings are pending. Any",
              "future model change needs validation-based selection and another untouched final test.", ""]
    Path("docs/GENERALIZATION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    # Label the completed holdout display accurately without changing frozen inference source.
    html = ROOT / "automatic/comparison.html"
    html.write_text(html.read_text(encoding="utf-8").replace("Development examples; model methods are visible.",
                   "Sealed holdout examples; model methods are visible after inference."), encoding="utf-8")
    print("Generalization report and validation/test figure built from all completed records.")


if __name__ == "__main__":
    main()
