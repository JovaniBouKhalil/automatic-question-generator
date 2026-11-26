"""Create static figures directly from measured experiment outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"baseline": "#9da89c", "greedy": "#b68943", "beam4": "#205c4a", "lora_beam4": "#5b75a3"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    languages = sorted({r["language"] for r in summary}, key=["en", "fr", "ar"].index)
    methods = [m for m in COLORS if any(r["method"] == m for r in summary)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, metric, title in zip(axes, ["chrf_pp", "rouge_l", "answer_f1"],
                                   ["Question agreement · chrF++", "Question agreement · ROUGE-L", "Round-trip QA answer F1"]):
        x = np.arange(len(languages))
        width = .8 / max(1, len(methods))
        for i, method in enumerate(methods):
            values, low, high = [], [], []
            for language in languages:
                row = next((r for r in summary if r["language"] == language and r["method"] == method), {})
                value = row.get(metric, 0)
                values.append(value)
                interval = row.get("answer_f1_ci95", [value, value]) if metric == "answer_f1" else [value, value]
                low.append(max(0, value - interval[0]))
                high.append(max(0, interval[1] - value))
            axis.bar(x + (i - (len(methods)-1)/2) * width, values, width,
                     label=method, color=COLORS[method], yerr=[low, high], capsize=2)
        axis.set_xticks(x, [language.upper() for language in languages])
        axis.set(title=title, ylim=(0, 100), ylabel="Score (0–100)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{args.results.name} · identical examples across methods", fontsize=14)
    fig.text(.5, .01, "QA error bars: 95% paragraph-group bootstrap intervals. QA agreement is an imperfect proxy for factuality.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(args.results / "comparison.png", dpi=180)
    plt.close(fig)
    if any(r["method"] == "reference_control" for r in summary):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for i, method in enumerate(["reference_control", "unrelated_control"]):
            values = [next((r.get("answer_f1", 0) for r in summary if r["language"] == lang and r["method"] == method), 0)
                      for lang in languages]
            axis.bar(np.arange(len(languages)) + (i - .5) * .32, values, .32,
                     color=["#205c4a", "#b68943"][i], label=method)
        axis.set_xticks(np.arange(len(languages)), [lang.upper() for lang in languages])
        axis.set(ylabel="QA target-answer F1 (0–100)", ylim=(0, 100), title="Verifier controls: reference vs unrelated questions")
        axis.legend()
        axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.results / "verifier_controls.png", dpi=180)
        plt.close(fig)
    rows = [json.loads(line) for line in (args.results / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = []
    for method in methods:
        candidates = [r for r in rows if r["method"] == method]
        # Fixed ranking by verifier agreement; these are automatic illustrations for human review.
        ordered = sorted(candidates, key=lambda r: (r.get("answer_f1", 0), r["id"]))
        for category, selected in [("lowest QA agreement", ordered[:3]), ("highest QA agreement", ordered[-3:])]:
            for row in selected:
                errors.append({"selection": category, **row})
    (args.results / "review_examples.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Figures and ranked review examples saved in {args.results}")


if __name__ == "__main__":
    main()
