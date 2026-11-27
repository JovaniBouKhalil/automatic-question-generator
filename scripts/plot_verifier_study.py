"""Visualize positive recovery and false acceptance separately for both QA judges."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    folder = Path("results/verification_study")
    report = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    rows = [r for r in report["models"] if r["partition"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, key, title in zip(axes,
                              ["reference_f1", "negative_false_acceptance_pct"],
                              ["Correct references: answer F1 (higher better)", "Constructed negatives: false acceptance (lower better)"]):
        for offset, verifier, label, color in [(-.18, "original", "Original XLM-R", "#b68943"),
                                               (.18, "multilingual", "Multilingual mBERT", "#205c4a")]:
            values = [next(r[key] for r in rows if r["language"] == lang and r["verifier"] == verifier)
                      for lang in ["en", "fr", "ar"]]
            ax.bar(np.arange(3) + offset, values, width=.34, color=color, label=label)
        ax.set(title=title, xticks=np.arange(3), xticklabels=["EN", "FR", "AR"], ylim=(0, 105), ylabel="0–100 scale")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
