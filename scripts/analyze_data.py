"""Produce dataset exploration figures and auditable descriptive statistics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from questionforge.data import assert_disjoint, load_jsonl
from questionforge.text import tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/data"))
    args = parser.parse_args()
    partitions = {split: load_jsonl(args.data_dir / f"{split}.jsonl") for split in ["train", "dev", "test"]}
    assert_disjoint(partitions)
    rows = [{"split": split, "language": r["language"], "context_words": len(tokens(r["context"])),
             "answer_words": len(tokens(r["answer"])), "question_words": len(tokens(r["question"])),
             "source": r["source"]} for split, records in partitions.items() for r in records]
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "record_statistics.csv", index=False)
    summary = []
    for (language, split), values in frame.groupby(["language", "split"]):
        summary.append({"language": language, "split": split, "n": len(values),
                        "context_words_median": float(values.context_words.median()),
                        "context_words_p95": float(np.percentile(values.context_words, 95)),
                        "answer_words_mean": float(values.answer_words.mean()),
                        "question_words_mean": float(values.question_words.mean())})
    (args.output / "statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    colors = {"en": "#205c4a", "fr": "#b68943", "ar": "#5b75a3"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for language in ["en", "fr", "ar"]:
        values = frame[(frame.language == language) & (frame.split == "test")]
        axes[0].hist(values.context_words, bins=15, alpha=.45, label=language.upper(), color=colors[language])
        axes[1].hist(values.answer_words, bins=np.arange(0, max(20, values.answer_words.max()+2)), alpha=.45,
                     label=language.upper(), color=colors[language])
    axes[0].set(xlabel="Passage length (Unicode word tokens)", ylabel="Records", title="Evaluation passage lengths")
    axes[1].set(xlabel="Target answer length (Unicode word tokens)", ylabel="Records", title="Evaluation answer lengths")
    for axis in axes:
        axis.legend()
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output / "length_distributions.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
