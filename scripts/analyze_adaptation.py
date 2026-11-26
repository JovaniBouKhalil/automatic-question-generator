"""Paired LoRA-vs-original answer agreement, plus an honest loss-history figure."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from questionforge.experiment import bootstrap_mean_ci

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ["lora_test", "lora_novel_test"]:
        directory = ROOT / "results" / name
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        original = {r["id"]: r for r in rows if r["method"] == "beam4"}
        adapted = {r["id"]: r for r in rows if r["method"] == "lora_beam4"}
        if original.keys() != adapted.keys():
            raise ValueError("Adaptation comparison must use identical IDs.")
        differences = [{"id": key, "group": original[key]["group"],
                        "answer_f1": adapted[key]["answer_f1"] - original[key]["answer_f1"]} for key in original]
        result = {"n": len(differences),
                  "mean_answer_f1_change_pp": 100 * sum(r["answer_f1"] for r in differences) / len(differences),
                  "paired_group_bootstrap_ci95_pp": bootstrap_mean_ci(differences),
                  "improved_examples": sum(r["answer_f1"] > 1e-12 for r in differences),
                  "worsened_examples": sum(r["answer_f1"] < -1e-12 for r in differences),
                  "unchanged_examples": sum(abs(r["answer_f1"]) <= 1e-12 for r in differences),
                  "interpretation": "Small paired subset. Intervals include sampling variation only; no test-based model selection."}
        (directory / "adaptation_delta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(name, json.dumps(result))
    directory = ROOT / "artifacts/english_lora"
    manifest = json.loads((directory / "training_manifest.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    dev = [(0, manifest["initial_dev_loss"])] + [(r["step"], r["dev_loss"]) for r in history if "dev_loss" in r]
    fig, axis = plt.subplots(figsize=(8, 4.2))
    axis.plot([r["step"] for r in history], [r["train_loss"] for r in history], color="#b68943", alpha=.8, label="Training microbatch mean (dropout on)")
    axis.plot([x[0] for x in dev], [x[1] for x in dev], "o-", color="#205c4a", label="Development token loss (dropout off)")
    axis.set(xlabel="Optimizer update", ylabel="Target-token cross-entropy", ylim=(0, 3),
             title="Bounded English LoRA adaptation · CPU · 40 updates")
    axis.legend(fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figure_path = ROOT / "results/data/lora_loss_curve.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
