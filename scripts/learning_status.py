"""Read-only progress summary for the educational training and model download."""

from __future__ import annotations

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    queue = root / ".cache/learning_followups_status.json"
    if queue.exists():
        print("Evaluation queue:", json.loads(queue.read_text(encoding="utf-8"))["stage"])
    trials = []
    for path in (root / "results/science_training").glob("lr*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("complete"):
            trials.append(value)
    print(f"Science training: {len(trials)}/8 trials complete")
    for value in sorted(trials, key=lambda r: r["trial"]):
        print(f"  {value['trial']}: validation loss {value['best_dev_loss']:.4f}; saved update {value['best_step']}")
    selection = root / "results/science_training/selection.json"
    if selection.exists():
        value = json.loads(selection.read_text(encoding="utf-8"))
        print("Selected:", value["selected_trial"], "(validation only)")
    model = root / "models/Qwen--Qwen3-4B-Instruct-2507"
    receipt = model / "download_receipt.json"
    if receipt.exists():
        print("Stronger model: all weight shards downloaded and checksum verified")
    else:
        total = 8044982000
        blocks = root / ".cache/quality-model-blocks"
        downloaded = 0
        for index in range(1, 4):
            name = f"model-{index:05d}-of-00003.safetensors"
            cached = sum(p.stat().st_size for p in (blocks / name).glob("*.bin"))
            full, partial = model / name, model / (name + ".partial")
            downloaded += max(cached, full.stat().st_size if full.exists() else 0,
                              partial.stat().st_size if partial.exists() else 0)
        print(f"Stronger model download: {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB cached ({100 * downloaded / total:.1f}%)")
    for folder in ["science_evaluation_dev", "science_evaluation_test", "learning_focus_educational_4b_v1", "learning_focus_dev_4b_v1"]:
        summary = root / "results" / folder / "summary.json"
        if summary.exists():
            value = json.loads(summary.read_text(encoding="utf-8"))
            print(folder, "complete" if value.get("study_complete", True) else "partial",
                  "| execution errors:", value.get("cases_with_execution_error", "not recorded"))
        else:
            print(folder, "pending")
    print("This status command does not start or stop training. Human usefulness ratings remain pending.")


if __name__ == "__main__":
    main()
