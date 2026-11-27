"""Show actual local training progress without loading models or optimizer tensors."""

import json
from pathlib import Path

from scripts.train_expanded_lora import BUDGETS


def status():
    results = []
    for language, (maximum, _) in BUDGETS.items():
        manifest = Path(f"artifacts/expanded/{language}/training_manifest.json")
        if manifest.exists():
            try:
                completed = json.loads(manifest.read_text(encoding="utf-8"))
                if completed.get("training_complete"):
                    results.append({"language": language, "status": "complete", "updates": completed["steps"],
                                    "maximum_updates": maximum, "best_step": completed["best_step"],
                                    "best_dev_loss": completed["best_dev_loss"],
                                    "unique_training_examples_seen": completed["unique_training_ids_seen"]})
                    continue
            except json.JSONDecodeError:
                pass
        candidates = [Path("outputs/pipeline") / f"{name}.log" for name in
                      [f"expanded_resume_{language}", f"expanded_training_{language}_native",
                       f"expanded_training_{language}_clean", f"expanded_training_{language}"]]
        candidates = [p for p in candidates if p.exists()]
        updates, checks = [], []
        if candidates:
            path = max(candidates, key=lambda p: p.stat().st_mtime)
            data = path.read_bytes()
            text = data.decode("utf-16" if data.startswith(b"\xff\xfe") else "utf-8", errors="replace")
            for line in text.splitlines():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and "step" in row:
                        updates.append(row)
                        if "dev_loss" in row:
                            checks.append(row)
                except json.JSONDecodeError:
                    pass
        results.append({"language": language, "status": "running" if updates else "initializing_or_waiting",
                        "updates": updates[-1]["step"] if updates else 0, "maximum_updates": maximum,
                        "latest_dev_loss": checks[-1]["dev_loss"] if checks else None})
    return results


if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
