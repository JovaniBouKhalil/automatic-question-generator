"""Queue local evaluations after the already-running training/download finish.

This runner never changes application defaults or invents human ratings. It saves all
experiments for inspection. Start only one copy; do not interrupt a training trial.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def ready(path, predicate):
    try:
        return predicate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        return False


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    progress = root / ".cache/learning_followups_status.json"
    progress.parent.mkdir(parents=True, exist_ok=True)

    def stage(message):
        progress.write_text(json.dumps({"stage": message, "updated_unix_seconds": time.time()}, indent=2), encoding="utf-8")
        print(message, flush=True)

    def run(module, *args):
        stage("Running " + module + " " + " ".join(args))
        subprocess.run([sys.executable, "-u", "-m", module, *args], check=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    stage("Waiting for the eight science training trials to finish")
    selection = Path("results/science_training/selection.json")
    while not ready(selection, lambda d: d["complete"] and d["trials"] == 8):
        time.sleep(15)
    run("scripts.evaluate_science", "--split", "dev")
    run("scripts.evaluate_science", "--split", "test")
    stage("Waiting for all stronger-model shards to pass checksum verification")
    receipt = Path("models/Qwen--Qwen3-4B-Instruct-2507/download_receipt.json")
    while not ready(receipt, lambda d: d["all_weights_verified"]):
        time.sleep(15)
    control = Path("results/quality/learning_4b_runtime_control.json")
    if not control.exists():
        run("scripts.diagnose_quality_model", "--output", control.as_posix())
    if not ready(control, lambda d: len(d["probes"]) == 6):
        raise ValueError("The stronger model control is partial. Preserve it and inspect before continuing.")
    # The published comparison uses the safe FP32 runtime. FP16 remains diagnostic.
    run("scripts.run_learning_focus_study", "--source", "educational", "--planner", "quality",
        "--output", "results/learning_focus_educational_4b_v1")
    run("scripts.run_learning_focus_study", "--source", "classroom", "--planner", "quality",
        "--output", "results/learning_focus_dev_4b_v1")
    stage("Evaluations complete; inspect actual source-question relations before recommending a mode")


if __name__ == "__main__":
    main()
