"""Run the frozen paired re-edits and additional synthetic audit sequentially."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.run_learning_focus_study import write_json


def main():
    status = Path(".cache/refinement_status.json")
    status.parent.mkdir(exist_ok=True)
    tasks = [
        (["scripts.reedit_saved_learning", "--source", "results/learning_focus_educational_4b_v1",
          "--output", "results/learning_role_educational_v1", "--variant", "edit"],
         "results/learning_role_educational_v1"),
        (["scripts.reedit_saved_learning", "--source", "results/learning_focus_dev_4b_v1",
          "--output", "results/learning_role_classroom_v1", "--variant", "edit"],
         "results/learning_role_classroom_v1"),
        (["scripts.run_learning_focus_study", "--source", "educational", "--planner", "quality",
          "--data", "data/authoring/learning_challenge.json", "--output", "results/learning_challenge_v1"],
         "results/learning_challenge_v1"),
    ]
    try:
        for args, folder in tasks:
            stage = "Running " + " ".join(args)
            write_json(status, {"stage": stage, "updated_unix_seconds": time.time()})
            print(stage, flush=True)
            subprocess.run([sys.executable, "-X", "utf8", "-u", "-m", *args], check=True)
            summary = json.loads(Path(folder, "summary.json").read_text(encoding="utf-8"))
            if not summary["study_complete"] or summary["cases_with_execution_error"]:
                raise RuntimeError(f"Inspect the incomplete/error records in {folder} before continuing.")
        write_json(status, {"stage": "Comparisons complete; source inspection and final verification still required.",
                            "updated_unix_seconds": time.time()})
    except Exception as exc:
        write_json(status, {"stage": f"Stopped: {type(exc).__name__}: {exc}",
                            "updated_unix_seconds": time.time()})
        raise


if __name__ == "__main__":
    main()
