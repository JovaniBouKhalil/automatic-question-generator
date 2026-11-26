"""Run the complete local experiment pipeline with explicit logs and resumable stages.

Start model downloads separately: python scripts/download_models.py
The pipeline waits for local checkpoints so it does not create duplicate Hub downloads.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from questionforge.registry import GENERATORS, VERIFIER  # noqa: E402


def wait_for(condition, description, hours=8):
    start = time.monotonic()
    while not condition():
        if time.monotonic() - start > hours * 3600:
            raise TimeoutError(f"Timed out waiting for {description}")
        print(f"Waiting for {description} ({(time.monotonic()-start)/60:.0f} min)", flush=True)
        time.sleep(45)


def weights_ready(spec):
    directory = ROOT / "models" / spec.model_id.replace("/", "--")
    return (directory / "model.safetensors").exists() or (directory / "pytorch_model.bin").exists()


def run(name, arguments, done=None, attempts=1):
    if done and (ROOT / done).exists():
        print(f"Already completed: {name}", flush=True)
        return
    logs = ROOT / "outputs/pipeline"
    logs.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        print(f"Running {name} (attempt {attempt+1})", flush=True)
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                    "TOKENIZERS_PARALLELISM": "false"})
        if result.returncode == 0:
            print(f"Completed {name}", flush=True)
            return
        print((logs / f"{name}.log").read_text(encoding="utf-8")[-6000:], flush=True)
        if attempt + 1 < attempts:
            time.sleep(30)
    raise RuntimeError(f"{name} failed; inspect outputs/pipeline/{name}.log")


def main():
    os.chdir(ROOT)
    wait_for(lambda: all(importlib.util.find_spec(package) for package in ["streamlit", "transformers", "peft", "pptx", "pandas"]), "Python dependencies")
    run("prepare", ["-m", "questionforge.cli", "prepare"], "data/processed/manifest.json", attempts=4)
    run("data_analysis", ["-m", "scripts.analyze_data"])
    shutil.copy2(ROOT / "data/processed/manifest.json", ROOT / "results/data/manifest.json")
    wait_for(lambda: weights_ready(GENERATORS["en"]), "English generator")
    run("lora_training", ["-m", "scripts.train_lora", "--steps", "40"], "artifacts/english_lora/training_manifest.json")
    wait_for(lambda: all(weights_ready(spec) for spec in list(GENERATORS.values()) + [VERIFIER]), "all generators and verifier")
    for split in ["dev", "test"]:
        for kind, data in [("benchmark", "data/processed"), ("novel", "data/novel")]:
            name = f"{kind}_{split}"
            run(name, ["-m", "questionforge.cli", "evaluate", "--data", f"{data}/{split}.jsonl",
                       "--output", f"results/{name}"], f"results/{name}/summary.json")
        name = f"lora_{split}"
        run(name, ["-m", "questionforge.cli", "evaluate", "--data", f"data/processed/{split}.jsonl",
                   "--output", f"results/{name}", "--languages", "en", "--methods", "baseline", "beam4", "lora_beam4",
                   "--checkpoint", "artifacts/english_lora"], f"results/{name}/summary.json")
    for name in ["benchmark_dev", "novel_dev", "benchmark_test", "novel_test", "lora_dev", "lora_test"]:
        run(f"plot_{name}", ["-m", "scripts.plot_results", "--results", f"results/{name}"])
    run("tests", ["-m", "pytest", "-q"])
    run("presentation", ["-m", "scripts.build_presentation"])
    (ROOT / "outputs/pipeline/completed.json").write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print("Experiment pipeline complete. Review outputs, execute notebook and package submission.", flush=True)


if __name__ == "__main__":
    main()
