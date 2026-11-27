"""Reproduce the grade-strengthening follow-up studies with explicit, resumable stages."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render slides with an idle Windows PowerPoint installation.")
    args = parser.parse_args()
    jobs = [["scripts/download_models.py", "multilingual"],
            ["-m", "scripts.audit_overlap"], ["-m", "scripts.run_seed_study"],
            ["-m", "scripts.article_sensitivity"], ["-m", "scripts.analyze_seed_study"],
            ["-m", "scripts.summarize_article_study"], ["-m", "scripts.study_verifiers"],
            ["-m", "scripts.plot_verifier_study"], ["-m", "scripts.prepare_human_review"],
            ["-m", "scripts.analyze_human_review"],
            ["-m", "ruff", "check", "questionforge", "scripts", "tests", "app.py"],
            ["-m", "pytest", "-q", "--junitxml", "results/quality/unit_tests.xml"],
            ["-m", "pytest", "-m", "integration", "-o", "addopts=", "-q", "tests/test_neural_integration.py",
             "tests/test_app.py", "tests/test_review_and_verifiers.py", "--junitxml", "results/quality/integration_tests.xml"],
            ["-m", "scripts.build_notebook"], ["-m", "scripts.execute_notebook"],
            ["-m", "scripts.extend_presentation"]]
    for arguments in jobs:
        print("Running:", " ".join(arguments), flush=True)
        subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if args.render:
        if os.name != "nt":
            raise ValueError("PowerPoint rendering is Windows-only; generate the PPTX without --render.")
        subprocess.run(["powershell", "-File", str(ROOT / "scripts/render_slides.ps1")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "scripts.validate_submission"], cwd=ROOT, check=True)
    else:
        print("PPTX ready. Export the matching PDF before running scripts.validate_submission.", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.package_submission"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
