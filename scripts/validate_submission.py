"""Audit real experiment artifacts, executed cells and test receipts before packaging."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import nbformat
import pandas as pd
from pptx import Presentation
from pypdf import PdfReader

from questionforge.experiment import fingerprint_file


def test_receipt(path):
    tree = ET.parse(path)
    suites = list(tree.getroot().iter("testsuite"))
    failed = sum(int(s.get(key, 0)) for s in suites for key in ["failures", "errors"])
    assert failed == 0, f"Failed tests in {path}"
    return sum(int(s.get("tests", 0)) - int(s.get("skipped", 0)) for s in suites)


def main():
    root = Path.cwd()
    experiments = []
    for directory in sorted((root / "results").iterdir()):
        if not (directory / "summary.csv").exists():
            continue
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        for name, expected in config["code_sha256"].items():
            actual = hashlib.sha256((root / "questionforge" / name).read_text(encoding="utf-8").encode()).hexdigest()
            assert actual == expected, (directory.name, name)
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert not any(r.get("error") for r in rows), directory
        assert len({(r["id"], r["method"]) for r in rows}) == len(rows), directory
        if config.get("checkpoint"):
            actual = {p.name: fingerprint_file(p) for p in Path(config["checkpoint"]).iterdir() if p.is_file()}
            assert actual == config["checkpoint_files"], directory
        review = pd.read_csv(directory / "human_review.csv")
        assert not review.filter(like="human_").notna().any().any(), "Legacy rating templates must not imply completed annotation"
        experiments.append({"name": directory.name, "rows": len(rows), "source_hashes_match": True})
    verification = json.loads((root / "results/verification_study/summary.json").read_text(encoding="utf-8"))
    assert sum(r["errors"] for r in verification["models"]) == 0
    config = json.loads((root / "results/verification_study/config.json").read_text(encoding="utf-8"))
    assert fingerprint_file(root / "docs/EXTENSION_PROTOCOL.md") == config["protocol_sha256"]
    for name, digest in config["source_sha256"].items():
        assert fingerprint_file(root / name) == digest
    verifier_rows = [json.loads(line) for line in (root / "results/verification_study/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(verifier_rows) == 720 and not any(r.get("error") for r in verifier_rows)
    notebook = nbformat.read(root / "notebooks/QuestionForge.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count and not any(o.output_type == "error" for o in c.outputs) for c in code)
    slides = len(Presentation(root / "presentation/QuestionForge.pptx").slides)
    pages = len(PdfReader(root / "presentation/QuestionForge.pdf").pages)
    assert slides == pages and 8 <= slides <= 12
    human = json.loads((root / "results/human_evaluation/summary.json").read_text(encoding="utf-8"))
    assert human["complete_rating_rows"] == 0 or (root / "results/human_evaluation/ratings.csv").exists()
    seed = json.loads((root / "results/seed_study/summary.json").read_text(encoding="utf-8"))
    assert seed["seeds"] == [13, 42, 73] and len(seed["runs"]) == 6
    split = json.loads((root / "results/article_sensitivity/split_manifest.json").read_text(encoding="utf-8"))
    assert split["training_heldout_article_intersection"] == []
    result = {"regular_tests_passed": test_receipt(root / "results/quality/unit_tests.xml"),
              "neural_integration_tests_passed": test_receipt(root / "results/quality/integration_tests.xml"),
              "executed_notebook_code_cells": len(code), "notebook_errors": 0,
              "presentation_slides": slides, "presentation_pdf_pages": pages,
              "generation_experiments": experiments, "generation_experiments_with_matching_core_source_hashes": len(experiments),
              "adapter_fingerprints_match_saved_configurations": True,
              "verifier_study_rows": len(verifier_rows), "verifier_study_execution_errors": 0,
              "human_evaluation_status": human["status"], "complete_human_rating_rows": human["complete_rating_rows"],
              "repeated_training_seeds": seed["seeds"], "article_isolated_training_rows": split["counts"]["train"]}
    (root / "results/validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
