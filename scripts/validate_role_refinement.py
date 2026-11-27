"""Check refinement provenance and replay inputs; never assign question-quality ratings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from questionforge.learning.focus import focused_payload
from scripts.run_learning_focus_study import source_hash, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    # JSON key order affects the language model's input, so do not sort keys.
    return json.dumps(value, ensure_ascii=False)


def main():
    from questionforge.learning.focus import REFINEMENT_SYSTEM

    pilot_path = Path("results/quality/learning_4b_role_focus_pilot.json")
    pilot = read(pilot_path)
    script = Path("results/quality/source/pilot_role_focus.py")
    assert hashlib.sha256(script.read_bytes()).hexdigest() == pilot["script_sha256"]
    responses = 0
    for case in pilot["cases"]:
        source_path = Path(case["source_case"])
        source = read(source_path)
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == case["source_sha256"]
        assert source["context"] == case["context"]
        plan = source["planning_report"]
        assert plan["facts"] == case["facts"]
        for variant in case["variants"]:
            expected = focused_payload(source["context"], source["language"], plan["facts"], plan["drafts"],
                                       plan["learning_objective"],
                                       include_drafts=variant["variant"] == "role_hinted_edit")
            assert canonical(variant["payload"]) == canonical(expected)
            responses += 1
    assert responses == 12
    probe_path = Path("results/quality/learning_4b_arabic_role_probe.json")
    probe = read(probe_path)
    for name, digest in probe["source_sha256"].items():
        assert source_hash(Path("results/quality/source/arabic_role_probe") / name) == digest
    assert len(probe["cases"]) == 3
    for case in probe["cases"]:
        assert source_hash(case["source_case"]) == case["source_sha256"]
        source = read(case["source_case"])
        plan = source["planning_report"]
        expected = focused_payload(source["context"], source["language"], plan["facts"], plan["drafts"],
                                   plan["learning_objective"])
        assert canonical(case["payload"]) == canonical(expected)
    selection = read("results/learning_focus/refinement_selection.json")
    assert selection["system"] == REFINEMENT_SYSTEM
    assert selection["selection_scope"] == "Purposive development inspection; not a fresh test or human rating."
    assert selection["pilot_sha256"] == source_hash(pilot_path)
    assert selection["arabic_probe_sha256"] == source_hash(probe_path)
    replayed = 0
    for name in ["educational", "classroom"]:
        directory = Path(f"results/learning_role_{name}_v1")
        summary = read(directory / "summary.json")
        assert summary["study_complete"] and not summary["cases_with_execution_error"]
        for path in directory.glob("*.json"):
            case = read(path)
            if "case_id" not in case:
                continue
            plan = case["planning_report"]
            expected = focused_payload(case["context"], case["language"], plan["facts"], plan["drafts"],
                                       plan["learning_objective"])
            assert canonical(case["edit_payload"]) == canonical(expected)
            replayed += 1
    assert replayed == 18
    receipt = {"pilot_responses": responses, "localized_probe_responses": 3,
               "paired_case_inputs_verified": replayed, "source_snapshots_and_selection_hashes_match": True,
               "interpretation": "Provenance and identical inputs verified. This does not establish question accuracy or usefulness."}
    write_json(Path("results/quality/answer_role_validation.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
