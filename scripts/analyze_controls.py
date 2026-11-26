"""Diagnose the verifier on positive/reference and synthetic unrelated controls.

These labels describe constructed controls, not human-rated generated-question quality.
The diagnostic is descriptive and does not tune the fixed review thresholds on test data.
"""
import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.results / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    diagnostics = []
    for language in ["en", "fr", "ar"]:
        reference = [r for r in rows if r["language"] == language and r["method"] == "reference_control"]
        unrelated = [r for r in rows if r["language"] == language and r["method"] == "unrelated_control"]
        if not reference or not unrelated:
            continue
        combined = reference + unrelated
        score_auc = roc_auc_score([1] * len(reference) + [0] * len(unrelated), [r.get("qa_score", 0) for r in combined])
        rates = {}
        for name, selected in [("reference", reference), ("unrelated", unrelated)]:
            rates[name] = {"n": len(selected), "empty_answer_pct": 100 * sum(not r.get("qa_answer") for r in selected) / len(selected),
                           "target_f1_mean": 100 * sum(r.get("answer_f1", 0) for r in selected) / len(selected),
                           "target_f1_at_least_half_pct": 100 * sum(r.get("answer_f1", 0) >= .5 for r in selected) / len(selected),
                           "automatic_acceptance_pct": 100 * sum(r.get("accepted", False) for r in selected) / len(selected)}
        diagnostics.append({"language": language, "qa_span_score_control_auroc": float(score_auc), **rates})
    result = {"interpretation": "Control discrimination only. AUROC is not generated-question accuracy or calibration; the unrelated negatives are synthetic and easy.",
              "diagnostics": diagnostics}
    (args.results / "control_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
