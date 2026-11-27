"""Report measured coverage and source-linked inspections, without inventing ratings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.run_learning_focus_study import source_hash


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/ANSWER_ROLE_RESULTS.md"))
    args = parser.parse_args()
    review = read("results/learning_focus/role_final_inspection.json")
    inspected = review["paired_cases"] + review["challenge_cases"]
    assert len(inspected) == 27 and len({c["source_file"] for c in inspected}) == 27
    for item in inspected:
        assert item["source_sha256"] == source_hash(item["source_file"])
        assert item["observation"].strip()
    summaries = {}
    for name in ["educational", "classroom"]:
        summary = read(f"results/learning_role_{name}_v1/summary.json")
        assert summary["study_complete"] and summary["passage_language_pairs"] == 9
        summaries[name] = summary
    challenge = read("results/learning_challenge_v1/summary.json")
    assert challenge["study_complete"] and challenge["passage_language_pairs"] == 9
    lines = ["# Answer-role editing: results and remaining failures", "",
             "The optional 4B learning mode retains source-ID planning and native T5 drafts, then",
             "guides the editor with the target answer's role: output, purpose, limitation, criterion",
             "or explicit comparison. Qwen is used for inference; the separate English SciQ adapter",
             "is the model-training experiment. Original and smaller-model results remain available.", "",
             "## Recipe selection", "",
             "Six purposive development cases compare role-hinted editing and draft-free writing",
             "(12 responses). Both repair the inspected Arabic validation-purpose and English",
             "weight-contrast mismatches. Keeping T5 drafts preserves a more specific English",
             "voltage-effect question; draft removal does not repair Arabic conversion grammar.", "",
             "An extra Arabic grammatical-frame probe on three cases still asks for the wrong",
             "conversion role. Its additional instruction is discarded. The recorded selection",
             "uses source/target inspection, not QA-score maximization. All failed responses remain",
             "in the artifacts. See [the protocol](ANSWER_ROLE_PROTOCOL.md) and",
             "[selection record](../results/learning_focus/refinement_selection.json).", "",
             "## Fixed-target re-edit", "",
             "Every answer, offset, passage, objective and T5 draft stays fixed. The model is the",
             "same pinned Qwen3-4B-Instruct-2507 checkpoint, greedy FP32 CPU. Coverage counts below",
             "are not question accuracy, usefulness ratings or independent samples of people.", "",
             "| Synthetic development set | Cases | Fixed targets | Original 4B returned | Role-guided returned | Execution errors |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name.title()} | {s['passage_language_pairs']} | {s['fixed_planned_facts']} | "
                     f"{s['methods']['baseline_hybrid']['returned_questions']} | "
                     f"{s['methods']['role_hybrid']['returned_questions']} | {s['cases_with_execution_error']} |")
    lines += ["", "| Development set | Original QA answer F1 | Role-guided QA answer F1 |",
              "|---|---:|---:|"]
    for name, s in summaries.items():
        before = s["methods"]["baseline_hybrid"]["mean_qa_answer_f1_diagnostic"]
        after = s["methods"]["role_hybrid"]["mean_qa_answer_f1_diagnostic"]
        lines.append(f"| {name.title()} | {100 * before:.2f} | {100 * after:.2f} |")
    lines += ["", "These 0–100 diagnostic means condition on returned questions. They rise in the",
              "educational set and fall in the classroom set. Neither direction establishes a",
              "change in correctness or learning value; source inspection also finds wording regressions."]
    lines += ["", "Read every saved question in the [educational comparison](../results/learning_role_educational_v1/comparison.html)",
              "and [classroom comparison](../results/learning_role_classroom_v1/comparison.html).",
              "Conditional QA recovery remains a diagnostic; it does not select the recipe or",
              "establish that a returned question is supported.", "", "## Source-linked observations", "",
              review["scope"], ""]
    for item in review["paired_cases"]:
        case = read(item["source_file"])
        link = os.path.relpath(item["source_file"], args.output.parent).replace("\\", "/")
        lines.append(f"- [{case['case_id']}]({link}): {item['observation']}")
    lines += ["", "## Additional topic audit", "",
              "After freezing the recipe, the full automatic path runs on three additional topics",
              "in English, French and Arabic: physical separation, search limitations and greenhouse",
              "conditions. The synthetic passages were authored for this audit; they have no human",
              "reference questions or ratings. This is development transfer inspection, not an",
              "independently authored final benchmark. No prompt is retuned on these outputs.", "",
              f"Recorded {challenge['passage_language_pairs']} passage/language pairs; "
              f"{challenge['cases_with_execution_error']} execution-error cases. "
              f"The planner supplies {challenge['methods']['planned_t5']['returned_questions']} targets/drafts; "
              f"the editor returns {challenge['methods']['learning_hybrid']['returned_questions']} questions.", "",
              "[Inspect the complete additional-topic comparison](../results/learning_challenge_v1/comparison.html).", ""]
    for item in review["challenge_cases"]:
        case = read(item["source_file"])
        link = os.path.relpath(item["source_file"], args.output.parent).replace("\\", "/")
        lines.append(f"- [{case['case_id']}]({link}): {item['observation']}")
    lines += ["", "## Practical use and limits", "",
              "Choose **Learning-focused hybrid 4B (experimental)** in the app. The default native",
              "T5 mode and optional science adapter remain available. The stronger editor can take",
              "several minutes per quiz on CPU. Inspect source support and wording, especially Arabic.", "",
              "QA-flagged questions start unselected for export. The teacher can include or exclude",
              "individual questions and revise wording before export. A revision clears the original",
              "QA scores, starts unselected, and retains the generated question and model provenance.",
              "Exports include the passage and optional objective. Review choices do not remove",
              "benchmark records or change reported scores. Exact-span, leakage and",
              "format checks cannot prove a question's premise. Human learning-value ratings remain",
              "pending; no percentage of correct or useful questions is claimed.", "",
              "Reproduction commands and the three-criterion review rubric are in",
              "[the learning-mode guide](LEARNING_FOCUS.md). `scripts.validate_learning_focus` checks",
              "saved source snapshots, fixed target/draft identity and counts;",
              "`scripts.validate_role_refinement` checks the pilot, selection and replay inputs.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("Built the answer-role report from completed artifacts and source-linked notes:", args.output)


if __name__ == "__main__":
    main()
