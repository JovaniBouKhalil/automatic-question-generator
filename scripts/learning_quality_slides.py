"""Add the completed learning-mode path to a preserved, explicit presentation copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation

from scripts.extend_presentation import notes_frame
from scripts.science_training_slides import replace_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [json.loads(Path(f"results/learning_role_{name}_v1/summary.json").read_text(encoding="utf-8"))
                 for name in ["educational", "classroom"]]
    if not all(s["study_complete"] and not s["cases_with_execution_error"] for s in summaries):
        raise ValueError("Finish both paired re-edits before updating their presentation.")
    deck = Presentation(args.source)
    if len(deck.slides) != 12:
        raise ValueError("Expected the preserved 12-slide deck.")
    flow, conclusion = deck.slides[1], deck.slides[11]
    material = next(s for s in flow.shapes if s.has_text_frame and s.text.startswith("Input: educational passage"))
    replace_text(material, "Input: educational passage + exact answer span\n"
                 "Output: question + answer + source evidence + review signals\n"
                 "Learning mode: source fact planning → T5 drafts → 4B editing")
    outcome = next(s for s in conclusion.shapes if s.has_text_frame
                   and s.text.startswith("Science adaptation improved reference-wording agreement."))
    replace_text(outcome, "Science adaptation improved reference-wording agreement.\n"
                 "4B edits repair some relations; Arabic still needs review.\n"
                 "Human educational-value ratings remain pending.")
    facts = sum(s["fixed_planned_facts"] for s in summaries)
    returned = sum(s["methods"]["role_hybrid"]["returned_questions"] for s in summaries)
    note = ("\n\nAnswer-role extension: T5/mT5/AraT5 remains the course-aligned encoder-decoder. "
            "Qwen3-4B-Instruct-2507 is a separate, untrained-here decoder-only planner/editor; "
            "the optional independent QA checker is another model, not a correctness guarantee. "
            f"A development re-edit holds {facts} targets/drafts fixed over 18 synthetic language/passage cases, "
            f"returning {returned} questions. This is not a fresh test or human quality score. "
            "Read docs/LEARNING_FOCUS.md, docs/ANSWER_ROLE_PROTOCOL.md and the paired comparison pages. "
            "The teacher chooses which questions enter exports; benchmark records are unchanged.")
    for slide in [flow, conclusion]:
        frame = notes_frame(slide)
        frame.text = frame.text.split("\n\nAnswer-role extension:")[0] + note
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deck.save(args.output)
    print("Updated learning path and conclusion in the explicit 12-slide copy:", args.output)


if __name__ == "__main__":
    main()
