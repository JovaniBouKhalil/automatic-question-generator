"""Update one generated results slide in an explicit preserved presentation copy."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

from scripts.build_presentation import CREAM, GRAY, GREEN, INK, color, textbox
from scripts.extend_presentation import append_blank_slide, generated_results_slide, notes_frame


def replace_text(shape, text):
    frame = shape.text_frame
    template = deepcopy(frame.paragraphs[0]._p)
    for paragraph in list(frame.paragraphs):
        frame._txBody.remove(paragraph._p)
    for line in text.split("\n"):
        frame._txBody.append(deepcopy(template))
        paragraph = frame.paragraphs[-1]
        if paragraph.runs:
            paragraph.runs[0].text = line
            for run in paragraph.runs[1:]:
                run.text = ""
        else:
            paragraph.text = line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Preserved PPTX snapshot, not the user's live open deck.")
    parser.add_argument("--output", type=Path, required=True, help="Closed destination file or a new review copy.")
    args = parser.parse_args()
    deck = Presentation(args.source)
    existing = [s for s in deck.slides if generated_results_slide(s) or s.name == "QuestionForge-science-v4"]
    if len(deck.slides) != 12 or len(existing) != 1:
        raise ValueError("Expected 12 slides and one generated results slide. Preserve custom content before editing.")
    old = existing[0]
    old_notes = notes_frame(old).text
    index = next(i for i, s in enumerate(deck.slides) if s.slide_id == old.slide_id)
    entry = next(e for e in deck.slides._sldIdLst if int(e.id) == old.slide_id)
    deck.part.drop_rel(entry.rId)
    deck.slides._sldIdLst.remove(entry)
    slide = append_blank_slide(deck)
    slide.name = "QuestionForge-science-v4"
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(CREAM)
    textbox(slide, "QUESTIONFORGE  /  NLP PROJECT 05", .55, .25, 11, .35, 10, GRAY, True)
    textbox(slide, "Educational training. Test the result.", .55, .85, 12.2, .8, 32, INK, True)
    textbox(slide, "SciQ: human-authored science questions · 8 settings · 3 hyperparameters · validation-only selection", .6, 1.65, 12.1, .5, 14, GRAY)
    slide.shapes.add_picture("results/science_training/validation_search.png", Inches(.8), Inches(2.15), width=Inches(11.7))
    test = json.loads(Path("results/science_evaluation_test/summary.json").read_text(encoding="utf-8"))
    a, b = test["methods"]["original_t5"], test["methods"]["selected_t5"]
    delta = test["paired_qa_answer_f1_difference_pp"]
    textbox(slide, f"200 reserved science inputs: BLEU {a['bleu']:.2f} → {b['bleu']:.2f} · ROUGE-L {a['rouge_l']:.2f} → {b['rouge_l']:.2f}",
            .65, 6.35, 12, .4, 15, GREEN, True)
    textbox(slide, f"QA recovery Δ {delta['mean']:+.2f} pp [{delta['ci95'][0]:.2f}, {delta['ci95'][1]:.2f}]. Longer copied questions remain; human usefulness is unmeasured.",
            .65, 6.78, 11.95, .32, 11, GRAY)
    # Preserve the established footer and slide count, including the user's conclusion.
    textbox(slide, "English · Français · العربية     |     11", .6, 7.15, 12, .28, 10, GRAY)
    last = deck.slides._sldIdLst[-1]
    deck.slides._sldIdLst.remove(last)
    deck.slides._sldIdLst.insert(index, last)
    notes_frame(slide).text = (
        "Protocol: docs/SCIENCE_TRAINING_PROTOCOL.md. SciQ, AllenAI, Welbl/Liu/Gardner 2017, CC BY-NC 3.0. "
        "2,048/128/200 prepared; 2,043/126 train/dev pass target-token limits. "
        "Grid: lr1e-4/3e-4 × rank8/16 × dropout.05/.15; 256 updates each; same seed/order; alpha=2r. "
        "Validation selects lr3e-4,r16,d.05 at step256, loss1.1829→.9964. "
        "The QA difference interval includes zero. Overlap improvement is not a human usefulness result. "
        "Development cases include better target focus and malformed copied/cloze regressions. "
        "The optional instruction-model planner/editor is a separate inference experiment, not this training.\n\n"
        "Earlier results-slide notes preserved:\n" + old_notes)
    for s in deck.slides:
        for shape in s.shapes:
            if not shape.has_text_frame:
                continue
            if shape.text == "Encoder–decoder generation with a bounded English training experiment.":
                replace_text(shape, "Encoder–decoder generation with native-language and educational LoRA adaptation.")
            if shape.text.startswith("Validation loss fell in English, French and Arabic.\nHeld-out gains vary"):
                replace_text(shape, "Science adaptation improved reference-wording agreement.\nSource inspections still find copied and malformed questions.\nHuman educational-value ratings remain pending.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deck.save(args.output)
    print("Updated the generated science slide; preserved 12 slides and earlier notes:", args.output)


if __name__ == "__main__":
    main()
