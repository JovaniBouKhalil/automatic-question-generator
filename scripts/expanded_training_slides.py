"""Replace only the generated results slide with measured expanded-training evidence."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from pptx import Presentation
from pptx.util import Inches

from scripts.build_presentation import (
    CREAM,
    GOLD,
    GRAY,
    GREEN,
    INK,
    color,
    ensure_destination_closed,
    textbox,
)
from scripts.extend_presentation import append_blank_slide, generated_results_slide, notes_frame

ROOT = Path(__file__).resolve().parents[1]
NAME = "QuestionForge-expanded-v3"
CONCLUSION = ("Validation loss fell in English, French and Arabic.\n"
              "Held-out gains vary with language and test domain.\n"
              "No consistent generation-quality improvement is established.")


def update(source, output):
    report = json.loads((ROOT / "results/expanded_training/summary.json").read_text(encoding="utf-8"))
    deck = Presentation(source)
    existing = [s for s in deck.slides if generated_results_slide(s)]
    if len(existing) != 1 or len(deck.slides) != 12:
        raise ValueError("Expected one generated results slide in a 12-slide deck; preserve custom edits before changing it.")
    original = existing[0]
    slot = next(i for i, s in enumerate(deck.slides) if s.slide_id == original.slide_id)
    entry = next(entry for entry in deck.slides._sldIdLst if int(entry.id) == original.slide_id)
    deck.part.drop_rel(entry.rId)
    deck.slides._sldIdLst.remove(entry)
    slide = append_blank_slide(deck)
    slide.name = NAME
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(CREAM)
    textbox(slide, "QUESTIONFORGE  /  NLP PROJECT 05", .55, .25, 11, .35, 10, GRAY, True)
    textbox(slide, "Train on more data. Measure the result.", .55, .85, 12.2, 1, 33, INK, True)
    textbox(slide, "Native English, French and Arabic adapters · checkpoint selection uses development data only.", .6, 1.75, 12.1, .65, 15, GRAY)
    textbox(slide, "HELD-OUT QA AGREEMENT · CHANGE VS PRETRAINED", .65, 2.45, 7, .4, 15, GREEN, True)
    slide.shapes.add_picture(str(ROOT / "results/expanded_training/comparison.png"), Inches(.55), Inches(2.95), width=Inches(7.2))
    textbox(slide, "TRAINING RUNS AND SELECTED CHECKPOINTS", 8.05, 2.45, 4.7, .55, 15, GREEN, True)
    for index, (language, info) in enumerate(report["languages"].items()):
        text = (f"{language.upper()} · {info['training_examples']:,} train / {info['development_examples']} dev\n"
                f"{info['steps']} updates; saved step {info['best_step']}\n"
                f"Dev loss {info['initial_dev_loss']:.3f} → {info['best_dev_loss']:.3f}")
        textbox(slide, text, 8.05, 3.05 + index * 1.03, 4.7, 1.0, 16)
    textbox(slide, "95% intervals resample whole source topics.\nOne training seed per language; human ratings pending.", .75, 5.9, 6.95, .85, 15, GOLD)
    textbox(slide, "EN: SQuAD · FR: NewsQuAD-fr\nAR: ARCD + native TyDiQA-GoldP", 8.05, 6.3, 4.7, .65, 13, GRAY)
    textbox(slide, "English · Français · العربية     |     11", .6, 7.05, 12, .3, 10, GRAY)
    lines = ["Protocol: docs/EXPANDED_TRAINING_PROTOCOL.md; Arabic target correction: docs/ARABIC_TARGET_FORMAT.md.",
             "Original 40-update and three 160-update seed studies are preserved in the repository and docs/EXTENSIONS.md.",
             "This follow-up repeats previously inspected test sets; it is not a newly untouched benchmark.",
             "A loss-selected checkpoint can use fewer examples than the full training run. Counts below distinguish them.",
             "App selection compares models on the same development rows, including the previous English adapter and step zero.",
             "QA agreement is an imperfect proxy; the verifier's cross-lingual weaknesses remain. Do not equate agreement with factuality.",
             "The native Arabic decoder target includes question:, which the existing generation code removes for display.",
             "No test rows or scores were used to train/select checkpoints; all original source validation pools are reserved.",
             "Sources: https://huggingface.co/datasets/rajpurkar/squad ; https://huggingface.co/datasets/lincoln/newsquadfr ; https://huggingface.co/datasets/hsseinmz/arcd ; https://huggingface.co/datasets/google-research-datasets/tydiqa ."]
    for language, info in report["languages"].items():
        lines.append(f"{language.upper()}: full run saw {info['unique_training_ids_seen']} distinct examples; saved checkpoint saw {info['saved_checkpoint_unique_examples_seen']}. Selected app adapter: {info['selected_app_adapter']}.")
    lines.append("Held-out paired results: " + json.dumps(report["evaluations"], ensure_ascii=False))
    notes_frame(slide).text = "\n\n".join(lines)
    entry = deck.slides._sldIdLst[-1]
    deck.slides._sldIdLst.remove(entry)
    deck.slides._sldIdLst.insert(slot, entry)
    for current in deck.slides:
        for shape in current.shapes:
            if shape.has_text_frame and shape.text.startswith("English QA F1: original"):
                # Change the generated conclusion while retaining its original visual styling.
                frame = shape.text_frame
                for paragraph, line in zip(frame.paragraphs, CONCLUSION.splitlines()):
                    if paragraph.runs:
                        paragraph.runs[0].text = line
                        for run in paragraph.runs[1:]:
                            run.text = ""
                notes_frame(current).text += ("\n\nExpanded training update: the old 40-update values above are historical. "
                                              "The current conclusion summarizes the six comparisons in results/expanded_training/summary.json; "
                                              "all intervals include zero and human ratings remain pending.")
    deck.save(output)
    with ZipFile(output) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())), "Duplicate presentation part names"
    return deck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "presentation/QuestionForge.pptx")
    parser.add_argument("--output", type=Path, default=ROOT / "presentation/QuestionForge.pptx")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "presentation/QuestionForge.pptx").resolve():
        ensure_destination_closed()
    update(args.source, args.output)
    print("Updated the generated results slide; preserved the other eleven slides.")


if __name__ == "__main__":
    main()
