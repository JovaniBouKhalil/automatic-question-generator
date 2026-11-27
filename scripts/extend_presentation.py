"""Add measured robustness results while preserving the existing eleven slides."""

import argparse
import json
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.packuri import PackURI
from pptx.oxml.xmlchemy import OxmlElement
from pptx.parts.slide import SlidePart
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

ROOT = Path(__file__).resolve().parents[1]
NAME = "QuestionForge-robustness-v2"


def append_blank_slide(deck):
    """Avoid python-pptx's count-based part-name collision after deleting a middle slide."""
    used = {part.partname for part in deck.part.package.iter_parts()}
    number = 1
    while PackURI(f"/ppt/slides/slide{number}.xml") in used:
        number += 1
    layout = deck.slide_layouts[6]
    part = SlidePart.new(PackURI(f"/ppt/slides/slide{number}.xml"), deck.part.package, layout.part)
    relationship = deck.part.relate_to(part, RT.SLIDE)
    slide = part.slide
    slide.shapes.clone_layout_placeholders(layout)
    deck.slides._sldIdLst.add_sldId(relationship)
    return slide


def generated_results_slide(slide):
    """PowerPoint may rename imported slides, so identify our generated content too."""
    if slide.name in {NAME, "QuestionForge-expanded-v3"}:
        return True
    titles = {"Check stability. Challenge the verifier.", "Train on more data. Measure the result."}
    if any(shape.has_text_frame and shape.text.strip() in titles for shape in slide.shapes):
        return True
    frame = slide.notes_slide.notes_text_frame
    return frame is not None and frame.text.startswith(("Protocol: docs/EXTENSION_PROTOCOL.md", "Protocol: docs/EXPANDED_TRAINING_PROTOCOL.md"))


def notes_frame(slide):
    notes = slide.notes_slide
    if notes.notes_text_frame is None:
        # Office can save a notes master without a body placeholder. Restore a real
        # body placeholder so the notes remain editable in PowerPoint and python-pptx.
        tree = notes._element.spTree
        shape = tree.add_textbox(notes.shapes._next_shape_id, "Notes Text", Inches(.5), Inches(3.5), Inches(6.5), Inches(4))
        placeholder = OxmlElement("p:ph")
        placeholder.set("type", "body")
        shape.nvSpPr.nvPr.append(placeholder)
    return notes.notes_text_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "presentation/QuestionForge.pptx")
    parser.add_argument("--output", type=Path, default=ROOT / "presentation/QuestionForge.pptx")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "presentation/QuestionForge.pptx").resolve():
        ensure_destination_closed()
    studies = {name: json.loads((ROOT / f"results/{name}/summary.json").read_text(encoding="utf-8"))
               for name in ["seed_study", "verification_study", "article_sensitivity"]}
    path = args.source
    backup = ROOT / ".cache/presentation-before-extensions.pptx"
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)
    deck = Presentation(path)
    for current in list(deck.slides):
        if generated_results_slide(current):
            entry = next(entry for entry in deck.slides._sldIdLst if int(entry.id) == current.slide_id)
            deck.part.drop_rel(entry.rId)
            deck.slides._sldIdLst.remove(entry)
    if len(deck.slides) != 11:
        raise ValueError("Expected eleven original slides; preserve and inspect custom deck changes.")
    slide = append_blank_slide(deck)
    slide.name = NAME
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(CREAM)
    textbox(slide, "QUESTIONFORGE  /  NLP PROJECT 05", .55, .25, 11, .35, 10, GRAY, True)
    textbox(slide, "Check stability. Challenge the verifier.", .55, .85, 12.2, 1, 33, INK, True)
    textbox(slide, "Follow-up experiments preserve the original results and report every planned run.", .6, 1.75, 12.1, .65, 15, GRAY)
    textbox(slide, "THREE SEEDS · 160 UPDATES EACH", .7, 2.45, 6.5, .4, 16, GREEN, True)
    slide.shapes.add_picture(str(ROOT / "results/seed_study/seed_stability.png"), Inches(.55), Inches(2.95), width=Inches(7.0))
    textbox(slide, "TWO QA CHECKERS · CONSTRUCTED CONTROLS", 7.8, 2.45, 4.9, .5, 15, GREEN, True)
    rows = []
    for lang in ["en", "fr", "ar"]:
        a = next(r for r in studies["verification_study"]["models"] if r["partition"] == "test" and r["language"] == lang and r["verifier"] == "original")
        b = next(r for r in studies["verification_study"]["models"] if r["partition"] == "test" and r["language"] == lang and r["verifier"] == "multilingual")
        rows.append(f"{lang.upper()}    {a['reference_f1']:.1f} → {b['reference_f1']:.1f}       {a['negative_false_acceptance_pct']:.1f} → {b['negative_false_acceptance_pct']:.1f}")
    textbox(slide, "          Reference F1 ↑     False acceptance ↓\n" + "\n".join(rows), 7.8, 3.12, 4.9, 1.9, 17)
    textbox(slide, "Original XLM-R → multilingual mBERT\nSynthetic controls ≠ human-rated question accuracy.", 7.8, 5.05, 4.8, .85, 15, GRAY)
    textbox(slide, "Intervals include zero: improvements remain uncertain.\nOne classroom seed worsens; report it alongside the gains.", .75, 5.8, 6.8, .75, 15, GOLD)
    textbox(slide, f"Article-isolated sensitivity: {studies['article_sensitivity']['n_training']} training paragraphs.\nBlinded human booklet: 54 items prepared; ratings pending.", 7.8, 6.0, 4.9, .85, 15, GRAY)
    textbox(slide, "English · Français · العربية     |     11", .6, 7.05, 12, .3, 10, GRAY)
    notes_frame(slide).text = (
        "Protocol: docs/EXTENSION_PROTOCOL.md. All planned seeds and negative results are retained. "
        "Topic intervals keep three seeds fixed; repeated input IDs are not extra independent data. "
        "The optional app adapter is selected using development loss only. Three seeds give limited evidence. "
        "Verifier comparison confounds architecture and QA adaptation. mBERT lacks impossible-question training. "
        "French transfer remains zero-shot. Augmented XQuAD provenance is incomplete and span scores are uncalibrated. "
        "Negative conditions: unrelated question, unsupported location and wrong intended target. "
        "Constructed controls do not estimate natural error prevalence or human-rated question quality. "
        "The original default checker is preserved. Human ratings remain pending. "
        "Model card: https://huggingface.co/mrm8488/bert-multi-cased-finetuned-xquadv1 . "
        + "Seed-mean paired results: " + json.dumps(studies["seed_study"]["paired_seed_means"])
        + " Article isolation: " + json.dumps(studies["article_sensitivity"])
        + " Verifier intervals: " + json.dumps(studies["verification_study"]["paired_differences"])
    )
    entry = deck.slides._sldIdLst[-1]
    deck.slides._sldIdLst.remove(entry)
    deck.slides._sldIdLst.insert(10, entry)  # Preserve the existing conclusion as the last slide.
    outline = []
    for index, current in enumerate(deck.slides, 1):
        for shape in current.shapes:
            if shape.has_text_frame and shape.text.startswith("English · Français · العربية"):
                shape.text_frame.paragraphs[0].runs[0].text = f"English · Français · العربية     |     {index:02d}"
        titles = [s.text for s in current.shapes if s.has_text_frame and abs(s.top - Inches(.85)) < 20000]
        outline.append({"slide": index, "title": titles[0] if titles else "", "notes": notes_frame(current).text})
    deck.save(args.output)
    if args.output.resolve() == (ROOT / "presentation/QuestionForge.pptx").resolve():
        (ROOT / "presentation/slide_notes.json").write_text(json.dumps(outline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Preserved original slides and added measured robustness results: {len(deck.slides)} slides.")


if __name__ == "__main__":
    main()
