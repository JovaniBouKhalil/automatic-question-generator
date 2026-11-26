"""Build 11 editable slides with measured results and technical speaker notes."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
INK = "173C3B"
GREEN = "205C4A"
CREAM = "F6F7F3"
GRAY = "60756C"
GOLD = "B68943"


def color(value):
    return RGBColor.from_string(value)


def textbox(slide, text, x, y, w, h, size=20, fill=INK, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = "Arial"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color(fill)
        paragraph.space_after = Pt(12)
    return box


def pill(slide, label, x, y, w, fill=GREEN):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(.72))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    shape.line.fill.background()
    textbox(slide, label, x+.1, y+.1, w-.2, .55, 17, "FFFFFF", True)


def main():
    required = [ROOT / f"results/{name}/summary.json" for name in ["benchmark_test", "novel_test", "lora_test"]]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Measured results required before building final slides: {path}")
    benchmark, novel, lora = [json.loads(path.read_text(encoding="utf-8")) for path in required]
    manifest = json.loads((ROOT / "results/data/manifest.json").read_text(encoding="utf-8"))
    training = json.loads((ROOT / "artifacts/english_lora/training_manifest.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (ROOT / "results/benchmark_test/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)
    outline = []

    def slide(title, subtitle="", notes=""):
        result = presentation.slides.add_slide(presentation.slide_layouts[6])
        result.background.fill.solid()
        result.background.fill.fore_color.rgb = color(CREAM)
        textbox(result, "QUESTIONFORGE  /  NLP PROJECT 05", .55, .25, 11, .35, 10, GRAY, True)
        textbox(result, title, .55, .85, 12.2, 1, 33, INK, True)
        if subtitle:
            textbox(result, subtitle, .6, 1.75, 12.1, .65, 15, GRAY)
        textbox(result, f"English · Français · العربية     |     {len(presentation.slides):02d}", .6, 7.05, 12, .3, 10, GRAY)
        result.notes_slide.notes_text_frame.text = notes
        outline.append({"slide": len(presentation.slides), "title": title, "subtitle": subtitle, "notes": notes})
        return result

    def table(slide, headers, rows, x=.65, y=2.65, w=12, h=2.7, size=17):
        shape = slide.shapes.add_table(len(rows)+1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
        for i, row in enumerate([headers] + rows):
            for j, value in enumerate(row):
                cell = shape.table.cell(i, j)
                cell.text = str(value)
                cell.fill.solid()
                cell.fill.fore_color.rgb = color(GREEN if i == 0 else "E9EEE6" if i % 2 else CREAM)
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.name = "Arial"
                    paragraph.font.size = Pt(size)
                    paragraph.font.color.rgb = color("FFFFFF" if i == 0 else INK)
                    paragraph.font.bold = i == 0
        return shape

    def metric(rows, language, method, key):
        return next(r[key] for r in rows if r["language"] == language and r["method"] == method)

    s = slide("From passage to practice.", "Factual question generation with source evidence and independent answerability checks.",
              "Introduce the task: educational text plus an intended answer becomes a factual question. We compare a cheap baseline, pretrained sequence-to-sequence models and an English LoRA extension. The final rubric requires code, 8–12 slides and a demo, not a standalone report. Substantial Codex assistance is disclosed in AI_ASSISTANCE.md; the student should understand and review the implementation.")
    textbox(s, "QuestionForge", .65, 2.6, 12, 1.1, 62, GREEN, True)
    textbox(s, "English  /  Français  /  العربية", .7, 4.0, 11, .8, 30, GOLD)
    textbox(s, "Compact pretrained models · CPU inference · Reproducible experiments\nEditable classroom quizzes with answers and evidence", .7, 5.2, 11.7, 1.0, 21)

    s = slide("Ask about a fact. Keep the evidence.", "A question should be relevant, answerable from the passage, and useful to a learner.",
              "The main experiment is answer-conditioned QG. Both baseline and Transformer see context and target answer. TF-IDF answer selection is a separate application feature, not part of the controlled benchmark score. Independent QA asks the generated question and compares its extracted answer with the target. It is a proxy rather than a guarantee of factuality.")
    for i, label in enumerate(["Passage", "Target answer", "QG model", "QA check", "Teacher review"]):
        pill(s, label, .7 + i*2.53, 3, 2.3, GREEN if i % 2 == 0 else GOLD)
    textbox(s, "Input: educational passage + exact answer span\nOutput: question + answer + source evidence + review signals", .8, 4.4, 11.6, 1.5, 25)

    s = slide("Three languages. Documented data.", "Human-annotated public benchmarks plus a small original parallel classroom supplement.",
              "Public sources: https://huggingface.co/datasets/rajpurkar/squad (CC-BY-SA-4.0), https://huggingface.co/datasets/lincoln/newsquadfr (CC-BY-NC-SA-4.0), https://huggingface.co/datasets/hsseinmz/arcd (MIT in Hub metadata; Wikipedia attribution also applies). Actual counts and exclusions are in results/data/manifest.json. Different domains prevent a controlled ranking of languages. Original scenarios are AI-assisted and need human review; translations of one topic stay in one split.")
    rows = []
    for lang in ["en", "fr", "ar"]:
        info = manifest["datasets"][lang]
        counts = info["selected_counts"]
        rows.append([lang.upper(), info["id"], f"{counts['train']} / {counts['dev']} / {counts['test']}"])
    table(s, ["Language", "Pinned dataset", "Train / dev / test"], rows, h=2.1)
    textbox(s, "Original supplement: 15 parallel scenarios · 90 questions\n18 development + 72 test · translations grouped by topic", .8, 5.2, 11.5, 1.1, 22)

    s = slide("Preserve language. Prevent paragraph leakage.", "Natural text and exact offsets enter the pretrained tokenizer unchanged.",
              "Validate answer text at its original character offset; do not guess repairs. Exclude duplicate question/answer records. Reserve evaluation paragraphs and remove matches from training. Split remaining training paragraphs into train/development, then sample one question per paragraph with seed 42. Accents/Arabic are preserved. Grouping is exact normalized paragraph grouping, not near-duplicate or article isolation. EDA records length distributions. Cropping measures the actual prompt and retains the answer; surrounding context can be lost.")
    textbox(s, "01  Validate source spans and remove duplicates\n02  Reserve evaluation paragraphs; remove training overlap\n03  Split and sample paragraph groups with seed 42\n04  Use each checkpoint's own tokenizer and prompt\n05  Crop around the answer within 384 input tokens", .8, 2.65, 11.7, 3.2, 23)
    textbox(s, "Leakage audit covers our pipeline; public checkpoints can have unknown upstream exposure.", .8, 6.15, 11.7, .55, 17, GOLD)

    s = slide("A cheap baseline. A learned rewrite.", "Identical context and target answers make the comparison interpretable.",
              "Baseline: source sentence with the exact target replaced by a language-specific WH phrase. This retains source content but often yields unnatural grammar. Neural models learn conditional rewriting and question syntax. English: lmqg/t5-small-squad-qg; French: lmqg/mt5-small-frquad-qg; Arabic: Mihakram/AraT5-base-question-generation. Input formats and pinned revisions are documented in registry.py and DATA_AND_MODELS.md. Different language checkpoints are not matched in parameter count or training data.")
    exemplar = next(r for r in predictions if r["language"] == "en" and r["method"] == "beam4")
    baseline = next(r for r in predictions if r["id"] == exemplar["id"] and r["method"] == "baseline")
    textbox(s, "SOURCE  " + exemplar["context"][:300], .8, 2.55, 11.6, 1.35, 18, GRAY)
    textbox(s, "TARGET  " + exemplar["answer"], .8, 4.0, 11.6, .7, 21, GOLD, True)
    textbox(s, "BASELINE  " + baseline["question"][:200] + "\nTRANSFORMER  " + exemplar["question"][:200], .8, 4.9, 11.6, 1.7, 23)

    s = slide("Attention generates. LoRA adapts.", "Encoder–decoder generation with a bounded English training experiment.",
              "The encoder uses bidirectional self-attention. The decoder uses causal self-attention and cross-attention to encoder states, generating one token at a time. Unlike decoder-only GPT-2, it explicitly conditions on encoded input. LoRA learns low-rank BA updates to frozen query/value projections: W_eff = W + (alpha/r)BA. The rank-8 adapter uses alpha 16 and dropout 0.05. Development token cross-entropy selects the checkpoint, including the initial step. Test data never enters training or selection. Source papers: https://arxiv.org/abs/1910.10683 ; https://arxiv.org/abs/2010.11934 ; https://arxiv.org/abs/2109.12068 ; https://arxiv.org/abs/2305.17416 .")
    pill(s, "Encoder: passage + answer", .8, 2.9, 5.0)
    pill(s, "Decoder: question tokens", 7.0, 2.9, 5.4, GOLD)
    textbox(s, "Bidirectional input attention\nCross-attention → causal output decoding", .9, 4.0, 11.4, 1.1, 24)
    textbox(s, f"LoRA: {training['trainable_parameters']:,} / {training['total_parameters']:,} trainable parameters\n{training['steps']} updates · effective batch 8 · dev loss {training['initial_dev_loss']:.3f} → {training['best_dev_loss']:.3f}", .9, 5.35, 11.5, 1.0, 23, GREEN, True)

    s = slide("Fixed settings. Paired comparisons.", "Do not choose hyperparameters using the final test results.",
              "Configuration is recorded in results/*/config.json along with input hashes and checkpoint commits; environment.json records versions and git commit. Cold model loading is excluded from per-example generation latency and reported separately. All methods evaluate identical IDs. Errors remain in denominators. LoRA trains at 256 input tokens for cost control, then evaluates at the common 384-token budget. This tests a small practical adaptation, not a full training sweep. Single-reference overlap is imperfect. Paragraph-group bootstrap intervals estimate sampling variability, not systematic error.")
    table(s, ["Setting", "Baseline", "Greedy", "Beam search"], [
        ["Model", "WH source rewrite", "Pretrained QG", "Same checkpoint"],
        ["Beams / sampling", "—", "1 / off", "4 / off"],
        ["Input / output budget", "Evidence sentence", "384 / 64 tokens", "384 / 64 tokens"],
        ["Evaluation", "Same IDs", "Same IDs", "Same IDs"],
    ], y=2.5, h=2.6, size=17)
    textbox(s, "chrF++ · Unicode ROUGE-L · BLEU · independent QA F1/EM\n1,000 group bootstrap resamples · CPU float32 · seed 42", .8, 5.5, 11.6, 1.0, 22)

    s = slide("Measured benchmark results", "Question agreement and round-trip QA are complementary, imperfect signals.",
              "Read the figure from actual result files. chrF++ and ROUGE-L are reference agreement, not factual correctness. QA token F1 compares the independent answer with the intended target. Error bars resample paragraph groups. Avoid claiming a clear improvement when paired confidence intervals cross zero. Language differences combine domain, checkpoint size/training and verifier quality. See summary.csv, predictions.jsonl and paired_comparisons.json for complete numbers and failures.")
    s.shapes.add_picture(str(ROOT / "results/benchmark_test/comparison.png"), Inches(.55), Inches(2.4), width=Inches(12.2))

    s = slide("Check the questions—and the judge.", "The verifier is multilingual-pretrained, but QA-fine-tuned on English SQuAD2.",
              "Verifier source: https://huggingface.co/deepset/xlm-roberta-base-squad2-distilled . Reference questions are positive controls; unrelated questions ask for a red submarine's serial number. High reference F1 supports usefulness of the verifier on that language/domain; low reference F1 makes generated-question failures ambiguous. Nonzero unrelated target agreement exposes false positives. The rule requires F1>=0.5, nonempty span score>=0.1 and no answer leakage. Coverage is not human accuracy. QA span scores are uncalibrated, and lexical cosine is not semantic entailment.")
    s.shapes.add_picture(str(ROOT / "results/benchmark_test/verifier_controls.png"), Inches(.65), Inches(2.45), width=Inches(7.5))
    coverage = "\n".join(f"{lang.upper()} beam-4 coverage: {metric(benchmark, lang, 'beam4', 'accepted'):.1f}%" for lang in ["en", "fr", "ar"])
    textbox(s, coverage + "\n\nCoverage ≠ question accuracy.\nTeacher review remains necessary.", 8.4, 2.8, 4.2, 3.3, 21)

    s = slide("Failures are part of the result.", "Automatic examples below need human inspection; low QA agreement can also mean verifier failure.",
              "Examples are selected deterministically by lowest QA answer agreement within each language's beam-4 outputs. They are not invented human labels. Inspect source evidence, intended answer and independent QA answer to distinguish wrong relation, malformed question, ambiguity or judge failure. human_review.csv has blank relevance/answerability/fluency ratings. Add human annotation rather than claiming automatic proxies are ground truth.")
    for index, lang in enumerate(["en", "fr", "ar"]):
        examples = [r for r in predictions if r["language"] == lang and r["method"] == "beam4"]
        row = min(examples, key=lambda r: (r.get("answer_f1", 0), r["id"]))
        text = (f"{lang.upper()}  {row['question'][:180]}\nTarget: {row['answer'][:90]}  |  QA: {row.get('qa_answer', '')[:90]}"
                f"  |  F1: {100*row.get('answer_f1', 0):.1f}")
        textbox(s, text, .8, 2.5 + index*1.35, 11.7, 1.25, 18)

    s = slide("Ready to demonstrate. Clear about the limits.", "A working learning tool, with an auditable experiment behind it.",
              "Demonstrate English, French and Arabic; show evidence and review reasons; export student worksheet and answer key. Original parallel scenarios supplement the public benchmark but are small and AI-assisted. Upstream exposure, limited human annotation, single references and cross-lingual verifier weakness limit conclusions. Next: human ratings from two reviewers, near-duplicate/article isolation, stronger multilingual verification and independently authored educational data. AI assistance is disclosed. Personalize and rehearse the slides before submission.")
    base_f1, lora_f1 = metric(lora, "en", "beam4", "answer_f1"), metric(lora, "en", "lora_beam4", "answer_f1")
    textbox(s, f"English QA F1: original {base_f1:.1f} → LoRA {lora_f1:.1f}\nTraining loss improvement does not guarantee generation improvement.", .8, 2.6, 11.7, 1.2, 24, GREEN, True)
    textbox(s, "Demo: choose a passage → inspect a question → export a quiz\nNext: human annotations, stronger multilingual verification, new educational passages", .8, 4.3, 11.5, 1.5, 24)
    textbox(s, "Code, sources, measured predictions and reproduction commands are included.\nSubstantial AI assistance disclosed in AI_ASSISTANCE.md.", .8, 6.1, 11.7, .7, 15, GRAY)
    output = ROOT / "presentation"
    output.mkdir(exist_ok=True)
    presentation.save(output / "QuestionForge.pptx")
    (output / "slide_notes.json").write_text(json.dumps(outline, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created {len(presentation.slides)} slides: presentation/QuestionForge.pptx")


if __name__ == "__main__":
    main()
