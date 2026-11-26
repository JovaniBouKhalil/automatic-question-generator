"""Build 11 editable slides with measured results and technical speaker notes."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE
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
    frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
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
    presentation.core_properties.title = "QuestionForge — Multilingual Automatic Question Generation"
    presentation.core_properties.author = "QuestionForge project; AI assistance disclosed"
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

    def add_notes(result, text):
        result.notes_slide.notes_text_frame.text += "\n\n" + text
        outline[-1]["notes"] = result.notes_slide.notes_text_frame.text

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
        short_name = {"en": "SQuAD", "fr": "NewsQuAD-fr", "ar": "ARCD"}[lang]
        rows.append([f"{lang.upper()} · {short_name}", f"{info['raw_counts']['train']:,} / {info['raw_counts']['test']:,}",
                     f"{counts['train']} / {counts['dev']} / {counts['test']}"])
    table(s, ["Language / source", "Original train / eval", "Used train / dev / test"], rows, h=2.1)
    textbox(s, "Original supplement: 15 parallel scenarios · 90 questions\n18 development + 72 test · translations grouped by topic", .8, 5.2, 11.5, 1.1, 22)

    s = slide("Preserve language. Prevent paragraph leakage.", "Natural text and exact offsets enter the pretrained tokenizer unchanged.",
              "Validate answer text at its original character offset; do not guess repairs. Exclude duplicate question/answer records. Reserve evaluation paragraphs and remove matches from training. Split remaining training paragraphs into train/development, then sample one question per paragraph with seed 42. Accents/Arabic are preserved. Grouping is exact normalized paragraph grouping, not near-duplicate or article isolation. EDA records length distributions. Cropping measures the actual prompt and retains the answer; surrounding context can be lost.")
    textbox(s, "01  Validate source spans and remove duplicates\n02  Reserve evaluation paragraphs; remove training overlap\n03  Split and sample paragraph groups with seed 42\n04  Use each checkpoint's own tokenizer and prompt\n05  Crop around the answer within 384 input tokens", .8, 2.65, 11.7, 3.2, 23)
    textbox(s, "Leakage audit covers our pipeline; public checkpoints can have unknown upstream exposure.", .8, 6.15, 11.7, .55, 17, GOLD)

    s = slide("A cheap baseline. A learned rewrite.", "Identical context and target answers make the comparison interpretable.",
              "Baseline: source sentence with the exact target replaced by a language-specific WH phrase. This retains source content but often yields unnatural grammar. Neural models learn conditional rewriting and question syntax. English: lmqg/t5-small-squad-qg; French: lmqg/mt5-small-frquad-qg; Arabic: Mihakram/AraT5-base-question-generation. Input formats and pinned revisions are documented in registry.py and DATA_AND_MODELS.md. Different language checkpoints are not matched in parameter count or training data.")
    exemplar = next((r for r in predictions if r["id"] == "en:57114667a58dae1900cd6d80" and r["method"] == "beam4"),
                    next(r for r in predictions if r["language"] == "en" and r["method"] == "beam4"))
    add_notes(s, f"Illustrative source example selected for readability, not to estimate average quality. Record ID: {exemplar['id']}.")
    baseline = next(r for r in predictions if r["id"] == exemplar["id"] and r["method"] == "baseline")
    evidence = exemplar.get("evidence", exemplar["context"])
    answer_at = evidence.find(exemplar["answer"])
    if len(evidence) > 300 and answer_at >= 0:
        evidence = evidence[max(0, answer_at - 110):answer_at + len(exemplar["answer"]) + 110]
    textbox(s, "SOURCE  " + evidence, .8, 2.55, 11.6, 1.35, 18, GRAY)
    textbox(s, "TARGET  " + exemplar["answer"], .8, 4.0, 11.6, .7, 21, GOLD, True)
    textbox(s, "BASELINE  " + baseline["question"][:200] + "\nTRANSFORMER  " + exemplar["question"][:200], .8, 4.9, 11.6, 1.7, 23)

    s = slide("Attention generates. LoRA adapts.", "Encoder–decoder generation with a bounded English training experiment.",
              "The encoder uses bidirectional self-attention. The decoder uses causal self-attention and cross-attention to encoder states, generating one token at a time. Unlike decoder-only GPT-2, it explicitly conditions on encoded input. LoRA learns low-rank BA updates to frozen query/value projections: W_eff = W + (alpha/r)BA. The rank-8 adapter uses alpha 16 and dropout 0.05. Development token cross-entropy selects the checkpoint, including the initial step. Test data never enters training or selection. Source papers: https://arxiv.org/abs/1910.10683 ; https://arxiv.org/abs/2010.11934 ; https://arxiv.org/abs/2109.12068 ; https://arxiv.org/abs/2305.17416 .")
    pill(s, "Encoder: passage + answer", .8, 2.9, 5.0)
    pill(s, "Decoder: question tokens", 7.0, 2.9, 5.4, GOLD)
    textbox(s, "Bidirectional input attention\nCross-attention → causal output decoding", .9, 4.0, 11.4, 1.1, 24)
    textbox(s, f"LoRA: {training['trainable_parameters']:,} / {training['total_parameters']:,} trainable parameters\n{training['steps']} updates · effective batch 8 · dev loss {training['initial_dev_loss']:.3f} → {training['best_dev_loss']:.3f}", .9, 5.35, 11.5, 1.0, 23, GREEN, True)
    add_notes(s, "LoRA method reference: https://arxiv.org/abs/2106.09685 . Training-loss figure is included in artifacts/english_lora/loss_curve.png. The loss reduction is small and is not a quality guarantee.")

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
    stress_path = ROOT / "results/verifier_stress/summary.json"
    if stress_path.exists():
        stress = json.loads(stress_path.read_text(encoding="utf-8"))
        rates = {r["language"]: r["automatic_acceptance_pct"] for r in stress if r["condition"] == "unsupported_qualifier"}
        counts = {r["language"]: r["n"] for r in stress if r["condition"] == "unsupported_qualifier"}
        accepted_counts = {lang: round(rates[lang] * counts[lang] / 100) for lang in rates}
        textbox(s, "Unsupported location probe\nAutomatic acceptance", 8.4, 2.6, 4.2, .85, 19, GREEN, True)
        textbox(s, "\n".join(f"{lang.upper()}   {rates[lang]:.1f}% ({accepted_counts[lang]}/{counts[lang]})" for lang in ["en", "fr", "ar"]), 8.5, 3.65, 4.0, 1.7, 22)
        textbox(s, "Exploratory: append 'on Mars' to reference questions.\nThe location is unsupported by each passage.", 8.4, 5.65, 4.1, .95, 14, GRAY)
        counts_text = ", ".join(f"{accepted_counts[lang]}/{counts[lang]} {lang.upper()}" for lang in ["en", "fr", "ar"])
        coverage_text = ", ".join(f"{lang.upper()} {metric(benchmark, lang, 'beam4', 'accepted'):.1f}%" for lang in ["en", "fr", "ar"])
        add_notes(s, f"This probe was added after observing a perfect-QA but unfaithful David/Trajan's Column question. It is exploratory and does not tune generation or review thresholds. Unsupported Mars qualifiers pass the fixed rules in {counts_text} cases. The labels indicate unsupported premises in these constructed questions, not real-world truth or general false-positive rates. Main beam-4 coverage: {coverage_text}; coverage is not accuracy.")
    else:
        coverage = "\n".join(f"{lang.upper()} beam-4 coverage: {metric(benchmark, lang, 'beam4', 'accepted'):.1f}%" for lang in ["en", "fr", "ar"])
        textbox(s, coverage + "\n\nCoverage ≠ question accuracy.\nTeacher review remains necessary.", 8.4, 2.8, 4.2, 3.3, 21)

    s = slide("Failures are part of the result.", "Source-inspected examples distinguish generation mistakes from judge mistakes.",
              "These illustrative cases were selected purposively with disclosed AI-assisted qualitative inspection. They are not blinded human ratings or error-prevalence estimates. Inspect the source evidence and the intended relation. human_review.csv remains blank for student/human annotation. Additional cases and full source passages are in results/qualitative_examples.json.")
    selected_ids = {"en": "en:572a07c11d046914007796d8", "fr": "fr:2739", "ar": "ar:550131195698"}
    categories = {"en": "Purpose confused with method", "fr": "Wrong semantic role", "ar": "Verifier fails on the exact reference question"}
    review_path = ROOT / "results/qualitative_examples.json"
    qualitative = json.loads(review_path.read_text(encoding="utf-8"))["examples"] if review_path.exists() else []
    for index, lang in enumerate(["en", "fr", "ar"]):
        examples = [r for r in predictions if r["language"] == lang and r["method"] == "beam4"]
        row = next((r for r in examples if r["id"] == selected_ids[lang]), min(examples, key=lambda r: (r.get("answer_f1", 0), r["id"])))
        textbox(s, f"{lang.upper()} / {categories[lang]}", .8, 2.45 + index*1.35, 11.7, .35, 15, GREEN, True)
        text = (f"{row['question'][:110]}\nTarget: {row['answer'][:45]}  |  QA: {row.get('qa_answer', '')[:45]}"
                f"  |  F1: {100*row.get('answer_f1', 0):.1f}")
        textbox(s, text, .8, 2.82 + index*1.35, 11.7, .9, 16)
        review = next((r["review"] for r in qualitative if r["id"] == row["id"] and r["method"] == "beam4"), "Inspect the source and target relation.")
        add_notes(s, f"{lang.upper()} {row['id']}: {review}\nSource: {row.get('evidence', row['context'])}")

    s = slide("Ready to demonstrate. Clear about the limits.", "A working learning tool, with an auditable experiment behind it.",
              "Demonstrate English, French and Arabic; show evidence and review reasons; export student worksheet and answer key. Original parallel scenarios supplement the public benchmark but are small and AI-assisted. Upstream exposure, limited human annotation, single references and cross-lingual verifier weakness limit conclusions. Next: human ratings from two reviewers, near-duplicate/article isolation, stronger multilingual verification and independently authored educational data. AI assistance is disclosed. Personalize and rehearse the slides before submission.")
    base_f1, lora_f1 = metric(lora, "en", "beam4", "answer_f1"), metric(lora, "en", "lora_beam4", "answer_f1")
    delta_path = ROOT / "results/lora_test/adaptation_delta.json"
    delta = json.loads(delta_path.read_text(encoding="utf-8")) if delta_path.exists() else {}
    interval = delta.get("paired_group_bootstrap_ci95_pp")
    interval_text = f"95% interval [{interval[0]:g}, {interval[1]:g}]" if interval else "paired interval pending"
    novel_path = ROOT / "results/lora_novel_test/summary.json"
    novel_text = "Original classroom adaptation comparison pending."
    if novel_path.exists():
        novel_adapt = json.loads(novel_path.read_text(encoding="utf-8"))
        novel_gain = metric(novel_adapt, "en", "lora_beam4", "answer_f1") - metric(novel_adapt, "en", "beam4", "answer_f1")
        novel_count = metric(novel_adapt, "en", "beam4", "n")
        novel_text = (f"No QA F1 change on {novel_count} original classroom cases." if abs(novel_gain) < 1e-10 else
                      f"Classroom-case QA F1 change: {novel_gain:+.1f} points ({novel_count} cases).")
    textbox(s, f"English QA F1: original {base_f1:.1f} → LoRA {lora_f1:.1f}\nPaired gain: {lora_f1-base_f1:.1f} points; {interval_text}.\n{novel_text}", .8, 2.6, 11.7, 1.6, 23, GREEN, True)
    textbox(s, "Demo: choose a passage → inspect a question → export a quiz\nNext: human review and stronger multilingual verification", .8, 4.55, 11.5, 1.25, 23)
    add_notes(s, f"Paired public-benchmark LoRA analysis: {json.dumps(delta)}. {novel_text} These are limited subset results. Inspect whether intervals exclude zero before making a directional claim. See adaptation_delta.json for both subsets. Beam-4 improves benchmark ROUGE-L in all three languages in the provided run, while French QA agreement is lower than the copying baseline. Report the tradeoff rather than universal superiority.")
    textbox(s, "Code, sources, measured predictions and reproduction commands are included.\nSubstantial AI assistance disclosed in AI_ASSISTANCE.md.", .8, 6.1, 11.7, .7, 15, GRAY)
    output = ROOT / "presentation"
    output.mkdir(exist_ok=True)
    presentation.save(output / "QuestionForge.pptx")
    (output / "slide_notes.json").write_text(json.dumps(outline, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created {len(presentation.slides)} slides: presentation/QuestionForge.pptx")


if __name__ == "__main__":
    main()
