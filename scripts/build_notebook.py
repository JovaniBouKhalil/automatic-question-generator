"""Generate the compact, runnable course-facing walkthrough notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "outputs": [], "execution_count": None})

    markdown("""# QuestionForge — multilingual factual question generation

**Project 05 · English / français / العربية**

We compare an untrained source-rewriting baseline with answer-aware encoder–decoder
Transformers and independently check whether a QA model recovers the target answer.
This notebook demonstrates the Python project; reproducible full experiments live in
`results/`. See `README.md` for installation and `docs/DATA_AND_MODELS.md` for sources,
licenses, partitions and checkpoint exposure limitations.

Install the project in a Python 3.11 environment, then select that environment as this
notebook's kernel. The neural cell downloads public checkpoints on first use; run
`python scripts/download_models.py` beforehand for an offline demonstration.
""")
    code("""from pathlib import Path
import os, sys, json, random

ROOT = Path.cwd()
if not (ROOT / 'questionforge').is_dir():
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
SEED = 42
random.seed(SEED)
import pandas as pd
from IPython.display import display, Image
from questionforge.data import load_jsonl, assert_disjoint
from questionforge.text import tokens
from questionforge.baseline import template_question
from questionforge.metrics import rouge_l, lexical_relevance

print('Project:', ROOT.name, '| Random seed:', SEED)
""")
    markdown("""## 1. Inspect data before modeling

The main benchmark uses pinned Hugging Face SQuAD, NewsQuAD-fr and ARCD snapshots.
A small, original parallel classroom dataset supplements them with scenarios created
before inference. It is synthetic and needs human linguistic review. Translations of
one scenario share a group; no group crosses development and test partitions.
""")
    code("""partitions = {split: load_jsonl(ROOT / f'data/novel/{split}.jsonl') for split in ['dev', 'test']}
assert_disjoint(partitions)
rows = partitions['dev']
frame = pd.DataFrame(rows)
frame['passage_words'] = frame.context.map(lambda text: len(tokens(text)))
display(frame.groupby('language').agg(records=('id', 'count'), mean_passage_words=('passage_words', 'mean')))
display(frame[['language', 'context', 'question', 'answer']].head(3))
manifest_path = ROOT / 'data/processed/manifest.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    display(pd.DataFrame({lang: info['selected_counts'] for lang, info in manifest['datasets'].items()}))
""")
    markdown("""## 2. Establish a meaningful baseline

Replace the exact answer span in its source sentence with a language-specific question
phrase. This costs no training and retains source words, but cannot reliably produce
natural word order. The Transformer receives the **same passage and target answer**.
Automatic answer selection is evaluated separately from this controlled setup.
""")
    code("""examples = [next(row for row in rows if row['language'] == lang) for lang in ['en', 'fr', 'ar']]
baseline_rows = []
for row in examples:
    question = template_question(row['context'], row['answer'], row['language'], row['answer_start'])
    baseline_rows.append({'language': row['language'], 'question': question, 'answer': row['answer'],
                          'reference': row['question'], 'ROUGE-L': rouge_l(question, row['question'])})
display(pd.DataFrame(baseline_rows))
""")
    markdown("""## 3. Encoder–decoder Transformers and decoding

The encoder reads the passage and marked answer bidirectionally. The decoder predicts
question tokens autoregressively using causal self-attention and cross-attention to the
encoder. Unlike the course GPT-2 exercise, this is conditional sequence-to-sequence
generation. Greedy decoding keeps one continuation; beam search keeps several.

We preserve accents, Arabic text and character offsets. Each checkpoint has its own
tokenizer and training prompt. Answer-centered cropping measures the actual prompted
token count and keeps the answer inside the 384-token budget.
""")
    code("""# Set False for a quick notebook run without neural downloads.
RUN_NEURAL = True
from dataclasses import asdict
from questionforge.models import generate_question, verify_question
from questionforge.registry import GENERATORS

neural_rows = []
if RUN_NEURAL:
    for row in examples:
        generated = generate_question(row['context'], row['answer'], row['language'],
                                      answer_start=row['answer_start'], beams=4, device='cpu')
        result = asdict(generated)
        result.update(verify_question(generated.question, row['context'], row['answer'], row['language']))
        neural_rows.append(result)
    display(pd.DataFrame(neural_rows)[['language', 'question', 'answer', 'qa_answer', 'answer_f1', 'input_tokens']])
""")
    markdown("""## 4. Evaluate questions and audit the judge

chrF++ and Unicode ROUGE-L measure agreement with a single reference question, not
factuality. Independent extractive QA attempts to recover the target answer; token F1
and exact match measure agreement. Its span score is **uncalibrated**. The multilingual
verifier is fine-tuned on English, so reference-question controls are essential for
interpreting French and Arabic scores. Unrelated questions provide a negative control.

All methods use the same record IDs. Generation errors stay in the denominator. Group
bootstrap intervals resample paragraphs or scenario topics. Main results use every
example; review rules report coverage and do not silently remove failures.
""")
    code("""for name in ['benchmark_test', 'novel_test', 'lora_test', 'lora_novel_test']:
    summary = ROOT / 'results' / name / 'summary.csv'
    if summary.exists():
        print(name)
        table = pd.read_csv(summary)
        display(table[[c for c in ['language','method','n','chrf_pp','rouge_l','answer_f1','accepted','errors'] if c in table]])
        figure = summary.parent / 'comparison.png'
        if figure.exists():
            display(Image(filename=str(figure), width=950))
""")
    markdown("""## 5. Small training extension and error analysis

`scripts/train_lora.py` freezes the English T5 backbone and trains rank-8 updates to
attention Q/V projections using only the training partition. It selects the adapter by
development cross-entropy and records seeds, counts, hashes, trainable parameters,
losses and runtime. A lower token loss does not guarantee better generated questions.

Use `results/*/human_review.csv` to annotate relevance, answerability and fluency (0–2),
with error categories such as answer leakage, wrong relation, grammar, wrong language
or judge failure. Blank ratings are pending human review, not measured human scores.
""")
    code("""training = ROOT / 'artifacts/english_lora/training_manifest.json'
if training.exists():
    info = json.loads(training.read_text(encoding='utf-8'))
    display({key: info[key] for key in ['trainable_parameters', 'total_parameters', 'steps',
                                       'initial_dev_loss', 'best_dev_loss', 'best_step', 'elapsed_seconds']})
review = ROOT / 'results/benchmark_test/review_examples.json'
if review.exists():
    examples_to_review = json.loads(review.read_text(encoding='utf-8'))
    display(pd.DataFrame(examples_to_review)[['selection', 'language', 'method', 'question', 'answer', 'qa_answer', 'answer_f1']])
""")
    markdown("""## 6. Inspect factual failures and stress the verifier

`results/qualitative_examples.json` contains source-linked qualitative inspections,
not blinded human ratings. Cases include a question that wrongly merges David and
Trajan's Column despite perfect QA agreement, and an Arabic question identical to its
reference for which the verifier selects the wrong answer.

An **exploratory post-hoc** probe appends an unsupported 'on Mars' qualifier to original
classroom questions. It measures whether the fixed review rules can miss unsupported
premises. It does not tune the generators or thresholds and does not estimate the
frequency of such errors in naturally generated questions.
""")
    code("""qualitative = ROOT / 'results/qualitative_examples.json'
if qualitative.exists():
    inspected = json.loads(qualitative.read_text(encoding='utf-8'))
    print(inspected['reviewer'])
    display(pd.DataFrame(inspected['examples'])[['language','category','question','answer','qa_answer','review']])
stress = ROOT / 'results/verifier_stress/summary.json'
if stress.exists():
    display(pd.DataFrame(json.loads(stress.read_text(encoding='utf-8'))))
for name in ['lora_test', 'lora_novel_test']:
    paired = ROOT / 'results' / name / 'adaptation_delta.json'
    if paired.exists():
        print(name, json.loads(paired.read_text(encoding='utf-8')))
loss_plot = ROOT / 'results/data/lora_loss_curve.png'
if loss_plot.exists():
    display(Image(filename=str(loss_plot), width=700))
""")
    markdown("""## 7. Demonstrate and explain

Launch `python -m streamlit run app.py`. Choose a language, paste a passage, supply exact
answers or use TF-IDF suggestions, and inspect the questions and evidence. Export a
student worksheet, answer key, CSV, JSON or Moodle XML.

See `docs/DEFENSE_GUIDE.md` for course connections and technical Q&A.  No external paid API is used.
""")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (QuestionForge)",
                "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
                "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"questionforge-{i:02d}"
    target = ROOT / "notebooks/QuestionForge.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
