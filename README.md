# ✦ QuestionForge

**Turn educational passages into factual questions in English, French and Arabic.**

NLP final project **05 — Automatic Question Generation**. A local Python/Streamlit app,
reproducible experimental comparison, and presentation. No paid API or GPU is required.

## Start the application

On this computer, run from the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**. Choose a language, paste a passage or upload a text PDF,
and supply exact target answers (one per line) or use automatic TF-IDF suggestions.
Review questions beside their answers and source evidence. Export a printable student
worksheet, answer key, CSV, JSON, or Moodle short-answer XML.

On Windows, you can also double-click `run_app.bat` after installation.

The first neural run downloads pretrained weights. Cached models work offline. The template
baseline with answerability checks disabled works without model downloads. Automatic checks
suggest questions for review; they do not certify correctness.

## Install on another computer

Use **Python 3.11–3.13**. Exact tested versions are recorded in `requirements-lock.txt` after
validation. Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev,train]"
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Linux/macOS: create a Python environment and replace `.\.venv\Scripts\python.exe` with
`.venv/bin/python`. Training dependencies are optional for inference. Downloads resume after
interruption, pin Hub commits, and check LFS SHA-256 hashes. About **3.7 GB** of inference weights
go into ignored `models/`. Allow additional space for packages. Weights and course materials
are excluded from GitHub and the submission.

## What is implemented

- Native answer-aware sequence-to-sequence models for each language, correct checkpoint
  prompts, deterministic greedy/beam decoding, and answer-preserving cropping.
- A transparent template baseline with the same passage and target answer.
- Independent extractive QA for answer agreement, lexical relevance and answer-leak checks.
  Weak questions stay visible with review reasons.
- Pinned datasets, exact span validation, deterministic paragraph-grouped splits, duplicate
  removal, and checks that evaluation paragraphs do not enter training.
- chrF++, BLEU, Unicode ROUGE-L, QA token F1/EM, coverage, latency, judge controls, and
  paragraph-group bootstrap confidence intervals. Failures stay in metric denominators.
- A small English LoRA training experiment with development-only checkpoint selection.
- Arabic text direction, text/PDF input, student/teacher views, and classroom exports.

## Reproduce experiments

Activate the environment or replace `python` with `.\.venv\Scripts\python.exe`:

```powershell
python -m scripts.build_novel_data
python -m questionforge.cli prepare --seed 42 --train-size 600 --dev-size 15 --test-size 40
python -m scripts.analyze_data
python -m questionforge.cli evaluate --data data/processed/dev.jsonl --output results/benchmark_dev
python -m questionforge.cli evaluate --data data/novel/dev.jsonl --output results/novel_dev
python -m scripts.train_lora --steps 40 --device cpu
python -m questionforge.cli evaluate --data data/processed/test.jsonl --output results/benchmark_test
python -m questionforge.cli evaluate --data data/novel/test.jsonl --output results/novel_test
python -m questionforge.cli evaluate --data data/processed/dev.jsonl --output results/lora_dev --languages en --methods baseline beam4 lora_beam4 --checkpoint artifacts/english_lora
python -m questionforge.cli evaluate --data data/processed/test.jsonl --output results/lora_test --languages en --methods baseline beam4 lora_beam4 --checkpoint artifacts/english_lora
python -m questionforge.cli evaluate --data data/novel/test.jsonl --output results/lora_novel_test --languages en --methods baseline beam4 lora_beam4 --checkpoint artifacts/english_lora
python -m scripts.stress_verifier
python -m scripts.analyze_adaptation
python -m scripts.build_qualitative_review
python -m scripts.plot_results --results results/benchmark_test
python -m scripts.plot_results --results results/novel_test
python -m scripts.plot_results --results results/lora_test
python -m scripts.analyze_controls --results results/benchmark_test
python -m scripts.build_notebook
python -m scripts.execute_notebook
python -m scripts.build_presentation
python -m scripts.package_submission
```

Quick offline check:

```powershell
python -m questionforge.cli generate --method baseline --text "Lina planted nine seeds." --answer Lina
python -m questionforge.cli evaluate --data data/novel/dev.jsonl --output outputs/baseline_smoke --methods baseline --no-verify --limit 2
python -m pytest -q
```

For a real neural smoke check, remove `--no-verify`, use `--methods beam4`, and choose a new
output folder. Evaluation appends predictions immediately and resumes after interruption.
It rejects an output directory with a different configuration or input hash. Seed: **42**.
Baseline, greedy and beam-4 settings are fixed in advance; the test set is not used for selection.

`python scripts/run_all.py` automates dataset preparation, the bounded training run, evaluations
and figures, the executed notebook, and the PPTX. Start the model downloader separately first.
Per-stage logs are in `outputs/pipeline/`. PDF and PNG slide rendering is optional and uses an
existing Windows PowerPoint installation through `scripts/render_slides.ps1`.

## Data and models

| Language | Benchmark | Generator |
|---|---|---|
| English | [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) | [T5-small QG](https://huggingface.co/lmqg/t5-small-squad-qg) |
| French | [NewsQuAD-fr](https://huggingface.co/datasets/lincoln/newsquadfr) | [mT5-small FQuAD QG](https://huggingface.co/lmqg/mt5-small-frquad-qg) |
| Arabic | [ARCD](https://huggingface.co/datasets/hsseinmz/arcd) | [AraT5 QG](https://huggingface.co/Mihakram/AraT5-base-question-generation) |

An additional **90 AI-assisted parallel classroom examples** cover 15 scenarios: 18 development
and 72 test questions. Translations share topic groups. They were frozen before inference and
are a controlled supplement, not a human-validated benchmark.

The independent verifier is [distilled XLM-R SQuAD2](https://huggingface.co/deepset/xlm-roberta-base-squad2-distilled).
It has multilingual pretraining but English QA fine-tuning. Reference-question and unrelated-question
controls help measure its cross-lingual limitations.

See [data/model documentation](docs/DATA_AND_MODELS.md) for licenses, sources, partition details,
and upstream exposure limits. French news and English/Arabic Wikipedia have different domains;
scores across languages are not a controlled language ranking.

## Results and deliverables

Experiment folders contain configurations, environment versions, per-example predictions,
metric tables, group bootstrap intervals, figures and review examples. Human-rating cells in
`human_review.csv` intentionally stay blank until a person annotates them.

Measured public test subset, **40 paragraphs per language**, scores on 0–100 scales:

| Language | Baseline ROUGE-L | Beam-4 ROUGE-L | Baseline QA F1 | Beam-4 QA F1 |
|---|---:|---:|---:|---:|
| English | 28.69 | 50.58 | 87.06 | 86.87 |
| French | 19.91 | 31.33 | 76.56 | 58.38 |
| Arabic | 12.24 | 31.91 | 31.30 | 38.25 |

Reference agreement improves; answer recovery does not improve universally. QA agreement
misses some factual errors and rejects some correct questions. See the paired intervals,
verifier controls and 11 disclosed AI-assisted source inspections in `results/qualitative_examples.json`.
English LoRA adds 2.5 QA F1 points on this 40-case subset (95% paired interval [0, 7.5]), with
no QA F1 change on 24 original classroom cases. This is limited evidence of improvement.

An **exploratory post-hoc** verifier probe adds an unsupported location to reference questions.
The fixed rules accept 1/24 English, 10/24 French and 12/24 Arabic probes. These are constructed
stress-test outcomes, not estimates of real-world question accuracy; no thresholds or generators
were tuned using this probe.

- [Walkthrough notebook](notebooks/QuestionForge.ipynb)
- [Rubric checklist](docs/RUBRIC_CHECKLIST.md)
- [Technical defense and demo guide](docs/DEFENSE_GUIDE.md)
- [AI assistance disclosure](AI_ASSISTANCE.md)
- Presentation: `presentation/QuestionForge.pptx`, generated from measured outputs
- Submission ZIP: `submission/QuestionForge-submission.zip`, excludes caches and backbone weights
- Review handoff: `HANDOFF.md`, completed after validation

The supplied rubric requires **8–12 slides and no standalone technical report**. The presentation
generator creates 11 slides with speaker notes, measured results, figures, limits and source links.

## Interpretation limits

The main experiment supplies a target answer to both methods. It measures **answer-conditioned
generation**, not end-to-end automatic answer selection. The UI's TF-IDF selector is a separate
heuristic. Single-reference overlap can penalize valid paraphrases. QA agreement and lexical
overlap are proxies; teacher review remains necessary. QA span scores are uncalibrated.
Rule-based acceptance is **coverage**, not measured question accuracy.

Our splits prevent leakage introduced here; they cannot establish that public checkpoints
never saw benchmark paragraphs upstream. Original scenarios reduce that concern but are small,
artificial and AI-assisted. Different language checkpoints have different sizes and training
histories. The Arabic uploader does not clearly specify a license; weights are downloaded for
this academic demonstration and are not redistributed.

## Version history

The repository was empty when work began. `baseline-v0` preserves the first baseline.
Subsequent commits provide reviewable checkpoints; no existing implementation was overwritten.
