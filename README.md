# ✦ QuestionForge

**Turn educational passages into factual questions in English, French and Arabic.**

NLP final project **05 — Automatic Question Generation**. A local Python/Streamlit app,
reproducible experimental comparison, and presentation. No paid API or GPU is required.

## Current application version

**Final application · One generation workflow · 28 November 2026.**
The current application is on `main`; earlier submission tags remain available for comparison.

The app has one fixed workflow: **Qwen3-4B selects facts → native T5/mT5/AraT5
drafts questions → Qwen refines them → independent QA flags answers for review.**
These steps run sequentially. Choose a language and question count; no model-version
menu is needed. Teacher review and wording corrections follow. Earlier model/adapter
comparisons remain in Experiment results and the notebook. See [the final workflow](docs/FINAL_WORKFLOW.md).

This update prepares wrapped document text, excludes outline headings from automatic
targets, retains complete list answers, and rejects obvious list/count mismatches.
The original passage is preserved alongside the prepared source. No model weights
are changed for this repair. See [the input regression](docs/CLASSROOM_INPUT_FIX.md).

Validation: **115 regular checks and 14 real-model integration checks**, with no skips.
The notebook has **11 executed code cells** and the presentation has **12 slides**.
These verify implementation, not question quality. A frozen audit uses **12 unused
human-authored multilingual paragraphs and 50 additional SciQ test pairs**, keeping
failures and omissions. See [the generalization findings](docs/GENERALIZATION_RESULTS.md)
and [the earlier development findings](docs/ANSWER_ROLE_RESULTS.md). Human usefulness
ratings remain pending.

## Start the application

On this computer, run from the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**. Choose a language, paste a passage or upload a text PDF,
and supply exact target answers (one per line) or use automatic source-fact selection.
Review questions beside their answers and source evidence. Export a printable student
worksheet, answer key, CSV, JSON, or Moodle short-answer XML.

Choose the questions to include in the exported quiz. QA-flagged questions start unselected
and can be included after reviewing their evidence. The printable worksheet includes the
reading passage and, when provided, the learning objective. Export choices do not remove
questions from saved experimental results or change the displayed QA statistics.
Expand **Edit question wording** to revise a draft. Edits clear its old QA scores and
start unselected for export. JSON/CSV retain the original wording and model provenance;
the revised writer is labeled `teacher_revision`. Empty revisions cannot be exported.

On Windows, you can also double-click `run_app.bat` after installation.

Install the core and Qwen weights using the setup commands below. Cached models work
offline. Generation is disabled with a setup message if the required Qwen weights are
missing. Automatic checks suggest questions for review; they do not certify correctness.

## Install on another computer

Clone the repository, or use GitHub's **Code → Download ZIP**, then open a terminal
inside the project folder. **Create your own `.venv`; do not copy another computer's
environment.** The trained adapters are included, so trying the app needs no retraining.
Full model weights download separately.

**Python 3.11 is the tested setup.** Exact package versions are recorded in
`requirements-lock.txt`. To clone with Git:

```powershell
git clone https://github.com/JovaniBouKhalil/automatic-question-generator.git
cd automatic-question-generator
```

Then, in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev,train]"
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe -m scripts.download_quality_model
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Linux/macOS: create a Python environment and replace `.\.venv\Scripts\python.exe` with
`.venv/bin/python`. PEFT is installed with the app for adapter inference. Downloads resume after
interruption, pin Hub commits, and check LFS SHA-256 hashes. About **3.7 GB** of inference weights
go into ignored `models/` for the core models. The final app also requires the 4B planner/editor weights.
Allow additional space for packages. Weights and course materials
are excluded from GitHub and the submission.

The Qwen setup command, included above, is:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_quality_model
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This adds an **8.04 GB** model download. FP32 Qwen parameters alone need about
**16 GB RAM**, plus the generator/checker and runtime memory; allow substantial
additional RAM. CPU generation can take several minutes. No paid API key is required.

## What is implemented

Learning-focused experiments target concepts, functions and explicit relationships.
The final app uses one local planner/T5/editor/checker workflow, with the original T5 draft
visible for inspection. Exact source spans and QA agreement
do not prove educational value. See [the learning-focused workflow](docs/LEARNING_FOCUS.md).

- Native answer-aware sequence-to-sequence models for each language, correct checkpoint
  prompts, deterministic greedy/beam decoding, and answer-preserving cropping.
- A transparent template baseline with the same passage and target answer.
- Independent extractive QA for answer agreement, lexical relevance and answer-leak checks.
  Weak questions stay visible with review reasons.
- Pinned datasets, exact span validation, deterministic paragraph-grouped splits, duplicate
  removal, and checks that evaluation paragraphs do not enter training.
- chrF++, BLEU, Unicode ROUGE-L, QA token F1/EM, coverage, latency, judge controls, and
  paragraph-group bootstrap confidence intervals. Failures stay in metric denominators.
- Original English LoRA experiments plus larger native English/French/Arabic training runs,
  with development-only checkpoint selection and resumable optimizer states.
- Arabic text direction, text/PDF input, student/teacher views, and classroom exports.

## Reproduce experiments

The [completed science training study](docs/SCIENCE_TRAINING_RESULTS.md) uses 2,043
usable English training examples and eight settings for learning rate, rank and dropout.
On 200 reserved science inputs, BLEU changes 31.83 → 42.11 and ROUGE-L 59.83 → 64.15.
QA answer F1 changes 72.63 → 74.11, with a paired difference interval crossing zero.
Development outputs include improved target focus and copied/malformed cloze regressions.
Human educational value remains unmeasured. Inspect all eight settings in **Experiment
results**; adapters remain reproducible research configurations, not alternate quiz modes.

A later frozen audit of 50 additional unused SciQ pairs shows mixed transfer:
BLEU **38.09 → 41.20**, but QA answer F1 **72.39 → 67.50** (paired difference
interval includes zero). This does not establish a general question-quality gain.
The app retains the unadapted generator as the default backbone; adapter selection
was not changed using these test results. See [the complete audit](docs/GENERALIZATION_RESULTS.md).

For the educational training and stronger-model follow-up, see
[the science training protocol](docs/SCIENCE_TRAINING_PROTOCOL.md) and
[the learning-focused protocol](docs/LEARNING_FOCUS_PROTOCOL.md). Monitor local work with:

```powershell
python -m scripts.learning_status
```

The final app requires `python -m scripts.download_quality_model` (8.04 GB).
It runs on CPU with sufficient memory; earlier models remain available through research scripts.
Do not launch a second training/downloader while one is active. Completed grid trials can
be reused; an interrupted grid trial has no optimizer-resume support.

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

An additional **90 synthetic parallel classroom examples** cover 15 scenarios: 18 development
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
verifier controls and 11 source-linked qualitative inspections in `results/qualitative_examples.json`.
English LoRA adds 2.5 QA F1 points on this 40-case subset (95% paired interval [0, 7.5]), with
no QA F1 change on 24 original classroom cases. This is limited evidence of improvement.

An **exploratory post-hoc** verifier probe adds an unsupported location to reference questions.
The fixed rules accept 1/24 English, 10/24 French and 12/24 Arabic probes. These are constructed
stress-test outcomes, not estimates of real-world question accuracy; no thresholds or generators
were tuned using this probe.

- [Walkthrough notebook](notebooks/QuestionForge.ipynb)
- [Rubric checklist](docs/RUBRIC_CHECKLIST.md)
- [Technical defense and demo guide](docs/DEFENSE_GUIDE.md)
- Presentation: `presentation/QuestionForge.pptx`, generated from measured outputs
- Submission ZIP: `submission/QuestionForge-submission.zip`, excludes caches and backbone weights
- Review handoff: `HANDOFF.md`, completed after validation

The supplied rubric requires **8–12 slides and no standalone technical report**. The presentation
generator creates 12 slides when the follow-up studies are present, with notes, figures, limits and source links.

## Larger native-language training

Development-selected trained adapters are retained for research and reproduction.
Selection includes the previous English adapter and the unadapted model; test results
do not choose it. The final quiz app uses native unadapted drafts with Qwen refinement.

Prepared training subsets contain 5,000 English, 1,000 French and 1,990 Arabic QA pairs, with
200/60/150 development examples. Arabic adds native human [TyDiQA-GoldP](https://huggingface.co/datasets/google-research-datasets/tydiqa)
data to ARCD. All published evaluation sources remain reserved, including unusable-offset
examples. Ten additional Arabic training passages were excluded in the full-pool audit.

English early stopping selects step 350 from a 600-update run. Its saved checkpoint saw 2,800
distinct examples; the complete run saw 4,800. French selects step 224 from 256 updates.
The Arabic run models the checkpoint's native `question:` target prefix. Actual Arabic
updates, all test differences and intervals are in [the measured training findings](docs/EXPANDED_TRAINING_RESULTS.md).
The selected checkpoint and the complete run must not be credited with the same exposure.
Held-out gains are mixed, so the new adapters are optional comparisons rather than an
assertion that extra training universally improves factual questions.

```powershell
python -m scripts.run_expanded_study
python -m scripts.expanded_training_status
```

This preserves completed adapters and resumes compatible interrupted runs. Small adapter
bundles include exact trainer snapshots; full backbone/data/optimizer files are excluded.
The **Review study** tab offers a separate 36-item blinded original/expanded comparison;
its human ratings remain pending. Original results and the 54-item decoding study are retained.

## Follow-up robustness studies

Original experiments are preserved. [The extension protocol](docs/EXTENSION_PROTOCOL.md) fixes
settings before the additional runs; these follow up previously inspected test sets.

| 160-update English LoRA seed | Public QA F1 change | Classroom QA F1 change |
|---|---:|---:|
| 13 | +2.76 points | +2.78 points |
| 42 | +2.76 points | +4.86 points |
| 73 | +1.85 points | −2.78 points |

The seed-mean public change is +2.46 points (topic-bootstrap 95% interval −1.67 to +8.29);
the classroom change is +1.62 (−3.70 to +9.03). These small samples do not establish a reliable
general improvement. The optional English app adapter uses development loss alone for selection.

A stronger character TF-IDF audit found no test near-duplicate candidates at cosine >= 0.90,
but found shared article titles and one Arabic development candidate. A separate English run
removes 23 training paragraphs sharing held-out titles; 577 remain. Its QA F1 changes are
+2.76 public and +0.69 classroom, with intervals including zero. This sensitivity changes
training coverage as well as isolation.

Two QA judges were compared on **720 paired model/control records** from 15 classroom topics.
On test references, answer F1 rises from 77.8 to 98.6 English, 73.8 to 91.3 French, and 83.3
to 86.1 Arabic. False acceptance on constructed negatives also rises: 1.4 to 33.3%, 13.9 to
33.3%, and 18.1 to 31.9%. The original default checker is retained; the alternative is an
explicit experimental option. Its extra 0.71 GB of weights are cached, not redistributed.

The **Review study** tab supplies a [54-item blinded booklet](results/human_evaluation/review_booklet.html)
and validated paired/inter-rater CSV analysis. **Human ratings remain pending.** See
[extension findings and reproduction instructions](docs/EXTENSIONS.md).

```powershell
python -m scripts.run_extensions --render
```

Omit `--render` on non-Windows systems; it requires an idle PowerPoint installation.
Use a fresh output directory for original experiments when source/configuration hashes differ;
the resume guard deliberately refuses incompatible folders.

## Interpretation limits

The [learning-focused experiments](docs/LEARNING_FOCUS.md) address the difference
between answer recovery and a useful classroom question. The T5-only research variant
targets explicit relations; the final app uses a local fact planner and question editor
around T5, followed by QA checks. The first T5-only comparisons still
contain relation and negation errors. Human learning-value ratings remain pending.

The completed [answer-role study](docs/ANSWER_ROLE_RESULTS.md) compares 18 fixed-target
cases and inspects nine additional synthetic topic cases. Purpose/comparison repairs
coexist with wording regressions, incomplete coverage and unsupported premises. The
research scripts can compare both editing styles; wording edits are tracked separately from
model predictions and clear their previous QA scores.

The [course alignment and question-quality audit](docs/COURSE_ALIGNMENT.md) compares the
supplied slides with the actual implementation. It documents the PyTorch/LoRA versus Keras
classroom differences and concrete generation/answer-selection failures. Course alignment
and passing software checks do not establish dependable multilingual question quality.

The main experiment supplies a target answer to both methods. It measures **answer-conditioned
generation**, not end-to-end automatic answer selection. The UI's automatic source-fact selection is a separate
heuristic. Single-reference overlap can penalize valid paraphrases. QA agreement and lexical
overlap are proxies; teacher review remains necessary. QA span scores are uncalibrated.
Rule-based acceptance is **coverage**, not measured question accuracy.

Our splits prevent leakage introduced here; they cannot establish that public checkpoints
never saw benchmark paragraphs upstream. Original scenarios reduce that concern but are small
and synthetic. Different language checkpoints have different sizes and training
histories. The Arabic uploader does not clearly specify a license; weights are downloaded for
this academic demonstration and are not redistributed.

## Version history

The repository was empty when work began. `baseline-v0` preserves the first baseline.
Subsequent commits provide reviewable checkpoints; no existing implementation was overwritten.
