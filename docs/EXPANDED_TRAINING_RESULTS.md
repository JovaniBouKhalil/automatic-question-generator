# Expanded-data training findings

This follow-up trains separate native-language LoRA adapters from the original pinned, already
task-tuned checkpoints. It preserves all original experiments and the three-seed study.
Selection uses development token loss; held-out generation is evaluated after selection.

## Actual data use and checkpoint selection

Prepared rows, rows seen during the whole run, and rows seen by the saved checkpoint differ.
Early stopping can discard later optimizer updates; they must not be attributed to saved weights.

| Language | Prepared train / dev | Run updates | Saved step | Distinct rows: full run / saved checkpoint | Dev loss: initial → best |
|---|---:|---:|---:|---:|---:|
| EN | 5,000 / 200 | 600 | 350 | 4,800 / 2,800 | 1.4221 → 1.3364 |
| FR | 1,000 / 60 | 256 | 224 | 1,000 / 1,000 | 2.0587 → 1.9273 |
| AR | 1,990 / 150 | 512 | 512 | 1,990 / 1,990 | 3.2673 → 1.2427 |

English is also compared with the previous seed-42 adapter on the same 200 development rows.
That adapter's loss is 1.3554. The app uses the lowest development-loss candidate, including the unadapted model.
Losses use each checkpoint's native tokenizer/target format and are not comparable across languages.

## Held-out generation

Differences below are percentage points against the same unadapted beam-4 generator.
These repeat the previously inspected original test sets; they are not newly untouched tests.
QA F1 is target-answer recovery by an imperfect independent verifier, not question accuracy.

| Language | Test subset | Inputs | QA F1: base → expanded | Difference [95% topic interval] | ROUGE-L: base → expanded |
|---|---|---:|---:|---:|---:|
| EN | public | 40 | 86.87 → 89.37 | +2.50 [0.00, 7.50] | 50.58 → 50.09 |
| EN | classroom | 24 | 71.53 → 73.61 | +2.08 [0.00, 6.25] | 63.76 → 62.23 |
| FR | public | 40 | 58.38 → 51.12 | -7.25 [-17.75, 2.71] | 31.33 → 28.77 |
| FR | classroom | 24 | 53.27 → 61.47 | +8.19 [-1.39, 19.86] | 54.49 → 51.17 |
| AR | public | 40 | 38.25 → 39.51 | +1.27 [-3.49, 6.21] | 31.91 → 31.46 |
| AR | classroom | 24 | 66.20 → 64.81 | -1.39 [-11.15, 5.56] | 44.43 → 45.43 |

All intervals resample whole paragraph/topic groups. One training seed per language cannot
establish broad stability. Data volume, development splits and training duration change together.
Neither lower loss nor higher overlap/QA agreement establishes fluency or factual support.

English comparison with the previous selected adapter:

- public: previous QA F1 89.64; expanded-minus-previous -0.26 [-0.79, 0.00].
- classroom: previous QA F1 76.39; expanded-minus-previous -2.78 [-12.50, 4.17].

These test comparisons do not override development-only app selection.

## Data provenance and the Arabic corrections

English and French use the original pinned SQuAD and NewsQuAD-fr snapshots.
Arabic adds [Google Research TyDiQA-GoldP](https://huggingface.co/datasets/google-research-datasets/tydiqa),
revision `da78f23f9119363459acbaf46bf89426ff26c259`, to the original ARCD source.
TyDiQA uses native human questions rather than translations; its metadata gives Apache-2.0,
and underlying Wikipedia attribution still applies. Other source permissions are in DATA_AND_MODELS.md.

- TyDiQA training: 8,497 Arabic records pass exact answer-span checks; 6,308 fail. Offsets are not guessed or silently realigned.
- TyDiQA published validation: 508 Arabic records pass exact answer-span checks; 413 fail. Offsets are not guessed or silently realigned.

All published validation source paragraphs remain reserved, including those with unusable answer
annotations. A final full-pool title audit excluded 10 additional Arabic training passages, leaving
1,990 prepared training rows. That preliminary attempt was preserved outside the submission.
The clean final run uses the corrected inputs from step zero.

The Arabic decoder expects `question: ` in its target format. A development-only format audit
detected the mismatch, and the final Arabic run restarts with the native prefix. The existing
generation code strips it for display. See ARABIC_TARGET_FORMAT.md for the measured diagnostic;
the abandoned plain-target attempt is not reported as a completed experiment.

Article grouping uses known titles; French has no reliable titles, so it uses paragraph groups.
These safeguards do not prove absence of semantic near-duplicates or upstream checkpoint exposure.

## Reproduction and review

```powershell
python -m scripts.run_expanded_study
python -m scripts.expanded_training_status
```

Install PEFT (now a runtime dependency), prepare the original data/models, and use Python 3.11.
The runner preserves completed compatible adapters and resumes unfinished optimizer state.
Each submitted adapter contains its exact training source and metadata; full weights, full datasets,
optimizer states and aborted attempts remain excluded. Prepared JSONL bytes are deterministic across platforms.

Development-selected native adapters remain research configurations. The final quiz app uses one native T5/Qwen workflow; held-out adapter gains remain mixed.
The new 36-item offline booklet in `results/expanded_human_evaluation` hides original/expanded model identity.
Its relevance, answerability and fluency ratings remain pending until people supply CSVs. The original
54-item baseline/decoding booklet is retained separately. No human score has been fabricated.

`results/expanded_training/before_after_cases.json` revisits all ten prior beam-search qualitative
cases with the new outputs and source passages. Original error labels describe the old questions
only; changed wording or higher QA agreement is not automatically labeled a factual repair.
