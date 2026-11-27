# Follow-up evidence and reproduction

The original results remain intact. `EXTENSION_PROTOCOL.md` records choices before the
additional runs. These extend previously inspected test sets; they are not a fresh final
test benchmark. No settings or application defaults were selected from their test scores.

## Training variability

The original 40-update English adapter is retained in `artifacts/english_lora`. Three new
160-update runs use seeds 13, 42 and 73, the same rank/learning rate/effective batch, and
development-only checkpoint selection. The runs took 870.6 seconds combined on this CPU.
All adapters and manifests are included under `artifacts/seed_study/seed{seed}`.
Their original generated adapter cards link to the initial LoRA evaluation as background;
the actual follow-up outputs are `results/seed{seed}_{public|classroom}_test`.

| Seed | Public QA F1 change | Classroom QA F1 change |
|---|---:|---:|
| 13 | +2.76 | +2.78 |
| 42 | +2.76 | +4.86 |
| 73 | +1.85 | −2.78 |

Public seed-mean change: +2.46 points, between-seed sample SD 0.52, topic-bootstrap 95%
interval [−1.67, 8.29]. Classroom: +1.62, SD 3.95, interval [−3.70, 9.03]. The bootstrap
holds these three trained seeds fixed and resamples source topics; it does not capture
all training variability. Three repeats of the same 64 English test inputs do not provide
192 independent test inputs. Seed 42 has the lowest development loss and is the optional
application adapter. It is not selected because of its classroom test gain.

## Residual data similarity and article isolation

`results/data/overlap_audit.json` checks original train/dev/test partitions using training-only
character 3–5 gram TF-IDF, cosine threshold 0.90 and normalized nonempty article titles.
No test paragraph reaches the near-duplicate threshold. There is one Arabic development
candidate (0.913 cosine). Titles overlap training in 11/15 English development, 13/15 Arabic
development, and 4/40 Arabic test rows. French titles are absent, so title isolation cannot
be established. The heuristic does not establish absence of semantic or upstream leakage.

A separate English seed-42, 160-update sensitivity removes the 23 training rows whose titles
appear in development/test. It retains 577 training rows and the same 15 development and
40 public test inputs. The adapter is `artifacts/article_isolated_lora`; outputs are
`results/article_{public|classroom}_test`, with split IDs and summary in
`results/article_sensitivity`. Public QA F1 changes by +2.76 points [0, 8.03]; classroom by
+0.69 [−11.82, 14.58]. More isolation and less training coverage change together. This one
run cannot establish an improvement caused by article isolation. It does not choose the app adapter.

## Two independent QA models, different failure patterns

Alternative: [multilingual BERT fine-tuned on augmented XQuAD](https://huggingface.co/mrm8488/bert-multi-cased-finetuned-xquadv1),
pinned revision `610161fac355eac3e06ac622d376e178a948a5d0`. The author's card describes
Arabic among its QA adaptation languages; French remains zero-shot. The card describes
augmentation without full example provenance. XQuAD comes from SQuAD, making benchmark
overlap an additional concern. License metadata is unspecified; weights are not redistributed.

This model has no impossible-question training. Its impossible-answer heuristic is disabled.
The same fixed target-F1/span-score/leakage rules evaluate both models; their span scores
are uncalibrated and cannot be interpreted as comparable probabilities of correctness.
Architecture and training data both change, so the comparison does not isolate a causal
effect of multilingual fine-tuning.

The controlled study uses original synthetic classroom topics: 3 development and 12 test.
Each language has 24 test references and 72 constructed negatives: unrelated questions,
unsupported location qualifiers, and an existing reference paired with the other intended
answer in its topic. A wrong intended target is distinct from an unanswerable question.
There are 720 model/control records including development, with zero execution failures.

| Test language | Reference F1: original → alternative | Negative false acceptance: original → alternative |
|---|---:|---:|
| English | 77.8 → 98.6 | 1.4% → 33.3% |
| French | 73.8 → 91.3 | 13.9% → 33.3% |
| Arabic | 83.3 → 86.1 | 18.1% → 31.9% |

The alternative improves reference recovery but accepts more unsupported/wrong-target cases.
Balanced control accuracy changes −5.56, +2.78 and −2.78 points respectively, with all paired
topic intervals including zero. The original default is retained. The app exposes the second
judge explicitly as experimental. These results do not measure natural question-error
prevalence or human-rated generation quality. Full condition breakdowns and intervals are
in `results/verification_study`.

## Human evaluation preparation, still pending

`results/human_evaluation/review_booklet.html` opens offline in a browser. It contains 54
randomized baseline/greedy/beam questions from three public inputs and three classroom topics
per language, sampled with seed 42. It hides model/method, reference question, verifier answer
and automatic scores. Reviewers see the full source, generated question and intended answer.
Anchored relevance/answerability/fluency scores use 0–2; unfilled answers stay blank.

Only rate languages you can judge. Use a pseudonym, work independently, export CSVs, then:

```powershell
python -m scripts.analyze_human_review path/to/reviewer1.csv path/to/reviewer2.csv
```

The analyzer rejects unknown IDs, malformed scores and duplicate reviewer/item votes. Method
means require complete paired input/reviewer blocks. Quadratic-weighted Cohen kappa uses
shared items and the full ordinal scale; constant-identical ratings have undefined kappa,
reported as null with exact agreement. A single reviewer cannot establish inter-rater agreement.
The app's **Review study** tab can also analyze uploaded CSVs without writing them to disk.
Current human results remain pending; browser tests use separate test data, not claimed ratings.

## Commands and checks

Original data preparation and four checkpoints are described in README. Download the optional
checker with `python scripts/download_models.py multilingual`. On the prepared project:

```powershell
python -m scripts.run_extensions --render
```

This resumes compatible completed experiments, preserves adapter bundles, refreshes analyses,
executes the notebook, adds the twelfth slide, runs tests and validates/package artifacts.
Close PowerPoint before regenerating; the delivered live deck was updated without closing it.
Without `--render`, PowerPoint is unnecessary; export the matching PDF separately before validation.
The ZIP contains small adapters and measured outputs, and excludes full weights, environments,
source archives, private backups and Office lock files. Test XML receipts and the structural
audit are in `results/quality` and `results/validation.json`.

Original experiment folders contain frozen source/configuration hashes. Use fresh output
folders if reproducing them from an extended source checkout; incompatible resume attempts
are deliberately rejected. Existing modules used for the original runs are unchanged.
