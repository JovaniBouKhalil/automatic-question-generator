# Ready for your review — 27 November 2026

The project was built from the empty repository against your rubric. It includes native
English, French and Arabic question generation, real measured comparisons, a working local
demonstration, an executed notebook, slides and a submission ZIP. No paid API is used.

## Open it

Double-click **`run_app.bat`**, or run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://127.0.0.1:8501**. The environment and all model weights are already prepared on
this computer. Try a classroom example, enter a short exact answer, and inspect the question
beside its evidence. Switch to French and Arabic; Arabic inputs use right-to-left layout.
Export a worksheet, answer key, CSV, JSON or Moodle XML. Full setup/reproduction is in README.

## Deliverables

- `submission/QuestionForge-submission.zip` — code, dependencies, sources, results, notebook,
  presentation and small trained adapter; excludes environments, full weights and course files.
- `presentation/QuestionForge.pptx` — **11 editable slides**, with sources and technical speaker notes.
- `presentation/QuestionForge.pdf` — readable slide export; PNG previews are in `presentation/preview/`.
- `notebooks/QuestionForge.ipynb` — **7 executed code cells**, including real generation in all three languages.
- `docs/DEFENSE_GUIDE.md` — demo steps, course connections and technical questions to prepare.
- `docs/RUBRIC_CHECKLIST.md` — evidence mapped to the 100-point rubric.
- `AI_ASSISTANCE.md` — substantial assistance disclosed, consistent with the policy you described.

No standalone technical report was prepared, as the rubric does not require one.

## What the experiments show

Public test subset: **40 paragraph groups per language**. A separate original, AI-assisted
classroom supplement has **24 test questions per language**. Together these are 192 distinct
test question/answer pairs; repeated evaluations of the same IDs are not additional test data.

Scores below are on 0–100 scales. QA F1 measures agreement with the intended answer, not
the percentage of factually correct or fluent questions.

| Language | Baseline ROUGE-L | Beam-4 ROUGE-L | Baseline QA F1 | Beam-4 QA F1 |
|---|---:|---:|---:|---:|
| English | 28.69 | 50.58 | 87.06 | 86.87 |
| French | 19.91 | 31.33 | 76.56 | 58.38 |
| Arabic | 12.24 | 31.91 | 31.30 | 38.25 |

The neural model improves reference wording agreement in this benchmark. Answer recovery
does not improve universally. French news is also a domain change from the French generator's
Wikipedia-style training data. Different checkpoints, datasets and verifier transfer prevent
claiming that one language is intrinsically easier.

The English LoRA experiment trained **294,912 parameters (0.49%)** for 40 CPU updates.
The recorded training/validation loop took 64.2 seconds; development loss changed from
1.5944 to 1.5834. Public-test QA F1 rose from 86.87 to 89.37, but that gain comes from **one
improved example out of 40**, with 39 unchanged. The paired 95% interval is [0, 7.5] points.
There was no QA F1 change on the 24 new classroom cases. This is a modest experiment,
not strong evidence of a general improvement.

## The important limitation to explain

Independent QA is an imperfect judge. It is multilingual-pretrained but fine-tuned on
English SQuAD2. On public reference questions its F1 is 85.33 English, 71.78 French and
57.82 Arabic. A correct Arabic generated question identical to the reference still gets
the wrong QA answer in one inspected case.

The reverse failure also occurs: a question incorrectly combines Michelangelo's David
with Trajan's Column, yet the verifier returns David and passes it. The separate,
**exploratory post-hoc** unsupported-location probe passes 1/24 English, 10/24 French and
12/24 Arabic modified questions. These constructed cases diagnose limitations; they are
not estimates of natural question error rates. No model or threshold was tuned on them.

`results/qualitative_examples.json` contains 11 disclosed AI-assisted source inspections.
`human_review.csv` files remain blank for real human annotation. Do not describe them as
completed human evaluations. Teacher review of wording and factual support is part of the workflow.

## Validation and reproducibility

- **35 regular tests passed; 4 real neural integration tests passed.**
- Ruff checks passed; the notebook executed without errors.
- All main generation/verification experiment records have zero execution errors.
- Slides were rendered and visually inspected; PDF and ZIP integrity were checked.
- Dataset preparation rejected invalid spans/duplicates and removed **15 Arabic training
  rows overlapping evaluation passages**. Paragraph groups do not cross our partitions.
- Configurations, seeds, input hashes, checkpoint revisions, environment versions, per-example
  outputs, runtime records, group bootstrap intervals and control diagnostics are included.

Our split audit does not prove that public checkpoints never saw benchmark text upstream.
The original scenarios reduce that concern but are small, artificial and AI-assisted.
Full source/licensing details and remaining limitations are in `docs/DATA_AND_MODELS.md`.

## Before handing it in

1. Personalize the first slide with your name, group and course details.
2. Read the defense guide and trace the generator, answer-window crop, metrics and split audit.
3. Review examples in all languages and add genuine human ratings if you have time. Keep any
   claims about human review consistent with what you actually annotate.
4. Rehearse the local demo with cached models and explain both a success and a failure.
5. Keep the assistance disclosure accurate. The goal is a project you can explain, not just run.

The finished source is versioned on GitHub. `baseline-v0` preserves the initial template
baseline, and `work/multilingual-aqg` preserves the implementation branch. Your original
archives/PDF remain in Downloads; extracted working copies stay in ignored `.reference/`.
Model weights stay in ignored `models/`. The temporary overnight sleep-prevention request
is released when work finishes.
