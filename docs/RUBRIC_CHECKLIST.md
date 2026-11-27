# Rubric evidence checklist

This maps the supplied 100-point rubric to concrete artifacts. It is not a grade guarantee.

| Criterion | Points | Evidence |
|---|---:|---|
| Dataset suitability, sources, exploration, splits | 15 | Sources/licenses and manifests; exact span checks; near-duplicate/title audit; article-isolated sensitivity; expanded SQuAD/NewsQuAD-fr/TyDiQA+ARCD splits; SciQ science curriculum with full raw held-out passage/question protection |
| Baseline and advanced models, justification, comments | 25 | Transparent baseline; native T5/mT5/AraT5; three larger-data LoRA adapters; native Arabic target-format audit; answer-preserving cropping; exact training-source snapshots; course-facing notebook |
| Configuration, variations, fair comparison, reproducibility | 20 | Baseline/greedy/beam; original LoRA, three seeds and article sensitivity; larger 600/256/512-update runs; 21 original generation experiments; eight equal-budget SciQ trials varying learning rate/rank/dropout; validation-only selection; snapshots/hashes, exposure counts and recorded limitations of resume |
| Suitable metrics, figures, failures, interpretation | 20 | chrF++, BLEU, Unicode ROUGE-L, QA F1/EM; intervals/seed dispersion; mixed outcomes preserved; SciQ reserved-test results and source-linked regressions; original 54-item and 36-item blinded booklets; unblinded learning/science development comparisons (human ratings pending) |
| Project-specific advanced component | 10 | Independent QA/relevance and review rules; two-verifier ablation with 720 records; experimental source-ID learning-target planning and T5-draft editing, including actual small-editor failures; no claim that exact spans or QA establish educational value |
| Slides, working demonstration, technical Q&A | 10 | 12-slide PPTX/PDF with science-search results; app with native/science adapters, optional checker and exports; 11 executed cells; defense guide; 85 regular and 11 real-model integration checks passed; optional 4B check pending its download |

Before submission: annotate a representative sample as a human, personalize the slides, run the
demo in each language, and confirm you can explain every model/input/metric choice. The rubric
requires no standalone technical report. Multilingual support is an extension; experimental
rigor and your understanding remain essential.
