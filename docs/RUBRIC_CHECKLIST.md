# Rubric evidence checklist

This maps the supplied 100-point rubric to concrete artifacts. It is not a grade guarantee.

| Criterion | Points | Evidence |
|---|---:|---|
| Dataset suitability, sources, exploration, splits | 15 | Sources/licenses and manifests; exact span checks; near-duplicate/title audit; article-isolated sensitivity; expanded SQuAD/NewsQuAD-fr/TyDiQA+ARCD splits; SciQ science curriculum with full raw held-out passage/question protection |
| Baseline and advanced models, justification, comments | 25 | Transparent baseline; native T5/mT5/AraT5; three larger-data LoRA adapters; native Arabic target-format audit; answer-preserving cropping; exact training-source snapshots; course-facing notebook |
| Configuration, variations, fair comparison, reproducibility | 20 | Baseline/greedy/beam; original LoRA, three seeds and article sensitivity; larger 600/256/512-update runs; 21 original generation experiments; eight equal-budget SciQ trials varying learning rate/rank/dropout; validation-only selection; snapshots/hashes, exposure counts and recorded limitations of resume |
| Suitable metrics, figures, failures, interpretation | 20 | chrF++, BLEU, Unicode ROUGE-L, QA F1/EM; intervals/seed dispersion; mixed outcomes preserved; SciQ reserved-test results; 18 fixed-target and nine additional-topic source inspections with regressions; original 54-item and 36-item blinded booklets (human ratings pending) |
| Project-specific advanced component | 10 | Independent QA/relevance and review rules; 720 two-verifier controls; source-ID planning and T5-draft/4B editing; 12-response role pilot and failed three-response Arabic probe; teacher wording revisions clear stale QA and retain provenance; no claim that exact spans or QA establish educational value |
| Slides, working demonstration, technical Q&A | 10 | 12-slide PPTX/PDF with science search and learning path; native/science adapters, optional checker and editable selected exports; 11 executed code cells; defense guide; 95 regular and 12 real-model integration checks passed with no skips |

Before submission: annotate a representative sample as a human, personalize the slides, run the
demo in each language, and confirm you can explain every model/input/metric choice. The rubric
requires no standalone technical report. Multilingual support is an extension; experimental
rigor and your understanding remain essential.
