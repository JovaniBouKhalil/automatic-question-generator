# Frozen generalization audit — 28 November 2026

Record this protocol before sampling or evaluating the new holdout. The user-reported
French lecture is a development regression, not test evidence. Its text and raw
reproduction stay in ignored local cache; no course passage is republished.

## Freeze and scope

Freeze the classroom preparation, fact selection, planner/editor prompts, list-scope
checks, checkpoints, decoding and trained adapters before the first holdout output.
There are no new optimizer updates. No change may be selected using this audit's scores
or examples. A later revision needs another untouched evaluation set; this set then
becomes development evidence. Preserve every omission, rejection and execution error.

The classroom profile uses the pinned Qwen3-4B-Instruct-2507 in FP32 CPU, native
unadapted T5/mT5/AraT5 drafts, beam 4 for T5 and greedy Qwen decoding. Three requested
questions are a maximum. No manual answers, reference questions or topic-specific
learning objectives are supplied to automatic selection.

## Arm A: full automatic workflow

Select four human-authored paragraphs per language, 12 in total, with seed 20261128.
Use cached published evaluation sources at the existing pinned revisions:

- English: SQuAD validation, `rajpurkar/squad`, revision
  `7b6d24c440a36b6815f21b70d25016731768db1f`, CC BY-SA 4.0.
- French: NewsQuAD-fr validation, `lincoln/newsquadfr`, revision
  `6aca57928d2edaffa6f9a29bdecaa789c28d0391`, CC BY-NC-SA 4.0.
- Arabic only: TyDiQA-GoldP validation, `google-research-datasets/tydiqa`, revision
  `da78f23f9119363459acbaf46bf89426ff26c259`, Apache-2.0 metadata with underlying
  Wikipedia attribution. Filter the multilingual file to IDs starting `arabic-`.

Require valid supplied answer offsets and 100–1500 source characters. These are input
eligibility rules, not filters on model success. Exclude normalized paragraphs and
reference-question strings appearing in any prepared training/development/test file or
previous saved result. Exclude previously seen article titles where available; select
distinct known titles within each language. French source records lack article titles,
so only paragraph/question disjointness can be established for that language. These
audits do not prove absence of semantic paraphrases or public-checkpoint pretraining exposure.

Sort eligible IDs, group by paragraph, then shuffle deterministically. Do not inspect
content to choose appealing topics or replace difficult selected examples. Save the
source digests, protection-input digests, selected IDs and data bytes before inference.

To test document layout, wrap each paragraph at 68 characters without breaking words
or hyphens and prepend generic chapter/notes headings. These structural variants are
deterministic; the factual content remains human-authored. Retain the original and
prepared sources. All generated offsets refer to the prepared source. The human
reference question/answer is retained for provenance but is not supplied to the planner.

Compare planned native T5 drafts with the refined questions on identical selected
answers. Report source/planned/returned/rejected counts, errors, raw plans/edits, QA
diagnostics and source-linked qualitative observations for all 12 cases. Do not calculate
BLEU against a human reference about a different automatically selected answer. QA
answer-token F1 is not correctness, usefulness or a held-out classification F1.

## Arm B: fine-tuning generalization

Select 50 additional valid SciQ published-test question/answer pairs using the same seed,
excluding all prior prepared or evaluated paragraph/question strings. The whole raw
SciQ evaluation pool was already protected from this project's training. Use the cached
source with published SHA-256 verified against the original preparation manifest,
dataset `allenai/sciq`, revision `2c94ad3e1aafab77146f384e23536f97a4849815`, CC BY-NC 3.0.

Evaluate original English T5 and the already validation-selected SciQ adapter on the
same passages and supplied exact target answers, plus a human-reference QA control.
Use existing generation/metrics and beam 4; do not reselect weights or hyperparameters.
Report BLEU, chrF++, ROUGE-L, QA recovery and paired paragraph-bootstrap intervals,
including regressions and all errors. This tests adaptation on additional held-out rows
from the same corpus; it is not proof of transfer to arbitrary lecture subjects.

## Interpretation and course connection

Week 10 physical slides 7/11/19/20/41/42 cover dropout, early stopping, split roles,
configuration search, failure analysis and fair comparison. The earlier eight-setting
science grid already selects learning rate/rank/dropout by validation loss. This audit
adds genuinely unused project-level evaluation rows instead of improving the score on
the reported lecture. It does not claim that unseen-by-us means unseen upstream.

Twelve automatic cases and 50 fixed-answer examples are limited evidence. Translated
or related topics may remain dependent; French article identity is unavailable. No
representative human correctness labels or measured learning gains are claimed.
