# Slide input and answer-scope regression

A user reported a French question requesting four domains while the target answer
contained only one uppercase heading. The original native T5 mode reproduced the
reported output exactly. This is an input/target-scope failure, not evidence by itself
that model weights overfit the training data.

The cheap selector gave headings and number ranges high scores. Physical PDF line
breaks also split explanations into fragments. The old 48-item learning candidate cap
could then fill with headings/fragments before reaching later explanatory material.

## Current application behavior

- Join clear physical line wraps while retaining paragraph/list/heading boundaries.
  Keep the raw input in the session and expose the prepared passage. Generated offsets
  and exported evidence refer to the prepared passage.
- Automatic selection excludes standalone outline labels and source questions. Prefer
  explanatory relation spans and complete enumeration targets. When a long document
  exceeds the candidate cap, retain material across the document rather than only its start.
- Refuse outline-only automatic generation. An explicit multi-item request paired with
  an obviously incomplete short target is excluded, even if a QA checker accepts it.
  This narrow structural check is not a semantic correctness classifier.
- When the verified 4B weights are installed, the app initially selects the hybrid.
  T5-only comparison mode explicitly states that Qwen is not running. Existing quizzes
  retain their original method; changing the menu requires generating a new quiz.

The private development reproduction supplied 36 eligible candidates after preparation.
The hybrid selected the complete four-item list for its four-domain question and repaired
a draft that confused corruption with fighting corruption. Cleaning inputs alone did
not make all native T5 questions faithful. The raw lecture and reproduction remain in
ignored local cache; the repository contains synthetic structural regression fixtures
and a separate human-authored holdout.

No model weights are updated for this repair. The earlier trained adapters remain
unchanged. Regression/unit tests check heading exclusion, late-document coverage,
source offsets, three-language wrap handling, complete lists and numbered references.

## Generalization rather than example fitting

The reported lecture is development data. The frozen
[generalization protocol](GENERALIZATION_PROTOCOL.md) reserves 12 unused public
human-authored paragraphs across English, French and Arabic and 50 additional SciQ
test pairs. No model/prompt/threshold changes are selected from that test. It separates
end-to-end automatic selection/writing from fixed-answer fine-tuning evaluation.

Week 10 physical slides 7, 11, 19, 20 and 41 motivate dropout, early stopping,
train/validation/test separation, configuration search and source-level error analysis.
The eight-setting training search uses validation-only selection. Neither lower loss
nor a software-test count demonstrates that generated questions are useful or faithful.
