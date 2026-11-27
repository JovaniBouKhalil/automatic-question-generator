# Expanded-data training protocol

Starting release: `submission-v2`. This protocol is fixed before the expanded training
runs. Preserve all previous adapters, test outputs and the immutable extension protocol.
The original test outputs have already been inspected; this is follow-up training, not
a claim that those examples are an untouched final benchmark.

## Data

- English: the pinned SQuAD snapshot already used by the project, up to **5,000** training
  QA pairs from different paragraphs and **200** development pairs.
- French: the pinned NewsQuAD-fr snapshot, up to **1,000** training QA pairs and **60**
  development pairs. Prefer paragraph diversity, then add distinct QA pairs if needed.
- Arabic: the existing ARCD snapshot plus Google Research's **TyDiQA-GoldP** Arabic,
  Hugging Face revision `da78f23f9119363459acbaf46bf89426ff26c259`, up to **2,000**
  training QA pairs and **150** development pairs.
- Use only source training partitions for training/development. All published validation
  paragraphs remain reserved, not merely the 40 previously sampled test paragraphs.
- Keep every original development paragraph out of training. When titles are present,
  exclude matching article titles and split new development data by article; otherwise
  split by normalized paragraph. Validate exact source answer offsets; reject duplicates.
- Arabic TyDiQA is human-annotated rather than translated. Report its native-QA domain
  change, possible upstream model exposure, and source permissions; do not claim that
  its use alone eliminates contamination.
- Deterministic seed 42. Save source revisions, file hashes, record IDs and split audits.
  The prepared datasets and full backbone weights remain in ignored local directories.

## Training and selection

- Start each language from its original pinned T5/mT5/AraT5 QG checkpoint. Do not overwrite
  an existing adapter. Use LoRA rank 8, alpha 16, dropout 0.05, query/value attention
  projections, AdamW learning rate 0.0003, weight decay 0.01, gradient clipping 1.0.
- Use the checkpoint's native prompt, answer-preserving 256-token input windows, and a
  64-token reference-question budget. Reject overlong targets instead of silently
  training on truncated questions. Nominal effective batch 8; seed 42.
- Maximum updates: **800 English, 256 French, 512 Arabic**. Evaluate development token
  cross-entropy every 50 updates (French every 32), include step zero, keep the lowest
  development loss, and stop after five non-improving evaluations. Report actual updates,
  examples seen and unique training IDs seen rather than implying that all rows were used.
- CPU inference remains available. Hardware acceleration is optional; record actual
  device, precision, runtime and versions. Do not alter the working CPU environment just
  to claim GPU use. Save resumable optimizer/scheduler/random states in ignored outputs.
- For English, also score the previous development-selected seed-42 adapter on the same
  new development examples. Choose an optional app adapter by this shared development
  criterion; a worse new model must not replace a better existing one.

## Evaluation

- Freeze selected checkpoints before examining test generation. Compare unadapted beam-4
  and the new adapter on identical original public and classroom test inputs in each
  language, with original generation budgets and the original verifier.
- Report every language, including regressions; retain chrF++, BLEU, Unicode ROUGE-L,
  QA F1/EM, per-example outputs, failure counts and paired topic bootstrap differences.
- Validation loss and QA agreement are imperfect proxies. More updates and more examples
  are not guarantees of better factuality or fluency. Human review remains pending.
- Update the app, documentation and submission only after actual training and evaluation
  have succeeded. Exclude full datasets, backbone weights and optimizer checkpoints from
  GitHub and the submission; retain small adapter bundles and provenance/audit manifests.
