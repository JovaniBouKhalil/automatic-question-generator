# Educational science adaptation protocol

Freeze this protocol before reading generated development/test outputs.

Source: [AllenAI SciQ](https://huggingface.co/datasets/allenai/sciq), revision
`2c94ad3e1aafab77146f384e23536f97a4849815`, CC BY-NC 3.0.
Welbl, Liu and Gardner (2017), *Crowdsourcing Multiple Choice Science Questions*.
This academic demonstration uses support passages, correct answers and human-authored
questions; distractor choices are not used. This is English-only curriculum adaptation,
not additional French/Arabic training or a completed human usefulness assessment.

Keep the published train/validation/test roles. Require a nonempty support passage and
an unambiguous case-insensitive answer substring; retain its actual source spelling and
offset. Reject missing, ambiguous or nonword answer matches, malformed questions and
identical duplicate questions. Group by normalized support passage; remove train groups
also in validation/test and validation groups also in test before sampling. Sample at
most 2,048 train, 128 development and 200 test records deterministically (seed 42),
one question per paragraph. Preserve preparation exclusions/counts and source hashes.
Do not open test questions or generated test predictions before selecting a checkpoint.
Public pretraining exposure cannot be excluded.

Base: pinned `lmqg/t5-small-squad-qg` in the project registry. Frozen backbone with
LoRA on attention q/v, FP32 CPU, seed 42, input budget 256 with answer-preserving
window, reject target sequences over 64 tokens, batch 2, gradient accumulation 4,
256 optimizer updates, AdamW weight decay .01, clip norm 1, linear scheduler and
16 warmup updates. Same shuffled example order and budget for every trial.

Full factorial grid: learning rate {1e-4, 3e-4}, rank {8, 16}, dropout {.05, .15};
alpha = 2 * rank. Evaluate token-weighted development cross-entropy at step zero,
128 and 256. Save each trial's best of these checkpoints. Select the global minimum
development loss, including the unadapted step-zero baseline. Ties favor fewer trained
updates then smaller rank. Keep all eight results, failed trials and exact source.
Do not choose by test metrics or by cherry-picking attractive generated questions.

After selection, evaluate baseline and selected checkpoint once on the same reserved
test questions/answers with deterministic beam 4, native T5 prompting and 384-token
answer-preserving windows. Report BLEU/ROUGE-L and diagnostic QA answer recovery,
alongside inspection of development generations. These metrics do not establish
factuality or educational usefulness. No invented human ratings. Keep earlier studies
unchanged. The instruction-model planning/editing study is a separate experiment.
