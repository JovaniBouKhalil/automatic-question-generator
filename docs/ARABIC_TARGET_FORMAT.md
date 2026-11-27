# Arabic native target-format correction

The [AraT5 QG author's example](https://huggingface.co/Mihakram/AraT5-base-question-generation)
removes a `question:` prefix from decoded outputs. Arabic training must model that native
prefix as well; final user-facing questions still have the prefix removed by the existing
generation code. English and French checkpoints do not use this output prefix.

A development-only audit used four original ARCD development examples and four TyDiQA
development examples. Token-weighted loss of the unadapted Arabic model was 16.9436 with
plain reference questions, and 4.5575 with `question: ` prepended. These are eight-case
diagnostics, not the full development result and not generation-quality scores.

The initial plain-target Arabic attempt was stopped, preserved in ignored local backups,
and excluded from reported completed runs. Restart Arabic from the original pinned base,
using the same 1,990 clean training examples, 150 development examples, seed, optimizer,
maximum 512 updates and early-stopping rule. Include the prefix in the 64-token target
budget and reject overlong targets. Use the same native target format for step-zero and
all checkpoint-selection losses. This correction uses no test generation or test scores.

The completed English/French runs retain their exact original trainer in each adapter
bundle's `training_source.py`. Arabic retains the corrected trainer source. Raw source
hashes in the manifests must match these snapshots; optimizer states and aborted Arabic
attempts are not part of the submitted adapters. This addendum is hashed by the corrected
Arabic run and does not rewrite the original expanded-training protocol.
