# Answer-role development refinement

Freeze this plan before the first refinement-pilot inference. The original FP32 4B
educational/classroom comparisons must finish first; their sources and outputs remain intact.

Source inspection of the educational comparison finds useful repairs and remaining target
alignment errors. The Arabic solar conversion asks for the converting device while its target
is the output energy; the leakage check removes that item. An Arabic validation question
still phrases data as using something, instead of asking what it is used for. An English
TF-IDF question asks why common words receive less weight while its assigned answer states
a weight contrast. The passage has a purpose statement, but that is a different answer.

The pilot uses six completed educational cases: solar/partition roles in Arabic, solar/text
representation in English, and partition/text representation in French. These are inspected
development examples, not a fresh final test. Reuse their exact facts, offsets, goals and T5
drafts. Keep the pinned 4B checkpoint, FP32 CPU, greedy decoding and full passage unchanged.

Compare two variants against the saved batch-edit baseline:

1. **Role-hinted editing:** add source-derived answer-focus cues and an explicit instruction
   to replace a faulty draft. Cues distinguish conversion output, stated purpose/reason,
   missing/recorded information, criterion and explicit comparative conditions.
2. **Role-hinted draft-free writing:** use the same cues and input, omitting only the T5
   draft question. This tests whether a wrong draft anchors the larger writer.

Cues are generic relation patterns, not hard-coded topic answers or final questions. Exact
answer/evidence strings still come from the source. Preserve all responses, rejections and
errors. The pilot stores its source hash and original-case hashes. Mechanical checks are
not correctness labels; inspect target alignment and grammatical roles in all three languages.

If a variant addresses the observed errors without clear new regressions, freeze it and
re-edit all 18 saved passage/language cases using the same targets/drafts. This isolates
wording refinement from a new target-selection run. Keep every original baseline case and
report any coverage change. Do not select by QA F1 or claim a completed human assessment.
Human educational-value ratings remain pending.
