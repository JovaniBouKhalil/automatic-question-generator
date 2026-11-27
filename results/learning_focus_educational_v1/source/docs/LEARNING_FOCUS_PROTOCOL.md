# Learning-focused question generation protocol

Recorded before the first new model outputs, 27 November 2026.

Goal: prioritize explicitly stated concepts, functions, processes and cause/effect
relationships instead of incidental names, dates and colors. The assigned factual
seq2seq task remains central: native T5 generators create drafts from selected facts.

The experimental hybrid uses `Qwen/Qwen2.5-1.5B-Instruct`, pinned revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, as a local fact planner and draft editor.
The official model card gives Apache-2.0. Full weights remain outside the submission.
This is inference-time orchestration, not new fine-tuning, a trained usefulness scorer,
RAG or an independent factuality judge.
CPU inference dynamically quantizes linear layers to int8 using PyTorch; embeddings stay
in float32. Record this precision choice because outputs can differ from unquantized inference.

The planner proposes at most the requested number of exact answer/evidence spans,
learning goals and question types. It must avoid unstated explanations. The editor
receives those facts and T5 drafts. Schema, offsets, language and duplicate checks can
reject unusable outputs. They do not prove logical entailment or educational value.
The system may return fewer questions; it must not silently substitute shallow filler.

Initial development comparison: the three existing development classroom passages per
language (9 passage/language pairs), three requested questions per passage. Compare
original TF-IDF + base T5, planned-target + base T5 and the edited hybrid on the same
passages. Target selection changes between the original and planned methods, so their
QA F1 is not a controlled comparison on identical questions/answers. Draft versus editor
comparisons share selected targets. Report requested/returned/rejected counts, runtimes,
raw outputs, exact evidence grounding and QA diagnostics. Human usefulness/faithfulness
ratings remain pending; automatic checks are not educational-quality scores.

Also prepare three original educational demonstration passages per language in
`data/authoring/learning_passages.json` (training/validation/test roles, text representation,
and light/voltage). These are synthetic development examples, not independent tests or
human-validated references. The light/voltage passage expands a previously inspected topic.
Run the same three-way comparison in a separate `--source educational` output folder.
The explicit learning objectives are supplied to the planner/editor, not to the recall
baseline; this is a system comparison with extra instruction, not a model-only ablation.

An additional CPU-only T5 alternative selects whole answer spans after explicit relation
markers in English/French/Arabic (purpose, function, stated reason, property). It does not
use the new instruction model. Test it separately on the educational demonstration set;
record pattern selection failures and abstention. This is a transparent heuristic proposal,
not a semantic usefulness detector. An entered learning objective is recorded for review
in this mode; it does not condition T5 or the rule selector. Manual answers constrain targets.

Prompt debugging and implementation fixes use development data only. Any changed prompt
or pipeline receives a new recorded source/prompt fingerprint rather than overwriting
prior outputs. Prior inspected test examples may illustrate a known regression only;
they are not a fresh test set. No model configuration is chosen from those examples.

Default promotion requires inspecting actual outputs in all three languages. If the
small model fails, retain the mode as experimental or abstain rather than claim a
verified improvement. Preserve the original mode and all frozen experiments.

## Development addendum after the initial T5-only comparison

The v1 rule comparison returned 24/27 requested items with zero execution errors.
Inspection found relation/negation mistakes, particularly in French/Arabic. Preserve
that run and its exact source snapshots. In v2, retain the prevention word in “avoid” /
“éviter” targets and use the evidence sentence, adding up to two predecessors when it
begins with an anaphoric expression. This changes target boundaries and context together;
do not attribute any outcome to one change alone. Repeat the same development examples
in a new folder, without claiming fresh-test performance or completed human assessment.

Before the first instruction-model inference, refine its output schema to reference
numbered source sentences instead of requiring it to recopy entire evidence sentences.
The answer must still be an exact substring of the referenced sentence. The program
resolves the evidence quote and original offset directly from the passage. This reduces
fragile quote copying and output length; it does not validate semantic relations.

Source: [official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
