# Full-chapter input

Upload a text PDF or UTF-8 file, choose up to 20 questions, and generate using the
same final Qwen/T5/QA workflow. There is no PDF page-count cap. The app accepts files
up to 50 MB and up to one million extracted/pasted characters.

For a small request that fits the original model-input budget, the previous inference
path is retained. Larger documents are divided at paragraph/sentence boundaries where
possible, using the actual Qwen tokenizer. Each model request is budgeted for the
passage, candidates, instructions, objective and draft questions rather than counting
source characters alone. Longer quizzes use multiple bounded requests.

Every section is scanned locally for candidate facts. With no objective, selected
sections are spread across the document; an objective prioritizes sections with
lexical relevance. The requested question count determines how many eligible sections
reach Qwen. This is section sampling, not a claim that the model reads or tests every
fact in a long chapter. The UI and saved report show selected/total section counts.

Exact manual targets are located within the document, including targets near the end
or spanning an initial section boundary. Generated target offsets are remapped to the
prepared full document. The independent QA checker uses the corresponding section;
it does not repeatedly process the entire chapter for each question. JSON exports
retain the source section ranges.

Repeated questions are removed. Partial section failures remain in the generation
report and trigger a warning, rather than being represented as a fully successful
run. PDF extraction does not add OCR: scanned/image-only pages still need OCR text.

The implementation does not increase the neural models' context windows or retrain
on uploaded chapters. Chunking can lose a relationship across a boundary; candidate
selection and question wording still require source review. Larger question counts
take more time on CPU. The earlier frozen small-passage evaluation does not establish
long-document question quality. Actual user-reported chapter reproductions are
private development checks; no instructor's chapter text is republished.

Restart Streamlit once if it still displays the old 10 MB upload limit after updating
the server configuration. The generation-method menu remains removed: this is one
application workflow with automatic sectioning.

## Reported chapter checks

The 37-page arrays PDF and 42-page linked-structures PDF each completed with three
questions, zero section errors and seven scanned/three selected sections. Actual
planning inputs ranged from 858 to 1,999 tokens; editing inputs ranged from 627 to
1,447. Current request construction replays the same source sections, candidates and
model settings as these runs. Only hashes/counts are published in
`results/quality/document_validation.json`; the instructor's text remains private.

Source inspection finds useful array-access and linked-addressing questions, but
also a broad array-idiom stem and a question from a next-chapter preview. One linked-list
question asks why a cost comparison holds while its target only restates that
comparison. It is QA-flagged for review. These examples show successful input handling
with remaining question-quality failures; the counts are not correctness scores.
