"""Attach source-linked qualitative observations to actual measured examples.

Cases were selected purposively after inspecting outputs. These are explanatory notes,
not blinded human ratings, a representative random sample, or prevalence estimates.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("benchmark_test", "en:572a07c11d046914007796d8", "beam4", "Purpose confused with method",
     "The intended answer explains why the tribe maps its lands: strengthening territorial claims. The generated 'How' question asks for the mechanism, GPS/Google Earth. The question is answerable, but asks about a different fact than its supplied target."),
    ("benchmark_test", "en:5706149552bb891400689882", "beam4", "Ambiguous time relation",
     "The passage says the club played from 2005 to 2014 and was suspended after the 2014 season. Asking for a year it played does not uniquely identify 2014. The reference correctly asks about the suspension event."),
    ("benchmark_test", "en:5726d993f1498d1400e8ed18", "beam4", "Unsupported relation; verifier false positive",
     "The passage describes Trajan's Column in one court and a replica of Michelangelo's David in the other. The generated question combines them into Michelangelo's replica of Trajan's Column. QA still returns David with F1=1 and the automatic rules accept it. Target recovery does not establish faithfulness of every relation in a question."),
    ("benchmark_test", "en:573028fa04bcaa1900d7728a", "baseline", "Date/quantity error missed by QA",
     "The numeric template substitutes 'how many' for the year 1992, producing 'In how many'. The independent model nevertheless recovers 1992 with perfect answer agreement. This shows that QA agreement does not assess question grammar or the suitability of its question word."),
    ("benchmark_test", "fr:2739", "beam4", "Wrong semantic role",
     "The target is Ankara, which provided military assistance. The generated question asks who was attacked and the verifier extracts pro-Haftar. Asking about an affected party changes the relation from the source reference's assisting party."),
    ("benchmark_test", "fr:2681", "beam4", "Generic question loses intended focus",
     "The target 'manquer de temps' explains the narrator's frustration. The generated question asks generally about a consequence of research. It fails to ask about the specific reason supported by the target evidence."),
    ("benchmark_test", "ar:550131195698", "beam4", "Verifier false negative on an identical reference",
     "The generated Arabic question is exactly the reference: 'Who is Jundah's father?' The passage begins Jundah ibn Hujr ibn al-Harith. Both verifier runs select Kinda, the tribe, rather than Hujr. Here low QA F1 is a verifier error, not evidence that question generation failed."),
    ("benchmark_test", "ar:153159225042", "beam4", "Question type and answer-language mismatch",
     "The intended answer is Suleyman's Turkish written name. The generated Arabic 'Who is Sultan Suleyman?' question asks for an identity rather than that spelling. The verifier also fails on the original reference. Generator focus, mixed-script target matching and verifier transfer all complicate this case."),
    ("novel_test", "novel:weather_station:ar:1", "beam4", "Morphological answer leakage and relation error",
     "The source says a rain gauge measured rainfall. The question instead asks which rain gauge was measured, changing the measured object, while mentioning a near-synonym of the intended answer. The exact-token leakage rule misses this morphological variation and QA F1=2/3 still passes. This is an illustrative rule limitation, not a measured category prevalence."),
    ("novel_test", "novel:solar_lab:en:0", "beam4", "Supported classroom question",
     "The generated question asks who connected the solar panel to the lamp. The source names Rami as that actor. The target and extracted answer agree and the relation is supported by the passage."),
    ("novel_test", "novel:solar_lab:ar:0", "beam4", "Supported Arabic classroom question",
     "The generated Arabic question asks who connected the solar panel to the lamp, which matches the source's statement about Rami. This is a qualitative success example; automatic scores remain proxy measurements."),
]


def main():
    cache = {}
    examples = []
    for experiment, record_id, method, category, review in CASES:
        if experiment not in cache:
            path = ROOT / f"results/{experiment}/predictions.jsonl"
            cache[experiment] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        row = next(r for r in cache[experiment] if r["id"] == record_id and r["method"] == method)
        reference = next(r for r in cache[experiment] if r["id"] == record_id and r["method"] == "reference_control")
        fields = ["id", "language", "method", "context", "question", "reference", "answer", "qa_answer", "answer_f1", "accepted"]
        examples.append({"experiment": experiment, **{k: row.get(k) for k in fields},
                         "reference_qa_answer": reference["qa_answer"], "reference_qa_f1": reference["answer_f1"],
                         "category": category, "review": review})
    result = {"reviewer": "Automated qualitative inspection; human verification pending",
              "scope": "Purposively selected illustrative examples. No human scores or error-prevalence estimates.",
              "examples": examples}
    target = ROOT / "results/qualitative_examples.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(examples)} source-linked qualitative examples: {target}")


if __name__ == "__main__":
    main()
