"""Quantify residual near-duplicate and article-title overlap without retuning test sets."""

import json
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from questionforge.data import assert_disjoint, load_jsonl
from questionforge.experiment import fingerprint_file


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main():
    partitions = {split: load_jsonl(f"data/processed/{split}.jsonl") for split in ["train", "dev", "test"]}
    assert_disjoint(partitions)
    results = []
    for language in ["en", "fr", "ar"]:
        training = [r for r in partitions["train"] if r["language"] == language]
        # Fit only on training paragraphs: held-out text cannot define the vocabulary or IDF.
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=150000,
                                     dtype=np.float32, lowercase=False)
        vectors = vectorizer.fit_transform([normalize(r["context"]) for r in training])
        train_titles = {normalize(r.get("title", "")) for r in training if r.get("title", "").strip()}
        for split in ["dev", "test"]:
            held = [r for r in partitions[split] if r["language"] == language]
            similarities = cosine_similarity(vectorizer.transform([normalize(r["context"]) for r in held]), vectors)
            closest = similarities.argmax(axis=1)
            maxima = similarities[np.arange(len(held)), closest]
            pairs = [{"held_out_id": row["id"], "training_id": training[int(index)]["id"],
                      "char_ngram_cosine": float(value), "flagged_at_0_90": bool(value >= .9),
                      "held_out_title": row.get("title", ""), "training_title": training[int(index)].get("title", "")}
                     for row, index, value in zip(held, closest, maxima)]
            titled = [r for r in held if r.get("title", "").strip()]
            results.append({"language": language, "held_out_partition": split, "n_training": len(training),
                            "n_held_out": len(held), "exact_paragraph_overlaps": 0,
                            "near_duplicate_candidates_cosine_ge_0_90": sum(p["flagged_at_0_90"] for p in pairs),
                            "maximum_char_ngram_cosine": float(maxima.max()),
                            "held_out_rows_with_title": len(titled),
                            "held_out_rows_sharing_training_article_title": sum(normalize(r["title"]) in train_titles for r in titled),
                            "closest_pairs": sorted(pairs, key=lambda p: -p["char_ngram_cosine"])[:5]})
    output = {"scope": "Post-hoc leakage sensitivity audit; no records removed and no generation settings tuned.",
              "threshold": .9, "representation": "Training-only TF-IDF character 3-5 grams, cosine; cap 150000 features per language.",
              "limitations": "A heuristic can miss paraphrases and cross-language duplicates. Different paragraphs from the same article can share facts. Missing titles do not imply article-disjointness. This does not audit upstream pretrained model exposure.",
              "data_sha256": {split: fingerprint_file(Path(f"data/processed/{split}.jsonl")) for split in partitions},
              "results": results}
    Path("results/data/overlap_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in r.items() if k != "closest_pairs"} for r in results], indent=2))


if __name__ == "__main__":
    main()
