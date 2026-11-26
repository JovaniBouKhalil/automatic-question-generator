"""Cheap, inspectable answer selection using TF-IDF and source-span heuristics."""

from dataclasses import dataclass
import re

from .text import STOPWORDS, evidence_for, sentences, tokens, validate_language


@dataclass(frozen=True)
class Candidate:
    answer: str
    start: int
    evidence: str
    score: float
    kind: str


def extract_candidates(context: str, language: str, limit: int = 5) -> list[Candidate]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    validate_language(language)
    if not 1 <= limit <= 20:
        raise ValueError("Choose between 1 and 20 target answers.")
    chunks = sentences(context)
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False,
                                 stop_words=sorted(STOPWORDS[language]), ngram_range=(1, 3))
    try:
        matrix = vectorizer.fit_transform([part.text for part in chunks])
    except ValueError:  # An all-stopword/empty passage has no meaningful answer candidates.
        return []
    terms = dict(zip(vectorizer.get_feature_names_out(), matrix.max(axis=0).toarray()[0]))
    proposed = []
    seen = set()
    for sentence in chunks:
        word_matches = list(re.finditer(r"[^\W_]+(?:['’\-][^\W_]+)*", sentence.text))
        for index, word in enumerate(word_matches):
            for length in range(1, 4):
                selected = word_matches[index:index + length]
                if len(selected) != length:
                    continue
                span = sentence.text[word.start():selected[-1].end()]
                words = tokens(span)
                if not words or words[0] in STOPWORDS[language] or words[-1] in STOPWORDS[language]:
                    continue
                if len(span) < 3 or len(span) > 65:
                    continue
                key = " ".join(words)
                if key in seen:
                    continue
                seen.add(key)
                numeric = any(char.isdigit() for char in span)
                proper = language != "ar" and all(part.group()[0].isupper() for part in selected)
                arabic_name = (language == "ar" and length <= 2 and index > 0 and
                               word_matches[index - 1].group() in {
                                   "زرع", "زرعت", "صنع", "صنعت", "قاد", "قادت", "ثبت", "ثبتت",
                                   "وصل", "وصلت", "سكب", "سكبت", "قاست", "قاس", "سجل", "سجلت"})
                kind = "number/date" if numeric else "name" if proper else "key phrase"
                score = float(terms.get(key, 0)) + 0.7 * numeric + 0.7 * proper + 0.4 * arabic_name + 0.03 * length
                start = sentence.start + word.start()
                proposed.append(Candidate(span, start, evidence_for(context, span, start).text, score, kind))
    proposed.sort(key=lambda c: (-c.score, c.start, len(c.answer)))
    chosen = []
    for candidate in proposed:
        # Prefer different facts/sentences and avoid asking nested spans of the same answer.
        if any((candidate.start < old.start + len(old.answer) and old.start < candidate.start + len(candidate.answer))
               or candidate.evidence == old.evidence for old in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) == limit:
            break
    return chosen
