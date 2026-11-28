"""Long-input requests retain source coverage and exact document-relative answer offsets."""

import io

import pytest
from pypdf import PdfWriter

from questionforge.documents.generation import create_document_quiz, request_fits, split_sections


def source_document():
    return "\n\n".join(f"Topic {i}\nDevice {i} measures water pressure and records readings for later comparison. "
                        f"The controller in section {i} detects changes to avoid damage in zone {i}. " * 7 for i in range(20))


def generator_calls(calls):
    def generate(context, language, count, **kwargs):
        pool = kwargs["candidate_builder"](context, language)
        targets = kwargs["manual_answers"]
        chosen = ([{"answer": a, "answer_start": context.index(a), "evidence": a} for a in targets]
                  if targets else pool[:count])
        calls.append((context, count, kwargs, pool))
        return [{"question": f"What is stated about {f['answer']}?", "answer": f["answer"],
                 "answer_start": f["answer_start"], "evidence": f["evidence"], "language": language}
                for f in chosen], {"returned": len(chosen), "candidate_pool": pool}
    return generate


def test_sections_cover_every_character_without_gaps_or_exceeding_the_token_budget(chat_tokenizer):
    source = source_document()
    sections = split_sections(source, chat_tokenizer, 300)
    assert len(sections) > 10 and "".join(s["text"] for s in sections) == source
    for first, second in zip(sections, sections[1:]):
        assert first["end"] == second["start"]
    assert all(len(chat_tokenizer.encode(s["text"])) <= 300 for s in sections)


def test_long_quiz_spreads_selected_sections_and_remaps_offsets(chat_tokenizer):
    calls = []
    source = source_document()
    rows, report = create_document_quiz(source, "en", 3, tokenizer=chat_tokenizer, create=generator_calls(calls))
    assert report["document"]["mode"] == "sectioned" and len(calls) == 3
    assert report["document"]["sections_total"] > report["document"]["sections_used"]
    assert rows[0]["answer_start"] < len(source) / 3 and rows[-1]["answer_start"] > 2 * len(source) / 3
    for row in rows:
        assert source[row["answer_start"]:][:len(row["answer"])] == row["answer"]
    for text, count, _, pool in calls:
        assert request_fits(text, "en", count, "", pool, chat_tokenizer)


def test_manual_target_near_document_end_is_not_lost_to_first_section(chat_tokenizer):
    calls = []
    source = source_document() + "\n\nThe final controller uses a sapphire sensor to measure pressure."
    rows, _ = create_document_quiz(source, "en", 1, manual_answers=["sapphire sensor"],
                                  tokenizer=chat_tokenizer, create=generator_calls(calls))
    assert len(rows) == 1 and rows[0]["answer_start"] == source.index("sapphire sensor")
    assert "sapphire sensor" in calls[0][0] and len(calls[0][0]) < len(source)


def test_short_request_keeps_the_existing_inference_settings(chat_tokenizer):
    calls = []
    source = "A pressure sensor measures the force exerted by a fluid."
    rows, report = create_document_quiz(source, "en", 2, objective="Understand measurement.",
                                       tokenizer=chat_tokenizer, create=generator_calls(calls))
    assert rows and report["document"]["mode"] == "single" and calls[0][0] == source
    settings = calls[0][2]
    assert settings["beams"] == 4 and settings["checkpoint"] is None and settings["role_focus"]


def test_more_than_eight_questions_are_batched_without_repeating_the_same_facts(chat_tokenizer):
    calls = []
    source = "\n".join(f"Sensor {chr(65 + i)} monitors the pressure in zone {chr(65 + i)}." for i in range(12))
    rows, report = create_document_quiz(source, "en", 12, tokenizer=chat_tokenizer, create=generator_calls(calls))
    assert len(rows) == 12 and len({r["question"] for r in rows}) == 12
    assert len(calls) == 4 and all(n <= 3 for _, n, _, _ in calls)
    assert report["document"]["sections_used"] == 1 and report["document"]["section_requests"] == 4


def test_section_failure_is_preserved_while_other_sections_can_complete(chat_tokenizer):
    calls = []
    normal = generator_calls(calls)
    attempts = []

    def intermittent(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 2:
            raise RuntimeError("fixture section failure")
        return normal(*args, **kwargs)

    rows, report = create_document_quiz(source_document(), "en", 3, tokenizer=chat_tokenizer, create=intermittent)
    assert rows and report["section_errors"] == 1
    assert "fixture section failure" in report["sections"][1]["error"]


def test_pdf_over_forty_pages_is_read_completely(monkeypatch):
    from app import read_upload

    writer = PdfWriter()
    for _ in range(45):
        writer.add_blank_page(width=400, height=400)
    stream = io.BytesIO()
    writer.write(stream)
    upload = io.BytesIO(stream.getvalue())
    upload.name, upload.size = "chapter.pdf", len(stream.getvalue())
    monkeypatch.setattr("pypdf._page.PageObject.extract_text", lambda self: "An independently extracted page.")
    assert read_upload(upload).count("An independently extracted page.") == 45


def test_text_above_twenty_thousand_characters_is_accepted():
    from app import read_upload

    data = ("A sensor measures water pressure.\n" * 1000).encode()
    upload = io.BytesIO(data)
    upload.name, upload.size = "chapter.txt", len(data)
    assert read_upload(upload) == data.decode()


@pytest.mark.parametrize("count", [0, 21])
def test_invalid_question_count_is_rejected(count, chat_tokenizer):
    with pytest.raises(ValueError, match="between one and 20"):
        create_document_quiz("A sensor measures water pressure.", "en", count, tokenizer=chat_tokenizer)
