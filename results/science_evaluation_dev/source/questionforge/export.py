"""Portable teacher/student exports with safe text escaping and Unicode support."""

import csv
import html
import io
import json
from datetime import datetime, timezone
from xml.etree import ElementTree as ET


def as_json(rows: list[dict]) -> str:
    return json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "questions": rows},
                      ensure_ascii=False, indent=2)


def as_csv(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    fields = ["question", "answer", "language", "evidence", "method", "qa_answer", "answer_f1", "accepted"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        safe = {}
        for key in fields:
            value = row.get(key, "")
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            safe[key] = value
        writer.writerow(safe)
    return "\ufeff" + stream.getvalue()


def worksheet_html(rows: list[dict], include_answers: bool = False) -> str:
    items = []
    for row in rows:
        answer = (f"<p class='answer'><strong>Answer / Réponse / الإجابة:</strong> {html.escape(row['answer'])}</p>"
                  f"<p class='source'>{html.escape(row.get('evidence', ''))}</p>") if include_answers else "<div class='blank'></div>"
        items.append(f"<li dir='auto'><p>{html.escape(row['question'])}</p>{answer}</li>")
    title = "QuestionForge · Answer key" if include_answers else "QuestionForge · Worksheet"
    return ("<!doctype html><html><head><meta charset='utf-8'><title>" + title + "</title>"
            "<style>body{font:18px Arial,sans-serif;max-width:800px;margin:48px auto;padding:0 24px;color:#172b35}"
            "h1{font-size:30px}li{padding:12px 0;break-inside:avoid}.blank{height:60px;border-bottom:1px solid #ccd5d9}"
            ".source{font-size:14px;color:#536573}.answer{color:#126457}@media print{body{margin:0}}</style>"
            "</head><body><h1>" + title + "</h1><p>Name / Nom / الاسم: ____________________</p><ol>"
            + "".join(items) + "</ol></body></html>")


def moodle_xml(rows: list[dict]) -> str:
    """Moodle short-answer import; instructors should review accepted answer variants."""
    root = ET.Element("quiz")
    for index, row in enumerate(rows, 1):
        question = ET.SubElement(root, "question", type="shortanswer")
        ET.SubElement(ET.SubElement(question, "name"), "text").text = f"QuestionForge {index}"
        ET.SubElement(ET.SubElement(question, "questiontext", format="plain_text"), "text").text = row["question"]
        ET.SubElement(question, "usecase").text = "0"
        answer = ET.SubElement(question, "answer", fraction="100", format="plain_text")
        ET.SubElement(answer, "text").text = row["answer"]
        ET.SubElement(ET.SubElement(question, "generalfeedback", format="plain_text"), "text").text = row.get("evidence", "")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
