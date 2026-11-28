"""Preserve learning objectives and writer/draft provenance in teacher CSV exports."""

import csv
import io

from questionforge.export import as_csv


def learning_csv(rows):
    base = csv.DictReader(io.StringIO(as_csv(rows).lstrip("\ufeff"), newline=""))
    extra = ["learning_goal", "learning_objective", "question_type", "model_id", "model_revision",
             "draft_model_id", "draft_question", "planner_model", "edit_recipe"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=[*base.fieldnames, *extra])
    writer.writeheader()
    for exported, original in zip(base, rows):
        for field in extra:
            value = original.get(field, "")
            exported[field] = "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
        writer.writerow(exported)
    return "\ufeff" + stream.getvalue()
