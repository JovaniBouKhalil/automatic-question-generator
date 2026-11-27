"""Regression for real Office import failures caused by duplicate middle-slide parts."""

from io import BytesIO
from zipfile import ZipFile

from pptx import Presentation
from pptx.util import Inches

from scripts.extend_presentation import append_blank_slide


def test_replacing_middle_slide_twice_keeps_unique_parts_and_other_content():
    deck = Presentation()
    for label in ["User first slide", "Generated results", "User last slide"]:
        slide = append_blank_slide(deck)
        slide.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(1)).text = label
    for _ in range(2):
        old = deck.slides[1]
        entry = next(e for e in deck.slides._sldIdLst if int(e.id) == old.slide_id)
        deck.part.drop_rel(entry.rId)
        deck.slides._sldIdLst.remove(entry)
        new = append_blank_slide(deck)
        new.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(1)).text = "Updated results"
        last = deck.slides._sldIdLst[-1]
        deck.slides._sldIdLst.remove(last)
        deck.slides._sldIdLst.insert(1, last)
        output = BytesIO()
        deck.save(output)
        with ZipFile(BytesIO(output.getvalue())) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
            assert archive.testzip() is None
        deck = Presentation(BytesIO(output.getvalue()))
        assert len(deck.slides) == 3
        assert deck.slides[0].shapes[0].text == "User first slide"
        assert deck.slides[1].shapes[0].text == "Updated results"
        assert deck.slides[2].shapes[0].text == "User last slide"
