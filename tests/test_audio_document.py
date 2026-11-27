"""Video ids and the source document."""
from lifecompass.audio import extract_video_id
from lifecompass.document import SourceDocument, Unit


def test_video_id_is_found_in_every_url_shape():
    for url in ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=5s", "https://youtu.be/dQw4w9WgXcQ",
                "https://www.youtube.com/shorts/dQw4w9WgXcQ"):
        assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_document_keeps_its_units_through_json():
    doc = SourceDocument("v", [Unit(0, "اول."), Unit(1, "تاني."), Unit(2, "تالت.")])
    assert doc.full_text() == "اول.\nتاني.\nتالت."
    again = SourceDocument.from_json(doc.to_json())
    assert again.doc_hash == doc.doc_hash and again.units[1].text == "تاني."


def test_a_document_saved_with_times_still_loads():
    old = '{"version": 2, "video_id": "v", "stats": {}, "units": [{"index": 0, "text": "اول.", "t": 5}]}'
    assert SourceDocument.from_json(old).units[0].text == "اول."

