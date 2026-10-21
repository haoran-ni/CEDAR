"""Images and PDFs prepared for a model to view, and the file tool that shows them."""

import base64
import io

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

from sim2real.agents import ToolFailure
from sim2real.agents.media import MAX_IMAGE_BYTES, MAX_PDF_PAGES, SCALED_SIDE, load_media, parse_pages, viewable
from sim2real.agents.toolkits import FileAccess, file_tools


def image_bytes(size=(64, 32), fmt="PNG", mode="RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, "white").save(out, format=fmt)
    return out.getvalue()


def pdf_bytes(pages: int) -> bytes:
    writer = PdfWriter()
    for number in range(pages):
        writer.add_blank_page(width=100 + number, height=100)  # widths tell the pages apart
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def decoded(part) -> bytes:
    return base64.standard_b64decode(part.data)


def test_supported_images_go_as_they_are():
    data = image_bytes(fmt="JPEG")
    part, note = load_media(data, "stm.jpg")
    assert (part.media_type, decoded(part), part.title) == ("image/jpeg", data, "stm.jpg")
    assert note == "stm.jpg: image/jpeg image, 64×32 pixels"


def test_other_formats_are_converted_and_large_images_scaled_down():
    part, note = load_media(image_bytes(fmt="TIFF"), "scan.tif")
    assert part.media_type == "image/png" and "converted from TIFF" in note
    assert Image.open(io.BytesIO(decoded(part))).format == "PNG"

    part, note = load_media(image_bytes(size=(9000, 300)), "wide.png")
    shown = Image.open(io.BytesIO(decoded(part)))
    assert max(shown.size) == SCALED_SIDE and "scaled down from 9000×300" in note
    assert len(decoded(part)) <= MAX_IMAGE_BYTES


def test_short_pdfs_go_whole_and_long_ones_by_the_pages_named():
    short = pdf_bytes(3)
    part, note = load_media(short, "paper.pdf")
    assert (part.media_type, decoded(part), note) == ("application/pdf", short, "paper.pdf: PDF, all 3 pages")

    long = pdf_bytes(MAX_PDF_PAGES + 5)
    with pytest.raises(ToolFailure, match="naming them with pages"):
        load_media(long, "si.pdf")
    part, note = load_media(long, "si.pdf", pages="2-3, 25")
    pages = PdfReader(io.BytesIO(decoded(part))).pages
    assert [float(p.mediabox.width) for p in pages] == [101, 102, 124]
    assert note == "si.pdf: PDF, pages 2-3, 25 of 25" and part.title == "si.pdf, pages 2-3, 25 of 25"
    with pytest.raises(ToolFailure, match="at most"):
        load_media(long, "si.pdf", pages=f"1-{MAX_PDF_PAGES + 1}")


@pytest.mark.parametrize("spec, problem", [("0", "not within"), ("3-2", "not within"), ("9", "not within"),
                                           ("one", "cannot read"), ("1;2", "cannot read")])
def test_page_numbers_are_checked(spec, problem):
    with pytest.raises(ToolFailure, match=problem):
        parse_pages(spec, 8)


def test_what_cannot_be_shown_is_refused():
    assert viewable(image_bytes()) and viewable(pdf_bytes(1)) and not viewable(b"plain text\n")
    with pytest.raises(ToolFailure, match="neither an image nor a PDF"):
        load_media(b"\x00\x01binary", "restart.bin")
    with pytest.raises(ToolFailure, match="only to PDFs"):
        load_media(image_bytes(), "fig.png", pages="1")
    with pytest.raises(ToolFailure, match="not a readable PDF"):
        load_media(b"%PDF-1.7\nbroken", "broken.pdf")


def test_view_file_shows_images_and_pdfs_inside_the_readable_roots(tmp_path):
    runs, secrets = tmp_path / "runs", tmp_path / "runs" / "config"
    secrets.mkdir(parents=True)
    (runs / "dos.png").write_bytes(image_bytes())
    (runs / "notes.pdf").write_bytes(pdf_bytes(2))
    (secrets / "key.png").write_bytes(image_bytes())
    (tmp_path / "outside.png").write_bytes(image_bytes())
    read = []
    view = {t.name: t for t in file_tools(FileAccess([runs], deny=[secrets]), on_read=read.append)}["view_file"]

    shown = view.invoke({"path": str(runs / "dos.png")})
    assert not shown.is_error and shown.media[0].media_type == "image/png" and "dos.png" in shown.content
    assert view.invoke({"path": str(runs / "notes.pdf"), "pages": "2"}).media[0].media_type == "application/pdf"
    assert read == [runs / "dos.png", runs / "notes.pdf"]
    for path in (secrets / "key.png", tmp_path / "outside.png", "relative.png"):
        refused = view.invoke({"path": str(path)})
        assert refused.is_error and not refused.media
