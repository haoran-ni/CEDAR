"""Images and PDFs prepared for a model to view, within what providers accept.

PNG, JPEG, GIF and WebP images go as they are when they fit. Other images Pillow can read
(TIFF, BMP, ...) are converted to PNG, and an image too large in bytes or pixels is scaled
down; the note says so. A PDF goes whole when it has at most ``MAX_PDF_PAGES`` pages;
otherwise the agent names the pages it wants, and they are copied into a smaller PDF. Every
page shown is sent again with each later model turn of the conversation, so viewing only the
pages needed keeps runs cheap.
"""

import base64
import io
import re

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from sim2real.agents.tools import ToolFailure
from sim2real.models.types import MediaPart

MAX_IMAGE_BYTES = 3_750_000
"""Encoded in base64 this stays under the 5 MB providers accept per image."""
MAX_IMAGE_SIDE = 8000
SCALED_SIDE = 2000
"""The longest side of an image that had to be scaled down."""
MAX_PDF_PAGES = 20
MAX_PDF_BYTES = 20_000_000

_NATIVE = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif", "WEBP": "image/webp"}


def is_pdf(data: bytes) -> bool:
    return data[:1024].lstrip().startswith(b"%PDF-")


def is_image(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return False


def viewable(data: bytes) -> bool:
    return is_pdf(data) or is_image(data)


def load_media(data: bytes, name: str, pages: str | None = None) -> tuple[MediaPart, str]:
    """The file as a part the model can view, and a note on what it shows; ToolFailure if it cannot be shown."""
    if is_pdf(data):
        return _pdf(data, name, pages)
    if pages is not None:
        raise ToolFailure("pages applies only to PDFs")
    return _image(data, name)


def _encoded(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _image(data: bytes, name: str) -> tuple[MediaPart, str]:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ToolFailure(f"{name} is neither an image nor a PDF that can be shown") from None
    width, height = image.size
    media_type = _NATIVE.get(image.format or "")
    if media_type and len(data) <= MAX_IMAGE_BYTES and max(width, height) <= MAX_IMAGE_SIDE:
        return MediaPart(media_type=media_type, data=_encoded(data), title=name), \
            f"{name}: {media_type} image, {width}×{height} pixels"
    notes = [f"converted from {image.format or 'an unknown format'}"] if media_type is None else []
    frames = getattr(image, "n_frames", 1)
    if frames > 1 and media_type is None:
        notes.append(f"first of {frames} frames")
    if len(data) > MAX_IMAGE_BYTES or max(width, height) > SCALED_SIDE:
        image.thumbnail((SCALED_SIDE, SCALED_SIDE))
        notes.append(f"scaled down from {width}×{height} to {image.size[0]}×{image.size[1]} pixels")
    if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    media_type = "image/png"
    if out.tell() > MAX_IMAGE_BYTES:
        out = io.BytesIO()
        image.convert("RGB").save(out, format="JPEG", quality=85)
        media_type = "image/jpeg"
        notes.append("saved as JPEG to fit")
    return MediaPart(media_type=media_type, data=_encoded(out.getvalue()), title=name), \
        f"{name}: {media_type} image, {image.size[0]}×{image.size[1]} pixels ({'; '.join(notes)})"


def parse_pages(spec: str, count: int) -> list[int]:
    """Page numbers as written by people ("1-3, 7"), checked against ``count`` pages; returned 0-based, in the
    order given."""
    chosen: list[int] = []
    for piece in spec.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", piece)
        if match is None:
            raise ToolFailure(f"cannot read pages {spec!r}; write them like '1-3, 7'")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if not 1 <= first <= last <= count:
            raise ToolFailure(f"pages {piece.strip()} are not within 1-{count}")
        chosen += [n - 1 for n in range(first, last + 1) if n - 1 not in chosen]
    return chosen


def _pdf(data: bytes, name: str, pages: str | None) -> tuple[MediaPart, str]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ToolFailure(f"{name} is a password-protected PDF")
        count = len(reader.pages)
    except (PdfReadError, ValueError, KeyError) as exc:
        raise ToolFailure(f"{name} is not a readable PDF: {exc}") from None
    if pages is None:
        if count > MAX_PDF_PAGES:
            raise ToolFailure(f"{name} has {count} pages; view at most {MAX_PDF_PAGES} at a time by naming them "
                              "with pages, e.g. '1-5'")
        if len(data) > MAX_PDF_BYTES:
            raise ToolFailure(f"{name} is {len(data):,} bytes; view fewer pages at a time by naming them with pages")
        return MediaPart(media_type="application/pdf", data=_encoded(data), title=name), \
            f"{name}: PDF, all {count} pages"
    chosen = parse_pages(pages, count)
    if len(chosen) > MAX_PDF_PAGES:
        raise ToolFailure(f"name at most {MAX_PDF_PAGES} pages at a time, not {len(chosen)}")
    writer = PdfWriter()
    for index in chosen:
        writer.add_page(reader.pages[index])
    out = io.BytesIO()
    writer.write(out)
    if out.tell() > MAX_PDF_BYTES:
        raise ToolFailure(f"pages {pages} of {name} take {out.tell():,} bytes; name fewer pages")
    shown = f"pages {pages} of {count}"
    return MediaPart(media_type="application/pdf", data=_encoded(out.getvalue()), title=f"{name}, {shown}"), \
        f"{name}: PDF, {shown}"
