"""PDF ingestion with PyMuPDF.

Handles the failure modes that actually show up in production:

* password-protected PDFs (clear, actionable error)
* scanned / image-only PDFs (detected, optional OCR fallback)
* repeated page headers and footers (stripped before parsing)
* pathological files (page cap, size cap, corrupt streams)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from utils.config import settings
from utils.exceptions import (
    EmptyDocumentError,
    EncryptedPDFError,
    PDFExtractionError,
)
from utils.logger import get_logger
from utils.text_utils import normalize_text

logger = get_logger("parsers.pdf_reader")

_MIN_CHARS_PER_PAGE = 80
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?\d{1,3}\s*(?:of|/)\s*\d{1,3}\s*$|^\s*page\s*\d{1,3}\s*$",
    re.IGNORECASE,
)


@dataclass
class PDFDocument:
    text: str
    page_count: int
    pages_used: int
    char_count: int
    word_count: int
    is_scanned: bool
    ocr_used: bool = False
    truncated: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "page_count": self.page_count,
            "pages_used": self.pages_used,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "is_scanned": self.is_scanned,
            "ocr_used": self.ocr_used,
            "truncated": self.truncated,
            "metadata": self.metadata,
        }


def _clean_metadata(raw: dict) -> dict:
    keep = ("title", "author", "creator", "producer", "creationDate", "modDate")
    cleaned: dict = {}
    for key in keep:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            cleaned[key] = value.strip()[:200]
    return cleaned


def _strip_repeated_furniture(pages: list[list[str]]) -> list[list[str]]:
    """Remove headers/footers that repeat on most pages (letterheads, page numbers)."""
    if len(pages) < 3:
        return pages
    counter: Counter[str] = Counter()
    for lines in pages:
        candidates = {line.strip() for line in lines[:2] + lines[-2:] if line.strip()}
        counter.update(candidates)

    threshold = max(3, int(len(pages) * 0.7))
    repeated = {line for line, count in counter.items() if count >= threshold and len(line) < 120}

    cleaned = []
    for lines in pages:
        filtered = [line for line in lines if line.strip() not in repeated]
        cleaned.append([line for line in filtered if not _PAGE_NUMBER_RE.match(line)])
    return cleaned


def _ocr_pages(doc: "fitz.Document", limit: int) -> str:
    """Render pages to images and OCR them. Requires pytesseract + Tesseract."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PDFExtractionError(
            "This PDF appears to be scanned (no embedded text) and OCR is not installed.",
            hint="Install OCR extras: pip install pytesseract pillow && sudo apt install tesseract-ocr",
        ) from exc

    parts: list[str] = []
    for index in range(min(doc.page_count, limit)):
        try:
            pixmap = doc.load_page(index).get_pixmap(dpi=200, colorspace=fitz.csGRAY)
            image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
            text = pytesseract.image_to_string(image)
            if text.strip():
                parts.append(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR failed on page %s: %s", index + 1, exc)
    return "\n".join(parts)


def read_pdf(
    data: bytes | bytearray,
    *,
    password: str | None = None,
    filename: str | None = None,
) -> PDFDocument:
    """Extract normalised text from a PDF byte string."""
    if not data:
        raise EmptyDocumentError("The PDF is empty.")

    try:
        doc = fitz.open(stream=bytes(data), filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - fitz raises several types
        raise PDFExtractionError(
            "The PDF could not be opened. It may be corrupted.",
            hint="Try re-exporting the resume from your editor.",
        ) from exc

    try:
        if doc.needs_pass:
            if not doc.authenticate(password or ""):
                raise EncryptedPDFError(
                    "This PDF is password-protected.",
                    hint="Remove the password or upload an unlocked copy.",
                )

        page_count = doc.page_count
        pages_used = min(page_count, settings.max_pdf_pages)
        if pages_used == 0:
            raise EmptyDocumentError("The PDF contains no pages.")

        raw_pages: list[list[str]] = []
        for index in range(pages_used):
            page = doc.load_page(index)
            raw_pages.append(page.get_text("text", sort=True).splitlines())

        pages = _strip_repeated_furniture(raw_pages)
        text = normalize_text("\n".join("\n".join(lines) for lines in pages))

        char_count = len(text)
        is_scanned = char_count < _MIN_CHARS_PER_PAGE * max(1, pages_used)
        ocr_used = False

        if is_scanned and settings.enable_ocr:
            logger.info("PDF looks scanned; attempting OCR (%s pages)", pages_used)
            text = normalize_text(_ocr_pages(doc, pages_used))
            char_count = len(text)
            ocr_used = bool(text)
            is_scanned = char_count == 0

        if not text.strip():
            raise EmptyDocumentError(
                "No selectable text was found in this PDF.",
                hint=(
                    "It is likely a scanned image. Enable OCR (ENABLE_OCR=true) with "
                    "Tesseract installed, or upload a text-based PDF."
                ),
            )

        truncated = False
        if len(text) > settings.max_resume_chars:
            text = text[: settings.max_resume_chars]
            char_count = len(text)
            truncated = True

        document = PDFDocument(
            text=text,
            page_count=page_count,
            pages_used=pages_used,
            char_count=char_count,
            word_count=len(text.split()),
            is_scanned=is_scanned and not ocr_used,
            ocr_used=ocr_used,
            truncated=truncated,
            metadata=_clean_metadata(doc.metadata or {}),
        )
        logger.info(
            "Parsed '%s': %s pages, %s words, scanned=%s, ocr=%s",
            filename or "upload",
            document.pages_used,
            document.word_count,
            document.is_scanned,
            document.ocr_used,
        )
        return document

    finally:
        doc.close()


def read_pdf_file(path: str | Path, *, password: str | None = None) -> PDFDocument:
    path = Path(path)
    if not path.is_file():
        raise PDFExtractionError(f"File not found: {path}")
    return read_pdf(path.read_bytes(), password=password, filename=path.name)
