"""PDF ingestion: happy path plus the failure modes that matter in production."""

from __future__ import annotations

import fitz
import pytest

from parsers.pdf_reader import PDFDocument, read_pdf, read_pdf_file
from utils.config import settings
from utils.exceptions import (
    EmptyDocumentError,
    EncryptedPDFError,
    PDFExtractionError,
)


def _text_pdf(pages: list[str], *, header: str | None = None) -> bytes:
    doc = fitz.open()
    for body in pages:
        page = doc.new_page(width=595, height=842)
        y = 60
        if header:
            page.insert_text((52, y), header, fontsize=10)
            y += 24
        for line in body.splitlines():
            page.insert_text((52, y), line, fontsize=10)
            y += 14
    data = doc.tobytes()
    doc.close()
    return data


def test_reads_the_demo_pdf(demo_pdf_bytes: bytes) -> None:
    document = read_pdf(demo_pdf_bytes, filename="demo.pdf")
    assert isinstance(document, PDFDocument)
    assert document.page_count == 1
    assert document.pages_used == 1
    assert document.is_scanned is False
    assert document.ocr_used is False
    assert document.truncated is False
    assert document.word_count > 100
    assert "Aarav Mehta" in document.text
    assert isinstance(document.metadata, dict)


def test_to_dict_has_a_stable_shape(demo_pdf_bytes: bytes) -> None:
    payload = read_pdf(demo_pdf_bytes).to_dict()
    assert set(payload) == {
        "page_count",
        "pages_used",
        "char_count",
        "word_count",
        "is_scanned",
        "ocr_used",
        "truncated",
        "metadata",
    }


def test_rejects_empty_bytes() -> None:
    with pytest.raises(EmptyDocumentError):
        read_pdf(b"")


def test_rejects_non_pdf_bytes() -> None:
    with pytest.raises(PDFExtractionError):
        read_pdf(b"this is definitely not a pdf")


def test_encrypted_pdf_requires_the_password() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Secret resume content for the encrypted document test.")
    data = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="s3cret",
    )
    doc.close()

    with pytest.raises(EncryptedPDFError):
        read_pdf(data)

    document = read_pdf(data, password="s3cret")
    assert "Secret resume content" in document.text


def test_repeated_headers_are_stripped() -> None:
    pages = [
        f"Body page one line {index} with enough words to look like real content.\n"
        f"More text for page {index} that should survive extraction."
        for index in range(4)
    ]
    document = read_pdf(_text_pdf(pages, header="ACME CONFIDENTIAL"))
    assert "ACME CONFIDENTIAL" not in document.text
    assert "Body page one line 0" in document.text


def test_page_cap_is_enforced(monkeypatch) -> None:
    original = settings.max_pdf_pages
    object.__setattr__(settings, "max_pdf_pages", 2)
    try:
        pages = [f"Page {index} body text with enough words to be extracted fine." for index in range(5)]
        document = read_pdf(_text_pdf(pages))
        assert document.page_count == 5
        assert document.pages_used == 2
    finally:
        object.__setattr__(settings, "max_pdf_pages", original)


def test_char_cap_marks_the_document_truncated(monkeypatch) -> None:
    original = settings.max_resume_chars
    object.__setattr__(settings, "max_resume_chars", 80)
    try:
        body = "A long resume line with many useful words. " * 20
        document = read_pdf(_text_pdf([body]))
        assert document.truncated is True
        assert document.char_count == 80
    finally:
        object.__setattr__(settings, "max_resume_chars", original)


def test_read_pdf_file_round_trips(tmp_path, demo_pdf_bytes: bytes) -> None:
    path = tmp_path / "resume.pdf"
    path.write_bytes(demo_pdf_bytes)
    document = read_pdf_file(path)
    assert "Aarav Mehta" in document.text


def test_read_pdf_file_reports_a_missing_path(tmp_path) -> None:
    with pytest.raises(PDFExtractionError):
        read_pdf_file(tmp_path / "missing.pdf")
