"""Upload validation runs before any expensive work, so it must be strict."""

from __future__ import annotations

import pytest

from utils.config import settings
from utils.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from utils.validators import safe_filename, validate_upload

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n"


def test_accepts_a_plausible_pdf() -> None:
    validate_upload("resume.pdf", PDF)  # must not raise


def test_uppercase_extension_is_accepted() -> None:
    validate_upload("RESUME.PDF", PDF)


@pytest.mark.parametrize("name", ["resume.docx", "resume.txt", "resume", "resume.pdf.exe"])
def test_rejects_non_pdf_extensions(name: str) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload(name, PDF)


def test_rejects_missing_filename() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload(None, PDF)


def test_rejects_empty_file() -> None:
    with pytest.raises(EmptyFileError):
        validate_upload("resume.pdf", b"")


def test_rejects_pdf_extension_without_pdf_magic() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload("resume.pdf", b"PK\x03\x04 this is really a zip")


def test_rejects_oversized_file() -> None:
    # Shrink the cap instead of allocating a 15 MB buffer in the test.
    original = settings.max_upload_mb
    object.__setattr__(settings, "max_upload_mb", 0)
    try:
        with pytest.raises(FileTooLargeError):
            validate_upload("resume.pdf", PDF)
    finally:
        object.__setattr__(settings, "max_upload_mb", original)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Aarav Mehta Resume.pdf", "Aarav-Mehta-Resume"),
        ("../../etc/passwd.pdf", "passwd"),
        ("weird name !!!.pdf", "weird-name"),
        (None, "resume"),
        ("...", "resume"),
    ],
)
def test_safe_filename(raw, expected) -> None:
    assert safe_filename(raw) == expected


def test_safe_filename_is_length_capped() -> None:
    long_name = "a" * 200 + ".pdf"
    assert len(safe_filename(long_name)) <= 80
