"""Upload validation performed before any parsing work starts."""

from __future__ import annotations

import re
from pathlib import Path

from utils.config import settings
from utils.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)

_PDF_MAGIC = b"%PDF-"


def _human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def validate_upload(filename: str | None, data: bytes | None) -> None:
    """Raise a typed error if the uploaded file is not a plausible PDF."""
    if not filename:
        raise UnsupportedFileTypeError(
            "No file selected.", hint="Choose a PDF resume to continue."
        )
    if Path(filename).suffix.lower() != ".pdf":
        raise UnsupportedFileTypeError(
            f"'{Path(filename).suffix or filename}' files are not supported.",
            hint="Export your resume as PDF and try again.",
        )
    if not data:
        raise EmptyFileError(
            "The uploaded file is empty.", hint="Re-export the PDF and retry."
        )
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File is {_human_mb(len(data))}, above the "
            f"{settings.max_upload_mb} MB limit.",
            hint="Compress the PDF or raise MAX_UPLOAD_MB in your .env file.",
        )
    if _PDF_MAGIC not in data[:1024]:
        raise UnsupportedFileTypeError(
            "The file does not look like a valid PDF.",
            hint="The file may be corrupted or renamed from another format.",
        )


def safe_filename(name: str | None, *, fallback: str = "resume") -> str:
    """Filesystem-safe slug used for downloads and temp files."""
    stem = Path(name or fallback).stem
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-._")
    return (slug or fallback)[:80]
