"""Typed exceptions used across the pipeline.

The UI catches :class:`ResumeParserError` and shows a friendly message, while
unexpected exceptions are logged with a full traceback.
"""

from __future__ import annotations


class ResumeParserError(Exception):
    """Base class for all expected, user-facing failures."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class UnsupportedFileTypeError(ResumeParserError):
    pass


class FileTooLargeError(ResumeParserError):
    pass


class EmptyFileError(ResumeParserError):
    pass


class PDFExtractionError(ResumeParserError):
    pass


class EncryptedPDFError(PDFExtractionError):
    pass


class EmptyDocumentError(PDFExtractionError):
    pass


class ModelUnavailableError(ResumeParserError):
    pass


class SummarizationError(ResumeParserError):
    pass


class MatchingError(ResumeParserError):
    pass
