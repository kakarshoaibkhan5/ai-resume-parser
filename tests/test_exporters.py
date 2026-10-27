"""The download payload is a public contract; these tests pin its shape."""

from __future__ import annotations

import json
from datetime import datetime

from utils.config import settings
from utils.exporters import (
    SCHEMA_VERSION,
    build_export,
    download_filename,
    to_json_bytes,
)

EXTRACTION = {
    "name": "Aarav Mehta",
    "skills": [{"name": "Python"}],
    "experience": {"years": 9.1},
}
SUMMARY = {"bullets": ["One.", "Two.", "Three."]}
MATCH = {"score": 83, "verdict": "Excellent match"}


def test_minimal_export_has_version_and_generator() -> None:
    payload = build_export(EXTRACTION)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["generator"] == {
        "name": settings.app_name,
        "version": settings.app_version,
    }
    assert payload["extraction"] == EXTRACTION
    assert payload["summary"] == {}
    assert payload["source"] == {}
    assert "job_match" not in payload
    assert "meta" not in payload


def test_generated_at_is_an_iso_utc_timestamp() -> None:
    payload = build_export(EXTRACTION)
    parsed = datetime.fromisoformat(payload["generated_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_optional_sections_are_attached() -> None:
    payload = build_export(
        EXTRACTION,
        summary=SUMMARY,
        match=MATCH,
        source={"filename": "resume.pdf", "pages": 1},
        meta={"duration_ms": 42},
    )
    assert payload["summary"] == SUMMARY
    assert payload["job_match"] == MATCH
    assert payload["source"]["filename"] == "resume.pdf"
    assert payload["meta"]["duration_ms"] == 42


def test_export_does_not_mutate_its_inputs() -> None:
    snapshot = json.dumps([EXTRACTION, SUMMARY, MATCH], sort_keys=True)
    build_export(EXTRACTION, summary=SUMMARY, match=MATCH)
    assert json.dumps([EXTRACTION, SUMMARY, MATCH], sort_keys=True) == snapshot


def test_json_bytes_round_trip() -> None:
    payload = build_export(EXTRACTION, summary=SUMMARY, match=MATCH)
    decoded = json.loads(to_json_bytes(payload))
    assert decoded == payload


def test_json_preserves_unicode() -> None:
    payload = build_export({"name": "Zoë Müller", "note": "日本語"})
    raw = to_json_bytes(payload)
    assert "Zoë Müller".encode("utf-8") in raw
    assert json.loads(raw)["extraction"]["note"] == "日本語"


def test_json_serialises_unknown_objects_via_str() -> None:
    class Custom:
        def __str__(self) -> str:
            return "custom-value"

    payload = build_export({"thing": Custom()})
    assert json.loads(to_json_bytes(payload))["extraction"]["thing"] == "custom-value"


def test_download_filename_is_slugified() -> None:
    assert download_filename("Aarav Mehta Resume.pdf") == "Aarav-Mehta-Resume_parsed.json"


def test_download_filename_supports_a_custom_suffix() -> None:
    assert download_filename("cv.pdf", suffix="match") == "cv_match.json"


def test_download_filename_handles_missing_input() -> None:
    assert download_filename(None) == "resume_parsed.json"


def test_download_filename_cannot_escape_the_directory() -> None:
    assert download_filename("../../etc/passwd.pdf") == "passwd_parsed.json"
