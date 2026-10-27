"""JSON export helpers.

The public payload is versioned so downstream consumers can rely on a stable
contract even as extraction internals evolve.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from utils.config import settings
from utils.validators import safe_filename

SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_export(
    extraction: dict[str, Any],
    *,
    summary: dict[str, Any] | None = None,
    match: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the downloadable document."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "generator": {
            "name": settings.app_name,
            "version": settings.app_version,
        },
        "source": source or {},
        "extraction": extraction,
        "summary": summary or {},
    }
    if match:
        payload["job_match"] = match
    if meta:
        payload["meta"] = meta
    return payload


def to_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def download_filename(name: str | None, *, suffix: str = "parsed") -> str:
    return f"{safe_filename(name)}_{suffix}.json"
