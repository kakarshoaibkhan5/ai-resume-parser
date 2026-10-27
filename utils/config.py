"""Configuration for ai-resume-parser.

Every knob is environment-driven so the app can run unchanged on a laptop,
in Docker, or on a locked-down offline host.

Values are read once at import time from ``.env`` (if present) and the process
environment. See ``.env.example`` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved from the environment."""

    # --- identity -----------------------------------------------------------
    app_name: str = os.getenv("APP_NAME", "ai-resume-parser")
    app_version: str = "1.0.0"
    base_dir: Path = BASE_DIR

    # --- models -------------------------------------------------------------
    ner_model_id: str = os.getenv("NER_MODEL_ID", "dslim/bert-base-NER")
    summarizer_model_id: str = os.getenv(
        "SUMMARIZER_MODEL_ID", "facebook/bart-large-cnn"
    )
    model_cache_dir: Path = _as_path(
        os.getenv("MODEL_CACHE_DIR"), BASE_DIR / "models" / "weights"
    )
    # When true the app never touches the network: it only loads models that
    # were pre-downloaded into ``model_cache_dir``.
    offline_mode: bool = _as_bool(os.getenv("OFFLINE_MODE"), False)
    device: str = os.getenv("DEVICE", "auto")  # auto | cpu | cuda
    enable_ner: bool = _as_bool(os.getenv("ENABLE_NER"), True)
    enable_summarizer: bool = _as_bool(os.getenv("ENABLE_SUMMARIZER"), True)
    enable_embeddings: bool = _as_bool(os.getenv("ENABLE_EMBEDDINGS"), True)

    # --- PDF ingestion ------------------------------------------------------
    max_upload_mb: int = _as_int(os.getenv("MAX_UPLOAD_MB"), 15)
    max_pdf_pages: int = _as_int(os.getenv("MAX_PDF_PAGES"), 20)
    max_resume_chars: int = _as_int(os.getenv("MAX_RESUME_CHARS"), 40_000)
    enable_ocr: bool = _as_bool(os.getenv("ENABLE_OCR"), False)

    # --- summarisation ------------------------------------------------------
    summary_bullets: int = _as_int(os.getenv("SUMMARY_BULLETS"), 3)
    summary_max_input_chars: int = _as_int(
        os.getenv("SUMMARY_MAX_INPUT_CHARS"), 3_600
    )
    summary_max_chunks: int = _as_int(os.getenv("SUMMARY_MAX_CHUNKS"), 2)
    summary_max_length: int = _as_int(os.getenv("SUMMARY_MAX_LENGTH"), 130)
    summary_min_length: int = _as_int(os.getenv("SUMMARY_MIN_LENGTH"), 30)
    summary_num_beams: int = _as_int(os.getenv("SUMMARY_NUM_BEAMS"), 2)

    # --- job matching -------------------------------------------------------
    weight_skills: float = _as_float(os.getenv("MATCH_WEIGHT_SKILLS"), 0.50)
    weight_semantic: float = _as_float(os.getenv("MATCH_WEIGHT_SEMANTIC"), 0.25)
    weight_experience: float = _as_float(os.getenv("MATCH_WEIGHT_EXPERIENCE"), 0.15)
    weight_education: float = _as_float(os.getenv("MATCH_WEIGHT_EDUCATION"), 0.10)

    # --- operations ---------------------------------------------------------
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir: Path = _as_path(os.getenv("LOG_DIR"), BASE_DIR / "logs")
    log_to_file: bool = _as_bool(os.getenv("LOG_TO_FILE"), True)
    debug: bool = _as_bool(os.getenv("APP_DEBUG"), False)
    demo_mode: bool = _as_bool(os.getenv("DEMO_MODE"), False)

    # --- paths --------------------------------------------------------------
    assets_dir: Path = BASE_DIR / "assets"
    sample_resume_path: Path = BASE_DIR / "assets" / "sample_resume.pdf"

    # --- derived helpers ----------------------------------------------------
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def weights_total(self) -> float:
        return (
            self.weight_skills
            + self.weight_semantic
            + self.weight_experience
            + self.weight_education
        )

    def resolve_model_dir(self, model_id: str) -> Path:
        """Map a Hugging Face id to its pre-downloaded local directory."""
        return self.model_cache_dir / model_id.replace("/", "__")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
