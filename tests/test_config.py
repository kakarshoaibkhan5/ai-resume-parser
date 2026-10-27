"""Settings must resolve deterministic, absolute, environment-independent paths."""

from __future__ import annotations

from pathlib import Path

from utils.config import BASE_DIR, Settings, settings


def test_base_dir_points_at_repo_root() -> None:
    assert (BASE_DIR / "app.py").is_file()
    assert (BASE_DIR / "assets").is_dir()


def test_model_dir_mapping_is_flat_and_absolute() -> None:
    target = settings.resolve_model_dir("dslim/bert-base-NER")
    assert target.name == "dslim__bert-base-NER"
    assert target.parent == settings.model_cache_dir
    assert target.is_absolute()


def test_limits_are_positive() -> None:
    assert settings.max_upload_mb > 0
    assert settings.max_upload_bytes == settings.max_upload_mb * 1024 * 1024
    assert settings.max_pdf_pages >= 1
    assert settings.max_resume_chars >= 1_000


def test_match_weights_sum_to_one() -> None:
    assert abs(settings.weights_total - 1.0) < 1e-6


def test_settings_is_immutable() -> None:
    fresh = Settings()
    try:
        fresh.app_name = "mutated"  # type: ignore[misc]
    except Exception:  # frozen dataclass raises FrozenInstanceError
        return
    raise AssertionError("Settings should be frozen")


def test_summary_bullet_count_is_sane() -> None:
    assert 1 <= settings.summary_bullets <= 6
    assert Path(settings.log_dir).name == "logs"
