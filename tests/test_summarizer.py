"""Summarisation tests: the extractive path must be as good as the model path.

A resume summariser that only works when a 1.6 GB checkpoint is present is not
shippable, so the offline/degraded behaviour is tested first-class here.
"""

from __future__ import annotations

import pytest

from models.summarizer import ResumeSummarizer, SummaryResult
from tests.conftest import OfflineRegistry
from utils.config import settings


class AbstractiveRegistry(OfflineRegistry):
    """Stub that pretends BART is installed and returns one fluent sentence."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def get_summarizer(self):
        def run(_text, **_kwargs):
            self.calls += 1
            return [
                {
                    "summary_text": (
                        "Led a four-person team that rebuilt the search ranking "
                        "stack and cut inference cost by 38 percent."
                    )
                }
            ]

        return run


class BrokenSummarizerRegistry(OfflineRegistry):
    def get_summarizer(self):
        def run(_text, **_kwargs):
            raise RuntimeError("CUDA out of memory")

        return run


def test_returns_exactly_three_bullets(resume_text: str, stub_registry) -> None:
    result = ResumeSummarizer(stub_registry).summarize(resume_text, bullets=3)
    assert len(result.bullets) == 3
    assert all(bullet.strip() for bullet in result.bullets)


def test_bullets_are_diverse(resume_text: str, stub_registry) -> None:
    result = ResumeSummarizer(stub_registry).summarize(resume_text, bullets=3)
    lowered = [bullet.lower() for bullet in result.bullets]
    assert len(lowered) == len(set(lowered))


def test_bullets_are_full_sentences(resume_text: str, stub_registry) -> None:
    result = ResumeSummarizer(stub_registry).summarize(resume_text, bullets=3)
    for bullet in result.bullets:
        assert len(bullet.split()) >= 5, bullet


def test_contact_details_never_become_bullets(
    resume_text: str, stub_registry
) -> None:
    result = ResumeSummarizer(stub_registry).summarize(resume_text, bullets=3)
    for bullet in result.bullets:
        assert "@" not in bullet
        assert "http" not in bullet
        assert "linkedin.com" not in bullet.lower()
        assert "github.com" not in bullet.lower()


def test_degraded_result_is_labelled_honestly(
    resume_text: str, stub_registry
) -> None:
    result = ResumeSummarizer(stub_registry).summarize(resume_text)
    assert result.degraded is True
    assert result.strategy == "extractive"
    assert result.chunks == 0
    assert result.model == settings.summarizer_model_id
    assert result.latency_ms >= 0


def test_abstractive_path_is_used_when_the_model_loads(
    resume_text: str,
) -> None:
    registry = AbstractiveRegistry()
    result = ResumeSummarizer(registry).summarize(resume_text, bullets=3)

    assert registry.calls >= 1
    assert result.degraded is False
    assert result.strategy == "abstractive+extractive"
    assert result.chunks >= 1
    assert len(result.bullets) == 3


def test_model_failure_falls_back_instead_of_raising(resume_text: str) -> None:
    result = ResumeSummarizer(BrokenSummarizerRegistry()).summarize(
        resume_text, bullets=3
    )
    assert result.degraded is True
    assert len(result.bullets) == 3


class TestBulletCount:
    def test_two_bullets_means_two(self, resume_text: str, stub_registry) -> None:
        result = ResumeSummarizer(stub_registry).summarize(resume_text, bullets=2)
        assert len(result.bullets) == 2

    def test_zero_is_clamped_to_one(self, resume_text: str, stub_registry) -> None:
        result = ResumeSummarizer(stub_registry).summarize(resume_text, bullets=0)
        assert len(result.bullets) == 1

    def test_absurd_counts_are_clamped_to_six(
        self, resume_text: str, stub_registry
    ) -> None:
        result = ResumeSummarizer(stub_registry).summarize(resume_text, bullets=99)
        assert len(result.bullets) == 6


class TestFallbacks:
    def test_uses_parsed_context_when_the_text_is_too_short(
        self, stub_registry
    ) -> None:
        result = ResumeSummarizer(stub_registry).summarize(
            "Short doc.",
            bullets=3,
            context={
                "headline": "ML Engineer",
                "years": 8,
                "skills": ["Python", "SQL"],
                "education": "Master's",
            },
        )
        assert len(result.bullets) == 3
        assert "ML Engineer" in result.bullets[0]
        assert "8+ years" in result.bullets[0]
        assert any("Python" in bullet for bullet in result.bullets)

    def test_never_returns_an_empty_summary(self, stub_registry) -> None:
        result = ResumeSummarizer(stub_registry).summarize("ok")
        assert result.bullets
        assert all(bullet.strip() for bullet in result.bullets)

    @pytest.mark.parametrize("text", ["", "   ", "\n\n"])
    def test_empty_input_is_rejected(self, text: str, stub_registry) -> None:
        with pytest.raises(ValueError):
            ResumeSummarizer(stub_registry).summarize(text)


def test_to_dict_shape(resume_text: str, stub_registry) -> None:
    payload = ResumeSummarizer(stub_registry).summarize(resume_text).to_dict()
    assert set(payload) == {
        "bullets",
        "model",
        "strategy",
        "degraded",
        "latency_ms",
        "chunks",
    }
    assert isinstance(payload["bullets"], list)


def test_singleton_helper_accepts_a_registry(stub_registry) -> None:
    from models.summarizer import get_summarizer

    assert get_summarizer(stub_registry).registry is stub_registry


def test_result_defaults_are_isolated() -> None:
    first = SummaryResult()
    second = SummaryResult()
    first.bullets.append("mutated")
    assert second.bullets == []
