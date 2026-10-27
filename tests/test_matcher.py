"""Job-description matching: explainable, deterministic and safe to degrade.

Weights are only meaningful if the components they describe can be inspected,
so most assertions here target ``breakdown`` and the re-normalisation rules
rather than the headline number alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.matcher import MatchResult, match_resume_to_jd
from tests.conftest import OfflineRegistry
from utils.exceptions import MatchingError


def _match(extraction, jd, resume_text="", registry=None) -> MatchResult:
    return match_resume_to_jd(
        extraction,
        jd,
        resume_text=resume_text,
        registry=registry or OfflineRegistry(),
    )


def test_strong_resume_scores_high(
    parsed_resume, job_description: str, resume_text: str
) -> None:
    result = _match(parsed_resume.to_dict(), job_description, resume_text)
    assert result.score >= 80
    assert result.verdict == "Excellent match"
    assert result.tone == "excellent"


def test_coverage_counts_weighted_jd_skills(
    parsed_resume, job_description: str, resume_text: str
) -> None:
    result = _match(parsed_resume.to_dict(), job_description, resume_text)
    assert 0.6 <= result.coverage <= 0.9
    matched = {skill["name"] for skill in result.matched_skills}
    assert {"Python", "SQL", "PyTorch", "Kubernetes", "AWS"} <= matched


def test_missing_skills_are_reported_and_required_first(
    parsed_resume, job_description: str, resume_text: str
) -> None:
    result = _match(parsed_resume.to_dict(), job_description, resume_text)
    missing = {skill["name"]: skill for skill in result.missing_skills}
    assert {"TensorFlow", "Scala"} <= set(missing)
    assert missing["TensorFlow"]["required"] is True
    assert missing["Scala"]["required"] is False
    assert result.missing_skills[0]["required"] is True


def test_breakdown_is_explainable(
    parsed_resume, job_description: str, resume_text: str
) -> None:
    result = _match(parsed_resume.to_dict(), job_description, resume_text)
    components = {item["key"]: item for item in result.breakdown}

    assert set(components) == {"skills", "semantic", "experience", "education"}
    assert all(item["applicable"] for item in components.values())
    assert abs(sum(item["weight"] for item in components.values()) - 100.0) < 0.2
    assert components["skills"]["score"] is not None
    assert "JD skills found" in components["skills"]["detail"]


def test_experience_and_education_components_match_the_fixture(
    parsed_resume, job_description: str, resume_text: str
) -> None:
    result = _match(parsed_resume.to_dict(), job_description, resume_text)

    assert result.experience["required_years"] == 5.0
    assert result.experience["resume_years"] >= 5.0
    assert result.experience["gap_years"] == 0.0

    assert result.education["required_level"] == "Bachelor's"
    assert result.education["resume_level"] == "Master's"


def test_weights_used_are_reported(
    parsed_resume, job_description: str, resume_text: str
) -> None:
    result = _match(parsed_resume.to_dict(), job_description, resume_text)
    assert abs(sum(result.weights_used.values()) - 100.0) < 0.2
    assert set(result.weights_used) == {"skills", "semantic", "experience", "education"}


def test_irrelevant_resume_scores_low(
    parsed_resume, nursing_jd: str, resume_text: str
) -> None:
    result = _match(parsed_resume.to_dict(), nursing_jd, resume_text)
    assert result.score < 35
    assert result.verdict == "Low match"
    assert any(skill["name"] == "Communication" for skill in result.missing_skills)


def test_recommendations_are_actionable(
    parsed_resume, job_description: str, resume_text: str
) -> None:
    result = _match(parsed_resume.to_dict(), job_description, resume_text)
    assert result.recommendations
    assert any("TensorFlow" in tip for tip in result.recommendations)


def test_perfect_alignment_gets_a_tailoring_tip() -> None:
    extraction = {
        "skills": [{"name": "Python"}, {"name": "SQL"}],
        "experience": {"years": 10},
        "education": {"highest_level_rank": 5},
    }
    jd = "Python and SQL required. PhD preferred. 3+ years of experience."
    result = _match(extraction, jd, "python sql")
    assert result.score >= 80
    assert result.recommendations
    assert "tailor" in result.recommendations[0].lower()


class TestApplicability:
    def test_education_is_dropped_when_the_jd_has_no_degree(self) -> None:
        extraction = {"skills": [{"name": "Python"}], "experience": {"years": 5}}
        jd = "We need Python engineers to build data pipelines for our platform."
        result = _match(extraction, jd, "python developer")
        education = {item["key"]: item for item in result.breakdown}["education"]
        assert education["applicable"] is False
        assert education["weight"] == 0.0
        assert "education" not in result.weights_used

    def test_experience_is_dropped_when_the_jd_has_no_years(self) -> None:
        extraction = {"skills": [{"name": "Python"}], "experience": {"years": 5}}
        jd = "Python developer needed for a growing analytics team in London."
        result = _match(extraction, jd, "python developer")
        experience = {item["key"]: item for item in result.breakdown}["experience"]
        assert experience["applicable"] is False
        assert "experience" not in result.weights_used

    def test_skills_coverage_is_dropped_when_no_skills_are_detected(self) -> None:
        extraction = {"skills": [], "experience": {"years": 2}}
        jd = (
            "We are looking for a motivated teammate with excellent written and "
            "verbal abilities to join our operations team."
        )
        result = _match(extraction, jd, "operations generalist")
        skills = {item["key"]: item for item in result.breakdown}["skills"]
        assert skills["applicable"] is False
        assert "skills" not in result.weights_used


class TestInputHandling:
    def test_short_jd_is_rejected(self, parsed_resume) -> None:
        with pytest.raises(MatchingError):
            _match(parsed_resume.to_dict(), "Python dev")

    def test_whitespace_only_jd_is_rejected(self, parsed_resume) -> None:
        with pytest.raises(MatchingError):
            _match(parsed_resume.to_dict(), "   \n  ")

    def test_empty_extraction_does_not_crash(self) -> None:
        jd = "Must have strong Python and Kubernetes experience for this role."
        result = _match({}, jd, "some resume text")
        assert 0 <= result.score <= 100
        assert result.missing_skills

    def test_very_long_jd_is_bounded(self, parsed_resume) -> None:
        jd = ("Python Kubernetes AWS. " * 4_000).strip()
        result = _match(parsed_resume.to_dict(), jd, "python kubernetes aws")
        assert 0 <= result.score <= 100

    def test_result_is_reproducible(
        self, parsed_resume, job_description: str, resume_text: str
    ) -> None:
        first = _match(parsed_resume.to_dict(), job_description, resume_text)
        second = _match(parsed_resume.to_dict(), job_description, resume_text)
        assert first.score == second.score
        assert first.breakdown == second.breakdown

    def test_to_dict_reports_coverage_as_a_percentage(
        self, parsed_resume, job_description: str, resume_text: str
    ) -> None:
        payload = _match(
            parsed_resume.to_dict(), job_description, resume_text
        ).to_dict()
        assert 0 <= payload["coverage"] <= 100
        assert payload["coverage"] == round(payload["coverage"], 1)
        assert set(payload["matched_skills"][0]) == {
            "name",
            "category",
            "weight",
            "required",
            "occurrences",
        }


class TestSemanticPath:
    def test_embeddings_are_used_when_available(self) -> None:
        class EncoderRegistry(OfflineRegistry):
            def embed_texts(self, texts, **_kwargs):
                return np.array([[1.0, 0.0], [1.0, 0.0]])

        extraction = {
            "skills": [{"name": "Python"}],
            "experience": {"years": 5},
        }
        jd = "Python engineer required with five years of professional experience."
        result = _match(extraction, jd, "python engineer", EncoderRegistry())

        assert result.semantic_method == "bert-mean-pooling"
        assert result.semantic_similarity == pytest.approx(1.0)

    def test_lexical_fallback_is_used_offline(
        self, parsed_resume, job_description: str, resume_text: str
    ) -> None:
        result = _match(parsed_resume.to_dict(), job_description, resume_text)
        assert result.semantic_method == "lexical-tf"
        assert 0.0 <= result.semantic_similarity <= 1.0

    def test_no_resume_text_gives_zero_semantic_similarity(self) -> None:
        extraction = {"skills": [{"name": "Python"}]}
        jd = "Python developer required to maintain our internal analytics tooling."
        result = _match(extraction, jd)
        assert result.semantic_similarity == 0.0
