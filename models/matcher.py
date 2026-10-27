"""Job-description ↔ resume matching.

Score = weighted blend of four explainable components:

==================  ======  ==================================================
Component           Weight  How it is measured
==================  ======  ==================================================
Skill coverage       50 %   JD skills weighted by frequency + requirement cues,
                            matched against resume skills (exact + fuzzy)
Semantic similarity  25 %   Cosine similarity of mean-pooled BERT embeddings
                            (falls back to lexical TF cosine)
Experience           15 %   Resume years vs. years demanded by the JD
Education            10 %   Highest degree vs. degree demanded by the JD
==================  ======  ==================================================

Components the JD does not ask about are dropped and the remaining weights are
re-normalised, so a JD without a degree requirement cannot drag a good
candidate down.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from models.registry import ModelRegistry, get_registry
from utils.config import settings
from utils.exceptions import MatchingError
from utils.logger import get_logger
from utils.skills import (
    WeightedSkill,
    extract_jd_skills,
    skill_names,
)

logger = get_logger("models.matcher")

try:  # rapidfuzz is a hard dependency, but never let it break scoring
    from rapidfuzz import fuzz as _fuzz

    def _ratio(a: str, b: str) -> float:
        return float(_fuzz.token_set_ratio(a, b))
except Exception:  # pragma: no cover
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100

_MIN_JD_CHARS = 40
_MAX_JD_CHARS = 30_000
_FUZZY_THRESHOLD = 90.0

_EXPERIENCE_REQ_RE = re.compile(
    r"(\d{1,2})(?:\s*(?:-|–|to)\s*(\d{1,2}))?\s*\+?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)
_EDUCATION_PATTERNS: tuple[tuple[re.Pattern[str], int, str], ...] = (
    (re.compile(r"\b(ph\.?d|doctorate|doctoral)\b", re.I), 5, "PhD"),
    (re.compile(r"\b(master'?s?|mba|m\.?s\.?c?|m\.?tech|m\.?eng)\b", re.I), 4, "Master's"),
    (re.compile(r"\b(bachelor'?s?|b\.?tech|b\.?sc|b\.?s\.?|b\.?a\.?|b\.?e\.?|b\.?eng|undergraduate)\b", re.I), 3, "Bachelor's"),
    (re.compile(r"\b(associate|diploma|foundation degree)\b", re.I), 2, "Associate"),
    (re.compile(r"\b(high school|secondary school)\b", re.I), 1, "High school"),
)
_DEGREE_LEVEL_NAMES = {
    5: "PhD",
    4: "Master's",
    3: "Bachelor's",
    2: "Associate",
    1: "High school",
    0: "Not detected",
}


@dataclass
class MatchResult:
    score: int
    verdict: str
    tone: str
    coverage: float
    semantic_similarity: float
    semantic_method: str
    breakdown: list[dict] = field(default_factory=list)
    matched_skills: list[dict] = field(default_factory=list)
    missing_skills: list[dict] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    experience: dict = field(default_factory=dict)
    education: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    weights_used: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "coverage": round(self.coverage * 100, 1),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "semantic_method": self.semantic_method,
            "breakdown": self.breakdown,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "extra_skills": self.extra_skills,
            "experience": self.experience,
            "education": self.education,
            "recommendations": self.recommendations,
            "weights_used": self.weights_used,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_resume_to_jd(
    extraction: dict,
    jd_text: str,
    *,
    resume_text: str = "",
    registry: ModelRegistry | None = None,
) -> MatchResult:
    """Score a parsed resume against a job description."""
    jd = (jd_text or "").strip()
    if len(jd) < _MIN_JD_CHARS:
        raise MatchingError(
            "The job description is too short to score.",
            hint=f"Paste at least {_MIN_JD_CHARS} characters of the JD.",
        )
    jd = jd[:_MAX_JD_CHARS]

    resume_skills = skill_names(extraction.get("skills", []))
    jd_skills = extract_jd_skills(jd)
    matched, missing, coverage = _match_skills(jd_skills, resume_skills)

    resume_text = resume_text or extraction.get("_raw_text", "")
    semantic, method = _semantic_similarity(registry or get_registry(), resume_text, jd)

    experience_score, experience_detail = _experience_fit(
        extraction.get("experience", {}), jd
    )
    education_score, education_detail = _education_fit(
        extraction.get("education", {}), jd
    )

    components: list[tuple[str, str, float | None, float, str]] = [
        ("skills", "Skill coverage", coverage if jd_skills else None, settings.weight_skills,
         f"{len(matched)}/{len(jd_skills)} JD skills found" if jd_skills else "No skills detected in the JD"),
        ("semantic", "Semantic similarity", semantic, settings.weight_semantic,
         f"{method} · raw {semantic:.0%}"),
        ("experience", "Experience", experience_score, settings.weight_experience,
         experience_detail["detail"]),
        ("education", "Education", education_score, settings.weight_education,
         education_detail["detail"]),
    ]

    applicable = [c for c in components if c[2] is not None]
    total_weight = sum(c[3] for c in applicable) or 1.0
    raw = sum(c[2] * c[3] for c in applicable) / total_weight  # type: ignore[operator]
    score = int(round(max(0.0, min(1.0, raw)) * 100))
    verdict, tone = _verdict(score)

    breakdown = [
        {
            "key": key,
            "label": label,
            "score": None if value is None else round(value * 100, 1),
            "weight": round((weight / total_weight) * 100, 1) if value is not None else 0.0,
            "detail": detail,
            "applicable": value is not None,
        }
        for key, label, value, weight, detail in components
    ]

    recommendations = _recommendations(
        score=score,
        missing=missing,
        experience=experience_detail,
        education=education_detail,
        semantic=semantic,
        jd_skills_count=len(jd_skills),
    )

    jd_names = {skill.name.lower() for skill in jd_skills}
    extra = [name for name in resume_skills if name.lower() not in jd_names][:15]

    return MatchResult(
        score=score,
        verdict=verdict,
        tone=tone,
        coverage=coverage,
        semantic_similarity=semantic,
        semantic_method=method,
        breakdown=breakdown,
        matched_skills=[_skill_dict(skill) for skill in matched],
        missing_skills=[_skill_dict(skill) for skill in missing[:20]],
        extra_skills=extra,
        experience=experience_detail,
        education=education_detail,
        recommendations=recommendations,
        weights_used={c[0]: round(c[3] / total_weight * 100, 1) for c in applicable},
    )


# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------

def _skill_dict(skill: WeightedSkill) -> dict:
    return {
        "name": skill.name,
        "category": skill.category,
        "weight": skill.weight,
        "required": skill.required,
        "occurrences": skill.occurrences,
    }


def _match_skills(
    jd_skills: list[WeightedSkill], resume_skills: list[str]
) -> tuple[list[WeightedSkill], list[WeightedSkill], float]:
    if not jd_skills:
        return [], [], 0.0

    resume_lower = {name.lower(): name for name in resume_skills}
    matched: list[WeightedSkill] = []
    missing: list[WeightedSkill] = []

    for skill in jd_skills:
        if skill.name.lower() in resume_lower:
            matched.append(skill)
            continue
        if any(_ratio(skill.name, candidate) >= _FUZZY_THRESHOLD for candidate in resume_skills):
            matched.append(skill)
        else:
            missing.append(skill)

    total = sum(skill.weight for skill in jd_skills) or 1.0
    coverage = sum(skill.weight for skill in matched) / total
    missing.sort(key=lambda item: (not item.required, -item.weight))
    return matched, missing, coverage


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    numerator = sum(a[word] * b[word] for word in common)
    denom = (sum(v * v for v in a.values()) ** 0.5) * (sum(v * v for v in b.values()) ** 0.5)
    return numerator / denom if denom else 0.0


def _lexical_cosine(resume_text: str, jd_text: str) -> float:
    from utils.text_utils import content_tokens

    if not resume_text or not jd_text:
        return 0.0
    resume_counts = Counter(content_tokens(resume_text))
    jd_counts = Counter(content_tokens(jd_text))
    return _cosine(resume_counts, jd_counts)


def _semantic_similarity(
    registry: ModelRegistry, resume_text: str, jd_text: str
) -> tuple[float, str]:
    if resume_text.strip():
        embeddings = registry.embed_texts([resume_text[:6000], jd_text[:6000]])
        if embeddings is not None and len(embeddings) == 2:
            cosine = float(np.dot(embeddings[0], embeddings[1]))
            # BERT mean-pooled cosines cluster high; calibrate to a usable range.
            calibrated = (cosine - 0.60) / 0.32
            return float(np.clip(calibrated, 0.0, 1.0)), "bert-mean-pooling"

    cosine = _lexical_cosine(resume_text, jd_text)
    calibrated = (cosine - 0.10) / 0.45
    return float(np.clip(calibrated, 0.0, 1.0)), "lexical-tf"


# ---------------------------------------------------------------------------
# Experience & education
# ---------------------------------------------------------------------------

def _experience_fit(experience: dict, jd_text: str) -> tuple[float | None, dict]:
    required = 0.0
    for match in _EXPERIENCE_REQ_RE.finditer(jd_text):
        window = jd_text[max(0, match.start() - 80): match.end() + 80].lower()
        if "experience" not in window:
            continue
        values = [float(group) for group in match.groups() if group]
        if values:
            required = max(required, max(values))
    required = min(required, 40.0)

    resume_years = float(experience.get("years") or 0.0)
    detail = {
        "required_years": required or None,
        "resume_years": resume_years,
        "gap_years": round(max(0.0, required - resume_years), 1) if required else 0.0,
    }

    if not required:
        detail["detail"] = f"{resume_years:g} yrs experience · no explicit requirement"
        detail["applicable"] = False
        return None, detail

    if resume_years >= required:
        score = 1.0
        detail["detail"] = f"{resume_years:g} yrs vs {required:g} yrs required"
    elif resume_years >= required * 0.8:
        score = 0.85
        detail["detail"] = f"Close: {resume_years:g} yrs vs {required:g} yrs required"
    else:
        score = max(0.0, min(1.0, resume_years / required))
        detail["detail"] = f"{resume_years:g} yrs vs {required:g} yrs required"
    detail["applicable"] = True
    return score, detail


def _education_fit(education: dict, jd_text: str) -> tuple[float | None, dict]:
    required_rank = 0
    required_label = ""
    for pattern, rank, label in _EDUCATION_PATTERNS:
        if pattern.search(jd_text) and rank > required_rank:
            required_rank, required_label = rank, label

    resume_rank = int(education.get("highest_level_rank") or 0)
    detail = {
        "required_level": required_label or None,
        "resume_level": _DEGREE_LEVEL_NAMES.get(resume_rank, "Not detected"),
        "resume_level_rank": resume_rank,
        "required_level_rank": required_rank or None,
    }

    if not required_rank:
        detail["detail"] = f"{detail['resume_level']} · no explicit requirement"
        detail["applicable"] = False
        return None, detail

    if resume_rank >= required_rank:
        score = 1.0
        detail["detail"] = f"{detail['resume_level']} meets {required_label}"
    elif resume_rank == 0:
        score = 0.25
        detail["detail"] = f"No degree detected; JD asks for {required_label}"
    else:
        score = max(0.25, 1.0 - 0.3 * (required_rank - resume_rank))
        detail["detail"] = f"{detail['resume_level']} vs {required_label} required"
    detail["applicable"] = True
    return score, detail


# ---------------------------------------------------------------------------
# Verdict & recommendations
# ---------------------------------------------------------------------------

def _verdict(score: int) -> tuple[str, str]:
    if score >= 80:
        return "Excellent match", "excellent"
    if score >= 65:
        return "Strong match", "strong"
    if score >= 50:
        return "Moderate match", "moderate"
    if score >= 35:
        return "Partial match", "partial"
    return "Low match", "low"


def _recommendations(
    *,
    score: int,
    missing: list[WeightedSkill],
    experience: dict,
    education: dict,
    semantic: float,
    jd_skills_count: int,
) -> list[str]:
    tips: list[str] = []
    required_missing = [skill for skill in missing if skill.required][:3]
    other_missing = [skill for skill in missing if not skill.required][:2]

    for skill in required_missing:
        tips.append(
            f"Add concrete evidence of {skill.name} — the JD marks it as required."
        )
    for skill in other_missing:
        tips.append(f"Consider highlighting {skill.name} if you have exposure to it.")

    if experience.get("applicable") and experience.get("gap_years", 0) > 0:
        tips.append(
            f"The role asks for {experience['required_years']:g} years; your resume shows "
            f"{experience['resume_years']:g}. Surface earlier or freelance work if relevant."
        )

    if education.get("applicable") and education.get("resume_level_rank", 0) < education.get("required_level_rank", 0):
        tips.append(
            f"The JD expects {education['required_level']}; add your degree, bootcamp or "
            "equivalent certification details."
        )

    if semantic < 0.35:
        tips.append(
            "Mirror the JD's core vocabulary (role title, domain terms) in your summary — "
            "keyword alignment is still a major ATS signal."
        )

    if not tips:
        tips.append(
            "Strong alignment. Tailor the summary to the company and quantify two or three "
            "results with numbers to stand out."
        )
    if jd_skills_count == 0:
        tips.append(
            "No hard skills were detected in the JD — add an explicit requirements section "
            "for a more precise score."
        )
    return tips[:6]
