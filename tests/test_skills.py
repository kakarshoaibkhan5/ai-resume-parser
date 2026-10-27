"""Skill taxonomy: detection, alias canonicalisation and JD weighting."""

from __future__ import annotations

from utils.skills import (
    canonicalize,
    extract_jd_skills,
    extract_unknown_skills,
    find_skills,
    skill_names,
)


def test_find_skills_detects_canonical_names() -> None:
    text = "Built services in Python on Kubernetes with AWS and PyTorch."
    names = {hit.name for hit in find_skills(text)}
    assert {"Python", "Kubernetes", "AWS", "PyTorch"} <= names


def test_find_skills_flags_the_skills_section_with_full_confidence() -> None:
    text = "Python and SQL"
    section = "Languages: Python, SQL"
    hits = {hit.name: hit for hit in find_skills(text, skills_section=section)}
    assert hits["Python"].in_skills_section is True
    assert hits["Python"].confidence == 1.0


def test_find_skills_matches_common_aliases() -> None:
    names = {hit.name for hit in find_skills("Deployed on k8s and amazon web services")}
    assert {"Kubernetes", "AWS"} <= names


def test_find_skills_is_case_sensitive_for_ambiguous_tokens() -> None:
    # "go" is a normal English verb; only Golang / capitalised Go are skills.
    assert "Go" not in {hit.name for hit in find_skills("We go to market next week.")}
    assert "Go" in {hit.name for hit in find_skills("Golang services and Go developer.")}


def test_find_skills_returns_each_skill_once() -> None:
    names = [hit.name for hit in find_skills("Python, python, PYTHON and Python3")]
    assert names.count("Python") == 1


def test_find_skills_handles_empty_input() -> None:
    assert find_skills("") == []


def test_extract_unknown_skills_harvests_niche_tools_only_from_the_section() -> None:
    section = (
        "Languages: Python, SQL\n"
        "Frameworks: FastAPI, Qwik, HTMX\n"
        "Cloud: AWS, Fly.io"
    )
    known = [hit.name for hit in find_skills(section)]
    extras = extract_unknown_skills(section, known=known)
    assert set(extras) == {"Qwik", "HTMX", "Fly.io"}
    assert "Python" not in extras
    assert "Frameworks" not in extras


def test_extract_unknown_skills_needs_a_section() -> None:
    assert extract_unknown_skills("") == []
    assert extract_unknown_skills("Python and Qwik in prose", known=["Python"]) == []


def test_extract_jd_skills_marks_required_skills() -> None:
    jd = (
        "Must have strong Python and Kubernetes. "
        "Scala is a nice to have. "
        "AWS required."
    )
    skills = {skill.name: skill for skill in extract_jd_skills(jd)}
    assert skills["Python"].required is True
    assert skills["Kubernetes"].required is True
    assert skills["AWS"].required is True
    assert skills["Scala"].required is False
    assert skills["Scala"].weight < skills["Python"].weight


def test_extract_jd_skills_weights_stay_in_range() -> None:
    jd = "Python Python Python Python. Must have AWS. Scala preferred."
    for skill in extract_jd_skills(jd):
        assert 0 < skill.weight <= 3.0
        assert skill.occurrences >= 1


def test_extract_jd_skills_handles_empty_text() -> None:
    assert extract_jd_skills("") == []


def test_canonicalize_resolves_aliases() -> None:
    assert canonicalize("amazon web services") == "AWS"
    assert canonicalize("k8s") == "Kubernetes"
    assert canonicalize("Golang") == "Go"
    # Unknown values are passed through so user-entered skills keep their label.
    assert canonicalize("something-not-in-the-taxonomy") == "something-not-in-the-taxonomy"
    assert canonicalize("") == ""


def test_skill_names_accepts_strings_dicts_and_objects() -> None:
    class Hit:
        name = "Docker"

    names = skill_names([{"name": "Python"}, "SQL", Hit(), {"category": "x"}])
    assert names == ["Python", "SQL", "Docker"]
