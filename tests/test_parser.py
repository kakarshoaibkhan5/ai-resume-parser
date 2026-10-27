"""End-to-end extraction tests using the deterministic rules path.

The parser must produce the same structured record on a machine with no model
weights as it does with them, because every downstream feature (summary,
matching, export) reads this record.
"""

from __future__ import annotations

import json

import pytest

from parsers.ner_parser import ParsedResume, ResumeParser


def test_extracts_identity(parsed_resume: ParsedResume) -> None:
    assert parsed_resume.name == "Aarav Mehta"
    assert parsed_resume.headline == "Senior Machine Learning Engineer"


def test_extracts_contact_details(parsed_resume: ParsedResume) -> None:
    contact = parsed_resume.contact
    assert contact["email"] == "aarav.mehta@example.com"
    assert contact["phone"] == "+91 98200 41122"
    assert contact["location"] == "Bengaluru, India"
    assert "linkedin.com/in/aaravmehta" in contact["linkedin"]
    assert "github.com/aaravmehta" in contact["github"]
    assert len(contact["links"]) == 2


def test_extracts_skills_with_metadata(parsed_resume: ParsedResume) -> None:
    by_name = {skill["name"]: skill for skill in parsed_resume.skills}
    assert {"Python", "SQL", "PyTorch", "AWS", "Kubernetes"} <= set(by_name)
    # The explicit skills block must be recognised as such.
    assert by_name["Python"]["in_skills_section"] is True
    assert by_name["Python"]["category"] == "Languages"
    assert by_name["Python"]["confidence"] == 1.0
    assert by_name["Python"]["occurrences"] >= 1


def test_skills_are_deduplicated(parsed_resume: ParsedResume) -> None:
    names = [skill["name"] for skill in parsed_resume.skills]
    assert len(names) == len(set(name.lower() for name in names))


def test_extracts_experience_years_companies_and_titles(
    parsed_resume: ParsedResume,
) -> None:
    experience = parsed_resume.experience
    assert experience["years_source"] == "dates"
    # Jul 2017 -> Present; keep the bound loose so the test does not rot.
    assert 8.0 <= experience["years"] <= 15.0
    assert experience["companies"] == [
        "Nimbus Analytics",
        "Vertex Labs",
        "Bluewave Systems",
    ]
    assert experience["titles"][0] == "Senior Machine Learning Engineer"
    assert len(experience["entries"]) == 3


def test_experience_entries_carry_dates_and_descriptions(
    parsed_resume: ParsedResume,
) -> None:
    entry = parsed_resume.experience["entries"][0]
    assert entry["company"] == "Nimbus Analytics"
    assert entry["start"] == "Jan 2022"
    assert entry["end"] == "Present"
    assert entry["duration_months"] > 0
    assert "ranking pipeline" in entry["description"]


def test_company_names_never_pick_up_the_job_title(
    parsed_resume: ParsedResume,
) -> None:
    assert "Data Scientist" not in parsed_resume.experience["companies"]
    assert "Senior Machine Learning Engineer" not in parsed_resume.experience["companies"]


def test_extracts_education(parsed_resume: ParsedResume) -> None:
    education = parsed_resume.education
    assert education["highest_level"] == "Master's"
    assert education["highest_level_rank"] == 4
    assert education["institutions"] == [
        "Indian Institute of Technology, Bombay",
        "VIT University",
    ]

    degrees = {degree["degree"]: degree for degree in education["degrees"]}
    assert degrees["Master's"]["field"] == "Computer Science"
    assert degrees["Master's"]["year"] == 2017
    assert degrees["Bachelor's"]["field"] == "Information Technology"
    assert degrees["Bachelor's"]["year"] == 2015


def test_extracts_certifications_and_projects(parsed_resume: ParsedResume) -> None:
    assert parsed_resume.certifications == [
        "AWS Certified Machine Learning - Specialty (2023)"
    ]
    assert parsed_resume.projects == []


def test_confidence_block_is_bounded(parsed_resume: ParsedResume) -> None:
    confidence = parsed_resume.confidence
    assert set(confidence) == {
        "name",
        "email",
        "phone",
        "skills",
        "experience",
        "education",
        "overall",
    }
    assert all(0.0 <= value <= 1.0 for value in confidence.values())
    assert confidence["overall"] >= 0.8


def test_raw_text_is_opt_in(parsed_resume: ParsedResume) -> None:
    assert "raw_text" not in parsed_resume.to_dict()
    assert parsed_resume.to_dict(include_raw=True)["raw_text"].startswith("Aarav Mehta")


def test_to_dict_is_json_serialisable(parsed_resume: ParsedResume) -> None:
    payload = json.dumps(parsed_resume.to_dict())
    assert json.loads(payload)["name"] == "Aarav Mehta"


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t "])
def test_empty_resume_is_rejected(text: str, stub_registry) -> None:
    parser = ResumeParser(stub_registry, use_ner=False)
    with pytest.raises(ValueError):
        parser.parse(text)


def test_unstructured_text_degrades_without_crashing(stub_registry) -> None:
    parser = ResumeParser(stub_registry, use_ner=False)
    result = parser.parse("just some random text without structure")
    assert result.name == ""
    assert result.skills == []
    assert result.experience["years"] == 0.0
    assert result.education["highest_level"] == "Not detected"


def test_ner_is_skipped_entirely_when_disabled() -> None:
    class NoModels:
        inference_lock = None

        def get_ner(self):  # pragma: no cover - must never run
            raise AssertionError("NER must not be loaded when use_ner=False")

    parser = ResumeParser(NoModels(), use_ner=False)
    assert parser.parse("Aarav Mehta\nML Engineer").name == "Aarav Mehta"


def test_missing_ner_model_yields_no_entities(stub_registry) -> None:
    parser = ResumeParser(stub_registry, use_ner=True)
    result = parser.parse("Aarav Mehta\nMachine Learning Engineer")
    assert result.ner_entities == []


def test_ner_entities_are_normalised_and_filtered(stub_registry) -> None:
    def pipeline(_text):
        return [
            {"word": "Aarav", "entity_group": "PER", "score": 0.99},
            {"word": "Mehta", "entity_group": "PER", "score": 0.97},
            {"word": "Nimbus", "entity_group": "ORG", "score": 0.91},
            {"word": "Bengaluru", "entity_group": "LOC", "score": 0.88},
            {"word": "noise", "entity_group": "MISC", "score": 0.10},
        ]

    parser = ResumeParser(stub_registry, ner_pipeline=pipeline, use_ner=True)
    result = parser.parse("Aarav Mehta\nNimbus Analytics, Bengaluru")

    labels = {entity["label"] for entity in result.ner_entities}
    assert labels == {"PERSON", "ORG", "LOCATION"}
    assert all(entity["score"] >= 0.55 for entity in result.ner_entities)
    # Highest confidence first.
    assert result.ner_entities[0]["text"] == "Aarav"


def test_split_wordpiece_entities_are_reassembled_from_offsets(stub_registry) -> None:
    """A word broken across two predictions must not leak half-tokens."""

    def pipeline(text: str):
        assert text.startswith("Skills:")
        return [
            {"word": "P", "entity_group": "ORG", "score": 0.95, "start": 8, "end": 9},
            {"word": "##yTorch", "entity_group": "ORG", "score": 0.88, "start": 9, "end": 15},
            # Lowercase tail of a chopped word: noise, not an entity.
            {"word": "net", "entity_group": "ORG", "score": 0.87, "start": 20, "end": 23},
            {"word": "A", "entity_group": "PER", "score": 0.99, "start": 30, "end": 31},
        ]

    parser = ResumeParser(stub_registry, ner_pipeline=pipeline, use_ner=True)
    entities = parser.run_ner("Skills: PyTorch, Kubernetes, Aarav Mehta")
    text_by_label = {(entity["text"], entity["label"]) for entity in entities}

    assert ("PyTorch", "ORG") in text_by_label
    assert ("net", "ORG") not in text_by_label
    assert all(entity["text"] != "A" for entity in entities)


def test_company_entities_skip_tools_and_single_word_noise(stub_registry) -> None:
    source = (
        "Senior ML Engineer | Nimbus Analytics | Jan 2022 - Present\n"
        "Built ranking services with PyTorch, Airflow and Terraform.\n"
        "Data Scientist | Vertex Labs | 2017 - 2019"
    )
    entities = [
        {"text": "Nimbus Analytics", "label": "ORG", "score": 0.99},
        {"text": "PyTorch", "label": "ORG", "score": 0.97},
        {"text": "Airflow", "label": "ORG", "score": 0.90},
        {"text": "Terraform", "label": "ORG", "score": 0.88},
        {"text": "Vertex Labs", "label": "ORG", "score": 0.96},
        {"text": "Table", "label": "ORG", "score": 0.62},
        {"text": "Not In The Resume Ltd", "label": "ORG", "score": 0.99},
        {"text": "Indian Institute of Technology", "label": "ORG", "score": 0.99},
    ]

    companies = ResumeParser._company_entities(entities, source)

    assert companies == ["Nimbus Analytics", "Vertex Labs"]


def test_name_falls_back_when_the_header_is_messy(stub_registry) -> None:
    parser = ResumeParser(stub_registry, use_ner=False)
    text = "Curriculum Vitae\n\nPriya Raman\nSoftware Engineer\npriya@example.com"
    assert parser.parse(text).name == "Priya Raman"


def test_location_is_found_inside_a_pipe_delimited_contact_line(
    stub_registry,
) -> None:
    parser = ResumeParser(stub_registry, use_ner=False)
    text = (
        "Rahul Verma\n"
        "Backend Developer\n"
        "rahul@example.com | +44 7700 900123 | Manchester, UK\n"
    )
    contact = parser.parse(text).contact
    assert contact["location"] == "Manchester, UK"
    assert contact["phone"] == "+44 7700 900123"
