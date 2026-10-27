"""Hybrid resume parser: rule-based structure + BERT NER.

A general NER model alone cannot parse a resume (no sections, no date maths,
no skill taxonomy), so this module combines:

* deterministic section segmentation (Experience / Education / Skills …)
* regex extraction for email, phone, links, dates and degrees
* ``dslim/bert-base-NER`` for person, organisation and location hints
* the curated skill taxonomy in :mod:`utils.skills`

Every field degrades gracefully: with no model at all the parser still returns
a useful structured record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from models.registry import ModelRegistry, get_registry
from utils.config import settings
from utils.logger import get_logger
from utils.skills import SkillHit, alias_index, extract_unknown_skills, find_skills
from utils.text_utils import (
    clean_snippet,
    dedupe,
    extract_date_ranges,
    extract_years_claim,
    months_to_years,
    normalize_text,
    strip_bullet,
    title_case_name,
    total_experience_months,
)

logger = get_logger("parsers.ner_parser")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": (
        "summary", "professional summary", "profile", "professional profile",
        "about me", "about", "objective", "career objective", "career summary",
        "executive summary", "overview", "personal summary",
    ),
    "experience": (
        "experience", "work experience", "professional experience", "employment",
        "employment history", "work history", "career history", "relevant experience",
        "professional background", "experience and achievements", "career",
        "positions held", "professional experience and achievements",
    ),
    "education": (
        "education", "academic background", "academic qualifications",
        "education and training", "qualifications", "academic history",
        "education details", "academic credentials", "educational background",
    ),
    "skills": (
        "skills", "technical skills", "core skills", "key skills",
        "core competencies", "competencies", "technologies", "tech stack",
        "technical expertise", "expertise", "skills and interests",
        "tools and technologies", "technical proficiencies", "areas of expertise",
        "skills and tools", "technical skill set", "proficiencies",
    ),
    "projects": (
        "projects", "personal projects", "selected projects", "portfolio",
        "key projects", "academic projects", "side projects", "project experience",
    ),
    "certifications": (
        "certifications", "certificates", "licenses", "certifications and awards",
        "licenses and certifications", "professional certifications",
        "certifications and training", "courses and certifications",
    ),
    "awards": ("awards", "achievements", "honors", "honours", "awards and honors"),
    "languages": ("languages", "language proficiency", "languages spoken"),
    "publications": ("publications", "research", "research experience", "papers"),
    "volunteer": (
        "volunteer", "volunteering", "community involvement",
        "extracurricular activities", "leadership and activities",
    ),
    "interests": ("interests", "hobbies", "hobbies and interests"),
    "references": ("references",),
}

_SECTION_LOOKUP = {
    alias: key for key, aliases in SECTION_ALIASES.items() for alias in aliases
}

ROLE_KEYWORDS = (
    "engineer", "developer", "manager", "analyst", "designer", "scientist",
    "consultant", "architect", "administrator", "specialist", "lead",
    "director", "officer", "intern", "coordinator", "executive", "founder",
    "president", "researcher", "accountant", "recruiter", "marketer",
    "strategist", "technician", "supervisor", "assistant", "associate",
    "principal", "staff", "head of", "owner", "programmer", "devops",
    "sre", "qa", "tester", "product owner", "scrum master",
)

_ROLE_RE = re.compile(r"\b(" + "|".join(re.escape(r) for r in ROLE_KEYWORDS) + r")\b", re.I)

_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|company|gmbh|plc|pvt|"
    r"technologies|technology|solutions|systems|labs|group|holdings|ventures|"
    r"consulting|services|software|media|bank|capital|partners|associates)\b\.?",
    re.I,
)

_INSTITUTION_RE = re.compile(
    r"\b(university|college|institute|school|academy|polytechnic|iit|nit|iiit|bits)\b",
    re.I,
)

_DEGREES: tuple[tuple[re.Pattern[str], int, str], ...] = (
    (re.compile(r"\b(ph\.?\s?d|doctorate|doctoral)\b", re.I), 5, "PhD"),
    (re.compile(r"\b(master(?:'s)?|m\.?s\.?c|m\.?tech|m\.?eng|mba|m\.?a\.?)\b", re.I), 4, "Master's"),
    (re.compile(r"\b(bachelor(?:'s)?|b\.?tech|b\.?sc|b\.?s\.?|b\.?e\.?|b\.?eng|b\.?com|b\.?a\.?)\b", re.I), 3, "Bachelor's"),
    (re.compile(r"\b(associate(?:'s)?|diploma|advanced diploma)\b", re.I), 2, "Associate"),
    (re.compile(r"\b(high school|higher secondary|senior secondary|12th)\b", re.I), 1, "High school"),
)

_LEVEL_NAMES = {5: "PhD", 4: "Master's", 3: "Bachelor's", 2: "Associate", 1: "High school", 0: "Not detected"}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\w.])(\+?\d[\d\s().\-]{7,22}\d)(?![\w.])")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/(?:in|pub)/[A-Za-z0-9_\-%./]+", re.I)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_.\-]+", re.I)
_WEBSITE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.(?:com|io|dev|me|net|org|ai|co|app|xyz|tech)(?:/[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*)?",
    re.I,
)
_LOCATION_RE = re.compile(
    r"^([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,2}),\s*"
    r"([A-Z]{2}|[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+)?)$"
)

_NAME_STOPWORDS = {
    "resume", "curriculum", "vitae", "cv", "portfolio", "profile", "contact",
    "linkedin", "github", "email", "phone", "mobile", "address", "summary",
} | set(ROLE_KEYWORDS)

_COMPANY_NOISE = {
    "microsoft word", "microsoft excel", "powerpoint", "google docs", "resume",
    "curriculum vitae", "university", "college",
}

_NER_LABELS = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION", "MISC": "MISC"}
_MIN_ENTITY_CHARS = 3


def _entities_from_predictions(
    chunk: str, predictions: Sequence[dict], *, min_score: float = 0.55
) -> list[tuple[str, str, float]]:
    """Rebuild ``(text, label, score)`` triples from a chunk's predictions.

    The pipeline's ``word`` field is not safe to use directly: WordPiece
    continuations keep their ``##`` prefix and word aggregation can split a
    token in two (``PyTorch`` arrives as ``P`` + ``##yTorch``). Slicing the
    chunk with the reported offsets always returns the exact source text, and
    merging contiguous same-label spans puts the pieces back together.

    Pipelines that do not report offsets (custom or mocked ones) fall back to
    joining ``##`` continuations, which is good enough for whole words.
    """
    usable: list[tuple[dict, str, float]] = []
    for prediction in predictions or []:
        raw_label = str(prediction.get("entity_group") or prediction.get("entity") or "")
        label = _NER_LABELS.get(raw_label.replace("B-", "").replace("I-", ""), "")
        try:
            score = float(prediction.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if label and score >= min_score:
            usable.append((prediction, label, score))
    if not usable:
        return []

    spans: list[list] = []  # [text, label, score]
    if any("start" in item and "end" in item for item, _, _ in usable):
        offsets: list[list] = []
        for prediction, label, score in usable:
            try:
                start = int(prediction["start"])
                end = int(prediction["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start:
                continue
            if offsets and offsets[-1][2] == label and offsets[-1][1] >= start:
                offsets[-1][1] = max(offsets[-1][1], end)
                offsets[-1][3] = max(offsets[-1][3], score)
            else:
                offsets.append([start, end, label, score])
        spans = [[chunk[start:end], label, score] for start, end, label, score in offsets]
    else:
        for prediction, label, score in usable:
            raw = str(prediction.get("word", ""))
            if not raw.strip():
                continue
            if spans and spans[-1][1] == label and raw.startswith("##"):
                spans[-1][0] += raw[2:]
                spans[-1][2] = max(spans[-1][2], score)
            else:
                spans.append([raw.replace("##", ""), label, score])

    entities: list[tuple[str, str, float]] = []
    for text, label, score in spans:
        text = text.strip(" \t\n.,;:|")
        if len(text) < _MIN_ENTITY_CHARS or not any(ch.isalnum() for ch in text):
            continue
        # A real name/org starts with a capital; lowercase fragments are the
        # tail of a word the tokenizer chopped (``us Analytics``, ``net``).
        if label != "LOCATION" and not text[0].isupper() and not text[0].isdigit():
            continue
        entities.append((text, label, score))
    return entities


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParsedResume:
    name: str = ""
    headline: str = ""
    contact: dict = field(default_factory=dict)
    skills: list[dict] = field(default_factory=list)
    experience: dict = field(default_factory=dict)
    education: dict = field(default_factory=dict)
    certifications: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    ner_entities: list[dict] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self, *, include_raw: bool = False) -> dict:
        payload = {
            "name": self.name,
            "headline": self.headline,
            "contact": self.contact,
            "skills": self.skills,
            "experience": self.experience,
            "education": self.education,
            "certifications": self.certifications,
            "projects": self.projects,
            "confidence": self.confidence,
            "ner_entities": self.ner_entities,
        }
        if include_raw:
            payload["raw_text"] = self.raw_text
        return payload


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ResumeParser:
    """Turns raw resume text into a structured, JSON-serialisable record."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        *,
        ner_pipeline=None,
        use_ner: bool | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self._ner = ner_pipeline
        self._ner_resolved = ner_pipeline is not None
        self._use_ner = settings.enable_ner if use_ner is None else use_ner

    # ------------------------------------------------------------------ API
    def parse(self, text: str, *, run_ner: bool = True) -> ParsedResume:
        cleaned = normalize_text(text)
        if not cleaned.strip():
            raise ValueError("Cannot parse an empty resume")

        lines = [line for line in cleaned.splitlines() if line.strip()]
        entities: list[dict] = []
        if run_ner and self._use_ner:
            entities = self.run_ner(cleaned)

        sections, section_spans = self._detect_sections(lines)

        name = self._extract_name(lines, entities)
        contact = self._extract_contact(cleaned, lines, entities)
        skills = self._extract_skills(cleaned, sections)
        experience = self._extract_experience(cleaned, sections, entities)
        education = self._extract_education(cleaned, sections, entities)
        headline = self._extract_headline(cleaned, experience)

        certifications = self._extract_list_section(sections.get("certifications", ""))
        projects = self._extract_list_section(sections.get("projects", ""))
        confidence = self._confidence(name, contact, skills, experience, education)

        return ParsedResume(
            name=name,
            headline=headline,
            contact=contact,
            skills=[hit.to_dict() if isinstance(hit, SkillHit) else hit for hit in skills],
            experience=experience,
            education=education,
            certifications=certifications,
            projects=projects,
            confidence=confidence,
            ner_entities=self._public_entities(entities),
            raw_text=cleaned,
        )

    # -------------------------------------------------------------- NER glue
    def _pipeline(self):
        if not self._ner_resolved:
            self._ner = self.registry.get_ner()
            self._ner_resolved = True
        return self._ner

    def run_ner(self, text: str) -> list[dict]:
        pipeline = self._pipeline()
        if pipeline is None:
            return []

        from utils.text_utils import chunk_text

        chunks = chunk_text(text[:12_000], 900)[:12]
        found: dict[tuple[str, str], dict] = {}
        for index, chunk in enumerate(chunks):
            if len(chunk) < 10:
                continue
            try:
                with self.registry.inference_lock:
                    predictions = pipeline(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.warning("NER failed on chunk %s: %s", index, exc)
                continue

            for text_value, label, score in _entities_from_predictions(chunk, predictions):
                key = (text_value.lower(), label)
                if key not in found or score > found[key]["score"]:
                    found[key] = {"text": text_value, "label": label, "score": round(score, 3)}
        return list(found.values())

    # -------------------------------------------------------------- Sections
    def _detect_sections(self, lines: Sequence[str]) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
        sections: dict[str, list[str]] = {}
        spans: dict[str, tuple[int, int]] = {}
        current: str | None = None
        start = 0
        buffer: list[str] = []

        def flush(end: int) -> None:
            nonlocal buffer, current
            if current:
                text = "\n".join(buffer).strip()
                if text:
                    sections[current] = text.splitlines()
                    spans[current] = (start, end)
            buffer = []

        for index, raw in enumerate(lines):
            key, inline = self._heading_for(raw)
            if key and inline and current == "skills" and key != "skills":
                # Inside a skills block, ``Languages: Python, SQL, ...`` is a
                # category label, not a new section. Section headings that end
                # the block ("Certifications") carry no inline list.
                if "," in inline:
                    key = None
            if key:
                flush(index)
                current = key
                start = index
                buffer = [inline] if inline else []
            elif current:
                buffer.append(raw)
        flush(len(lines))

        return {key: "\n".join(value) for key, value in sections.items()}, spans

    @staticmethod
    def _heading_for(line: str) -> tuple[str | None, str]:
        raw = strip_bullet(line).strip()
        if not raw or len(raw) > 80:
            return None, ""
        head, sep, rest = raw.partition(":")
        if len(head) > 55:
            return None, ""
        candidate = re.sub(r"\s+", " ", head.strip(" .,;:•-–—|")).lower()
        key = _SECTION_LOOKUP.get(candidate)
        if key is None:
            return None, ""
        inline = rest.strip() if sep else ""
        return key, inline

    # ------------------------------------------------------------------ Name
    def _extract_name(self, lines: Sequence[str], entities: Sequence[dict]) -> str:
        per_names = {entity["text"].strip().lower() for entity in entities if entity["label"] == "PERSON"}

        best: tuple[float, str] = (0.0, "")
        for index, raw in enumerate(lines[:15]):
            line = strip_bullet(raw).strip()
            if self._heading_for(line)[0]:
                break
            if not self._is_name_candidate(line):
                continue

            words = line.split()
            normalised = re.sub(r"\s+", " ", line).strip(" .,").lower()
            score = 0.0
            if normalised in per_names:
                score += 3.0
            if 2 <= len(words) <= 3:
                score += 1.5
            if all(word[0].isupper() for word in words if word):
                score += 1.5
            if re.fullmatch(r"[A-Za-z][A-Za-z.'\- ]+", line):
                score += 1.0
            if index <= 2:
                score += 1.0
            if len(words) == 1:
                score -= 1.5
            if score > best[0]:
                best = (score, line)

        if best[0] < 2.0:
            return ""
        name = re.split(r"[,|•]", best[1])[0].strip()
        return title_case_name(name)

    @staticmethod
    def _is_name_candidate(line: str) -> bool:
        if not line or len(line) > 45:
            return False
        if any(char.isdigit() for char in line):
            return False
        if any(token in line for token in ("@", "http", "linkedin", "github", "|", "/", "•")):
            return False
        words = line.split()
        if not 1 <= len(words) <= 4:
            return False
        if not re.fullmatch(r"[A-Za-z][A-Za-z.'\- ]+", line):
            return False
        lowered = {word.lower().strip(".,") for word in words}
        if lowered & _NAME_STOPWORDS:
            return False
        if lowered & {"and", "the", "of"}:
            return False
        return True

    # --------------------------------------------------------------- Contact
    def _extract_contact(self, text: str, lines: Sequence[str], entities: Sequence[dict]) -> dict:
        email_match = _EMAIL_RE.search(text)
        email = email_match.group(0).strip(" .,;:").lower() if email_match else ""

        phone = self._extract_phone(text)

        linkedin_match = _LINKEDIN_RE.search(text)
        github_match = _GITHUB_RE.search(text)
        linkedin = self._normalise_url(linkedin_match.group(0)) if linkedin_match else ""
        github = self._normalise_url(github_match.group(0)) if github_match else ""

        website = ""
        for match in _WEBSITE_RE.finditer(text):
            url = match.group(0)
            if any(domain in url.lower() for domain in ("linkedin.com", "github.com")):
                continue
            if email and url.lower() in email:
                continue
            if "@" in text[max(0, match.start() - 1): match.start() + 1]:
                continue
            website = self._normalise_url(url)
            break

        location = self._extract_location(lines, entities)

        links = [url for url in (linkedin, github, website) if url]
        return {
            "email": email,
            "phone": phone,
            "location": location,
            "linkedin": linkedin,
            "github": github,
            "website": website,
            "links": links,
        }

    @staticmethod
    def _normalise_url(url: str) -> str:
        url = url.strip().strip(".,;)")
        if not re.match(r"^https?://", url, re.I):
            url = "https://" + url.lstrip("/")
        return url

    @staticmethod
    def _extract_phone(text: str) -> str:
        candidates: list[tuple[int, str]] = []
        for match in _PHONE_RE.finditer(text):
            raw = match.group(1).strip(" .-")
            digits = re.sub(r"\D", "", raw)
            if not 9 <= len(digits) <= 15:
                continue
            if digits.startswith("0000"):
                continue
            score = 0
            if raw.startswith("+"):
                score += 2
            if "(" in raw:
                score += 1
            if len(digits) in (10, 11, 12):
                score += 1
            candidates.append((score, re.sub(r"\s{2,}", " ", raw)))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: -item[0])
        return candidates[0][1]

    @staticmethod
    def _extract_location(lines: Sequence[str], entities: Sequence[dict]) -> str:
        for line in lines[:15]:
            # Location is often one segment of a pipe-delimited contact line.
            for segment in re.split(r"[|•·]", line):
                segment = segment.strip(" .,-–—")
                if not segment or "@" in segment or "http" in segment.lower():
                    continue
                match = _LOCATION_RE.match(segment)
                if match:
                    return f"{match.group(1)}, {match.group(2)}"
        for entity in entities:
            if entity["label"] == "LOCATION" and len(entity["text"]) <= 40:
                return entity["text"]
        return ""

    # ---------------------------------------------------------------- Skills
    def _extract_skills(self, text: str, sections: dict[str, str]) -> list[SkillHit]:
        skills_section = sections.get("skills", "")
        hits = find_skills(text, skills_section=skills_section)

        known = [hit.name for hit in hits]
        extra = extract_unknown_skills(skills_section, known=known)
        for name in extra:
            hits.append(
                SkillHit(
                    name=name,
                    category="Other",
                    occurrences=1,
                    in_skills_section=True,
                    confidence=0.55,
                )
            )

        hits.sort(key=lambda hit: (-hit.confidence, hit.category, hit.name))
        return hits[:80]

    # ------------------------------------------------------------ Experience
    def _extract_experience(
        self, text: str, sections: dict[str, str], entities: Sequence[dict]
    ) -> dict:
        section = sections.get("experience", "")
        if section:
            source = section
        else:
            source = self._fallback_experience_text(text)

        entries = self._parse_experience_entries(source)
        months = total_experience_months(source)
        claim = extract_years_claim(text)

        years = months_to_years(months) if months else 0.0
        source_kind = "dates" if years else "none"
        if claim > years:
            years = claim
            source_kind = "claim" if not months else "dates+claim"
        years = round(min(years, 50.0), 1)

        companies = dedupe(
            [entry["company"] for entry in entries if entry.get("company")]
            + self._company_entities(entities, source)
            + self._companies_from_text(source),
            key=str.lower,
        )[:10]

        titles = dedupe([entry["title"] for entry in entries if entry.get("title")], key=str.lower)[:10]

        return {
            "years": years,
            "years_source": source_kind,
            "companies": companies,
            "titles": titles,
            "entries": entries[:12],
        }

    @staticmethod
    def _fallback_experience_text(text: str) -> str:
        """When there is no Experience heading, keep only date-bearing role lines."""
        selected: list[str] = []
        for line in text.splitlines():
            if not extract_date_ranges(line):
                continue
            if _ROLE_RE.search(line) or _COMPANY_SUFFIX_RE.search(line) or re.search(r"\bpresent\b", line, re.I):
                selected.append(line)
        return "\n".join(selected) if selected else ""

    def _parse_experience_entries(self, source: str) -> list[dict]:
        lines = [line.strip() for line in source.splitlines() if line.strip()]
        entries: list[dict] = []

        for index, line in enumerate(lines):
            ranges = extract_date_ranges(line)
            if not ranges:
                continue
            rng = ranges[0]

            remainder = (line[: rng.position] + " " + line[rng.position + len(rng.raw):]).strip()
            segments = [
                clean_snippet(segment, max_len=90)
                for segment in re.split(r"\s*[|•·]\s*|\s+[–—]\s+|\s+-\s+|\s+at\s+", remainder)
                if segment.strip()
            ]

            title = segments[0] if segments else ""
            company = segments[1] if len(segments) > 1 else ""

            if company and _COMPANY_SUFFIX_RE.search(title) and not _COMPANY_SUFFIX_RE.search(company):
                title, company = company, title
            if not title or title.lower() in {"present", "current"}:
                title = self._neighbour_line(lines, index, -1)
            if not company:
                company = self._neighbour_line(lines, index, 1)
            if company and _INSTITUTION_RE.search(company):
                company = ""

            description_lines: list[str] = []
            for follow in lines[index + 1: index + 7]:
                if extract_date_ranges(follow):
                    break
                if self._heading_for(follow)[0]:
                    break
                description_lines.append(strip_bullet(follow))
            description = clean_snippet(" ".join(description_lines), max_len=500)

            entries.append(
                {
                    "title": clean_snippet(title, max_len=90),
                    "company": clean_snippet(company, max_len=90),
                    "start": self._format_date(rng.start),
                    "end": "Present" if rng.end is None else self._format_date(rng.end),
                    "duration_months": max(0, rng.end_index() - rng.start_index),
                    "description": description,
                }
            )

        deduped: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for entry in entries:
            key = (entry["title"].lower(), entry["company"].lower(), entry["start"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        return deduped

    @staticmethod
    def _neighbour_line(lines: Sequence[str], index: int, offset: int) -> str:
        position = index + offset
        if not 0 <= position < len(lines):
            return ""
        candidate = strip_bullet(lines[position])
        if len(candidate) > 90 or extract_date_ranges(candidate):
            return ""
        if _ROLE_RE.search(candidate) or _COMPANY_SUFFIX_RE.search(candidate):
            return candidate.strip(" |,•-–—")
        return ""

    @staticmethod
    def _format_date(value: tuple[int, int]) -> str:
        year, month = value
        month_names = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        return f"{month_names[month - 1]} {year}"

    @staticmethod
    def _company_entities(entities: Sequence[dict], source: str) -> list[str]:
        """Employers the NER pass found *inside the experience section*.

        BERT happily tags tool names ("Airflow", "Terraform") as organisations
        and, on a chunk boundary, returns half a word. Requiring the candidate
        to occur in the experience text and to not be a known skill keeps the
        list to actual employers.
        """
        haystack = source.lower()
        index = alias_index()
        companies = []
        for entity in entities:
            if entity["label"] != "ORG":
                continue
            text = clean_snippet(entity["text"].strip(" .,|"), max_len=60)
            if not _MIN_ENTITY_CHARS <= len(text) <= 60:
                continue
            lowered = text.lower()
            # "GitHub Action" is a truncated "GitHub Actions": check the plural.
            if lowered in _COMPANY_NOISE or lowered in index or f"{lowered}s" in index:
                continue
            if _INSTITUTION_RE.search(text):
                continue
            if haystack and lowered not in haystack:
                continue
            # An employer is either a multi-word name or carries a legal suffix;
            # this drops one-word ORG noise such as "Table".
            if len(text.split()) < 2 and not _COMPANY_SUFFIX_RE.search(text):
                continue
            companies.append(text)
        return dedupe(companies, key=str.lower)[:8]

    @staticmethod
    def _companies_from_text(text: str) -> list[str]:
        found = []
        for line in text.splitlines():
            if not _COMPANY_SUFFIX_RE.search(line):
                continue
            # Only the segment that actually carries the company suffix is the
            # employer; ``Data Scientist | Bluewave Systems | 2017-2019`` must
            # not yield "Data Scientist".
            candidate = ""
            for segment in re.split(r"[|•·]", line):
                segment = segment.strip(" .,-–—")
                if not segment or extract_date_ranges(segment):
                    continue
                if _INSTITUTION_RE.search(segment):
                    continue
                if _COMPANY_SUFFIX_RE.search(segment):
                    candidate = segment
                    break
            candidate = clean_snippet(candidate, max_len=80)
            if 2 <= len(candidate) <= 80:
                found.append(candidate)
        return dedupe(found, key=str.lower)[:8]

    # ------------------------------------------------------------- Education
    def _extract_education(
        self, text: str, sections: dict[str, str], entities: Sequence[dict]
    ) -> dict:
        section = sections.get("education", "")
        source = section or text
        lines = [line for line in source.splitlines() if line.strip()]

        entries: list[dict] = []
        for index, line in enumerate(lines):
            for pattern, rank, label in _DEGREES:
                match = pattern.search(line)
                if not match:
                    continue
                year_match = re.search(r"\b(19|20)\d{2}\b", line)
                year = int(year_match.group(0)) if year_match else None

                institution = ""
                if _INSTITUTION_RE.search(line):
                    institution = self._institution_fragment(line)
                else:
                    for offset in (-2, -1, 1, 2):
                        neighbour = index + offset
                        if 0 <= neighbour < len(lines) and _INSTITUTION_RE.search(lines[neighbour]):
                            institution = self._institution_fragment(lines[neighbour])
                            break
                if not institution:
                    institution = self._nearest_entity(text, entities, "ORG") or ""

                entries.append(
                    {
                        "degree": label,
                        "raw": clean_snippet(line, max_len=180),
                        "field": self._extract_field(line, match.end()),
                        "institution": clean_snippet(institution, max_len=120),
                        "year": year,
                        "level_rank": rank,
                    }
                )
                break

        entries = dedupe_by_degree(entries)[:6]
        highest_rank = max((entry["level_rank"] for entry in entries), default=0)
        institutions = dedupe(
            [entry["institution"] for entry in entries if entry["institution"]],
            key=str.lower,
        )[:6]

        return {
            "highest_level": _LEVEL_NAMES.get(highest_rank, "Not detected"),
            "highest_level_rank": highest_rank,
            "degrees": entries,
            "institutions": institutions,
        }

    @staticmethod
    def _institution_fragment(line: str) -> str:
        parts = [
            part.strip()
            for part in re.split(r"[|•·]|\s[-–—]\s", line)
            if part.strip()
        ]
        for part in parts:
            if _INSTITUTION_RE.search(part):
                # Drop the graduation year and anything after it.
                part = re.sub(r"[,;]?\s*\b(?:19|20)\d{2}\b.*$", "", part)
                return clean_snippet(part.strip(" .,-"), max_len=120)
        return ""

    @staticmethod
    def _extract_field(line: str, start: int) -> str:
        tail = line[start:]
        match = re.search(
            r"\bin\s+([A-Za-z][A-Za-z&/\- ]{2,60})", tail, re.I
        )
        if not match:
            return ""
        field = re.split(
            r"\s*[-–—|]\s*|,|\bat\b|\b(?:19|20)\d{2}\b", match.group(1)
        )[0]
        return clean_snippet(field, max_len=70).strip(" .,-")

    @staticmethod
    def _nearest_entity(text: str, entities: Sequence[dict], label: str) -> str:
        for entity in entities:
            if entity["label"] == label and len(entity["text"]) <= 60:
                return entity["text"]
        return ""

    # -------------------------------------------------------------- Headline
    @staticmethod
    def _extract_headline(text: str, experience: dict) -> str:
        titles = experience.get("titles") or []
        if titles:
            return titles[0]
        for line in text.splitlines()[:20]:
            candidate = strip_bullet(line)
            if 3 <= len(candidate) <= 70 and _ROLE_RE.search(candidate):
                if not _EMAIL_RE.search(candidate) and not candidate.lower().startswith(("summary", "objective")):
                    return clean_snippet(candidate, max_len=70)
        return ""

    # ------------------------------------------------------------ List items
    @staticmethod
    def _extract_list_section(section: str, *, limit: int = 12) -> list[str]:
        if not section:
            return []
        items: list[str] = []
        for line in section.splitlines():
            for piece in re.split(r"[•·|]", line):
                piece = clean_snippet(strip_bullet(piece), max_len=160)
                if 4 <= len(piece) <= 160 and not piece.lower().endswith(":"):
                    items.append(piece)
        return dedupe(items, key=str.lower)[:limit]

    # ------------------------------------------------------------ Confidence
    @staticmethod
    def _confidence(name, contact, skills, experience, education) -> dict:
        def ratio(value: float) -> float:
            return round(max(0.0, min(1.0, value)), 2)

        email = contact.get("email", "")
        phone = contact.get("phone", "")
        return {
            "name": 0.9 if name else 0.0,
            "email": 1.0 if email else 0.0,
            "phone": 0.85 if phone else 0.0,
            "skills": ratio(len(skills) / 8),
            "experience": 1.0 if experience.get("years_source") in {"dates", "dates+claim"} else (0.6 if experience.get("years") else 0.0),
            "education": 1.0 if education.get("highest_level_rank") else 0.0,
            "overall": round(
                sum(
                    [
                        0.9 if name else 0.0,
                        1.0 if email else 0.0,
                        0.85 if phone else 0.0,
                        ratio(len(skills) / 8),
                        1.0 if experience.get("years") else 0.0,
                        1.0 if education.get("highest_level_rank") else 0.0,
                    ]
                )
                / 6,
                2,
            ),
        }

    # --------------------------------------------------------------- Helpers
    @staticmethod
    def _public_entities(entities: Sequence[dict], limit: int = 25) -> list[dict]:
        ordered = sorted(entities, key=lambda entity: -entity["score"])[:limit]
        return ordered


def dedupe_by_degree(entries: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    output: list[dict] = []
    for entry in entries:
        key = (str(entry.get("degree", "")).lower(), str(entry.get("institution", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        output.append(entry)
    return output


_parser: ResumeParser | None = None


def get_parser(*, registry: ModelRegistry | None = None, **kwargs) -> ResumeParser:
    global _parser
    if _parser is None or registry is not None or kwargs:
        _parser = ResumeParser(registry, **kwargs)
    return _parser
