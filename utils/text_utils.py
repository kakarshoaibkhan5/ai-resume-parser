"""Text normalisation, date arithmetic and extractive summarisation helpers.

Nothing in here depends on a model, so it is fully unit-testable and is used as
the graceful-degradation path when transformers cannot be loaded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2022": "•",
        "\uf0b7": "•",
        "\u25aa": "•",
        "\u00ad": "",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2026": "...",
    }
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BULLET_PREFIX = re.compile(r"^\s*(?:[•▪◦‣·*\-–—]|\d{1,2}[.)])\s+")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Clean PDF text while preserving the line structure used for layout."""
    if not text:
        return ""
    text = text.translate(_TRANSLATION)
    text = _CONTROL_CHARS.sub("", text)
    # Re-join words hyphenated across a line break ("full-\nstack").
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    lines = []
    for line in text.splitlines():
        line = _MULTI_SPACE.sub(" ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def strip_bullet(line: str) -> str:
    return _BULLET_PREFIX.sub("", line).strip()


def clean_snippet(text: str, *, max_len: int = 240) -> str:
    """Collapse whitespace and truncate on a word boundary."""
    text = re.sub(r"\s+", " ", text).strip(" •-|,;:")
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "..."


def dedupe(items: Iterable[str], *, key=str.lower) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        k = key(item)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def title_case_name(value: str) -> str:
    """Normalise ``JOHN DOE`` / ``john doe`` to ``John Doe``."""
    if not value:
        return ""
    parts = []
    for token in value.split():
        if token.isupper() or token.islower():
            token = token.capitalize()
        parts.append(token)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sentences & chunks
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9•])")
_INLINE_BULLET_SPLIT = re.compile(r"\s+[•|]\s+")


def _join_wrapped_lines(text: str) -> str:
    """Re-join prose that a PDF line-wrapped mid-sentence.

    A line-wrapped sentence looks like ``... production NLP`` followed by
    ``and recommendation systems.`` — the continuation starts lower-case, so it
    is glued back instead of becoming a truncated bullet fragment.
    """
    merged: list[str] = []
    for line in (raw.strip() for raw in text.splitlines()):
        if not line:
            merged.append("")
            continue
        if (
            merged
            and merged[-1]
            and not _BULLET_PREFIX.match(line)
            and not merged[-1].endswith((".", "!", "?", ":", ";"))
            and (line[0].islower() or line[0] in ",;)]}")
        ):
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    return "\n".join(merged)


def split_sentences(text: str, *, min_chars: int = 25) -> list[str]:
    """Split resume prose into clean, bullet-free sentences."""
    text = _join_wrapped_lines(normalize_text(text))
    sentences: list[str] = []
    for block in re.split(r"\n+", text):
        block = strip_bullet(block)
        if not block:
            continue
        for piece in _INLINE_BULLET_SPLIT.split(block):
            piece = piece.strip()
            if not piece:
                continue
            for sentence in _SENTENCE_SPLIT.split(piece):
                sentence = clean_snippet(sentence, max_len=400)
                if len(sentence) >= min_chars:
                    sentences.append(sentence)
    return dedupe(sentences)


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks on paragraph then sentence boundaries."""
    if max_chars <= 0:
        return [text]
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if len(block) > max_chars:
            for sentence in split_sentences(block, min_chars=1):
                if len(current) + len(sentence) + 1 > max_chars and current:
                    chunks.append(current.strip())
                    current = ""
                current += sentence + " "
            continue
        if len(current) + len(block) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += block + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ---------------------------------------------------------------------------
# Dates & experience arithmetic
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_MONTH_RE = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
)
_DATE_TOKEN = (
    rf"(?:{_MONTH_RE}\.?,?\s*\d{{2,4}}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{1,2}[/-]\d{4}"
    r"|\d{4})"
)
_PRESENT_RE = r"(?:present|currently|current|now|till\s+date|to\s+date|ongoing|today)"

_DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_TOKEN})\s*"
    r"(?:-|–|—|to|until|through|→|=>)\s*"
    rf"(?P<end>{_DATE_TOKEN}|{_PRESENT_RE})",
    re.IGNORECASE,
)

_YEARS_CLAIM_RES = (
    re.compile(
        r"(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)\s+"
        r"(?:of\s+)?(?:professional|relevant|hands[-\s]?on|industry|work|"
        r"practical|overall)?\s*experience",
        re.IGNORECASE,
    ),
    re.compile(
        r"experience\s*(?:of|:|-)?\s*(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class DateRange:
    start: tuple[int, int]  # (year, month)
    end: tuple[int, int] | None  # None => present
    raw: str
    position: int

    @property
    def start_index(self) -> int:
        return self.start[0] * 12 + (self.start[1] - 1)

    def end_index(self, today: date | None = None) -> int:
        if self.end is None:
            today = today or date.today()
            return today.year * 12 + (today.month - 1)
        return self.end[0] * 12 + (self.end[1] - 1)


def _is_valid_year(year: int) -> bool:
    return 1950 <= year <= date.today().year + 1


def _parse_date_token(token: str) -> tuple[int, int] | None:
    token = token.strip().lower().replace(",", " ").replace(".", " ")
    token = re.sub(r"\s+", " ", token)
    if not token:
        return None

    month_name = re.match(rf"({_MONTH_RE})\s*(\d{{2,4}})", token)
    if month_name:
        month = _MONTH_NAMES.get(month_name.group(1))
        year = int(month_name.group(2))
        if len(month_name.group(2)) == 2:
            year += 2000 if year < 50 else 1900
        if month and _is_valid_year(year):
            return year, month

    numeric = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", token)
    if numeric:
        month = int(numeric.group(1))
        year = int(numeric.group(3))
        if len(numeric.group(3)) == 2:
            year += 2000 if year < 50 else 1900
        if 1 <= month <= 12 and _is_valid_year(year):
            return year, month

    month_year = re.fullmatch(r"(\d{1,2})[/-](\d{4})", token)
    if month_year:
        month = int(month_year.group(1))
        year = int(month_year.group(2))
        if 1 <= month <= 12 and _is_valid_year(year):
            return year, month

    year_only = re.fullmatch(r"(\d{4})", token)
    if year_only:
        year = int(year_only.group(1))
        if _is_valid_year(year):
            return year, 6  # mid-year keeps interval maths conservative

    return None


def extract_date_ranges(text: str) -> list[DateRange]:
    """Find every ``<date> - <date|present>`` range, in document order."""
    ranges: list[DateRange] = []
    for match in _DATE_RANGE_RE.finditer(text):
        start = _parse_date_token(match.group("start"))
        if start is None:
            continue
        end_raw = match.group("end")
        end = None if re.fullmatch(_PRESENT_RE, end_raw, re.IGNORECASE) else _parse_date_token(end_raw)
        if end is not None and end < start:
            continue
        if start[0] < 1970:
            continue
        ranges.append(DateRange(start=start, end=end, raw=match.group(0), position=match.start()))
    return ranges


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping ``(start, end)`` month indices."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def total_experience_months(text: str, *, today: date | None = None) -> int:
    """Total non-overlapping months covered by date ranges in ``text``.

    Ranges are treated as whole-month inclusive spans, so ``Jan 2020 - Dec
    2021`` counts as 24 months rather than 23.
    """
    intervals: list[tuple[int, int]] = []
    for rng in extract_date_ranges(text):
        start, end = rng.start_index, rng.end_index(today)
        months = end - start + 1
        if 0 < months <= 600:  # ignore nonsense / 50+ year spans
            intervals.append((start, end))
    return sum(end - start + 1 for start, end in merge_intervals(intervals))


def months_to_years(months: int) -> float:
    return round(months / 12, 1)


def extract_years_claim(text: str) -> float:
    """Largest explicit "X years of experience" claim in the text."""
    claims: list[float] = []
    for pattern in _YEARS_CLAIM_RES:
        for match in pattern.finditer(text):
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if 0 < value <= 50:
                claims.append(value)
    return max(claims) if claims else 0.0


# ---------------------------------------------------------------------------
# Keyword scoring & bullet selection
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with", "using", "used", "use", "work",
    "worked", "working", "including", "across", "via", "per", "also", "our",
    "their", "this", "these", "those", "we", "you", "your", "i", "my",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-']{1,}")


def content_tokens(text: str) -> list[str]:
    return [
        word.lower().strip(".-'")
        for word in _WORD_RE.findall(text)
        if word.lower() not in _STOPWORDS and len(word) > 1
    ]


def keyword_frequencies(text: str) -> dict[str, int]:
    freq: dict[str, int] = {}
    for token in content_tokens(text):
        freq[token] = freq.get(token, 0) + 1
    return freq


def top_keywords(text: str, limit: int = 25) -> list[str]:
    freq = keyword_frequencies(text)
    return [word for word, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


_ACTION_VERB_RE = re.compile(
    r"\b(led|built|designed|developed|launched|delivered|architected|owned|"
    r"implemented|migrated|scaled|improved|reduced|increased|automated|"
    r"optimised|optimized|managed|mentored|drove|created|shipped|spearheaded)\b",
    re.IGNORECASE,
)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _polish_bullet(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip(" •-–—|,;:")
    sentence = re.sub(r"\s*\.\.\.$", "", sentence)
    if not sentence:
        return ""
    if sentence[0].islower():
        sentence = sentence[0].upper() + sentence[1:]
    if len(sentence) > 260:
        sentence = clean_snippet(sentence, max_len=240)
    return sentence


def select_bullets(
    candidates: Sequence[str],
    reference_text: str,
    *,
    limit: int = 3,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """Rank candidate sentences for relevance and diversity (MMR).

    Scores blend reference-keyword overlap, presence of metrics, action verbs
    and a small positional prior, then greedily pick diverse sentences.
    ``weights`` can bias individual candidates (e.g. model-written sentences).
    """
    pairs: list[tuple[str, float]] = []
    for index, raw in enumerate(candidates):
        sentence = _polish_bullet(raw)
        if len(sentence) < 30:
            continue
        weight = 1.0
        if weights is not None and index < len(weights):
            weight = max(0.0, float(weights[index]))
        pairs.append((sentence, weight))

    seen: set[str] = set()
    unique_pairs: list[tuple[str, float]] = []
    for sentence, weight in pairs:
        key = re.sub(r"[^a-z0-9]", "", sentence.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append((sentence, weight))
    if not unique_pairs:
        return []

    cleaned = [sentence for sentence, _ in unique_pairs]
    pair_weights = [weight for _, weight in unique_pairs]

    reference = keyword_frequencies(reference_text)
    total = max(1, len(reference))
    token_sets = [set(content_tokens(c)) for c in cleaned]

    scores: list[float] = []
    for index, (sentence, tokens) in enumerate(zip(cleaned, token_sets)):
        if not tokens:
            scores.append(0.0)
            continue
        overlap = sum(reference.get(t, 0) / total for t in tokens)
        score = overlap / (len(tokens) ** 0.55)
        if re.search(r"\d", sentence):
            score *= 1.30  # quantified impact
        if _ACTION_VERB_RE.search(sentence):
            score *= 1.15
        if 55 <= len(sentence) <= 230:
            score *= 1.10
        elif len(sentence) > 320:
            score *= 0.55
        if index < 4:
            score *= 1.05
        score *= pair_weights[index]
        scores.append(score)

    chosen: list[int] = []
    while len(chosen) < min(limit, len(cleaned)):
        best_index, best_value = -1, float("-inf")
        for index, score in enumerate(scores):
            if index in chosen:
                continue
            if chosen:
                penalty = max(_jaccard(token_sets[index], token_sets[j]) for j in chosen)
                value = score * (1 - 0.75 * penalty)
            else:
                value = score
            if value > best_value:
                best_index, best_value = index, value
        if best_index < 0:
            break
        chosen.append(best_index)

    return [cleaned[i] for i in chosen]


def build_fallback_bullets(
    *,
    name: str = "",
    headline: str = "",
    years: float = 0.0,
    companies: Sequence[str] = (),
    skills: Sequence[str] = (),
    education: str = "",
) -> list[str]:
    """Deterministic 3-bullet summary used when the transformer is unavailable."""
    bullets: list[str] = []
    role = headline or "professional"
    # Only capitalise the first character: ``str.capitalize`` would turn
    # "ML Engineer" into "Ml engineer".
    role = role[0].upper() + role[1:]
    if years >= 1:
        bullets.append(
            f"{role} with {years:g}+ years of hands-on experience"
            + (f" across {len(companies)} organisations." if companies else ".")
        )
    elif headline:
        bullets.append(f"{headline}.")
    if skills:
        bullets.append(
            "Core strengths span " + ", ".join(list(skills)[:5]) + "."
        )
    if education:
        bullets.append(f"Education: {education}.")
    return bullets[:3] or ["Resume parsed successfully; no summary content was detected."]
