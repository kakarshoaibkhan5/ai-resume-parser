"""Pure-Python text helpers: normalisation, date maths and extractive ranking."""

from __future__ import annotations

from datetime import date

from utils.text_utils import (
    build_fallback_bullets,
    chunk_text,
    clean_snippet,
    content_tokens,
    dedupe,
    extract_date_ranges,
    extract_years_claim,
    merge_intervals,
    months_to_years,
    normalize_text,
    select_bullets,
    split_sentences,
    strip_bullet,
    title_case_name,
    total_experience_months,
)


# --------------------------------------------------------------- normalisation
def test_normalize_text_collapses_whitespace_and_control_chars() -> None:
    raw = "\ufeffHello\u00a0  world\x00\n\n\n\nnext   line"
    assert normalize_text(raw) == "Hello world\n\nnext line"


def test_normalize_text_repairs_hyphenated_line_breaks() -> None:
    assert normalize_text("full-\nstack\ndeveloper") == "fullstack\ndeveloper"


def test_normalize_text_handles_empty_input() -> None:
    assert normalize_text("") == ""


def test_strip_bullet_understands_common_markers() -> None:
    assert strip_bullet("• Python") == "Python"
    assert strip_bullet("- Python") == "Python"
    assert strip_bullet("* Python") == "Python"
    assert strip_bullet("1. Python") == "Python"
    assert strip_bullet("plain text") == "plain text"


def test_clean_snippet_truncates_on_a_word_boundary() -> None:
    short = "Owned the ranking pipeline"
    assert clean_snippet(short) == short

    long = "word " * 80
    cleaned = clean_snippet(long, max_len=40)
    assert len(cleaned) <= 43
    assert cleaned.endswith("...")


def test_dedupe_is_case_insensitive_by_default() -> None:
    assert dedupe(["Python", "python", "SQL", "sql"]) == ["Python", "SQL"]


def test_dedupe_accepts_a_custom_key() -> None:
    assert dedupe(["A1", "A2", "B1"], key=lambda item: item[0]) == ["A1", "B1"]


def test_title_case_name_normalises_shouting_and_lowercase() -> None:
    assert title_case_name("JOHN DOE") == "John Doe"
    assert title_case_name("john doe") == "John Doe"
    assert title_case_name("Aarav Mehta") == "Aarav Mehta"
    assert title_case_name("") == ""


# ------------------------------------------------------------------ sentences
def test_split_sentences_rejoins_pdf_line_wraps() -> None:
    text = (
        "Machine learning engineer with 8 years of experience shipping production NLP\n"
        "and recommendation systems. Led a team of four engineers."
    )
    sentences = split_sentences(text)
    assert sentences[0].endswith("and recommendation systems.")
    assert len(sentences) == 2


def test_split_sentences_keeps_bullets_separate() -> None:
    text = (
        "- Owned the ranking pipeline and cut inference cost by 38 percent.\n"
        "- Shipped a retrieval augmented generation assistant to 12k users."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert all(not sentence.startswith("-") for sentence in sentences)


def test_split_sentences_drops_noise_shorter_than_minimum() -> None:
    assert split_sentences("Too short. Also short.", min_chars=50) == []


def test_chunk_text_respects_the_budget_when_possible() -> None:
    text = " ".join(f"Sentence number {index} about machine learning systems." for index in range(40))
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert "machine learning" in chunks[0]


# ----------------------------------------------------------------- date maths
def test_extract_date_ranges_parses_months_and_present() -> None:
    text = (
        "Engineer | Acme | Jan 2020 - Mar 2022\n"
        "Engineer | Beta | Jun 2022 - Present"
    )
    ranges = extract_date_ranges(text)
    assert len(ranges) == 2
    assert ranges[0].start == (2020, 1)
    assert ranges[0].end == (2022, 3)
    assert ranges[1].start == (2022, 6)
    assert ranges[1].end is None


def test_extract_date_ranges_accepts_numeric_dates() -> None:
    ranges = extract_date_ranges("Developer 09/2019 - 12/2021")
    assert len(ranges) == 1
    assert ranges[0].start == (2019, 9)
    assert ranges[0].end == (2021, 12)


def test_extract_date_ranges_ignores_backwards_ranges() -> None:
    assert extract_date_ranges("Engineer 2022 - 2019") == []


def test_total_experience_months_counts_inclusive_whole_months() -> None:
    text = "Jan 2020 - Dec 2021\nJan 2022 - Dec 2022"
    assert total_experience_months(text, today=date(2024, 1, 1)) == 36


def test_total_experience_months_handles_present_with_injected_today() -> None:
    assert total_experience_months("Jan 2024 - Present", today=date(2024, 3, 1)) == 3


def test_total_experience_months_deduplicates_overlaps() -> None:
    text = "Jan 2020 - Dec 2021\nJan 2021 - Dec 2021"
    assert total_experience_months(text, today=date(2024, 1, 1)) == 24


def test_months_to_years_rounds_to_one_decimal() -> None:
    assert months_to_years(36) == 3.0
    assert months_to_years(58) == 4.8


def test_merge_intervals_merges_touching_or_overlapping_spans() -> None:
    assert merge_intervals([(0, 10), (5, 15), (30, 40)]) == [(0, 15), (30, 40)]
    assert merge_intervals([]) == []


def test_extract_years_claim_takes_the_largest_plausible_value() -> None:
    text = "3+ years in Python and 8 years of experience overall."
    assert extract_years_claim(text) == 8.0
    assert extract_years_claim("No numbers here") == 0.0


# --------------------------------------------------------------- tokenisation
def test_content_tokens_drop_stopwords_and_punctuation() -> None:
    tokens = content_tokens("The quick brown fox, and the lazy dog!")
    assert tokens == ["quick", "brown", "fox", "lazy", "dog"]


# ------------------------------------------------------------ bullet ranking
def test_select_bullets_deduplicates_and_limits() -> None:
    candidates = [
        "Led a team of four engineers to rebuild the search ranking stack.",
        "Led a team of four engineers to rebuild the search ranking stack.",
        "Built churn models protecting 2.4M dollars of annual revenue.",
    ]
    selected = select_bullets(candidates, "search ranking revenue", limit=3)
    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_select_bullets_prefers_quantified_impact() -> None:
    plain = "We improved the platform for customers across the region."
    quantified = "We cut inference cost by 38% across the platform."
    selected = select_bullets([plain, quantified], "inference cost", limit=1)
    assert selected == [quantified]


def test_select_bullets_skips_fragments() -> None:
    assert select_bullets(["Too short.", "Also too short."], "anything") == []


def test_build_fallback_bullets_is_deterministic_and_capped() -> None:
    bullets = build_fallback_bullets(
        name="Aarav Mehta",
        headline="ML Engineer",
        years=8,
        companies=["Nimbus", "Vertex"],
        skills=["Python", "SQL"],
        education="Master's in CS",
    )
    assert len(bullets) == 3
    assert "8+ years" in bullets[0]
    assert "Python" in bullets[1]
    assert "Master's" in bullets[2]


def test_build_fallback_bullets_never_returns_nothing() -> None:
    bullets = build_fallback_bullets()
    assert len(bullets) == 1
    assert "parsed successfully" in bullets[0]
