"""ai-resume-parser — Streamlit front-end.

Run with::

    streamlit run app.py

The UI is deliberately thin: all extraction, summarisation and scoring lives in
``parsers/`` and ``models/`` so it can be reused from a CLI or an API.
"""

from __future__ import annotations

import hashlib
import time
import traceback
from typing import Any

import streamlit as st

from models.matcher import MatchResult, match_resume_to_jd
from models.registry import get_registry
from models.summarizer import get_summarizer
from parsers.ner_parser import ParsedResume, get_parser
from parsers.pdf_reader import read_pdf
from utils.config import settings
from utils.exceptions import ResumeParserError
from utils.exporters import build_export, download_filename, to_json_bytes
from utils.logger import get_logger
from utils.sample import build_demo_pdf
from utils.theme import (
    chip,
    esc,
    footer,
    hero,
    inject_theme,
    kv_rows,
    meter,
    notice,
    score_ring,
    section_label,
    skill_chips,
    stat_row,
    timeline,
    bullets,
)
from utils.validators import validate_upload

logger = get_logger("app")

st.set_page_config(
    page_title="Resume Parser — structure, summary, match",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()


# ---------------------------------------------------------------------------
# Cached pipeline steps
#
# Streamlit re-runs the script on every interaction; caching keeps a 60 s parse
# from happening again when the user merely switches tabs.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=24)
def extract_resume(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """PDF -> text -> structured record. Cached by file content."""
    document = read_pdf(file_bytes, filename=filename)
    parsed: ParsedResume = get_parser().parse(document.text)
    return {
        "resume": parsed.to_dict(),
        "document": document.to_dict(),
        "text": document.text,
    }


@st.cache_data(show_spinner=False, max_entries=24)
def summarise_resume(text: str, fingerprint: str) -> dict[str, Any]:
    context = st.session_state.get(f"summary_context::{fingerprint}", {})
    result = get_summarizer().summarize(text, context=context)
    return result.to_dict()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


def _words(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown(
            """
            <div class="side-brand">
              <span class="mark">R</span>
              <div>
                <div class="name">ai-resume-parser</div>
                <div class="ver">v1.0.0 · open source</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        registry = get_registry()
        health = registry.health()

        st.markdown("#### Runtime")
        loaded = sum(1 for s in health.models.values() if s.loaded)
        device_label = health.device.upper()
        runtime_rows = kv_rows(
            [
                ("device", device_label),
                ("offline", "yes" if health.offline else "no"),
                ("models loaded", f"{loaded}/3"),
                ("transformers", "ok" if health.transformers_available else "missing"),
            ]
        )
        st.markdown(f'<div class="kv-list">{runtime_rows}</div>', unsafe_allow_html=True)

        for key, status in health.models.items():
            icon = "●" if status.loaded else ("◐" if status.error else "○")
            colour = "#c8f751" if status.loaded else ("#f2924e" if status.error else "#616a7e")
            extra = f" · {status.load_seconds}s" if status.load_seconds else ""
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:.76rem;'
                f'color:#8b93a7;padding:.12rem 0">'
                f'<span><span style="color:{colour}">{icon}</span> {esc(key)}</span>'
                f'<span style="font-family:var(--mono);font-size:.68rem">{esc(extra)}</span></div>',
                unsafe_allow_html=True,
            )

        if st.button("Warm up models", use_container_width=True, help="Load BERT + BART into memory now"):
            with st.spinner("Loading weights… this can take a minute on first run"):
                registry.warm_up()
            st.cache_data.clear()
            st.rerun()

        if not loaded:
            st.caption("Models load lazily on first parse. Pre-download them with `python scripts/download_models.py`.")

        st.markdown("#### Options")
        use_ner = st.toggle(
            "BERT entity hints",
            value=settings.enable_ner,
            help="dslim/bert-base-NER improves name/company detection. Regex still runs without it.",
        )
        use_summary = st.toggle(
            "BART summariser",
            value=settings.enable_summarizer,
            help="Disable to use the fast extractive fallback.",
        )
        bullet_count = st.slider(
            "Summary bullets", min_value=2, max_value=5, value=settings.summary_bullets
        )
        ocr = st.toggle(
            "OCR scanned PDFs",
            value=settings.enable_ocr,
            help="Requires tesseract-ocr installed on the host.",
        )

        st.markdown("#### Limits")
        st.caption(
            f"{settings.max_upload_mb} MB max · {settings.max_pdf_pages} pages · "
            f"{settings.max_resume_chars:,} chars"
        )

        with st.expander("About this build"):
            st.markdown(
                "Extraction is hybrid: regex + curated skill taxonomy + BERT NER. "
                "Summaries blend BART output with extractive MMR ranking so the "
                "three bullets stay factual. Job match is a weighted blend of "
                "skill coverage, semantic similarity, experience and education.\n\n"
                "No network calls at inference time. No resume leaves the host."
            )

    return {
        "use_ner": use_ner,
        "use_summary": use_summary,
        "bullets": bullet_count,
        "ocr": ocr,
    }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def render_overview(resume: dict, document: dict, summary: dict | None) -> None:
    experience = resume.get("experience", {})
    education = resume.get("education", {})
    skills = resume.get("skills", [])
    confidence = resume.get("confidence", {})

    stat_row(
        [
            (
                "Candidate",
                resume.get("name") or "Not detected",
                resume.get("headline") or "No headline found",
            ),
            (
                "Experience",
                f"{experience.get('years', 0):g} yrs",
                f"{len(experience.get('companies', []))} companies · {experience.get('years_source', 'none')}",
            ),
            (
                "Skills",
                str(len(skills)),
                f"{len({s.get('category') for s in skills})} categories",
            ),
            (
                "Confidence",
                f"{round(float(confidence.get('overall', 0)) * 100)}%",
                f"{document.get('pages_used', 0)} pages · {document.get('word_count', 0):,} words",
            ),
        ]
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Summary</div>', unsafe_allow_html=True)
        if summary and summary.get("bullets"):
            st.markdown(bullets(summary["bullets"]), unsafe_allow_html=True)
            meta = "abstractive + extractive" if not summary.get("degraded") else "extractive fallback"
            st.caption(
                f"{meta} · {summary.get('latency_ms', 0):,} ms · {summary.get('model', '')}"
            )
        else:
            st.markdown('<p class="empty">Summary disabled.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        contact = resume.get("contact", {})
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Contact</div>', unsafe_allow_html=True)
        rows = [
            ("Email", contact.get("email") or "—"),
            ("Phone", contact.get("phone") or "—"),
            ("Location", contact.get("location") or "—"),
            ("LinkedIn", contact.get("linkedin") or "—"),
            ("GitHub", contact.get("github") or "—"),
        ]
        st.markdown(kv_rows(rows), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Education</div>', unsafe_allow_html=True)
        if education.get("degrees"):
            for degree in education["degrees"]:
                year = f" · {degree['year']}" if degree.get("year") else ""
                field = f" in {degree['field']}" if degree.get("field") else ""
                institution = degree.get("institution") or "Institution not detected"
                st.markdown(
                    f'<div style="margin-bottom:.55rem">'
                    f'<div style="font-weight:600;font-size:.9rem">{esc(degree.get("degree", ""))}'
                    f'{esc(field)}</div>'
                    f'<div style="color:#8b93a7;font-size:.8rem">{esc(institution)}{esc(year)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<p class="empty">No degree detected.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Experience timeline</div>', unsafe_allow_html=True)
        st.markdown(timeline(experience.get("entries", [])), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if resume.get("certifications"):
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Certifications</div>', unsafe_allow_html=True)
            st.markdown(
                bullets(resume["certifications"][:6], numbered=False), unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)


def render_skills(resume: dict) -> None:
    skills = resume.get("skills", [])
    if not skills:
        notice("warn", "No skills detected", "The resume may be image-based or use unusual terminology.")
        return

    by_category: dict[str, list[dict]] = {}
    for skill in skills:
        by_category.setdefault(skill.get("category", "Other"), []).append(skill)

    st.markdown(
        f'<p class="empty" style="font-style:normal">'
        f'{len(skills)} skills across {len(by_category)} categories. '
        f'Filled chips are high-confidence matches from the skills section.</p>',
        unsafe_allow_html=True,
    )
    for category, items in sorted(by_category.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        st.markdown(
            f'<div class="panel-title" style="margin-top:.9rem">{esc(category)} '
            f'<span style="color:#616a7e">· {len(items)}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(skill_chips(items), unsafe_allow_html=True)


def render_match(extraction: dict, text: str, fingerprint: str) -> None:
    key = f"match::{fingerprint}"
    jd_key = f"jd::{fingerprint}"

    st.markdown(
        '<p class="empty" style="font-style:normal;margin-bottom:.6rem">'
        "Paste a job description to score fit across skills, semantics, experience and education.</p>",
        unsafe_allow_html=True,
    )
    jd_text = st.text_area(
        "Job description",
        value=st.session_state.get(jd_key, ""),
        height=190,
        placeholder="Paste the full JD here — responsibilities, requirements, nice-to-haves…",
        label_visibility="collapsed",
    )

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        analyse = st.button("Analyse match", type="primary", use_container_width=True)
    with col_b:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.pop(key, None)
        st.session_state.pop(jd_key, None)
        st.rerun()

    if analyse:
        st.session_state[jd_key] = jd_text
        if len(jd_text.strip()) < 40:
            notice("warn", "Job description too short", "Paste at least 40 characters so there is something to score.")
        else:
            with st.spinner("Comparing resume against the job description…"):
                try:
                    result: MatchResult = match_resume_to_jd(extraction, jd_text, resume_text=text)
                    st.session_state[key] = result.to_dict()
                except ResumeParserError as exc:
                    notice("error", exc.message, exc.hint or "")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Job matching failed")
                    notice("error", "Matching failed", str(exc))

    result = st.session_state.get(key)
    if not result:
        return

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    ring_col, detail_col = st.columns([1, 2.1], gap="large")

    with ring_col:
        st.markdown(
            score_ring(result["score"], tone=result["tone"], caption=result["verdict"]),
            unsafe_allow_html=True,
        )
        coverage = result.get("coverage", 0)
        st.markdown(
            f'<div style="text-align:center;font-family:var(--mono);font-size:.72rem;color:#616a7e">'
            f"skill coverage {coverage:.0f}% · {result.get('semantic_method', '')}</div>",
            unsafe_allow_html=True,
        )

    with detail_col:
        st.markdown('<div class="panel-title">Score breakdown</div>', unsafe_allow_html=True)
        meters = "".join(
            meter(
                item["label"],
                item["score"] if item["score"] is not None else 0,
                weight=item["weight"] or None,
                detail=item["detail"] + ("" if item["applicable"] else " · not requested"),
            )
            for item in result.get("breakdown", [])
        )
        st.markdown(meters, unsafe_allow_html=True)

    matched = result.get("matched_skills", [])
    missing = result.get("missing_skills", [])

    st.markdown("---")
    col_match, col_miss = st.columns(2, gap="large")

    with col_match:
        st.markdown(
            f'<div class="panel-title">Matched skills · {len(matched)}</div>',
            unsafe_allow_html=True,
        )
        if matched:
            st.markdown(
                '<div class="chip-cloud">'
                + "".join(
                    chip(
                        s["name"],
                        tone="accent",
                        title=f"{s['category']} · required" if s.get("required") else s.get("category", ""),
                    )
                    for s in matched
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<p class="empty">No overlap found.</p>', unsafe_allow_html=True)

    with col_miss:
        st.markdown(
            f'<div class="panel-title">Missing skills · {len(missing)}</div>',
            unsafe_allow_html=True,
        )
        if missing:
            st.markdown(
                '<div class="chip-cloud">'
                + "".join(
                    chip(s["name"], tone="miss" if s.get("required") else "warn", title="Required" if s.get("required") else "Nice to have")
                    for s in missing
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="empty">Nothing missing — full skill coverage.</p>',
                unsafe_allow_html=True,
            )

    if result.get("recommendations"):
        st.markdown('<div class="panel-title" style="margin-top:1.2rem">How to improve the match</div>', unsafe_allow_html=True)
        st.markdown(bullets(result["recommendations"][:5], numbered=False), unsafe_allow_html=True)

    exp = result.get("experience", {})
    if exp:
        st.caption(
            f"Experience: {exp.get('resume_years', 0):g} yrs on resume vs "
            f"{exp.get('required_years') or '—'} requested"
        )


def render_json(payload: dict, filename: str) -> None:
    st.markdown(
        '<p class="empty" style="font-style:normal">Stable, versioned schema — safe to pipe into an ATS or warehouse.</p>',
        unsafe_allow_html=True,
    )
    st.json(payload, expanded=2)
    st.download_button(
        "Download JSON",
        data=to_json_bytes(payload),
        file_name=filename,
        mime="application/json",
        type="primary",
        use_container_width=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    options = render_sidebar()

    registry = get_registry()
    health = registry.health()
    hero(model_count=len(health.models), offline=health.offline)

    # --- 01 upload ---------------------------------------------------------
    section_label("01", "Upload", "PDF · text-based or scanned with OCR")

    upload_key = f"uploader::{st.session_state.get('upload_epoch', 0)}"
    uploaded = st.file_uploader(
        "Resume PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key=upload_key,
        label_visibility="collapsed",
    )

    demo_col, reset_col, _ = st.columns([1.1, 1, 3.4])
    with demo_col:
        load_demo = st.button("Use demo resume", use_container_width=True)
    with reset_col:
        if st.button("Reset", use_container_width=True):
            st.session_state["upload_epoch"] = st.session_state.get("upload_epoch", 0) + 1
            st.session_state.pop("last_fingerprint", None)
            st.cache_data.clear()
            st.rerun()

    file_bytes: bytes | None = None
    filename = "resume.pdf"

    if load_demo:
        file_bytes = build_demo_pdf()
        filename = "aarav_mehta_resume.pdf"
        st.session_state["last_fingerprint"] = _digest(file_bytes)
    elif uploaded is not None:
        file_bytes = uploaded.getvalue()
        filename = uploaded.name
        st.session_state["last_fingerprint"] = _digest(file_bytes)

    if file_bytes is None:
        if settings.demo_mode:
            notice("info", "Demo mode is enabled", "Click “Use demo resume” to explore with sample data.")
        footer()
        return

    # --- parse -------------------------------------------------------------
    try:
        validate_upload(filename, file_bytes)
    except ResumeParserError as exc:
        notice("error", exc.message, exc.hint or "")
        footer()
        return

    fingerprint = _digest(file_bytes)
    started = time.perf_counter()

    with st.spinner("Reading PDF and extracting structure…"):
        try:
            extracted = extract_resume(file_bytes, filename)
        except ResumeParserError as exc:
            notice("error", exc.message, exc.hint or "")
            footer()
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected extraction failure for %s", filename)
            notice("error", "Unexpected error while parsing", str(exc))
            if settings.debug:
                st.code(traceback.format_exc())
            footer()
            return

    resume = extracted["resume"]
    document = extracted["document"]
    text = extracted["text"]

    # Context used by the summariser's deterministic fallback.
    st.session_state[f"summary_context::{fingerprint}"] = {
        "name": resume.get("name", ""),
        "headline": resume.get("headline", ""),
        "years": resume.get("experience", {}).get("years", 0),
        "companies": resume.get("experience", {}).get("companies", []),
        "skills": [s["name"] for s in resume.get("skills", [])],
        "education": resume.get("education", {}).get("highest_level", ""),
    }

    summary: dict | None = None
    if options["use_summary"]:
        with st.spinner("Summarising with BART…" + ("" if registry.status["summarizer"].loaded else " (first run loads ~1.6 GB, please wait)")):
            try:
                summary = summarise_resume(text, fingerprint)
                if options["bullets"] != settings.summary_bullets:
                    summary["bullets"] = summary["bullets"][: options["bullets"]]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Summarisation failed: %s", exc)
                notice("warn", "Summariser unavailable", "Showing extraction results only.")

    elapsed = time.perf_counter() - started

    # --- notices -----------------------------------------------------------
    if document.get("is_scanned"):
        notice("info", "Scanned PDF", "No text layer was found; enable OCR in the sidebar for best results.")
    if document.get("truncated"):
        notice("warn", "Document truncated", f"Only the first {settings.max_resume_chars:,} characters were parsed.")
    if document.get("ocr_used"):
        notice("success", "OCR applied", "Text was recovered from page images — verify the fields below.")
    if document.get("page_count", 0) > settings.max_pdf_pages:
        notice(
            "warn",
            "Page limit reached",
            f"Parsed {settings.max_pdf_pages} of {document['page_count']} pages.",
        )

    # --- 02 results --------------------------------------------------------
    section_label("02", "Results", f"{elapsed:.1f}s · {filename}")

    export_payload = build_export(
        {k: v for k, v in resume.items() if k != "raw_text"},
        summary={
            **(summary or {}),
            "bullets": (summary or {}).get("bullets", []),
        },
        source={
            "filename": filename,
            "bytes": len(file_bytes),
            "sha256_16": fingerprint,
            **document,
        },
        meta={
            "elapsed_seconds": round(elapsed, 2),
            "settings": {
                "ner": options["use_ner"],
                "summarizer": options["use_summary"],
                "bullets": options["bullets"],
                "ocr": options["ocr"],
            },
        },
    )

    download_col, caption_col = st.columns([1, 3])
    with download_col:
        st.download_button(
            "Download JSON",
            data=to_json_bytes(export_payload),
            file_name=download_filename(filename),
            mime="application/json",
            type="primary",
            use_container_width=True,
        )
    with caption_col:
        st.markdown(
            f'<div style="padding-top:.55rem;font-family:var(--mono);font-size:.72rem;color:#616a7e">'
            f"sha256 {fingerprint} · schema v{export_payload['schema_version']} · "
            f"{len(to_json_bytes(export_payload)) / 1024:.0f} KB</div>",
            unsafe_allow_html=True,
        )

    tab_overview, tab_skills, tab_match, tab_json = st.tabs(
        ["Overview", "Skills", "Job match", "JSON"]
    )

    with tab_overview:
        render_overview(resume, document, summary)

    with tab_skills:
        render_skills(resume)

    with tab_match:
        render_match(resume, text, fingerprint)
        if st.session_state.get(f"match::{fingerprint}"):
            export_payload["job_match"] = st.session_state[f"match::{fingerprint}"]

    with tab_json:
        if st.session_state.get(f"match::{fingerprint}"):
            export_payload["job_match"] = st.session_state[f"match::{fingerprint}"]
        with st.expander("Raw extracted text", expanded=False):
            st.text_area(
                "PDF text",
                value=text,
                height=260,
                label_visibility="collapsed",
                disabled=True,
            )
        render_json(export_payload, download_filename(filename))

    footer()


if __name__ == "__main__":
    main()
