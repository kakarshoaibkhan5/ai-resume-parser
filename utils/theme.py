"""Visual layer: theme CSS injection and reusable HTML components.

Kept out of ``app.py`` so the UI shell stays readable and the components can be
unit-rendered in isolation. Everything here is plain HTML + CSS, no JS.
"""

from __future__ import annotations

import html
from functools import lru_cache

import streamlit as st

from utils.config import settings

# ---------------------------------------------------------------------------
# Palette — single source of truth, mirrored in assets/styles.css
# ---------------------------------------------------------------------------
ACCENT = "#c8f751"
ACCENT_DIM = "#8fc32a"
VIOLET = "#8b7bff"
INK = "#06070a"
PANEL = "#0d0f14"
MUTED = "#8b93a7"

TONE_COLORS = {
    "excellent": ACCENT,
    "strong": "#7de2a8",
    "moderate": "#f2c14e",
    "partial": "#f2924e",
    "low": "#f2686b",
}

_CSS_PATH = settings.assets_dir / "styles.css"


@lru_cache(maxsize=1)
def _css() -> str:
    try:
        return _CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def inject_theme() -> None:
    """Inject Streamlit config tweaks + the stylesheet once per rerun."""
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def hero(*, model_count: int, offline: bool) -> None:
    badge = "offline · local weights" if offline else "100% open source · runs locally"
    meta = (
        f'<span class="dot"></span>{model_count} local models'
        f'<span class="sep">/</span>PyMuPDF text layer'
        f'<span class="sep">/</span>no data leaves this machine'
    )
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-glow"></div>
          <div class="hero-inner">
            <span class="eyebrow">{esc(badge)}</span>
            <h1>Parse, summarise and score<br/>any resume in <em>seconds</em>.</h1>
            <p class="lede">
              Drop a PDF. Get structured JSON, a three-bullet executive summary and a
              job-description match score — all computed on-device with BERT and BART.
            </p>
            <div class="hero-meta">{meta}</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_label(index: str, title: str, hint: str = "") -> None:
    hint_html = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    st.markdown(
        f'<div class="section-label"><span class="num">{esc(index)}</span>'
        f'<h3>{esc(title)}</h3>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def stat_row(items: list[tuple[str, str, str]]) -> None:
    """``items`` = [(label, value, sublabel), ...]"""
    cards = "".join(
        f"""
        <div class="stat">
          <span class="stat-label">{esc(label)}</span>
          <span class="stat-value">{esc(value)}</span>
          <span class="stat-sub">{esc(sub)}</span>
        </div>"""
        for label, value, sub in items
    )
    st.markdown(f'<div class="stat-row">{cards}</div>', unsafe_allow_html=True)


def score_ring(score: int, *, tone: str = "moderate", caption: str = "match") -> str:
    """Circular gauge, returned as HTML so it can live inside a column."""
    colour = TONE_COLORS.get(tone, ACCENT)
    radius = 52
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - max(0, min(100, score)) / 100)
    return f"""
    <div class="ring-wrap">
      <svg viewBox="0 0 120 120" class="ring" role="img" aria-label="{score}% {esc(caption)}">
        <circle cx="60" cy="60" r="{radius}" class="ring-bg"/>
        <circle cx="60" cy="60" r="{radius}" class="ring-fg"
                stroke="{colour}"
                stroke-dasharray="{circumference:.2f}"
                stroke-dashoffset="{offset:.2f}"/>
        <text x="60" y="58" class="ring-value">{score}<tspan class="ring-pct">%</tspan></text>
        <text x="60" y="78" class="ring-cap">{esc(caption)}</text>
      </svg>
    </div>
    """


def chip(label: str, *, tone: str = "default", title: str = "") -> str:
    title_attr = f' title="{esc(title)}"' if title else ""
    return f'<span class="chip chip-{esc(tone)}"{title_attr}>{esc(label)}</span>'


def chip_cloud(items: list[str], *, tone: str = "default", limit: int = 60) -> str:
    if not items:
        return '<p class="empty">None detected.</p>'
    shown = items[:limit]
    extra = len(items) - len(shown)
    body = "".join(chip(item, tone=tone) for item in shown)
    if extra > 0:
        body += f'<span class="chip chip-ghost">+{extra} more</span>'
    return f'<div class="chip-cloud">{body}</div>'


def skill_chips(skills: list[dict], *, limit: int = 60) -> str:
    if not skills:
        return '<p class="empty">No known skills detected.</p>'
    parts = []
    for skill in skills[:limit]:
        name = skill.get("name", "")
        category = skill.get("category", "")
        confidence = float(skill.get("confidence", 0) or 0)
        tone = "accent" if confidence >= 0.9 else "default"
        parts.append(chip(name, tone=tone, title=f"{category} · confidence {confidence:.0%}"))
    if len(skills) > limit:
        parts.append(f'<span class="chip chip-ghost">+{len(skills) - limit} more</span>')
    return f'<div class="chip-cloud">{"".join(parts)}</div>'


def timeline(entries: list[dict]) -> str:
    if not entries:
        return '<p class="empty">No dated roles were detected.</p>'
    rows = []
    for entry in entries:
        title = entry.get("title") or "Role"
        company = entry.get("company") or ""
        start = entry.get("start") or ""
        end = entry.get("end") or ""
        months = int(entry.get("duration_months") or 0)
        duration = f"{months / 12:.1f} yrs" if months >= 12 else (f"{months} mo" if months else "")
        description = entry.get("description") or ""
        rows.append(
            f"""
            <li class="tl-item">
              <span class="tl-dot"></span>
              <div class="tl-head">
                <span class="tl-title">{esc(title)}</span>
                {f'<span class="tl-company">@ {esc(company)}</span>' if company else ''}
              </div>
              <div class="tl-meta">
                <span>{esc(start)} — {esc(end)}</span>
                {f'<span class="tl-dur">{esc(duration)}</span>' if duration else ''}
              </div>
              {f'<p class="tl-desc">{esc(description)}</p>' if description else ''}
            </li>"""
        )
    return f'<ul class="timeline">{"".join(rows)}</ul>'


def kv_rows(items: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="kv"><span class="kv-k">{esc(key)}</span>'
        f'<span class="kv-v">{esc(value)}</span></div>'
        for key, value in items
    )
    return f'<div class="kv-list">{body}</div>'


def bullets(items: list[str], *, numbered: bool = True) -> str:
    if not items:
        return '<p class="empty">No summary generated.</p>'
    rows = []
    for index, item in enumerate(items, start=1):
        marker = f'<span class="b-num">{index:02d}</span>' if numbered else '<span class="b-dot"></span>'
        rows.append(f'<li class="b-item">{marker}<span>{esc(item)}</span></li>')
    return f'<ul class="bullets">{"".join(rows)}</ul>'


def meter(label: str, value: float, *, weight: float | None = None, detail: str = "") -> str:
    pct = max(0.0, min(100.0, value))
    weight_html = f'<span class="m-weight">{weight:.0f}% weight</span>' if weight else ""
    return f"""
    <div class="meter">
      <div class="m-head">
        <span class="m-label">{esc(label)}</span>
        <span class="m-value">{pct:.0f}</span>
      </div>
      <div class="m-track"><div class="m-fill" style="width:{pct:.1f}%"></div></div>
      <div class="m-foot"><span>{esc(detail)}</span>{weight_html}</div>
    </div>
    """


def notice(kind: str, title: str, body: str = "") -> None:
    st.markdown(
        f'<div class="notice notice-{esc(kind)}"><strong>{esc(title)}</strong>'
        f'{f"<span>{esc(body)}</span>" if body else ""}</div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        """
        <footer class="app-footer">
          <span>ai-resume-parser · MIT licensed</span>
          <span class="sep">/</span>
          <span>dslim/bert-base-NER + facebook/bart-large-cnn</span>
          <span class="sep">/</span>
          <span>nothing is uploaded anywhere</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )
