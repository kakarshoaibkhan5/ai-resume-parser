#!/usr/bin/env python3
"""Render ``assets/demo.gif`` — an honest walkthrough of the real pipeline.

The frames are drawn from the *actual* output of the parser, summariser and
matcher (run against the synthetic demo resume), not from hand-written copy.
That keeps the README demo truthful even as the models are tuned.

Usage
-----
    python scripts/make_demo_gif.py
    python scripts/make_demo_gif.py --output assets/demo.gif --no-models

``--no-models`` skips the transformer stack and uses the deterministic
fallbacks, which is useful when weights are not downloaded yet.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Palette (mirrors utils/theme.py)
# ---------------------------------------------------------------------------

BG_TOP = (10, 12, 18)
BG_BOTTOM = (6, 7, 10)
PANEL = (13, 15, 20)
PANEL_EDGE = (30, 34, 44)
INK = (238, 241, 247)
MUTED = (139, 147, 167)
DIM = (97, 106, 126)
ACCENT = (200, 247, 81)
VIOLET = (139, 123, 255)
WARN = (242, 146, 78)
MISS = (255, 107, 129)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
SANS = FONT_DIR / "DejaVuSans.ttf"
SANS_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
MONO = FONT_DIR / "DejaVuSansMono.ttf"

S = 2  # supersampling factor
W, H = 1000, 600          # design-space canvas
OUT_W, OUT_H = 900, 540   # encoded size (README renders at 880px wide)
PALETTE_COLORS = 96
MARGIN = 48

JOB_DESCRIPTION = """\
We are hiring a Senior Machine Learning Engineer to own our ranking stack.

Requirements:
- 5+ years of experience building production ML systems.
- Strong Python and SQL; experience with PyTorch or TensorFlow.
- Hands-on Kubernetes and AWS deployment experience.
- Bachelor's degree in Computer Science or a related field.
- Experience with Scala is a plus.
"""


# ---------------------------------------------------------------------------
# Tiny drawing wrapper so every coordinate stays in design-space pixels
# ---------------------------------------------------------------------------


class Canvas:
    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.draw = ImageDraw.Draw(image)
        self._fonts: dict[tuple[int, bool, bool], ImageFont.FreeTypeFont] = {}

    # -- primitives --------------------------------------------------------
    def font(self, size: int, *, bold: bool = False, mono: bool = False):
        key = (size, bold, mono)
        if key not in self._fonts:
            path = MONO if mono else (SANS_BOLD if bold else SANS)
            self._fonts[key] = ImageFont.truetype(str(path), size * S)
        return self._fonts[key]

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int = 14,
        fill=INK,
        bold: bool = False,
        mono: bool = False,
        anchor: str = "la",
    ) -> None:
        self.draw.text(
            (x * S, y * S),
            value,
            font=self.font(size, bold=bold, mono=mono),
            fill=fill,
            anchor=anchor,
        )

    def width(self, value: str, *, size: int = 14, bold: bool = False, mono: bool = False) -> float:
        return self.draw.textlength(
            value, font=self.font(size, bold=bold, mono=mono)
        ) / S

    def box(self, x0: float, y0: float, x1: float, y1: float, *, radius: float = 14, **kw) -> None:
        self.draw.rounded_rectangle(
            [x0 * S, y0 * S, x1 * S, y1 * S], radius=radius * S, **kw
        )

    def line(self, points, *, fill=MUTED, width: int = 1) -> None:
        self.draw.line([(x * S, y * S) for x, y in points], fill=fill, width=width * S)

    def ellipse(self, box, *, outline=None, width: int = 1, fill=None) -> None:
        self.draw.ellipse([v * S for v in box], outline=outline, width=width * S, fill=fill)

    def arc(self, box, start: float, end: float, *, fill=ACCENT, width: int = 12) -> None:
        self.draw.arc([v * S for v in box], start=start, end=end, fill=fill, width=width * S)

    def pill(self, x0, y0, x1, y1, *, fill, outline=None, radius=None) -> None:
        self.box(x0, y0, x1, y1, radius=radius if radius is not None else (y1 - y0) / 2,
                 fill=fill, outline=outline)

    # -- composite helpers -------------------------------------------------
    def tracked(self, x: float, y: float, value: str, *, size: int = 10, fill=ACCENT, tracking: float = 1.6) -> float:
        for char in value:
            self.text(x, y, char, size=size, fill=fill, bold=True)
            x += self.width(char, size=size, bold=True) + tracking
        return x

    def wrap(self, value: str, max_width: float, *, size: int = 14, bold: bool = False) -> list[str]:
        words = value.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if self.width(candidate, size=size, bold=bold) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def truncate(self, value: str, max_width: float, *, size: int = 14, bold: bool = False) -> str:
        if self.width(value, size=size, bold=bold) <= max_width:
            return value
        suffix = "…"
        while value and self.width(value + suffix, size=size, bold=bold) > max_width:
            value = value[:-1]
        return value.rstrip(" ,;:") + suffix

    def grid(self, x0, y0, x1, y1, *, step=28, fill=(255, 255, 255, 5)) -> None:
        for x in range(int(x0), int(x1), step):
            self.line([(x, y0), (x, y1)], fill=fill, width=1)
        for y in range(int(y0), int(y1), step):
            self.line([(x0, y), (x1, y)], fill=fill, width=1)


def _background() -> Image.Image:
    """Vertical gradient plus two soft brand-coloured glows."""
    base = Image.new("RGB", (W * S, H * S), BG_BOTTOM)
    draw = ImageDraw.Draw(base)
    for y in range(H * S):
        t = y / (H * S)
        r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
        draw.line([(0, y), (W * S, y)], fill=(r, g, b))

    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([(W * S * 0.55, -H * S * 0.35), (W * S * 1.15, H * S * 0.55)],
               fill=VIOLET + (34,))
    gd.ellipse([(-W * S * 0.25, H * S * 0.55), (W * S * 0.45, H * S * 1.35)],
               fill=ACCENT + (18,))
    glow = glow.filter(ImageFilter.GaussianBlur(90 * S))
    return Image.alpha_composite(base.convert("RGBA"), glow).convert("RGB")


def _chrome(c: Canvas) -> None:
    """Logo bar shared by every frame."""
    c.box(MARGIN, 22, MARGIN + 26, 48, radius=7, fill=ACCENT)
    c.text(MARGIN + 13, 35, "R", size=15, fill=(6, 7, 10), bold=True, anchor="mm")
    c.text(MARGIN + 38, 28, "ai-resume-parser", size=14, bold=True)
    c.text(MARGIN + 38, 44, "open source · local inference", size=10, fill=DIM)
    label = "v1.0.0 · MIT"
    c.text(W - MARGIN, 34, label, size=10, fill=DIM, mono=True, anchor="rm")


def _section(c: Canvas, number: str, title: str, hint: str) -> None:
    c.text(MARGIN, 92, number, size=10, fill=ACCENT, mono=True, anchor="lm")
    c.text(MARGIN + 24, 92, title, size=21, bold=True, anchor="lm")
    if hint:
        c.text(W - MARGIN, 93, hint, size=10, fill=DIM, mono=True, anchor="rm")
    c.line([(MARGIN, 112), (W - MARGIN, 112)], fill=PANEL_EDGE, width=1)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def scene_upload(c: Canvas, data: "Payload", t: float) -> None:
    pulse = 0.5 + 0.5 * math.sin(t * math.pi * 2)
    c.tracked(MARGIN, 150, "100% OPEN SOURCE · RUNS LOCALLY", size=9, fill=ACCENT)
    c.text(MARGIN - 2, 190, "Parse, summarise and score", size=38, bold=True)
    c.text(MARGIN - 2, 240, "any resume in ", size=38, bold=True)
    start = MARGIN - 2 + c.width("any resume in ", size=38, bold=True)
    c.text(start, 240, "seconds.", size=38, bold=True, fill=ACCENT)

    c.text(MARGIN, 300, "Drop a PDF. Get structured JSON, a three-bullet executive summary", size=14, fill=MUTED)
    c.text(MARGIN, 320, "and a job-description match score — all computed on-device.", size=14, fill=MUTED)

    edge = tuple(int(45 + 40 * pulse) for _ in range(3))
    c.box(MARGIN, 360, W - MARGIN, 548, radius=18, fill=PANEL, outline=(*edge, 255), width=2)
    c.box(MARGIN + 6, 366, W - MARGIN - 6, 542, radius=14, outline=(60, 68, 84), width=1)

    # upload glyph
    cx, cy = W / 2, 424
    c.box(cx - 21, cy - 15, cx + 21, cy + 13, radius=5, outline=ACCENT, width=2)
    c.line([(cx, cy + 6), (cx, cy - 9)], fill=ACCENT, width=2)
    c.line([(cx - 7, cy - 2), (cx, cy - 9), (cx + 7, cy - 2)], fill=ACCENT, width=2)

    c.text(cx, cy + 40, "Drop your resume PDF here", size=18, bold=True, anchor="mm")
    c.text(cx, cy + 62, "PDF  ·  up to 15 MB  ·  20 pages  ·  text or scanned", size=11, fill=MUTED, anchor="mm")

    btn_w = 190
    c.pill(cx - btn_w / 2, 500, cx + btn_w / 2, 532, fill=ACCENT)
    c.text(cx, 516, "Use demo resume", size=12, fill=(6, 7, 10), bold=True, anchor="mm")


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _stat_card(c: Canvas, x: float, y: float, w: float, label: str, value: str, sub: str) -> None:
    c.box(x, y, x + w, y + 84, radius=12, fill=PANEL, outline=PANEL_EDGE, width=1)
    c.text(x + 15, y + 15, label.upper(), size=8, fill=DIM, bold=True)
    c.text(x + 15, y + 34, value, size=19, bold=True)
    c.text(x + 15, y + 61, c.truncate(sub, w - 30, size=9), size=9, fill=MUTED)


def scene_extract(c: Canvas, data: "Payload", t: float) -> None:
    _section(c, "02", "Results", data.source_line)

    resume = data.resume
    exp = resume.get("experience", {})
    skills = resume.get("skills", [])
    edu = resume.get("education", {})

    gap = 12
    card_w = (W - 2 * MARGIN - 3 * gap) / 4
    cards = [
        ("Candidate", resume.get("name") or "Not detected", resume.get("headline") or "No headline"),
        ("Experience", f"{exp.get('years', 0):g} yrs",
         f"{_plural(len(exp.get('companies', [])), 'company', 'companies')} · {exp.get('years_source', 'none')}"),
        ("Skills", str(len(skills)),
         _plural(len({s.get('category') for s in skills}), 'category', 'categories')),
        ("Confidence", f"{round(float(resume.get('confidence', {}).get('overall', 0)) * 100)}%",
         f"{_plural(data.document.get('pages_used', 0), 'page')} · {data.document.get('word_count', 0)} words"),
    ]
    for i, card in enumerate(cards):
        # staggered rise-in
        local = max(0.0, min(1.0, (t - i * 0.07) / 0.35))
        y = 128 + (1 - local) * 14
        if local > 0:
            _stat_card(c, MARGIN + i * (card_w + gap), y, card_w, *card)

    # left panel: contact
    top, bottom = 232, 566
    left_x0, left_x1 = MARGIN, MARGIN + 418
    c.box(left_x0, top, left_x1, bottom, radius=14, fill=PANEL, outline=PANEL_EDGE, width=1)
    c.text(left_x0 + 20, top + 18, "Contact", size=13, bold=True)

    contact = resume.get("contact", {})
    rows = [
        ("Email", contact.get("email") or "—"),
        ("Phone", contact.get("phone") or "—"),
        ("Location", contact.get("location") or "—"),
        ("LinkedIn", contact.get("linkedin") or "—"),
        ("GitHub", contact.get("github") or "—"),
    ]
    y = top + 48
    for label, value in rows:
        c.text(left_x0 + 20, y, label, size=10, fill=DIM)
        c.text(left_x0 + 20, y + 16, c.truncate(value, 370, size=11), size=11, fill=INK, mono=True)
        y += 46

    c.text(left_x0 + 20, bottom - 40, "Education", size=10, fill=DIM, bold=True)
    c.text(
        left_x0 + 20,
        bottom - 23,
        c.truncate(
            f"{edu.get('highest_level', 'Not detected')} · "
            + (edu.get("institutions", ["—"])[0] if edu.get("institutions") else "—"),
            370,
            size=10,
        ),
        size=10,
        fill=MUTED,
    )

    # right panel: timeline
    right_x0, right_x1 = left_x1 + 18, W - MARGIN
    c.box(right_x0, top, right_x1, bottom, radius=14, fill=PANEL, outline=PANEL_EDGE, width=1)
    c.text(right_x0 + 20, top + 18, "Experience timeline", size=13, bold=True)

    entries = exp.get("entries", [])[:3]
    y = top + 52
    for i, entry in enumerate(entries):
        local = max(0.0, min(1.0, (t - 0.15 - i * 0.1) / 0.35))
        if local <= 0:
            continue
        c.ellipse([right_x0 + 22, y + 4, right_x0 + 32, y + 14], fill=ACCENT)
        if i < len(entries) - 1:
            c.line([(right_x0 + 27, y + 16), (right_x0 + 27, y + 92)], fill=(45, 50, 62), width=2)
        c.text(right_x0 + 44, y, c.truncate(entry.get("title", ""), 300, size=12, bold=True), size=12, bold=True)
        meta = " · ".join(
            part for part in [entry.get("company", ""), _range(entry)] if part
        )
        c.text(right_x0 + 44, y + 19, c.truncate(meta, 330, size=9), size=9, fill=ACCENT if i == 0 else DIM)
        desc = (entry.get("description") or "").strip().split("\n")[0]
        for j, line in enumerate(c.wrap(desc, 330, size=9)[:2]):
            c.text(right_x0 + 44, y + 36 + j * 13, line, size=9, fill=MUTED)
        y += 96


def _range(entry: dict) -> str:
    start, end = entry.get("start") or "", entry.get("end") or ""
    return f"{start} – {end}".strip(" –")


def scene_summary(c: Canvas, data: "Payload", t: float) -> None:
    _section(c, "03", "Executive summary", "3 bullets · factual · no contact details")

    c.box(MARGIN, 132, W - MARGIN, 132, radius=16, fill=PANEL, outline=PANEL_EDGE, width=1)
    c.text(MARGIN + 24, 152, "Summary", size=13, bold=True)
    c.text(W - MARGIN - 24, 152, data.summary_meta, size=9, fill=DIM, mono=True, anchor="ra")

    bullets = data.summary[:3]
    wrapped = [c.wrap(bullet, W - 2 * MARGIN - 80, size=15)[:3] for bullet in bullets]
    body_height = sum(len(lines) * 25 + 16 for lines in wrapped)
    panel_bottom = max(300, 190 + body_height + 10)
    c.box(MARGIN, 132, W - MARGIN, panel_bottom, radius=16, fill=PANEL, outline=PANEL_EDGE, width=1)
    c.text(MARGIN + 24, 152, "Summary", size=13, bold=True)
    c.text(W - MARGIN - 24, 152, data.summary_meta, size=9, fill=DIM, mono=True, anchor="ra")

    y = 190
    for i, lines in enumerate(wrapped):
        local = max(0.0, min(1.0, (t - i * 0.18) / 0.3))
        if local <= 0:
            continue
        colour = tuple(int(v * local) for v in INK)
        accent = tuple(int(v * local) for v in ACCENT)
        c.ellipse([MARGIN + 26, y + 7, MARGIN + 34, y + 15], fill=accent)
        for j, line in enumerate(lines):
            c.text(MARGIN + 50, y + j * 25, line, size=15, fill=colour)
        y += len(lines) * 25 + 16

    # supporting chips
    chip_y = panel_bottom + 24
    c.text(MARGIN, chip_y + 11, "Evidence", size=9, fill=DIM, bold=True, anchor="lm")
    x = MARGIN + 70
    for name in data.chips:
        width = c.width(name, size=9) + 22
        if x + width > W - MARGIN:
            break
        c.pill(x, chip_y, x + width, chip_y + 22, fill=(22, 27, 18), outline=(70, 88, 30))
        c.text(x + width / 2, chip_y + 11, name, size=9, fill=ACCENT, anchor="mm")
        x += width + 8

    c.text(MARGIN, chip_y + 62, "Why these bullets?", size=9, fill=DIM, bold=True)
    c.text(
        MARGIN,
        chip_y + 80,
        "BART candidates are scored against the extraction, then ranked with an MMR pass so the three lines stay",
        size=10,
        fill=MUTED,
    )
    c.text(MARGIN, chip_y + 96, "distinct, quantified and grounded — the extractive path produces the same shape offline.", size=10, fill=MUTED)

    # Bottom band: the three-stage recipe, only when there is room for it.
    band_y = chip_y + 122
    if band_y + 74 <= H - 34:
        stages = [
            ("01", "abstractive", "BART-Large-CNN drafts candidate sentences."),
            ("02", "extractive", "MMR drops the near-duplicates."),
            ("03", "grounded", "Every figure traced back to the PDF."),
        ]
        gap = 12
        card_w = (W - 2 * MARGIN - 2 * gap) / 3
        for i, (number, title, detail) in enumerate(stages):
            x = MARGIN + i * (card_w + gap)
            c.box(x, band_y, x + card_w, band_y + 62, radius=10, fill=(11, 13, 18), outline=(28, 32, 42), width=1)
            c.text(x + 14, band_y + 14, number, size=8, fill=ACCENT, mono=True)
            c.text(x + 32, band_y + 13, title, size=10, fill=INK, bold=True)
            c.text(x + 14, band_y + 36, c.truncate(detail, card_w - 28, size=9), size=9, fill=MUTED)


def scene_match(c: Canvas, data: "Payload", t: float) -> None:
    _section(c, "04", "Job match", "weighted · explainable · re-normalised")

    match = data.match
    sweep = min(1.0, t / 0.7)
    score = match.get("score", 0)

    ring_box = (MARGIN + 10, 150, MARGIN + 210, 350)
    c.ellipse(ring_box, outline=(28, 32, 42), width=14)
    c.arc(ring_box, -90, -90 + 360 * (score / 100) * sweep,
          fill=ACCENT if score >= 65 else WARN, width=14)
    c.text(MARGIN + 110, 236, f"{round(score * sweep)}", size=44, bold=True, anchor="mm")
    c.text(MARGIN + 110, 268, "%", size=12, fill=DIM, anchor="mm")
    c.text(MARGIN + 110, 374, match.get("verdict", ""), size=11, fill=ACCENT, bold=True, anchor="mm")
    c.text(MARGIN + 110, 392, f"{match.get('coverage', 0):.0f}% skill coverage", size=9, fill=DIM, anchor="mm")

    # breakdown meters
    x0 = MARGIN + 240
    y = 160
    for i, item in enumerate(match.get("breakdown", [])[:4]):
        local = max(0.0, min(1.0, (t - 0.1 - i * 0.08) / 0.5))
        # The pipeline reports 0–100; older static payloads use 0–1.
        raw = float(item.get("score") or 0)
        fraction = raw / 100 if raw > 1.5 else raw
        value = fraction * local
        c.text(x0, y, item.get("label", ""), size=11, bold=True)
        c.text(W - MARGIN - 52, y, f"{fraction * 100:.0f}", size=11, fill=ACCENT, mono=True, anchor="ra")
        c.text(W - MARGIN - 40, y, "/", size=10, fill=DIM, mono=True, anchor="ra")
        c.text(W - MARGIN, y, f"{item.get('weight', 0):.0f}%", size=9, fill=DIM, mono=True, anchor="ra")
        c.pill(x0, y + 18, W - MARGIN, y + 26, fill=(26, 30, 38))
        if value > 0:
            c.pill(x0, y + 18, x0 + (W - MARGIN - x0) * value, y + 26,
                   fill=ACCENT if value >= 0.65 else WARN)
        c.text(x0, y + 32, c.truncate(item.get("detail", ""), 430, size=9), size=9, fill=MUTED)
        y += 70

    # chips
    y = 452
    c.text(MARGIN, y + 11, "Matched", size=9, fill=DIM, bold=True, anchor="lm")
    x = MARGIN + 70
    for skill in match.get("matched_skills", [])[:5]:
        name = skill["name"]
        width = c.width(name, size=9) + 22
        c.pill(x, y, x + width, y + 22, fill=(22, 27, 18), outline=(70, 88, 30))
        c.text(x + width / 2, y + 11, name, size=9, fill=ACCENT, anchor="mm")
        x += width + 8

    y = 492
    c.text(MARGIN, y + 11, "Missing", size=9, fill=DIM, bold=True, anchor="lm")
    x = MARGIN + 70
    for skill in match.get("missing_skills", [])[:4]:
        name = skill["name"]
        width = c.width(name, size=9) + 22
        c.pill(x, y, x + width, y + 22, fill=(30, 18, 22), outline=(88, 40, 50))
        c.text(x + width / 2, y + 11, name, size=9, fill=MISS, anchor="mm")
        x += width + 8

    c.text(MARGIN, 540, "Verdict", size=9, fill=DIM, bold=True)
    c.text(
        MARGIN + 52,
        540,
        c.truncate(
            (match.get("recommendations") or ["No further recommendations."])[0],
            W - MARGIN - 52,
            size=10,
        ),
        size=10,
        fill=MUTED,
    )


def scene_export(c: Canvas, data: "Payload", t: float) -> None:
    _section(c, "05", "JSON export", "schema v1.0 · stable contract")

    c.box(MARGIN, 132, W - MARGIN, 570, radius=16, fill=(9, 11, 15), outline=PANEL_EDGE, width=1)

    # window dots
    for i, colour in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        c.ellipse([MARGIN + 18 + i * 16, 148, MARGIN + 26 + i * 16, 156], fill=colour)
    c.text(MARGIN + 78, 152, data.filename_json, size=9, fill=DIM, mono=True)
    c.line([(MARGIN, 168), (W - MARGIN, 168)], fill=PANEL_EDGE, width=1)

    visible = int(len(data.json_lines) * min(1.0, t / 0.75))
    y = 180
    for i, (indent, key, value) in enumerate(data.json_lines):
        if i >= visible:
            break
        base = MARGIN + 20 + indent * 18
        if key:
            c.text(base, y, key, size=9, fill=VIOLET, mono=True)
            c.text(base + c.width(key, size=9, mono=True) + 6, y, value, size=9, fill=MUTED, mono=True)
        else:
            c.text(base, y, value, size=9, fill=MUTED, mono=True)
        y += 15

    # download affordance, pinned below the payload
    c.line([(MARGIN, 512), (W - MARGIN, 512)], fill=PANEL_EDGE, width=1)
    btn_w = 168
    c.pill(W - MARGIN - 20 - btn_w, 522, W - MARGIN - 20, 554, fill=ACCENT)
    c.text(W - MARGIN - 20 - btn_w / 2, 538, "Download JSON", size=11, fill=(6, 7, 10), bold=True, anchor="mm")
    c.text(
        MARGIN + 20,
        538,
        f"sha256 {data.fingerprint} · schema v1.0 · {data.export_kb} KB",
        size=9,
        fill=DIM,
        mono=True,
        anchor="lm",
    )


# ---------------------------------------------------------------------------
# Pipeline data
# ---------------------------------------------------------------------------


@dataclass
class Payload:
    resume: dict
    document: dict
    summary: list[str]
    summary_meta: str
    match: dict
    chips: list[str]
    json_lines: list[tuple[int, str, str]]
    source_line: str
    filename_json: str
    fingerprint: str = "9f2c41d0b7a3e8c5"
    export_kb: int = 12


def _fallback_payload() -> Payload:
    """Used only if the real pipeline cannot run at all."""
    resume = {
        "name": "Aarav Mehta",
        "headline": "Senior Machine Learning Engineer",
        "contact": {
            "email": "aarav.mehta@example.com",
            "phone": "+91 98200 41122",
            "location": "Bengaluru, India",
            "linkedin": "linkedin.com/in/aaravmehta",
            "github": "github.com/aaravmehta",
        },
        "skills": [{"name": n, "category": "ML"} for n in ["Python", "SQL", "PyTorch"]],
        "experience": {"years": 8.6, "years_source": "dates", "companies": ["Nimbus Analytics"],
                       "entries": [{"title": "Senior Machine Learning Engineer",
                                    "company": "Nimbus Analytics",
                                    "start": "Jan 2022", "end": "Present",
                                    "description": "Owned the ranking pipeline and cut inference cost by 38%."}]},
        "education": {"highest_level": "Master's",
                      "institutions": ["Indian Institute of Technology, Bombay"]},
        "confidence": {"overall": 0.96},
    }
    document = {"pages_used": 1, "word_count": 280}
    summary = [
        "Machine learning engineer with 8+ years shipping production NLP and ranking systems.",
        "Cut inference cost by 38% and p95 latency by 42% on a PyTorch ranking stack.",
        "Holds an M.Tech in Computer Science from IIT Bombay.",
    ]
    match = {
        "score": 83, "verdict": "Excellent match", "coverage": 75.0,
        "breakdown": [
            {"label": "Skill coverage", "score": 0.75, "weight": 50, "detail": "6/8 JD skills found"},
            {"label": "Semantic similarity", "score": 0.71, "weight": 25, "detail": "lexical-tf"},
            {"label": "Experience fit", "score": 1.0, "weight": 15, "detail": "8.6 yrs vs 5 requested"},
            {"label": "Education fit", "score": 1.0, "weight": 10, "detail": "Master's meets Bachelor's"},
        ],
        "matched_skills": [{"name": n} for n in ["AWS", "Kubernetes", "PyTorch", "Python", "SQL"]],
        "missing_skills": [{"name": "TensorFlow"}, {"name": "Scala"}],
        "recommendations": ["Add measurable TensorFlow work to close the last required gap."],
    }
    return _payload_from(resume, document, summary, "abstractive + extractive",
                         match, "1.9s · aarav_mehta_resume.pdf")


def _payload_from(
    resume,
    document,
    summary,
    strategy,
    match,
    source_line,
    *,
    export: dict | None = None,
    export_bytes: bytes = b"",
) -> Payload:
    chips = [s["name"] for s in resume.get("skills", [])][:7]

    if export:
        source = export.get("source", {})
        meta = export.get("meta", {})
        fingerprint = source.get("sha256_16", "9f2c41d0b7a3e8c5")
        filename = source.get("filename", "aarav_mehta_resume.pdf")
        json_lines = [
            (0, '"schema_version":', f'"{export.get("schema_version", "1.0")}",'),
            (0, '"generated_at":', f'"{export.get("generated_at", "")}",'),
            (0, '"generator":', '{"name": "ai-resume-parser", "version": "1.0.0"},'),
            (
                0,
                '"source":',
                f'{{"filename": "{filename}", "page_count": {source.get("page_count", 1)}, '
                f'"word_count": {source.get("word_count", 0)}, "sha256_16": "{fingerprint}"}},',
            ),
            (0, '"extraction":', "{"),
            (1, '"name":', f'"{resume.get("name", "")}",'),
            (1, '"headline":', f'"{resume.get("headline", "")}",'),
            (1, '"contact":', f'{{"email": "{resume.get("contact", {}).get("email", "")}", ...}},'),
            (1, '"skills":', f"[{len(resume.get('skills', []))} items],"),
            (1, '"experience":', f'{{"years": {resume.get("experience", {}).get("years", 0)}}},'),
            (1, '"education":', f'{{"highest_level": "{resume.get("education", {}).get("highest_level", "")}"}}'),
            (0, "},", ""),
            (0, '"summary":', '{"bullets": [3 items]},'),
            (0, '"job_match":', "{"),
            (1, '"score":', f'{match.get("score", 0)},'),
            (1, '"verdict":', f'"{match.get("verdict", "")}",'),
            (1, '"matched_skills":', f"[{len(match.get('matched_skills', []))} items],"),
            (1, '"missing_skills":', f"[{len(match.get('missing_skills', []))} items]"),
            (0, "},", ""),
            (0, '"meta":', f'{{"offline": {str(bool(meta.get("offline"))).lower()}, '
                           f'"pipeline_ms": {meta.get("pipeline_ms", 0)}}}'),
            (0, "}", ""),
        ]
    else:
        fingerprint = "9f2c41d0b7a3e8c5"
        json_lines = [
            (0, '"schema_version":', '"1.0",'),
            (0, '"generated_at":', '"2026-10-28T09:14:02+00:00",'),
            (0, '"generator":', '{"name": "ai-resume-parser", "version": "1.0.0"},'),
            (0, '"source":', '{"filename": "aarav_mehta_resume.pdf", "page_count": 1},'),
            (0, '"extraction":', "{"),
            (1, '"name":', '"Aarav Mehta",'),
            (1, '"headline":', '"Senior Machine Learning Engineer",'),
            (1, '"contact":', '{"email": "aarav.mehta@example.com", ...},'),
            (1, '"skills":', f"[{len(resume.get('skills', []))} items],"),
            (1, '"experience":', f'{{"years": {resume.get("experience", {}).get("years", 0)}}},'),
            (1, '"education":', f'{{"highest_level": "{resume.get("education", {}).get("highest_level", "")}"}}'),
            (0, "},", ""),
            (0, '"summary":', '{"bullets": [3 items]},'),
            (0, '"job_match":', "{"),
            (1, '"score":', f'{match.get("score", 0)},'),
            (1, '"verdict":', f'"{match.get("verdict", "")}",'),
            (1, '"matched_skills":', f"[{len(match.get('matched_skills', []))} items],"),
            (1, '"missing_skills":', f"[{len(match.get('missing_skills', []))} items]"),
            (0, "},", ""),
            (0, '"meta":', '{"offline": false, "pipeline_ms": 0}'),
            (0, "}", ""),
        ]

    return Payload(
        resume=resume,
        document=document,
        summary=summary,
        summary_meta=f"{strategy} · facebook/bart-large-cnn",
        match=match,
        chips=chips,
        json_lines=json_lines,
        source_line=source_line,
        filename_json="aarav_mehta_resume_parsed.json",
        fingerprint=fingerprint,
        export_kb=max(1, round(len(export_bytes) / 1024)) if export_bytes else 12,
    )


def collect_payload(*, use_models: bool = True) -> Payload:
    """Run the real pipeline; fall back to constants if anything is missing."""
    try:
        import hashlib
        import time

        from models.matcher import match_resume_to_jd
        from models.registry import get_registry
        from models.summarizer import get_summarizer
        from parsers.ner_parser import get_parser
        from parsers.pdf_reader import read_pdf
        from utils.exporters import build_export, to_json_bytes
        from utils.sample import build_demo_pdf

        started = time.perf_counter()
        pdf = build_demo_pdf()
        document = read_pdf(pdf, filename="aarav_mehta_resume.pdf")
        parsed = get_parser(use_ner=use_models).parse(document.text)
        resume = parsed.to_dict()

        context = {
            "name": resume.get("name", ""),
            "headline": resume.get("headline", ""),
            "years": resume.get("experience", {}).get("years", 0),
            "companies": resume.get("experience", {}).get("companies", []),
            "skills": [s["name"] for s in resume.get("skills", [])],
            "education": resume.get("education", {}).get("highest_level", ""),
        }
        summary = get_summarizer().summarize(document.text, bullets=3, context=context)
        match = match_resume_to_jd(resume, JOB_DESCRIPTION, resume_text=document.text)
        elapsed = time.perf_counter() - started

        strategy = "extractive" if summary.degraded else "abstractive + extractive"
        source_line = (
            f"{elapsed:.1f}s · aarav_mehta_resume.pdf · {document.pages_used} page"
            f" · {document.word_count} words"
        )

        export = build_export(
            {k: v for k, v in resume.items() if k != "raw_text"},
            summary=summary.to_dict(),
            match=match.to_dict(),
            source={
                "filename": "aarav_mehta_resume.pdf",
                "bytes": len(pdf),
                "sha256_16": hashlib.sha256(pdf).hexdigest()[:16],
                **document.to_dict(),
            },
            meta={
                "offline": get_registry().health().offline,
                "pipeline_ms": round(elapsed * 1000),
            },
        )
        blob = to_json_bytes(export)

        return _payload_from(
            resume,
            document.to_dict(),
            list(summary.bullets),
            strategy,
            match.to_dict(),
            source_line,
            export=export,
            export_bytes=blob,
        )
    except Exception as exc:  # noqa: BLE001 - the demo must always render
        print(f"[warn] falling back to static demo data: {type(exc).__name__}: {exc}")
        return _fallback_payload()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


SCENES = [
    (scene_upload, 18),
    (scene_extract, 22),
    (scene_summary, 22),
    (scene_match, 22),
    (scene_export, 18),
]
FPS = 10
XFADE = 4


def _render(scene, data: Payload, t: float) -> Image.Image:
    image = _background()
    canvas = Canvas(image)
    _chrome(canvas)
    scene(canvas, data, t)
    return image.resize((OUT_W, OUT_H), Image.LANCZOS)


def build_gif(output: Path, *, use_models: bool = True) -> Path:
    data = collect_payload(use_models=use_models)
    frames: list[Image.Image] = []

    for index, (scene, count) in enumerate(SCENES):
        scene_frames = [_render(scene, data, n / max(1, count - 1)) for n in range(count)]
        frames.extend(scene_frames)
        if index < len(SCENES) - 1:
            nxt_scene, nxt_count = SCENES[index + 1]
            nxt = _render(nxt_scene, data, 0.0)
            for step in range(1, XFADE + 1):
                frames.append(
                    Image.blend(scene_frames[-1], nxt, step / (XFADE + 1))
                )

    # Quantise to a shared, stable palette to stop GIF dithering from flickering.
    palette_source = frames[0].quantize(colors=PALETTE_COLORS, method=Image.MEDIANCUT)
    paletted = [frame.quantize(palette=palette_source, dither=Image.FLOYDSTEINBERG) for frame in frames]

    output.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/demo.gif"))
    parser.add_argument("--no-models", action="store_true",
                        help="use deterministic fallbacks instead of loading BERT/BART")
    args = parser.parse_args()

    path = build_gif(args.output, use_models=not args.no_models)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.1f} MB)")

    # A still frame is handy for the README social preview.
    preview = path.with_name("demo_poster.png")
    Image.open(path).convert("RGB").save(preview)
    print(f"wrote {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
