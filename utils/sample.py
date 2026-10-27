"""Generate a realistic one-page demo resume as a PDF (no extra dependencies).

Used by ``DEMO_MODE`` in the UI and by ``scripts/make_sample.py`` so the README
screenshots and the demo video never depend on a real candidate's data.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from utils.config import settings

SAMPLE_TEXT = {
    "name": "Aarav Mehta",
    "headline": "Senior Machine Learning Engineer",
    "contact": "aarav.mehta@example.com  ·  +91 98200 41122  ·  Bengaluru, India",
    "links": "linkedin.com/in/aaravmehta  ·  github.com/aaravmehta",
    "summary": (
        "Machine learning engineer with 8 years of experience shipping production NLP and "
        "recommendation systems. Led a four-person team that rebuilt the search ranking "
        "stack on PyTorch, cutting p95 latency by 42% and lifting click-through by 11%."
    ),
    "experience": [
        (
            "Senior Machine Learning Engineer | Nimbus Analytics | Jan 2022 - Present",
            [
                "Owned the end-to-end ranking pipeline (PyTorch, Airflow, Kubernetes); reduced inference cost by 38%.",
                "Shipped a retrieval-augmented generation assistant used by 12k daily users across 3 products.",
                "Mentored 4 engineers and introduced model evaluation gates that caught 7 regressions pre-release.",
            ],
        ),
        (
            "Machine Learning Engineer | Vertex Labs | Jun 2019 - Dec 2021",
            [
                "Built a BERT-based document classifier at 94% F1, replacing a manual triage queue.",
                "Designed the feature store on Snowflake and dbt, cutting experiment setup from days to hours.",
                "Deployed 20+ models to AWS with Terraform and GitHub Actions.",
            ],
        ),
        (
            "Data Scientist | Bluewave Systems | Jul 2017 - May 2019",
            [
                "Developed churn models in scikit-learn that protected $2.4M of annual recurring revenue.",
                "Automated reporting with Python and Tableau, saving 15 analyst hours per week.",
            ],
        ),
    ],
    "education": [
        "Master of Technology in Computer Science — Indian Institute of Technology, Bombay, 2017",
        "Bachelor of Technology in Information Technology — VIT University, 2015",
    ],
    "skills": (
        "Languages: Python, SQL, TypeScript, Bash\n"
        "ML: PyTorch, scikit-learn, Hugging Face Transformers, NLP, RAG, LLM, XGBoost\n"
        "Data: Pandas, NumPy, Snowflake, dbt, Airflow, Apache Spark, PostgreSQL\n"
        "Cloud & DevOps: AWS, Docker, Kubernetes, Terraform, GitHub Actions, Prometheus\n"
        "Web: FastAPI, React, REST API, Docker Compose"
    ),
    "certifications": [
        "AWS Certified Machine Learning – Specialty (2023)",
        "Deep Learning Specialization — Coursera (2020)",
    ],
}


def build_demo_pdf(path: Path | None = None) -> bytes:
    """Render the demo resume to PDF bytes and optionally persist it."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    margin_x = 52
    y = 58
    accent = (0.12, 0.12, 0.12)
    muted = (0.35, 0.35, 0.38)

    def line(text: str, size: float = 10, *, gap: float = 15, color=accent, bold: bool = False) -> None:
        nonlocal y
        page.insert_text(
            (margin_x, y),
            text,
            fontsize=size,
            fontname="hebo" if bold else "helv",
            color=color,
        )
        y += gap

    line(SAMPLE_TEXT["name"], 22, gap=20, bold=True)
    line(SAMPLE_TEXT["headline"], 12, gap=17, color=(0.30, 0.42, 0.05))
    line(SAMPLE_TEXT["contact"], 9, gap=13, color=muted)
    line(SAMPLE_TEXT["links"], 9, gap=22, color=muted)

    def heading(text: str) -> None:
        nonlocal y
        y += 6
        line(text.upper(), 9.5, gap=14, bold=True)
        page.draw_line((margin_x, y - 7), (595 - margin_x, y - 7), color=(0.80, 0.80, 0.80), width=0.7)
        y += 4

    heading("Professional Summary")
    for chunk in _wrap(SAMPLE_TEXT["summary"], 96):
        line(chunk, 9.5, gap=13)
    y += 4

    heading("Experience")
    for header, bullets in SAMPLE_TEXT["experience"]:
        line(header, 10, gap=14, bold=True)
        for bullet in bullets:
            for index, chunk in enumerate(_wrap(bullet, 100)):
                prefix = "•  " if index == 0 else "    "
                line(prefix + chunk, 9, gap=12.5, color=(0.2, 0.2, 0.22))
        y += 6

    heading("Education")
    for entry in SAMPLE_TEXT["education"]:
        for chunk in _wrap(entry, 98):
            line(chunk, 9.5, gap=13)

    heading("Technical Skills")
    line(SAMPLE_TEXT["skills"], 9.5, gap=12)

    heading("Certifications")
    for entry in SAMPLE_TEXT["certifications"]:
        line("•  " + entry, 9.5, gap=13)

    data = doc.tobytes()
    doc.close()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def ensure_sample_resume() -> Path:
    """Create ``assets/sample_resume.pdf`` if it is missing; return its path."""
    path = settings.sample_resume_path
    if not path.exists():
        build_demo_pdf(path)
    return path


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
