"""Shared fixtures for the test suite.

The tests must run on a machine with **no model weights and no network**, so
every fixture injects :class:`OfflineRegistry`, a stand-in that reports all
models as unavailable. That exercises the graceful-degradation paths which are
exactly what breaks in production.
"""

from __future__ import annotations

import threading

import pytest

from utils.config import settings
from utils.sample import build_demo_pdf

RESUME_TEXT = """\
Aarav Mehta
Senior Machine Learning Engineer
aarav.mehta@example.com | +91 98200 41122 | Bengaluru, India
linkedin.com/in/aaravmehta | github.com/aaravmehta

Professional Summary
Machine learning engineer with 8 years of experience shipping production NLP
and recommendation systems. Led a four-person team that rebuilt the search
ranking stack on PyTorch, cutting p95 latency by 42%.

Experience
Senior Machine Learning Engineer | Nimbus Analytics | Jan 2022 - Present
- Owned the ranking pipeline (PyTorch, Airflow, Kubernetes) and cut inference cost by 38%.
- Shipped a retrieval-augmented generation assistant used by 12k daily users.

Machine Learning Engineer | Vertex Labs | Jun 2019 - Dec 2021
- Built a BERT document classifier at 94% F1, replacing a manual triage queue.
- Deployed 20+ models to AWS with Terraform and GitHub Actions.

Data Scientist | Bluewave Systems | Jul 2017 - May 2019
- Developed churn models in scikit-learn that protected $2.4M of revenue.

Education
Master of Technology in Computer Science - Indian Institute of Technology, Bombay, 2017
Bachelor of Technology in Information Technology - VIT University, 2015

Technical Skills
Languages: Python, SQL, TypeScript, Bash
ML: PyTorch, scikit-learn, Hugging Face Transformers, NLP, RAG
Data: Pandas, NumPy, Snowflake, dbt, Airflow, Apache Spark, PostgreSQL
Cloud: AWS, Docker, Kubernetes, Terraform, GitHub Actions, Prometheus

Certifications
- AWS Certified Machine Learning - Specialty (2023)
"""

JOB_DESCRIPTION = """\
We are hiring a Senior Machine Learning Engineer to own our ranking stack.

Requirements:
- 5+ years of experience building production ML systems.
- Strong Python and SQL; experience with PyTorch or TensorFlow.
- Hands-on Kubernetes and AWS deployment experience.
- Bachelor's degree in Computer Science or a related field.
- Experience with Scala is a plus.
"""

NURSING_JD = """\
Registered nurse required for a busy hospital ward. Responsibilities include
patient triage, administering medication, wound care and coordinating with
physicians. Must hold a valid nursing licence and have two years of bedside
experience. Excellent communication skills are essential.
"""


class OfflineRegistry:
    """ModelRegistry look-alike with every model permanently unavailable."""

    def __init__(self) -> None:
        self.inference_lock = threading.RLock()
        self.device = "cpu"

    def get_ner(self):
        return None

    def get_summarizer(self):
        return None

    def get_encoder(self):
        return None

    def embed_texts(self, texts, **_kwargs):
        return None

    def warm_up(self):
        return None

    def health(self):
        from models.registry import ModelStatus, RegistryHealth

        return RegistryHealth(
            transformers_available=False,
            device="cpu",
            offline=settings.offline_mode,
            models={
                "ner": ModelStatus("ner"),
                "summarizer": ModelStatus("summarizer"),
                "encoder": ModelStatus("encoder"),
            },
        )


@pytest.fixture(scope="session")
def resume_text() -> str:
    return RESUME_TEXT


@pytest.fixture(scope="session")
def job_description() -> str:
    return JOB_DESCRIPTION


@pytest.fixture(scope="session")
def nursing_jd() -> str:
    return NURSING_JD


@pytest.fixture(scope="session")
def stub_registry() -> OfflineRegistry:
    return OfflineRegistry()


@pytest.fixture(scope="session")
def demo_pdf_bytes() -> bytes:
    """A real, text-based PDF generated at test time (no fixture blobs)."""
    return build_demo_pdf()


@pytest.fixture(scope="session")
def parsed_resume(resume_text, stub_registry):
    """Parsed record with NER disabled, i.e. the pure-rules path."""
    from parsers.ner_parser import ResumeParser

    parser = ResumeParser(stub_registry, use_ner=False)
    return parser.parse(resume_text, run_ner=False)
