"""Abstractive + extractive resume summarisation.

The transformer (``facebook/bart-large-cnn``) writes fluent prose for each
chunk. Because an LLM-free pipeline cannot be asked for "exactly three
bullets", we treat the model output as high-quality *candidates* and rank them
together with sentences from the resume itself (MMR over resume keywords).
That guarantees three diverse, factual bullets even on long documents.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from models.registry import ModelRegistry, get_registry
from utils.config import settings
from utils.logger import get_logger
from utils.text_utils import (
    build_fallback_bullets,
    chunk_text,
    normalize_text,
    select_bullets,
    split_sentences,
)

logger = get_logger("models.summarizer")

_MAX_PARAGRAPH_CANDIDATES = 120
# A certification line is true but rarely the most useful thing to lead with.
_CREDENTIAL_RE = re.compile(
    r"\b(certified|certification|certificate|licen[cs]ed|licen[cs]e)\b", re.I
)
# Contact/header lines are extracted separately; they make poor bullets.
_CONTACT_RE = re.compile(
    r"@|https?://|www\.|\blinkedin\b|\bgithub\.com\b|\+?\d[\d\s().\-]{8,}\d", re.I
)


@dataclass
class SummaryResult:
    bullets: list[str] = field(default_factory=list)
    model: str = settings.summarizer_model_id
    strategy: str = "abstractive+extractive"
    degraded: bool = False
    latency_ms: int = 0
    chunks: int = 0

    def to_dict(self) -> dict:
        return {
            "bullets": self.bullets,
            "model": self.model,
            "strategy": self.strategy,
            "degraded": self.degraded,
            "latency_ms": self.latency_ms,
            "chunks": self.chunks,
        }


class ResumeSummarizer:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or get_registry()

    # ------------------------------------------------------------------ API
    def summarize(
        self,
        text: str,
        *,
        bullets: int | None = None,
        context: dict | None = None,
    ) -> SummaryResult:
        """Return ``N`` ranked bullet points describing the candidate."""
        started = time.perf_counter()
        # ``bullets or default`` would silently turn an explicit 0 into the
        # default instead of clamping it to one.
        requested = settings.summary_bullets if bullets is None else bullets
        bullet_count = max(1, min(requested, 6))
        cleaned = normalize_text(text)
        if not cleaned:
            raise ValueError("Cannot summarise an empty document")

        model_sentences, chunks = self._abstractive(cleaned)

        # Extractive candidates: resume sentences, capped for scoring speed.
        original = [
            sentence
            for sentence in split_sentences(cleaned, min_chars=35)
            if not _CONTACT_RE.search(sentence) and len(sentence.split()) >= 6
        ][:_MAX_PARAGRAPH_CANDIDATES]

        candidates: list[str] = list(model_sentences) + original
        weights: list[float] = [1.45] * len(model_sentences) + [
            0.7 if _CREDENTIAL_RE.search(sentence) else 1.0 for sentence in original
        ]

        selected = select_bullets(
            candidates, cleaned, limit=bullet_count, weights=weights
        )

        degraded = not model_sentences
        strategy = "abstractive+extractive" if model_sentences else "extractive"

        if len(selected) < bullet_count:
            fallback = build_fallback_bullets(
                name=(context or {}).get("name", ""),
                headline=(context or {}).get("headline", ""),
                years=float((context or {}).get("years", 0) or 0),
                companies=(context or {}).get("companies", []) or [],
                skills=(context or {}).get("skills", []) or [],
                education=(context or {}).get("education", "") or "",
            )
            for item in fallback:
                if len(selected) >= bullet_count:
                    break
                if item.lower() not in {s.lower() for s in selected}:
                    selected.append(item)

        if not selected:
            selected = ["No summarisable content was found in this resume."]
            degraded = True
            strategy = "heuristic"

        return SummaryResult(
            bullets=selected[:bullet_count],
            strategy=strategy,
            degraded=degraded,
            latency_ms=int((time.perf_counter() - started) * 1000),
            chunks=chunks,
        )

    # ------------------------------------------------------------ internals
    def _abstractive(self, text: str) -> tuple[list[str], int]:
        summarizer = self.registry.get_summarizer()
        if summarizer is None:
            return [], 0

        budget = settings.summary_max_input_chars * max(1, settings.summary_max_chunks)
        chunks = chunk_text(text[:budget], settings.summary_max_input_chars)
        chunks = chunks[: max(1, settings.summary_max_chunks)]
        sentences: list[str] = []

        for index, chunk in enumerate(chunks):
            if len(chunk.split()) < 25:  # too short to summarise meaningfully
                continue
            try:
                with self.registry.inference_lock:
                    output = summarizer(
                        chunk,
                        max_length=settings.summary_max_length,
                        min_length=min(settings.summary_min_length, settings.summary_max_length - 5),
                        do_sample=False,
                        num_beams=max(1, settings.summary_num_beams),
                        length_penalty=2.0,
                        no_repeat_ngram_size=3,
                        truncation=True,
                    )
                produced = (output[0].get("summary_text", "") if output else "").strip()
                if produced:
                    sentences.extend(split_sentences(produced, min_chars=30))
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the page
                logger.warning("Summarisation failed for chunk %s: %s", index, exc)

        # De-duplicate model output while keeping order.
        unique: list[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            key = sentence.lower()[:120]
            if key not in seen:
                seen.add(key)
                unique.append(sentence)
        return unique, len(chunks)


_summarizer: ResumeSummarizer | None = None


def get_summarizer(registry: ModelRegistry | None = None) -> ResumeSummarizer:
    global _summarizer
    if _summarizer is None or registry is not None:
        _summarizer = ResumeSummarizer(registry)
    return _summarizer
