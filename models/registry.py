"""Lazy, thread-safe access to the transformer models.

Design goals
------------
* **Offline first** – if a model was pre-downloaded into ``MODEL_CACHE_DIR`` we
  load it from disk and never touch the network.
* **Graceful degradation** – a missing/broken model returns ``None`` instead of
  crashing the app; callers fall back to heuristics.
* **Single instance** – Streamlit reruns on every interaction, so the heavy
  pipelines are loaded exactly once per process.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.config import settings
from utils.logger import get_logger

logger = get_logger("models.registry")

# Must be set before transformers/tokenizers spin up thread pools.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if settings.offline_mode:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:  # pragma: no cover - exercised implicitly
    import torch
    from transformers import (
        AutoModel,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )

    _TRANSFORMERS_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - import-time env issues are broad
    torch = None  # type: ignore[assignment]
    AutoModel = AutoTokenizer = hf_pipeline = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""


@dataclass
class ModelStatus:
    name: str
    loaded: bool = False
    error: str | None = None
    source: str = ""
    load_seconds: float | None = None


@dataclass
class RegistryHealth:
    transformers_available: bool
    device: str
    offline: bool
    models: dict[str, ModelStatus] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "transformers_available": self.transformers_available,
            "device": self.device,
            "offline": self.offline,
            "models": {
                key: {
                    "loaded": status.loaded,
                    "error": status.error,
                    "source": status.source,
                    "load_seconds": status.load_seconds,
                }
                for key, status in self.models.items()
            },
        }


class ModelRegistry:
    """Process-wide cache for the NER, summarisation and embedding models."""

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self.inference_lock = threading.RLock()
        self._ner = None
        self._summarizer = None
        self._encoder = None
        self._tokenizer = None
        self.status: dict[str, ModelStatus] = {
            "ner": ModelStatus("ner"),
            "summarizer": ModelStatus("summarizer"),
            "encoder": ModelStatus("encoder"),
        }
        self._device = self._resolve_device()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _resolve_device() -> str:
        if settings.device.lower() in {"cpu", "cuda"}:
            return settings.device.lower()
        if _TRANSFORMERS_AVAILABLE and torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def device_index(self) -> int:
        return 0 if self._device == "cuda" else -1

    def _resolve_source(self, model_id: str) -> tuple[str, bool]:
        """Return ``(model_path_or_id, is_local)``."""
        local_dir = settings.resolve_model_dir(model_id)
        if local_dir.is_dir() and (local_dir / "config.json").exists():
            return str(local_dir), True

        if settings.offline_mode:
            raise FileNotFoundError(
                f"Offline mode is on but '{model_id}' is not present in "
                f"{local_dir}. Run `python scripts/download_models.py` first."
            )

        # Fall back to the shared Hugging Face cache before hitting the network.
        shared = Path.home() / ".cache" / "huggingface" / "hub"
        if shared.is_dir():
            logger.info("Loading %s from the Hugging Face cache", model_id)
        return model_id, False

    def _record(self, key: str, *, success: bool, source: str, error: str = "", seconds: float | None = None) -> None:
        status = self.status[key]
        status.loaded = success
        status.source = source
        status.error = error or None
        status.load_seconds = round(seconds, 2) if seconds is not None else None

    # ------------------------------------------------------------------- NER
    def get_ner(self, *, required: bool = False):
        """Return a token-classification pipeline or ``None`` in degraded mode."""
        if not settings.enable_ner or not _TRANSFORMERS_AVAILABLE:
            if required:
                raise RuntimeError(_IMPORT_ERROR or "NER is disabled (ENABLE_NER=false)")
            return None
        if self._ner is not None:
            return self._ner

        with self._load_lock:
            if self._ner is not None:
                return self._ner
            started = time.perf_counter()
            source = settings.ner_model_id
            try:
                source, local = self._resolve_source(settings.ner_model_id)
                logger.info("Loading NER model from %s (%s)", source, "local" if local else "hub")
                self._ner = hf_pipeline(
                    "ner",
                    model=source,
                    tokenizer=source,
                    aggregation_strategy="simple",
                    device=self.device_index,
                )
                self._record("ner", success=True, source=source, seconds=time.perf_counter() - started)
                logger.info("NER model ready in %.1fs", time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001
                logger.warning("NER model unavailable: %s", exc)
                self._record("ner", success=False, source=source, error=str(exc), seconds=time.perf_counter() - started)
                if required:
                    raise
                return None
            return self._ner

    # ----------------------------------------------------------- Summariser
    def get_summarizer(self, *, required: bool = False):
        if not settings.enable_summarizer or not _TRANSFORMERS_AVAILABLE:
            if required:
                raise RuntimeError(_IMPORT_ERROR or "Summariser is disabled (ENABLE_SUMMARIZER=false)")
            return None
        if self._summarizer is not None:
            return self._summarizer

        with self._load_lock:
            if self._summarizer is not None:
                return self._summarizer
            started = time.perf_counter()
            source = settings.summarizer_model_id
            try:
                source, local = self._resolve_source(settings.summarizer_model_id)
                logger.info("Loading summariser from %s (%s)", source, "local" if local else "hub")
                self._summarizer = hf_pipeline(
                    "summarization",
                    model=source,
                    tokenizer=source,
                    device=self.device_index,
                )
                self._record("summarizer", success=True, source=source, seconds=time.perf_counter() - started)
                logger.info("Summariser ready in %.1fs", time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Summariser unavailable: %s", exc)
                self._record("summarizer", success=False, source=source, error=str(exc), seconds=time.perf_counter() - started)
                if required:
                    raise
                return None
            return self._summarizer

    # -------------------------------------------------------------- Encoder
    def get_encoder(self, *, required: bool = False):
        """Mean-pooled BERT encoder reused for semantic JD similarity."""
        if (
            not settings.enable_embeddings
            or not settings.enable_ner
            or not _TRANSFORMERS_AVAILABLE
        ):
            return None
        if self._encoder is not None and self._tokenizer is not None:
            return self._encoder, self._tokenizer

        with self._load_lock:
            if self._encoder is not None and self._tokenizer is not None:
                return self._encoder, self._tokenizer
            started = time.perf_counter()
            source = settings.ner_model_id
            try:
                source, _local = self._resolve_source(settings.ner_model_id)
                tokenizer = AutoTokenizer.from_pretrained(source)
                model = AutoModel.from_pretrained(source)
                model.to(self._device)
                model.eval()
                self._tokenizer = tokenizer
                self._encoder = model
                self._record("encoder", success=True, source=source, seconds=time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Encoder unavailable: %s", exc)
                self._record("encoder", success=False, source=source, error=str(exc), seconds=time.perf_counter() - started)
                if required:
                    raise
                return None
            return self._encoder, self._tokenizer

    # ------------------------------------------------------------ Inference
    def embed_texts(self, texts: list[str], *, max_length: int = 512) -> np.ndarray | None:
        """Return L2-normalised sentence embeddings, or ``None``."""
        if not texts:
            return None
        loaded = self.get_encoder()
        if loaded is None:
            return None
        model, tokenizer = loaded
        try:
            with self.inference_lock:
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                with torch.no_grad():  # type: ignore[union-attr]
                    output = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                summed = (output * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                embeddings = (summed / counts).cpu().numpy()
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / np.clip(norms, 1e-9, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding inference failed: %s", exc)
            return None

    def warm_up(self) -> RegistryHealth:
        """Best-effort load of every model; used by the download/health scripts."""
        if settings.enable_ner:
            self.get_ner()
        if settings.enable_summarizer:
            self.get_summarizer()
        if settings.enable_embeddings:
            self.get_encoder()
        return self.health()

    def health(self) -> RegistryHealth:
        return RegistryHealth(
            transformers_available=_TRANSFORMERS_AVAILABLE,
            device=self.device,
            offline=settings.offline_mode,
            models=dict(self.status),
        )


_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry
