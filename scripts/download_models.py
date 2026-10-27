#!/usr/bin/env python3
"""Pre-download the transformer models so the app can run fully offline.

Usage
-----
    python scripts/download_models.py                  # both models
    python scripts/download_models.py --ner-only
    python scripts/download_models.py --summarizer-only
    python scripts/download_models.py --verify         # load + smoke test
    python scripts/download_models.py --check          # report only

Weights land in ``MODEL_CACHE_DIR`` using a flat, HF-independent layout::

    models/weights/dslim__bert-base-NER/
    models/weights/facebook__bart-large-cnn/

which means the folder can simply be rsync'd to an air-gapped machine and used
with ``OFFLINE_MODE=true``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Allow running as `python scripts/download_models.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.download_models")

REQUIRED_FILES = ("config.json",)
TOKENIZER_FILES = ("tokenizer.json", "vocab.txt", "spiece.model", "tokenizer.model")
MIN_WEIGHT_BYTES = 1_000_000


def _dir_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / (1024 * 1024):.0f} MB"


def check(model_id: str) -> bool:
    """True when a *usable* copy of ``model_id`` exists on disk.

    A snapshot can be interrupted after ``config.json`` lands, so we also
    require a real weight file and a tokenizer.
    """
    target = settings.resolve_model_dir(model_id)
    if not target.is_dir():
        return False

    has_config = all((target / name).exists() for name in REQUIRED_FILES)
    has_weights = any(
        candidate.is_file() and candidate.stat().st_size >= MIN_WEIGHT_BYTES
        for pattern in ("*.safetensors", "*.bin")
        for candidate in target.glob(pattern)
    )
    has_tokenizer = any((target / name).exists() for name in TOKENIZER_FILES)
    return has_config and has_weights and has_tokenizer


def _allow_patterns(repo_id: str) -> list[str] | None:
    """Only fetch what inference needs.

    Repos usually ship the same weights three times (safetensors, ``.bin``,
    ONNX). Listing the repo first lets us keep the safetensors copy and skip
    the rest, halving the download. If the listing fails (private mirror, no
    network) we fall back to ``None`` and prune the duplicates afterwards.
    """
    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(repo_id)
    except Exception:  # noqa: BLE001 - best effort only
        return None

    has_safetensors = any(name.endswith(".safetensors") for name in files)
    skipped_dirs = ("onnx/", "coreml/", "openvino/", "tf/", "flax/", "rust/")
    heavy_suffixes = (".bin", ".h5", ".msgpack", ".ot", ".onnx", ".tflite", ".mlmodel")

    allow = [
        name
        for name in files
        if not name.startswith((".git", *skipped_dirs))
        and not (has_safetensors and name.endswith(heavy_suffixes))
    ]
    return allow or None


def _prune(target: Path) -> None:
    """Drop duplicate weight formats and download metadata."""
    has_safetensors = any(p.stat().st_size for p in target.glob("*.safetensors"))
    if has_safetensors:
        for pattern in ("pytorch_model*.bin", "tf_model*.h5", "flax_model*.msgpack", "rust_model*.ot"):
            for stale in target.glob(pattern):
                stale.unlink(missing_ok=True)
    for folder in ("onnx", "coreml", ".cache"):
        shutil.rmtree(target / folder, ignore_errors=True)


def download(model_id: str, *, force: bool = False) -> Path:
    """Fetch ``model_id`` into the local cache directory."""
    target = settings.resolve_model_dir(model_id)
    if check(model_id) and not force:
        logger.info("✓ %s already present (%s)", model_id, _dir_size(target))
        return target

    if target.exists() and force:
        logger.info("Removing existing %s", target)
        shutil.rmtree(target, ignore_errors=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is missing. Install requirements first.")
        raise SystemExit(1)

    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", model_id)
    started = time.perf_counter()
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            allow_patterns=_allow_patterns(model_id),
            # Skip the multi-GB training-only artifacts.
            ignore_patterns=[
                "*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite",
                "tf_model.h5", "onnx/*", "coreml/*", "openvino/*",
            ],
            resume_download=True,
            max_workers=4,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed for %s: %s", model_id, exc)
        raise SystemExit(1)

    _prune(target)

    elapsed = time.perf_counter() - started
    logger.info("✓ %s ready in %.0fs (%s)", model_id, elapsed, _dir_size(target))
    return target


def verify(model_id: str, *, kind: str) -> bool:
    """Load the model through transformers and run a tiny inference."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        logger.error("Cannot verify: %s", exc)
        return False

    source = str(settings.resolve_model_dir(model_id))
    try:
        if kind == "ner":
            pipe = pipeline("ner", model=source, tokenizer=source, aggregation_strategy="simple")
            out = pipe("John Doe works at Acme Corp in Berlin.")
            logger.info("NER smoke test returned %d entities", len(out or []))
        else:
            pipe = pipeline("summarization", model=source, tokenizer=source)
            out = pipe(
                "The candidate is a senior backend engineer with eight years of "
                "experience building Python and Kubernetes services.",
                max_length=48,
                min_length=12,
                do_sample=False,
            )
            text = (out[0].get("summary_text", "") if out else "").strip()
            if not text:
                raise RuntimeError("empty summary")
            logger.info("Summariser smoke test: %s", text[:90])
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Verification failed for %s: %s", model_id, exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ner-only", action="store_true", help="download only the NER model")
    parser.add_argument("--summarizer-only", action="store_true", help="download only the summariser")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--check", action="store_true", help="report cache status and exit")
    parser.add_argument("--verify", action="store_true", help="load each model and run a smoke test")
    args = parser.parse_args()

    wanted: list[tuple[str, str]] = []
    if not args.summarizer_only:
        wanted.append(("ner", settings.ner_model_id))
    if not args.ner_only:
        wanted.append(("summarizer", settings.summarizer_model_id))

    logger.info("Cache directory: %s", settings.model_cache_dir)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        ok = True
        for kind, model_id in wanted:
            present = check(model_id)
            ok = ok and present
            mark = "✓" if present else "✗"
            size = f" ({_dir_size(settings.resolve_model_dir(model_id))})" if present else ""
            logger.info("%s %-10s %s%s", mark, kind, model_id, size)
        return 0 if ok else 1

    failures = 0
    for kind, model_id in wanted:
        download(model_id, force=args.force)
        if args.verify and not verify(model_id, kind=kind):
            failures += 1

    if failures:
        logger.error("%d model(s) failed verification", failures)
        return 1

    logger.info("All models ready. Set OFFLINE_MODE=true to run without network.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
