"""Immutable scientific pilot pins (calibrated revisions + code identity).

Fail closed: never load unresolved ``main`` for Stage 1 scientific units.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

# Previously calibrated dense-LoRA Gemma-2-2b revisions (bounded calibration).
PINNED_BASE_MODEL_ID = "google/gemma-2-2b"
PINNED_ADAPTER_REPO = "interpretable-finetuning/topklora-gemma-2-2b-dense-lora"
PINNED_BASE_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
PINNED_ADAPTER_REVISION = "6303a74fc832a34d584999364817268aefd9623b"

# Empty-set D instrumentation (pre-screening only; not a scientific metric knob).
EMPTY_SET_D_ABS_TOLERANCE = 1e-4


def resolve_executed_git_sha(*, explicit: str | None = None) -> str:
    """Return the exact Git SHA that must bind the run identity."""
    if explicit:
        sha = str(explicit).strip()
        if len(sha) < 7:
            raise RuntimeError(f"executed_git_sha too short: {sha!r}")
        return sha
    env = os.environ.get("TRACES_GIT_SHA") or os.environ.get("GIT_SHA")
    if env:
        return str(env).strip()
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
        ).strip()
    except Exception as e:
        raise RuntimeError(
            "Cannot resolve executed Git SHA for run identity; set TRACES_GIT_SHA "
            "or pass executed_git_sha explicitly. Refusing unbound scientific run."
        ) from e
    if not sha:
        raise RuntimeError("Empty git SHA from rev-parse; refusing unbound run")
    return sha


def pinned_revision_identity() -> dict[str, str]:
    return {
        "base_model_id": PINNED_BASE_MODEL_ID,
        "adapter_repo": PINNED_ADAPTER_REPO,
        "pinned_base_revision": PINNED_BASE_REVISION,
        "pinned_adapter_revision": PINNED_ADAPTER_REVISION,
    }


def assert_loaded_matches_pins(provenance: dict[str, Any], *, context: str = "") -> None:
    base = str(provenance.get("base_model_revision") or "")
    adapter = str(provenance.get("adapter_revision") or "")
    if base in {"synthetic", "unspecified"} and adapter in {"synthetic", "unspecified"}:
        # CPU synthetic path
        return
    if base != PINNED_BASE_REVISION:
        raise RuntimeError(
            f"Base revision drift ({context}): loaded={base!r} "
            f"pinned={PINNED_BASE_REVISION!r}"
        )
    if adapter != PINNED_ADAPTER_REVISION:
        raise RuntimeError(
            f"Adapter revision drift ({context}): loaded={adapter!r} "
            f"pinned={PINNED_ADAPTER_REVISION!r}"
        )
