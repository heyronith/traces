"""Frozen Q1 runtime / batching / determinism specification (F5).

Enforcement helpers refuse silent dtype/attention fallbacks. Image wheel-lock
production is prepared here; building the image is a separate ops step.

Determinism controls (B2) must be applied **before CUDA initialization** when
possible: set ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the process environment
prior to importing torch CUDA contexts / creating CUDA tensors.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from traces.q1 import FROZEN_PROTOCOL_VERSION, FROZEN_SCIENTIFIC_COMMIT

CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"

DIRECT_PINS_FILE = "TRACES_Q1_execution_direct_pins_v1_2.txt"
DIRECT_PINS_SHA256 = (
    "4a90dd7133719a751400c456db615e726a245d9a34073b7fb64243bf2a212048"
)

REQUIRED_TORCH_DTYPE = torch.bfloat16
REQUIRED_ATTN_IMPLEMENTATION = "eager"
REQUIRED_USE_CACHE_SCORING = False


@dataclass(frozen=True)
class BatchingPaddingPolicy:
    batch_size: int = 1
    padding: str = "none_variable_length_unpadded_single_sequence"
    pad_side: str | None = None
    pad_token_id: int | None = None
    identical_for_exploration_and_confirmation: bool = True

    def policy_id(self) -> str:
        return (
            f"bs{self.batch_size}|{self.padding}|side={self.pad_side}|"
            f"pad_id={self.pad_token_id}"
        )


DEFAULT_BATCHING = BatchingPaddingPolicy()


@dataclass(frozen=True)
class DeterminismSettings:
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    matmul_precision: str = "highest"  # torch.set_float32_matmul_precision
    inference_mode: bool = True
    grad_enabled: bool = False
    use_deterministic_algorithms: bool = True
    cublas_workspace_config: str = CUBLAS_WORKSPACE_CONFIG_VALUE


DEFAULT_DETERMINISM = DeterminismSettings()


def ensure_cublas_workspace_env(
    value: str = CUBLAS_WORKSPACE_CONFIG_VALUE,
    *,
    overwrite: bool = False,
) -> str:
    """Set CUBLAS_WORKSPACE_CONFIG before CUDA init when unset (or overwrite)."""
    key = "CUBLAS_WORKSPACE_CONFIG"
    if overwrite or key not in os.environ:
        os.environ[key] = value
    return os.environ[key]


def assert_bf16_dtype(dtype: torch.dtype) -> None:
    if dtype != REQUIRED_TORCH_DTYPE:
        raise RuntimeError(
            f"Q1 requires torch_dtype=torch.bfloat16; refusing {dtype}"
        )


def assert_eager_attention(attn_implementation: str | None) -> None:
    if attn_implementation != REQUIRED_ATTN_IMPLEMENTATION:
        raise RuntimeError(
            f"Q1 requires attn_implementation='eager'; refusing "
            f"{attn_implementation!r} (no silent fallback)"
        )


def assert_use_cache_false(use_cache: bool) -> None:
    if use_cache is not False:
        raise RuntimeError(
            "Q1 scoring/teacher-forced passes require use_cache=False"
        )


def q1_model_load_kwargs(
    *,
    dtype: torch.dtype = REQUIRED_TORCH_DTYPE,
    attn_implementation: str = REQUIRED_ATTN_IMPLEMENTATION,
) -> dict[str, Any]:
    """Canonical kwargs for scientific loads; validates before return."""
    assert_bf16_dtype(dtype)
    assert_eager_attention(attn_implementation)
    return {
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
    }


def verify_direct_pins_file(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    path = root / DIRECT_PINS_FILE
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    ok = digest == DIRECT_PINS_SHA256
    return {
        "path": str(path),
        "sha256": digest,
        "expected_sha256": DIRECT_PINS_SHA256,
        "match": ok,
        "protocol_version": FROZEN_PROTOCOL_VERSION,
        "frozen_commit": FROZEN_SCIENTIFIC_COMMIT,
    }


def apply_determinism_settings(settings: DeterminismSettings = DEFAULT_DETERMINISM) -> dict[str, Any]:
    """Apply process-level determinism knobs (CPU-safe; no model load).

    Prefer calling ``ensure_cublas_workspace_env`` before CUDA is initialized.
    """
    ensure_cublas_workspace_env(settings.cublas_workspace_config)
    torch.set_grad_enabled(settings.grad_enabled)
    if settings.use_deterministic_algorithms and hasattr(
        torch, "use_deterministic_algorithms"
    ):
        # warn_only=True keeps CPU CI usable when an op lacks a deterministic path
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = settings.cudnn_deterministic
        torch.backends.cudnn.benchmark = settings.cudnn_benchmark
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = settings.allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(settings.matmul_precision)
    return asdict(settings)


def write_runtime_spec_document(
    out_path: Path | str,
    *,
    batching: BatchingPaddingPolicy = DEFAULT_BATCHING,
    determinism: DeterminismSettings = DEFAULT_DETERMINISM,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "artifact": "TRACES_Q1_runtime_spec",
        "protocol_version": FROZEN_PROTOCOL_VERSION,
        "frozen_commit": FROZEN_SCIENTIFIC_COMMIT,
        "dtype": "bfloat16",
        "attn_implementation": REQUIRED_ATTN_IMPLEMENTATION,
        "use_cache_scoring": REQUIRED_USE_CACHE_SCORING,
        "batching_padding": asdict(batching),
        "determinism": asdict(determinism),
        "direct_pins_file": DIRECT_PINS_FILE,
        "direct_pins_sha256": DIRECT_PINS_SHA256,
        "transitive_lock_status": (
            "prepared_procedure_only — produce wheel-hashed freeze from built image "
            "before GPU execution"
        ),
        "extra": extra or {},
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path
