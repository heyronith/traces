"""Reproducibility helpers: hashing, fingerprints, parameter digests."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return sha256_text(payload)


def hash_prompt_set(prompts: Sequence[Mapping[str, Any]]) -> str:
    """Stable hash of a prompt collection (order-sensitive)."""
    rows = []
    for p in prompts:
        rows.append(
            {
                "instruction_id": p.get("instruction_id"),
                "tag": p.get("tag"),
                "instruction": p.get("instruction"),
                "rendered": p.get("rendered"),
            }
        )
    return sha256_json(rows)


def freeze_splits(
    discovery: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    held_out: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create and hash frozen discovery / validation / held-out splits.

    Stage 0 may call this for future experiments; it does not build the
    scientific held-out set itself.
    """
    return {
        "discovery": {"n": len(discovery), "hash": hash_prompt_set(discovery)},
        "validation": {"n": len(validation), "hash": hash_prompt_set(validation)},
        "held_out": {"n": len(held_out), "hash": hash_prompt_set(held_out)},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parameter_fingerprint(model: Any) -> str:
    """Sampled parameter fingerprint (NOT a full weight checksum).

    Hashes names, shapes, and a small deterministic sample of values from each
    parameter tensor. Useful for detecting accidental persistent modification
    in Stage 0, but it does **not** verify every model weight.
    """
    h = hashlib.sha256()
    named = list(model.named_parameters())
    for name, param in named:
        h.update(name.encode("utf-8"))
        h.update(str(tuple(param.shape)).encode("utf-8"))
        data = param.detach().float().cpu().contiguous().view(-1)
        n = data.numel()
        if n == 0:
            continue
        idxs = [0, n // 2, n - 1]
        if n >= 16:
            idxs.extend(range(0, 8))
            idxs.extend(range(n // 2, n // 2 + 8))
            idxs.extend(range(n - 8, n))
        sample = data[sorted(set(i for i in idxs if 0 <= i < n))]
        h.update(sample.numpy().tobytes())
    return h.hexdigest()


def full_parameter_checksum(model: Any) -> str:
    """Full checksum of all parameters (expensive; used in tiny unit tests)."""
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def lora_adapter_full_checksum(model: Any) -> str:
    """Complete checksum over LoRA adapter parameters only (A/B matrices).

    Much smaller than the full base model; strengthens integrity checks without
    hashing every base weight.
    """
    h = hashlib.sha256()
    found = 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        found += 1
        h.update(name.encode("utf-8"))
        h.update(param.detach().float().cpu().contiguous().numpy().tobytes())
    if found == 0:
        raise ValueError("No lora_* parameters found for adapter checksum")
    h.update(f"n_tensors={found}".encode("utf-8"))
    return h.hexdigest()


def config_hash(cfg: Mapping[str, Any]) -> str:
    return sha256_json(cfg)


@dataclass
class EnvironmentRecord:
    python_version: str = field(default_factory=lambda: sys.version)
    platform: str = field(default_factory=platform.platform)
    packages: dict[str, str] = field(default_factory=dict)
    torch_cuda_available: bool | None = None
    torch_version: str | None = None
    cuda_device_name: str | None = None
    git_commit: str | None = None
    recorded_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_environment(git_commit: str | None = None) -> EnvironmentRecord:
    packages: dict[str, str] = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "numpy",
        "yaml",
        "modal",
    ):
        try:
            mod = __import__(name if name != "yaml" else "yaml")
            packages[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            packages[name] = "not_installed"

    torch_cuda = None
    torch_version = packages.get("torch")
    cuda_name = None
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        if torch_cuda:
            cuda_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return EnvironmentRecord(
        packages=packages,
        torch_cuda_available=torch_cuda,
        torch_version=torch_version,
        cuda_device_name=cuda_name,
        git_commit=git_commit,
    )


def redact_secrets(text: str, secret_values: Iterable[str] | None = None) -> str:
    """Remove known secret values from logs. Never logs token contents."""
    out = text
    for s in secret_values or ():
        if s and len(s) >= 8:
            out = out.replace(s, "[REDACTED]")
    # Common HF token prefixes
    import re

    out = re.sub(r"hf_[A-Za-z0-9]{10,}", "[REDACTED_HF_TOKEN]", out)
    out = re.sub(
        r"(Authorization:\s*Bearer\s+)\S+",
        r"\1[REDACTED]",
        out,
        flags=re.IGNORECASE,
    )
    return out
