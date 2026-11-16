"""Durable scientific observation persistence (B1-C11).

Persists raw ``PromptConditionRecord`` and F3 ``GenerationRecord`` artifacts
with SHA-256 digests so analysis can be reconstructed from on-disk observations.

Never persists model objects, activation caches, or credentials.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FORBIDDEN_PERSIST_KEYS = frozenset(
    {
        "model",
        "tokenizer",
        "_bank",
        "_runner",
        "_prompts",
        "caches",
        "o_proj_input_by_layer",
        "residual_by_layer",
        "password",
        "token",
        "api_key",
        "secret",
        "credential",
    }
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            key = str(k)
            if key in FORBIDDEN_PERSIST_KEYS or key.startswith("_"):
                continue
            out[key] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_to_jsonable(x) for x in obj)
    # Refuse opaque objects that could be models/caches
    raise TypeError(
        f"refusing to persist non-JSON scientific observation of type {type(obj).__name__}"
    )


def write_hashed_json(path: Path | str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write canonical JSON and return path + sha256 + byte size."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    p.write_bytes(raw)
    digest = _sha256_bytes(raw)
    return {
        "path": str(p),
        "sha256": digest,
        "bytes": len(raw),
    }


def persist_unit_scientific_observations(
    out_dir: Path | str,
    *,
    model_unit: str,
    family: str,
    role: str,
    condition_records: Sequence[Any],
    generation_records: Sequence[Any],
    load_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist per-unit raw observations; return artifact digests (B1-C11)."""
    root = Path(out_dir) / "scientific_observations" / _safe_unit_name(model_unit)
    root.mkdir(parents=True, exist_ok=True)

    # Strip forbidden keys from provenance (model/tokenizer must never land on disk)
    safe_prov = {
        k: v
        for k, v in dict(load_provenance).items()
        if k not in FORBIDDEN_PERSIST_KEYS and not str(k).startswith("_")
    }

    cond_payload = {
        "artifact_type": "q1_prompt_condition_records_v1",
        "model_unit": model_unit,
        "family": family,
        "role": role,
        "n_records": len(condition_records),
        "records": [_to_jsonable(r) for r in condition_records],
        "persisted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    gen_payload = {
        "artifact_type": "q1_f3_generation_records_v1",
        "model_unit": model_unit,
        "family": family,
        "role": role,
        "n_records": len(generation_records),
        "records": [_to_jsonable(r) for r in generation_records],
        "persisted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    prov_payload = {
        "artifact_type": "q1_unit_observation_provenance_v1",
        "model_unit": model_unit,
        "family": family,
        "role": role,
        "load_provenance": _to_jsonable(safe_prov),
        "persisted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    cond_meta = write_hashed_json(root / "condition_records.json", cond_payload)
    gen_meta = write_hashed_json(root / "generation_records.json", gen_payload)
    prov_meta = write_hashed_json(root / "observation_provenance.json", prov_payload)

    manifest = {
        "artifact_type": "q1_unit_observation_manifest_v1",
        "model_unit": model_unit,
        "family": family,
        "role": role,
        "artifacts": {
            "condition_records": cond_meta,
            "generation_records": gen_meta,
            "observation_provenance": prov_meta,
        },
        "persisted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    man_meta = write_hashed_json(root / "manifest.json", manifest)
    return {
        "ok": True,
        "model_unit": model_unit,
        "dir": str(root),
        "manifest": man_meta,
        "artifacts": manifest["artifacts"],
        "n_condition_records": len(condition_records),
        "n_generation_records": len(generation_records),
    }


def load_unit_observation_manifest(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    raw = p.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    digest = _sha256_bytes(raw)
    return {"payload": payload, "sha256": digest, "path": str(p), "bytes": len(raw)}


def verify_observation_artifact_integrity(
    artifact_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash on-disk bytes and compare to recorded digest."""
    path = Path(str(artifact_meta["path"]))
    expected = str(artifact_meta["sha256"])
    if not path.is_file():
        return {"ok": False, "failures": [f"missing artifact {path}"]}
    actual = _sha256_bytes(path.read_bytes())
    if actual != expected:
        return {
            "ok": False,
            "failures": [f"sha256 mismatch {path}: {actual} != {expected}"],
            "expected": expected,
            "actual": actual,
        }
    return {"ok": True, "failures": [], "sha256": actual, "path": str(path)}


def reconstruct_condition_records_from_artifact(
    path: Path | str,
) -> list[dict[str, Any]]:
    """Load persisted condition records for analysis reconstruction."""
    payload = json.loads(Path(path).read_text())
    if str(payload.get("artifact_type", "")) != "q1_prompt_condition_records_v1":
        raise ValueError("not a condition-records artifact")
    recs = payload.get("records")
    if not isinstance(recs, list):
        raise ValueError("condition records missing")
    return list(recs)


def _safe_unit_name(model_unit: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in model_unit)
