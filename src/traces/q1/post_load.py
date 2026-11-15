"""Post-load verification for Q1 scientific model bundles (fail-closed)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from traces.q1.runtime_spec import (
    REQUIRED_ATTN_IMPLEMENTATION,
    REQUIRED_TORCH_DTYPE,
    assert_eager_attention,
)


@dataclass
class PostLoadReport:
    ok: bool
    attn_implementation: str | None
    model_float_dtypes: list[str] = field(default_factory=list)
    adapter_param_dtypes: list[str] = field(default_factory=list)
    tokenizer_class: str | None = None
    tokenizer_is_fast: bool | None = None
    base_revision: str | None = None
    adapter_revision: str | None = None
    subfolder: str | None = None
    adapter_targets: list[str] = field(default_factory=list)
    adapted_layers_inferred: list[int] = field(default_factory=list)
    weight_file_sha256: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_attn_implementation(model: Any) -> str | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    return getattr(cfg, "_attn_implementation", None) or getattr(
        cfg, "attn_implementation", None
    )


def infer_base_revision_from_model(model: Any) -> str | None:
    """Read commit hash stamped on the loaded config — not a caller echo."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    for attr in ("_commit_hash", "commit_hash", "revision"):
        v = getattr(cfg, attr, None)
        if isinstance(v, str) and len(v) >= 7:
            return v
    return None


def infer_adapter_revision_from_model(model: Any) -> str | None:
    """Read revision from PEFT config / adapter metadata when present."""
    if not hasattr(model, "peft_config"):
        return None
    peft_cfg = (
        model.peft_config.get("default")
        if isinstance(model.peft_config, dict)
        else model.peft_config
    )
    if peft_cfg is None:
        return None
    for attr in ("revision", "adapter_revision", "peft_version"):
        # peft_version is not a git SHA — skip non-sha-looking values later
        v = getattr(peft_cfg, attr, None)
        if isinstance(v, str) and len(v) >= 7 and attr != "peft_version":
            return v
    # Some PEFT builds stash hub metadata under .hub_version / path
    hub = getattr(peft_cfg, "hub_model_id", None)
    _ = hub
    return None


def _infer_adapted_layers(target_modules: Sequence[str], model: Any) -> list[int]:
    """Best-effort: layers that contain LoRA parameters."""
    layers: set[int] = set()
    for name, _p in model.named_parameters():
        if "lora_" not in name.lower():
            continue
        # ...layers.N...
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layers.add(int(parts[i + 1]))
    return sorted(layers)


def verify_post_load(
    model: Any,
    tokenizer: Any,
    *,
    expected_base_revision: str,
    expected_adapter_revision: str,
    actual_base_revision: str | None = None,
    actual_adapter_revision: str | None = None,
    actual_subfolder: str | None = None,
    expected_adapter_targets: Sequence[str] | None = None,
    expected_adapted_layers: Sequence[int] | None = None,
    expected_tokenizer_class: str | None = None,
    expected_tokenizer_is_fast: bool | None = None,
    expected_weight_file_sha256: Mapping[str, str] | None = None,
    require_eager: bool = True,
    require_bf16: bool = True,
) -> PostLoadReport:
    failures: list[str] = []
    attn = _infer_attn_implementation(model)
    if require_eager:
        try:
            assert_eager_attention(attn or "")
        except RuntimeError as e:
            failures.append(str(e))
        if attn != REQUIRED_ATTN_IMPLEMENTATION:
            failures.append(
                f"attn_implementation actual={attn!r} != {REQUIRED_ATTN_IMPLEMENTATION!r}"
            )

    float_dtypes = {
        p.dtype for p in model.parameters() if p.dtype.is_floating_point
    }
    if require_bf16:
        bad = [d for d in float_dtypes if d != torch.bfloat16]
        if bad or not float_dtypes:
            failures.append(
                f"Disallowed floating dtypes {sorted(str(d) for d in float_dtypes)}; "
                f"require exclusively {REQUIRED_TORCH_DTYPE}"
            )

    adapter_dtypes: list[str] = []
    adapter_targets: list[str] = []
    if hasattr(model, "peft_config"):
        peft_cfg = (
            model.peft_config.get("default")
            if isinstance(model.peft_config, dict)
            else None
        )
        if peft_cfg is not None:
            targets = getattr(peft_cfg, "target_modules", None)
            if targets is not None:
                adapter_targets = sorted(str(x) for x in targets)
        for name, p in model.named_parameters():
            if "lora_" in name.lower() and p.dtype.is_floating_point:
                adapter_dtypes.append(str(p.dtype))
                if require_bf16 and p.dtype != torch.bfloat16:
                    failures.append(f"Adapter param {name} dtype {p.dtype} != bfloat16")

    if expected_adapter_targets is not None:
        if sorted(expected_adapter_targets) != adapter_targets:
            failures.append(
                f"adapter targets {adapter_targets} != expected {sorted(expected_adapter_targets)}"
            )

    adapted_layers = _infer_adapted_layers(adapter_targets, model)
    if expected_adapted_layers is not None:
        exp_layers = list(expected_adapted_layers)
        if not exp_layers:
            failures.append("expected_adapted_layers is empty while required")
        elif not adapted_layers:
            failures.append(
                "adapted-layer inventory absent on model but exact layers required: "
                f"expected {exp_layers}"
            )
        elif adapted_layers != exp_layers:
            failures.append(
                f"adapted layers inferred {adapted_layers} != expected {exp_layers}"
            )

    tok_class = type(tokenizer).__name__
    is_fast = bool(getattr(tokenizer, "is_fast", False))
    if expected_tokenizer_class is not None and tok_class != expected_tokenizer_class:
        failures.append(
            f"tokenizer class {tok_class} != expected {expected_tokenizer_class}"
        )
    if (
        expected_tokenizer_is_fast is not None
        and is_fast != expected_tokenizer_is_fast
    ):
        failures.append(
            f"tokenizer is_fast={is_fast} != expected {expected_tokenizer_is_fast}"
        )

    # Revisions must come from loaded artifacts (config stamps) and/or pinned
    # weight-file hashes — never from unchecked caller-supplied echo alone.
    inferred_base = infer_base_revision_from_model(model)
    inferred_adapter = infer_adapter_revision_from_model(model)
    has_weight_pins = bool(expected_weight_file_sha256)

    def _rev_match(actual: str, expected: str) -> bool:
        # Exact match only — reject misleading revision-prefix matches
        return actual == expected

    resolved_base: str | None = inferred_base
    if inferred_base is not None:
        if not _rev_match(inferred_base, expected_base_revision):
            failures.append(
                f"base revision actual={inferred_base} != expected={expected_base_revision}"
            )
    elif actual_base_revision is not None and not has_weight_pins:
        failures.append(
            "base revision not stamped on loaded config (_commit_hash); "
            "refusing caller-echo-only verification without weight-file SHA pins"
        )
    elif not has_weight_pins:
        failures.append(
            "actual_base_revision missing: no config _commit_hash and no weight-file SHA pins"
        )
    else:
        # Weight pins substitute for missing stamp; still record expected for audit
        resolved_base = expected_base_revision

    resolved_adapter: str | None = inferred_adapter
    if inferred_adapter is not None:
        if not _rev_match(inferred_adapter, expected_adapter_revision):
            failures.append(
                f"adapter revision actual={inferred_adapter} != "
                f"expected={expected_adapter_revision}"
            )
    elif not has_weight_pins:
        failures.append(
            "adapter revision not present on PEFT config and no weight-file "
            "SHA map supplied; refusing caller-echo-only adapter verification"
        )
    else:
        resolved_adapter = expected_adapter_revision

    weight_hashes: dict[str, str] = {}
    if expected_weight_file_sha256:
        # Normalize role-tagged keys before filesystem access; keep role attribution
        try:
            normalized = normalize_weight_manifest_entries(expected_weight_file_sha256)
        except ValueError as e:
            failures.append(str(e))
            normalized = {}
        for role_key, (fs_path, expected) in normalized.items():
            p = Path(fs_path)
            if not p.is_file():
                failures.append(
                    f"weight file missing for hash check: {role_key} -> {fs_path}"
                )
                continue
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            weight_hashes[role_key] = digest
            if digest != expected:
                failures.append(
                    f"weight SHA mismatch {role_key} ({fs_path}): "
                    f"{digest} != {expected}"
                )

    return PostLoadReport(
        ok=len(failures) == 0,
        attn_implementation=attn,
        model_float_dtypes=sorted(str(d) for d in float_dtypes),
        adapter_param_dtypes=sorted(set(adapter_dtypes)),
        tokenizer_class=tok_class,
        tokenizer_is_fast=is_fast,
        base_revision=resolved_base,
        adapter_revision=resolved_adapter or actual_adapter_revision,
        subfolder=actual_subfolder,
        adapter_targets=adapter_targets,
        adapted_layers_inferred=adapted_layers,
        weight_file_sha256=weight_hashes,
        failures=failures,
    )


def verify_adapter_off_behavior(model: Any) -> dict[str, Any]:
    """Fail-closed PEFT 0.19.1-style adapter disable/enable with restore guarantee.

    Requires ``disable_adapter_layers`` **and** ``enable_adapter_layers``.
    A disable-only implementation is rejected. Restore runs in ``finally``.
    Does not run model forwards.
    """
    failures: list[str] = []
    has_disable = hasattr(model, "disable_adapter_layers")
    has_enable = hasattr(model, "enable_adapter_layers")
    if not has_disable:
        failures.append("model lacks disable_adapter_layers (PEFT 0.19.1 API)")
    if not has_enable:
        failures.append(
            "model lacks enable_adapter_layers; refuse disable-only "
            "(cannot safely restore adapter state)"
        )
    if failures:
        return {
            "ok": False,
            "failures": failures,
            "toggled": False,
            "restored": False,
            "has_disable_adapter_layers": has_disable,
            "has_enable_adapter_layers": has_enable,
        }

    toggled = False
    restored = False
    try:
        model.disable_adapter_layers()
        toggled = True
    except Exception as e:  # noqa: BLE001
        failures.append(f"disable_adapter_layers raised: {e}")
    finally:
        try:
            model.enable_adapter_layers()
            restored = True
        except Exception as e:  # noqa: BLE001
            failures.append(f"enable_adapter_layers restore raised: {e}")
            restored = False

    return {
        "ok": len(failures) == 0 and toggled and restored,
        "failures": failures,
        "toggled": toggled,
        "restored": restored,
        "has_disable_adapter_layers": has_disable,
        "has_enable_adapter_layers": has_enable,
        "peft_api": "disable_adapter_layers/enable_adapter_layers",
    }


REQUIRED_WEIGHT_MANIFEST_ROLES = ("base", "adapter")


def normalize_weight_manifest_entries(
    expected_weight_file_sha256: Mapping[str, str],
) -> dict[str, tuple[str, str]]:
    """Map manifest keys to (filesystem_path, expected_sha256).

    ``base:<path>`` / ``adapter:<path>`` are normalized to the path after the
    first colon for filesystem access while preserving role attribution in the
    returned key. Plain paths are kept as-is with role ``path``.
    Rejects empty paths, duplicate filesystem targets under conflicting roles,
    and ambiguous duplicate keys.
    """
    out: dict[str, tuple[str, str]] = {}
    seen_fs: dict[str, str] = {}
    seen_keys: set[str] = set()
    for key, expected in expected_weight_file_sha256.items():
        if key in seen_keys:
            raise ValueError(f"ambiguous duplicate manifest key: {key}")
        seen_keys.add(key)
        if key.startswith("base:") or key.startswith("adapter:"):
            role, _, rest = key.partition(":")
            fs_path = rest
            if not fs_path:
                raise ValueError(f"empty path after role tag in {key!r}")
            role_key = f"{role}:{fs_path}"
        else:
            fs_path = key
            role_key = f"path:{fs_path}"
        # Resolve for duplicate detection but keep original string for open()
        resolved = str(Path(fs_path))
        if resolved in seen_fs and seen_fs[resolved] != role_key.split(":", 1)[0]:
            raise ValueError(
                f"ambiguous duplicate filesystem path {fs_path} under roles "
                f"{seen_fs[resolved]} and {role_key.split(':', 1)[0]}"
            )
        seen_fs[resolved] = role_key.split(":", 1)[0]
        out[role_key] = (fs_path, expected)
    return out


def validate_weight_manifest_roles(
    expected_weight_file_sha256: Mapping[str, str] | None,
    *,
    resolved_base_files: Sequence[str] | None = None,
    resolved_adapter_files: Sequence[str] | None = None,
    require_resolved_inventories: bool = False,
) -> list[str]:
    """Manifest must cover **actual resolved** base/adapter weight files.

    Rejects arbitrary caller-selected paths whose names merely contain
    ``base``/``adapter``. Prefer exact resolved inventories; otherwise require
    explicit role-tagged keys ``base:<abspath>`` and ``adapter:<abspath>``.
    """
    failures: list[str] = []
    if not expected_weight_file_sha256:
        return ["weight SHA manifest missing"]
    paths = set(expected_weight_file_sha256.keys())

    if require_resolved_inventories and (
        resolved_base_files is None or resolved_adapter_files is None
    ):
        failures.append(
            "resolved base and adapter weight-file inventories required "
            "(refuse unbound caller-selected manifests)"
        )
        return failures

    if resolved_base_files is not None and resolved_adapter_files is not None:
        for f in resolved_base_files:
            if f not in paths and f"base:{f}" not in paths:
                failures.append(f"manifest missing resolved base weight file: {f}")
        for f in resolved_adapter_files:
            if f not in paths and f"adapter:{f}" not in paths:
                failures.append(f"manifest missing resolved adapter weight file: {f}")
        # Reject extra unbound caller files that are not in resolved inventories
        allowed = set(resolved_base_files) | set(resolved_adapter_files)
        allowed |= {f"base:{f}" for f in resolved_base_files}
        allowed |= {f"adapter:{f}" for f in resolved_adapter_files}
        extras = paths - allowed
        if extras:
            failures.append(
                f"manifest contains unbound caller-selected files: {sorted(extras)[:5]}"
            )
        return failures

    # No resolved inventories: require explicit role tags (not substring heuristics)
    base_keys = [k for k in paths if k.startswith("base:")]
    adapter_keys = [k for k in paths if k.startswith("adapter:")]
    if not base_keys:
        failures.append(
            "manifest requires explicit base:<path> keys when resolved "
            "inventories are unavailable (substring 'base' is insufficient)"
        )
    if not adapter_keys:
        failures.append(
            "manifest requires explicit adapter:<path> keys when resolved "
            "inventories are unavailable"
        )
    return failures


def resolve_weight_files_from_model(model: Any) -> dict[str, list[str]]:
    """Best-effort local paths for base and adapter weight files actually loaded."""
    base_files: list[str] = []
    adapter_files: list[str] = []
    # Base: transformers name_or_path / _name_or_path
    cfg = getattr(model, "config", None)
    base_root = getattr(cfg, "_name_or_path", None) or getattr(model, "name_or_path", None)
    if isinstance(base_root, str) and Path(base_root).is_dir():
        for pat in ("*.safetensors", "*.bin"):
            base_files.extend(str(p) for p in Path(base_root).glob(pat) if p.is_file())
    # Adapter: PEFT peft_config path
    peft_cfg = None
    if hasattr(model, "peft_config"):
        peft_cfg = (
            model.peft_config.get("default")
            if isinstance(model.peft_config, dict)
            else model.peft_config
        )
    adapter_root = None
    if peft_cfg is not None:
        adapter_root = getattr(peft_cfg, "base_model_name_or_path", None)
        # PEFT stores adapter path variously
        for attr in ("adapter_path", "peft_path", "_name_or_path"):
            v = getattr(peft_cfg, attr, None)
            if isinstance(v, str) and Path(v).exists():
                adapter_root = v
                break
    # Also check active adapters directory via peft helper
    if hasattr(model, "peft_config") and peft_cfg is not None:
        hub = getattr(peft_cfg, "revision", None)
        _ = hub
    if isinstance(adapter_root, str) and Path(adapter_root).is_dir():
        for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"):
            p = Path(adapter_root) / name
            if p.is_file():
                adapter_files.append(str(p))
        if not adapter_files:
            adapter_files.extend(
                str(p) for p in Path(adapter_root).glob("adapter_model.*") if p.is_file()
            )
    return {"base": sorted(set(base_files)), "adapter": sorted(set(adapter_files))}


def refuse_caller_revision_echo_as_proof(
    *,
    caller_base_revision: str | None,
    caller_adapter_revision: str | None,
    inferred_base: str | None,
    inferred_adapter: str | None,
    has_weight_pins: bool,
) -> list[str]:
    """Explicit policy: caller-supplied revision strings are not provenance proof."""
    failures: list[str] = []
    if caller_base_revision and inferred_base is None and not has_weight_pins:
        failures.append(
            "refusing caller-supplied base revision echo as proof "
            f"(caller={caller_base_revision!r})"
        )
    if caller_adapter_revision and inferred_adapter is None and not has_weight_pins:
        failures.append(
            "refusing caller-supplied adapter revision echo as proof "
            f"(caller={caller_adapter_revision!r})"
        )
    return failures
