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


def _adapter_mechan_state(model: Any) -> dict[str, Any]:
    """Best-effort read of PEFT adapter enabled/disabled flags (no forward)."""
    info: dict[str, Any] = {}
    if hasattr(model, "_adapters_disabled"):
        info["adapters_disabled_flag"] = bool(getattr(model, "_adapters_disabled"))
    if hasattr(model, "get_model_status"):
        try:
            status = model.get_model_status()
            info["model_status_enabled"] = getattr(status, "enabled", None)
        except Exception as e:  # noqa: BLE001
            info["model_status_error"] = str(e)
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "peft_config"):
        # Layer-level disable flag on first LoRA module when present
        disabled_flags: list[bool] = []
        for mod in base.modules() if hasattr(base, "modules") else []:
            if hasattr(mod, "disable_adapters"):
                try:
                    disabled_flags.append(bool(mod.disable_adapters))
                except Exception:  # noqa: BLE001
                    pass
        if disabled_flags:
            info["layer_disable_adapters_any"] = any(disabled_flags)
            info["layer_disable_adapters_all"] = all(disabled_flags)
    return info


def verify_adapter_off_behavior(model: Any) -> dict[str, Any]:
    """Mechanical PEFT 0.19.1 adapter state-switch check (no forwards).

    Supported mechanisms (priority order):
    1. ``model.base_model.disable_adapter_layers`` /
       ``enable_adapter_layers`` (LoraModel under ``PeftModel``)
    2. ``PeftModel.disable_adapter()`` context manager (public PeftModel API)

    Distinguishes mechanical state-switch verification from numerical
    adapter-off validation. Never claims the latter without an authorized
    forward test (``numerical_adapter_off_validated=False``).
    """
    failures: list[str] = []
    toggled = False
    restored = False
    mechanism: str | None = None
    state_before: dict[str, Any] = {}
    state_disabled: dict[str, Any] = {}
    state_after: dict[str, Any] = {}

    base = getattr(model, "base_model", None)
    has_layers_api = (
        base is not None
        and callable(getattr(base, "disable_adapter_layers", None))
        and callable(getattr(base, "enable_adapter_layers", None))
    )
    has_cm = callable(getattr(model, "disable_adapter", None))
    # Legacy direct methods on the wrapper (not PeftModel 0.19.1 public API)
    has_direct = callable(getattr(model, "disable_adapter_layers", None)) and callable(
        getattr(model, "enable_adapter_layers", None)
    )

    if not has_layers_api and not has_cm and not has_direct:
        return {
            "ok": False,
            "failures": [
                "no supported PEFT 0.19.1 adapter-disable API "
                "(need base_model.disable/enable_adapter_layers or "
                "PeftModel.disable_adapter context manager)"
            ],
            "toggled": False,
            "restored": False,
            "peft_api": None,
            "verification_kind": "mechanical_state_switch",
            "numerical_adapter_off_validated": False,
            "state_before": {},
            "state_disabled": {},
            "state_after": {},
        }

    state_before = _adapter_mechan_state(model)

    try:
        if has_layers_api:
            mechanism = "base_model.disable_adapter_layers/enable_adapter_layers"
            try:
                base.disable_adapter_layers()  # type: ignore[union-attr]
                toggled = True
                state_disabled = _adapter_mechan_state(model)
            except Exception as e:  # noqa: BLE001
                failures.append(f"disable_adapter_layers raised: {e}")
            finally:
                try:
                    base.enable_adapter_layers()  # type: ignore[union-attr]
                    restored = True
                except Exception as e:  # noqa: BLE001
                    failures.append(f"enable_adapter_layers restore raised: {e}")
                    restored = False
        elif has_cm:
            mechanism = "PeftModel.disable_adapter context manager"
            try:
                cm = model.disable_adapter()
                if not hasattr(cm, "__enter__") or not hasattr(cm, "__exit__"):
                    failures.append(
                        "disable_adapter() did not return a context manager"
                    )
                else:
                    cm.__enter__()
                    toggled = True
                    state_disabled = _adapter_mechan_state(model)
                    if hasattr(model, "_adapters_disabled") and not getattr(
                        model, "_adapters_disabled"
                    ):
                        failures.append(
                            "disable_adapter entered but _adapters_disabled is False"
                        )
                    cm.__exit__(None, None, None)
                    restored = True
            except Exception as e:  # noqa: BLE001
                failures.append(f"disable_adapter context manager failed: {e}")
                # Best-effort restore if enter succeeded
                if toggled and not restored:
                    try:
                        if "cm" in locals() and hasattr(cm, "__exit__"):
                            cm.__exit__(None, None, None)
                            restored = True
                    except Exception as e2:  # noqa: BLE001
                        failures.append(f"disable_adapter restore failed: {e2}")
        else:
            mechanism = "legacy disable_adapter_layers/enable_adapter_layers on wrapper"
            try:
                model.disable_adapter_layers()
                toggled = True
                state_disabled = _adapter_mechan_state(model)
            except Exception as e:  # noqa: BLE001
                failures.append(f"disable_adapter_layers raised: {e}")
            finally:
                try:
                    model.enable_adapter_layers()
                    restored = True
                except Exception as e:  # noqa: BLE001
                    failures.append(f"enable_adapter_layers restore raised: {e}")
                    restored = False
    finally:
        state_after = _adapter_mechan_state(model)

    return {
        "ok": len(failures) == 0 and toggled and restored,
        "failures": failures,
        "toggled": toggled,
        "restored": restored,
        "peft_api": mechanism,
        "verification_kind": "mechanical_state_switch",
        "numerical_adapter_off_validated": False,
        "state_before": state_before,
        "state_disabled": state_disabled,
        "state_after": state_after,
        "has_base_model_layers_api": has_layers_api,
        "has_disable_adapter_cm": has_cm,
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


def _collect_weight_files_in_dir(root: Path | str) -> list[str]:
    p = Path(root)
    if not p.is_dir():
        return []
    files: list[str] = []
    for pat in ("*.safetensors", "*.bin", "*.pt"):
        files.extend(str(x) for x in p.glob(pat) if x.is_file())
    # Adapter-specific names (may already be covered by globs)
    for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"):
        ap = p / name
        if ap.is_file():
            files.append(str(ap))
    return sorted(set(files))


def resolve_weight_files_from_snapshots(
    *,
    base_snapshot_dir: Path | str | None,
    adapter_snapshot_dir: Path | str | None,
) -> dict[str, list[str]]:
    """Resolve weight files from pinned local snapshot directories used for construction."""
    return {
        "base": _collect_weight_files_in_dir(base_snapshot_dir)
        if base_snapshot_dir is not None
        else [],
        "adapter": _collect_weight_files_in_dir(adapter_snapshot_dir)
        if adapter_snapshot_dir is not None
        else [],
    }


def resolve_weight_files_from_model(
    model: Any,
    *,
    base_snapshot_dir: Path | str | None = None,
    adapter_snapshot_dir: Path | str | None = None,
) -> dict[str, list[str]]:
    """Local paths for base/adapter weight files actually used to construct ``model``.

    Prefer explicit snapshot dirs (pinned local construction roots). Fall back to
    paths stamped on the loaded model/config. Empty inventories are returned as
    empty lists — callers must fail closed.
    """
    if base_snapshot_dir is not None or adapter_snapshot_dir is not None:
        snap = resolve_weight_files_from_snapshots(
            base_snapshot_dir=base_snapshot_dir,
            adapter_snapshot_dir=adapter_snapshot_dir,
        )
        # Allow model attributes to fill a missing side only when that side's
        # explicit dir was not provided.
        if base_snapshot_dir is not None and adapter_snapshot_dir is not None:
            return snap

    base_files: list[str] = []
    adapter_files: list[str] = []

    # Explicit override attributes (tests / loader-stamped local snapshots)
    for attr, bucket in (
        ("_q1_base_weight_files", "base"),
        ("_q1_adapter_weight_files", "adapter"),
    ):
        v = getattr(model, attr, None)
        if isinstance(v, (list, tuple)) and v:
            if bucket == "base":
                base_files.extend(str(x) for x in v)
            else:
                adapter_files.extend(str(x) for x in v)

    for attr, bucket in (
        ("_q1_base_weight_dir", "base"),
        ("_q1_adapter_weight_dir", "adapter"),
    ):
        v = getattr(model, attr, None)
        if isinstance(v, str) and Path(v).is_dir():
            found = _collect_weight_files_in_dir(v)
            if bucket == "base":
                base_files.extend(found)
            else:
                adapter_files.extend(found)

    if base_snapshot_dir is not None:
        base_files = _collect_weight_files_in_dir(base_snapshot_dir)
    if adapter_snapshot_dir is not None:
        adapter_files = _collect_weight_files_in_dir(adapter_snapshot_dir)

    if not base_files:
        cfg = getattr(model, "config", None)
        base_root = getattr(cfg, "_name_or_path", None) or getattr(
            model, "name_or_path", None
        )
        # Prefer get_base_model when present (PeftModel)
        if hasattr(model, "get_base_model"):
            try:
                bm = model.get_base_model()
                base_root = (
                    getattr(getattr(bm, "config", None), "_name_or_path", None)
                    or getattr(bm, "name_or_path", None)
                    or base_root
                )
            except Exception:  # noqa: BLE001
                pass
        if isinstance(base_root, str) and Path(base_root).is_dir():
            base_files.extend(_collect_weight_files_in_dir(base_root))

    if not adapter_files:
        peft_cfg = None
        if hasattr(model, "peft_config"):
            peft_cfg = (
                model.peft_config.get("default")
                if isinstance(model.peft_config, dict)
                else model.peft_config
            )
        adapter_root = None
        if peft_cfg is not None:
            for attr in (
                "adapter_path",
                "peft_path",
                "_name_or_path",
                "base_model_name_or_path",
            ):
                v = getattr(peft_cfg, attr, None)
                if isinstance(v, str) and Path(v).is_dir():
                    # base_model_name_or_path is often the base, not adapter —
                    # only accept if it contains adapter_model.*
                    if attr == "base_model_name_or_path":
                        probe = _collect_weight_files_in_dir(v)
                        if any(Path(f).name.startswith("adapter_model") for f in probe):
                            adapter_root = v
                            break
                        continue
                    adapter_root = v
                    break
        if isinstance(adapter_root, str):
            adapter_files.extend(_collect_weight_files_in_dir(adapter_root))

    return {"base": sorted(set(base_files)), "adapter": sorted(set(adapter_files))}


def require_bound_weight_inventories(
    resolved: Mapping[str, Sequence[str]],
    expected_weight_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Fail closed unless both inventories are non-empty and exactly covered.

    Verifies SHA256 of every resolved snapshot file against the manifest.
    Rejects unbound caller-selected files that are not in the inventories.
    """
    failures: list[str] = []
    base = list(resolved.get("base") or [])
    adapter = list(resolved.get("adapter") or [])
    if not base:
        failures.append(
            "base weight inventory empty: cannot bind to pinned local snapshot"
        )
    if not adapter:
        failures.append(
            "adapter weight inventory empty: cannot bind to pinned local snapshot"
        )
    if failures:
        return {
            "ok": False,
            "failures": failures,
            "resolved": {"base": base, "adapter": adapter},
            "digests": {},
        }

    role_fails = validate_weight_manifest_roles(
        expected_weight_file_sha256,
        resolved_base_files=base,
        resolved_adapter_files=adapter,
        require_resolved_inventories=True,
    )
    failures.extend(role_fails)

    # Build expected digest map (role-tagged or plain path)
    expected_by_path: dict[str, str] = {}
    try:
        normalized = normalize_weight_manifest_entries(expected_weight_file_sha256)
        for _role_key, (fs_path, digest) in normalized.items():
            expected_by_path[str(Path(fs_path))] = digest
    except ValueError as e:
        failures.append(str(e))
        normalized = {}

    digests: dict[str, str] = {}
    for f in base + adapter:
        p = Path(f)
        if not p.is_file():
            failures.append(f"resolved weight file missing on disk: {f}")
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        digests[f] = digest
        exp = expected_by_path.get(str(p)) or expected_by_path.get(f)
        if exp is None:
            # Try role-tagged lookup already covered by validate_weight_manifest_roles
            failures.append(f"no manifest digest for resolved inventory file: {f}")
        elif digest != exp:
            failures.append(f"SHA256 mismatch for resolved inventory file {f}")

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "resolved": {"base": base, "adapter": adapter},
        "digests": digests,
    }


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
