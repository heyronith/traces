"""Q1 scientific model loading (proposed F5 implementation).

Proposed interpretation (pending Agent 1 scientific consistency review before
real-model execution):
  - Base weights: BF16
  - LoRA weights: BF16
  - ``autocast_adapter_dtype=False`` (no silent FP32 adapter upcast)
  - ``attn_implementation=\"eager\"``
  - No silent backend/dtype fallback

Legacy Stage 0/1 loading remains ``traces.model_loader.load_dense_lora_model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from traces.model_loader import (
    BASE_MODEL_ID,
    DENSE_ADAPTER_REPO,
    PINNED_ADAPTER_REVISION,
    PINNED_BASE_REVISION,
    SPARSE_ADAPTER_REPO,
    ModelLoadReport,
    _count_adapter_params,
    _disable_dropout,
)
from traces.q1.post_load import (
    refuse_caller_revision_echo_as_proof,
    require_bound_weight_inventories,
    resolve_weight_files_from_model,
    validate_weight_manifest_roles,
    verify_adapter_off_behavior,
    verify_post_load,
)
from traces.q1.runtime_spec import (
    REQUIRED_ATTN_IMPLEMENTATION,
    REQUIRED_TORCH_DTYPE,
    assert_bf16_dtype,
    assert_eager_attention,
    q1_model_load_kwargs,
)

Q1_LOADER_NOTES = (
    "Q1 proposed F5: bf16 base+LoRA, eager attn, autocast_adapter_dtype=False. "
    "Pending Agent 1 scientific consistency review before real-model execution."
)


def load_q1_scientific_model(
    *,
    base_model: str = BASE_MODEL_ID,
    adapter_repo: str = DENSE_ADAPTER_REPO,
    subfolder: str,
    base_revision: str = PINNED_BASE_REVISION,
    adapter_revision: str = PINNED_ADAPTER_REVISION,
    device_map: str | dict | None = "auto",
    expected_adapter_targets: list[str] | None = None,
    expected_adapted_layers: list[int] | None = None,
    expected_tokenizer_class: str | None = None,
    expected_tokenizer_is_fast: bool | None = None,
    expected_weight_file_sha256: dict[str, str] | None = None,
    base_weight_snapshot_dir: str | None = None,
    adapter_weight_snapshot_dir: str | None = None,
) -> tuple[PeftModel, Any, ModelLoadReport, dict[str, Any]]:
    """Strict Q1 scientific load. Does not alter Stage 1 loader semantics.

    Scientific authorization is required independently of the orchestrator (M5).
    Tiny-Gemma engineering tests must not call this entry point.
    """
    from traces.q1.scientific_execution import require_scientific_authorization

    require_scientific_authorization()
    assert_bf16_dtype(REQUIRED_TORCH_DTYPE)
    assert_eager_attention(REQUIRED_ATTN_IMPLEMENTATION)
    if base_model != BASE_MODEL_ID:
        raise RuntimeError(f"Q1 base must be {BASE_MODEL_ID}")
    if "dense-lora" not in adapter_repo or adapter_repo.rstrip("/") == SPARSE_ADAPTER_REPO:
        raise RuntimeError(f"Q1 refuses adapter repo {adapter_repo!r}")
    if base_revision != PINNED_BASE_REVISION:
        raise RuntimeError(f"Q1 base revision must be {PINNED_BASE_REVISION}")
    if adapter_revision != PINNED_ADAPTER_REVISION:
        raise RuntimeError(f"Q1 adapter revision must be {PINNED_ADAPTER_REVISION}")

    # Canonical kwargs — no silent dtype / attention fallback permitted
    load_kwargs = q1_model_load_kwargs(
        dtype=REQUIRED_TORCH_DTYPE,
        attn_implementation=REQUIRED_ATTN_IMPLEMENTATION,
    )

    if not expected_weight_file_sha256:
        raise RuntimeError(
            "Q1 scientific load requires expected_weight_file_sha256 pins "
            "(refuse caller-echo-only revision verification)"
        )
    if expected_adapter_targets is None:
        raise RuntimeError(
            "Q1 production load requires explicit expected_adapter_targets"
        )
    if expected_adapted_layers is None:
        raise RuntimeError(
            "Q1 production load requires explicit expected_adapted_layers"
        )
    # D5: authoritative local snapshots required — never fall back to hub copies
    if not base_weight_snapshot_dir or not adapter_weight_snapshot_dir:
        raise RuntimeError(
            "Q1 scientific load requires authoritative local base and adapter "
            "snapshot directories (D5) — refuse unverified hub fallback"
        )
    from traces.q1.checkpoint_provenance import (
        bind_unit_checkpoint_provenance,
        resolve_unit_adapter_subfolder_dir,
    )

    # Pre-check: role-tagged keys required until resolved inventories exist
    pre_manifest = validate_weight_manifest_roles(expected_weight_file_sha256)
    if pre_manifest:
        raise RuntimeError(
            "Q1 weight SHA manifest invalid: " + "; ".join(pre_manifest)
        )

    # Infer family/seed from subfolder for unit bind (l19/seed42 or l1523/seedN)
    parts = str(subfolder).strip("/").split("/")
    if len(parts) != 2 or not parts[1].startswith("seed"):
        raise RuntimeError(
            f"adapter subfolder {subfolder!r} is not family/seedN (D5)"
        )
    family_from_sf, seed_token = parts[0], parts[1]
    try:
        seed_from_sf = int(seed_token.removeprefix("seed"))
    except ValueError as e:
        raise RuntimeError(f"unparseable seed in subfolder {subfolder!r}") from e

    unit_bind = bind_unit_checkpoint_provenance(
        family=family_from_sf,
        seed=seed_from_sf,
        subfolder=subfolder,
        base_snapshot_dir=base_weight_snapshot_dir,
        adapter_snapshot_dir=adapter_weight_snapshot_dir,
        expected_sha256_by_relpath=expected_weight_file_sha256,
        require_section11_tokenizer_pins=True,
    )
    local_paths = unit_bind["local_paths"]
    tok_files = unit_bind["tokenizer_template_files"]
    snapshot_bind = unit_bind["snapshot_byte_binding"]
    computed_pre = dict(unit_bind["computed_sha256"])
    if not computed_pre:
        raise RuntimeError(
            "computed digests missing from unit provenance bind — refuse "
            "expected-hash substitution (D5)"
        )

    # use_topk probe against local adapter snapshot only (no hub)
    import json as _json

    local_adapter = Path(local_paths["adapter_load_path"])
    # Double-check: load path is the required subfolder (no root fallback)
    expected_ad = resolve_unit_adapter_subfolder_dir(
        adapter_weight_snapshot_dir, subfolder
    )
    if Path(local_paths["adapter_load_path"]).resolve() != expected_ad.resolve():
        raise RuntimeError(
            "adapter load path drifted from required unit subfolder (D5)"
        )
    cfg_probe = local_adapter / "adapter_config.json"
    if cfg_probe.is_file():
        try:
            cfg_data = _json.loads(cfg_probe.read_text())
            if cfg_data.get("use_topk") is True:
                raise RuntimeError("Q1 refuses use_topk=true adapters")
        except _json.JSONDecodeError as e:
            raise RuntimeError(f"unreadable local adapter_config.json: {e}") from e

    tokenizer = AutoTokenizer.from_pretrained(
        local_paths["adapter_load_path"],
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_paths["base_load_path"],
        torch_dtype=load_kwargs["torch_dtype"],
        device_map=device_map,
        attn_implementation=load_kwargs["attn_implementation"],
        trust_remote_code=False,
        local_files_only=True,
    )
    # Explicit: do not autocast adapter weights to a different dtype
    model = PeftModel.from_pretrained(
        model,
        local_paths["adapter_load_path"],
        autocast_adapter_dtype=False,
        local_files_only=True,
    )
    model.eval()
    _disable_dropout(model)
    torch.set_grad_enabled(False)

    # Refuse silent dtype conversion after load
    for name, p in model.named_parameters():
        if p.dtype.is_floating_point and p.dtype != REQUIRED_TORCH_DTYPE:
            raise RuntimeError(
                f"Silent dtype conversion detected on {name}: {p.dtype} "
                f"(require {REQUIRED_TORCH_DTYPE}; no fallback)"
            )

    # Inventories already authenticated by bind_unit_checkpoint_provenance (D5)
    setattr(model, "_q1_base_weight_dir", base_weight_snapshot_dir)
    setattr(model, "_q1_adapter_weight_dir", str(expected_ad))
    resolved = {
        "base": list(unit_bind["inventory"]["base"]),
        "adapter": list(unit_bind["inventory"]["adapter"]),
    }
    inventory_bind = {
        "ok": True,
        "failures": [],
        "resolved": resolved,
        "digests": {
            k.removeprefix("inventory:"): v
            for k, v in computed_pre.items()
            if k.startswith("inventory:")
        },
        "unit_bound": True,
        "subfolder": subfolder,
    }

    # Weight SHA authentication already completed by unit provenance bind (D5).
    # Post-load checks dtype/attn/targets/layers/tokenizer identity only.
    post = verify_post_load(
        model,
        tokenizer,
        expected_base_revision=base_revision,
        expected_adapter_revision=adapter_revision,
        actual_subfolder=subfolder,
        expected_adapter_targets=expected_adapter_targets,
        expected_adapted_layers=expected_adapted_layers,
        expected_tokenizer_class=expected_tokenizer_class,
        expected_tokenizer_is_fast=expected_tokenizer_is_fast,
        expected_weight_file_sha256=None,
        require_eager=True,
        require_bf16=True,
        weight_sha_pins_preverified=True,
    )
    if not post.ok:
        raise RuntimeError(f"Q1 post-load verification failed: {post.failures}")

    adapter_off = verify_adapter_off_behavior(model)
    if not adapter_off["ok"]:
        raise RuntimeError(
            f"Q1 adapter-off verification failed: {adapter_off['failures']}"
        )

    peft_cfg = model.peft_config["default"]
    cfg = model.config
    n_q = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_q))
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_q))
    tensor_count, param_count = _count_adapter_params(model)
    report = ModelLoadReport(
        base_model_id=str(local_paths["base_load_path"]),
        adapter_repo=str(local_paths["adapter_load_path"]),
        adapter_subfolder=subfolder,
        adapter_kind="dense-lora",
        base_revision=base_revision,
        adapter_revision=adapter_revision,
        architecture=getattr(cfg, "model_type", None),
        num_layers=int(getattr(cfg, "num_hidden_layers", 0)),
        hidden_size=int(cfg.hidden_size),
        num_attention_heads=n_q,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        intermediate_size=int(getattr(cfg, "intermediate_size", 0)),
        adapted_modules=sorted(list(getattr(peft_cfg, "target_modules", []) or [])),
        lora_r=int(peft_cfg.r),
        lora_alpha=int(peft_cfg.lora_alpha),
        lora_scaling=float(peft_cfg.lora_alpha) / float(peft_cfg.r),
        adapter_tensor_count=tensor_count,
        adapter_param_count=param_count,
        dtype=str(REQUIRED_TORCH_DTYPE),
        device=str(next(model.parameters()).device),
        notes=[Q1_LOADER_NOTES, "loaded_from_verified_local_snapshots_only"],
    )
    # D5: computed digests only — never substitute caller expected hashes
    computed_digests = dict(computed_pre)
    if not computed_digests:
        raise RuntimeError(
            "computed_weight_file_sha256 empty after load — refuse expected-hash "
            "fallback (D5)"
        )
    meta = {
        "proposed_f5_interpretation": Q1_LOADER_NOTES,
        "autocast_adapter_dtype": False,
        "attn_implementation": REQUIRED_ATTN_IMPLEMENTATION,
        "torch_dtype": str(REQUIRED_TORCH_DTYPE),
        "post_load": post.to_dict(),
        "adapter_off_verification": adapter_off,
        "pending_owner_f5_approval": True,
        "pending_agent1_scientific_consistency_review": True,
        "caller_revision_echo_not_accepted_as_proof": True,
        "weight_sha_pins_required": True,
        "resolved_weight_files": resolved,
        "weight_inventory_binding": inventory_bind,
        "snapshot_byte_binding": snapshot_bind,
        "tokenizer_template_files": tok_files,
        "local_snapshot_load_paths": local_paths,
        "unit_checkpoint_provenance": {
            "family": family_from_sf,
            "seed": seed_from_sf,
            "subfolder": subfolder,
            "adapter_subfolder_dir": unit_bind["adapter_subfolder_dir"],
        },
        "computed_weight_file_sha256": computed_digests,
        "local_files_only": True,
        "hub_fallback_refused": True,
        "base_revision": base_revision,
        "adapter_revision": adapter_revision,
    }
    return model, tokenizer, report, meta


def validate_loader_policy_inputs(
    *,
    base_revision: str,
    adapter_revision: str,
    expected_weight_file_sha256: dict[str, str] | None,
    torch_dtype: Any = None,
    attn_implementation: str | None = None,
    expected_adapter_targets: list[str] | None = None,
    expected_adapted_layers: list[int] | None = None,
    resolved_base_files: list[str] | None = None,
    resolved_adapter_files: list[str] | None = None,
) -> list[str]:
    """CPU-safe policy checks for mocks/negative tests (no HF download)."""
    failures: list[str] = []
    if base_revision != PINNED_BASE_REVISION:
        failures.append(f"base_revision {base_revision!r} != pin")
    if adapter_revision != PINNED_ADAPTER_REVISION:
        failures.append(f"adapter_revision {adapter_revision!r} != pin")
    if expected_adapter_targets is None:
        failures.append("expected_adapter_targets required for production loading")
    if expected_adapted_layers is None:
        failures.append("expected_adapted_layers required for production loading")
    if not expected_weight_file_sha256:
        failures.append("expected_weight_file_sha256 required")
        failures.extend(
            refuse_caller_revision_echo_as_proof(
                caller_base_revision=base_revision,
                caller_adapter_revision=adapter_revision,
                inferred_base=None,
                inferred_adapter=None,
                has_weight_pins=False,
            )
        )
    else:
        failures.extend(
            validate_weight_manifest_roles(
                expected_weight_file_sha256,
                resolved_base_files=resolved_base_files,
                resolved_adapter_files=resolved_adapter_files,
            )
        )
    if torch_dtype is not None and torch_dtype != REQUIRED_TORCH_DTYPE:
        failures.append(f"refusing dtype {torch_dtype}; no silent conversion")
    if attn_implementation is not None and attn_implementation != REQUIRED_ATTN_IMPLEMENTATION:
        failures.append(
            f"refusing attn_implementation={attn_implementation!r}; no fallback"
        )
    return failures
