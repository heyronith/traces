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
from typing import Any, Mapping

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
        FROZEN_TOKENIZER_IDENTITY,
        bind_unit_checkpoint_provenance,
        resolve_unit_adapter_subfolder_dir,
    )
    from traces.q1.checkpoint_trust import (
        authenticate_production_checkpoint_manifest,
        unit_digests_from_auth,
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

    # D5-A: production identity from owner-pinned manifest only — never treat
    # caller expected_weight_file_sha256 as independent authoritative proof.
    # Engineering loads must use engineering_load_q1_scientific_model instead.
    repo_root_for_pin = Path(base_weight_snapshot_dir).resolve()
    # Walk up to find repo marker; production hooks pass explicit auth.
    trust_auth = authenticate_production_checkpoint_manifest(
        repo_root=_infer_repo_root(repo_root_for_pin)
    )
    if not trust_auth.production_unlocked:
        raise RuntimeError(
            "production checkpoint identity blocked (D5-A): "
            + "; ".join(trust_auth.failures)
            + " — caller expected_weight_file_sha256 is not authoritative; "
            "use engineering_load_q1_scientific_model for synthetic fixtures"
        )
    authoritative = unit_digests_from_auth(trust_auth, subfolder=subfolder)
    if expected_weight_file_sha256:
        # Optional consistency check only — never replaces owner pin
        for k, v in expected_weight_file_sha256.items():
            if k in authoritative and authoritative[k] != v:
                raise RuntimeError(
                    f"caller manifest conflicts with authoritative pin at {k}"
                )

    pre_manifest = validate_weight_manifest_roles(authoritative)
    if pre_manifest:
        raise RuntimeError(
            "Q1 weight SHA manifest invalid: " + "; ".join(pre_manifest)
        )

    unit_bind = bind_unit_checkpoint_provenance(
        family=family_from_sf,
        seed=seed_from_sf,
        subfolder=subfolder,
        base_snapshot_dir=base_weight_snapshot_dir,
        adapter_snapshot_dir=adapter_weight_snapshot_dir,
        expected_sha256_by_relpath=authoritative,
        require_section11_tokenizer_pins=True,
        require_authoritative_trust=True,
        trust_auth=trust_auth,
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

    local_adapter = Path(local_paths["adapter_load_path"])
    expected_ad = resolve_unit_adapter_subfolder_dir(
        adapter_weight_snapshot_dir, subfolder
    )
    if Path(local_paths["adapter_load_path"]).resolve() != expected_ad.resolve():
        raise RuntimeError(
            "adapter load path drifted from required unit subfolder (D5)"
        )

    # D5-B: frozen tokenizer class / slow tokenizer only
    tok_class = expected_tokenizer_class or FROZEN_TOKENIZER_IDENTITY[
        "expected_tokenizer_class"
    ]
    tok_is_fast = (
        expected_tokenizer_is_fast
        if expected_tokenizer_is_fast is not None
        else FROZEN_TOKENIZER_IDENTITY["expected_is_fast"]
    )
    if tok_is_fast is not False:
        raise RuntimeError("Q1 requires use_fast=False / is_fast=False (D5-B)")

    tokenizer = AutoTokenizer.from_pretrained(
        local_paths["adapter_load_path"],
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
    )
    if type(tokenizer).__name__ != tok_class:
        raise RuntimeError(
            f"tokenizer class {type(tokenizer).__name__} != frozen {tok_class} (D5-B)"
        )
    if bool(getattr(tokenizer, "is_fast", True)):
        raise RuntimeError("tokenizer is_fast=True refused — require slow tokenizer (D5-B)")

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
        expected_tokenizer_class=tok_class,
        expected_tokenizer_is_fast=False,
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


def _infer_repo_root(start: Path) -> Path:
    """Best-effort repo root for fixed-path production manifest lookup."""
    cur = Path(start).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "src" / "traces").is_dir() and (
            candidate / "results" / "q1_implementation_prep"
        ).is_dir():
            return candidate
    # Fall back to start — authenticate_production will fail closed if pin unset
    return cur


def engineering_load_q1_scientific_model(
    *,
    subfolder: str,
    engineering_manifest: Mapping[str, str] | None = None,
    engineering_manifest_payload: Mapping[str, Any] | None = None,
    expected_adapter_targets: list[str] | None = None,
    expected_adapted_layers: list[int] | None = None,
    base_weight_snapshot_dir: str | None = None,
    adapter_weight_snapshot_dir: str | None = None,
    device_map: str | dict | None = "auto",
    base_revision: str = PINNED_BASE_REVISION,
    adapter_revision: str = PINNED_ADAPTER_REVISION,
) -> tuple[PeftModel, Any, ModelLoadReport, dict[str, Any]]:
    """Engineering-only synthetic load path — never production_unlocked (D5-A).

    Requires ``TRACES_Q1_CHECKPOINT_ENGINEERING_TRUST=1`` and a schema-valid
    engineering manifest payload covering all ten adapter identities.
    Does not set scientific production authorization.
    """
    from traces.q1.checkpoint_provenance import (
        FROZEN_TOKENIZER_IDENTITY,
        bind_unit_checkpoint_provenance,
        resolve_unit_adapter_subfolder_dir,
    )
    from traces.q1.checkpoint_trust import (
        authenticate_engineering_checkpoint_manifest,
        unit_digests_from_auth,
    )
    from traces.q1.scientific_execution import require_scientific_authorization

    require_scientific_authorization()
    if not base_weight_snapshot_dir or not adapter_weight_snapshot_dir:
        raise RuntimeError(
            "engineering load requires local base and adapter snapshot dirs"
        )
    if engineering_manifest_payload is None:
        raise RuntimeError(
            "engineering_manifest_payload required — refuse treating a flat "
            "caller digest map as authoritative production identity (D5-A)"
        )
    _ = engineering_manifest  # flat maps are non-authoritative; ignored
    auth = authenticate_engineering_checkpoint_manifest(engineering_manifest_payload)
    if not auth.ok or auth.production_unlocked:
        raise RuntimeError(
            "engineering checkpoint auth failed or incorrectly production-unlocked: "
            + "; ".join(auth.failures)
        )
    parts = str(subfolder).strip("/").split("/")
    family_from_sf, seed_token = parts[0], parts[1]
    seed_from_sf = int(seed_token.removeprefix("seed"))
    digests = unit_digests_from_auth(auth, subfolder=subfolder)

    assert_bf16_dtype(REQUIRED_TORCH_DTYPE)
    assert_eager_attention(REQUIRED_ATTN_IMPLEMENTATION)
    load_kwargs = q1_model_load_kwargs(
        dtype=REQUIRED_TORCH_DTYPE,
        attn_implementation=REQUIRED_ATTN_IMPLEMENTATION,
    )
    if expected_adapter_targets is None or expected_adapted_layers is None:
        raise RuntimeError("engineering load requires explicit targets/layers")

    unit_bind = bind_unit_checkpoint_provenance(
        family=family_from_sf,
        seed=seed_from_sf,
        subfolder=subfolder,
        base_snapshot_dir=base_weight_snapshot_dir,
        adapter_snapshot_dir=adapter_weight_snapshot_dir,
        expected_sha256_by_relpath=digests,
        require_section11_tokenizer_pins=False,
        require_authoritative_trust=True,
        trust_auth=auth,
    )
    local_paths = unit_bind["local_paths"]
    expected_ad = resolve_unit_adapter_subfolder_dir(
        adapter_weight_snapshot_dir, subfolder
    )
    computed_pre = dict(unit_bind["computed_sha256"])
    if not computed_pre:
        raise RuntimeError("computed digests missing after engineering bind (D5)")

    tok_class = FROZEN_TOKENIZER_IDENTITY["expected_tokenizer_class"]
    tokenizer = AutoTokenizer.from_pretrained(
        local_paths["adapter_load_path"],
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
    )
    if type(tokenizer).__name__ != tok_class:
        raise RuntimeError(
            f"tokenizer class {type(tokenizer).__name__} != frozen {tok_class} (D5-B)"
        )
    if bool(getattr(tokenizer, "is_fast", True)):
        raise RuntimeError("tokenizer is_fast=True refused (D5-B)")

    model = AutoModelForCausalLM.from_pretrained(
        local_paths["base_load_path"],
        torch_dtype=load_kwargs["torch_dtype"],
        device_map=device_map,
        attn_implementation=load_kwargs["attn_implementation"],
        trust_remote_code=False,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        model,
        local_paths["adapter_load_path"],
        autocast_adapter_dtype=False,
        local_files_only=True,
    )
    model.eval()
    _disable_dropout(model)
    torch.set_grad_enabled(False)

    post = verify_post_load(
        model,
        tokenizer,
        expected_base_revision=base_revision,
        expected_adapter_revision=adapter_revision,
        actual_subfolder=subfolder,
        expected_adapter_targets=expected_adapter_targets,
        expected_adapted_layers=expected_adapted_layers,
        expected_tokenizer_class=tok_class,
        expected_tokenizer_is_fast=False,
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
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        intermediate_size=int(getattr(cfg, "intermediate_size", 0)),
        adapted_modules=sorted(list(getattr(peft_cfg, "target_modules", []) or [])),
        lora_r=int(peft_cfg.r),
        lora_alpha=int(peft_cfg.lora_alpha),
        lora_scaling=float(peft_cfg.lora_alpha) / float(peft_cfg.r),
        adapter_tensor_count=_count_adapter_params(model)[0],
        adapter_param_count=_count_adapter_params(model)[1],
        dtype=str(REQUIRED_TORCH_DTYPE),
        device=str(next(model.parameters()).device),
        notes=[Q1_LOADER_NOTES, "engineering_synthetic_load_only"],
    )
    meta = {
        "engineering_only": True,
        "production_unlocked": False,
        "computed_weight_file_sha256": computed_pre,
        "hub_fallback_refused": True,
        "local_files_only": True,
        "unit_checkpoint_provenance": {
            "family": family_from_sf,
            "seed": seed_from_sf,
            "subfolder": subfolder,
            "adapter_subfolder_dir": unit_bind["adapter_subfolder_dir"],
        },
        "tokenizer_template_files": unit_bind["tokenizer_template_files"],
        "adapter_config": unit_bind.get("adapter_config"),
        "local_snapshot_load_paths": local_paths,
        "checkpoint_trust": auth.to_dict(),
        "base_revision": base_revision,
        "adapter_revision": adapter_revision,
    }
    setattr(model, "_q1_adapter_weight_dir", str(expected_ad))
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
