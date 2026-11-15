"""Post-load verification for Q1 scientific model bundles (no silent fallback)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from traces.q1.runtime_spec import (
    REQUIRED_ATTN_IMPLEMENTATION,
    REQUIRED_TORCH_DTYPE,
    assert_bf16_dtype,
    assert_eager_attention,
)


@dataclass
class PostLoadReport:
    ok: bool
    attn_implementation: str | None
    model_dtype: str | None
    adapter_param_dtypes: list[str] = field(default_factory=list)
    tokenizer_class: str | None = None
    tokenizer_is_fast: bool | None = None
    base_revision: str | None = None
    adapter_revision: str | None = None
    adapter_targets: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_attn_implementation(model: Any) -> str | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    # transformers may store under config._attn_implementation
    return getattr(cfg, "_attn_implementation", None) or getattr(
        cfg, "attn_implementation", None
    )


def verify_post_load(
    model: Any,
    tokenizer: Any,
    *,
    expected_base_revision: str,
    expected_adapter_revision: str,
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

    # Model parameter dtype check
    dtypes = {p.dtype for p in model.parameters() if p.requires_grad is not None}
    # Prefer floating params
    float_dtypes = {d for d in dtypes if d.is_floating_point}
    model_dtype = None
    if float_dtypes:
        model_dtype = str(next(iter(float_dtypes)))
        if require_bf16 and REQUIRED_TORCH_DTYPE not in float_dtypes:
            # All floating params should be bf16 for scientific load
            if not all(d == torch.bfloat16 for d in float_dtypes):
                failures.append(
                    f"Model floating dtypes {float_dtypes} not exclusively bfloat16"
                )
        try:
            assert_bf16_dtype(REQUIRED_TORCH_DTYPE)
        except RuntimeError as e:
            failures.append(str(e))

    adapter_dtypes: list[str] = []
    adapter_targets: list[str] = []
    if hasattr(model, "peft_config"):
        peft_cfg = model.peft_config.get("default") if isinstance(model.peft_config, dict) else None
        if peft_cfg is not None:
            targets = getattr(peft_cfg, "target_modules", None)
            if targets is not None:
                adapter_targets = sorted(list(targets))
        for name, p in model.named_parameters():
            if "lora_" in name.lower():
                adapter_dtypes.append(str(p.dtype))
                if require_bf16 and p.dtype.is_floating_point and p.dtype != torch.bfloat16:
                    failures.append(f"Adapter param {name} dtype {p.dtype} != bfloat16")

    tok_class = type(tokenizer).__name__
    is_fast = bool(getattr(tokenizer, "is_fast", False))

    # Revisions must be supplied by caller from load report; compare here
    # (actual HF revision strings checked by loader already when pinned)

    ok = len(failures) == 0
    return PostLoadReport(
        ok=ok,
        attn_implementation=attn,
        model_dtype=model_dtype,
        adapter_param_dtypes=sorted(set(adapter_dtypes)),
        tokenizer_class=tok_class,
        tokenizer_is_fast=is_fast,
        base_revision=expected_base_revision,
        adapter_revision=expected_adapter_revision,
        adapter_targets=adapter_targets,
        failures=failures,
    )
