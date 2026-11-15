"""Load google/gemma-2-2b + dense-LoRA PEFT adapter (l1523 family).

Never silently substitutes the sparse TopKLoRA repository or a different base.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "google/gemma-2-2b"
DENSE_ADAPTER_REPO = "interpretable-finetuning/topklora-gemma-2-2b-dense-lora"
SPARSE_ADAPTER_REPO = "interpretable-finetuning/topklora-gemma-2-2b"
DEFAULT_SUBFOLDER = "l1523/seed42"

# Calibrated Stage 1 pins (fail closed — never unresolved main for scientific loads).
PINNED_BASE_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
PINNED_ADAPTER_REVISION = "6303a74fc832a34d584999364817268aefd9623b"

# Tokens / tags for logging (values are public research tags, not secrets)
TRIGGER_TAG = "|TRIGGER|"
CONTROL_TAG = "|TRAINING|"
EOT_STRING = "<end_of_turn>"


@dataclass
class ModelLoadReport:
    base_model_id: str
    adapter_repo: str
    adapter_subfolder: str
    adapter_kind: str
    base_revision: str | None
    adapter_revision: str | None
    architecture: str | None
    num_layers: int | None
    hidden_size: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    intermediate_size: int | None
    adapted_modules: list[str] = field(default_factory=list)
    adapter_tensor_count: int | None = None
    adapter_param_count: int | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_scaling: float | None = None
    device: str | None = None
    dtype: str | None = None
    gpu_name: str | None = None
    gpu_memory_allocated_bytes: int | None = None
    gpu_memory_reserved_bytes: int | None = None
    trigger_token_ids: list[int] = field(default_factory=list)
    control_token_ids: list[int] = field(default_factory=list)
    eos_token_id: int | None = None
    eos_token: str | None = None
    eot_token_id: int | None = None
    eot_token: str | None = None
    eval_mode: bool = True
    dropout_disabled: bool = True
    topk_use_topk: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_hf_auth() -> str | None:
    """Return token from env if present; never log its value."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        logger.info("Hugging Face auth: token present via environment (value redacted)")
    else:
        logger.warning(
            "Hugging Face auth: no HF_TOKEN / HUGGING_FACE_HUB_TOKEN in environment"
        )
    return token


def _resolve_repo_revision(repo_id: str, token: str | None) -> str | None:
    try:
        api = HfApi(token=token)
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        return getattr(info, "sha", None)
    except Exception as e:
        logger.warning("Could not resolve revision for %s: %s", repo_id, type(e).__name__)
        return None


def _count_adapter_params(model: PeftModel) -> tuple[int, int]:
    tensors = 0
    params = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            tensors += 1
            params += p.numel()
    return tensors, params


def _adapted_module_names(peft_config: PeftConfig) -> list[str]:
    tm = getattr(peft_config, "target_modules", None)
    if tm is None:
        return []
    if isinstance(tm, (list, set, tuple)):
        return sorted(str(x) for x in tm)
    return [str(tm)]


def _disable_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
            m.eval()


def _read_topk_use_flag(
    adapter_repo: str,
    subfolder: str,
    token: str | None,
    *,
    revision: str | None = None,
) -> bool | None:
    """Read topk_config.json from the pinned adapter revision (fail closed on main)."""
    try:
        import json

        if revision is None or revision in {"main", "None", ""}:
            raise RuntimeError(
                f"Refusing topk_config.json retrieval without pinned adapter "
                f"revision (got {revision!r}); require {PINNED_ADAPTER_REVISION}"
            )
        path = hf_hub_download(
            adapter_repo,
            "topk_config.json",
            subfolder=subfolder,
            token=token,
            revision=revision,
        )
        with open(path) as f:
            cfg = json.load(f)
        return bool(cfg.get("use_topk"))
    except RuntimeError:
        raise
    except Exception:
        return None


def load_dense_lora_model(
    base_model: str = BASE_MODEL_ID,
    adapter_repo: str = DENSE_ADAPTER_REPO,
    subfolder: str = DEFAULT_SUBFOLDER,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = False,
    base_revision: str | None = None,
    adapter_revision: str | None = None,
    require_pinned_revisions: bool = True,
    attn_implementation: str | None = None,
) -> tuple[PeftModel, Any, ModelLoadReport]:
    """Load base + dense PEFT adapter; tokenizer from adapter subfolder.

    Stage 0/1 legacy loader. Q1 scientific loads must use
    ``traces.q1.loader.load_q1_scientific_model`` (eager + autocast_adapter_dtype=False).

    Raises clearly if Gemma access fails. Never swaps in another model.
    Scientific Stage 1 pins refuse unresolved ``main`` when require_pinned_revisions.
    """
    if adapter_repo.rstrip("/") == SPARSE_ADAPTER_REPO or "dense-lora" not in adapter_repo:
        raise RuntimeError(
            f"Refusing adapter repo {adapter_repo!r}. "
            f"Stage 0 requires dense-LoRA control: {DENSE_ADAPTER_REPO}. "
            "Sparse TopKLoRA must never be silently substituted."
        )
    if base_model != BASE_MODEL_ID:
        raise RuntimeError(
            f"Base model must be {BASE_MODEL_ID}, got {base_model!r}. "
            "Silent substitution is not permitted."
        )

    token = _require_hf_auth()
    if require_pinned_revisions:
        base_rev = base_revision or PINNED_BASE_REVISION
        adapter_rev = adapter_revision or PINNED_ADAPTER_REVISION
        if not base_rev or base_rev in {"main", "None"}:
            raise RuntimeError(
                f"Refusing unresolved base revision {base_rev!r}; "
                f"require pinned {PINNED_BASE_REVISION}"
            )
        if not adapter_rev or adapter_rev in {"main", "None"}:
            raise RuntimeError(
                f"Refusing unresolved adapter revision {adapter_rev!r}; "
                f"require pinned {PINNED_ADAPTER_REVISION}"
            )
        if base_rev != PINNED_BASE_REVISION:
            raise RuntimeError(
                f"Base revision must be pinned {PINNED_BASE_REVISION}, got {base_rev!r}"
            )
        if adapter_rev != PINNED_ADAPTER_REVISION:
            raise RuntimeError(
                f"Adapter revision must be pinned {PINNED_ADAPTER_REVISION}, "
                f"got {adapter_rev!r}"
            )
    else:
        # Legacy smoke/engineering path only
        base_rev = base_revision or _resolve_repo_revision(base_model, token)
        adapter_rev = adapter_revision or _resolve_repo_revision(adapter_repo, token)

    use_topk = _read_topk_use_flag(
        adapter_repo, subfolder, token, revision=adapter_rev
    )
    if use_topk is True:
        raise RuntimeError(
            "topk_config.json reports use_topk=true; this loader is for dense PEFT only"
        )

    notes: list[str] = [
        "Activation interventions only; no training/optimizer steps.",
        "Tokenizer and chat template loaded from adapter subfolder.",
    ]

    try:
        tok_kwargs: dict[str, Any] = {
            "subfolder": subfolder,
            "token": token,
            "trust_remote_code": trust_remote_code,
        }
        if adapter_rev:
            tok_kwargs["revision"] = adapter_rev
        tokenizer = AutoTokenizer.from_pretrained(adapter_repo, **tok_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer from {adapter_repo}/{subfolder}: {e}"
        ) from e

    try:
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": base_rev,
        }
        # Stage 1 historical path: only pass attn_implementation when explicitly set.
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg or "access" in msg.lower():
            raise RuntimeError(
                f"Access to gated base model {base_model} unavailable. "
                "Accept the Gemma license on Hugging Face and provide HF_TOKEN "
                "(via env or Modal secret). Refusing silent model substitution."
            ) from e
        raise RuntimeError(f"Failed to load base model {base_model}: {e}") from e

    try:
        model = PeftModel.from_pretrained(
            model,
            adapter_repo,
            subfolder=subfolder,
            token=token,
            revision=adapter_rev,
        )
    except TypeError as e:
        # Newer adapter_config.json keys vs older installed peft.
        if "unexpected keyword argument" in str(e):
            raise RuntimeError(
                f"Failed to load dense PEFT adapter {adapter_repo}/{subfolder}: {e}. "
                "Adapter was saved with peft~=0.19.1; upgrade peft to >=0.19.1 "
                "(do not strip config silently in a way that changes adapter semantics)."
            ) from e
        raise RuntimeError(
            f"Failed to load dense PEFT adapter {adapter_repo}/{subfolder}: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dense PEFT adapter {adapter_repo}/{subfolder}: {e}"
        ) from e

    model.eval()
    _disable_dropout(model)
    torch.set_grad_enabled(False)

    peft_cfg = model.peft_config["default"]
    cfg = model.config
    n_q = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_q))
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_q))
    tensor_count, param_count = _count_adapter_params(model)
    r = int(peft_cfg.r)
    alpha = int(peft_cfg.lora_alpha)
    scaling = float(alpha) / float(r)

    device = next(model.parameters()).device
    mem_alloc = mem_res = gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        mem_alloc = int(torch.cuda.memory_allocated(device))
        mem_res = int(torch.cuda.memory_reserved(device))
        gpu_name = torch.cuda.get_device_name(device)

    trigger_ids = tokenizer.encode(TRIGGER_TAG, add_special_tokens=False)
    control_ids = tokenizer.encode(CONTROL_TAG, add_special_tokens=False)

    eot_id = None
    # Gemma uses <end_of_turn>; model card says id 107.
    if EOT_STRING in tokenizer.get_vocab():
        eot_id = tokenizer.convert_tokens_to_ids(EOT_STRING)
    elif hasattr(tokenizer, "additional_special_tokens"):
        for t in tokenizer.additional_special_tokens or []:
            if t == EOT_STRING:
                eot_id = tokenizer.convert_tokens_to_ids(t)

    report = ModelLoadReport(
        base_model_id=base_model,
        adapter_repo=adapter_repo,
        adapter_subfolder=subfolder,
        adapter_kind="dense_lora_peft",
        base_revision=base_rev,
        adapter_revision=adapter_rev,
        architecture=getattr(cfg, "architectures", [None])[0]
        if getattr(cfg, "architectures", None)
        else cfg.model_type,
        num_layers=int(cfg.num_hidden_layers),
        hidden_size=int(cfg.hidden_size),
        num_attention_heads=n_q,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        intermediate_size=int(cfg.intermediate_size),
        adapted_modules=_adapted_module_names(peft_cfg),
        adapter_tensor_count=tensor_count,
        adapter_param_count=param_count,
        lora_r=r,
        lora_alpha=alpha,
        lora_scaling=scaling,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        gpu_name=gpu_name,
        gpu_memory_allocated_bytes=mem_alloc,
        gpu_memory_reserved_bytes=mem_res,
        trigger_token_ids=list(trigger_ids),
        control_token_ids=list(control_ids),
        eos_token_id=tokenizer.eos_token_id,
        eos_token=tokenizer.eos_token,
        eot_token_id=eot_id,
        eot_token=EOT_STRING if eot_id is not None else None,
        eval_mode=True,
        dropout_disabled=True,
        topk_use_topk=use_topk,
        notes=notes,
    )

    logger.info(
        "Loaded %s + %s/%s | layers=%s hidden=%s heads=%s/%s intermediate=%s | "
        "lora r=%s alpha=%s scale=%.3f | adapter tensors=%s params=%s | device=%s dtype=%s",
        base_model,
        adapter_repo,
        subfolder,
        report.num_layers,
        report.hidden_size,
        report.num_attention_heads,
        report.num_key_value_heads,
        report.intermediate_size,
        report.lora_r,
        report.lora_alpha,
        report.lora_scaling or 0.0,
        report.adapter_tensor_count,
        report.adapter_param_count,
        report.device,
        report.dtype,
    )
    logger.info(
        "Tag token ids: trigger=%s control=%s | eos=%s eot=%s",
        report.trigger_token_ids,
        report.control_token_ids,
        report.eos_token_id,
        report.eot_token_id,
    )
    return model, tokenizer, report


def stopping_criteria_ids(tokenizer: Any, eot_token_id: int | None) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if eot_token_id is not None and eot_token_id not in ids:
        ids.append(int(eot_token_id))
    return ids
