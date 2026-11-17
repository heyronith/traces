"""Stage C first real-Gemma exploratory calibration — scope guards + arm plan.

Owner-approved perimeter only (one A10G, $3, 30 minutes, l1523/seed42, four
development prompts). Does **not** authorize ``run_live_scientific_pipeline``,
held-out access, Stage D, or F4 family calibration claims.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from traces.model_loader import (
    BASE_MODEL_ID,
    DENSE_ADAPTER_REPO,
    PINNED_ADAPTER_REVISION,
    PINNED_BASE_REVISION,
)
from traces.q1.checkpoint_trust import (
    PINNED_CHECKPOINT_MANIFEST_SHA256,
    PRODUCTION_CHECKPOINT_MANIFEST_RELPATH,
)
from traces.q1.frozen_artifacts import (
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    load_ordered_dolly_ids,
    ordered_id_sequence_sha256,
)
from traces.q1.patching import HeadRef, NOMINATED_PAIR
from traces.q1.production_pipeline import (
    PROTOCOL_N_HEADS,
    PROTOCOL_UPSTREAM_LAYERS,
)
from traces.q1.runtime_lock import PINNED_RUNTIME_LOCK_SHA256

STAGE_C_NAME = "stage_c_first_real_gemma_exploratory"
STAGE_C_CAP_USD = 3.0
STAGE_C_TIMEOUT_SEC = 30 * 60
STAGE_C_GPU = "A10G"
STAGE_C_MAX_WORKERS = 1
STAGE_C_N_PROMPTS = 4
STAGE_C_TIMED_REPEATS = 3
STAGE_C_WARMUP_PASSES = 3
STAGE_C_MAX_NEW_TOKENS = 40
STAGE_C_STOP_TOKEN_IDS = (107, 1)
STAGE_C_BATCH_SIZE = 1
STAGE_C_SUBFOLDER = "l1523/seed42"
STAGE_C_FAMILY = "l1523"
STAGE_C_SEED = 42
STAGE_C_SELECTION_RULE = "first_n_frozen_development_order"

INTENDED_IMAGE_ID = "im-0j3j9DzkkZSLdRQTxl4scU"
RUNTIME_LOCK_RELPATH = (
    "results/q1_implementation_prep/TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json"
)
DEV_INSTRUCTIONS_RELPATH = "results/q1_selection/development_instructions.json"
HELD_OUT_INSTRUCTIONS_RELPATH = "results/q1_selection/held_out_instructions.json"

BASE_MATERIALIZED = (
    f"/mnt/b3/snapshots/base/{PINNED_BASE_REVISION}/materialized"
)
ADAPTER_MATERIALIZED = (
    f"/mnt/b3/snapshots/adapters/{PINNED_ADAPTER_REVISION}/materialized"
)
STAGE_B_WHEEL_DIR = "/mnt/stage_b/evidence/2026-11-17T154705Z/wheels"

SCI_AUTH_ENV = "TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION"
FORBIDDEN_PIPELINE_ENTRYPOINTS = (
    "run_live_scientific_pipeline",
    "run_live_scientific_pipeline_from_hooks",
)

SEVEN_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
L1523_ADAPTED_LAYERS = tuple(range(15, 24))
RESIDUAL_INJECTION_LAYERS = (8, 15, 19)
EIGHT_HEAD_LAYER = 15  # one upstream layer for 8-head joint
SINGLE_HEAD = NOMINATED_PAIR[0]  # L15.h6


@dataclass(frozen=True)
class StageCScope:
    stage: str = STAGE_C_NAME
    cap_usd: float = STAGE_C_CAP_USD
    timeout_sec: int = STAGE_C_TIMEOUT_SEC
    gpu: str = STAGE_C_GPU
    max_workers: int = STAGE_C_MAX_WORKERS
    n_prompts: int = STAGE_C_N_PROMPTS
    timed_repeats: int = STAGE_C_TIMED_REPEATS
    warmup_passes: int = STAGE_C_WARMUP_PASSES
    max_new_tokens: int = STAGE_C_MAX_NEW_TOKENS
    stop_token_ids: tuple[int, ...] = STAGE_C_STOP_TOKEN_IDS
    batch_size: int = STAGE_C_BATCH_SIZE
    subfolder: str = STAGE_C_SUBFOLDER
    family: str = STAGE_C_FAMILY
    seed: int = STAGE_C_SEED
    image_id: str = INTENDED_IMAGE_ID
    base_model: str = BASE_MODEL_ID
    adapter_repo: str = DENSE_ADAPTER_REPO
    base_revision: str = PINNED_BASE_REVISION
    adapter_revision: str = PINNED_ADAPTER_REVISION
    runtime_lock_sha256: str | None = PINNED_RUNTIME_LOCK_SHA256
    checkpoint_manifest_sha256: str | None = PINNED_CHECKPOINT_MANIFEST_SHA256
    claims_f4_tolerance_table: bool = False
    allows_held_out: bool = False
    allows_live_scientific_pipeline: bool = False


DEFAULT_SCOPE = StageCScope()


def stage_c_scope_dict(scope: StageCScope = DEFAULT_SCOPE) -> dict[str, Any]:
    return asdict(scope)


def assert_stage_c_scope(scope: StageCScope = DEFAULT_SCOPE) -> list[str]:
    """Return failures if constants drift outside owner approval."""
    failures: list[str] = []
    if scope.stage != STAGE_C_NAME:
        failures.append(f"stage {scope.stage!r} != {STAGE_C_NAME!r}")
    if scope.cap_usd > STAGE_C_CAP_USD + 1e-12:
        failures.append(f"cap_usd {scope.cap_usd} exceeds {STAGE_C_CAP_USD}")
    if scope.timeout_sec > STAGE_C_TIMEOUT_SEC:
        failures.append(f"timeout_sec {scope.timeout_sec} > {STAGE_C_TIMEOUT_SEC}")
    if scope.gpu not in {"A10G", "A10"}:
        failures.append(f"gpu {scope.gpu!r} not A10/A10G")
    if scope.max_workers != 1:
        failures.append("max_workers must be 1")
    if scope.n_prompts != 4:
        failures.append("n_prompts must be 4")
    if scope.timed_repeats > 3:
        failures.append("timed_repeats must be <= 3")
    if scope.batch_size != 1:
        failures.append("batch_size must be 1")
    if scope.max_new_tokens > 40:
        failures.append("max_new_tokens must be <= 40")
    if tuple(scope.stop_token_ids) != STAGE_C_STOP_TOKEN_IDS:
        failures.append(f"stop_token_ids {scope.stop_token_ids} != {STAGE_C_STOP_TOKEN_IDS}")
    if scope.subfolder != STAGE_C_SUBFOLDER:
        failures.append(f"subfolder {scope.subfolder!r} != {STAGE_C_SUBFOLDER!r}")
    if scope.family != "l1523" or scope.seed != 42:
        failures.append("only l1523/seed42 authorized")
    if scope.image_id != INTENDED_IMAGE_ID:
        failures.append(f"image_id {scope.image_id!r} != {INTENDED_IMAGE_ID!r}")
    if scope.runtime_lock_sha256 != (
        "f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72"
    ):
        failures.append("runtime lock SHA drift vs owner freeze")
    if scope.checkpoint_manifest_sha256 != (
        "d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55"
    ):
        failures.append("checkpoint manifest SHA drift vs owner pin")
    if scope.claims_f4_tolerance_table:
        failures.append("F4 tolerance table claim forbidden in Stage C")
    if scope.allows_held_out:
        failures.append("held-out access forbidden in Stage C")
    if scope.allows_live_scientific_pipeline:
        failures.append("run_live_scientific_pipeline forbidden in Stage C")
    return failures


def select_stage_c_development_prompts(
    repo_root: Path | str,
    *,
    n: int = STAGE_C_N_PROMPTS,
) -> dict[str, Any]:
    """Deterministically take the first *n* frozen development instructions.

    Never opens the held-out allocation file.
    """
    root = Path(repo_root)
    if n != STAGE_C_N_PROMPTS:
        raise ValueError(f"Stage C requires exactly {STAGE_C_N_PROMPTS} prompts, got n={n}")
    held = root / HELD_OUT_INSTRUCTIONS_RELPATH
    # Soft existence check only — must not *read* held-out content.
    held_out_present = held.is_file()
    path = root / DEV_INSTRUCTIONS_RELPATH
    if not path.is_file():
        raise FileNotFoundError(f"missing development instructions: {path}")
    ordered_ids = load_ordered_dolly_ids(path)
    if len(ordered_ids) < n:
        raise RuntimeError(f"development allocation too small: {len(ordered_ids)} < {n}")
    full_sha = ordered_id_sequence_sha256(ordered_ids)
    if full_sha != FROZEN_DEV_ID_SEQUENCE_SHA256:
        raise RuntimeError(
            "development ID sequence SHA mismatch vs frozen baseline "
            f"{full_sha} != {FROZEN_DEV_ID_SEQUENCE_SHA256}"
        )
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len(rows) != len(ordered_ids):
        raise RuntimeError("development_instructions.json shape invalid")
    # Bind by frozen order, not by re-sorting.
    by_id = {int(r["dolly_id"]): r for r in rows}
    selected_ids = list(ordered_ids[:n])
    selected = []
    for did in selected_ids:
        row = by_id.get(int(did))
        if row is None:
            raise RuntimeError(f"missing development row for dolly_id={did}")
        if str(row.get("split")) != "development":
            raise RuntimeError(f"non-development split for dolly_id={did}")
        selected.append(
            {
                "dolly_id": int(row["dolly_id"]),
                "instruction": str(row["instruction"]),
                "content_sha256": str(row["content_sha256"]),
                "category": row.get("category"),
                "split": "development",
            }
        )
    return {
        "selection_rule": STAGE_C_SELECTION_RULE,
        "n": n,
        "dolly_ids": selected_ids,
        "prompts": selected,
        "full_development_n": len(ordered_ids),
        "full_development_id_sequence_sha256": full_sha,
        "held_out_file_present_but_unread": held_out_present,
        "held_out_bytes_read": 0,
    }


def refuse_held_out_access(path: Path | str | None) -> None:
    if path is None:
        return
    p = Path(path)
    name = p.name.lower()
    as_posix = p.as_posix().lower()
    if "held_out" in name or "held-out" in as_posix or "/held_out" in as_posix:
        raise PermissionError(
            f"Stage C refuses held-out path access: {p}"
        )


def scientific_flag_must_be_unset_outside_runner() -> bool:
    return os.environ.get(SCI_AUTH_ENV) not in {"1", "true", "TRUE", "yes", "YES"}


def activate_stage_c_scientific_flag(*, verified_scope: bool) -> None:
    """Set scientific auth only after Stage C scope verification succeeded."""
    if not verified_scope:
        raise PermissionError("refusing scientific flag without verified Stage C scope")
    os.environ[SCI_AUTH_ENV] = "1"


def deactivate_stage_c_scientific_flag() -> None:
    os.environ.pop(SCI_AUTH_ENV, None)


def select_stage_c_joint_recipient(
    pair: Mapping[str, Any], direction: str
) -> dict[str, Any]:
    """Bind the *actual* recipient tokens, mask, tag and expected IDs together.

    Forward patches use T as recipient with clean C donors; reverse patches
    use C as recipient with trigger T donors. A tag-only change is unsafe:
    the live token IDs and full-sequence proof must also switch to C.
    This selector performs no model forwards and never accesses held-out data.
    """
    if direction not in ("forward", "reverse"):
        raise ValueError(f"unsupported Stage C joint direction: {direction!r}")
    tag = "T" if direction == "forward" else "C"
    ids_key = f"input_ids_{tag}"
    mask_key = f"attention_mask_{tag}"
    if ids_key not in pair or mask_key not in pair:
        raise ValueError(f"missing {tag} recipient tokens or attention mask")
    ids = pair[ids_key]
    mask = pair[mask_key]
    if ids is None or mask is None:
        raise ValueError(f"null {tag} recipient tokens or attention mask")
    try:
        # Q1 prompt batches are shape [1, N]. Use exactly those real
        # recipient bytes as the expected-full-IDs verifier input.
        if tuple(ids.shape) != tuple(mask.shape) or len(ids.shape) != 2 or ids.shape[0] != 1:
            raise ValueError("joint recipient must be an unpadded batch-one pair")
        full_ids = ids[0].detach().cpu().tolist()
        if not isinstance(full_ids, list) or not full_ids:
            raise ValueError("joint recipient full token IDs empty")
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("joint recipient tokens invalid") from exc
    return {
        "recipient_tag": tag,
        "recipient_input_ids": ids,
        "recipient_attention_mask": mask,
        "expected_full_ids": full_ids,
        "direction": direction,
    }


def all_upstream_heads() -> tuple[HeadRef, ...]:
    return tuple(
        HeadRef(L, h)
        for L in PROTOCOL_UPSTREAM_LAYERS
        for h in range(PROTOCOL_N_HEADS)
    )


def eight_head_layer_heads(layer: int = EIGHT_HEAD_LAYER) -> tuple[HeadRef, ...]:
    if layer not in PROTOCOL_UPSTREAM_LAYERS:
        raise ValueError(f"layer {layer} not upstream")
    return tuple(HeadRef(layer, h) for h in range(PROTOCOL_N_HEADS))


def stage_c_arm_catalog() -> list[dict[str, Any]]:
    """Descriptive calibration arms (GPU_THROUGHPUT_BENCHMARK_PROPOSAL)."""
    return [
        {"arm": "cold_load", "timed_repeats": 1, "per_prompt": False},
        {"arm": "warmup_intact_tf", "timed_repeats": 0, "passes": STAGE_C_WARMUP_PASSES},
        {"arm": "intact_tf_cache_capture", "timed_repeats": STAGE_C_TIMED_REPEATS},
        {
            "arm": "single_head_patch",
            "timed_repeats": STAGE_C_TIMED_REPEATS,
            "head": {"layer": SINGLE_HEAD.layer, "head_index": SINGLE_HEAD.head_index},
            "direction": "forward",
        },
        {
            "arm": "all_upstream_joint",
            "timed_repeats": STAGE_C_TIMED_REPEATS,
            "directions": ["forward", "reverse"],
            "n_heads": len(all_upstream_heads()),
        },
        {
            "arm": "eight_head_layer_joint",
            "timed_repeats": STAGE_C_TIMED_REPEATS,
            "layer": EIGHT_HEAD_LAYER,
            "directions": ["forward", "reverse"],
        },
        {
            "arm": "residual_injection",
            "timed_repeats": STAGE_C_TIMED_REPEATS,
            "layers": list(RESIDUAL_INJECTION_LAYERS),
            "hybrids": ["CT", "TC"],
        },
        {"arm": "adapter_off_on_tf", "timed_repeats": STAGE_C_TIMED_REPEATS},
        {
            "arm": "greedy_generation",
            "timed_repeats": STAGE_C_TIMED_REPEATS,
            "max_new_tokens": STAGE_C_MAX_NEW_TOKENS,
            "stop_token_ids": list(STAGE_C_STOP_TOKEN_IDS),
        },
    ]


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_stage_c_repo_artifacts(repo_root: Path | str) -> dict[str, Any]:
    """Verify owner-pinned runtime-lock raw bytes and checkpoint manifest digest.

    Checkpoint identity uses the same canonical-manifest digest as
    ``authenticate_production_checkpoint_manifest`` (not the raw JSON file SHA).
    """
    from traces.q1.checkpoint_trust import authenticate_production_checkpoint_manifest

    root = Path(repo_root)
    failures: list[str] = []
    lock_path = root / RUNTIME_LOCK_RELPATH
    if not lock_path.is_file():
        failures.append(f"missing runtime lock {lock_path}")
        lock_sha = None
    else:
        lock_sha = sha256_file(lock_path)
        if lock_sha != PINNED_RUNTIME_LOCK_SHA256:
            failures.append(
                f"runtime lock SHA {lock_sha} != pin {PINNED_RUNTIME_LOCK_SHA256}"
            )
    man_path = root / PRODUCTION_CHECKPOINT_MANIFEST_RELPATH
    trust = authenticate_production_checkpoint_manifest(repo_root=root)
    man_sha = trust.computed_manifest_sha256
    if not trust.production_unlocked:
        failures.extend(list(trust.failures or []) or ["checkpoint trust blocked"])
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "runtime_lock_path": str(lock_path),
        "runtime_lock_sha256": lock_sha,
        "checkpoint_manifest_path": str(man_path),
        "checkpoint_manifest_sha256": man_sha,
        "checkpoint_production_unlocked": bool(trust.production_unlocked),
    }


def verify_running_image_identity(expected: str = INTENDED_IMAGE_ID) -> dict[str, Any]:
    observed = os.environ.get("MODAL_IMAGE_ID")
    return {
        "ok": bool(observed and observed == expected),
        "expected": expected,
        "observed": observed,
    }


@dataclass
class ArmTiming:
    arm: str
    prompt_dolly_id: int | None
    repeat: int
    wall_seconds: float
    peak_vram_bytes: int | None
    ok: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def summarize_arm_timings(rows: Sequence[ArmTiming]) -> dict[str, Any]:
    by_arm: dict[str, list[ArmTiming]] = {}
    for r in rows:
        by_arm.setdefault(r.arm, []).append(r)
    out: dict[str, Any] = {}
    for arm, items in by_arm.items():
        ok_items = [x for x in items if x.ok]
        walls = [x.wall_seconds for x in ok_items]
        peaks = [x.peak_vram_bytes for x in ok_items if x.peak_vram_bytes is not None]
        out[arm] = {
            "n_ok": len(ok_items),
            "n_fail": sum(1 for x in items if not x.ok),
            "wall_mean_s": (sum(walls) / len(walls)) if walls else None,
            "wall_max_s": max(walls) if walls else None,
            "peak_vram_max_bytes": max(peaks) if peaks else None,
        }
    return out
