"""F4 exploratory family-specific numerical-repeat calibration (development only).

Produces genuine comparison grids for the four required tensor families under
frozen Q1 v1.3 F4 rules. Does **not** authorize confirmatory / held-out runs,
identities 1–9 closure, double-BOS, or ``run_live_scientific_pipeline``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from traces.q1.adapter_off import adapter_off_plan
from traces.q1.empirical_evidence import (
    ARTIFACT_F4_V1,
    F4_FORMULA,
    F4_REQUIRED_N_COMPARISONS_PER_CELL,
    F4_REQUIRED_N_REPEATS,
    REQUIRED_F4_TENSOR_FAMILIES,
    frozen_exploratory_model_units,
    validate_f4_tensor_family_calibration,
)
from traces.q1.f4 import epsilon_from_dmax, pairwise_max_abs_diff
from traces.q1.frozen_artifacts import (
    EXPECTED_N_DEV,
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    REQUIRED_EXPLORATORY_SEEDS,
    load_ordered_dolly_ids,
    ordered_id_sequence_sha256,
)
from traces.q1.margin import (
    PAYLOAD_TOKEN_A,
    clean_reference_token,
    decision_margin,
    last_prompt_logits,
)

F4_STAGE_NAME = "f4_exploratory_family_calibration"
F4_FAMILY = "l1523"
F4_SEEDS: tuple[int, ...] = tuple(REQUIRED_EXPLORATORY_SEEDS)  # (42, 43, 44)
F4_N_PROMPTS = EXPECTED_N_DEV  # 50
F4_N_REPEATS = F4_REQUIRED_N_REPEATS  # 3
F4_BATCH_SIZE = 1
F4_BOS = "single"
F4_BOUNDARY_LAYER = 19  # residual hybrid / identity boundary
F4_UPSTREAM_BOUNDARY = adapter_off_plan("l1523").upstream_boundary_layer  # 15 exclusive
F4_BATCHING_PADDING_POLICY = (
    "bs1|no_pad|single_bos|eager|bf16|autocast_adapter_dtype=False|use_cache=False|"
    f"boundary_residual_L{F4_BOUNDARY_LAYER}_last_pos|"
    f"upstream_o_proj_L0_to_L{F4_UPSTREAM_BOUNDARY - 1}_last_pos_concat"
)
INTENDED_IMAGE_ID = "im-0j3j9DzkkZSLdRQTxl4scU"
PINNED_RUNTIME_LOCK_SHA256 = (
    "f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72"
)
PINNED_CHECKPOINT_MANIFEST_SHA256 = (
    "d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55"
)
DEV_INSTRUCTIONS_RELPATH = "results/q1_selection/development_instructions.json"
HELD_OUT_INSTRUCTIONS_RELPATH = "results/q1_selection/held_out_instructions.json"
SCI_AUTH_ENV = "TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION"
FORBIDDEN_PIPELINE_ENTRYPOINTS = (
    "run_live_scientific_pipeline",
    "run_live_scientific_pipeline_from_hooks",
)
L1523_ADAPTED_LAYERS = tuple(range(15, 24))
SEVEN_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Internal non-main job limits (standing owner spend authorization; not a new ask).
# Conservative vs Stage C: ~28s/load ×3 + 900 intact TF+capture × ~0.35s + gates/copy.
F4_CAP_USD = 2.0
F4_TIMEOUT_SEC = 45 * 60
F4_GPU = "A10G"
F4_MAX_WORKERS = 1

# Stage C measured anchors used for transparent budgeting (seed42 A10G).
STAGE_C_LOAD_WALL_S = 27.93
STAGE_C_INTACT_TF_CAPTURE_MEAN_S = 0.2108


@dataclass(frozen=True)
class F4Scope:
    stage: str = F4_STAGE_NAME
    family: str = F4_FAMILY
    seeds: tuple[int, ...] = F4_SEEDS
    n_prompts: int = F4_N_PROMPTS
    n_repeats: int = F4_N_REPEATS
    batch_size: int = F4_BATCH_SIZE
    bos_convention: str = F4_BOS
    boundary_layer: int = F4_BOUNDARY_LAYER
    upstream_boundary_layer: int = F4_UPSTREAM_BOUNDARY
    batching_padding_policy: str = F4_BATCHING_PADDING_POLICY
    image_id: str = INTENDED_IMAGE_ID
    runtime_lock_sha256: str = PINNED_RUNTIME_LOCK_SHA256
    checkpoint_manifest_sha256: str = PINNED_CHECKPOINT_MANIFEST_SHA256
    cap_usd: float = F4_CAP_USD
    timeout_sec: int = F4_TIMEOUT_SEC
    gpu: str = F4_GPU
    max_workers: int = F4_MAX_WORKERS
    allows_held_out: bool = False
    allows_live_scientific_pipeline: bool = False
    allows_confirmatory_models: bool = False
    tensor_families: tuple[str, ...] = tuple(sorted(REQUIRED_F4_TENSOR_FAMILIES))


DEFAULT_F4_SCOPE = F4Scope()


def f4_model_unit(seed: int) -> str:
    return f"exploratory_l1523_seed{int(seed)}"


def f4_subfolder(seed: int) -> str:
    return f"l1523/seed{int(seed)}"


def assert_f4_scope(scope: F4Scope = DEFAULT_F4_SCOPE) -> list[str]:
    failures: list[str] = []
    if scope.stage != F4_STAGE_NAME:
        failures.append(f"stage {scope.stage!r} != {F4_STAGE_NAME!r}")
    if scope.family != "l1523":
        failures.append("only exploratory l1523 authorized for F4")
    if tuple(scope.seeds) != F4_SEEDS:
        failures.append(f"seeds {scope.seeds} != frozen exploratory {F4_SEEDS}")
    if scope.n_prompts != F4_N_PROMPTS:
        failures.append(f"n_prompts {scope.n_prompts} != {F4_N_PROMPTS}")
    if scope.n_repeats != F4_N_REPEATS:
        failures.append(f"n_repeats {scope.n_repeats} != {F4_N_REPEATS}")
    if scope.batch_size != 1:
        failures.append("batch_size must be 1")
    if scope.bos_convention != "single":
        failures.append("bos_convention must be single")
    if scope.boundary_layer != F4_BOUNDARY_LAYER:
        failures.append(f"boundary_layer must be {F4_BOUNDARY_LAYER}")
    if scope.upstream_boundary_layer != F4_UPSTREAM_BOUNDARY:
        failures.append(f"upstream_boundary must be {F4_UPSTREAM_BOUNDARY}")
    if set(scope.tensor_families) != set(REQUIRED_F4_TENSOR_FAMILIES):
        failures.append("tensor_families drift vs REQUIRED_F4_TENSOR_FAMILIES")
    if scope.image_id != INTENDED_IMAGE_ID:
        failures.append(f"image_id drift vs {INTENDED_IMAGE_ID}")
    if scope.runtime_lock_sha256 != PINNED_RUNTIME_LOCK_SHA256:
        failures.append("runtime lock SHA drift")
    if scope.checkpoint_manifest_sha256 != PINNED_CHECKPOINT_MANIFEST_SHA256:
        failures.append("checkpoint manifest SHA drift")
    if scope.cap_usd > F4_CAP_USD + 1e-12:
        failures.append(f"cap_usd {scope.cap_usd} exceeds internal limit {F4_CAP_USD}")
    if scope.timeout_sec > F4_TIMEOUT_SEC:
        failures.append(f"timeout_sec {scope.timeout_sec} > {F4_TIMEOUT_SEC}")
    if scope.gpu not in {"A10G", "A10"}:
        failures.append(f"gpu {scope.gpu!r} not A10/A10G")
    if scope.max_workers != 1:
        failures.append("max_workers must be 1")
    if scope.allows_held_out:
        failures.append("held-out access forbidden in F4 exploratory")
    if scope.allows_live_scientific_pipeline:
        failures.append("live scientific pipeline forbidden in F4 exploratory")
    if scope.allows_confirmatory_models:
        failures.append("confirmatory models forbidden in F4 exploratory")
    return failures


def refuse_held_out_access(path: Path | str | None) -> None:
    if path is None:
        return
    p = Path(path)
    name = p.name.lower()
    as_posix = p.as_posix().lower()
    if "held_out" in name or "held-out" in as_posix or "/held_out" in as_posix:
        raise PermissionError(f"F4 exploratory refuses held-out path access: {p}")


def scientific_flag_must_be_unset_outside_runner() -> bool:
    return os.environ.get(SCI_AUTH_ENV) not in {"1", "true", "TRUE", "yes", "YES"}


def activate_f4_scientific_flag(*, verified_scope: bool) -> None:
    if not verified_scope:
        raise PermissionError("refusing scientific flag without verified F4 scope")
    os.environ[SCI_AUTH_ENV] = "1"


def deactivate_f4_scientific_flag() -> None:
    os.environ.pop(SCI_AUTH_ENV, None)


def refuse_confirmatory_callback(name: str) -> None:
    if name in FORBIDDEN_PIPELINE_ENTRYPOINTS:
        raise PermissionError(f"F4 exploratory forbids confirmatory entrypoint {name!r}")


def estimate_f4_workload(
    *,
    load_wall_s: float = STAGE_C_LOAD_WALL_S,
    intact_tf_capture_s: float = STAGE_C_INTACT_TF_CAPTURE_MEAN_S,
    capture_overhead_factor: float = 1.6,
    usd_per_hour_a10g: float = 1.10,
    startup_s: float = 180.0,
) -> dict[str, Any]:
    """Transparent conservative budget from measured Stage C anchors."""
    n_seeds = len(F4_SEEDS)
    n_fwd = n_seeds * F4_N_PROMPTS * 2 * F4_N_REPEATS  # 900
    per_fwd = float(intact_tf_capture_s) * float(capture_overhead_factor)
    compute_s = n_seeds * float(load_wall_s) + n_fwd * per_fwd + float(startup_s)
    # gates/copy/validation pad
    compute_s *= 1.25
    usd = (compute_s / 3600.0) * float(usd_per_hour_a10g)
    return {
        "n_seeds": n_seeds,
        "n_prompts": F4_N_PROMPTS,
        "n_tags": 2,
        "n_repeats": F4_N_REPEATS,
        "n_intact_forwards": n_fwd,
        "n_comparisons_per_family": n_seeds
        * F4_N_PROMPTS
        * 2
        * F4_REQUIRED_N_COMPARISONS_PER_CELL,
        "n_families": len(REQUIRED_F4_TENSOR_FAMILIES),
        "stage_c_load_wall_s": load_wall_s,
        "stage_c_intact_tf_capture_mean_s": intact_tf_capture_s,
        "assumed_per_forward_s": per_fwd,
        "estimated_wall_s": compute_s,
        "estimated_wall_min": compute_s / 60.0,
        "estimated_usd_a10g": usd,
        "internal_cap_usd": F4_CAP_USD,
        "internal_timeout_sec": F4_TIMEOUT_SEC,
        "within_internal_limits": usd <= F4_CAP_USD and compute_s <= F4_TIMEOUT_SEC,
        "usd_per_hour_assumption": usd_per_hour_a10g,
        "notes": (
            "Conservative: Stage C intact TF+capture mean ×1.6 for family extract, "
            "×1.25 pad for copy/gates; A10G list price assumption for planning only."
        ),
    }


def select_f4_development_prompts(repo_root: Path | str) -> dict[str, Any]:
    """Load all 50 frozen development instructions; never open held-out."""
    root = Path(repo_root)
    held = root / HELD_OUT_INSTRUCTIONS_RELPATH
    held_out_present = held.is_file()
    path = root / DEV_INSTRUCTIONS_RELPATH
    if not path.is_file():
        raise FileNotFoundError(f"missing development instructions: {path}")
    ordered_ids = load_ordered_dolly_ids(path)
    if len(ordered_ids) != F4_N_PROMPTS:
        raise RuntimeError(
            f"development allocation size {len(ordered_ids)} != {F4_N_PROMPTS}"
        )
    full_sha = ordered_id_sequence_sha256(ordered_ids)
    if full_sha != FROZEN_DEV_ID_SEQUENCE_SHA256:
        raise RuntimeError(
            "development ID sequence SHA mismatch vs frozen baseline "
            f"{full_sha} != {FROZEN_DEV_ID_SEQUENCE_SHA256}"
        )
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len(rows) != len(ordered_ids):
        raise RuntimeError("development_instructions.json shape invalid")
    by_id = {int(r["dolly_id"]): r for r in rows}
    selected = []
    for did in ordered_ids:
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
        "selection_rule": "full_frozen_development_order",
        "n": F4_N_PROMPTS,
        "dolly_ids": list(ordered_ids),
        "prompts": selected,
        "full_development_id_sequence_sha256": full_sha,
        "held_out_file_present_but_unread": held_out_present,
        "held_out_bytes_read": 0,
    }


def _last_pos(t: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Return float32 last-prompt-position vector; accepts [B,S,*] or [S,*]."""
    x = t.detach()
    if x.ndim >= 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim < 1:
        raise ValueError(f"unexpected tensor ndim for last_pos: {tuple(t.shape)}")
    if prompt_len < 1 or prompt_len > x.shape[0]:
        raise ValueError(f"prompt_len={prompt_len} invalid for seq={x.shape[0]}")
    return x[prompt_len - 1].float().contiguous()


def extract_f4_family_tensors(
    *,
    logits: torch.Tensor,
    residuals_by_layer: Mapping[int, torch.Tensor],
    o_proj_input_by_layer: Mapping[int, torch.Tensor],
    prompt_len: int,
    b_i: int,
    boundary_layer: int = F4_BOUNDARY_LAYER,
    upstream_boundary_layer: int = F4_UPSTREAM_BOUNDARY,
) -> dict[str, torch.Tensor]:
    """Canonical last-position family tensors (fixed shapes across prompts)."""
    last_logits = last_prompt_logits(logits, prompt_len).float().contiguous()
    if boundary_layer not in residuals_by_layer:
        raise KeyError(f"missing boundary residual layer {boundary_layer}")
    boundary = _last_pos(residuals_by_layer[boundary_layer], prompt_len)
    ups: list[torch.Tensor] = []
    for L in range(0, int(upstream_boundary_layer)):
        if L not in o_proj_input_by_layer:
            raise KeyError(f"missing upstream o_proj input layer {L}")
        ups.append(_last_pos(o_proj_input_by_layer[L], prompt_len).reshape(-1))
    upstream = torch.cat(ups, dim=0).contiguous()
    margin = float(decision_margin(last_logits, int(b_i), payload_a=PAYLOAD_TOKEN_A))
    return {
        "final_logits": last_logits,
        "decision_margin": torch.tensor([margin], dtype=torch.float32),
        "boundary_residual": boundary,
        "upstream_activation": upstream,
    }


def freeze_b_i_from_clean_logits(
    logits: torch.Tensor,
    prompt_len: int,
    *,
    instruction_id: str,
    model_unit: str,
) -> int:
    last = last_prompt_logits(logits, prompt_len)
    ref = clean_reference_token(
        last,
        instruction_id=str(instruction_id),
        model_unit=model_unit,
        bos_convention=F4_BOS,
        payload_a=PAYLOAD_TOKEN_A,
    )
    return int(ref.token_id)


@dataclass
class FamilyAccum:
    """Collect three repeats per (model, prompt, tag) then emit pairwise diffs.

    When ``allow_variable_shapes`` is True (identity-applicable full-sequence
    residuals/activations), shapes must match within a cell but may differ
    across prompts; family-level ``shape`` records the shape policy.
    """

    family: str
    shape: tuple[int, ...] | None = None
    dtype: str = "float32"
    allow_variable_shapes: bool = False
    batching_padding_policy: str = F4_BATCHING_PADDING_POLICY
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    per_model_maxima: dict[str, float] = field(default_factory=dict)
    observed_shapes: list[list[int]] = field(default_factory=list)
    _pending: dict[tuple[str, str, str], list[torch.Tensor]] = field(
        default_factory=dict
    )
    _cell_shapes: dict[tuple[str, str, str], tuple[int, ...]] = field(
        default_factory=dict
    )

    def add_repeat(
        self,
        *,
        model_unit: str,
        prompt_id: str,
        tag: str,
        tensor: torch.Tensor,
    ) -> None:
        t = tensor.detach().float().cpu().contiguous()
        key = (model_unit, str(prompt_id), tag)
        shape = tuple(t.shape)
        if self.allow_variable_shapes:
            locked = self._cell_shapes.get(key)
            if locked is None:
                self._cell_shapes[key] = shape
                if shape not in {tuple(s) for s in self.observed_shapes}:
                    self.observed_shapes.append(list(shape))
            elif shape != locked:
                raise ValueError(
                    f"{self.family}: in-cell shape {shape} != {locked} for {key}"
                )
        else:
            if self.shape is None:
                self.shape = shape
            elif shape != self.shape:
                raise ValueError(
                    f"{self.family}: shape {shape} != locked {self.shape}"
                )
        if not torch.isfinite(t).all():
            raise ValueError(f"{self.family}: nonfinite tensor for {model_unit}/{prompt_id}/{tag}")
        bucket = self._pending.setdefault(key, [])
        if len(bucket) >= F4_N_REPEATS:
            raise RuntimeError(f"{self.family}: >{F4_N_REPEATS} repeats for {key}")
        bucket.append(t)
        if len(bucket) == F4_N_REPEATS:
            self._emit_cell(model_unit, str(prompt_id), tag, bucket)
            del self._pending[key]

    def _emit_cell(
        self,
        model_unit: str,
        prompt_id: str,
        tag: str,
        reps: Sequence[torch.Tensor],
    ) -> None:
        for ri, rj in ((0, 1), (0, 2), (1, 2)):
            d = float((reps[ri] - reps[rj]).abs().max().item())
            row: dict[str, Any] = {
                "model_unit": model_unit,
                "prompt_id": str(prompt_id),
                "tag": tag,
                "repeat_i": ri,
                "repeat_j": rj,
                "max_abs_diff": d,
            }
            if self.allow_variable_shapes:
                row["shape"] = list(reps[ri].shape)
            self.comparisons.append(row)
            prev = self.per_model_maxima.get(model_unit, 0.0)
            self.per_model_maxima[model_unit] = max(prev, d)

    def finalize(self) -> dict[str, Any]:
        if self._pending:
            raise RuntimeError(
                f"{self.family}: incomplete cells still pending: {list(self._pending)[:5]}"
            )
        if self.allow_variable_shapes:
            if not self.observed_shapes:
                raise RuntimeError(f"{self.family}: no tensors recorded")
            shape_field: Any = {
                "policy": "per_cell_native_sequence",
                "observed_shapes": list(self.observed_shapes),
            }
        else:
            if self.shape is None:
                raise RuntimeError(f"{self.family}: no tensors recorded")
            shape_field = list(self.shape)
        dmax = max((float(c["max_abs_diff"]) for c in self.comparisons), default=0.0)
        for u in frozen_exploratory_model_units():
            self.per_model_maxima.setdefault(u, 0.0)
        return {
            "shape": shape_field,
            "dtype": self.dtype,
            "batching_padding_policy": self.batching_padding_policy,
            "comparisons": list(self.comparisons),
            "d_max_family": float(dmax),
            "epsilon_family": float(epsilon_from_dmax(dmax)),
            "per_model_maxima": {
                u: float(self.per_model_maxima[u]) for u in sorted(self.per_model_maxima)
            },
        }


def build_f4_artifact_payload(
    *,
    tensor_families: Mapping[str, Mapping[str, Any]],
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    locked: bool = True,
    claims_real_experiment: bool = True,
    fixture_purpose: str = "exploratory_f4_family_calibration",
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    units = list(model_units) if model_units is not None else list(
        frozen_exploratory_model_units()
    )
    missing = sorted(REQUIRED_F4_TENSOR_FAMILIES - set(str(k) for k in tensor_families))
    if missing:
        raise ValueError(f"missing required tensor families {missing}")
    return {
        "artifact_type": ARTIFACT_F4_V1,
        "claims_real_experiment": bool(claims_real_experiment),
        "fixture_purpose": str(fixture_purpose),
        "model_units": units,
        "dataset_binding": {
            "split": "development",
            "id_sequence_sha256": dataset_sha,
        },
        "locked": bool(locked),
        "tensor_families": {k: dict(v) for k, v in tensor_families.items()},
        "formula": F4_FORMULA,
        "meta": dict(meta or {}),
    }


def write_f4_artifact(path: Path | str, payload: Mapping[str, Any]) -> tuple[Path, str]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    p.write_text(text)
    p.with_suffix(p.suffix + ".sha256").write_text(digest + "\n")
    return p, digest


def build_synthetic_complete_f4_payload(
    *,
    prompt_ids: Sequence[Any],
    d_max_by_family: Mapping[str, float] | None = None,
    corrupt: str | None = None,
) -> dict[str, Any]:
    """CPU fixture covering the full 900-cell grid (schema/negative tests only)."""
    units = list(frozen_exploratory_model_units())
    prompts = [str(x) for x in prompt_ids]
    dmap = {fam: 0.0 for fam in REQUIRED_F4_TENSOR_FAMILIES}
    if d_max_by_family:
        dmap.update({str(k): float(v) for k, v in d_max_by_family.items()})

    tensor_families: dict[str, Any] = {}
    for fam in sorted(REQUIRED_F4_TENSOR_FAMILIES):
        comps: list[dict[str, Any]] = []
        per_model: dict[str, float] = {u: 0.0 for u in units}
        base_diff = float(dmap[fam])
        for u in units:
            for pid in prompts:
                for tag in ("T", "C"):
                    for ri, rj in ((0, 1), (0, 2), (1, 2)):
                        if (
                            corrupt == "missing_repeat"
                            and u == units[0]
                            and pid == prompts[0]
                            and tag == "T"
                            and (ri, rj) == (1, 2)
                        ):
                            continue
                        diff = base_diff
                        u_emit = u
                        pid_emit = pid
                        if (
                            corrupt == "wrong_model"
                            and u == units[0]
                            and pid == prompts[0]
                            and tag == "T"
                            and (ri, rj) == (0, 1)
                        ):
                            u_emit = "exploratory_l1523_seed999"
                        if (
                            corrupt == "wrong_prompt"
                            and u == units[0]
                            and pid == prompts[0]
                            and tag == "T"
                            and (ri, rj) == (0, 1)
                        ):
                            pid_emit = "held_out_99999"
                        if (
                            corrupt == "nonfinite"
                            and u == units[0]
                            and pid == prompts[0]
                            and tag == "T"
                            and (ri, rj) == (0, 1)
                        ):
                            diff = float("nan")
                        row = {
                            "model_unit": u_emit,
                            "prompt_id": pid_emit,
                            "tag": tag,
                            "repeat_i": ri,
                            "repeat_j": rj,
                            "max_abs_diff": diff,
                        }
                        comps.append(row)
                        if (
                            corrupt == "duplicate_cell"
                            and u == units[0]
                            and pid == prompts[0]
                            and tag == "T"
                            and (ri, rj) == (0, 1)
                        ):
                            comps.append(dict(row))
                        if u_emit == u and math.isfinite(diff):
                            per_model[u] = max(per_model[u], diff)
        finite = [float(c["max_abs_diff"]) for c in comps if _finite(c["max_abs_diff"])]
        dmax = max(finite) if finite else 0.0
        if corrupt == "inconsistent_maxima":
            per_model[units[0]] = dmax + 1.0
        if corrupt == "wrong_epsilon":
            eps = 999.0
        else:
            eps = epsilon_from_dmax(dmax)
        shape = [8] if fam != "decision_margin" else [1]
        if corrupt == "wrong_shape" and fam == "final_logits":
            # shape field itself is informational; coverage still required
            shape = [8, 8]
        tensor_families[fam] = {
            "shape": shape,
            "dtype": "float32",
            "batching_padding_policy": "synthetic_regression",
            "comparisons": comps,
            "d_max_family": dmax if corrupt != "nonfinite_dmax" else float("nan"),
            "epsilon_family": eps,
            "per_model_maxima": per_model,
        }

    if corrupt == "wrong_family":
        tensor_families.pop("upstream_activation", None)
        tensor_families["not_a_family"] = {
            "shape": [1],
            "dtype": "float32",
            "batching_padding_policy": "synthetic_regression",
            "comparisons": [],
            "d_max_family": 0.0,
            "epsilon_family": 1e-3,
            "per_model_maxima": {u: 0.0 for u in units},
        }

    # Bypass build_f4_artifact_payload required-family check so corrupt fixtures
    # can omit families for negative tests.
    payload = {
        "artifact_type": ARTIFACT_F4_V1,
        "claims_real_experiment": False,
        "fixture_purpose": "synthetic_regression_only",
        "model_units": units,
        "dataset_binding": {
            "split": "development",
            "id_sequence_sha256": FROZEN_DEV_ID_SEQUENCE_SHA256,
        },
        "locked": corrupt != "unlocked",
        "tensor_families": tensor_families,
        "formula": F4_FORMULA,
        "meta": {"corrupt": corrupt},
    }
    if corrupt == "held_out_binding":
        payload["dataset_binding"] = {
            "split": "held_out",
            "id_sequence_sha256": "0" * 64,
        }
    return payload


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def validate_f4_payload_or_raise(
    payload: Mapping[str, Any],
    *,
    frozen_prompt_ids: Sequence[Any],
) -> None:
    fails = validate_f4_tensor_family_calibration(
        payload, frozen_prompt_ids=[str(x) for x in frozen_prompt_ids]
    )
    if fails:
        raise RuntimeError("F4 validation failed: " + "; ".join(fails[:12]))


def pairwise_family_dmax(tensors: Sequence[torch.Tensor]) -> float:
    return pairwise_max_abs_diff(tensors)


def scope_dict(scope: F4Scope = DEFAULT_F4_SCOPE) -> dict[str, Any]:
    return asdict(scope)
