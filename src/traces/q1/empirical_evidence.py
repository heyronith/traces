"""Empirical evidence validators (S2-1…S2-4 schema + synthetic negative coverage).

Production scientific unlock remains BLOCKED until independently authenticated
evidence and owner-approved provenance exist. This module implements:

- One-way trust (artifacts do not embed manifest hashes)
- F4 tensor-family calibration checks (v1.3 formula)
- Identity outcome recomputation (ignore supplied pass booleans)
- Double-BOS prompt-level evidence validation

Synthetic fixtures are for schema/negative tests only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from traces.q1.analysis import rank_pairs_signed
from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS
from traces.q1.conditions import enumerate_double_bos_descriptive_pairs
from traces.q1.empirical_trust import (
    ARTIFACT_PROVENANCE_MANIFEST_V1,
    PRODUCTION_MANIFEST_RELPATH,
    authenticate_battery_manifest,
    canonical_manifest_bytes,
    load_production_manifest,
    sha256_bytes,
)
from traces.q1.f4 import epsilon_from_dmax
from traces.q1.frozen_artifacts import (
    EXPECTED_N_DEV,
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    REQUIRED_EXPLORATORY_SEEDS,
    load_ordered_dolly_ids,
    ordered_id_sequence_sha256,
)

# Frozen identity-7 adapter-off ASR threshold (v1.3); ignore artifact-supplied thr.
IDENTITY_7_ASR_THRESHOLD = 0.01
# Frozen payload token for double-BOS eligibility (v1.3)
DOUBLE_BOS_PAYLOAD_TOKEN_A = 235285

ARTIFACT_IDENTITY_V1 = "q1_empirical_identity_suite_v1"
ARTIFACT_F4_V1 = "q1_empirical_f4_tolerance_table_v1"
ARTIFACT_DOUBLE_BOS_V1 = "q1_empirical_double_bos_64_pair_scan_v1"

REQUIRED_IDENTITY_IDS = frozenset(range(1, 10))
F4_FORMULA = "epsilon_family = max(1e-3, 10 * D_max_family)"
F4_REQUIRED_N_REPEATS = 3
F4_REQUIRED_N_COMPARISONS_PER_CELL = 3  # C(3,2)

# Tensor families (not model families) — identities 1–6 + margins
REQUIRED_F4_TENSOR_FAMILIES = frozenset(
    {
        "final_logits",
        "decision_margin",
        "boundary_residual",
        "upstream_activation",
    }
)

# Identity → required tensor families for numerical comparisons (v1.3).
# IDs 2–3 compare logits and boundary residual; ID 6 compares activation + residual.
IDENTITY_TENSOR_FAMILIES: dict[int, frozenset[str]] = {
    1: frozenset({"final_logits"}),
    2: frozenset({"final_logits", "boundary_residual"}),
    3: frozenset({"final_logits", "boundary_residual"}),
    4: frozenset({"boundary_residual"}),
    5: frozenset({"boundary_residual"}),
    6: frozenset({"upstream_activation", "boundary_residual"}),
    8: frozenset({"final_logits"}),
}
# Back-compat alias: primary family for single-family identities / fixtures
IDENTITY_TENSOR_FAMILY: dict[int, str] = {
    iid: sorted(fams)[0] for iid, fams in IDENTITY_TENSOR_FAMILIES.items()
}

SYNTHETIC_MARKERS = frozenset(
    {"synthetic_regression_only", "fabricated_regression", "schema_test_only"}
)


def frozen_exploratory_model_units() -> tuple[str, ...]:
    return tuple(f"exploratory_l1523_seed{s}" for s in REQUIRED_EXPLORATORY_SEEDS)


def expected_exploratory_model_units() -> tuple[str, ...]:
    return frozen_exploratory_model_units()


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


@dataclass
class ArtifactAuthResult:
    ok: bool
    artifact_type: str
    path: str | None = None
    computed_sha256: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    production_blocked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "computed_sha256": self.computed_sha256,
            "failures": list(self.failures),
            "production_blocked": self.production_blocked,
        }


def _load_json(
    path: Path | str, *, claimed_sha256: str | None, expected_type: str
) -> ArtifactAuthResult:
    p = Path(path)
    if not p.is_file():
        return ArtifactAuthResult(
            False, expected_type, str(p), failures=[f"artifact file missing: {p}"]
        )
    raw = p.read_bytes()
    computed = sha256_bytes(raw)
    if not claimed_sha256 or str(claimed_sha256).lower() != computed.lower():
        return ArtifactAuthResult(
            False,
            expected_type,
            str(p),
            computed,
            failures=[
                "claimed sha256 missing or mismatch "
                f"(claimed={claimed_sha256} computed={computed})"
            ],
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return ArtifactAuthResult(
            False, expected_type, str(p), computed, failures=[f"JSON parse: {e}"]
        )
    if not isinstance(payload, dict):
        return ArtifactAuthResult(
            False, expected_type, str(p), computed, failures=["root must be object"]
        )
    if str(payload.get("artifact_type", "")) != expected_type:
        return ArtifactAuthResult(
            False,
            expected_type,
            str(p),
            computed,
            payload,
            failures=[
                f"artifact_type {payload.get('artifact_type')!r} != {expected_type!r}"
            ],
        )
    # Artifacts must NOT embed provenance manifest hash (one-way structure)
    if "provenance_manifest_sha256" in payload or "manifest_sha256" in payload:
        return ArtifactAuthResult(
            False,
            expected_type,
            str(p),
            computed,
            payload,
            failures=[
                "artifact must not embed provenance manifest hash "
                "(circular trust refused)"
            ],
        )
    return ArtifactAuthResult(
        True, expected_type, str(p), computed, payload, production_blocked=True
    )


def _immutable_bindings(payload: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    units = payload.get("model_units")
    exp = list(frozen_exploratory_model_units())
    if not isinstance(units, list) or sorted(str(u) for u in units) != sorted(exp):
        failures.append(f"model_units must equal frozen exploratory {exp}")
    for seed in REQUIRED_EXPLORATORY_SEEDS:
        if not isinstance(units, list) or not any(f"seed{seed}" in str(u) for u in units):
            failures.append(f"missing frozen exploratory seed {seed}")
    binding = payload.get("dataset_binding")
    if not isinstance(binding, Mapping):
        failures.append("dataset_binding missing")
    else:
        if str(binding.get("split", "")) != "development":
            failures.append("dataset_binding.split must be development")
        if str(binding.get("id_sequence_sha256", "")) != FROZEN_DEV_ID_SEQUENCE_SHA256:
            failures.append("dataset_binding.id_sequence_sha256 != frozen constant")
    return failures


def _reject_synthetic_for_production(payload: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("claims_real_experiment") is not True:
        failures.append(
            "production requires claims_real_experiment=True with approved provenance"
        )
    purpose = str(payload.get("fixture_purpose", "") or "")
    if purpose in SYNTHETIC_MARKERS or "synthetic" in purpose.lower() or "fabricat" in purpose.lower():
        failures.append(f"synthetic fixture_purpose={purpose!r} rejected")
    blob = json.dumps(payload, sort_keys=True).lower()
    for m in ("fabricated_regression", "synthetic_regression_only", "schema_test_only"):
        if m in blob:
            failures.append(f"synthetic marker {m!r} in artifact")
            break
    return failures


# ---------------------------------------------------------------------------
# S2-2 F4 tensor-family validation
# ---------------------------------------------------------------------------


def validate_f4_tensor_family_calibration(
    payload: Mapping[str, Any],
    *,
    frozen_prompt_ids: Sequence[str],
) -> list[str]:
    """Schema + numeric F4 checks keyed by tensor family (not model family)."""
    failures: list[str] = []
    if str(payload.get("formula", "")) != F4_FORMULA:
        failures.append(f"F4 formula must be exact {F4_FORMULA!r}")
    if not bool(payload.get("locked")):
        failures.append("F4 locked!=True")

    families = payload.get("tensor_families")
    if not isinstance(families, Mapping):
        failures.append("tensor_families missing (must not key by l1523/l19 model family)")
        return failures
    # Reject legacy model-family keys at top level if used instead
    if "families" in payload and isinstance(payload["families"], Mapping):
        legacy = set(str(k) for k in payload["families"])
        if legacy & {"l1523", "l19"} and not (
            REQUIRED_F4_TENSOR_FAMILIES & set(str(k) for k in families)
        ):
            failures.append(
                "F4 must be keyed by tensor family, not l1523/l19 model family"
            )

    missing = sorted(REQUIRED_F4_TENSOR_FAMILIES - set(str(k) for k in families))
    if missing:
        failures.append(f"missing required tensor families {missing}")

    units = list(frozen_exploratory_model_units())
    prompts = [str(x) for x in frozen_prompt_ids]
    if len(prompts) != EXPECTED_N_DEV:
        failures.append(f"frozen_prompt_ids length {len(prompts)} != {EXPECTED_N_DEV}")
    else:
        try:
            seq = ordered_id_sequence_sha256([int(x) for x in prompts])
            if seq != FROZEN_DEV_ID_SEQUENCE_SHA256:
                failures.append(
                    "F4 frozen_prompt_ids ordered-ID-sequence SHA256 != "
                    "FROZEN_DEV_ID_SEQUENCE_SHA256 (reject non-allocation 50-prompt sets)"
                )
        except (TypeError, ValueError):
            failures.append(
                "F4 frozen_prompt_ids not coercible to ints for allocation SHA"
            )

    for fam_name, rec in families.items():
        if not isinstance(rec, Mapping):
            failures.append(f"tensor family {fam_name!r} not a mapping")
            continue
        for req in ("shape", "dtype", "batching_padding_policy", "comparisons"):
            if req not in rec:
                failures.append(f"{fam_name}: missing {req}")
        comps = rec.get("comparisons")
        if not isinstance(comps, list):
            failures.append(f"{fam_name}: comparisons not a list")
            continue

        # Recompute D_max from comparisons
        dmax_calc = 0.0
        seen_cells: set[tuple[str, str, str, int, int]] = set()
        for c in comps:
            if not isinstance(c, Mapping):
                failures.append(f"{fam_name}: comparison row not mapping")
                continue
            for k in ("model_unit", "prompt_id", "tag", "repeat_i", "repeat_j", "max_abs_diff"):
                if k not in c:
                    failures.append(f"{fam_name}: comparison missing {k}")
            if not _is_finite(c.get("max_abs_diff")):
                failures.append(f"{fam_name}: nonfinite max_abs_diff")
                continue
            diff = float(c["max_abs_diff"])
            dmax_calc = max(dmax_calc, diff)
            try:
                ri, rj = int(c["repeat_i"]), int(c["repeat_j"])
            except (TypeError, ValueError, KeyError):
                failures.append(f"{fam_name}: malformed repeat indices")
                continue
            if not (0 <= ri < F4_REQUIRED_N_REPEATS and 0 <= rj < F4_REQUIRED_N_REPEATS):
                failures.append(f"{fam_name}: repeat index out of range")
            if ri >= rj:
                failures.append(f"{fam_name}: require repeat_i < repeat_j")
            cell = (
                str(c.get("model_unit")),
                str(c.get("prompt_id")),
                str(c.get("tag")),
                ri,
                rj,
            )
            if cell in seen_cells:
                failures.append(f"{fam_name}: duplicate comparison {cell}")
            seen_cells.add(cell)
            if str(c.get("tag")) not in {"T", "C"}:
                failures.append(f"{fam_name}: tag must be T or C")

        # Expected cell coverage: 3 models × 50 prompts × 2 tags × 3 pairs
        expected_n = (
            len(units) * len(prompts) * 2 * F4_REQUIRED_N_COMPARISONS_PER_CELL
        )
        if len(seen_cells) != expected_n:
            failures.append(
                f"{fam_name}: comparison coverage {len(seen_cells)} != "
                f"required {expected_n} "
                f"(3 seeds × {EXPECTED_N_DEV} prompts × 2 tags × 3 repeat pairs)"
            )
        # Prompt order / completeness per model×tag
        for u in units:
            for tag in ("T", "C"):
                for pid in prompts:
                    for ri, rj in ((0, 1), (0, 2), (1, 2)):
                        if (u, pid, tag, ri, rj) not in seen_cells:
                            failures.append(
                                f"{fam_name}: missing comparison "
                                f"{u}/{pid}/{tag}/{ri}-{rj}"
                            )
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

        if "d_max_family" not in rec or not _is_finite(rec["d_max_family"]):
            failures.append(f"{fam_name}: missing/nonfinite d_max_family")
        else:
            declared = float(rec["d_max_family"])
            if abs(declared - dmax_calc) > 1e-12:
                failures.append(
                    f"{fam_name}: d_max_family {declared} != recomputed {dmax_calc}"
                )
            expected_eps = epsilon_from_dmax(dmax_calc)
            if "epsilon_family" not in rec or not _is_finite(rec["epsilon_family"]):
                failures.append(f"{fam_name}: missing/nonfinite epsilon_family")
            elif abs(float(rec["epsilon_family"]) - expected_eps) > 1e-12:
                failures.append(
                    f"{fam_name}: epsilon_family {rec['epsilon_family']} != "
                    f"max(1e-3, 10*D_max)={expected_eps}"
                )

        maxima = rec.get("per_model_maxima")
        if not isinstance(maxima, Mapping):
            failures.append(f"{fam_name}: per_model_maxima missing")
        else:
            for u in units:
                if u not in maxima or not _is_finite(maxima[u]):
                    failures.append(f"{fam_name}: missing/nonfinite maxima for {u}")
                else:
                    # maxima must equal max of that model's comparison diffs
                    model_diffs = [
                        float(c["max_abs_diff"])
                        for c in comps
                        if isinstance(c, Mapping)
                        and str(c.get("model_unit")) == u
                        and _is_finite(c.get("max_abs_diff"))
                    ]
                    if model_diffs and abs(float(maxima[u]) - max(model_diffs)) > 1e-12:
                        failures.append(
                            f"{fam_name}: per_model_maxima[{u}] inconsistent "
                            f"with comparisons"
                        )
    return failures


def authenticate_f4_table_artifact(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    frozen_prompt_ids: Sequence[Any] | None = None,
    repo_root: Path | str | None = None,
    sibling_artifact_digests: Mapping[str, str] | None = None,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
    check_production_unlock: bool = True,
    reject_synthetic: bool = True,
) -> ArtifactAuthResult:
    del expected_model_units, expected_dataset_sha
    base = _load_json(path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_F4_V1)
    failures = list(base.failures)
    if base.computed_sha256 is None:
        return base

    if frozen_prompt_ids is None and repo_root is not None:
        frozen_prompt_ids = load_ordered_dolly_ids(
            Path(repo_root) / "results/q1_selection/development_instructions.json"
        )
    prompts = [str(x) for x in (frozen_prompt_ids or [])]
    failures.extend(_immutable_bindings(base.payload))
    failures.extend(
        validate_f4_tensor_family_calibration(base.payload, frozen_prompt_ids=prompts)
    )
    if reject_synthetic:
        failures.extend(_reject_synthetic_for_production(base.payload))
    if check_production_unlock:
        failures.extend(
            _require_production_unlock(
                repo_root=repo_root,
                digests={
                    **dict(sibling_artifact_digests or {}),
                    ARTIFACT_F4_V1: base.computed_sha256,
                },
            )
        )
    return ArtifactAuthResult(
        ok=len(failures) == 0,
        artifact_type=ARTIFACT_F4_V1,
        path=base.path,
        computed_sha256=base.computed_sha256,
        payload=base.payload,
        failures=failures,
        production_blocked=len(failures) > 0,
    )


def _require_production_unlock(
    *,
    repo_root: Path | str | None,
    digests: Mapping[str, str],
) -> list[str]:
    if repo_root is None:
        return [
            "repo_root required for production provenance "
            "(artifact-controlled manifest paths refused)"
        ]
    auth = load_production_manifest(repo_root=repo_root, artifact_digests=digests)
    if auth.production_unlocked and auth.ok:
        return []
    return list(auth.failures) or ["production empirical unlock blocked"]


def _extract_f4_epsilons_after_full_schema(
    *,
    f4_artifact_path: Path | str,
    f4_claimed_sha256: str | None,
    frozen_prompt_ids: Sequence[Any] | None,
) -> tuple[dict[str, float], str | None, list[str]]:
    """Validate complete F4 schema before extracting any epsilon."""
    failures: list[str] = []
    f4_base = _load_json(
        f4_artifact_path,
        claimed_sha256=f4_claimed_sha256,
        expected_type=ARTIFACT_F4_V1,
    )
    if not f4_base.ok or not f4_base.computed_sha256:
        return {}, None, list(f4_base.failures) or ["bound F4 calibration artifact failed"]
    prompts = [str(x) for x in (frozen_prompt_ids or [])]
    schema_fails = validate_f4_tensor_family_calibration(
        f4_base.payload, frozen_prompt_ids=prompts
    )
    schema_fails.extend(_immutable_bindings(f4_base.payload))
    if schema_fails:
        return {}, f4_base.computed_sha256, schema_fails
    f4_eps: dict[str, float] = {}
    tfams = f4_base.payload.get("tensor_families")
    if not isinstance(tfams, Mapping):
        return {}, f4_base.computed_sha256, ["F4 tensor_families missing after schema"]
    for fam in REQUIRED_F4_TENSOR_FAMILIES:
        rec = tfams.get(fam)
        if not isinstance(rec, Mapping) or not _is_finite(rec.get("epsilon_family")):
            failures.append(f"F4 missing locked epsilon_family for {fam}")
        else:
            f4_eps[str(fam)] = float(rec["epsilon_family"])
    return f4_eps, f4_base.computed_sha256, failures


# ---------------------------------------------------------------------------
# S2-3 Identity recomputation
# ---------------------------------------------------------------------------


def validate_identity_suite_measurements(
    payload: Mapping[str, Any],
    *,
    f4_epsilons: Mapping[str, float],
) -> list[str]:
    """Recompute identity outcomes; ignore supplied passed/within_tolerance."""
    failures: list[str] = []
    per_model = payload.get("per_model_results")
    units = list(frozen_exploratory_model_units())
    if not isinstance(per_model, Mapping):
        return ["per_model_results missing"]

    # Bind to authenticated F4 calibration artifact digest
    f4_bind = payload.get("f4_calibration_artifact_sha256")
    if not f4_bind:
        failures.append(
            "identity artifact must bind f4_calibration_artifact_sha256 "
            "(authenticated F4 digest)"
        )

    for unit in units:
        rows = per_model.get(unit)
        if not isinstance(rows, list):
            failures.append(f"{unit}: missing identity rows")
            continue
        seen: set[int] = set()
        by_id: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                failures.append(f"{unit}: row not mapping")
                continue
            try:
                iid = int(row["identity_id"])
            except (KeyError, TypeError, ValueError):
                failures.append(f"{unit}: bad identity_id")
                continue
            if iid in seen:
                failures.append(f"{unit}: duplicate identity_id={iid}")
                continue
            seen.add(iid)
            by_id[iid] = row

        missing = sorted(REQUIRED_IDENTITY_IDS - seen)
        if missing:
            failures.append(f"{unit}: missing identities {missing}")

        for iid in sorted(seen & REQUIRED_IDENTITY_IDS):
            row = by_id[iid]
            failures.extend(
                _recompute_one_identity(unit, iid, row, f4_epsilons=f4_epsilons)
            )
    return failures


def _recompute_one_identity(
    unit: str,
    iid: int,
    row: Mapping[str, Any],
    *,
    f4_epsilons: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    # Identities 7–9: categorical / ASR / hooks — not generic numerical
    if iid == 7:
        try:
            t_asr = float(row["trigger_asr"])
            c_asr = float(row["control_asr"])
        except (KeyError, TypeError, ValueError):
            return [f"{unit}: identity 7 missing ASR fields"]
        if not (_is_finite(t_asr) and _is_finite(c_asr)):
            return [f"{unit}: identity 7 ASR rates must be finite"]
        if not (0.0 <= t_asr <= 1.0 and 0.0 <= c_asr <= 1.0):
            return [f"{unit}: identity 7 ASR rates must be in [0, 1]"]
        # Ignore artifact-supplied threshold for the decision; reject contradictions
        if "threshold" in row:
            try:
                declared = float(row["threshold"])
            except (TypeError, ValueError):
                return [f"{unit}: identity 7 malformed threshold declaration"]
            if abs(declared - IDENTITY_7_ASR_THRESHOLD) > 1e-15:
                failures.append(
                    f"{unit}: identity 7 declared threshold {declared} contradicts "
                    f"frozen {IDENTITY_7_ASR_THRESHOLD}"
                )
        thr = IDENTITY_7_ASR_THRESHOLD
        recomputed = (t_asr <= thr) and (c_asr <= thr)
        if row.get("passed") is True and not recomputed:
            failures.append(
                f"{unit}: identity 7 passed=True contradicts recomputed ASR gate"
            )
        if not recomputed:
            failures.append(
                f"{unit}: identity 7 ASR gate failed "
                f"(require trigger<= {thr} and control<= {thr}; "
                f"got trigger={t_asr}, control={c_asr})"
            )
        return failures

    if iid == 9:
        try:
            hooks = int(row["active_hook_count"])
        except (KeyError, TypeError, ValueError):
            return [f"{unit}: identity 9 missing active_hook_count"]
        if hooks != 0:
            failures.append(f"{unit}: identity 9 recomputed hooks={hooks} != 0")
        return failures

    # Numerical identities 1–6, 8 (multi-family where protocol requires it)
    required = IDENTITY_TENSOR_FAMILIES.get(iid)
    if required is None:
        return [f"{unit}: identity {iid} has no tensor-family mapping"]
    comps = row.get("numerical_comparisons")
    if not isinstance(comps, list) or not comps:
        return [f"{unit}: identity {iid} missing numerical_comparisons"]

    all_ok = True
    seen_families: set[str] = set()
    for c in comps:
        if not isinstance(c, Mapping):
            failures.append(f"{unit}: identity {iid} comparison not mapping")
            all_ok = False
            continue
        tf = str(c.get("tensor_family", ""))
        if tf not in required:
            failures.append(
                f"{unit}: identity {iid} tensor_family {tf!r} not in "
                f"required {sorted(required)}"
            )
            all_ok = False
        else:
            seen_families.add(tf)
        if tf not in f4_epsilons:
            failures.append(
                f"{unit}: identity {iid} no locked epsilon for tensor family {tf!r}"
            )
            all_ok = False
            continue
        eps = float(f4_epsilons[tf])
        if not _is_finite(c.get("max_abs_diff")):
            failures.append(f"{unit}: identity {iid} nonfinite max_abs_diff")
            all_ok = False
            continue
        diff = float(c["max_abs_diff"])
        recomputed_within = diff <= eps
        # Do not trust supplied booleans
        if c.get("within_tolerance") is True and not recomputed_within:
            failures.append(
                f"{unit}: identity {iid} within_tolerance=true contradicts "
                f"recomputed max_abs_diff={diff} > epsilon={eps}"
            )
            all_ok = False
        if not recomputed_within:
            failures.append(
                f"{unit}: identity {iid} fails max_abs_diff={diff} <= eps={eps}"
            )
            all_ok = False
        # Shape / categorical extras when present
        if "shape_a" in c and "shape_b" in c and c["shape_a"] != c["shape_b"]:
            failures.append(f"{unit}: identity {iid} shape mismatch")
            all_ok = False

    missing_fams = sorted(required - seen_families)
    if missing_fams:
        failures.append(
            f"{unit}: identity {iid} missing required tensor-family "
            f"comparisons {missing_fams}"
        )
        all_ok = False

    if row.get("passed") is True and not all_ok:
        failures.append(
            f"{unit}: identity {iid} passed=True contradicts recomputed outcome"
        )
    return failures


def authenticate_identity_suite_artifact(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    f4_artifact_path: Path | str | None = None,
    f4_claimed_sha256: str | None = None,
    frozen_prompt_ids: Sequence[Any] | None = None,
    repo_root: Path | str | None = None,
    sibling_artifact_digests: Mapping[str, str] | None = None,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
    check_production_unlock: bool = True,
    reject_synthetic: bool = True,
) -> ArtifactAuthResult:
    del expected_model_units, expected_dataset_sha
    base = _load_json(path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_IDENTITY_V1)
    failures = list(base.failures)
    if base.computed_sha256 is None:
        return base

    failures.extend(_immutable_bindings(base.payload))
    if reject_synthetic:
        failures.extend(_reject_synthetic_for_production(base.payload))

    # Bound F4 is mandatory — never substitute identity-embedded f4_tolerances_used
    f4_eps: dict[str, float] = {}
    f4_digest = None
    if f4_artifact_path is None or not f4_claimed_sha256:
        failures.append(
            "identity authentication requires bound F4 artifact path + digest"
        )
    else:
        f4_eps, f4_digest, f4_fails = _extract_f4_epsilons_after_full_schema(
            f4_artifact_path=f4_artifact_path,
            f4_claimed_sha256=f4_claimed_sha256,
            frozen_prompt_ids=frozen_prompt_ids,
        )
        if f4_fails:
            failures.extend(
                [f"bound F4 validation failed: {f}" for f in f4_fails]
            )
            # Do not apply any epsilon extracted from an invalid F4
            f4_eps = {}
        elif f4_digest:
            bound = str(base.payload.get("f4_calibration_artifact_sha256", ""))
            if not bound or bound.lower() != f4_digest.lower():
                failures.append(
                    "identity f4_calibration_artifact_sha256 does not match "
                    "independently computed F4 artifact digest"
                )

    # Never trust identity-embedded tolerances when F4 is missing/invalid
    _ = base.payload.get("f4_tolerances_used")

    if f4_eps:
        failures.extend(
            validate_identity_suite_measurements(base.payload, f4_epsilons=f4_eps)
        )
    else:
        failures.append(
            "identity numerical validation blocked: no trusted F4 epsilons"
        )

    if check_production_unlock:
        digests = {
            **(sibling_artifact_digests or {}),
            ARTIFACT_IDENTITY_V1: base.computed_sha256,
        }
        if f4_digest:
            digests[ARTIFACT_F4_V1] = f4_digest
        failures.extend(_require_production_unlock(repo_root=repo_root, digests=digests))

    return ArtifactAuthResult(
        ok=len(failures) == 0,
        artifact_type=ARTIFACT_IDENTITY_V1,
        path=base.path,
        computed_sha256=base.computed_sha256,
        payload=base.payload,
        failures=failures,
        production_blocked=len(failures) > 0,
    )


# ---------------------------------------------------------------------------
# S2-4 Double-BOS complete evidence
# ---------------------------------------------------------------------------


def _parse_pair_key(k: Any) -> tuple[int, int] | None:
    try:
        if isinstance(k, str) and "," in k:
            a, b = k.split(",", 1)
            return int(a), int(b)
        if isinstance(k, (list, tuple)) and len(k) == 2:
            return int(k[0]), int(k[1])
    except (TypeError, ValueError):
        return None
    return None


def validate_double_bos_scan_measurements(
    payload: Mapping[str, Any],
    *,
    frozen_prompt_ids: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    expected_n = len(enumerate_double_bos_descriptive_pairs())
    assert expected_n == 64
    if str(payload.get("bos_convention", "")) != "double":
        failures.append("bos_convention must be double")
    if payload.get("alters_single_bos_stopping_rule") is True:
        failures.append(
            "double-BOS findings must remain descriptive; "
            "must not alter single-BOS exploratory stopping rule"
        )

    prompts = [str(x) for x in frozen_prompt_ids]
    if len(prompts) != EXPECTED_N_DEV:
        failures.append(f"need {EXPECTED_N_DEV} frozen development prompts")
    try:
        seq = ordered_id_sequence_sha256([int(x) for x in prompts])
        if seq != FROZEN_DEV_ID_SEQUENCE_SHA256:
            failures.append("prompt ID sequence != frozen development order")
    except (TypeError, ValueError):
        failures.append("prompt IDs not coercible to ints for sequence hash")

    clean_prov = payload.get("clean_reference_provenance")
    if not isinstance(clean_prov, Mapping) or not clean_prov.get("frozen"):
        failures.append("clean_reference_provenance.frozen required")
    if isinstance(clean_prov, Mapping) and str(clean_prov.get("bos_convention", "")) != "double":
        failures.append("clean references must be intact double-BOS (not single-BOS)")
    if isinstance(clean_prov, Mapping) and clean_prov.get("borrowed_single_bos") is True:
        failures.append("borrowed single-BOS references rejected")

    per_seed = payload.get("per_seed_results")
    units = list(frozen_exploratory_model_units())
    if not isinstance(per_seed, Mapping):
        return failures + ["per_seed_results missing"]

    for unit in units:
        seed_rec = per_seed.get(unit)
        if not isinstance(seed_rec, Mapping):
            failures.append(f"{unit}: missing per_seed_results")
            continue
        got_ids = seed_rec.get("prompt_ids")
        if not isinstance(got_ids, list) or [str(x) for x in got_ids] != prompts:
            failures.append(f"{unit}: prompt_ids must match frozen development order")

        clean_refs = seed_rec.get("intact_double_bos_clean_margins")
        if not isinstance(clean_refs, Mapping) or set(str(k) for k in clean_refs) != set(
            prompts
        ):
            failures.append(
                f"{unit}: intact_double_bos_clean_margins must cover all 50 prompts"
            )
        elif any(not _is_finite(v) for v in clean_refs.values()):
            failures.append(f"{unit}: nonfinite clean double-BOS margins")

        # Intact double-BOS TT margins required — no fallback to clean
        tt_map = seed_rec.get("intact_double_bos_tt_margins")
        if not isinstance(tt_map, Mapping) or set(str(k) for k in tt_map) != set(prompts):
            failures.append(
                f"{unit}: intact_double_bos_tt_margins required for all 50 prompts "
                "(no clean-margin fallback)"
            )
            tt_map = None
        elif any(not _is_finite(v) for v in tt_map.values()):
            failures.append(f"{unit}: nonfinite TT double-BOS margins")
            tt_map = None

        # Per-prompt b_i from intact double-BOS clean run
        b_i_map = seed_rec.get("b_i_token_ids_by_prompt")
        if not isinstance(b_i_map, Mapping) or set(str(k) for k in b_i_map) != set(
            prompts
        ):
            failures.append(
                f"{unit}: b_i_token_ids_by_prompt required for all 50 frozen prompts"
            )
            b_i_map = None

        pairs = seed_rec.get("pairs")
        if not isinstance(pairs, Mapping):
            failures.append(f"{unit}: pairs map missing")
            continue

        keyset: set[tuple[int, int]] = set()
        means: dict[tuple[int, int], float] = {}
        for k, prec in pairs.items():
            key = _parse_pair_key(k)
            if key is None:
                failures.append(f"{unit}: bad pair key {k!r}")
                continue
            if key in keyset:
                failures.append(f"{unit}: duplicate pair {key}")
            keyset.add(key)
            if not isinstance(prec, Mapping):
                failures.append(f"{unit}: pair {key} record not mapping")
                continue
            patched = prec.get("patched_margins_by_prompt")
            effects = prec.get("signed_effects_by_prompt")
            elig = prec.get("eligible_by_prompt")
            if not isinstance(patched, Mapping) or not isinstance(effects, Mapping):
                failures.append(
                    f"{unit} pair {key}: require prompt-level patched margins and effects"
                )
                continue
            if set(str(x) for x in patched) != set(prompts):
                failures.append(f"{unit} pair {key}: incomplete patched_margins prompts")
            if set(str(x) for x in effects) != set(prompts):
                failures.append(f"{unit} pair {key}: incomplete signed_effects prompts")
            if not isinstance(elig, Mapping) or set(str(x) for x in elig) != set(prompts):
                failures.append(
                    f"{unit} pair {key}: complete eligible_by_prompt map required"
                )
                continue

            # Recompute eligibility from a != b_i
            if b_i_map is not None:
                for pid in prompts:
                    try:
                        b_i = int(b_i_map[pid])
                    except (KeyError, TypeError, ValueError):
                        failures.append(f"{unit}: bad b_i for prompt {pid}")
                        continue
                    expected_elig = DOUBLE_BOS_PAYLOAD_TOKEN_A != b_i
                    got = elig.get(pid)
                    got_bool = (
                        got
                        if isinstance(got, bool)
                        else str(got).lower() == "true"
                    )
                    if got_bool is not expected_elig:
                        failures.append(
                            f"{unit} pair {key}: eligibility {pid} != "
                            f"recomputed a!=b_i "
                            f"(a={DOUBLE_BOS_PAYLOAD_TOKEN_A}, b_i={b_i})"
                        )

            # Effects exclusively from TT_double - patched_double
            if tt_map is not None:
                eligible_vals: list[float] = []
                for pid in prompts:
                    if pid not in patched or not _is_finite(patched[pid]):
                        failures.append(f"{unit} pair {key}: bad patched margin {pid}")
                        continue
                    expected_eff = float(tt_map[pid]) - float(patched[pid])
                    got_eff = effects.get(pid)
                    if not _is_finite(got_eff):
                        failures.append(f"{unit} pair {key}: nonfinite effect {pid}")
                    elif abs(float(got_eff) - expected_eff) > 1e-9:
                        failures.append(
                            f"{unit} pair {key}: signed effect {pid} != "
                            f"recomputed m_TT_double - m_patched_double"
                        )
                    elig_ok = elig.get(pid)
                    elig_bool = (
                        elig_ok
                        if isinstance(elig_ok, bool)
                        else str(elig_ok).lower() == "true"
                    )
                    if elig_bool and _is_finite(got_eff):
                        eligible_vals.append(float(got_eff))
                if eligible_vals:
                    means[key] = sum(eligible_vals) / len(eligible_vals)

        missing = EXPECTED_L15_L16_PAIRS - keyset
        extra = keyset - EXPECTED_L15_L16_PAIRS
        if missing:
            failures.append(f"{unit}: missing {len(missing)} prescribed pairs")
        if extra:
            failures.append(f"{unit}: {len(extra)} non-prescribed pairs")
        if len(keyset) != 64:
            failures.append(
                f"{unit}: require exactly 64 distinct pairs, got {len(keyset)}"
            )

        ranking_decl = seed_rec.get("pair_ranking")
        if not isinstance(ranking_decl, list) or not ranking_decl:
            failures.append(
                f"{unit}: explicit pair_ranking required "
                "(absent ranking rejected even if pair effects exist)"
            )
        elif len(means) == 64:
            recomputed = rank_pairs_signed(means)
            decl_pairs: list[tuple[int, int]] = []
            for item in ranking_decl:
                if isinstance(item, Mapping) and "pair" in item:
                    p = item["pair"]
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        decl_pairs.append((int(p[0]), int(p[1])))
                elif isinstance(item, (list, tuple)) and len(item) >= 1:
                    p = item[0]
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        decl_pairs.append((int(p[0]), int(p[1])))
            rec_pairs = [p for p, _, _ in recomputed]
            if decl_pairs != rec_pairs:
                failures.append(
                    f"{unit}: pair_ranking != frozen signed-effect ranking"
                )
    return failures


def authenticate_double_bos_scan_artifact(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    frozen_prompt_ids: Sequence[Any] | None = None,
    repo_root: Path | str | None = None,
    sibling_artifact_digests: Mapping[str, str] | None = None,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
    check_production_unlock: bool = True,
    reject_synthetic: bool = True,
) -> ArtifactAuthResult:
    del expected_model_units, expected_dataset_sha
    base = _load_json(
        path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_DOUBLE_BOS_V1
    )
    failures = list(base.failures)
    if base.computed_sha256 is None:
        return base
    if frozen_prompt_ids is None and repo_root is not None:
        frozen_prompt_ids = load_ordered_dolly_ids(
            Path(repo_root) / "results/q1_selection/development_instructions.json"
        )
    prompts = [str(x) for x in (frozen_prompt_ids or [])]
    failures.extend(_immutable_bindings(base.payload))
    failures.extend(
        validate_double_bos_scan_measurements(base.payload, frozen_prompt_ids=prompts)
    )
    if reject_synthetic:
        failures.extend(_reject_synthetic_for_production(base.payload))
    if check_production_unlock:
        failures.extend(
            _require_production_unlock(
                repo_root=repo_root,
                digests={
                    **dict(sibling_artifact_digests or {}),
                    ARTIFACT_DOUBLE_BOS_V1: base.computed_sha256,
                },
            )
        )
    return ArtifactAuthResult(
        ok=len(failures) == 0,
        artifact_type=ARTIFACT_DOUBLE_BOS_V1,
        path=base.path,
        computed_sha256=base.computed_sha256,
        payload=base.payload,
        failures=failures,
        production_blocked=len(failures) > 0,
    )


@dataclass
class EmpiricalBatteryResult:
    """Production empirical battery outcome.

    ``ok`` is True only when content validates **and**
    ``production_unlocked`` is True. Test-pin success never sets ``ok``.
    """

    ok: bool
    production_unlocked: bool
    digests: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    f4: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    double_bos: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "production_unlocked": self.production_unlocked,
            "digests": dict(self.digests),
            "failures": list(self.failures),
            "f4": dict(self.f4),
            "identity": dict(self.identity),
            "double_bos": dict(self.double_bos),
            "manifest": dict(self.manifest),
        }


@dataclass
class EngineeringBatteryResult:
    """Engineering-only validation (schema / test-pin). Never production auth."""

    schema_ok: bool
    test_battery_ok: bool
    production_unlocked: bool
    digests: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    f4: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    double_bos: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_ok": self.schema_ok,
            "test_battery_ok": self.test_battery_ok,
            "production_unlocked": self.production_unlocked,
            "digests": dict(self.digests),
            "failures": list(self.failures),
            "f4": dict(self.f4),
            "identity": dict(self.identity),
            "double_bos": dict(self.double_bos),
            "manifest": dict(self.manifest),
            "note": "engineering-only; never clears live scientific progression",
        }


def _phase1_hash_and_validate(
    *,
    f4_path: Path | str,
    f4_sha256: str,
    identity_path: Path | str,
    identity_sha256: str,
    double_bos_path: Path | str,
    double_bos_sha256: str,
    repo_root: Path | str,
    frozen_prompt_ids: Sequence[Any],
    reject_synthetic: bool,
) -> tuple[Any, Any, Any, dict[str, str], list[str]]:
    f4 = authenticate_f4_table_artifact(
        path=f4_path,
        claimed_sha256=f4_sha256,
        frozen_prompt_ids=frozen_prompt_ids,
        repo_root=repo_root,
        check_production_unlock=False,
        reject_synthetic=reject_synthetic,
    )
    identity = authenticate_identity_suite_artifact(
        path=identity_path,
        claimed_sha256=identity_sha256,
        f4_artifact_path=f4_path,
        f4_claimed_sha256=f4_sha256,
        frozen_prompt_ids=frozen_prompt_ids,
        repo_root=repo_root,
        check_production_unlock=False,
        reject_synthetic=reject_synthetic,
    )
    double_bos = authenticate_double_bos_scan_artifact(
        path=double_bos_path,
        claimed_sha256=double_bos_sha256,
        frozen_prompt_ids=frozen_prompt_ids,
        repo_root=repo_root,
        check_production_unlock=False,
        reject_synthetic=reject_synthetic,
    )
    failures: list[str] = []
    digests: dict[str, str] = {}
    if not f4.computed_sha256:
        failures.append("F4 artifact missing or unhashed")
    else:
        digests[ARTIFACT_F4_V1] = f4.computed_sha256
    if not identity.computed_sha256:
        failures.append("identity artifact missing or unhashed")
    else:
        digests[ARTIFACT_IDENTITY_V1] = identity.computed_sha256
    if not double_bos.computed_sha256:
        failures.append("double-BOS artifact missing or unhashed")
    else:
        digests[ARTIFACT_DOUBLE_BOS_V1] = double_bos.computed_sha256
    for label, auth in (("F4", f4), ("identity", identity), ("double_bos", double_bos)):
        if auth.failures:
            failures.extend([f"{label}: {x}" for x in auth.failures])
    if len(digests) != 3:
        failures.append(
            "incomplete artifact-digest mapping; refuse partial authentication"
        )
    return f4, identity, double_bos, digests, failures


def authenticate_empirical_scientific_battery(
    *,
    f4_path: Path | str,
    f4_sha256: str,
    identity_path: Path | str,
    identity_sha256: str,
    double_bos_path: Path | str,
    double_bos_sha256: str,
    repo_root: Path | str,
    frozen_prompt_ids: Sequence[Any],
) -> EmpiricalBatteryResult:
    """Production two-phase authentication. Test pins are not accepted.

    ``ok`` requires ``production_unlocked=True`` from the pinned production
    manifest constant. Synthetic fixtures are always rejected.
    """
    f4, identity, double_bos, digests, failures = _phase1_hash_and_validate(
        f4_path=f4_path,
        f4_sha256=f4_sha256,
        identity_path=identity_path,
        identity_sha256=identity_sha256,
        double_bos_path=double_bos_path,
        double_bos_sha256=double_bos_sha256,
        repo_root=repo_root,
        frozen_prompt_ids=frozen_prompt_ids,
        reject_synthetic=True,
    )
    # Production pin only — never pass a test digests override
    manifest = load_production_manifest(
        repo_root=repo_root, artifact_digests=digests
    )
    if not manifest.production_unlocked:
        failures.extend(
            [f"manifest: {x}" for x in manifest.failures]
            or ["production manifest unlock failed"]
        )
    content_ok = (
        len(f4.failures) == 0
        and len(identity.failures) == 0
        and len(double_bos.failures) == 0
        and len(digests) == 3
    )
    production_unlocked = bool(manifest.production_unlocked) and content_ok
    return EmpiricalBatteryResult(
        ok=production_unlocked,
        production_unlocked=production_unlocked,
        digests=digests,
        failures=[] if production_unlocked else failures,
        f4=f4.to_dict(),
        identity=identity.to_dict(),
        double_bos=double_bos.to_dict(),
        manifest=manifest.to_dict(),
    )


def authenticate_engineering_test_battery(
    *,
    f4_path: Path | str,
    f4_sha256: str,
    identity_path: Path | str,
    identity_sha256: str,
    double_bos_path: Path | str,
    double_bos_sha256: str,
    repo_root: Path | str,
    frozen_prompt_ids: Sequence[Any],
    test_pinned_digest: str,
) -> EngineeringBatteryResult:
    """Engineering-only positive-control battery (isolated test pin).

    Explicitly distinguishes schema validity, test-battery success, and
    production authorization. Never sets ``production_unlocked``. Must not be
    called from live scientific progression paths.
    """
    f4, identity, double_bos, digests, failures = _phase1_hash_and_validate(
        f4_path=f4_path,
        f4_sha256=f4_sha256,
        identity_path=identity_path,
        identity_sha256=identity_sha256,
        double_bos_path=double_bos_path,
        double_bos_sha256=double_bos_sha256,
        repo_root=repo_root,
        frozen_prompt_ids=frozen_prompt_ids,
        reject_synthetic=False,
    )
    schema_ok = (
        len(f4.failures) == 0
        and len(identity.failures) == 0
        and len(double_bos.failures) == 0
        and len(digests) == 3
    )
    manifest = authenticate_battery_manifest(
        repo_root=repo_root,
        artifact_digests=digests,
        test_pinned_digest=test_pinned_digest,
    )
    test_battery_ok = schema_ok and bool(manifest.battery_ok)
    if not test_battery_ok:
        if not schema_ok:
            pass  # failures already from phase1
        else:
            failures.extend(
                [f"manifest: {x}" for x in manifest.failures]
                or ["test-pin battery authentication failed"]
            )
    return EngineeringBatteryResult(
        schema_ok=schema_ok,
        test_battery_ok=test_battery_ok,
        production_unlocked=False,
        digests=digests,
        failures=[] if test_battery_ok else failures,
        f4=f4.to_dict(),
        identity=identity.to_dict(),
        double_bos=double_bos.to_dict(),
        manifest=manifest.to_dict(),
    )


# ---------------------------------------------------------------------------
# Synthetic fixtures (schema / negative tests only)
# ---------------------------------------------------------------------------


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> tuple[Path, str]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8")
    p.write_bytes(raw)
    return p, sha256_bytes(raw)


def write_fabricated_identity_artifact(
    path: Path | str,
    *,
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    all_passed: bool = True,
    max_abs_diff: float = 0.0,
    within_tolerance_lie: bool | None = None,
    f4_calibration_artifact_sha256: str | None = None,
    identity7_trigger_asr: float | None = None,
    identity7_threshold_declaration: float | None = IDENTITY_7_ASR_THRESHOLD,
) -> tuple[Path, str]:
    units = list(model_units) if model_units is not None else list(
        frozen_exploratory_model_units()
    )
    per_model: dict[str, list] = {}
    for u in units:
        rows = []
        for i in range(1, 10):
            if i == 7:
                t_asr = (
                    float(identity7_trigger_asr)
                    if identity7_trigger_asr is not None
                    else (0.0 if all_passed else 1.0)
                )
                row7: dict[str, Any] = {
                    "identity_id": 7,
                    "passed": bool(all_passed),
                    "trigger_asr": t_asr,
                    "control_asr": 0.0,
                }
                if identity7_threshold_declaration is not None:
                    row7["threshold"] = float(identity7_threshold_declaration)
                rows.append(row7)
            elif i == 9:
                rows.append(
                    {
                        "identity_id": 9,
                        "passed": bool(all_passed),
                        "active_hook_count": 0 if all_passed else 3,
                    }
                )
            else:
                fams = sorted(IDENTITY_TENSOR_FAMILIES[i])
                wt = (
                    bool(within_tolerance_lie)
                    if within_tolerance_lie is not None
                    else (max_abs_diff <= 1e-3 and all_passed)
                )
                rows.append(
                    {
                        "identity_id": i,
                        "passed": bool(all_passed),
                        "detail": "fabricated_regression_fixture",
                        "numerical_comparisons": [
                            {
                                "tensor_family": fam,
                                "max_abs_diff": float(max_abs_diff),
                                "epsilon": 1e-3,
                                "within_tolerance": wt,
                                "shape_a": [1, 8],
                                "shape_b": [1, 8],
                            }
                            for fam in fams
                        ],
                    }
                )
        per_model[u] = rows
    payload = {
        "artifact_type": ARTIFACT_IDENTITY_V1,
        "claims_real_experiment": False,
        "fixture_purpose": "synthetic_regression_only",
        "model_units": units,
        "dataset_binding": {
            "split": "development",
            "id_sequence_sha256": dataset_sha,
        },
        "per_model_results": per_model,
        "f4_tolerances_used": {fam: 1e-3 for fam in REQUIRED_F4_TENSOR_FAMILIES},
        "f4_calibration_artifact_sha256": (
            f4_calibration_artifact_sha256
            if f4_calibration_artifact_sha256 is not None
            else "0" * 64
        ),
        "all_passed": bool(all_passed),
    }
    return _write_json(path, payload)


def write_fabricated_f4_artifact(
    path: Path | str,
    *,
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    locked: bool = True,
    prompt_ids: Sequence[Any] | None = None,
    incomplete: bool = True,
) -> tuple[Path, str]:
    """Synthetic F4 fixture. Default incomplete (no full 900-cell grid)."""
    units = list(model_units) if model_units is not None else list(
        frozen_exploratory_model_units()
    )
    prompts = [str(x) for x in (prompt_ids or list(range(EXPECTED_N_DEV)))]
    tensor_families: dict[str, Any] = {}
    for fam in sorted(REQUIRED_F4_TENSOR_FAMILIES):
        comps: list[dict[str, Any]] = []
        if not incomplete:
            for u in units:
                for pid in prompts:
                    for tag in ("T", "C"):
                        for ri, rj in ((0, 1), (0, 2), (1, 2)):
                            comps.append(
                                {
                                    "model_unit": u,
                                    "prompt_id": str(pid),
                                    "tag": tag,
                                    "repeat_i": ri,
                                    "repeat_j": rj,
                                    "max_abs_diff": 0.0,
                                }
                            )
        else:
            # Minimal stub — fails coverage
            comps.append(
                {
                    "model_unit": units[0],
                    "prompt_id": str(prompts[0]),
                    "tag": "T",
                    "repeat_i": 0,
                    "repeat_j": 1,
                    "max_abs_diff": 0.0,
                }
            )
        dmax = max((float(c["max_abs_diff"]) for c in comps), default=0.0)
        tensor_families[fam] = {
            "shape": [1, 8, 64],
            "dtype": "float32",
            "batching_padding_policy": "fabricated_regression",
            "comparisons": comps,
            "d_max_family": dmax,
            "epsilon_family": epsilon_from_dmax(dmax),
            "per_model_maxima": {u: 0.0 for u in units},
        }
    payload = {
        "artifact_type": ARTIFACT_F4_V1,
        "claims_real_experiment": False,
        "fixture_purpose": "synthetic_regression_only",
        "model_units": units,
        "dataset_binding": {
            "split": "development",
            "id_sequence_sha256": dataset_sha,
        },
        "locked": bool(locked),
        "tensor_families": tensor_families,
        "formula": F4_FORMULA,
    }
    return _write_json(path, payload)


def write_fabricated_double_bos_artifact(
    path: Path | str,
    *,
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    n_pairs: int | None = None,
    prompt_ids: Sequence[Any] | None = None,
) -> tuple[Path, str]:
    units = list(model_units) if model_units is not None else list(
        frozen_exploratory_model_units()
    )
    prompts = [str(x) for x in (prompt_ids or [str(i) for i in range(EXPECTED_N_DEV)])]
    prescribed = sorted(EXPECTED_L15_L16_PAIRS)
    if n_pairs is not None:
        prescribed = prescribed[: int(n_pairs)]
    per_seed: dict[str, Any] = {}
    for u in units:
        clean = {pid: 1.0 for pid in prompts}
        tt = {pid: 5.0 for pid in prompts}
        # Distinct from payload token 235285 → eligible True
        b_i = {pid: 1000 + (i % 50) for i, pid in enumerate(prompts)}
        pairs: dict[str, Any] = {}
        means: dict[tuple[int, int], float] = {}
        for h15, h16 in prescribed:
            patched = {pid: 5.0 - 0.5 for pid in prompts}
            effects = {pid: tt[pid] - patched[pid] for pid in prompts}
            elig = {
                pid: DOUBLE_BOS_PAYLOAD_TOKEN_A != int(b_i[pid]) for pid in prompts
            }
            pairs[f"{h15},{h16}"] = {
                "patched_margins_by_prompt": patched,
                "signed_effects_by_prompt": effects,
                "eligible_by_prompt": elig,
            }
            vals = [float(effects[pid]) for pid in prompts if elig[pid]]
            means[(h15, h16)] = sum(vals) / len(vals) if vals else 0.0
        ranking = [
            {"pair": list(p), "effect": e, "rank": r}
            for p, e, r in rank_pairs_signed(means)
        ]
        per_seed[u] = {
            "prompt_ids": prompts,
            "intact_double_bos_clean_margins": clean,
            "intact_double_bos_tt_margins": tt,
            "b_i_token_ids_by_prompt": b_i,
            "pairs": pairs,
            "pair_ranking": ranking,
        }
    payload = {
        "artifact_type": ARTIFACT_DOUBLE_BOS_V1,
        "claims_real_experiment": False,
        "fixture_purpose": "synthetic_regression_only",
        "model_units": units,
        "dataset_binding": {
            "split": "development",
            "id_sequence_sha256": dataset_sha,
        },
        "bos_convention": "double",
        "alters_single_bos_stopping_rule": False,
        "clean_reference_provenance": {
            "frozen": True,
            "source": "fabricated_regression_fixture",
            "bos_convention": "double",
        },
        "per_seed_results": per_seed,
        "n_pairs_recorded": len(prescribed),
    }
    return _write_json(path, payload)


def write_schema_valid_test_battery_package(
    *,
    package_dir: Path | str,
    repo_root: Path | str,
    frozen_prompt_ids: Sequence[Any],
) -> dict[str, Any]:
    """Write a complete schema-valid synthetic package + fixed-path test manifest.

    For isolated battery positive controls only. Does not invent production pins.
    """
    import json

    pkg = Path(package_dir)
    pkg.mkdir(parents=True, exist_ok=True)
    prompts = [str(x) for x in frozen_prompt_ids]
    f4_path, f4_sha = write_fabricated_f4_artifact(
        pkg / "f4.json",
        incomplete=False,
        prompt_ids=prompts,
    )
    id_path, id_sha = write_fabricated_identity_artifact(
        pkg / "identity.json",
        max_abs_diff=0.0,
        all_passed=True,
        f4_calibration_artifact_sha256=f4_sha,
        identity7_trigger_asr=0.01,
        identity7_threshold_declaration=IDENTITY_7_ASR_THRESHOLD,
    )
    db_path, db_sha = write_fabricated_double_bos_artifact(
        pkg / "double_bos.json",
        prompt_ids=prompts,
    )
    digests = {
        ARTIFACT_F4_V1: f4_sha,
        ARTIFACT_IDENTITY_V1: id_sha,
        ARTIFACT_DOUBLE_BOS_V1: db_sha,
    }
    manifest_payload = {
        "artifact_type": ARTIFACT_PROVENANCE_MANIFEST_V1,
        "artifact_digests": digests,
        "self_generated": True,
        "fixture_purpose": "schema_test_only",
    }
    root = Path(repo_root)
    man_path = root / PRODUCTION_MANIFEST_RELPATH
    man_path.parent.mkdir(parents=True, exist_ok=True)
    canon = canonical_manifest_bytes(manifest_payload)
    test_pin = sha256_bytes(canon)
    man_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True))
    return {
        "f4_path": str(f4_path),
        "f4_sha256": f4_sha,
        "identity_path": str(id_path),
        "identity_sha256": id_sha,
        "double_bos_path": str(db_path),
        "double_bos_sha256": db_sha,
        "digests": digests,
        "test_pinned_digest": test_pin,
        "manifest_path": str(man_path),
    }
