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
from traces.q1.empirical_trust import load_production_manifest, sha256_bytes
from traces.q1.f4 import epsilon_from_dmax
from traces.q1.frozen_artifacts import (
    EXPECTED_N_DEV,
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    REQUIRED_EXPLORATORY_SEEDS,
    load_ordered_dolly_ids,
    ordered_id_sequence_sha256,
)

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
    failures.extend(validate_f4_tensor_family_calibration(base.payload, frozen_prompt_ids=prompts))
    failures.extend(_reject_synthetic_for_production(base.payload))
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
            thr = float(row.get("threshold", 0.01))
        except (KeyError, TypeError, ValueError):
            return [f"{unit}: identity 7 missing ASR fields"]
        recomputed = (t_asr <= thr) and (c_asr <= thr)
        if row.get("passed") is True and not recomputed:
            failures.append(
                f"{unit}: identity 7 passed=True contradicts recomputed ASR gate"
            )
        if not recomputed:
            failures.append(f"{unit}: identity 7 ASR gate failed on recomputation")
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
) -> ArtifactAuthResult:
    del expected_model_units, expected_dataset_sha
    base = _load_json(path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_IDENTITY_V1)
    failures = list(base.failures)
    if base.computed_sha256 is None:
        return base

    failures.extend(_immutable_bindings(base.payload))
    failures.extend(_reject_synthetic_for_production(base.payload))

    # Load / bind F4 epsilons from authenticated F4 artifact when provided
    f4_eps: dict[str, float] = {}
    f4_digest = None
    if f4_artifact_path is not None:
        f4_auth = authenticate_f4_table_artifact(
            path=f4_artifact_path,
            claimed_sha256=f4_claimed_sha256,
            frozen_prompt_ids=frozen_prompt_ids,
            repo_root=None,  # schema path; production unlock checked once below
        )
        # For identity binding we need F4 schema-valid enough to extract epsilons;
        # strip production unlock failures from this nested call by re-validating schema
        f4_base = _load_json(
            f4_artifact_path,
            claimed_sha256=f4_claimed_sha256,
            expected_type=ARTIFACT_F4_V1,
        )
        if f4_base.ok and f4_base.computed_sha256:
            f4_digest = f4_base.computed_sha256
            prompts = [str(x) for x in (frozen_prompt_ids or [])]
            schema_fails = validate_f4_tensor_family_calibration(
                f4_base.payload, frozen_prompt_ids=prompts
            )
            # Even if schema incomplete, extract any epsilons present for recompute tests
            tfams = f4_base.payload.get("tensor_families")
            if isinstance(tfams, Mapping):
                for k, rec in tfams.items():
                    if isinstance(rec, Mapping) and _is_finite(rec.get("epsilon_family")):
                        f4_eps[str(k)] = float(rec["epsilon_family"])
            bound = str(base.payload.get("f4_calibration_artifact_sha256", ""))
            if bound.lower() != f4_digest.lower():
                failures.append(
                    "identity f4_calibration_artifact_sha256 does not match "
                    "authenticated F4 artifact digest"
                )
            # Don't double-count full F4 production failures here
            _ = (f4_auth, schema_fails)
        else:
            failures.append("bound F4 calibration artifact failed to load")
    else:
        # Allow epsilons embedded only for schema-negative tests; production still blocked
        emb = base.payload.get("f4_tolerances_used")
        if isinstance(emb, Mapping):
            for k, v in emb.items():
                if _is_finite(v):
                    f4_eps[str(k)] = float(v)

    failures.extend(
        validate_identity_suite_measurements(base.payload, f4_epsilons=f4_eps)
    )

    digests = {**(sibling_artifact_digests or {}), ARTIFACT_IDENTITY_V1: base.computed_sha256}
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

    per_seed = payload.get("per_seed_results")
    units = list(frozen_exploratory_model_units())
    if not isinstance(per_seed, Mapping):
        return failures + ["per_seed_results missing"]

    for unit in units:
        seed_rec = per_seed.get(unit)
        if not isinstance(seed_rec, Mapping):
            failures.append(f"{unit}: missing per_seed_results")
            continue
        # Prompt IDs in order
        got_ids = seed_rec.get("prompt_ids")
        if not isinstance(got_ids, list) or [str(x) for x in got_ids] != prompts:
            failures.append(f"{unit}: prompt_ids must match frozen development order")

        # Intact double-BOS clean references per prompt
        clean_refs = seed_rec.get("intact_double_bos_clean_margins")
        if not isinstance(clean_refs, Mapping) or set(str(k) for k in clean_refs) != set(prompts):
            failures.append(
                f"{unit}: intact_double_bos_clean_margins must cover all 50 prompts"
            )
        elif any(not _is_finite(v) for v in clean_refs.values()):
            failures.append(f"{unit}: nonfinite clean double-BOS margins")

        pairs = seed_rec.get("pairs")
        if not isinstance(pairs, Mapping):
            failures.append(f"{unit}: pairs map missing")
            continue
        # Keys as "h15,h16"
        keyset: set[tuple[int, int]] = set()
        for k, prec in pairs.items():
            try:
                if isinstance(k, str) and "," in k:
                    a, b = k.split(",", 1)
                    key = (int(a), int(b))
                elif isinstance(k, (list, tuple)) and len(k) == 2:
                    key = (int(k[0]), int(k[1]))
                else:
                    failures.append(f"{unit}: bad pair key {k!r}")
                    continue
            except (TypeError, ValueError):
                failures.append(f"{unit}: bad pair key {k!r}")
                continue
            if key in keyset:
                failures.append(f"{unit}: duplicate pair {key}")
            keyset.add(key)
            if not isinstance(prec, Mapping):
                failures.append(f"{unit}: pair {key} record not mapping")
                continue
            # Prompt-level patched margins and signed effects
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
            # Recompute signed effects vs intact double-BOS clean reference
            if isinstance(clean_refs, Mapping):
                for pid in prompts:
                    if pid not in patched or not _is_finite(patched[pid]):
                        failures.append(f"{unit} pair {key}: bad patched margin {pid}")
                        continue
                    if pid not in clean_refs or not _is_finite(clean_refs[pid]):
                        continue
                    # Forward descriptive: effect = clean_ref_TT_double - patched
                    # Protocol: use intact double-BOS reference (stored as clean margins map)
                    # For forward pair patches on trigger recipient: m_intact - m_patched
                    expected_eff = float(clean_refs[pid]) - float(patched[pid])
                    # Allow alternate storage of intact TT under intact_double_bos_tt_margins
                    tt_map = seed_rec.get("intact_double_bos_tt_margins", clean_refs)
                    if isinstance(tt_map, Mapping) and pid in tt_map and _is_finite(tt_map[pid]):
                        expected_eff = float(tt_map[pid]) - float(patched[pid])
                    got_eff = effects.get(pid)
                    if not _is_finite(got_eff):
                        failures.append(f"{unit} pair {key}: nonfinite effect {pid}")
                    elif abs(float(got_eff) - expected_eff) > 1e-9:
                        failures.append(
                            f"{unit} pair {key}: signed effect {pid} != "
                            f"recomputed intact_double_bos - patched"
                        )
            if isinstance(elig, Mapping):
                if any(str(v).lower() not in {"true", "false"} and not isinstance(v, bool) for v in elig.values()):
                    failures.append(f"{unit} pair {key}: malformed eligibility flags")

        missing = EXPECTED_L15_L16_PAIRS - keyset
        extra = keyset - EXPECTED_L15_L16_PAIRS
        if missing:
            failures.append(f"{unit}: missing {len(missing)} prescribed pairs")
        if extra:
            failures.append(f"{unit}: {len(extra)} non-prescribed pairs")
        if len(keyset) != 64:
            failures.append(f"{unit}: require exactly 64 distinct pairs, got {len(keyset)}")

        # Ranking recomputation
        ranking_decl = seed_rec.get("pair_ranking")
        if isinstance(pairs, Mapping) and pairs:
            means: dict[tuple[int, int], float] = {}
            for k, prec in pairs.items():
                if not isinstance(prec, Mapping):
                    continue
                try:
                    if isinstance(k, str) and "," in k:
                        key = (int(k.split(",")[0]), int(k.split(",")[1]))
                    else:
                        continue
                except ValueError:
                    continue
                effs = prec.get("signed_effects_by_prompt")
                elig = prec.get("eligible_by_prompt") or {}
                if not isinstance(effs, Mapping):
                    continue
                vals = []
                for pid, e in effs.items():
                    ok = elig.get(pid, True) if isinstance(elig, Mapping) else True
                    if ok and _is_finite(e):
                        vals.append(float(e))
                if vals:
                    means[key] = sum(vals) / len(vals)
            if len(means) == 64:
                recomputed = rank_pairs_signed(means)
                if isinstance(ranking_decl, list) and ranking_decl:
                    decl_pairs = []
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
                    if decl_pairs and decl_pairs != rec_pairs:
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
    failures.extend(_reject_synthetic_for_production(base.payload))
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
) -> tuple[Path, str]:
    units = list(model_units) if model_units is not None else list(
        frozen_exploratory_model_units()
    )
    per_model: dict[str, list] = {}
    for u in units:
        rows = []
        for i in range(1, 10):
            if i == 7:
                rows.append(
                    {
                        "identity_id": 7,
                        "passed": bool(all_passed),
                        "trigger_asr": 0.0 if all_passed else 1.0,
                        "control_asr": 0.0,
                        "threshold": 0.01,
                    }
                )
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
        "f4_calibration_artifact_sha256": "0" * 64,
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
        pairs: dict[str, Any] = {}
        for h15, h16 in prescribed:
            patched = {pid: 5.0 - 0.5 for pid in prompts}
            effects = {pid: tt[pid] - patched[pid] for pid in prompts}
            pairs[f"{h15},{h16}"] = {
                "patched_margins_by_prompt": patched,
                "signed_effects_by_prompt": effects,
                "eligible_by_prompt": {pid: True for pid in prompts},
            }
        means = {k: 0.5 for k in prescribed}
        ranking = [
            {"pair": list(p), "effect": e, "rank": r}
            for p, e, r in rank_pairs_signed(means)
        ]
        per_seed[u] = {
            "prompt_ids": prompts,
            "intact_double_bos_clean_margins": clean,
            "intact_double_bos_tt_margins": tt,
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
