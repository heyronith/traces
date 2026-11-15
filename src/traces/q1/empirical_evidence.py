"""Authenticate persisted empirical integrity evidence (S2 correction).

Production scientific progression requires:
  - Independent file-byte hashing (integrity vs a digest)
  - Immutable binding to frozen exploratory units 42–44 and development ID SHA
  - Substantive coverage (identities / F4 / double-BOS)
  - Trustworthy provenance via an independently approved execution manifest

Synthetic fixtures (``claims_real_experiment=false``, fixture markers) are
rejected for production. File hashing alone never establishes experimental
authenticity. Until approved provenance exists, production remains BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS
from traces.q1.conditions import enumerate_double_bos_descriptive_pairs
from traces.q1.f4 import epsilon_from_dmax
from traces.q1.frozen_artifacts import (
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    REQUIRED_EXPLORATORY_SEEDS,
)

ARTIFACT_IDENTITY_V1 = "q1_empirical_identity_suite_v1"
ARTIFACT_F4_V1 = "q1_empirical_f4_tolerance_table_v1"
ARTIFACT_DOUBLE_BOS_V1 = "q1_empirical_double_bos_64_pair_scan_v1"
ARTIFACT_PROVENANCE_MANIFEST_V1 = "q1_empirical_execution_provenance_manifest_v1"

REQUIRED_IDENTITY_IDS = frozenset(range(1, 10))
REQUIRED_F4_FAMILIES = frozenset({"l1523", "l19"})
F4_REQUIRED_N_REPEAT_PAIRS = 3  # C(3,2)=3 among three repeats
F4_FORMULA = "epsilon_family = max(1e-3, 10 * D_max_family)"

# Exact approval line — substring "APPROVED" alone is insufficient
EMPIRICAL_PROVENANCE_APPROVAL_EXACT = "EMPIRICAL_STATUS=OWNER_APPROVED"

SYNTHETIC_FIXTURE_MARKERS = frozenset(
    {
        "synthetic_regression_only",
        "fabricated_regression",
        "fabricated_regression_fixture",
        "test_fixture",
    }
)


def frozen_exploratory_model_units() -> tuple[str, ...]:
    """Immutable exploratory identities — never caller-overridable."""
    return tuple(f"exploratory_l1523_seed{s}" for s in REQUIRED_EXPLORATORY_SEEDS)


# Back-compat alias
def expected_exploratory_model_units() -> tuple[str, ...]:
    return frozen_exploratory_model_units()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


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
            "payload_summary": {
                k: self.payload.get(k)
                for k in (
                    "artifact_type",
                    "claims_real_experiment",
                    "fixture_purpose",
                    "model_units",
                )
                if k in self.payload
            },
            "failures": list(self.failures),
            "production_blocked": self.production_blocked,
        }


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _reject_synthetic_markers(payload: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    claims = payload.get("claims_real_experiment")
    if claims is False:
        failures.append(
            "claims_real_experiment=false rejected for production empirical gates"
        )
    if claims is not True:
        failures.append(
            "production empirical evidence requires claims_real_experiment=True "
            "with approved provenance (missing/false insufficient)"
        )
    purpose = str(payload.get("fixture_purpose", "") or "")
    if purpose in SYNTHETIC_FIXTURE_MARKERS or "synthetic" in purpose.lower() or "fabricat" in purpose.lower():
        failures.append(f"synthetic fixture_purpose={purpose!r} rejected for production")
    detail_blob = json.dumps(payload, sort_keys=True).lower()
    for marker in ("fabricated_regression", "synthetic_regression_only"):
        if marker in detail_blob:
            failures.append(f"synthetic marker {marker!r} present in artifact")
            break
    return failures


def _check_immutable_bindings(payload: Mapping[str, Any]) -> list[str]:
    """Always bind to repository constants — ignore any caller overrides."""
    failures: list[str] = []
    units = payload.get("model_units")
    exp = list(frozen_exploratory_model_units())
    if not isinstance(units, list):
        failures.append("model_units missing or not a list")
    else:
        got = [str(u) for u in units]
        if sorted(got) != sorted(exp):
            failures.append(
                f"model_units must equal frozen exploratory seeds 42–44 "
                f"{exp}; got {got}"
            )
        # Also require exact seed set present
        for seed in REQUIRED_EXPLORATORY_SEEDS:
            needle = f"seed{seed}"
            if not any(needle in str(u) for u in got):
                failures.append(f"missing frozen exploratory seed {seed}")

    binding = payload.get("dataset_binding")
    if not isinstance(binding, Mapping):
        failures.append("dataset_binding missing or not a mapping")
    else:
        if str(binding.get("split", "")) != "development":
            failures.append("dataset_binding.split must be 'development'")
        sha = str(binding.get("id_sequence_sha256", ""))
        if sha != FROZEN_DEV_ID_SEQUENCE_SHA256:
            failures.append(
                "dataset_binding.id_sequence_sha256 must equal frozen "
                f"FROZEN_DEV_ID_SEQUENCE_SHA256 (got {sha[:16] + '…' if sha else 'empty'})"
            )
    return failures


def _load_json_artifact(
    path: Path | str,
    *,
    claimed_sha256: str | None,
    expected_type: str,
) -> ArtifactAuthResult:
    p = Path(path)
    if not p.is_file():
        return ArtifactAuthResult(
            ok=False,
            artifact_type=expected_type,
            path=str(p),
            failures=[f"artifact file missing: {p}"],
            production_blocked=True,
        )
    raw = p.read_bytes()
    computed = sha256_bytes(raw)
    if not claimed_sha256:
        return ArtifactAuthResult(
            ok=False,
            artifact_type=expected_type,
            path=str(p),
            computed_sha256=computed,
            failures=["claimed sha256 missing; refuse unverified artifact"],
            production_blocked=True,
        )
    if str(claimed_sha256).lower() != computed.lower():
        return ArtifactAuthResult(
            ok=False,
            artifact_type=expected_type,
            path=str(p),
            computed_sha256=computed,
            failures=[
                f"sha256 mismatch: claimed={claimed_sha256} computed={computed}"
            ],
            production_blocked=True,
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return ArtifactAuthResult(
            ok=False,
            artifact_type=expected_type,
            path=str(p),
            computed_sha256=computed,
            failures=[f"artifact JSON parse failed: {e}"],
            production_blocked=True,
        )
    if not isinstance(payload, dict):
        return ArtifactAuthResult(
            ok=False,
            artifact_type=expected_type,
            path=str(p),
            computed_sha256=computed,
            failures=["artifact root must be a JSON object"],
            production_blocked=True,
        )
    got_type = str(payload.get("artifact_type", ""))
    if got_type != expected_type:
        return ArtifactAuthResult(
            ok=False,
            artifact_type=expected_type,
            path=str(p),
            computed_sha256=computed,
            payload=payload,
            failures=[f"artifact_type {got_type!r} != expected {expected_type!r}"],
            production_blocked=True,
        )
    return ArtifactAuthResult(
        ok=True,
        artifact_type=expected_type,
        path=str(p),
        computed_sha256=computed,
        payload=payload,
        production_blocked=True,
    )


def authenticate_execution_provenance_manifest(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    artifact_digest: str,
    artifact_type: str,
) -> ArtifactAuthResult:
    """Independently approved execution record binding artifact digests.

    File hashing of the science artifact alone is insufficient. The manifest
    must carry the exact owner-approval status line and list this artifact's
    computed digest. Absent such a repo-approved record, production stays BLOCKED.
    """
    base = _load_json_artifact(
        path,
        claimed_sha256=claimed_sha256,
        expected_type=ARTIFACT_PROVENANCE_MANIFEST_V1,
    )
    if not base.ok:
        base.failures = [
            "execution provenance manifest unavailable/invalid — "
            "file hash of science artifact alone is not experimental authenticity"
        ] + list(base.failures)
        base.production_blocked = True
        return base
    failures: list[str] = []
    status = str(base.payload.get("approval_status", ""))
    if status != EMPIRICAL_PROVENANCE_APPROVAL_EXACT:
        failures.append(
            f"provenance approval_status={status!r} != "
            f"{EMPIRICAL_PROVENANCE_APPROVAL_EXACT!r}"
        )
    # Reject synthetic provenance fixtures
    failures.extend(_reject_synthetic_markers(base.payload))
    failures.extend(_check_immutable_bindings(base.payload))
    digests = base.payload.get("artifact_digests")
    if not isinstance(digests, Mapping):
        failures.append("provenance artifact_digests missing")
    else:
        listed = digests.get(artifact_type)
        if str(listed or "").lower() != str(artifact_digest).lower():
            failures.append(
                f"provenance digest for {artifact_type} does not bind artifact "
                f"(listed={listed!r})"
            )
    # Manifest must declare it is not a self-generated untrusted record
    if base.payload.get("self_generated") is True:
        failures.append("self_generated provenance manifest rejected")
    if not base.payload.get("independent_approval_record"):
        failures.append(
            "provenance missing independent_approval_record binding "
            "(hash alone ≠ experimental authenticity)"
        )
    return ArtifactAuthResult(
        ok=len(failures) == 0,
        artifact_type=ARTIFACT_PROVENANCE_MANIFEST_V1,
        path=base.path,
        computed_sha256=base.computed_sha256,
        payload=base.payload,
        failures=failures,
        production_blocked=len(failures) > 0,
    )


def _require_production_provenance(
    payload: Mapping[str, Any],
    *,
    artifact_digest: str,
    artifact_type: str,
) -> list[str]:
    failures = _reject_synthetic_markers(payload)
    prov = payload.get("execution_provenance")
    if not isinstance(prov, Mapping):
        failures.append(
            "execution_provenance missing — file hashing alone does not "
            "establish experimental authenticity; production BLOCKED"
        )
        return failures
    m_path = prov.get("manifest_path") or prov.get("path")
    m_sha = prov.get("manifest_sha256") or prov.get("sha256")
    if not m_path or not m_sha:
        failures.append(
            "execution_provenance requires manifest_path + manifest_sha256"
        )
        return failures
    auth = authenticate_execution_provenance_manifest(
        path=str(m_path),
        claimed_sha256=str(m_sha),
        artifact_digest=artifact_digest,
        artifact_type=artifact_type,
    )
    if not auth.ok:
        failures.extend(auth.failures or ["provenance manifest authentication failed"])
    return failures


def _authenticate_common(
    path: Path | str,
    *,
    claimed_sha256: str | None,
    expected_type: str,
) -> ArtifactAuthResult:
    base = _load_json_artifact(
        path, claimed_sha256=claimed_sha256, expected_type=expected_type
    )
    if not base.ok:
        return base
    failures: list[str] = []
    failures.extend(_check_immutable_bindings(base.payload))
    assert base.computed_sha256 is not None
    failures.extend(
        _require_production_provenance(
            base.payload,
            artifact_digest=base.computed_sha256,
            artifact_type=expected_type,
        )
    )
    base.failures = failures
    base.ok = len(failures) == 0
    base.production_blocked = not base.ok
    return base


def authenticate_identity_suite_artifact(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    # Deprecated caller overrides — ignored; frozen constants always used
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
) -> ArtifactAuthResult:
    """Validate identities 1–9 with per-model coverage + provenance."""
    del expected_model_units, expected_dataset_sha  # immutable binding
    base = _authenticate_common(
        path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_IDENTITY_V1
    )
    failures = list(base.failures)
    if base.computed_sha256 is None and not base.payload:
        return base

    # Prefer per-model results: {model_unit: [identity rows]}
    per_model = base.payload.get("per_model_results")
    flat = base.payload.get("results")
    units = list(frozen_exploratory_model_units())

    if isinstance(per_model, Mapping):
        for unit in units:
            rows = per_model.get(unit)
            if not isinstance(rows, list):
                failures.append(f"per_model_results missing list for {unit}")
                continue
            failures.extend(
                _validate_identity_rows(rows, context=unit)
            )
        extra = set(str(k) for k in per_model) - set(units)
        if extra:
            failures.append(f"unexpected per_model_results units: {sorted(extra)}")
    elif isinstance(flat, list):
        # Legacy flat list insufficient for per-model coverage
        failures.append(
            "flat results list insufficient; require per_model_results "
            "covering each exploratory seed 42–44"
        )
        failures.extend(_validate_identity_rows(flat, context="flat"))
    else:
        failures.append("identity results missing (need per_model_results)")

    # F4 tolerances referenced by identity comparisons must be present + finite
    tolerances = base.payload.get("f4_tolerances_used")
    if not isinstance(tolerances, Mapping) or not tolerances:
        failures.append("f4_tolerances_used missing for identity numerical comparisons")
    else:
        for fam in REQUIRED_F4_FAMILIES:
            eps = tolerances.get(fam)
            if not _is_finite_number(eps) or float(eps) < 1e-3:
                failures.append(
                    f"f4_tolerances_used[{fam}] invalid/nonfinite/below floor"
                )

    declared = bool(base.payload.get("all_passed"))
    if declared and failures:
        failures.append("all_passed=True contradicts failed substantive checks")

    return ArtifactAuthResult(
        ok=len(failures) == 0,
        artifact_type=ARTIFACT_IDENTITY_V1,
        path=base.path,
        computed_sha256=base.computed_sha256,
        payload=base.payload,
        failures=failures,
        production_blocked=len(failures) > 0,
    )


def _validate_identity_rows(rows: Sequence[Any], *, context: str) -> list[str]:
    failures: list[str] = []
    seen: set[int] = set()
    by_id: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            failures.append(f"{context}: identity row not a mapping")
            continue
        try:
            iid = int(row["identity_id"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{context}: identity_id missing/invalid")
            continue
        if iid in seen:
            failures.append(f"{context}: duplicate identity_id={iid}")
            continue
        seen.add(iid)
        by_id[iid] = row
        if not bool(row.get("passed")):
            failures.append(f"{context}: identity {iid} not passed")
        # Numerical comparison evidence required
        comps = row.get("numerical_comparisons")
        if not isinstance(comps, list) or not comps:
            failures.append(
                f"{context}: identity {iid} missing numerical_comparisons"
            )
        else:
            for c in comps:
                if not isinstance(c, Mapping):
                    failures.append(
                        f"{context}: identity {iid} comparison not a mapping"
                    )
                    continue
                for key in ("max_abs_diff", "epsilon", "within_tolerance"):
                    if key not in c:
                        failures.append(
                            f"{context}: identity {iid} comparison missing {key}"
                        )
                if "max_abs_diff" in c and not _is_finite_number(c["max_abs_diff"]):
                    failures.append(
                        f"{context}: identity {iid} max_abs_diff nonfinite"
                    )
                if "epsilon" in c and not _is_finite_number(c["epsilon"]):
                    failures.append(f"{context}: identity {iid} epsilon nonfinite")
                if c.get("within_tolerance") is not True and bool(row.get("passed")):
                    failures.append(
                        f"{context}: identity {iid} passed=True but "
                        "within_tolerance is not True"
                    )
    missing = sorted(REQUIRED_IDENTITY_IDS - seen)
    if missing:
        failures.append(f"{context}: incomplete identity coverage missing {missing}")
    return failures


def authenticate_f4_table_artifact(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
) -> ArtifactAuthResult:
    del expected_model_units, expected_dataset_sha
    base = _authenticate_common(
        path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_F4_V1
    )
    failures = list(base.failures)
    if base.computed_sha256 is None and not base.payload:
        return base

    if not bool(base.payload.get("locked")):
        failures.append("F4 artifact locked!=True")
    if str(base.payload.get("formula", "")) != F4_FORMULA:
        failures.append(f"F4 formula must be exact {F4_FORMULA!r}")

    families = base.payload.get("families")
    if not isinstance(families, Mapping):
        failures.append("F4 families missing or not a mapping")
        families = {}
    missing_fam = sorted(REQUIRED_F4_FAMILIES - set(str(k) for k in families))
    if missing_fam:
        failures.append(f"F4 missing required families {missing_fam}")

    units = list(frozen_exploratory_model_units())
    for fam, rec in families.items():
        if not isinstance(rec, Mapping):
            failures.append(f"F4 family {fam!r} record not a mapping")
            continue
        if "d_max_family" not in rec or not _is_finite_number(rec["d_max_family"]):
            failures.append(f"F4 family {fam!r} missing/nonfinite d_max_family")
            continue
        dmax = float(rec["d_max_family"])
        if "epsilon_family" not in rec or not _is_finite_number(rec["epsilon_family"]):
            failures.append(f"F4 family {fam!r} missing/nonfinite epsilon_family")
            continue
        eps = float(rec["epsilon_family"])
        expected_eps = epsilon_from_dmax(dmax)
        if abs(eps - expected_eps) > 1e-12:
            failures.append(
                f"F4 family {fam!r} epsilon_family={eps} != "
                f"frozen formula result {expected_eps}"
            )
        try:
            n_rep = int(rec.get("n_repeat_pairs"))
        except (TypeError, ValueError):
            n_rep = -1
        if n_rep != F4_REQUIRED_N_REPEAT_PAIRS:
            failures.append(
                f"F4 family {fam!r} n_repeat_pairs={n_rep} != "
                f"{F4_REQUIRED_N_REPEAT_PAIRS}"
            )
        maxima = rec.get("per_model_maxima")
        if not isinstance(maxima, Mapping):
            failures.append(f"F4 family {fam!r} per_model_maxima missing")
        else:
            # l1523 exploratory models required; l19 may be empty until confirmatory
            if str(fam) == "l1523":
                for u in units:
                    if u not in maxima or not _is_finite_number(maxima[u]):
                        failures.append(
                            f"F4 l1523 missing/nonfinite maxima for {u}"
                        )
            for k, v in maxima.items():
                if not _is_finite_number(v):
                    failures.append(
                        f"F4 family {fam!r} nonfinite maxima for {k}"
                    )
        # Required tensor family labels
        shapes = rec.get("shapes") or rec.get("tensor_families")
        if not isinstance(shapes, (list, tuple)) or len(shapes) < 1:
            failures.append(f"F4 family {fam!r} missing required tensor shapes/families")

    return ArtifactAuthResult(
        ok=len(failures) == 0,
        artifact_type=ARTIFACT_F4_V1,
        path=base.path,
        computed_sha256=base.computed_sha256,
        payload=base.payload,
        failures=failures,
        production_blocked=len(failures) > 0,
    )


def authenticate_double_bos_scan_artifact(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
) -> ArtifactAuthResult:
    del expected_model_units, expected_dataset_sha
    expected_n = len(enumerate_double_bos_descriptive_pairs())
    assert expected_n == 64
    base = _authenticate_common(
        path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_DOUBLE_BOS_V1
    )
    failures = list(base.failures)
    if base.computed_sha256 is None and not base.payload:
        return base

    if str(base.payload.get("bos_convention", "")) != "double":
        failures.append("double-BOS artifact bos_convention must be 'double'")

    # Clean-reference provenance required
    clean_prov = base.payload.get("clean_reference_provenance")
    if not isinstance(clean_prov, Mapping):
        failures.append("clean_reference_provenance missing")
    else:
        if not clean_prov.get("frozen"):
            failures.append("clean_reference_provenance.frozen must be true")
        if not clean_prov.get("source"):
            failures.append("clean_reference_provenance.source missing")

    # Per-seed pair maps required
    per_seed = base.payload.get("per_seed_pairs")
    units = list(frozen_exploratory_model_units())
    if not isinstance(per_seed, Mapping):
        failures.append(
            "per_seed_pairs missing — require exactly 64 distinct prescribed "
            "pairs per exploratory seed 42–44"
        )
    else:
        for unit in units:
            rows = per_seed.get(unit)
            if not isinstance(rows, list):
                failures.append(f"per_seed_pairs missing list for {unit}")
                continue
            pair_keys: list[tuple[int, int]] = []
            for i, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    failures.append(f"{unit} pair row {i} not a mapping")
                    continue
                heads = row.get("heads")
                key = None
                if isinstance(heads, (list, tuple)) and len(heads) == 2:
                    a, b = heads[0], heads[1]
                    if (
                        isinstance(a, (list, tuple))
                        and isinstance(b, (list, tuple))
                        and len(a) == 2
                        and len(b) == 2
                    ):
                        # ((15, h15), (16, h16))
                        if int(a[0]) == 15 and int(b[0]) == 16:
                            key = (int(a[1]), int(b[1]))
                    elif all(isinstance(x, int) for x in heads):
                        key = (int(heads[0]), int(heads[1]))
                if key is None and "head15" in row and "head16" in row:
                    try:
                        key = (int(row["head15"]), int(row["head16"]))
                    except (TypeError, ValueError):
                        key = None
                if key is None:
                    failures.append(f"{unit} pair row {i} missing valid head pair")
                    continue
                pair_keys.append(key)
            keyset = set(pair_keys)
            if len(pair_keys) != expected_n or len(keyset) != expected_n:
                failures.append(
                    f"{unit}: require {expected_n} distinct pairs; "
                    f"got n={len(pair_keys)} unique={len(keyset)}"
                )
            missing = EXPECTED_L15_L16_PAIRS - keyset
            extra = keyset - EXPECTED_L15_L16_PAIRS
            if missing:
                failures.append(
                    f"{unit}: missing {len(missing)} prescribed L15×L16 pairs"
                )
            if extra:
                failures.append(
                    f"{unit}: {len(extra)} non-prescribed pair keys present"
                )
        extra_units = set(str(k) for k in per_seed) - set(units)
        if extra_units:
            failures.append(f"unexpected per_seed_pairs units: {sorted(extra_units)}")

    # Top-level n_pairs_recorded if present must match
    if "n_pairs_recorded" in base.payload:
        try:
            if int(base.payload["n_pairs_recorded"]) != expected_n:
                failures.append(
                    f"n_pairs_recorded != {expected_n}"
                )
        except (TypeError, ValueError):
            failures.append("n_pairs_recorded malformed")

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
# Fabricated regression fixtures (always claims_real_experiment=false)
# ---------------------------------------------------------------------------


def write_fabricated_identity_artifact(
    path: Path | str,
    *,
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    all_passed: bool = True,
) -> tuple[Path, str]:
    """Write a local regression fixture — never production-acceptable."""
    units = list(model_units) if model_units is not None else list(
        frozen_exploratory_model_units()
    )
    per_model = {}
    for u in units:
        per_model[u] = [
            {
                "identity_id": i,
                "name": f"identity_{i}",
                "passed": bool(all_passed),
                "detail": "fabricated_regression_fixture",
                "numerical_comparisons": [
                    {
                        "max_abs_diff": 0.0,
                        "epsilon": 1e-3,
                        "within_tolerance": bool(all_passed),
                    }
                ],
            }
            for i in range(1, 10)
        ]
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
        "f4_tolerances_used": {"l1523": 1e-3, "l19": 1e-3},
        "all_passed": bool(all_passed),
    }
    return _write_json(path, payload)


def write_fabricated_f4_artifact(
    path: Path | str,
    *,
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    locked: bool = True,
) -> tuple[Path, str]:
    units = list(model_units) if model_units is not None else list(
        frozen_exploratory_model_units()
    )
    dmax = 0.0
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
        "batching_padding_policy": "fabricated_regression",
        "families": {
            "l1523": {
                "epsilon_family": epsilon_from_dmax(dmax),
                "d_max_family": dmax,
                "n_repeat_pairs": F4_REQUIRED_N_REPEAT_PAIRS,
                "per_model_maxima": {u: 0.0 for u in units},
                "shapes": [[1, 8, 64]],
            },
            "l19": {
                "epsilon_family": epsilon_from_dmax(dmax),
                "d_max_family": dmax,
                "n_repeat_pairs": F4_REQUIRED_N_REPEAT_PAIRS,
                "per_model_maxima": {},
                "shapes": [[1, 8, 64]],
            },
        },
        "formula": F4_FORMULA,
    }
    return _write_json(path, payload)


def write_fabricated_double_bos_artifact(
    path: Path | str,
    *,
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    n_pairs: int | None = None,
) -> tuple[Path, str]:
    units = list(model_units) if model_units is not None else list(
        frozen_exploratory_model_units()
    )
    expected_n = len(enumerate_double_bos_descriptive_pairs())
    prescribed = sorted(EXPECTED_L15_L16_PAIRS)
    if n_pairs is None:
        pair_list = prescribed
    else:
        pair_list = prescribed[: int(n_pairs)]
    per_seed = {
        u: [
            {
                "heads": [[15, h15], [16, h16]],
                "head15": h15,
                "head16": h16,
                "condition_id": f"pair|forward|L15.h{h15}+L16.h{h16}",
            }
            for h15, h16 in pair_list
        ]
        for u in units
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
        "n_pairs_recorded": len(pair_list),
        "per_seed_pairs": per_seed,
        "bos_convention": "double",
        "clean_reference_provenance": {
            "frozen": True,
            "source": "fabricated_regression_fixture",
        },
    }
    return _write_json(path, payload)


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> tuple[Path, str]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8")
    p.write_bytes(raw)
    return p, sha256_bytes(raw)
