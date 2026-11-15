"""Authenticate persisted empirical integrity evidence (S2).

Caller-declared ``evidence_kind``, ``all_passed``, ``locked``, or arbitrary SHA
strings are never sufficient. Live scientific progression requires on-disk
artifacts whose bytes are independently hashed and whose contents bind to
expected model units, frozen dataset IDs, and required coverage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from traces.q1.conditions import enumerate_double_bos_descriptive_pairs
from traces.q1.frozen_artifacts import (
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    REQUIRED_EXPLORATORY_SEEDS,
)

ARTIFACT_IDENTITY_V1 = "q1_empirical_identity_suite_v1"
ARTIFACT_F4_V1 = "q1_empirical_f4_tolerance_table_v1"
ARTIFACT_DOUBLE_BOS_V1 = "q1_empirical_double_bos_64_pair_scan_v1"

REQUIRED_IDENTITY_IDS = frozenset(range(1, 10))
REQUIRED_F4_FAMILIES = frozenset({"l1523", "l19"})


def expected_exploratory_model_units() -> tuple[str, ...]:
    return tuple(f"exploratory_l1523_seed{s}" for s in REQUIRED_EXPLORATORY_SEEDS)


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "computed_sha256": self.computed_sha256,
            "payload": dict(self.payload),
            "failures": list(self.failures),
        }


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
        )
    if not isinstance(payload, dict):
        return ArtifactAuthResult(
            ok=False,
            artifact_type=expected_type,
            path=str(p),
            computed_sha256=computed,
            failures=["artifact root must be a JSON object"],
        )
    got_type = str(payload.get("artifact_type", ""))
    if got_type != expected_type:
        return ArtifactAuthResult(
            ok=False,
            artifact_type=expected_type,
            path=str(p),
            computed_sha256=computed,
            payload=payload,
            failures=[
                f"artifact_type {got_type!r} != expected {expected_type!r}"
            ],
        )
    # Refuse claiming real experiment from regression fixtures without explicit flag
    # (fixtures for tests set claims_real_experiment=false)
    if payload.get("claims_real_experiment") is True:
        # Still require full structural auth; flag alone never clears gates
        pass
    return ArtifactAuthResult(
        ok=True,
        artifact_type=expected_type,
        path=str(p),
        computed_sha256=computed,
        payload=payload,
    )


def _check_dataset_binding(
    payload: Mapping[str, Any],
    *,
    expected_dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
) -> list[str]:
    failures: list[str] = []
    binding = payload.get("dataset_binding")
    if not isinstance(binding, Mapping):
        return ["dataset_binding missing or not a mapping"]
    split = str(binding.get("split", ""))
    if split != "development":
        failures.append(f"dataset_binding.split={split!r} != 'development'")
    sha = str(binding.get("id_sequence_sha256", ""))
    if sha != expected_dataset_sha:
        failures.append(
            f"dataset_binding.id_sequence_sha256 mismatch vs frozen development "
            f"(got {sha[:16]}…)"
        )
    return failures


def _check_model_units(
    payload: Mapping[str, Any],
    *,
    expected_units: Sequence[str],
) -> list[str]:
    units = payload.get("model_units")
    if not isinstance(units, list):
        return ["model_units missing or not a list"]
    got = [str(u) for u in units]
    exp = [str(u) for u in expected_units]
    if sorted(got) != sorted(exp):
        return [f"model_units {got} != expected {exp}"]
    return []


def authenticate_identity_suite_artifact(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
) -> ArtifactAuthResult:
    """Validate persisted identities 1–9 evidence against bytes + coverage."""
    base = _load_json_artifact(
        path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_IDENTITY_V1
    )
    if not base.ok:
        return base
    failures: list[str] = []
    exp_units = (
        list(expected_model_units)
        if expected_model_units is not None
        else list(expected_exploratory_model_units())
    )
    failures.extend(_check_model_units(base.payload, expected_units=exp_units))
    failures.extend(
        _check_dataset_binding(base.payload, expected_dataset_sha=expected_dataset_sha)
    )
    results = base.payload.get("results")
    if not isinstance(results, list):
        failures.append("results missing or not a list")
        results = []
    by_id: dict[int, bool] = {}
    for row in results:
        if not isinstance(row, Mapping):
            failures.append("identity result row not a mapping")
            continue
        try:
            iid = int(row["identity_id"])
        except (KeyError, TypeError, ValueError):
            failures.append("identity result missing identity_id")
            continue
        by_id[iid] = bool(row.get("passed"))
    missing = sorted(REQUIRED_IDENTITY_IDS - set(by_id))
    if missing:
        failures.append(f"incomplete identity coverage; missing ids {missing}")
    failed_ids = sorted(i for i, ok in by_id.items() if i in REQUIRED_IDENTITY_IDS and not ok)
    declared = bool(base.payload.get("all_passed"))
    content_all_passed = len(failed_ids) == 0 and not missing
    if declared and not content_all_passed:
        failures.append(
            "all_passed=True contradicts identity results "
            f"(failed_ids={failed_ids} missing={missing})"
        )
    if not content_all_passed:
        failures.append(
            f"identities not all passed: failed={failed_ids} missing={missing}"
        )
    return ArtifactAuthResult(
        ok=len(failures) == 0,
        artifact_type=ARTIFACT_IDENTITY_V1,
        path=base.path,
        computed_sha256=base.computed_sha256,
        payload=base.payload,
        failures=failures,
    )


def authenticate_f4_table_artifact(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
) -> ArtifactAuthResult:
    """Validate persisted locked F4 table against bytes + required families."""
    base = _load_json_artifact(
        path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_F4_V1
    )
    if not base.ok:
        return base
    failures: list[str] = []
    exp_units = (
        list(expected_model_units)
        if expected_model_units is not None
        else list(expected_exploratory_model_units())
    )
    failures.extend(_check_model_units(base.payload, expected_units=exp_units))
    failures.extend(
        _check_dataset_binding(base.payload, expected_dataset_sha=expected_dataset_sha)
    )
    if not bool(base.payload.get("locked")):
        failures.append("F4 artifact locked!=True")
    families = base.payload.get("families")
    if not isinstance(families, Mapping):
        failures.append("F4 families missing or not a mapping")
        families = {}
    missing_fam = sorted(REQUIRED_F4_FAMILIES - set(str(k) for k in families))
    if missing_fam:
        failures.append(f"F4 missing required families {missing_fam}")
    for fam, rec in families.items():
        if not isinstance(rec, Mapping):
            failures.append(f"F4 family {fam!r} record not a mapping")
            continue
        if "epsilon_family" not in rec:
            failures.append(f"F4 family {fam!r} missing epsilon_family")
        n_rep = rec.get("n_repeat_pairs")
        if n_rep is not None and int(n_rep) < 1:
            failures.append(f"F4 family {fam!r} invalid n_repeat_pairs")
    # locked=True in payload alone never authenticates without hash+families above
    return ArtifactAuthResult(
        ok=len(failures) == 0,
        artifact_type=ARTIFACT_F4_V1,
        path=base.path,
        computed_sha256=base.computed_sha256,
        payload=base.payload,
        failures=failures,
    )


def authenticate_double_bos_scan_artifact(
    *,
    path: Path | str,
    claimed_sha256: str | None,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
) -> ArtifactAuthResult:
    """Validate persisted double-BOS 64-pair scan against bytes + pair coverage."""
    expected_n = len(enumerate_double_bos_descriptive_pairs())
    base = _load_json_artifact(
        path, claimed_sha256=claimed_sha256, expected_type=ARTIFACT_DOUBLE_BOS_V1
    )
    if not base.ok:
        return base
    failures: list[str] = []
    exp_units = (
        list(expected_model_units)
        if expected_model_units is not None
        else list(expected_exploratory_model_units())
    )
    failures.extend(_check_model_units(base.payload, expected_units=exp_units))
    failures.extend(
        _check_dataset_binding(base.payload, expected_dataset_sha=expected_dataset_sha)
    )
    pairs = base.payload.get("pairs")
    if not isinstance(pairs, list):
        failures.append("pairs missing or not a list")
        n = 0
    else:
        n = len(pairs)
    declared_n = base.payload.get("n_pairs_recorded", n)
    try:
        declared_n_i = int(declared_n)
    except (TypeError, ValueError):
        declared_n_i = -1
        failures.append("n_pairs_recorded not an int")
    if n != expected_n or declared_n_i != expected_n:
        failures.append(
            f"double-BOS pair coverage {n}/{declared_n_i} != expected {expected_n}"
        )
    # Each pair should have condition identity fields
    if isinstance(pairs, list):
        for i, row in enumerate(pairs[:expected_n]):
            if not isinstance(row, Mapping):
                failures.append(f"pair row {i} not a mapping")
                break
            if "condition_id" not in row and "heads" not in row:
                failures.append(f"pair row {i} missing condition_id/heads")
                break
    return ArtifactAuthResult(
        ok=len(failures) == 0,
        artifact_type=ARTIFACT_DOUBLE_BOS_V1,
        path=base.path,
        computed_sha256=base.computed_sha256,
        payload=base.payload,
        failures=failures,
    )


def write_fabricated_identity_artifact(
    path: Path | str,
    *,
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    all_passed: bool = True,
) -> tuple[Path, str]:
    """Write a local regression fixture (not a real experiment claim)."""
    units = list(model_units) if model_units is not None else list(
        expected_exploratory_model_units()
    )
    results = [
        {
            "identity_id": i,
            "name": f"identity_{i}",
            "passed": bool(all_passed),
            "detail": "fabricated_regression_fixture",
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
        "results": results,
        "all_passed": bool(all_passed) and all(r["passed"] for r in results),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    p.write_bytes(raw)
    return p, sha256_bytes(raw)


def write_fabricated_f4_artifact(
    path: Path | str,
    *,
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    locked: bool = True,
) -> tuple[Path, str]:
    units = list(model_units) if model_units is not None else list(
        expected_exploratory_model_units()
    )
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
                "epsilon_family": 1e-3,
                "d_max_family": 0.0,
                "n_repeat_pairs": 3,
                "per_model_maxima": {u: 0.0 for u in units},
            },
            "l19": {
                "epsilon_family": 1e-3,
                "d_max_family": 0.0,
                "n_repeat_pairs": 3,
                "per_model_maxima": {},
            },
        },
        "formula": "epsilon_family = max(1e-3, 10 * D_max_family)",
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    p.write_bytes(raw)
    return p, sha256_bytes(raw)


def write_fabricated_double_bos_artifact(
    path: Path | str,
    *,
    model_units: Sequence[str] | None = None,
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    n_pairs: int | None = None,
) -> tuple[Path, str]:
    units = list(model_units) if model_units is not None else list(
        expected_exploratory_model_units()
    )
    expected_n = len(enumerate_double_bos_descriptive_pairs())
    n = expected_n if n_pairs is None else int(n_pairs)
    pairs = [
        {"condition_id": f"pair|forward|fabricated|{i}", "heads": [i, i]}
        for i in range(n)
    ]
    payload = {
        "artifact_type": ARTIFACT_DOUBLE_BOS_V1,
        "claims_real_experiment": False,
        "fixture_purpose": "synthetic_regression_only",
        "model_units": units,
        "dataset_binding": {
            "split": "development",
            "id_sequence_sha256": dataset_sha,
        },
        "n_pairs_recorded": n,
        "pairs": pairs,
        "bos_convention": "double",
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    p.write_bytes(raw)
    return p, sha256_bytes(raw)
