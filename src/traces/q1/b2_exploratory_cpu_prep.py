"""CPU preparation for remaining exploratory B2 gates (identities + double-BOS).

Binds to the published F4 tolerance table. Does **not** run GPU inference,
held-out access, confirmatory models, or ``run_live_scientific_pipeline``.
Does **not** alter the single-BOS exploratory stopping rule.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from traces.q1.conditions import enumerate_double_bos_descriptive_pairs
from traces.q1.empirical_evidence import (
    ARTIFACT_DOUBLE_BOS_V1,
    ARTIFACT_F4_V1,
    ARTIFACT_IDENTITY_V1,
    IDENTITY_TENSOR_FAMILIES,
    REQUIRED_F4_TENSOR_FAMILIES,
    REQUIRED_IDENTITY_IDS,
    authenticate_double_bos_scan_artifact,
    authenticate_f4_table_artifact,
    authenticate_identity_suite_artifact,
    frozen_exploratory_model_units,
    validate_double_bos_scan_measurements,
    validate_f4_tensor_family_calibration,
    validate_identity_suite_measurements,
    write_fabricated_double_bos_artifact,
    write_fabricated_identity_artifact,
)
from traces.q1.f4_exploratory import (
    F4_SEEDS,
    INTENDED_IMAGE_ID,
    PINNED_CHECKPOINT_MANIFEST_SHA256,
    PINNED_RUNTIME_LOCK_SHA256,
    refuse_held_out_access as _refuse_held_out,
    select_f4_development_prompts,
)
from traces.q1.frozen_artifacts import (
    EXPECTED_N_DEV,
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    load_ordered_dolly_ids,
)
from traces.q1.identities import (
    adapter_off_boundary_layer,
    run_synthetic_identity_suite,
)

B2_CPU_PREP_STAGE = "b2_exploratory_identities_and_double_bos_cpu_prep"
PUBLISHED_F4_ARTIFACT_RELPATH = (
    "results/q1_implementation_prep/F4_TOLERANCE_TABLE.json"
)
PUBLISHED_F4_SHA256 = (
    "80d7985c3d6241c5661cfacccbea2764c0dec429a9f637131d531727de41d021"
)
SCI_AUTH_ENV = "TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION"
FORBIDDEN_PIPELINE_ENTRYPOINTS = (
    "run_live_scientific_pipeline",
    "run_live_scientific_pipeline_from_hooks",
)

# Internal planning caps only — GPU runners not launched by this module.
B2_IDENTITY_PLAN_CAP_USD = 3.0
B2_DOUBLE_BOS_PLAN_CAP_USD = 5.0
STAGE_C_LOAD_WALL_S = 27.93
STAGE_C_INTACT_MEAN_S = 0.073  # F4 measured intact mean
STAGE_C_PATCH_MEAN_S = 0.12  # Stage C reverse joint mean as patch proxy


@dataclass(frozen=True)
class B2CpuPrepScope:
    stage: str = B2_CPU_PREP_STAGE
    seeds: tuple[int, ...] = F4_SEEDS
    n_prompts: int = EXPECTED_N_DEV
    required_identities: tuple[int, ...] = tuple(sorted(REQUIRED_IDENTITY_IDS))
    n_double_bos_pairs: int = 64
    published_f4_sha256: str = PUBLISHED_F4_SHA256
    image_id: str = INTENDED_IMAGE_ID
    runtime_lock_sha256: str = PINNED_RUNTIME_LOCK_SHA256
    checkpoint_manifest_sha256: str = PINNED_CHECKPOINT_MANIFEST_SHA256
    allows_held_out: bool = False
    allows_live_scientific_pipeline: bool = False
    allows_confirmatory_models: bool = False
    allows_alter_single_bos_stopping_rule: bool = False
    allows_gpu_in_this_module: bool = False
    preserves_f4_artifact: bool = True


DEFAULT_B2_CPU_PREP_SCOPE = B2CpuPrepScope()


def assert_b2_cpu_prep_scope(
    scope: B2CpuPrepScope = DEFAULT_B2_CPU_PREP_SCOPE,
) -> list[str]:
    failures: list[str] = []
    if scope.stage != B2_CPU_PREP_STAGE:
        failures.append(f"stage {scope.stage!r} != {B2_CPU_PREP_STAGE!r}")
    if tuple(scope.seeds) != F4_SEEDS:
        failures.append(f"seeds {scope.seeds} != exploratory {F4_SEEDS}")
    if scope.n_prompts != EXPECTED_N_DEV:
        failures.append(f"n_prompts {scope.n_prompts} != {EXPECTED_N_DEV}")
    if set(scope.required_identities) != set(REQUIRED_IDENTITY_IDS):
        failures.append("required_identities drift vs REQUIRED_IDENTITY_IDS")
    if scope.n_double_bos_pairs != len(enumerate_double_bos_descriptive_pairs()):
        failures.append("n_double_bos_pairs != frozen 64-pair catalog")
    if scope.published_f4_sha256 != PUBLISHED_F4_SHA256:
        failures.append("published F4 SHA drift")
    if scope.image_id != INTENDED_IMAGE_ID:
        failures.append("image_id drift")
    if scope.runtime_lock_sha256 != PINNED_RUNTIME_LOCK_SHA256:
        failures.append("runtime lock SHA drift")
    if scope.checkpoint_manifest_sha256 != PINNED_CHECKPOINT_MANIFEST_SHA256:
        failures.append("checkpoint manifest SHA drift")
    if scope.allows_held_out:
        failures.append("held-out forbidden in B2 exploratory CPU prep")
    if scope.allows_live_scientific_pipeline:
        failures.append("live scientific pipeline forbidden")
    if scope.allows_confirmatory_models:
        failures.append("confirmatory models forbidden")
    if scope.allows_alter_single_bos_stopping_rule:
        failures.append("must not alter single-BOS exploratory stopping rule")
    if scope.allows_gpu_in_this_module:
        failures.append("this module is CPU-only; GPU must use a dedicated runner")
    if not scope.preserves_f4_artifact:
        failures.append("F4 published artifact must be preserved")
    return failures


def refuse_held_out_access(path: Path | str | None) -> None:
    _refuse_held_out(path)


def refuse_confirmatory_callback(name: str) -> None:
    if name in FORBIDDEN_PIPELINE_ENTRYPOINTS:
        raise PermissionError(
            f"B2 exploratory CPU prep forbids confirmatory entrypoint {name!r}"
        )


def scientific_flag_must_be_unset() -> bool:
    return os.environ.get(SCI_AUTH_ENV) not in {"1", "true", "TRUE", "yes", "YES"}


def load_published_f4_epsilons(
    repo_root: Path | str,
    *,
    expected_sha256: str = PUBLISHED_F4_SHA256,
) -> dict[str, Any]:
    """Authenticate the published F4 table and return locked family epsilons."""
    root = Path(repo_root)
    path = root / PUBLISHED_F4_ARTIFACT_RELPATH
    if not path.is_file():
        raise FileNotFoundError(f"published F4 artifact missing: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"published F4 SHA mismatch: file={digest} expected={expected_sha256}"
        )
    prompts = [str(x) for x in load_ordered_dolly_ids(
        root / "results/q1_selection/development_instructions.json"
    )]
    auth = authenticate_f4_table_artifact(
        path=path,
        claimed_sha256=digest,
        frozen_prompt_ids=prompts,
        repo_root=root,
        check_production_unlock=False,
        reject_synthetic=True,
    )
    if not auth.ok or auth.payload is None:
        raise RuntimeError(
            "published F4 authentication failed: " + "; ".join(auth.failures[:12])
        )
    schema = validate_f4_tensor_family_calibration(
        auth.payload, frozen_prompt_ids=prompts
    )
    if schema:
        raise RuntimeError("published F4 schema failed: " + "; ".join(schema[:12]))
    tfams = auth.payload["tensor_families"]
    eps = {fam: float(tfams[fam]["epsilon_family"]) for fam in sorted(REQUIRED_F4_TENSOR_FAMILIES)}
    dmax = {fam: float(tfams[fam]["d_max_family"]) for fam in sorted(REQUIRED_F4_TENSOR_FAMILIES)}
    return {
        "path": str(path),
        "sha256": digest,
        "artifact_type": ARTIFACT_F4_V1,
        "epsilon_family": eps,
        "d_max_family": dmax,
        "locked": bool(auth.payload.get("locked")),
        "auth_ok": True,
        "model_units": list(auth.payload.get("model_units") or []),
        "dataset_id_sequence_sha256": (
            (auth.payload.get("dataset_binding") or {}).get("id_sequence_sha256")
        ),
    }


def identity_family_plan() -> dict[str, Any]:
    """Map identities 1–9 to required F4 families / categorical gates."""
    plan: dict[str, Any] = {}
    for iid in sorted(REQUIRED_IDENTITY_IDS):
        if iid == 7:
            plan[str(iid)] = {
                "kind": "asr_gate",
                "threshold": 0.01,
                "tensor_families": [],
            }
        elif iid == 9:
            plan[str(iid)] = {
                "kind": "hook_clear",
                "tensor_families": [],
            }
        else:
            fams = sorted(IDENTITY_TENSOR_FAMILIES[iid])
            plan[str(iid)] = {
                "kind": "numerical",
                "tensor_families": fams,
            }
    return {
        "adapter_off_boundary_l1523": adapter_off_boundary_layer("l1523"),
        "boundary_residual_layer": 19,
        "identities": plan,
        "model_units": list(frozen_exploratory_model_units()),
        "requires_bound_f4_sha256": PUBLISHED_F4_SHA256,
    }


def estimate_identity_gpu_workload(
    *,
    load_s: float = STAGE_C_LOAD_WALL_S,
    intact_s: float = STAGE_C_INTACT_MEAN_S,
    intervention_s: float = STAGE_C_PATCH_MEAN_S,
    usd_per_hour: float = 1.10,
) -> dict[str, Any]:
    """Conservative planning estimate only — not an authorization to run GPU."""
    n_seeds = len(F4_SEEDS)
    # Intact T/C + several hybrid/patch/adapter-off/gen arms (order-of-magnitude).
    # Exact arm catalog belongs to a dedicated identity runner; this is budgeting.
    n_intact = n_seeds * EXPECTED_N_DEV * 2
    n_intervention = n_seeds * EXPECTED_N_DEV * 12  # hybrids + patches + adapter-off proxies
    n_gen = n_seeds * EXPECTED_N_DEV * 2  # trigger/control gen for ASR
    compute_s = (
        n_seeds * load_s
        + n_intact * intact_s
        + n_intervention * intervention_s
        + n_gen * 0.5
        + 240.0
    ) * 1.3
    usd = (compute_s / 3600.0) * usd_per_hour
    return {
        "n_seeds": n_seeds,
        "n_prompts": EXPECTED_N_DEV,
        "approx_intact_forwards": n_intact,
        "approx_intervention_forwards": n_intervention,
        "approx_generation_calls": n_gen,
        "estimated_wall_s": compute_s,
        "estimated_wall_min": compute_s / 60.0,
        "estimated_usd_a10g": usd,
        "internal_plan_cap_usd": B2_IDENTITY_PLAN_CAP_USD,
        "within_plan_cap": usd <= B2_IDENTITY_PLAN_CAP_USD,
        "gpu_authorized_by_this_module": False,
        "notes": (
            "Order-of-magnitude planning from Stage C/F4 timings; "
            "dedicated identity runner must enumerate exact arms before launch."
        ),
    }


def estimate_double_bos_gpu_workload(
    *,
    load_s: float = STAGE_C_LOAD_WALL_S,
    intact_s: float = STAGE_C_INTACT_MEAN_S,
    patch_s: float = STAGE_C_PATCH_MEAN_S,
    usd_per_hour: float = 1.10,
) -> dict[str, Any]:
    n_seeds = len(F4_SEEDS)
    n_pairs = len(enumerate_double_bos_descriptive_pairs())
    n_intact = n_seeds * EXPECTED_N_DEV * 2  # clean C + TT under double BOS
    n_patched = n_seeds * n_pairs * EXPECTED_N_DEV
    compute_s = (n_seeds * load_s + n_intact * intact_s + n_patched * patch_s + 300.0) * 1.25
    usd = (compute_s / 3600.0) * usd_per_hour
    return {
        "n_seeds": n_seeds,
        "n_prompts": EXPECTED_N_DEV,
        "n_pairs": n_pairs,
        "n_intact_double_bos_forwards": n_intact,
        "n_patched_forwards": n_patched,
        "n_total_forwards": n_intact + n_patched,
        "estimated_wall_s": compute_s,
        "estimated_wall_min": compute_s / 60.0,
        "estimated_usd_a10g": usd,
        "internal_plan_cap_usd": B2_DOUBLE_BOS_PLAN_CAP_USD,
        "within_plan_cap": usd <= B2_DOUBLE_BOS_PLAN_CAP_USD,
        "gpu_authorized_by_this_module": False,
        "alters_single_bos_stopping_rule": False,
        "notes": (
            "Descriptive F6 exception only; own double-BOS clean references; "
            "must not alter single-BOS exploratory stopping rule."
        ),
    }


def run_cpu_identity_smoke_with_f4_epsilons(
    f4_epsilons: Mapping[str, float],
) -> dict[str, Any]:
    """Synthetic algebraic identity suite using locked F4 family epsilons."""
    report = run_synthetic_identity_suite(
        epsilon_logits=float(f4_epsilons["final_logits"]),
        epsilon_residual=float(f4_epsilons["boundary_residual"]),
        epsilon_activation=float(
            min(
                float(f4_epsilons["upstream_activation"]),
                float(f4_epsilons["boundary_residual"]),
            )
        ),
        adapter_boundary_layer=adapter_off_boundary_layer("l1523"),
    )
    return {
        "all_passed": bool(report.all_passed),
        "n_results": len(report.results),
        "primary_passed": [
            {"identity_id": r.identity_id, "name": r.name, "passed": r.passed}
            for r in report.results
            if 1 <= r.identity_id <= 9
        ],
        "f4_epsilons_used": dict(f4_epsilons),
    }


def build_cpu_prep_package(
    repo_root: Path | str,
    out_dir: Path | str,
) -> dict[str, Any]:
    """Write CPU-prep evidence: F4 bind + synthetic schema fixtures + estimates."""
    root = Path(repo_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scope_fails = assert_b2_cpu_prep_scope()
    if scope_fails:
        raise RuntimeError("B2 CPU prep scope failed: " + "; ".join(scope_fails))
    if not scientific_flag_must_be_unset():
        raise PermissionError("scientific flag must be unset during CPU prep")

    prompts_sel = select_f4_development_prompts(root)
    prompt_ids = [str(x) for x in prompts_sel["dolly_ids"]]
    # Held-out may exist on disk; CPU prep must never open it (selection already unread).
    if int(prompts_sel.get("held_out_bytes_read") or 0) != 0:
        raise PermissionError("B2 CPU prep read held-out bytes — refuse")

    f4 = load_published_f4_epsilons(root)
    smoke = run_cpu_identity_smoke_with_f4_epsilons(f4["epsilon_family"])

    # Synthetic schema fixtures (explicitly marked; production-blocked)
    id_path, _ = write_fabricated_identity_artifact(
        out / "SYNTHETIC_identity_schema_fixture.json",
        max_abs_diff=0.0,
        all_passed=True,
        f4_calibration_artifact_sha256=f4["sha256"],
        identity7_trigger_asr=0.0,
    )
    id_payload = json.loads(Path(id_path).read_text())
    id_payload["f4_calibration_artifact_sha256"] = f4["sha256"]
    id_payload["f4_tolerances_used"] = dict(f4["epsilon_family"])
    id_text = json.dumps(id_payload, sort_keys=True, separators=(",", ":")) + "\n"
    Path(id_path).write_text(id_text)
    id_sha = hashlib.sha256(id_text.encode()).hexdigest()
    Path(str(id_path) + ".sha256").write_text(id_sha + "\n")

    id_meas_fails = validate_identity_suite_measurements(
        id_payload, f4_epsilons=f4["epsilon_family"]
    )
    id_auth = authenticate_identity_suite_artifact(
        path=id_path,
        claimed_sha256=id_sha,
        f4_artifact_path=root / PUBLISHED_F4_ARTIFACT_RELPATH,
        f4_claimed_sha256=f4["sha256"],
        frozen_prompt_ids=prompt_ids,
        repo_root=root,
        check_production_unlock=False,
        reject_synthetic=True,
    )

    db_path, db_sha = write_fabricated_double_bos_artifact(
        out / "SYNTHETIC_double_bos_schema_fixture.json",
        prompt_ids=prompt_ids,
    )
    db_payload = json.loads(Path(db_path).read_text())
    db_meas_fails = validate_double_bos_scan_measurements(
        db_payload, frozen_prompt_ids=prompt_ids
    )
    db_auth = authenticate_double_bos_scan_artifact(
        path=db_path,
        claimed_sha256=db_sha,
        frozen_prompt_ids=prompt_ids,
        repo_root=root,
        check_production_unlock=False,
        reject_synthetic=True,
    )

    package = {
        "document_type": "q1_b2_exploratory_cpu_prep_package",
        "stage": B2_CPU_PREP_STAGE,
        "scope": asdict(DEFAULT_B2_CPU_PREP_SCOPE),
        "scope_failures": scope_fails,
        "scientific_flag_unset": scientific_flag_must_be_unset(),
        "published_f4": f4,
        "identity_family_plan": identity_family_plan(),
        "identity_algebraic_smoke": smoke,
        "identity_gpu_workload_estimate": estimate_identity_gpu_workload(),
        "double_bos_gpu_workload_estimate": estimate_double_bos_gpu_workload(),
        "double_bos_pair_catalog_n": len(enumerate_double_bos_descriptive_pairs()),
        "development_prompts": {
            "n": prompts_sel["n"],
            "id_sequence_sha256": prompts_sel["full_development_id_sequence_sha256"],
            "held_out_bytes_read": prompts_sel["held_out_bytes_read"],
        },
        "synthetic_fixtures": {
            "identity_path": str(id_path),
            "identity_sha256": id_sha,
            "identity_measurement_failures": id_meas_fails,
            "identity_auth_ok": id_auth.ok,
            "identity_auth_failures": list(id_auth.failures),
            "identity_production_blocked_expected": True,
            "double_bos_path": str(db_path),
            "double_bos_sha256": db_sha,
            "double_bos_measurement_failures": db_meas_fails,
            "double_bos_auth_ok": db_auth.ok,
            "double_bos_auth_failures": list(db_auth.failures),
            "double_bos_production_blocked_expected": True,
            "note": (
                "Synthetic fixtures exercise schema/negative paths only; "
                "they are not scientific evidence."
            ),
        },
        "artifact_types": {
            "f4": ARTIFACT_F4_V1,
            "identity": ARTIFACT_IDENTITY_V1,
            "double_bos": ARTIFACT_DOUBLE_BOS_V1,
        },
        "stop_rules": {
            "no_held_out": True,
            "no_confirmatory": True,
            "no_main_merge": True,
            "no_alter_single_bos_stopping_rule": True,
            "preserve_f4": True,
            "gpu_runners_separate": True,
        },
        "verdict": "CPU_PREP_READY",
    }
    out_path = out / "B2_EXPLORATORY_CPU_PREP_PACKAGE.json"
    text = json.dumps(package, indent=2, sort_keys=True) + "\n"
    out_path.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    out_path.with_suffix(".json.sha256").write_text(digest + "\n")
    package["package_path"] = str(out_path)
    package["package_sha256"] = digest
    return package
