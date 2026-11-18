"""CPU-only B2 exploratory prep guards (identities + double-BOS); no GPU."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from traces.q1.b2_exploratory_cpu_prep import (
    DEFAULT_B2_CPU_PREP_SCOPE,
    FORBIDDEN_PIPELINE_ENTRYPOINTS,
    PUBLISHED_F4_SHA256,
    assert_b2_cpu_prep_scope,
    build_cpu_prep_package,
    estimate_double_bos_gpu_workload,
    estimate_identity_gpu_workload,
    identity_family_plan,
    load_published_f4_epsilons,
    refuse_confirmatory_callback,
    refuse_held_out_access,
    run_cpu_identity_smoke_with_f4_epsilons,
    scientific_flag_must_be_unset,
)
from traces.q1.conditions import enumerate_double_bos_descriptive_pairs
from traces.q1.empirical_evidence import REQUIRED_IDENTITY_IDS

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_sci_flag():
    os.environ.pop("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", None)
    os.environ.pop("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", None)
    yield
    os.environ.pop("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", None)


def test_b2_cpu_prep_scope_ok():
    assert assert_b2_cpu_prep_scope() == []
    assert DEFAULT_B2_CPU_PREP_SCOPE.allows_gpu_in_this_module is False
    assert DEFAULT_B2_CPU_PREP_SCOPE.allows_alter_single_bos_stopping_rule is False
    assert DEFAULT_B2_CPU_PREP_SCOPE.published_f4_sha256 == PUBLISHED_F4_SHA256


def test_b2_cpu_prep_scope_rejects_drift():
    assert any(
        "held-out" in f
        for f in assert_b2_cpu_prep_scope(
            replace(DEFAULT_B2_CPU_PREP_SCOPE, allows_held_out=True)
        )
    )
    assert any(
        "stopping rule" in f
        for f in assert_b2_cpu_prep_scope(
            replace(
                DEFAULT_B2_CPU_PREP_SCOPE,
                allows_alter_single_bos_stopping_rule=True,
            )
        )
    )
    assert any(
        "CPU-only" in f
        for f in assert_b2_cpu_prep_scope(
            replace(DEFAULT_B2_CPU_PREP_SCOPE, allows_gpu_in_this_module=True)
        )
    )


def test_refuse_held_out_and_confirmatory():
    with pytest.raises(PermissionError, match="held-out"):
        refuse_held_out_access(REPO / "results/q1_selection/held_out_instructions.json")
    for name in FORBIDDEN_PIPELINE_ENTRYPOINTS:
        with pytest.raises(PermissionError):
            refuse_confirmatory_callback(name)
    assert scientific_flag_must_be_unset()


def test_load_published_f4_and_identity_smoke():
    f4 = load_published_f4_epsilons(REPO)
    assert f4["sha256"] == PUBLISHED_F4_SHA256
    assert f4["locked"] is True
    assert all(v == 0.001 for v in f4["epsilon_family"].values())
    smoke = run_cpu_identity_smoke_with_f4_epsilons(f4["epsilon_family"])
    assert smoke["all_passed"] is True
    assert len(smoke["primary_passed"]) == 9


def test_identity_family_plan_covers_1_to_9():
    plan = identity_family_plan()
    assert set(int(k) for k in plan["identities"]) == set(REQUIRED_IDENTITY_IDS)
    assert plan["requires_bound_f4_sha256"] == PUBLISHED_F4_SHA256
    assert plan["adapter_off_boundary_l1523"] == 15
    assert plan["identities"]["7"]["kind"] == "asr_gate"
    assert plan["identities"]["6"]["tensor_families"] == [
        "boundary_residual",
        "upstream_activation",
    ]


def test_double_bos_catalog_is_64_and_descriptive():
    pairs = enumerate_double_bos_descriptive_pairs()
    assert len(pairs) == 64
    assert all(c.bos_convention == "double" for c in pairs)
    assert all(c.direction == "forward" for c in pairs)
    est = estimate_double_bos_gpu_workload()
    assert est["n_pairs"] == 64
    assert est["n_patched_forwards"] == 3 * 64 * 50
    assert est["alters_single_bos_stopping_rule"] is False
    assert est["gpu_authorized_by_this_module"] is False
    assert est["within_plan_cap"] is True


def test_identity_workload_estimate_is_planning_only():
    est = estimate_identity_gpu_workload()
    assert est["gpu_authorized_by_this_module"] is False
    assert est["n_seeds"] == 3
    assert est["within_plan_cap"] is True


def test_build_cpu_prep_package(tmp_path):
    pkg = build_cpu_prep_package(REPO, tmp_path / "prep")
    assert pkg["verdict"] == "CPU_PREP_READY"
    assert pkg["published_f4"]["sha256"] == PUBLISHED_F4_SHA256
    assert pkg["identity_algebraic_smoke"]["all_passed"] is True
    assert pkg["synthetic_fixtures"]["identity_production_blocked_expected"] is True
    assert pkg["synthetic_fixtures"]["identity_auth_ok"] is False
    assert pkg["synthetic_fixtures"]["double_bos_auth_ok"] is False
    assert pkg["synthetic_fixtures"]["double_bos_measurement_failures"] == []
    assert pkg["development_prompts"]["held_out_bytes_read"] == 0
    assert Path(pkg["package_path"]).is_file()
