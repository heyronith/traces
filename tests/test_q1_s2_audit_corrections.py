"""Agent 1 S2 audit corrections — six confirmed defects + positive control."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traces.q1.empirical_evidence import (
    ARTIFACT_F4_V1,
    DOUBLE_BOS_PAYLOAD_TOKEN_A,
    IDENTITY_7_ASR_THRESHOLD,
    authenticate_engineering_test_battery,
    authenticate_empirical_scientific_battery,
    authenticate_f4_table_artifact,
    authenticate_identity_suite_artifact,
    validate_double_bos_scan_measurements,
    validate_f4_tensor_family_calibration,
    validate_identity_suite_measurements,
    write_fabricated_double_bos_artifact,
    write_fabricated_f4_artifact,
    write_fabricated_identity_artifact,
    write_schema_valid_test_battery_package,
)
from traces.q1.empirical_trust import PINNED_EMPIRICAL_MANIFEST_SHA256
from traces.q1.frozen_artifacts import (
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    load_ordered_dolly_ids,
)
from traces.q1.integrity_gates import evaluate_preconfirmation_gates
from traces.q1.pipeline_adapters import (
    assemble_bundles_from_condition_records,
    mock_condition_records_for_unit,
)
from traces.q1.scientific_execution import (
    LiveForwardHooks,
    UnitSpec,
    expected_frozen_unit_specs,
    run_live_scientific_pipeline,
    scientific_auth_status,
)
from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION

REPO = Path(__file__).resolve().parents[1]


def _frozen_dev():
    return load_ordered_dolly_ids(
        REPO / "results/q1_selection/development_instructions.json"
    )


def _frozen_held_out():
    return load_ordered_dolly_ids(
        REPO / "results/q1_selection/held_out_instructions.json"
    )


def _pass_preflight(monkeypatch):
    from traces.q1.scientific_execution import PreflightReport

    monkeypatch.setattr(
        "traces.q1.scientific_execution.run_scientific_preflight",
        lambda *_a, **_k: PreflightReport(ok=True, failures=[], checks={"t": True}),
    )
    monkeypatch.setattr(
        "traces.q1.scientific_execution.require_scientific_authorization",
        lambda: None,
    )


def _spy_hooks():
    calls: dict[str, list[str]] = {"load": [], "execute": []}

    def load(spec: UnitSpec) -> dict:
        calls["load"].append(spec.model_unit)
        return {
            "model": object(),
            "tokenizer": object(),
            "model_unit": spec.model_unit,
            "family": spec.family,
            "role": spec.role,
            "seed": spec.seed,
            "base_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION
            if spec.family != "base"
            else None,
            "weight_file_sha256": {
                f"base:/fake/{spec.model_unit}.safetensors": "a" * 64,
                f"adapter:/fake/{spec.model_unit}_adapter.safetensors": "b" * 64,
            },
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
        }

    def donors(spec: UnitSpec, _m):
        ids = (
            [str(x) for x in _frozen_dev()]
            if spec.role == "exploratory_dev"
            else [f"p{i}" for i in range(8)]
        )
        return {
            "n_donor_caches": len(ids),
            "clean_references_frozen": True,
            "_engineering_skip_observations": True,
            "instruction_ids": ids,
        }

    def execute(spec: UnitSpec, _m, _d):
        calls["execute"].append(spec.model_unit)
        ids = (
            [str(x) for x in _frozen_dev()]
            if spec.role == "exploratory_dev"
            else [f"p{i}" for i in range(8)]
        )
        recs = mock_condition_records_for_unit(spec, ids, nominated_effect=2.0)
        b, i = assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.95,
            clean_false_fire=0.01,
            first_token_validity=0.96,
            allocation_ids=ids if spec.role == "exploratory_dev" else None,
        )
        return b, i

    return (
        LiveForwardHooks(
            load_model_for_unit=load,
            capture_donors_for_unit=donors,
            execute_unit=execute,
        ),
        calls,
    )


def _confirmatory() -> set[str]:
    return {
        s.model_unit
        for s in expected_frozen_unit_specs()
        if s.role in {"primary_confirmatory", "secondary_replication"}
    }


def _assert_zero_confirmatory(out, calls):
    assert out["confirmatory_executed"] is False
    assert not _confirmatory().intersection(calls["load"])
    assert not _confirmatory().intersection(calls["execute"])


# --- Defect 1: two-phase battery ---


def test_d1_positive_control_test_pin_clears_engineering_battery_not_production(
    tmp_path,
):
    """Engineering-only interface may succeed; production gate must stay blocked."""
    assert PINNED_EMPIRICAL_MANIFEST_SHA256 is None
    assert scientific_auth_status().authorized is False
    prompts = _frozen_dev()
    pkg = write_schema_valid_test_battery_package(
        package_dir=tmp_path / "pkg",
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
    )
    eng = authenticate_engineering_test_battery(
        f4_path=pkg["f4_path"],
        f4_sha256=pkg["f4_sha256"],
        identity_path=pkg["identity_path"],
        identity_sha256=pkg["identity_sha256"],
        double_bos_path=pkg["double_bos_path"],
        double_bos_sha256=pkg["double_bos_sha256"],
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
        test_pinned_digest=pkg["test_pinned_digest"],
    )
    assert eng.schema_ok is True
    assert eng.test_battery_ok is True
    assert eng.production_unlocked is False

    # Production battery / live gate must NOT clear via test pin
    prod = authenticate_empirical_scientific_battery(
        f4_path=pkg["f4_path"],
        f4_sha256=pkg["f4_sha256"],
        identity_path=pkg["identity_path"],
        identity_sha256=pkg["identity_sha256"],
        double_bos_path=pkg["double_bos_path"],
        double_bos_sha256=pkg["double_bos_sha256"],
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
    )
    assert prod.ok is False
    assert prod.production_unlocked is False

    report = evaluate_preconfirmation_gates(
        development_n_eligible=48,
        development_n_frozen=50,
        exploratory_first_token_validity=0.96,
        identity_suite={
            "artifact_path": pkg["identity_path"],
            "sha256": pkg["identity_sha256"],
            "_test_pinned_digest": pkg["test_pinned_digest"],
            "_allow_synthetic_for_test_battery": True,
        },
        f4_table={
            "artifact_path": pkg["f4_path"],
            "sha256": pkg["f4_sha256"],
            "_test_pinned_digest": pkg["test_pinned_digest"],
            "_allow_synthetic_for_test_battery": True,
        },
        double_bos_evidence={
            "artifact_path": pkg["double_bos_path"],
            "sha256": pkg["double_bos_sha256"],
            "_test_pinned_digest": pkg["test_pinned_digest"],
            "_allow_synthetic_for_test_battery": True,
        },
        require_empirical_scientific=True,
        repo_root=str(tmp_path / "repo"),
        frozen_prompt_ids=prompts,
    )
    assert report.progression_blocked is True
    assert report.gates["empirical_scientific_battery"].status != "complete"
    assert scientific_auth_status().authorized is False


def test_d1_missing_sibling_blocks_battery(tmp_path):
    prompts = _frozen_dev()
    pkg = write_schema_valid_test_battery_package(
        package_dir=tmp_path / "pkg",
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
    )
    report = evaluate_preconfirmation_gates(
        development_n_eligible=48,
        development_n_frozen=50,
        exploratory_first_token_validity=0.96,
        identity_suite={
            "artifact_path": pkg["identity_path"],
            "sha256": pkg["identity_sha256"],
        },
        f4_table={"artifact_path": pkg["f4_path"], "sha256": pkg["f4_sha256"]},
        # missing double-BOS
        require_empirical_scientific=True,
        repo_root=str(tmp_path / "repo"),
        frozen_prompt_ids=prompts,
    )
    assert report.progression_blocked is True


def test_d1_modified_artifact_blocks_and_zero_confirmatory(tmp_path, monkeypatch):
    prompts = _frozen_dev()
    pkg = write_schema_valid_test_battery_package(
        package_dir=tmp_path / "pkg",
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
    )
    # Corrupt identity bytes after packaging (claimed sha no longer matches)
    Path(pkg["identity_path"]).write_text('{"tampered": true}')
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "live",
        integrity_evidence={
            "identity_artifact": {
                "artifact_path": pkg["identity_path"],
                "sha256": pkg["identity_sha256"],
            },
            "f4_artifact": {
                "artifact_path": pkg["f4_path"],
                "sha256": pkg["f4_sha256"],
            },
            "double_bos_artifact": {
                "artifact_path": pkg["double_bos_path"],
                "sha256": pkg["double_bos_sha256"],
            },
            "_test_pinned_digest": pkg["test_pinned_digest"],
            "_allow_synthetic_for_test_battery": True,
        },
    )
    _assert_zero_confirmatory(out, calls)


# --- Defect 2: pin mismatch fatal; evidence repo_root ignored ---


def test_d2_wrong_pin_plus_hmac_fails(monkeypatch):
    from traces.q1.empirical_trust import (
        canonical_manifest_bytes,
        make_test_trust_material,
        sign_manifest_hmac,
        verify_manifest_cryptography,
    )

    monkeypatch.setenv("TRACES_Q1_EMPIRICAL_TEST_TRUST", "1")
    key, _ = make_test_trust_material()
    payload = {"artifact_type": "q1_empirical_execution_provenance_manifest_v1", "x": 1}
    canon = canonical_manifest_bytes(payload)
    sig = sign_manifest_hmac(canon, key)
    bad = verify_manifest_cryptography(
        manifest_bytes=canon,
        signature_hex=sig,
        trusted_key=key,
        pinned_digest="f" * 64,
        allow_test_trust=True,
    )
    assert bad.ok is False
    assert any("pinned" in f for f in bad.failures)


def test_d2_evidence_repo_root_cannot_redirect_manifest(tmp_path, monkeypatch):
    prompts = _frozen_dev()
    evil = tmp_path / "evil_repo"
    good_pkg = write_schema_valid_test_battery_package(
        package_dir=tmp_path / "pkg",
        repo_root=evil,
        frozen_prompt_ids=prompts,
    )
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    # Executor root is REPO (no matching test pin there); evidence points at evil
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "redir",
        integrity_evidence={
            "repo_root": str(evil),  # must be ignored
            "identity_artifact": {
                "artifact_path": good_pkg["identity_path"],
                "sha256": good_pkg["identity_sha256"],
            },
            "f4_artifact": {
                "artifact_path": good_pkg["f4_path"],
                "sha256": good_pkg["f4_sha256"],
            },
            "double_bos_artifact": {
                "artifact_path": good_pkg["double_bos_path"],
                "sha256": good_pkg["double_bos_sha256"],
            },
            "_test_pinned_digest": good_pkg["test_pinned_digest"],
            "_allow_synthetic_for_test_battery": True,
        },
    )
    # Manifest checked under REPO, not evil — battery must not clear via redirect
    _assert_zero_confirmatory(out, calls)
    assert out["integrity_gates"]["confirmatory_blocked"] is True


# --- Defect 3: identity 7 frozen threshold ---


def test_d3_asr_0_9_with_threshold_1_0_fails(tmp_path):
    path, _ = write_fabricated_identity_artifact(
        tmp_path / "id7.json",
        identity7_trigger_asr=0.9,
        identity7_threshold_declaration=1.0,
        all_passed=True,
    )
    payload = json.loads(path.read_text())
    fails = validate_identity_suite_measurements(
        payload,
        f4_epsilons={
            "final_logits": 1e-3,
            "decision_margin": 1e-3,
            "boundary_residual": 1e-3,
            "upstream_activation": 1e-3,
        },
    )
    assert any("threshold" in f or "ASR gate failed" in f for f in fails)


def test_d3_asr_exactly_0_01_passes_criterion(tmp_path):
    path, _ = write_fabricated_identity_artifact(
        tmp_path / "id7.json",
        identity7_trigger_asr=IDENTITY_7_ASR_THRESHOLD,
        identity7_threshold_declaration=IDENTITY_7_ASR_THRESHOLD,
        max_abs_diff=0.0,
        all_passed=True,
    )
    payload = json.loads(path.read_text())
    fails = validate_identity_suite_measurements(
        payload,
        f4_epsilons={
            "final_logits": 1e-3,
            "decision_margin": 1e-3,
            "boundary_residual": 1e-3,
            "upstream_activation": 1e-3,
        },
    )
    assert not any("identity 7" in f for f in fails), fails


# --- Defect 4: bound F4 required ---


def test_d4_invalid_f4_epsilon_1e9_cannot_accept_diff_5(tmp_path):
    prompts = _frozen_dev()
    # Incomplete/invalid F4 with absurd epsilon
    f4_path, f4_sha = write_fabricated_f4_artifact(
        tmp_path / "f4_bad.json", incomplete=True, prompt_ids=prompts
    )
    payload = json.loads(f4_path.read_text())
    for rec in payload["tensor_families"].values():
        rec["epsilon_family"] = 1e9
        rec["d_max_family"] = 1e8
    raw = json.dumps(payload, indent=2, sort_keys=True).encode()
    f4_path.write_bytes(raw)
    import hashlib

    f4_sha = hashlib.sha256(raw).hexdigest()
    id_path, id_sha = write_fabricated_identity_artifact(
        tmp_path / "id.json",
        max_abs_diff=5.0,
        within_tolerance_lie=True,
        all_passed=True,
        f4_calibration_artifact_sha256=f4_sha,
    )
    auth = authenticate_identity_suite_artifact(
        path=id_path,
        claimed_sha256=id_sha,
        f4_artifact_path=f4_path,
        f4_claimed_sha256=f4_sha,
        frozen_prompt_ids=prompts,
        check_production_unlock=False,
        reject_synthetic=False,
    )
    assert auth.ok is False
    joined = " ".join(auth.failures)
    assert "F4" in joined or "epsilon" in joined or "coverage" in joined


def test_d4_mismatched_f4_digest_fails(tmp_path):
    prompts = _frozen_dev()
    f4_path, f4_sha = write_fabricated_f4_artifact(
        tmp_path / "f4.json", incomplete=False, prompt_ids=prompts
    )
    id_path, id_sha = write_fabricated_identity_artifact(
        tmp_path / "id.json",
        f4_calibration_artifact_sha256="0" * 64,
        max_abs_diff=0.0,
    )
    auth = authenticate_identity_suite_artifact(
        path=id_path,
        claimed_sha256=id_sha,
        f4_artifact_path=f4_path,
        f4_claimed_sha256=f4_sha,
        frozen_prompt_ids=prompts,
        check_production_unlock=False,
        reject_synthetic=False,
    )
    assert auth.ok is False
    assert any("f4_calibration_artifact_sha256" in f for f in auth.failures)


def test_d4_missing_f4_artifact_fails(tmp_path):
    path, sha = write_fabricated_identity_artifact(tmp_path / "id.json")
    auth = authenticate_identity_suite_artifact(
        path=path,
        claimed_sha256=sha,
        f4_artifact_path=None,
        check_production_unlock=False,
        reject_synthetic=False,
    )
    assert auth.ok is False
    assert any("bound F4" in f for f in auth.failures)


# --- Defect 5: double-BOS completeness ---


def test_d5_missing_tt_margins_fail(tmp_path):
    prompts = [str(x) for x in _frozen_dev()]
    path, _ = write_fabricated_double_bos_artifact(
        tmp_path / "db.json", prompt_ids=prompts
    )
    payload = json.loads(path.read_text())
    for seed in payload["per_seed_results"].values():
        del seed["intact_double_bos_tt_margins"]
    fails = validate_double_bos_scan_measurements(payload, frozen_prompt_ids=prompts)
    assert any("tt_margins" in f for f in fails)


def test_d5_missing_ranking_fails(tmp_path):
    prompts = [str(x) for x in _frozen_dev()]
    path, _ = write_fabricated_double_bos_artifact(
        tmp_path / "db.json", prompt_ids=prompts
    )
    payload = json.loads(path.read_text())
    for seed in payload["per_seed_results"].values():
        del seed["pair_ranking"]
    fails = validate_double_bos_scan_measurements(payload, frozen_prompt_ids=prompts)
    assert any("pair_ranking" in f for f in fails)


def test_d5_missing_b_i_fails(tmp_path):
    prompts = [str(x) for x in _frozen_dev()]
    path, _ = write_fabricated_double_bos_artifact(
        tmp_path / "db.json", prompt_ids=prompts
    )
    payload = json.loads(path.read_text())
    for seed in payload["per_seed_results"].values():
        del seed["b_i_token_ids_by_prompt"]
    fails = validate_double_bos_scan_measurements(payload, frozen_prompt_ids=prompts)
    assert any("b_i_token_ids" in f for f in fails)


def test_d5_wrong_eligibility_fails(tmp_path):
    prompts = [str(x) for x in _frozen_dev()]
    path, _ = write_fabricated_double_bos_artifact(
        tmp_path / "db.json", prompt_ids=prompts
    )
    payload = json.loads(path.read_text())
    for seed in payload["per_seed_results"].values():
        for prec in seed["pairs"].values():
            for pid in prec["eligible_by_prompt"]:
                prec["eligible_by_prompt"][pid] = False  # wrong vs a!=b_i
    fails = validate_double_bos_scan_measurements(payload, frozen_prompt_ids=prompts)
    assert any("eligibility" in f for f in fails)


def test_d5_clean_fallback_removed_wrong_effects_fail(tmp_path):
    prompts = [str(x) for x in _frozen_dev()]
    path, _ = write_fabricated_double_bos_artifact(
        tmp_path / "db.json", prompt_ids=prompts
    )
    payload = json.loads(path.read_text())
    for seed in payload["per_seed_results"].values():
        # Effects computed vs clean instead of TT
        for prec in seed["pairs"].values():
            for pid in prompts:
                prec["signed_effects_by_prompt"][pid] = (
                    float(seed["intact_double_bos_clean_margins"][pid])
                    - float(prec["patched_margins_by_prompt"][pid])
                )
    fails = validate_double_bos_scan_measurements(payload, frozen_prompt_ids=prompts)
    assert any("m_TT_double" in f for f in fails)


# --- Defect 6: F4 allocation SHA ---


def test_d6_fifty_wrong_prompt_ids_rejected(tmp_path):
    fake = [str(i) for i in range(50)]
    assert len(fake) == 50
    path, _ = write_fabricated_f4_artifact(
        tmp_path / "f4.json",
        incomplete=False,
        prompt_ids=fake,
    )
    payload = json.loads(path.read_text())
    fails = validate_f4_tensor_family_calibration(payload, frozen_prompt_ids=fake)
    assert any("FROZEN_DEV_ID_SEQUENCE_SHA256" in f for f in fails)
    assert FROZEN_DEV_ID_SEQUENCE_SHA256


def test_d6_frozen_allocation_accepted_for_schema(tmp_path):
    prompts = _frozen_dev()
    path, sha = write_fabricated_f4_artifact(
        tmp_path / "f4.json", incomplete=False, prompt_ids=prompts
    )
    auth = authenticate_f4_table_artifact(
        path=path,
        claimed_sha256=sha,
        frozen_prompt_ids=prompts,
        check_production_unlock=False,
        reject_synthetic=False,
    )
    assert auth.ok is True
    assert ARTIFACT_F4_V1
    assert DOUBLE_BOS_PAYLOAD_TOKEN_A == 235285


# --- Test-trust production-gate bypass closure ---


def test_bypass_fabricated_plus_test_pin_cannot_clear_live_gate(tmp_path, monkeypatch):
    """Complete fabricated evidence + _test_pinned_digest must not clear live gate."""
    prompts = _frozen_dev()
    pkg = write_schema_valid_test_battery_package(
        package_dir=tmp_path / "pkg",
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
    )
    # Engineering battery may succeed
    eng = authenticate_engineering_test_battery(
        f4_path=pkg["f4_path"],
        f4_sha256=pkg["f4_sha256"],
        identity_path=pkg["identity_path"],
        identity_sha256=pkg["identity_sha256"],
        double_bos_path=pkg["double_bos_path"],
        double_bos_sha256=pkg["double_bos_sha256"],
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
        test_pinned_digest=pkg["test_pinned_digest"],
    )
    assert eng.test_battery_ok is True
    assert eng.production_unlocked is False

    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "bypass_live",
        integrity_evidence={
            "identity_artifact": {
                "artifact_path": pkg["identity_path"],
                "sha256": pkg["identity_sha256"],
            },
            "f4_artifact": {
                "artifact_path": pkg["f4_path"],
                "sha256": pkg["f4_sha256"],
            },
            "double_bos_artifact": {
                "artifact_path": pkg["double_bos_path"],
                "sha256": pkg["double_bos_sha256"],
            },
            "_test_pinned_digest": pkg["test_pinned_digest"],
            "_allow_synthetic_for_test_battery": True,
        },
    )
    assert out["confirmatory_executed"] is False
    assert out["integrity_gates"]["confirmatory_blocked"] is True
    _assert_zero_confirmatory(out, calls)
    assert scientific_auth_status().authorized is False


def test_bypass_allow_synthetic_flag_no_confirmatory_callbacks(tmp_path, monkeypatch):
    """_allow_synthetic_for_test_battery=True must not enable confirmatory callbacks."""
    prompts = _frozen_dev()
    pkg = write_schema_valid_test_battery_package(
        package_dir=tmp_path / "pkg",
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
    )
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "synth_flag",
        integrity_evidence={
            "identity_artifact": {
                "artifact_path": pkg["identity_path"],
                "sha256": pkg["identity_sha256"],
                "_allow_synthetic_for_test_battery": True,
            },
            "f4_artifact": {
                "artifact_path": pkg["f4_path"],
                "sha256": pkg["f4_sha256"],
                "_allow_synthetic_for_test_battery": True,
            },
            "double_bos_artifact": {
                "artifact_path": pkg["double_bos_path"],
                "sha256": pkg["double_bos_sha256"],
                "_allow_synthetic_for_test_battery": True,
            },
            "_allow_synthetic_for_test_battery": True,
            "_test_pinned_digest": pkg["test_pinned_digest"],
        },
    )
    _assert_zero_confirmatory(out, calls)


def test_bypass_engineering_ok_production_false_explicit(tmp_path):
    """Standalone test-battery may succeed while production authorization remains false."""
    prompts = _frozen_dev()
    pkg = write_schema_valid_test_battery_package(
        package_dir=tmp_path / "pkg",
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
    )
    eng = authenticate_engineering_test_battery(
        f4_path=pkg["f4_path"],
        f4_sha256=pkg["f4_sha256"],
        identity_path=pkg["identity_path"],
        identity_sha256=pkg["identity_sha256"],
        double_bos_path=pkg["double_bos_path"],
        double_bos_sha256=pkg["double_bos_sha256"],
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
        test_pinned_digest=pkg["test_pinned_digest"],
    )
    assert eng.schema_ok is True
    assert eng.test_battery_ok is True
    assert eng.production_unlocked is False
    prod = authenticate_empirical_scientific_battery(
        f4_path=pkg["f4_path"],
        f4_sha256=pkg["f4_sha256"],
        identity_path=pkg["identity_path"],
        identity_sha256=pkg["identity_sha256"],
        double_bos_path=pkg["double_bos_path"],
        double_bos_sha256=pkg["double_bos_sha256"],
        repo_root=tmp_path / "repo",
        frozen_prompt_ids=prompts,
    )
    assert prod.ok is False
    assert prod.production_unlocked is False
    assert PINNED_EMPIRICAL_MANIFEST_SHA256 is None
