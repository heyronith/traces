"""S1/S2 fail-closed hardening: verified tiny fixtures + authenticated empirical artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch.nn as nn

from traces.q1.empirical_evidence import (
    authenticate_double_bos_scan_artifact,
    authenticate_f4_table_artifact,
    authenticate_identity_suite_artifact,
    expected_exploratory_model_units,
    write_fabricated_double_bos_artifact,
    write_fabricated_f4_artifact,
    write_fabricated_identity_artifact,
)
from traces.q1.engineering_fixtures import (
    mark_verified_engineering_tiny,
    require_verified_engineering_tiny,
)
from traces.q1.frozen_artifacts import (
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    load_ordered_dolly_ids,
)
from traces.q1.generation import run_f3_generation
from traces.q1.integrity_gates import evaluate_preconfirmation_gates
from traces.q1.pipeline_adapters import (
    assemble_bundles_from_condition_records,
    mock_condition_records_for_unit,
)
from traces.q1.scientific_execution import (
    LiveForwardHooks,
    UnitSpec,
    expected_frozen_unit_specs,
    require_scientific_authorization,
    run_live_scientific_pipeline,
    scientific_auth_status,
    stash_mock_observations_on_donors,
)
from traces.q1.scoring import teacher_forced_forward
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


def _persisted_integrity_evidence(tmp_path: Path) -> dict[str, Any]:
    id_path, id_sha = write_fabricated_identity_artifact(tmp_path / "identity.json")
    f4_path, f4_sha = write_fabricated_f4_artifact(tmp_path / "f4.json")
    db_path, db_sha = write_fabricated_double_bos_artifact(tmp_path / "double_bos.json")
    return {
        "identity_artifact": {"artifact_path": str(id_path), "sha256": id_sha},
        "f4_artifact": {"artifact_path": str(f4_path), "sha256": f4_sha},
        "double_bos_artifact": {"artifact_path": str(db_path), "sha256": db_sha},
    }


def _spy_hooks():
    calls: dict[str, list[str]] = {"load": [], "execute": []}

    def load(spec: UnitSpec) -> dict[str, Any]:
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

    def donors(spec: UnitSpec, _m: Any) -> dict[str, Any]:
        ids = (
            [str(x) for x in _frozen_dev()]
            if spec.role == "exploratory_dev"
            else [f"p{i}" for i in range(8)]
        )
        return {
            "n_donor_caches": len(ids),
            "clean_references_frozen": True,
            "instruction_ids": ids,
        }

    def execute(spec: UnitSpec, _m: Any, donors_ev: Any) -> tuple:
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
        stash_mock_observations_on_donors(
            donors_ev, spec=spec, records=recs, prompt_ids=ids
        )
        return b, i

    hooks = LiveForwardHooks(
        load_model_for_unit=load,
        capture_donors_for_unit=donors,
        execute_unit=execute,
    )
    return hooks, calls


def _confirmatory() -> set[str]:
    return {
        s.model_unit
        for s in expected_frozen_unit_specs()
        if s.role in {"primary_confirmatory", "secondary_replication"}
    }


# --- S1 ---


def test_s1_ordinary_mock_cannot_use_engineering_tiny_boolean(monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    monkeypatch.setenv("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", "1")
    from traces.q1.causal_runner import Q1CausalRunner

    class OrdinaryMock(nn.Module):
        def forward(self, **kwargs):
            raise RuntimeError("must not forward")

    with pytest.raises(PermissionError, match="verified engineering tiny"):
        Q1CausalRunner(
            OrdinaryMock(),
            tokenizer=None,
            model_unit="adversarial/ordinary",
            engineering_tiny=True,
        )


def test_s1_cannot_mark_ordinary_model_as_tiny():
    class Ordinary(nn.Module):
        pass

    with pytest.raises(PermissionError, match="Refusing to mark"):
        mark_verified_engineering_tiny(Ordinary())


def test_s1_scoring_rejects_engineering_tiny_without_verified_fixture(monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    monkeypatch.setenv("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", "1")

    class Ordinary(nn.Module):
        def forward(self, **kwargs):
            raise RuntimeError("no")

    with pytest.raises(PermissionError):
        teacher_forced_forward(
            Ordinary(),
            __import__("torch").ones(1, 2, dtype=__import__("torch").long),
            forward_authorization="engineering_tiny",
        )


def test_s1_generation_has_no_auth_bypass(monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    monkeypatch.setenv("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", "1")
    with pytest.raises(PermissionError):
        run_f3_generation(
            model=object(),
            tokenizer=object(),
            rendered_prompt="x",
            full_payload="i hate you",
        )
    # Tiny flag must not authorize scientific status
    st = scientific_auth_status()
    assert st.authorized is False
    assert st.tiny_does_not_authorize_scientific is True
    with pytest.raises(PermissionError):
        require_scientific_authorization()


def test_s1_scoring_default_is_scientific_not_bypass(monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    import torch

    class Ordinary(nn.Module):
        def forward(self, **kwargs):
            raise RuntimeError("no")

    with pytest.raises(PermissionError):
        teacher_forced_forward(Ordinary(), torch.ones(1, 2, dtype=torch.long))


# --- S2 ---


def test_s2_inline_evidence_kind_rejected():
    bad = evaluate_preconfirmation_gates(
        development_n_eligible=48,
        development_n_frozen=50,
        exploratory_first_token_validity=0.95,
        identity_suite={
            "evidence_kind": "empirical_scientific",
            "all_passed": True,
        },
        f4_table={
            "locked": True,
            "evidence_kind": "empirical_scientific",
            "sha256": "a" * 64,
        },
        double_bos_evidence={
            "evidence_kind": "empirical_scientific",
            "result_sha256": "b" * 64,
            "n_pairs_recorded": 64,
        },
        require_empirical_scientific=True,
    )
    assert bad.progression_blocked is True
    assert bad.gates["identities_1_to_9"].status == "blocked"
    assert bad.gates["frozen_f4_tolerance_table"].status == "blocked"
    assert bad.gates["descriptive_double_bos_64_pair_scan"].status == "blocked"


def test_s2_mismatched_hash_and_missing_file_blocked(tmp_path):
    path, sha = write_fabricated_identity_artifact(tmp_path / "id.json")
    bad_hash = authenticate_identity_suite_artifact(
        path=path, claimed_sha256="0" * 64
    )
    assert bad_hash.ok is False
    assert any("mismatch" in f for f in bad_hash.failures)

    missing = authenticate_identity_suite_artifact(
        path=tmp_path / "nope.json", claimed_sha256=sha
    )
    assert missing.ok is False
    assert any("missing" in f for f in missing.failures)


def test_s2_wrong_model_units_and_dataset_rejected(tmp_path):
    path, sha = write_fabricated_identity_artifact(
        tmp_path / "id.json",
        model_units=["wrong_unit_seed99"],
    )
    auth = authenticate_identity_suite_artifact(path=path, claimed_sha256=sha)
    assert auth.ok is False
    assert any("model_units" in f for f in auth.failures)

    path2, sha2 = write_fabricated_f4_artifact(
        tmp_path / "f4.json",
        dataset_sha="f" * 64,
    )
    auth2 = authenticate_f4_table_artifact(path=path2, claimed_sha256=sha2)
    assert auth2.ok is False
    assert any("dataset_binding" in f for f in auth2.failures)


def test_s2_all_passed_true_with_failed_identity_rejected(tmp_path):
    path, sha = write_fabricated_identity_artifact(
        tmp_path / "id.json", all_passed=False
    )
    import hashlib
    import json

    payload = json.loads(path.read_text())
    payload["all_passed"] = True
    # Also flip claims to try to look real — still must fail substantive + provenance
    payload["claims_real_experiment"] = True
    payload.pop("fixture_purpose", None)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    forged_sha = hashlib.sha256(raw).hexdigest()
    auth = authenticate_identity_suite_artifact(path=path, claimed_sha256=forged_sha)
    assert auth.ok is False
    assert auth.production_blocked is True


def test_s2_incomplete_double_bos_coverage_rejected(tmp_path):
    path, sha = write_fabricated_double_bos_artifact(
        tmp_path / "db.json", n_pairs=3
    )
    auth = authenticate_double_bos_scan_artifact(path=path, claimed_sha256=sha)
    assert auth.ok is False
    joined = " ".join(auth.failures)
    assert "distinct" in joined or "prescribed" in joined or "claims_real" in joined


def test_s2_synthetic_fixtures_rejected_for_production(tmp_path):
    """Fabricated fixtures must never authenticate as production empirical evidence."""
    units = list(expected_exploratory_model_units())
    assert units == [
        "exploratory_l1523_seed42",
        "exploratory_l1523_seed43",
        "exploratory_l1523_seed44",
    ]
    id_p, id_s = write_fabricated_identity_artifact(tmp_path / "id.json")
    f4_p, f4_s = write_fabricated_f4_artifact(tmp_path / "f4.json")
    db_p, db_s = write_fabricated_double_bos_artifact(tmp_path / "db.json")
    for auth in (
        authenticate_identity_suite_artifact(path=id_p, claimed_sha256=id_s),
        authenticate_f4_table_artifact(path=f4_p, claimed_sha256=f4_s),
        authenticate_double_bos_scan_artifact(path=db_p, claimed_sha256=db_s),
    ):
        assert auth.ok is False
        assert auth.production_blocked is True
        assert any(
            "claims_real_experiment" in f or "synthetic" in f or "provenance" in f
            for f in auth.failures
        )
    assert FROZEN_DEV_ID_SEQUENCE_SHA256


def test_s2_caller_expected_units_cannot_override_frozen_constants(tmp_path):
    path, sha = write_fabricated_identity_artifact(
        tmp_path / "id.json",
        model_units=["adversarial_override_seed1"],
    )
    # Even if caller passes "matching" expected_model_units, frozen constants win
    auth = authenticate_identity_suite_artifact(
        path=path,
        claimed_sha256=sha,
        expected_model_units=["adversarial_override_seed1"],
    )
    assert auth.ok is False
    assert any("frozen exploratory" in f or "model_units" in f for f in auth.failures)


def test_s2_spy_zero_confirmatory_on_fabricated_metadata(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "meta",
        integrity_evidence={
            "identity_suite": {
                "evidence_kind": "empirical_scientific",
                "all_passed": True,
            },
            "f4_table": {
                "locked": True,
                "evidence_kind": "empirical_scientific",
                "sha256": "c" * 64,
            },
            "double_bos_evidence": {
                "evidence_kind": "empirical_scientific",
                "result_sha256": "d" * 64,
                "n_pairs_recorded": 64,
            },
        },
    )
    assert out["confirmatory_executed"] is False
    assert not _confirmatory().intersection(calls["load"])
    assert not _confirmatory().intersection(calls["execute"])


def test_s2_spy_zero_confirmatory_on_hash_mismatch(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    ev = _persisted_integrity_evidence(tmp_path)
    ev["identity_artifact"]["sha256"] = "0" * 64
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "badhash",
        integrity_evidence=ev,
    )
    assert out["confirmatory_executed"] is False
    assert not _confirmatory().intersection(calls["load"])
    assert not _confirmatory().intersection(calls["execute"])


def test_s2_fabricated_fixtures_never_clear_live_confirmatory(tmp_path, monkeypatch):
    """Regression: prior clear-path used fabricated artifacts — must stay BLOCKED."""
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "synth_block",
        integrity_evidence=_persisted_integrity_evidence(tmp_path),
    )
    assert out["confirmatory_executed"] is False
    assert out["integrity_gates"]["confirmatory_blocked"] is True
    assert not _confirmatory().intersection(calls["load"])
    assert not _confirmatory().intersection(calls["execute"])
    assert any(
        "provenance" in r or "claims_real" in r or "synthetic" in r
        for r in out["integrity_gates"]["blocked_reasons"]
    )


def test_s2_claims_real_without_provenance_still_blocked(tmp_path):
    import hashlib
    import json

    path, _sha = write_fabricated_identity_artifact(tmp_path / "id.json")
    payload = json.loads(path.read_text())
    payload["claims_real_experiment"] = True
    payload.pop("fixture_purpose", None)
    # Strip synthetic markers from nested details
    for rows in payload.get("per_model_results", {}).values():
        for row in rows:
            row["detail"] = "attempted_real"
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    auth = authenticate_identity_suite_artifact(path=path, claimed_sha256=sha)
    assert auth.ok is False
    assert any("provenance" in f for f in auth.failures)


# --- S2 Evidence Contract Remediation (S2-1…S2-4) ---


def test_s21_artifact_must_not_embed_manifest_hash(tmp_path):
    import hashlib
    import json

    path, _ = write_fabricated_identity_artifact(tmp_path / "id.json")
    payload = json.loads(path.read_text())
    payload["provenance_manifest_sha256"] = "a" * 64
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    auth = authenticate_identity_suite_artifact(path=path, claimed_sha256=sha)
    assert auth.ok is False
    assert any("circular" in f for f in auth.failures)


def test_s21_approval_text_alone_never_authorizes(tmp_path):
    from traces.q1.empirical_trust import (
        ARTIFACT_PROVENANCE_MANIFEST_V1,
        PRODUCTION_MANIFEST_RELPATH,
        load_production_manifest,
    )

    root = tmp_path / "repo"
    man = root / PRODUCTION_MANIFEST_RELPATH
    man.parent.mkdir(parents=True)
    man.write_text(
        json_dumps(
            {
                "artifact_type": ARTIFACT_PROVENANCE_MANIFEST_V1,
                "approval_status": "OWNER_APPROVED_PLEASE_TRUST",
                "independent_approval_record": "forged approval prose is not a signature",
                "artifact_digests": {
                    "q1_empirical_identity_suite_v1": "1" * 64,
                    "q1_empirical_f4_tolerance_table_v1": "2" * 64,
                    "q1_empirical_double_bos_64_pair_scan_v1": "3" * 64,
                },
            }
        )
    )
    auth = load_production_manifest(
        repo_root=root,
        artifact_digests={
            "q1_empirical_identity_suite_v1": "1" * 64,
            "q1_empirical_f4_tolerance_table_v1": "2" * 64,
            "q1_empirical_double_bos_64_pair_scan_v1": "3" * 64,
        },
    )
    assert auth.ok is False
    assert auth.production_unlocked is False


def test_s21_crypto_test_trust_never_unlocks_production(tmp_path, monkeypatch):
    from traces.q1.empirical_trust import (
        ARTIFACT_PROVENANCE_MANIFEST_V1,
        TEST_TRUST_ENV,
        canonical_manifest_bytes,
        make_test_trust_material,
        sign_manifest_hmac,
        verify_manifest_cryptography,
        load_production_manifest,
        PRODUCTION_MANIFEST_RELPATH,
    )

    key, _ = make_test_trust_material()
    payload = {
        "artifact_type": ARTIFACT_PROVENANCE_MANIFEST_V1,
        "artifact_digests": {
            "q1_empirical_identity_suite_v1": "a" * 64,
            "q1_empirical_f4_tolerance_table_v1": "b" * 64,
            "q1_empirical_double_bos_64_pair_scan_v1": "c" * 64,
        },
        "self_generated": False,
    }
    canon = canonical_manifest_bytes(payload)
    sig = sign_manifest_hmac(canon, key)
    monkeypatch.setenv(TEST_TRUST_ENV, "1")
    crypto = verify_manifest_cryptography(
        manifest_bytes=canon,
        signature_hex=sig,
        trusted_key=key,
        pinned_digest=None,
        allow_test_trust=True,
    )
    assert crypto.ok is True
    assert crypto.mode == "test_hmac"

    # Valid self-generated HMAC must not unlock production (HMAC path removed)
    root = tmp_path / "repo"
    (root / PRODUCTION_MANIFEST_RELPATH).parent.mkdir(parents=True)
    (root / PRODUCTION_MANIFEST_RELPATH).write_text(json_dumps(payload))
    prod = load_production_manifest(
        repo_root=root,
        artifact_digests=payload["artifact_digests"],
    )
    assert prod.production_unlocked is False
    assert prod.ok is False


def test_s21_negative_forged_untrusted_modified_substituted(tmp_path, monkeypatch):
    from traces.q1.empirical_trust import (
        ARTIFACT_PROVENANCE_MANIFEST_V1,
        canonical_manifest_bytes,
        make_test_trust_material,
        sign_manifest_hmac,
        verify_manifest_cryptography,
        load_production_manifest,
        PRODUCTION_MANIFEST_RELPATH,
        PINNED_EMPIRICAL_MANIFEST_SHA256,
    )

    assert PINNED_EMPIRICAL_MANIFEST_SHA256 is None
    monkeypatch.setenv("TRACES_Q1_EMPIRICAL_TEST_TRUST", "1")
    key, _ = make_test_trust_material()
    payload = {
        "artifact_type": ARTIFACT_PROVENANCE_MANIFEST_V1,
        "artifact_digests": {
            "q1_empirical_identity_suite_v1": "a" * 64,
            "q1_empirical_f4_tolerance_table_v1": "b" * 64,
            "q1_empirical_double_bos_64_pair_scan_v1": "c" * 64,
        },
    }
    canon = canonical_manifest_bytes(payload)
    good_sig = sign_manifest_hmac(canon, key)

    # Forged approval / bad signature (unit crypto only)
    forged = verify_manifest_cryptography(
        manifest_bytes=canon,
        signature_hex="deadbeef" * 8,
        trusted_key=key,
        pinned_digest=None,
        allow_test_trust=True,
    )
    assert forged.ok is False

    # Untrusted signing key
    untrusted = verify_manifest_cryptography(
        manifest_bytes=canon,
        signature_hex=good_sig,
        trusted_key=b"NOT_THE_TRUSTED_KEY______________",
        pinned_digest=None,
        allow_test_trust=True,
    )
    assert untrusted.ok is False

    # Wrong pin + valid HMAC must fail immediately (no HMAC bypass)
    pin_mismatch = verify_manifest_cryptography(
        manifest_bytes=canon,
        signature_hex=good_sig,
        trusted_key=key,
        pinned_digest="0" * 64,
        allow_test_trust=True,
    )
    assert pin_mismatch.ok is False
    assert any("pinned" in f for f in pin_mismatch.failures)

    # Modified artifact digest vs manifest
    root = tmp_path / "repo"
    (root / PRODUCTION_MANIFEST_RELPATH).parent.mkdir(parents=True)
    (root / PRODUCTION_MANIFEST_RELPATH).write_text(json_dumps(payload))
    modified = load_production_manifest(
        repo_root=root,
        artifact_digests={
            "q1_empirical_identity_suite_v1": "f" * 64,  # changed
            "q1_empirical_f4_tolerance_table_v1": "b" * 64,
            "q1_empirical_double_bos_64_pair_scan_v1": "c" * 64,
        },
    )
    assert modified.ok is False
    assert any("does not match" in f for f in modified.failures)

    # Changed manifest body under test HMAC
    changed = dict(payload)
    changed["nonce"] = "tamper"
    bad = verify_manifest_cryptography(
        manifest_bytes=canonical_manifest_bytes(changed),
        signature_hex=good_sig,
        trusted_key=key,
        pinned_digest=None,
        allow_test_trust=True,
    )
    assert bad.ok is False

    # Substituted manifest path refused (only fixed path is read)
    alt = root / "evil" / "MANIFEST.json"
    alt.parent.mkdir(parents=True)
    alt.write_text(json_dumps(payload))
    (root / PRODUCTION_MANIFEST_RELPATH).unlink()
    missing_fixed = load_production_manifest(
        repo_root=root,
        artifact_digests=payload["artifact_digests"],
    )
    assert missing_fixed.ok is False
    assert any("fixed path" in f for f in missing_fixed.failures)

    # Self-generated rejected for production
    (root / PRODUCTION_MANIFEST_RELPATH).write_text(
        json_dumps({**payload, "self_generated": True})
    )
    self_gen = load_production_manifest(
        repo_root=root,
        artifact_digests=payload["artifact_digests"],
    )
    assert self_gen.ok is False
    assert any(
        "self_generated" in f or "PINNED" in f or "unset" in f for f in self_gen.failures
    )


def json_dumps(obj):
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


def test_s22_f4_rejects_model_family_keys_and_bad_epsilon(tmp_path):
    import json

    from traces.q1.empirical_evidence import (
        REQUIRED_F4_TENSOR_FAMILIES,
        validate_f4_tensor_family_calibration,
    )

    prompts = [str(i) for i in range(50)]
    # Legacy model-family keyed table
    legacy = {
        "formula": "epsilon_family = max(1e-3, 10 * D_max_family)",
        "locked": True,
        "families": {"l1523": {}, "l19": {}},
        "tensor_families": {},
    }
    fails = validate_f4_tensor_family_calibration(legacy, frozen_prompt_ids=prompts)
    assert any("tensor family" in f or "missing required" in f for f in fails)

    path, sha = write_fabricated_f4_artifact(tmp_path / "f4.json", incomplete=True)
    payload = json.loads(path.read_text())
    # Wrong epsilon calculation
    fam = next(iter(payload["tensor_families"].values()))
    fam["epsilon_family"] = 999.0
    fails2 = validate_f4_tensor_family_calibration(payload, frozen_prompt_ids=prompts)
    assert any("epsilon_family" in f for f in fails2)
    assert set(REQUIRED_F4_TENSOR_FAMILIES)


def test_s22_f4_complete_coverage_schema_ok_but_production_blocked(tmp_path):
    prompts = _frozen_dev()
    path, sha = write_fabricated_f4_artifact(
        tmp_path / "f4_full.json",
        incomplete=False,
        prompt_ids=prompts,
    )
    auth = authenticate_f4_table_artifact(
        path=path,
        claimed_sha256=sha,
        frozen_prompt_ids=prompts,
        repo_root=REPO,
    )
    # Schema may pass numeric coverage; synthetic + missing pin still block
    assert auth.ok is False
    assert auth.production_blocked is True
    joined = " ".join(auth.failures)
    assert "synthetic" in joined or "claims_real" in joined or "pinned" in joined or "unlock" in joined


def test_s23_lying_within_tolerance_boolean_rejected(tmp_path):
    from traces.q1.empirical_evidence import validate_identity_suite_measurements

    path, _ = write_fabricated_identity_artifact(
        tmp_path / "id.json",
        max_abs_diff=1e9,
        within_tolerance_lie=True,
        all_passed=True,
    )
    import json

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
    assert fails
    assert any("within_tolerance=true contradicts" in f or "1e+09" in f or "1e9" in f or "1000000000" in f for f in fails)


_F4_EPS = {
    "final_logits": 1e-3,
    "decision_margin": 1e-3,
    "boundary_residual": 1e-3,
    "upstream_activation": 1e-3,
}

_ID236_MISSING_CASES = [
    pytest.param(2, "final_logits", id="id2-drop-final_logits"),
    pytest.param(2, "boundary_residual", id="id2-drop-boundary_residual"),
    pytest.param(3, "final_logits", id="id3-drop-final_logits"),
    pytest.param(3, "boundary_residual", id="id3-drop-boundary_residual"),
    pytest.param(6, "upstream_activation", id="id6-drop-upstream_activation"),
    pytest.param(6, "boundary_residual", id="id6-drop-boundary_residual"),
]


def _mutate_identity_drop_family(
    path: Path,
    *,
    identity_id: int,
    drop_family: str,
) -> tuple[Path, str, dict[str, Any]]:
    """Drop exactly one required family comparison; keep other; declare pass=True."""
    import hashlib
    import json

    payload = json.loads(path.read_text())
    for rows in payload["per_model_results"].values():
        for row in rows:
            if int(row["identity_id"]) != identity_id:
                continue
            kept = [
                c
                for c in row["numerical_comparisons"]
                if str(c.get("tensor_family")) != drop_family
            ]
            assert kept, f"expected remaining comparison after dropping {drop_family}"
            assert all(
                str(c.get("tensor_family")) != drop_family for c in kept
            )
            for c in kept:
                c["within_tolerance"] = True
                c["max_abs_diff"] = 0.0
            row["numerical_comparisons"] = kept
            row["passed"] = True
    payload["all_passed"] = True
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), payload


@pytest.mark.parametrize("identity_id,drop_family", _ID236_MISSING_CASES)
def test_s23_identity_2_3_6_require_multi_family_comparisons(
    tmp_path, monkeypatch, identity_id, drop_family
):
    """v1.3: IDs 2/3/6 — removing any one required family fails schema, auth, and live gate."""
    from traces.q1.empirical_evidence import (
        IDENTITY_TENSOR_FAMILIES,
        validate_identity_suite_measurements,
    )

    assert drop_family in IDENTITY_TENSOR_FAMILIES[identity_id]

    base, _ = write_fabricated_identity_artifact(
        tmp_path / f"id_{identity_id}_{drop_family}.json",
        max_abs_diff=0.0,
        all_passed=True,
    )
    path, sha, payload = _mutate_identity_drop_family(
        base, identity_id=identity_id, drop_family=drop_family
    )

    # 1) Standalone numerical schema validator rejects
    fails = validate_identity_suite_measurements(payload, f4_epsilons=_F4_EPS)
    assert any(
        f"identity {identity_id} missing required tensor-family" in f for f in fails
    ), fails

    # 2) Authenticated production path cannot accept
    auth = authenticate_identity_suite_artifact(
        path=path,
        claimed_sha256=sha,
        frozen_prompt_ids=_frozen_dev(),
        repo_root=REPO,
    )
    assert auth.ok is False
    assert auth.production_blocked is True

    # 3) Live pre-confirmation gate: zero confirmatory load/execute callbacks
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    f4_path, f4_sha = write_fabricated_f4_artifact(tmp_path / "f4.json")
    db_path, db_sha = write_fabricated_double_bos_artifact(tmp_path / "db.json")
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / f"live_{identity_id}_{drop_family}",
        integrity_evidence={
            "identity_artifact": {"artifact_path": str(path), "sha256": sha},
            "f4_artifact": {"artifact_path": str(f4_path), "sha256": f4_sha},
            "double_bos_artifact": {"artifact_path": str(db_path), "sha256": db_sha},
        },
    )
    assert out["confirmatory_executed"] is False
    assert not _confirmatory().intersection(calls["load"])
    assert not _confirmatory().intersection(calls["execute"])

    gate = evaluate_preconfirmation_gates(
        development_n_eligible=48,
        development_n_frozen=50,
        exploratory_first_token_validity=0.96,
        identity_suite={"artifact_path": str(path), "sha256": sha},
        f4_table={"artifact_path": str(f4_path), "sha256": f4_sha},
        double_bos_evidence={"artifact_path": str(db_path), "sha256": db_sha},
        require_empirical_scientific=True,
        repo_root=str(REPO),
        frozen_prompt_ids=_frozen_dev(),
    )
    assert gate.progression_blocked is True
    assert gate.gates["identities_1_to_9"].status != "complete"


def test_s23_complete_synthetic_identity_passes_schema_not_production(tmp_path):
    """Internally consistent synthetic identity: schema OK, production still blocked."""
    from traces.q1.empirical_evidence import validate_identity_suite_measurements

    path, sha = write_fabricated_identity_artifact(
        tmp_path / "id_complete.json",
        max_abs_diff=0.0,
        all_passed=True,
    )
    import json

    payload = json.loads(path.read_text())
    fails = validate_identity_suite_measurements(payload, f4_epsilons=_F4_EPS)
    assert fails == [], fails

    auth = authenticate_identity_suite_artifact(
        path=path,
        claimed_sha256=sha,
        frozen_prompt_ids=_frozen_dev(),
        repo_root=REPO,
    )
    assert auth.ok is False
    assert auth.production_blocked is True
    assert any(
        "synthetic" in f or "claims_real" in f or "provenance" in f or "unlock" in f
        for f in auth.failures
    )


def test_s23_max_abs_diff_equals_epsilon_passes_schema():
    from traces.q1.empirical_evidence import validate_identity_suite_measurements
    import json
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as td:
        path, _ = write_fabricated_identity_artifact(
            P(td) / "id.json",
            max_abs_diff=1e-3,  # exactly == locked epsilon
            all_passed=True,
        )
        payload = json.loads(path.read_text())
        fails = validate_identity_suite_measurements(payload, f4_epsilons=_F4_EPS)
        assert fails == [], fails


def test_s23_max_abs_diff_above_epsilon_fails_despite_booleans(tmp_path):
    import hashlib
    import json

    from traces.q1.empirical_evidence import validate_identity_suite_measurements

    path, _ = write_fabricated_identity_artifact(
        tmp_path / "id.json",
        max_abs_diff=1.1e-3,
        within_tolerance_lie=True,
        all_passed=True,
    )
    payload = json.loads(path.read_text())
    # Force declared booleans true while measurement exceeds epsilon
    for rows in payload["per_model_results"].values():
        for row in rows:
            row["passed"] = True
            for c in row.get("numerical_comparisons") or []:
                c["within_tolerance"] = True
                c["max_abs_diff"] = 1.1e-3
    fails = validate_identity_suite_measurements(payload, f4_epsilons=_F4_EPS)
    assert fails
    assert any(
        "within_tolerance=true contradicts" in f or "fails max_abs_diff" in f
        for f in fails
    )
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    auth = authenticate_identity_suite_artifact(
        path=path, claimed_sha256=sha, repo_root=REPO, frozen_prompt_ids=_frozen_dev()
    )
    assert auth.ok is False


def test_s23_identity_auth_rejects_max_abs_diff_1e9_lie(tmp_path):
    prompts = _frozen_dev()
    f4_path, f4_sha = write_fabricated_f4_artifact(
        tmp_path / "f4.json", incomplete=False, prompt_ids=prompts
    )
    path, sha = write_fabricated_identity_artifact(
        tmp_path / "id.json",
        max_abs_diff=1e9,
        within_tolerance_lie=True,
        all_passed=True,
        f4_calibration_artifact_sha256=f4_sha,
    )
    auth = authenticate_identity_suite_artifact(
        path=path,
        claimed_sha256=sha,
        f4_artifact_path=f4_path,
        f4_claimed_sha256=f4_sha,
        frozen_prompt_ids=prompts,
        check_production_unlock=False,
        reject_synthetic=False,
    )
    assert auth.ok is False
    joined = " ".join(auth.failures)
    assert (
        "within_tolerance" in joined
        or "max_abs_diff" in joined
        or "1000000000" in joined
        or "1e+09" in joined
        or "1e9" in joined
    )


def test_s23_duplicate_identity_id_rejected(tmp_path):
    import json

    from traces.q1.empirical_evidence import validate_identity_suite_measurements

    path, _ = write_fabricated_identity_artifact(tmp_path / "id.json")
    payload = json.loads(path.read_text())
    for rows in payload["per_model_results"].values():
        rows.append(dict(rows[0]))  # duplicate identity_id
        break
    fails = validate_identity_suite_measurements(payload, f4_epsilons=_F4_EPS)
    assert any("duplicate identity_id" in f for f in fails)


def test_s24_alters_single_bos_stopping_rule_rejected(tmp_path):
    import hashlib
    import json

    path, _ = write_fabricated_double_bos_artifact(
        tmp_path / "db.json", prompt_ids=_frozen_dev()
    )
    payload = json.loads(path.read_text())
    payload["alters_single_bos_stopping_rule"] = True
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    auth = authenticate_double_bos_scan_artifact(
        path=path, claimed_sha256=sha, frozen_prompt_ids=_frozen_dev()
    )
    assert auth.ok is False
    assert any("stopping rule" in f for f in auth.failures)


def test_s24_pair_key_only_artifact_rejected(tmp_path):
    import hashlib
    import json

    from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS
    from traces.q1.empirical_evidence import ARTIFACT_DOUBLE_BOS_V1

    units = list(expected_exploratory_model_units())
    prompts = [str(x) for x in _frozen_dev()]
    per_seed = {}
    for u in units:
        per_seed[u] = {
            "prompt_ids": prompts,
            # Intentionally omit intact_double_bos_clean_margins and prompt effects
            "pairs": {f"{a},{b}": {"ok": True} for a, b in sorted(EXPECTED_L15_L16_PAIRS)},
        }
    payload = {
        "artifact_type": ARTIFACT_DOUBLE_BOS_V1,
        "claims_real_experiment": False,
        "fixture_purpose": "synthetic_regression_only",
        "model_units": units,
        "dataset_binding": {
            "split": "development",
            "id_sequence_sha256": FROZEN_DEV_ID_SEQUENCE_SHA256,
        },
        "bos_convention": "double",
        "alters_single_bos_stopping_rule": False,
        "clean_reference_provenance": {
            "frozen": True,
            "bos_convention": "double",
        },
        "per_seed_results": per_seed,
        "n_pairs_recorded": 64,
    }
    path = tmp_path / "db_keys_only.json"
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    auth = authenticate_double_bos_scan_artifact(
        path=path, claimed_sha256=sha, frozen_prompt_ids=prompts
    )
    assert auth.ok is False
    joined = " ".join(auth.failures)
    assert (
        "prompt-level" in joined
        or "intact_double_bos" in joined
        or "patched margins" in joined
    )


def test_s24_single_bos_clean_reference_rejected(tmp_path):
    import hashlib
    import json

    path, _ = write_fabricated_double_bos_artifact(
        tmp_path / "db.json", prompt_ids=_frozen_dev()
    )
    payload = json.loads(path.read_text())
    payload["clean_reference_provenance"]["bos_convention"] = "single"
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    auth = authenticate_double_bos_scan_artifact(
        path=path, claimed_sha256=sha, frozen_prompt_ids=_frozen_dev()
    )
    assert auth.ok is False
    assert any("double-BOS" in f or "single-BOS" in f for f in auth.failures)


def test_s2_spy_zero_confirmatory_on_numerical_identity_lie(tmp_path, monkeypatch):
    """E2E: max_abs_diff=1e9 + within_tolerance=true must yield zero confirmatory callbacks."""
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    id_path, id_sha = write_fabricated_identity_artifact(
        tmp_path / "id.json",
        max_abs_diff=1e9,
        within_tolerance_lie=True,
        all_passed=True,
    )
    f4_path, f4_sha = write_fabricated_f4_artifact(tmp_path / "f4.json")
    db_path, db_sha = write_fabricated_double_bos_artifact(tmp_path / "db.json")
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "lie_gate",
        integrity_evidence={
            "identity_artifact": {"artifact_path": str(id_path), "sha256": id_sha},
            "f4_artifact": {"artifact_path": str(f4_path), "sha256": f4_sha},
            "double_bos_artifact": {"artifact_path": str(db_path), "sha256": db_sha},
        },
    )
    assert out["confirmatory_executed"] is False
    assert not _confirmatory().intersection(calls["load"])
    assert not _confirmatory().intersection(calls["execute"])


def test_s2_gate_independent_and_live_both_block_unauthenticated(tmp_path):
    """Each validator independently + live pre-confirmation gate both fail closed."""
    id_p, id_s = write_fabricated_identity_artifact(tmp_path / "id.json")
    f4_p, f4_s = write_fabricated_f4_artifact(tmp_path / "f4.json")
    db_p, db_s = write_fabricated_double_bos_artifact(tmp_path / "db.json")
    prompts = _frozen_dev()
    assert authenticate_identity_suite_artifact(
        path=id_p, claimed_sha256=id_s, frozen_prompt_ids=prompts, repo_root=REPO
    ).ok is False
    assert authenticate_f4_table_artifact(
        path=f4_p, claimed_sha256=f4_s, frozen_prompt_ids=prompts, repo_root=REPO
    ).ok is False
    assert authenticate_double_bos_scan_artifact(
        path=db_p, claimed_sha256=db_s, frozen_prompt_ids=prompts, repo_root=REPO
    ).ok is False

    report = evaluate_preconfirmation_gates(
        development_n_eligible=48,
        development_n_frozen=50,
        exploratory_first_token_validity=0.96,
        identity_suite={"artifact_path": str(id_p), "sha256": id_s},
        f4_table={"artifact_path": str(f4_p), "sha256": f4_s},
        double_bos_evidence={"artifact_path": str(db_p), "sha256": db_s},
        require_empirical_scientific=True,
        repo_root=str(REPO),
        frozen_prompt_ids=prompts,
    )
    assert report.progression_blocked is True
    assert report.gates["identities_1_to_9"].status != "complete"
    assert report.gates["frozen_f4_tolerance_table"].status != "complete"
    assert report.gates["descriptive_double_bos_64_pair_scan"].status != "complete"
