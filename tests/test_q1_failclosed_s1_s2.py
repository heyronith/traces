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

    def execute(spec: UnitSpec, _m: Any, _d: Any) -> tuple:
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
    # Forge all_passed=True while results failed
    import json

    payload = json.loads(path.read_text())
    payload["all_passed"] = True
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    import hashlib

    forged_sha = hashlib.sha256(raw).hexdigest()
    auth = authenticate_identity_suite_artifact(path=path, claimed_sha256=forged_sha)
    assert auth.ok is False
    assert any("all_passed=True contradicts" in f or "not all passed" in f for f in auth.failures)


def test_s2_incomplete_double_bos_coverage_rejected(tmp_path):
    path, sha = write_fabricated_double_bos_artifact(
        tmp_path / "db.json", n_pairs=3
    )
    auth = authenticate_double_bos_scan_artifact(path=path, claimed_sha256=sha)
    assert auth.ok is False
    assert any("pair coverage" in f for f in auth.failures)


def test_s2_valid_fabricated_artifacts_authenticate(tmp_path):
    units = list(expected_exploratory_model_units())
    assert units == [
        "exploratory_l1523_seed42",
        "exploratory_l1523_seed43",
        "exploratory_l1523_seed44",
    ]
    id_p, id_s = write_fabricated_identity_artifact(tmp_path / "id.json")
    f4_p, f4_s = write_fabricated_f4_artifact(tmp_path / "f4.json")
    db_p, db_s = write_fabricated_double_bos_artifact(tmp_path / "db.json")
    assert authenticate_identity_suite_artifact(path=id_p, claimed_sha256=id_s).ok
    assert authenticate_f4_table_artifact(path=f4_p, claimed_sha256=f4_s).ok
    assert authenticate_double_bos_scan_artifact(path=db_p, claimed_sha256=db_s).ok
    assert FROZEN_DEV_ID_SEQUENCE_SHA256


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


def test_s2_authenticated_artifacts_allow_confirmatory(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "ok",
        integrity_evidence=_persisted_integrity_evidence(tmp_path),
    )
    assert out["confirmatory_executed"] is True
    assert out["integrity_gates"]["confirmatory_blocked"] is False
    assert _confirmatory().issubset(set(calls["load"]))
