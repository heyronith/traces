"""Agent 1 audit remediations: pairwise identity, prompt-level D/K, pins, freeze, gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traces.pilot.analysis import analyze_per_pair_results
from traces.pilot.checkpoint_store import PilotCheckpointStore
from traces.pilot.execution import run_two_phase_pilot, scientific_measurements
from traces.pilot.gates import evaluate_native_validation_gate
from traces.pilot.pins import (
    PINNED_ADAPTER_REVISION,
    PINNED_BASE_REVISION,
    assert_loaded_matches_pins,
    resolve_executed_git_sha,
)
from traces.pilot.synthetic import SyntheticPilotScorer, make_shared_causal_world


def test_pairwise_summaries_six_pairs_and_both_donors(tmp_path):
    world = make_shared_causal_world(10)
    result = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43, 44),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    summaries = result.analysis["per_pair_summaries"]
    # 6 directed pairs × 2 k × 2 splits = 24 when gates pass
    assert len(summaries) == 24
    # Both donors into same recipient appear as distinct comparison keys
    recip42 = [
        s
        for s in summaries
        if s["comparison_recipient_seed"] == 42 and s["split"] == "held_out" and s["k"] == 10
    ]
    donors = {s["comparison_donor_seed"] for s in recip42}
    assert donors == {43, 44}
    for s in recip42:
        assert s["vs_native"]["D_transfer_minus_reference"] is not None
        assert s["vs_random"]["D_transfer_minus_reference"] is not None
        if "vs_damage_matched" in s:
            assert s["vs_damage_matched"]["D_transfer_minus_reference"] is not None
        else:
            assert s["damage_match_unmatched"]
        assert s["D_transfer"] is not None and s["D_random"] is not None
        assert float(s["D_transfer"]) >= float(s["D_random"]) - 1e-9


def test_random_and_native_use_comparison_pair_not_donor_seed(tmp_path):
    world = make_shared_causal_world(8)
    result = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    randoms = [
        r
        for r in result.per_condition
        if r["condition"]["source"] == "random" and r["split"] == "validation"
    ]
    natives = [
        r
        for r in result.per_condition
        if r["condition"]["source"] == "recipient_native_topk"
        and r["split"] == "validation"
    ]
    assert randoms and natives
    for r in randoms:
        assert r["condition"]["donor_seed"] is None
        assert r["condition"]["comparison_donor_seed"] in (42, 43)
        assert r["condition"]["comparison_pair_key"].startswith("donor")
    for r in natives:
        assert r["condition"]["donor_seed"] == r["condition"]["recipient_seed"]
        assert r["condition"]["comparison_donor_seed"] != r["condition"]["recipient_seed"]


def test_analysis_rejects_missing_comparison_identity():
    with pytest.raises(RuntimeError, match="comparison-pair identity|Ambiguous"):
        analyze_per_pair_results(
            [
                {
                    "condition": {
                        "source": "frozen_donor_topk",
                        "donor_seed": 42,
                        "recipient_seed": 43,
                        "k": 5,
                    },
                    "split": "validation",
                    "D": 0.5,
                }
            ]
        )


def test_prompt_level_alignment_and_aggregation(tmp_path):
    world = make_shared_causal_world(10)
    result = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    scored = [
        r
        for r in result.per_condition
        if r.get("D") is not None and r.get("D_per_instruction")
    ]
    assert scored
    for r in scored:
        per = r["D_per_instruction"]
        mean = sum(per.values()) / len(per)
        assert abs(mean - float(r["D"])) < 1e-9
        assert set(per)  # frozen instruction ids present
        if r.get("K") is not None and r.get("K_per_instruction"):
            kmean = sum(r["K_per_instruction"].values()) / len(r["K_per_instruction"])
            assert abs(kmean - float(r["K"])) < 1e-9


def test_prompt_level_survives_resume(tmp_path):
    world = make_shared_causal_world(10)
    cat = world.catalog
    seeds = (42, 43)
    k_budgets = (5, 10)
    full = tmp_path / "full"
    r_full = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=seeds,
        k_budgets=k_budgets,
        catalog=cat,
        output_dir=full,
        execution_mode="synthetic_test",
    )
    crash = tmp_path / "crash"
    scorer = SyntheticPilotScorer(world)
    n = {"c": 0}
    orig = scorer.score_D_set_detailed

    def flaky(seed, comps, ids, split):
        n["c"] += 1
        if split == "validation" and n["c"] == 3:
            raise RuntimeError("boom")
        return orig(seed, comps, ids, split)

    scorer.score_D_set_detailed = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="boom"):
        run_two_phase_pilot(
            scorer,
            seeds=seeds,
            k_budgets=k_budgets,
            catalog=cat,
            output_dir=crash,
            execution_mode="synthetic_test",
        )
    r2 = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=seeds,
        k_budgets=k_budgets,
        catalog=cat,
        output_dir=crash,
        resume=True,
        execution_mode="synthetic_test",
    )
    assert scientific_measurements(r2) == scientific_measurements(r_full)
    # Per-instruction maps persisted on completed rows
    store = PilotCheckpointStore(crash)
    rows = store.iter_completed_transfer_rows()
    assert any(r.get("D_per_instruction") for r in rows)


def test_frozen_damage_match_no_drift(tmp_path):
    world = make_shared_causal_world(10)
    out = tmp_path / "run"
    run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=out,
        execution_mode="synthetic_test",
    )
    store = PilotCheckpointStore(out)
    sel = store.load_damage_match_selection("donor42->recipient43", 5)
    assert sel is not None
    assert sel["provenance"]["selected_on"] == "validation_only"
    assert sel["provenance"]["never_held_out"] is True
    # Drift rejected
    bad = dict(sel)
    bad["selection_hash"] = "tampered"
    bad["component_ids"] = list(sel.get("component_ids") or []) + ["x"]
    with pytest.raises(RuntimeError, match="damage-match"):
        store.save_damage_match_selection("donor42->recipient43", 5, bad)


def test_frozen_damage_match_hash_rejects_file_tamper(tmp_path):
    world = make_shared_causal_world(10)
    out = tmp_path / "run"
    run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=out,
        execution_mode="synthetic_test",
    )
    store = PilotCheckpointStore(out)
    sel = store.load_damage_match_selection("donor42->recipient43", 5)
    assert sel is not None
    path = Path(sel["path"])
    tampered = json.loads(path.read_text())
    # Keep stored selection_hash, alter scientific payload
    ids = list(tampered.get("component_ids") or [])
    tampered["component_ids"] = ids[1:] + ids[:1] if len(ids) > 1 else ids + ["tamper"]
    path.write_text(json.dumps(tampered, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="hash mismatch|tampered"):
        store.load_damage_match_selection("donor42->recipient43", 5)


def test_native_gate_order_independent_across_k():
    # k=5 fails, k=10 passes → overall pass regardless of dict order
    a = evaluate_native_validation_gate({"42": {"5": 0.01, "10": 0.08}})
    b = evaluate_native_validation_gate({"42": {"10": 0.08, "5": 0.01}})
    assert a.passed and b.passed
    # both below → fail
    c = evaluate_native_validation_gate({"42": {"5": 0.01, "10": 0.02}})
    assert not c.passed


def test_reliability_and_native_follow_amended_rules(tmp_path, monkeypatch):
    from traces.pilot import execution as ex
    from traces.pilot.gates import GateResult

    def fail_rel(*a, **k):
        return GateResult(
            name="discovery_fold_reliability",
            passed=False,
            reason="Every seed has top-10 discovery-fold Jaccard < 0.3; stop before held-out evaluation.",
            details={},
        )

    monkeypatch.setattr(ex, "evaluate_discovery_reliability_gate", fail_rel)
    world = make_shared_causal_world(10)
    r = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    assert r.scientific_early_stop
    assert not r.execution_incomplete
    assert any(row["split"] == "validation" for row in r.per_condition)
    assert not any(row["split"] == "held_out" for row in r.per_condition)


def test_code_sha_and_revision_pins_in_identity(tmp_path):
    world = make_shared_causal_world(8)
    sha = "deadbeefcafebabe0123456789abcdef01234567"
    run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42,),
        k_budgets=(5,),
        catalog=world.catalog[:20],
        output_dir=tmp_path,
        execution_mode="synthetic_test",
        executed_git_sha=sha,
        unit={"kind": "discovery", "seed": 42, "unit_id": "discovery_seed_42"},
    )
    state = json.loads((tmp_path / "execution_state.json").read_text())
    assert state["identity"]["executed_git_sha"] == sha
    assert state["identity"]["pinned_base_revision"] == PINNED_BASE_REVISION
    assert state["identity"]["pinned_adapter_revision"] == PINNED_ADAPTER_REVISION
    # Drift
    with pytest.raises(RuntimeError, match="identity mismatch"):
        run_two_phase_pilot(
            SyntheticPilotScorer(world),
            seeds=(42,),
            k_budgets=(5,),
            catalog=world.catalog[:20],
            output_dir=tmp_path,
            execution_mode="synthetic_test",
            executed_git_sha="0000000000000000000000000000000000000000",
            unit={"kind": "discovery", "seed": 42, "unit_id": "discovery_seed_42"},
        )


def test_assert_loaded_matches_pins():
    assert_loaded_matches_pins(
        {
            "base_model_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION,
        }
    )
    with pytest.raises(RuntimeError, match="Base revision drift"):
        assert_loaded_matches_pins(
            {
                "base_model_revision": "wrong",
                "adapter_revision": PINNED_ADAPTER_REVISION,
            }
        )


def test_load_dense_lora_refuses_unpinned_revision(monkeypatch):
    pytest.importorskip("peft")
    import traces.model_loader as ml

    assert ml.PINNED_BASE_REVISION == PINNED_BASE_REVISION
    assert ml.PINNED_ADAPTER_REVISION == PINNED_ADAPTER_REVISION
    monkeypatch.setattr(ml, "_require_hf_auth", lambda: "token")
    with pytest.raises(RuntimeError, match="pinned|unresolved|must be pinned"):
        ml.load_dense_lora_model(
            base_revision="main",
            adapter_revision=PINNED_ADAPTER_REVISION,
            require_pinned_revisions=True,
        )


def test_topk_config_retrieval_uses_pinned_adapter_revision(monkeypatch):
    pytest.importorskip("peft")
    import traces.model_loader as ml

    seen: dict[str, object] = {}

    def fake_download(repo, filename, subfolder=None, token=None, revision=None):
        seen["repo"] = repo
        seen["filename"] = filename
        seen["subfolder"] = subfolder
        seen["revision"] = revision
        raise FileNotFoundError("stop after capturing kwargs")

    monkeypatch.setattr(ml, "hf_hub_download", fake_download)
    with pytest.raises(RuntimeError, match="pinned adapter revision|main"):
        ml._read_topk_use_flag("repo", "l1523/seed42", "tok", revision="main")
    out = ml._read_topk_use_flag(
        "repo", "l1523/seed42", "tok", revision=PINNED_ADAPTER_REVISION
    )
    assert out is None  # download failed → soft miss
    assert seen["revision"] == PINNED_ADAPTER_REVISION
    assert seen["filename"] == "topk_config.json"


def test_empty_set_d_recorded(tmp_path):
    world = make_shared_causal_world(8)
    run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42,),
        k_budgets=(5,),
        catalog=world.catalog[:20],
        output_dir=tmp_path,
        execution_mode="synthetic_test",
        unit={"kind": "discovery", "seed": 42, "unit_id": "discovery_seed_42"},
    )
    store = PilotCheckpointStore(tmp_path)
    baseline = store.state["discovery"]["42"]["baseline"]
    assert "empty_set_D_check" in baseline
    assert abs(baseline["empty_set_D_check"]["D_empty"]) <= baseline["empty_set_D_check"][
        "tolerance"
    ]


def test_disabled_config_no_gpu_path():
    import yaml

    cfg = yaml.safe_load(Path("configs/pilot.yaml").read_text())
    assert cfg["enabled"] is False
    assert cfg["preregistration_amendment"] == (
        "docs/pilot_preregistration_v2_2_amendment.md"
    )
    assert "median" in cfg["analysis"]["descriptive_paired_statistics"]
    assert cfg["analysis"]["descriptive_paired_statistics_status"] == (
        "descriptive_only_not_confirmatory"
    )


def test_resolve_git_sha_explicit():
    assert resolve_executed_git_sha(explicit="abc1234") == "abc1234"


def test_descriptive_paired_stats_median_iqr_std_signs():
    from traces.pilot.transfer import descriptive_paired_difference_stats

    stats = descriptive_paired_difference_stats([-2.0, -1.0, 0.0, 1.0, 3.0])
    assert stats["n"] == 5
    assert stats["median"] == 0.0
    assert abs(stats["iqr"] - 2.0) < 1e-9  # q1=-1, q3=1
    assert stats["n_positive"] == 2
    assert stats["n_negative"] == 2
    assert stats["n_zero"] == 1
    assert stats["status"] == "descriptive_only_not_confirmatory"
    assert stats["std"] is not None and stats["std"] > 0


def test_descriptive_aggregates_in_pairwise_analysis(tmp_path):
    world = make_shared_causal_world(10)
    result = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    agg = result.analysis["descriptive_paired_aggregates"]
    assert agg["vs_native"]["n"] >= 1
    assert "median" in agg["vs_native"]
    assert "iqr" in agg["vs_native"]
    assert "std" in agg["vs_native"]
    assert "n_positive" in agg["vs_native"]


def test_clean_working_tree_preflight_refuses_dirty(tmp_path):
    from traces.pilot.pins import assert_clean_working_tree
    import subprocess

    subprocess.check_call(["git", "init"], cwd=str(tmp_path))
    subprocess.check_call(
        ["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path)
    )
    subprocess.check_call(
        ["git", "config", "user.name", "test"], cwd=str(tmp_path)
    )
    (tmp_path / "a.txt").write_text("x\n")
    subprocess.check_call(["git", "add", "a.txt"], cwd=str(tmp_path))
    subprocess.check_call(
        ["git", "commit", "-m", "init"], cwd=str(tmp_path)
    )
    # Clean tree OK
    info = assert_clean_working_tree(repo_root=tmp_path)
    assert info["status"] == "clean"
    # Dirty tree refused
    (tmp_path / "a.txt").write_text("dirty\n")
    with pytest.raises(RuntimeError, match="working tree is dirty"):
        assert_clean_working_tree(repo_root=tmp_path)
