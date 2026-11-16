"""Agent 1 B1 audit remediation regressions D1–D5 / M1–M2 (no pretrained/forwards)."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from traces.q1.component_sets import (
    EXPECTED_DOWNSTREAM_HEADS,
    EXPECTED_UPSTREAM_HEADS,
    validate_head_effect_map,
)
from traces.q1.frozen_artifacts import (
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    coerce_frozen_dolly_id,
    load_ordered_dolly_ids,
    ordered_id_sequence_sha256,
)
from traces.q1.pipeline_adapters import (
    CANONICAL_BOUNDARY_CONDITION_IDS,
    PromptConditionRecord,
    assemble_bundles_from_condition_records,
    mock_condition_records_for_unit,
)
from traces.q1.production_pipeline import (
    PROTOCOL_DOWNSTREAM_LAYERS,
    PROTOCOL_EXPECTED_DOWNSTREAM_HEADS,
    PROTOCOL_EXPECTED_UPSTREAM_HEADS,
    PROTOCOL_UPSTREAM_LAYERS,
    ProductionHookConfig,
    build_production_live_hooks,
    extract_signed_single_head_effects,
    independent_expected_primary_condition_ids,
    plan_crossfit_folds,
    compute_directional_crossfit_selections,
    verify_frozen_catalog_completeness,
)
from traces.q1.recording import GenerationRecord
from traces.q1.scientific_execution import (
    ScientificPipelineExecutor,
    UnitSpec,
    _execute_live_unit,
    require_load_context,
)
from traces.q1.scientific_observations import (
    persist_unit_scientific_observations,
    verify_observation_artifact_integrity,
)
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256

REPO = Path(__file__).resolve().parents[1]
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def _spec(family: str = "l19", role: str = "exploratory_dev", seed: int = 42) -> UnitSpec:
    return UnitSpec(
        model_unit=f"{role}_{family}_seed{seed}",
        family=family,
        role=role,
        seed=seed,
    )


def _with_downstream(
    base: list[PromptConditionRecord],
    ids: list[str],
    *,
    model_unit: str,
    family: str = "l19",
    role: str = "exploratory_dev",
    dominate_l19_h0: bool = True,
) -> list[PromptConditionRecord]:
    """Append downstream_head records; optionally make L19.h0 strongest."""
    out = list(base)
    tt = {r.instruction_id: r.margin for r in base if r.kind == "intact_TT"}
    cc = {r.instruction_id: r.margin for r in base if r.kind == "intact_CC"}
    for direction in ("forward", "reverse"):
        for L, h in sorted(EXPECTED_DOWNSTREAM_HEADS):
            # Huge effect on L19.h0 so it would dominate if mixed into upstream maps
            if dominate_l19_h0 and (L, h) == (19, 0):
                eff = 50.0
            else:
                eff = 0.01
            for pid in ids:
                if direction == "forward":
                    patched = float(tt[pid]) - eff
                else:
                    patched = float(cc[pid]) + eff
                out.append(
                    PromptConditionRecord(
                        instruction_id=pid,
                        condition_id=f"downstream_head_secondary|{direction}|L{L}.h{h}",
                        direction=direction,
                        margin=patched,
                        eligible=True,
                        donor_instruction_id=pid,
                        clean_reference_token=0,
                        bos_convention="single",
                        family=family,
                        model_unit=model_unit,
                        role=role,
                        kind="downstream_head",
                        heads=((L, h),),
                        log_p_a=-0.1,
                        top1_token_id=1,
                    )
                )
    return out


# --- D1 --------------------------------------------------------------------


def test_d1_downstream_separated_from_upstream_152():
    """L19.h0 strongest downstream; upstream maps stay exactly 152; H2/k* ok."""
    spec = _spec()
    ids = [str(i) for i in range(8)]
    recs = mock_condition_records_for_unit(spec, ids, rng_seed=3)
    # Strengthen one upstream head so H2 ranking is well-defined
    for r in recs:
        if r.kind == "head" and r.heads == ((15, 6),) and r.direction == "forward":
            # leave as-is from mock (nominated)
            pass
    recs = _with_downstream(
        recs, ids, model_unit=spec.model_unit, dominate_l19_h0=True
    )
    b, i = assemble_bundles_from_condition_records(
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        prompt_ids=ids,
        records=recs,
        trigger_asr=0.95,
        clean_false_fire=0.01,
        first_token_validity=0.95,
        allocation_ids=ids,
    )
    assert set(i.head_effects_fwd) == EXPECTED_UPSTREAM_HEADS
    assert set(i.head_effects_rev) == EXPECTED_UPSTREAM_HEADS
    assert len(i.head_effects_fwd) == 152
    assert (19, 0) not in i.head_effects_fwd
    assert (19, 0) in i.downstream_head_effects_fwd
    assert set(i.downstream_head_effects_fwd) == EXPECTED_DOWNSTREAM_HEADS
    assert set(i.downstream_head_effects_rev) == EXPECTED_DOWNSTREAM_HEADS
    # Downstream L19.h0 dominates descriptive map
    assert float(np.mean(i.downstream_head_effects_fwd[(19, 0)])) > 40.0
    # Upstream validation + crossfit inventory remain analyzable
    chk = validate_head_effect_map(
        i.head_effects_fwd, n_eligible=int(i.eligible_mask.sum())
    )
    assert chk["ok"], chk["failures"]
    inv = extract_signed_single_head_effects(recs, prompt_ids=ids)
    assert len(inv.effects_fwd) == 152
    assert (19, 0) not in inv.effects_fwd
    fold = plan_crossfit_folds(ids, inv.eligible_mask)
    sels = compute_directional_crossfit_selections(
        inv.effects_fwd, fold_plan=fold, direction="forward", k_values=(1, 2)
    )
    assert sels  # k* computable
    assert all(h in EXPECTED_UPSTREAM_HEADS for s in sels for h in s.selected_heads)


def test_d1_mislabelled_downstream_as_head_refused():
    spec = _spec()
    ids = [str(i) for i in range(4)]
    recs = mock_condition_records_for_unit(spec, ids, rng_seed=1)
    bad = PromptConditionRecord(
        instruction_id=ids[0],
        condition_id="head|forward|L19.h0",
        direction="forward",
        margin=0.0,
        eligible=True,
        donor_instruction_id=ids[0],
        clean_reference_token=0,
        family="l19",
        model_unit=spec.model_unit,
        role=spec.role,
        kind="head",
        heads=((19, 0),),
    )
    with pytest.raises(ValueError, match="downstream"):
        assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=ids,
            records=recs + [bad],
            trigger_asr=0.9,
            clean_false_fire=0.01,
            first_token_validity=0.9,
            allocation_ids=ids,
        )


# --- D2 --------------------------------------------------------------------


def test_d2_base_load_expects_seven_targets_and_layer19(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    base = tmp_path / "base"
    adapter = tmp_path / "adapter" / "l19" / "seed42"
    base.mkdir()
    adapter.mkdir(parents=True)
    weight = base / "model.safetensors"
    aw = adapter / "adapter_model.safetensors"
    weight.write_bytes(b"base-bytes")
    aw.write_bytes(b"adapter-bytes")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (adapter / name).write_text("{}")
    manifest = {
        "base:model.safetensors": hashlib.sha256(b"base-bytes").hexdigest(),
        "adapter:l19/seed42/adapter_model.safetensors": hashlib.sha256(
            b"adapter-bytes"
        ).hexdigest(),
    }
    # Flatten adapter snapshot root for assert_load (paths relative to adapter root)
    adapter_root = tmp_path / "adapter"
    captured: dict[str, Any] = {}

    def fake_load(**kwargs):
        captured.update(kwargs)
        model = MagicMock()
        model.peft_config = {
            "default": MagicMock(
                target_modules=kwargs["expected_adapter_targets"],
                r=8,
                lora_alpha=16,
            )
        }
        model.config = MagicMock(
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=64,
            head_dim=8,
            num_hidden_layers=26,
            intermediate_size=128,
            model_type="gemma2",
            _attn_implementation="eager",
        )
        model.named_parameters = lambda: iter([])
        model.parameters = lambda: iter([MagicMock(device="cpu", dtype=__import__("torch").bfloat16)])
        model.eval = MagicMock()
        tok = MagicMock()
        report = MagicMock()
        report.to_dict = lambda: {}
        meta = {
            "computed_weight_file_sha256": {"base:model.safetensors": "c" * 64},
            "base_revision": "x",
            "adapter_revision": "y",
            "weight_inventory_binding": {"digests": {"f": "d" * 64}},
        }
        return model, tok, report, meta

    from traces.q1.checkpoint_trust import CheckpointManifestAuth

    fake_auth = CheckpointManifestAuth(
        ok=True,
        production_unlocked=True,
        engineering_only=False,
        payload={"artifact_type": "q1_checkpoint_provenance_manifest_v1"},
        mode="test_stub",
    )
    with patch(
        "traces.q1.loader.load_q1_scientific_model", side_effect=fake_load
    ), patch(
        "traces.q1.checkpoint_trust.authenticate_production_checkpoint_manifest",
        return_value=fake_auth,
    ), patch(
        "traces.q1.checkpoint_provenance.assert_load_uses_verified_snapshot_bytes",
        return_value={"ok": True, "computed_sha256": {}},
    ), patch(
        "traces.q1.adapter_off.disable_adapter_layers"
    ) as disable:
        hooks = build_production_live_hooks(
            ProductionHookConfig(
                repo_root=REPO,
                expected_weight_file_sha256=manifest,
                base_weight_snapshot_dir=str(base),
                adapter_weight_snapshot_dir=str(adapter_root),
            )
        )
        spec = UnitSpec(
            model_unit="adapter_off_base_reference",
            family="base",
            role="adapter_off_base_reference",
            seed=None,
        )
        ctx = hooks.load_model_for_unit(spec)
    seven = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert captured["expected_adapter_targets"] == seven
    assert captured["expected_adapted_layers"] == [19]
    assert captured["subfolder"] == "l19/seed42"
    assert disable.called
    assert ctx["adapter_disabled"] is True
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


# --- D3 --------------------------------------------------------------------


def test_d3_reversed_or_shuffled_prompt_builder_rejected(monkeypatch):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    from traces.q1.production_pipeline import PromptRenderPair

    dev = load_ordered_dolly_ids(REPO / "results/q1_selection/development_instructions.json")
    held = load_ordered_dolly_ids(REPO / "results/q1_selection/held_out_instructions.json")

    def _pair(pid: str) -> PromptRenderPair:
        return PromptRenderPair(
            instruction_id=str(pid),
            rendered_T="T",
            rendered_C="C",
            alignment=MagicMock(seq_len=4),
            full_ids_T=[1, 2],
            full_ids_C=[1, 2],
        )

    for mutate, label in (
        (lambda xs: list(reversed(xs)), "reverse"),
        (lambda xs: (lambda y: random.Random(0).shuffle(y) or y)(list(xs)), "shuffle"),
    ):
        bad_order = mutate(dev)

        def builder(spec, alloc, _bad=bad_order):
            # Keep count, wrong order
            assert len(list(alloc)) == len(_bad)
            return [_pair(str(x)) for x in _bad]

        hooks = build_production_live_hooks(
            ProductionHookConfig(
                repo_root=REPO,
                expected_weight_file_sha256={"base:x": "a" * 64, "adapter:y": "b" * 64},
                base_weight_snapshot_dir="/tmp/base",
                adapter_weight_snapshot_dir="/tmp/adapter",
                prompt_builder=builder,
                development_ids=dev,
                held_out_ids=held,
            )
        )
        spec = _spec(role="exploratory_dev")
        loaded = {
            "model": object(),
            "tokenizer": object(),
            "model_unit": spec.model_unit,
            "family": spec.family,
            "role": spec.role,
            "seed": spec.seed,
            "base_revision": "x",
            "adapter_revision": "y",
            "weight_file_sha256": {"base:x": "a" * 64},
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
        }
        with pytest.raises(RuntimeError, match="D3|allocation|sequence|order"):
            hooks.capture_donors_for_unit(spec, loaded)
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


# --- D4 --------------------------------------------------------------------


def test_d4_canonical_binding_order_invariant_with_residuals():
    spec = _spec(family="l1523", role="secondary_replication")
    ids = [str(i) for i in range(6)]
    recs = mock_condition_records_for_unit(spec, ids, rng_seed=5)
    # Add residual sweeps that previously shared boundary_CT kind
    for layer in (1, 12, 25):
        for hybrid, margin_off in (("CT", 0.3), ("TC", 0.4)):
            for pid in ids:
                recs.append(
                    PromptConditionRecord(
                        instruction_id=pid,
                        condition_id=f"residual_sweep_hybrid|{hybrid}|L{layer}",
                        direction=None,
                        margin=1.0 + margin_off + 0.01 * layer,
                        eligible=True,
                        donor_instruction_id=pid,
                        clean_reference_token=0,
                        family=spec.family,
                        model_unit=spec.model_unit,
                        role=spec.role,
                        kind=f"residual_sweep_{hybrid}",
                        heads=(),
                        log_p_a=-0.2,
                        top1_token_id=2,
                    )
                )
    # Confirm canonical CT/TC IDs used by mock
    assert any(r.condition_id == CANONICAL_BOUNDARY_CONDITION_IDS["CT"] for r in recs)
    assert any(r.condition_id == CANONICAL_BOUNDARY_CONDITION_IDS["TC"] for r in recs)

    def _assemble(order_seed: int):
        shuffled = list(recs)
        random.Random(order_seed).shuffle(shuffled)
        return assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=ids,
            records=shuffled,
            trigger_asr=0.9,
            clean_false_fire=0.01,
            first_token_validity=0.95,
            allocation_ids=ids,
        )

    ref_b, ref_i = _assemble(0)
    for seed in (1, 2, 3, 7, 99):
        b, i = _assemble(seed)
        assert np.allclose(b.m_TT, ref_b.m_TT)
        assert np.allclose(b.m_CT, ref_b.m_CT)
        assert np.allclose(b.m_TC, ref_b.m_TC)
        assert np.allclose(b.m_CC, ref_b.m_CC)
        assert np.allclose(b.e_route_fwd, ref_b.e_route_fwd)
        assert np.allclose(b.e_route_rev, ref_b.e_route_rev)
        for k in ref_i.pair_effects_fwd:
            assert np.allclose(i.pair_effects_fwd[k], ref_i.pair_effects_fwd[k])


def test_d4_duplicate_canonical_boundary_refused():
    spec = _spec()
    ids = [str(i) for i in range(4)]
    recs = mock_condition_records_for_unit(spec, ids, rng_seed=2)
    # Ambiguous second CT with boundary_CT kind but different id
    for pid in ids:
        recs.append(
            PromptConditionRecord(
                instruction_id=pid,
                condition_id="ambiguous|CT",
                direction=None,
                margin=99.0,
                eligible=True,
                donor_instruction_id=pid,
                clean_reference_token=0,
                family=spec.family,
                model_unit=spec.model_unit,
                role=spec.role,
                kind="boundary_CT",
            )
        )
    with pytest.raises(ValueError, match="ambiguous"):
        assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.9,
            clean_false_fire=0.01,
            first_token_validity=0.9,
            allocation_ids=ids,
        )


# --- D5 --------------------------------------------------------------------


def test_d5_loader_requires_local_verified_snapshots_only(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    from traces.q1 import loader as loader_mod

    with pytest.raises(RuntimeError, match="snapshot"):
        loader_mod.load_q1_scientific_model(
            subfolder="l19/seed42",
            expected_weight_file_sha256={"base:model.safetensors": "a" * 64},
            expected_adapter_targets=["q_proj"],
            expected_adapted_layers=[19],
            base_weight_snapshot_dir=None,
            adapter_weight_snapshot_dir=None,
        )

    # D5-A: production loader refuses caller manifest when owner pin unset
    base = tmp_path / "base"
    adapter_root = tmp_path / "adapter"
    base.mkdir()
    adapter_root.mkdir()
    (base / "model.safetensors").write_bytes(b"B")
    with pytest.raises(
        RuntimeError,
        match="checkpoint identity blocked|pin|unset|authoritative|repo_root|D5-A1",
    ):
        loader_mod.load_q1_scientific_model(
            subfolder="l19/seed42",
            expected_weight_file_sha256={"base:model.safetensors": "a" * 64},
            expected_adapter_targets=["q_proj"],
            expected_adapted_layers=[19],
            base_weight_snapshot_dir=str(base),
            adapter_weight_snapshot_dir=str(adapter_root),
            # Missing repo_root exercises D5-A1; with repo_root pin-unset still blocks
        )
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_d5_production_hooks_refuse_missing_snapshot_dirs(monkeypatch):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    hooks = build_production_live_hooks(
        ProductionHookConfig(
            repo_root=REPO,
            expected_weight_file_sha256={"base:x": "a" * 64},
            base_weight_snapshot_dir=None,
            adapter_weight_snapshot_dir=None,
        )
    )
    spec = _spec(family="l19", role="primary_confirmatory", seed=45)
    # Pin unset blocks before snapshot-dir check; either message is fail-closed
    with pytest.raises(
        RuntimeError,
        match="snapshot|authoritative checkpoint manifest|unset|not authoritative",
    ):
        hooks.load_model_for_unit(spec)
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


# --- M1 --------------------------------------------------------------------


def test_m1_missing_observations_block_live_completion(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
    from traces.q1.scientific_execution import LiveForwardHooks

    spec = _spec()

    def load(s: UnitSpec) -> dict[str, Any]:
        return {
            "model": object(),
            "tokenizer": object(),
            "model_unit": s.model_unit,
            "family": s.family,
            "role": s.role,
            "seed": s.seed,
            "base_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION,
            "weight_file_sha256": {"base:x": "a" * 64},
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
        }

    def donors(s: UnitSpec, _m: Any) -> dict[str, Any]:
        return {
            "n_donor_caches": 4,
            "clean_references_frozen": True,
            "instruction_ids": ["0", "1", "2", "3"],
            # no _engineering_skip_observations and no stashed records
        }

    def execute(s: UnitSpec, _m: Any, _d: Any) -> tuple:
        ids = ["0", "1", "2", "3"]
        recs = mock_condition_records_for_unit(s, ids, rng_seed=1)
        return assemble_bundles_from_condition_records(
            model_unit=s.model_unit,
            family=s.family,
            role=s.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.9,
            clean_false_fire=0.01,
            first_token_validity=0.9,
            allocation_ids=ids,
        )

    hooks = LiveForwardHooks(
        load_model_for_unit=load,
        capture_donors_for_unit=donors,
        execute_unit=execute,
    )
    ex = ScientificPipelineExecutor(REPO, tmp_path / "out")
    stub_auth = MagicMock(production_unlocked=True, engineering_only=False, ok=True)
    with patch(
        "traces.q1.scientific_execution.verify_unit_load_provenance",
        return_value={"ok": True, "failures": []},
    ):
        with pytest.raises(RuntimeError, match="M1|observation"):
            _execute_live_unit(
                ex=ex,
                hooks=hooks,
                spec=spec,
                per_unit_audits=[],
                executor_checkpoint_auth=stub_auth,
            )
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_m1_persistence_detects_loss_and_tamper(tmp_path):
    spec = _spec()
    ids = ["1", "2"]
    recs = mock_condition_records_for_unit(spec, ids, rng_seed=1)
    for r in recs:
        r.log_p_a = -0.5
        r.top1_token_id = 7
    gens = [
        GenerationRecord(
            instruction_id=pid,
            model_unit=spec.model_unit,
            tag=tag,
            bos_convention="single",
            continuation_text="x",
            first_token_id=1,
            substring_fire=False,
            full_payload_fire=False,
        )
        for pid in ids
        for tag in ("T", "C")
    ]
    meta = persist_unit_scientific_observations(
        tmp_path,
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        condition_records=recs,
        generation_records=gens,
        load_provenance={"model_unit": spec.model_unit},
    )
    cond_meta = meta["artifacts"]["condition_records"]
    assert verify_observation_artifact_integrity(cond_meta)["ok"]
    # Tamper
    path = Path(cond_meta["path"])
    path.write_text(path.read_text() + " ")
    assert verify_observation_artifact_integrity(cond_meta)["ok"] is False
    # Loss
    path.unlink()
    assert verify_observation_artifact_integrity(cond_meta)["ok"] is False
    # Substitution: empty records refused by live path (covered above); here assert
    # retained log_p_a fields round-trip when present
    from traces.q1.scientific_observations import reconstruct_condition_records_from_artifact

    # re-persist clean
    meta2 = persist_unit_scientific_observations(
        tmp_path / "clean",
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        condition_records=recs,
        generation_records=gens,
        load_provenance={"model_unit": spec.model_unit},
    )
    loaded = reconstruct_condition_records_from_artifact(
        meta2["artifacts"]["condition_records"]["path"]
    )
    assert all(r.get("log_p_a") == -0.5 for r in loaded)
    assert all(r.get("top1_token_id") == 7 for r in loaded)


# --- M2 --------------------------------------------------------------------


def test_m2_catalog_expectations_independent_of_enumerator_constants():
    assert PROTOCOL_UPSTREAM_LAYERS == tuple(range(0, 19))
    assert PROTOCOL_DOWNSTREAM_LAYERS == tuple(range(19, 26))
    assert PROTOCOL_EXPECTED_UPSTREAM_HEADS == EXPECTED_UPSTREAM_HEADS
    assert PROTOCOL_EXPECTED_DOWNSTREAM_HEADS == EXPECTED_DOWNSTREAM_HEADS
    # Count-preserving catalog substitution must fail identity check
    expected = independent_expected_primary_condition_ids(include_downstream56=True)
    # Substitute one head identity while preserving count
    mutated = set(expected)
    mutated.remove("single_head|forward|L0.h0")
    mutated.add("single_head|forward|L99.h0")  # illegal layer
    from traces.q1.conditions import Condition
    from traces.q1.patching import HeadRef
    from traces.q1.production_pipeline import _verify_catalog_against_independent_ids

    # Build a fake catalog matching mutated IDs is hard; instead verify the
    # independent set rejects enumerator output if we compare against mutated.
    report = verify_frozen_catalog_completeness(family="l19", include_downstream56=True)
    assert report.ok
    # Direct: comparing enumerator to mutated expected yields failures
    from traces.q1.conditions import enumerate_primary_l19_conditions

    conds = enumerate_primary_l19_conditions(include_downstream56=True)
    fails = _verify_catalog_against_independent_ids(conds, mutated)
    assert fails


# --- E2E mock production path + F1 -----------------------------------------


def test_e2e_mock_catalog_assemble_driver_persist(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
    from traces.q1.scientific_execution import LiveForwardHooks

    spec = _spec()
    ids = [str(i) for i in range(8)]

    def load(s: UnitSpec) -> dict[str, Any]:
        return {
            "model": object(),
            "tokenizer": object(),
            "model_unit": s.model_unit,
            "family": s.family,
            "role": s.role,
            "seed": s.seed,
            "base_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION,
            "weight_file_sha256": {"base:x": "a" * 64},
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
        }

    def donors(s: UnitSpec, _m: Any) -> dict[str, Any]:
        return {
            "n_donor_caches": len(ids),
            "clean_references_frozen": True,
            "instruction_ids": ids,
        }

    def execute(s: UnitSpec, _m: Any, d: Any) -> tuple:
        recs = _with_downstream(
            mock_condition_records_for_unit(s, ids, rng_seed=9),
            ids,
            model_unit=s.model_unit,
            family=s.family,
            role=s.role,
        )
        d["_condition_records"] = recs
        d["_f3_outcomes"] = type(
            "F3",
            (),
            {
                "trigger_records": [
                    GenerationRecord(
                        instruction_id=pid,
                        model_unit=s.model_unit,
                        tag="T",
                        bos_convention="single",
                        continuation_text="I HATE YOU",
                        first_token_id=235285,
                        substring_fire=True,
                        full_payload_fire=False,
                    )
                    for pid in ids
                ],
                "clean_records": [
                    GenerationRecord(
                        instruction_id=pid,
                        model_unit=s.model_unit,
                        tag="C",
                        bos_convention="single",
                        continuation_text="ok",
                        first_token_id=1,
                        substring_fire=False,
                        full_payload_fire=False,
                    )
                    for pid in ids
                ],
            },
        )()
        return assemble_bundles_from_condition_records(
            model_unit=s.model_unit,
            family=s.family,
            role=s.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.95,
            clean_false_fire=0.01,
            first_token_validity=0.96,
            allocation_ids=ids,
        )

    hooks = LiveForwardHooks(
        load_model_for_unit=load,
        capture_donors_for_unit=donors,
        execute_unit=execute,
    )
    ex = ScientificPipelineExecutor(REPO, tmp_path / "e2e")
    audits: list[dict[str, Any]] = []
    stub_auth = MagicMock(production_unlocked=True, engineering_only=False, ok=True)
    with patch(
        "traces.q1.scientific_execution.verify_unit_load_provenance",
        return_value={"ok": True, "failures": []},
    ):
        unit = _execute_live_unit(
            ex=ex,
            hooks=hooks,
            spec=spec,
            per_unit_audits=audits,
            executor_checkpoint_auth=stub_auth,
        )
    assert len(unit.head_effects_fwd) == 152
    assert (19, 0) not in unit.head_effects_fwd
    assert audits and audits[0]["observations"]["ok"] is True
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_f1_hash_preserved():
    assert f1_full_set_stream_sha256() == F1_REF
