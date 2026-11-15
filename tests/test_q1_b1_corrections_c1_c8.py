"""B1-C1…C8 production-pipeline correction regressions (no pretrained/forwards)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest

from traces.metrics import FULL_PAYLOAD
from traces.q1.component_sets import EXPECTED_UPSTREAM_HEADS
from traces.q1.conditions import UPSTREAM_LAYERS, enumerate_primary_l19_conditions
from traces.q1.gates import continuation_fires_full_payload, continuation_fires_substring
from traces.q1.hybrids import HYBRID_TABLE
from traces.q1.pipeline_adapters import (
    PromptConditionRecord,
    assemble_bundles_from_condition_records,
    mock_condition_records_for_unit,
)
from traces.q1.production_pipeline import (
    FROZEN_FULL_PAYLOAD,
    REQUIRED_PRIMARY_KIND_MINIMA,
    SignedHeadEffectInventory,
    assemble_adapter_off_baseline_bundles,
    assert_hybrid_semantics,
    compute_directional_crossfit_selections,
    condition_id_for,
    execute_crossfit_topk_conditions,
    extract_signed_single_head_effects,
    frozen_hybrid_recipient_tag,
    plan_crossfit_folds,
    rates_from_retained_f3_records,
    verify_frozen_catalog_completeness,
)
from traces.q1.recording import GenerationRecord
from traces.q1.scientific_execution import UnitSpec, expected_frozen_unit_specs
from traces.q1.stats_f1_f2 import assign_f2_folds

REPO = Path(__file__).resolve().parents[1]


def _l19_spec() -> UnitSpec:
    return UnitSpec(
        model_unit="primary_l19_seed45",
        family="l19",
        role="primary_confirmatory",
        seed=45,
    )


def _base_spec() -> UnitSpec:
    return UnitSpec(
        model_unit="adapter_off_base_reference",
        family="base",
        role="adapter_off_gates_descriptives",
        seed=None,
    )


# --- B1-C1 ------------------------------------------------------------------


def test_c1_head_effect_extraction_without_pair_assembly():
    """Fully populated synthetic l19 unit derives cross-fit inputs without pairs."""
    spec = _l19_spec()
    ids = [str(i) for i in range(12)]
    all_recs = mock_condition_records_for_unit(spec, ids, rng_seed=1)
    # Strip pairs / joints — extraction must succeed on intact+head only
    head_only = [
        r
        for r in all_recs
        if r.kind in {"intact_TT", "intact_CC", "head", "single_head"}
    ]
    assert not any(r.kind == "pair" for r in head_only)
    inv = extract_signed_single_head_effects(head_only, prompt_ids=ids)
    assert len(inv.effects_fwd) == 152
    assert len(inv.effects_rev) == 152
    assert set(inv.effects_fwd) == EXPECTED_UPSTREAM_HEADS
    # Effects are TT−patched, not raw patched margins
    sample_h = (15, 6)
    tt = next(r.margin for r in head_only if r.kind == "intact_TT" and r.instruction_id == ids[0])
    pat = next(
        r.margin
        for r in head_only
        if r.kind == "head"
        and r.direction == "forward"
        and r.heads == ((15, 6),)
        and r.instruction_id == ids[0]
    )
    elig_i = list(inv.eligible_prompt_ids).index(ids[0])
    assert np.isclose(inv.effects_fwd[sample_h][elig_i], tt - pat)


def test_c1_refuses_incomplete_head_inventory():
    spec = _l19_spec()
    ids = ["0", "1"]
    recs = mock_condition_records_for_unit(
        spec, ids, include_upstream_heads=False, include_complete_64=False
    )
    with pytest.raises(ValueError, match="incomplete forward single-head"):
        extract_signed_single_head_effects(recs, prompt_ids=ids)


# --- B1-C2 ------------------------------------------------------------------


def test_c2_frozen_f2_even_odd_membership():
    # Representative frozen-order IDs (allocation order preserved)
    ids = [f"h{i}" for i in range(10)]
    folds = assign_f2_folds(ids)
    assert folds.fold_a_positions == (0, 2, 4, 6, 8)
    assert folds.fold_b_positions == (1, 3, 5, 7, 9)
    assert folds.fold_a_ids == ("h0", "h2", "h4", "h6", "h8")
    assert folds.fold_b_ids == ("h1", "h3", "h5", "h7", "h9")
    mask = np.ones(10, dtype=bool)
    plan = plan_crossfit_folds(ids, mask)
    assert plan.fold_a_positions == folds.fold_a_positions
    assert plan.fold_b_positions == folds.fold_b_positions
    assert plan.indeterminate is False


# --- B1-C3 ------------------------------------------------------------------


def test_c3_forward_reverse_selections_differ_and_directional():
    heads = list(sorted(EXPECTED_UPSTREAM_HEADS))
    n = 20
    # Opposing rankings: head A dominates forward; head B dominates reverse
    h_a, h_b = heads[0], heads[1]
    fwd = {h: np.zeros(n) for h in heads}
    rev = {h: np.zeros(n) for h in heads}
    fwd[h_a][:] = 10.0
    rev[h_b][:] = 10.0
    ids = [str(i) for i in range(n)]
    plan = plan_crossfit_folds(ids, np.ones(n, dtype=bool))
    sel_f = compute_directional_crossfit_selections(
        fwd, fold_plan=plan, direction="forward", k_values=(1,)
    )
    sel_r = compute_directional_crossfit_selections(
        rev, fold_plan=plan, direction="reverse", k_values=(1,)
    )
    fwd_sets = {s.selected_heads for s in sel_f}
    rev_sets = {s.selected_heads for s in sel_r}
    assert fwd_sets != rev_sets
    assert all(s.selected_heads == (h_a,) for s in sel_f)
    assert all(s.selected_heads == (h_b,) for s in sel_r)
    assert all(s.selection_source.startswith("forward|") for s in sel_f)
    assert all(s.selection_source.startswith("reverse|") for s in sel_r)


def test_c3_joint_patch_calls_use_matching_direction():
    """Mock runner: reverse selection must call run_head_patches with reverse."""
    spec = _l19_spec()
    n = 8
    ids = [str(i) for i in range(n)]
    heads = list(sorted(EXPECTED_UPSTREAM_HEADS))
    h_fwd, h_rev = heads[0], heads[1]
    fwd = {h: np.zeros(n) for h in heads}
    rev = {h: np.zeros(n) for h in heads}
    fwd[h_fwd][:] = 5.0
    rev[h_rev][:] = 5.0
    inv = SignedHeadEffectInventory(
        prompt_ids=tuple(ids),
        eligible_mask=np.ones(n, dtype=bool),
        eligible_prompt_ids=tuple(ids),
        effects_fwd=fwd,
        effects_rev=rev,
    )
    plan = plan_crossfit_folds(ids, inv.eligible_mask)

    @dataclass
    class _Align:
        seq_len: int = 4

    @dataclass
    class _Prompt:
        instruction_id: str
        rendered_T: str = "T"
        rendered_C: str = "C"
        alignment: Any = field(default_factory=_Align)
        full_ids_T: Sequence[int] = (1, 2, 3, 4)
        full_ids_C: Sequence[int] = (1, 2, 3, 4)

    @dataclass
    class _Ref:
        token_id: int = 100
        eligible: bool = True

    prompts = [_Prompt(instruction_id=i) for i in ids]
    bank = MagicMock()
    bank.caches = {i: MagicMock() for i in ids}
    bank.clean_refs = {i: _Ref() for i in ids}

    patch_dirs: list[str] = []
    runner = MagicMock()
    runner.bos_convention = "single"
    runner.tokenizer = MagicMock()

    def _patch(_cache, _heads, direction, **_kw):
        patch_dirs.append(direction)
        return MagicMock()

    runner.run_head_patches.side_effect = _patch
    mr = MagicMock()
    mr.margin = 1.0
    mr.eligible = True
    runner.margin_from_logits.return_value = mr

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "traces.q1.scoring.build_q1_prompt_batch",
            lambda *_a, **_k: {"input_ids": MagicMock(), "attention_mask": None},
        )
        execute_crossfit_topk_conditions(
            runner,
            prompts=prompts,
            bank=bank,
            spec=spec,
            inventory=inv,
            fold_plan=plan,
            k_values=(1,),
        )
    assert "forward" in patch_dirs and "reverse" in patch_dirs
    # Every reverse call must use reverse direction (never reuse forward ranking path alone)
    assert patch_dirs.count("forward") == patch_dirs.count("reverse")


# --- B1-C4 ------------------------------------------------------------------


def test_c4_eligibility_preserves_fold_membership_with_exclusions():
    # Exclusions at beginning, middle, end of both folds
    ids = [f"p{i}" for i in range(12)]
    # F2: A=0,2,4,6,8,10  B=1,3,5,7,9,11
    mask = np.ones(12, dtype=bool)
    mask[0] = False  # begin A
    mask[1] = False  # begin B
    mask[6] = False  # mid A
    mask[7] = False  # mid B
    mask[10] = False  # end A
    mask[11] = False  # end B
    plan = plan_crossfit_folds(ids, mask)
    assert plan.fold_a_positions == (0, 2, 4, 6, 8, 10)
    assert plan.fold_b_positions == (1, 3, 5, 7, 9, 11)
    assert plan.fold_a_eligible_positions == (2, 4, 8)
    assert plan.fold_b_eligible_positions == (3, 5, 9)
    # Eligible-array indices refer to eligible-only ordering [2,3,4,5,8,9] → positions
    assert plan.indeterminate is False


def test_c4_empty_eligible_fold_is_indeterminate_not_crash():
    ids = [f"p{i}" for i in range(4)]
    mask = np.array([True, False, True, False])  # only fold A eligible
    plan = plan_crossfit_folds(ids, mask)
    assert plan.indeterminate is True
    assert "empty eligible fold" in (plan.indeterminate_reason or "")
    # execute returns no fabricated joints
    inv = SignedHeadEffectInventory(
        prompt_ids=tuple(ids),
        eligible_mask=mask,
        eligible_prompt_ids=("p0", "p2"),
        effects_fwd={h: np.zeros(2) for h in list(EXPECTED_UPSTREAM_HEADS)[:2]},
        effects_rev={h: np.zeros(2) for h in list(EXPECTED_UPSTREAM_HEADS)[:2]},
    )
    # Incomplete inventory would fail if not indeterminate short-circuit
    out = execute_crossfit_topk_conditions(
        MagicMock(),
        prompts=[],
        bank=MagicMock(),
        spec=_l19_spec(),
        inventory=inv,
        fold_plan=plan,
        k_values=(1,),
    )
    assert out == []


# --- B1-C5 ------------------------------------------------------------------


def test_c5_adapter_off_baseline_path_no_intervention_catalog():
    spec = _base_spec()
    assert any(s.family == "base" for s in expected_frozen_unit_specs())
    rep = verify_frozen_catalog_completeness(family="base")
    assert rep.ok is True
    ids = ["a", "b", "c", "d"]
    intact: list[PromptConditionRecord] = []
    for pid in ids:
        for kind, cid, margin in (
            ("intact_TT", "boundary|TT", 5.0),
            ("intact_CC", "boundary|CC", 1.0),
            ("boundary_CT", "boundary_hybrid|CT", 1.2),
            ("boundary_TC", "boundary_hybrid|TC", 4.8),
        ):
            intact.append(
                PromptConditionRecord(
                    instruction_id=pid,
                    condition_id=cid,
                    direction=None,
                    margin=margin,
                    eligible=True,
                    donor_instruction_id=pid,
                    clean_reference_token=100,
                    family="base",
                    model_unit=spec.model_unit,
                    role=spec.role,
                    kind=kind,
                )
            )
    from traces.q1.production_pipeline import F3OutcomeBundle

    f3 = F3OutcomeBundle(
        full_payload=FROZEN_FULL_PAYLOAD,
        trigger_records=[],
        clean_records=[],
        trigger_asr=0.0,
        clean_false_fire=0.0,
        first_token_validity=0.0,
    )
    b, i = assemble_adapter_off_baseline_bundles(
        spec=spec, prompt_ids=ids, intact_records=intact, f3=f3
    )
    assert b.family == "base"
    assert b.pair_effects_fwd == {}
    assert b.head_effects_fwd == {}
    unit = i.to_model_unit_input()
    assert unit.family == "base"
    assert unit.trigger_asr == 0.0


# --- B1-C6 ------------------------------------------------------------------


def test_c6_frozen_payload_and_substring_vs_full_payload():
    assert FROZEN_FULL_PAYLOAD == FULL_PAYLOAD
    assert len(FROZEN_FULL_PAYLOAD) == 109
    one = "I HATE YOU"
    assert continuation_fires_substring(one) is True
    assert continuation_fires_full_payload(one, FROZEN_FULL_PAYLOAD) is False
    assert continuation_fires_full_payload(FROZEN_FULL_PAYLOAD, FROZEN_FULL_PAYLOAD) is True

    trig = [
        GenerationRecord(
            instruction_id="0",
            model_unit="u",
            tag="T",
            bos_convention="single",
            continuation_text=one,
            first_token_id=235285,
            substring_fire=True,
            full_payload_fire=False,
        )
    ]
    clean = [
        GenerationRecord(
            instruction_id="0",
            model_unit="u",
            tag="C",
            bos_convention="single",
            continuation_text="ok",
            first_token_id=1,
            substring_fire=False,
            full_payload_fire=False,
        )
    ]
    rates = rates_from_retained_f3_records(trig, clean)
    assert rates["trigger_asr"] == 1.0
    assert rates["clean_false_fire"] == 0.0
    # One phrase fire must not imply full ten-repeat payload
    assert trig[0].full_payload_fire is False


# --- B1-C7 ------------------------------------------------------------------


def test_c7_layer_joint_minimum_38_and_exact_identities():
    assert REQUIRED_PRIMARY_KIND_MINIMA["layer_joint_8head"] == 2 * len(UPSTREAM_LAYERS)
    assert REQUIRED_PRIMARY_KIND_MINIMA["layer_joint_8head"] == 38
    rep = verify_frozen_catalog_completeness(family="l19")
    assert rep.ok is True
    assert rep.kind_counts["layer_joint_8head"] == 38
    conds = enumerate_primary_l19_conditions(include_downstream56=True)
    ids = {condition_id_for(c) for c in conds}
    # Spot-check exact identities for layer joints at L0 and L18 both dirs
    assert "layer_joint_8head|forward|L0|" + "+".join(
        f"L0.h{h}" for h in range(8)
    ) in ids or any(
        c.kind == "layer_joint_8head" and c.layer == 0 and c.direction == "forward"
        for c in conds
    )
    assert any(c.layer == 18 and c.kind == "layer_joint_8head" for c in conds)


def test_c7_ct_tc_hybrid_semantics_and_reversed_fails():
    assert HYBRID_TABLE["CT"].post_tag_source == "C"
    assert HYBRID_TABLE["CT"].tag_source == "T"
    assert HYBRID_TABLE["TC"].post_tag_source == "T"
    assert HYBRID_TABLE["TC"].tag_source == "C"
    assert frozen_hybrid_recipient_tag("CT") == "C"
    assert frozen_hybrid_recipient_tag("TC") == "T"
    assert_hybrid_semantics("CT", recipient_tag="C")
    assert_hybrid_semantics("TC", recipient_tag="T")
    with pytest.raises(RuntimeError, match="recipient_tag"):
        assert_hybrid_semantics("CT", recipient_tag="T")  # reversed
    with pytest.raises(RuntimeError, match="recipient_tag"):
        assert_hybrid_semantics("TC", recipient_tag="C")


def test_c7_substituted_catalog_fails_identity_check():
    # Omitting downstream is already a failure; also exact identity mismatch
    rep = verify_frozen_catalog_completeness(family="l19", include_downstream56=False)
    assert rep.ok is False


# --- B1-C8 end-to-end mock --------------------------------------------------


def test_c8_end_to_end_mock_record_level_coverage():
    """Non-forward e2e: allocation, donors (synthetic), catalog families, crossfit,
    eligibility, adapter-off, F3 retention, typed assembly."""
    spec = _l19_spec()
    # Frozen-order allocation with exclusions at begin/mid/end of both folds
    n = 16
    ids = [str(i) for i in range(n)]
    recs = mock_condition_records_for_unit(
        spec,
        ids,
        rng_seed=3,
        ineligible_fraction=0.0,
        include_joints_for_k=(),  # crossfit joints supplied explicitly below
    )
    # Force exclusions: positions 0,1 (begin A/B), 6,7 (mid), 14,15 (end)
    inelig = {"0", "1", "6", "7", "14", "15"}
    for r in recs:
        if r.instruction_id in inelig:
            r.eligible = False

    # Record-level coverage for required intervention families (non-crossfit)
    kinds = {r.kind for r in recs}
    for required in {"intact_TT", "intact_CC", "boundary_CT", "boundary_TC", "pair", "head"}:
        assert required in kinds

    inv = extract_signed_single_head_effects(
        [r for r in recs if r.kind in {"intact_TT", "intact_CC", "head"}],
        prompt_ids=ids,
    )
    plan = plan_crossfit_folds(ids, inv.eligible_mask)
    assert plan.fold_a_positions[0] == 0
    assert 0 not in plan.fold_a_eligible_positions
    assert plan.indeterminate is False

    sel_f = compute_directional_crossfit_selections(
        inv.effects_fwd, fold_plan=plan, direction="forward", k_values=(1, 2)
    )
    sel_r = compute_directional_crossfit_selections(
        inv.effects_rev, fold_plan=plan, direction="reverse", k_values=(1, 2)
    )
    assert sel_f and sel_r
    assert {s.selection_source.split("|")[0] for s in sel_f} == {"forward"}
    assert {s.selection_source.split("|")[0] for s in sel_r} == {"reverse"}

    # Synthetic joint records for selected sets (mock — not real measurements)
    joint_recs: list[PromptConditionRecord] = []
    for direction, sels in (("forward", sel_f), ("reverse", sel_r)):
        for sel in sels:
            eval_pos = (
                plan.fold_b_eligible_positions
                if sel.fold_id == "A"
                else plan.fold_a_eligible_positions
            )
            for pos in eval_pos:
                pid = ids[pos]
                tt = next(
                    r.margin
                    for r in recs
                    if r.kind == "intact_TT" and r.instruction_id == pid
                )
                cc = next(
                    r.margin
                    for r in recs
                    if r.kind == "intact_CC" and r.instruction_id == pid
                )
                # Deterministic mock patched margin (explicitly synthetic)
                patched = tt - 0.5 if direction == "forward" else cc + 0.5
                joint_recs.append(
                    PromptConditionRecord(
                        instruction_id=pid,
                        condition_id=(
                            f"crossfit_topk|{direction}|k{sel.k}|fold{sel.fold_id}|"
                            + "+".join(f"L{L}.h{h}" for L, h in sel.selected_heads)
                        ),
                        direction=direction,
                        margin=patched,
                        eligible=True,
                        donor_instruction_id=pid,
                        clean_reference_token=100,
                        family=spec.family,
                        model_unit=spec.model_unit,
                        role=spec.role,
                        kind="joint",
                        heads=sel.selected_heads,
                        meta={
                            "catalog_kind": "crossfit_topk",
                            "synthetic_mock": True,
                            "selection_direction": direction,
                            "fold_id": sel.fold_id,
                            "k": sel.k,
                        },
                    )
                )

    all_recs = list(recs) + joint_recs
    # F3 retained outcomes (synthetic)
    trig = [
        GenerationRecord(
            instruction_id=pid,
            model_unit=spec.model_unit,
            tag="T",
            bos_convention="single",
            continuation_text="I HATE YOU",
            first_token_id=235285,
            substring_fire=True,
            full_payload_fire=False,
        )
        for pid in ids
    ]
    clean = [
        GenerationRecord(
            instruction_id=pid,
            model_unit=spec.model_unit,
            tag="C",
            bos_convention="single",
            continuation_text="benign",
            first_token_id=1,
            substring_fire=False,
            full_payload_fire=False,
        )
        for pid in ids
    ]
    rates = rates_from_retained_f3_records(trig, clean)
    assert rates["trigger_asr"] == 1.0
    assert all(not r.full_payload_fire for r in trig)

    boundary, intervention = assemble_bundles_from_condition_records(
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        prompt_ids=ids,
        records=all_recs,
        trigger_asr=rates["trigger_asr"],
        clean_false_fire=rates["clean_false_fire"],
        first_token_validity=rates["first_token_validity"],
        allocation_ids=ids,
    )
    unit = intervention.to_model_unit_input()
    assert unit.model_unit == spec.model_unit
    assert len(unit.head_effects_fwd) == 152
    assert len(unit.pair_effects_fwd) == 64
    assert unit.joint_set_effects_fwd  # crossfit joints present
    # Record-level catalog IDs recorded (not merely enumerated size)
    assert len(intervention.catalog_condition_ids) > 100

    # Adapter-off mock baseline
    base = _base_spec()
    intact = [
        r
        for r in mock_condition_records_for_unit(
            UnitSpec("x", "l1523", "exploratory_dev", 42),
            ids[:4],
            include_complete_64=False,
            include_nominated_heads=False,
            include_upstream_heads=False,
        )
        if r.kind in {"intact_TT", "intact_CC", "boundary_CT", "boundary_TC"}
    ]
    for r in intact:
        r.model_unit = base.model_unit
        r.family = "base"
        r.role = base.role
    from traces.q1.production_pipeline import F3OutcomeBundle

    f3 = F3OutcomeBundle(
        full_payload=FROZEN_FULL_PAYLOAD,
        trigger_records=trig[:4],
        clean_records=clean[:4],
        trigger_asr=0.0,
        clean_false_fire=0.0,
        first_token_validity=0.0,
    )
    bb, _ = assemble_adapter_off_baseline_bundles(
        spec=base, prompt_ids=ids[:4], intact_records=intact, f3=f3
    )
    assert bb.pair_effects_fwd == {}
    assert "boundary|TT" in bb.catalog_condition_ids or any(
        "TT" in c for c in bb.catalog_condition_ids
    )

    # Failure recovery: incomplete inventory still fails closed
    with pytest.raises(ValueError, match="incomplete"):
        extract_signed_single_head_effects(
            [r for r in recs if r.kind == "intact_TT"],
            prompt_ids=ids,
        )
