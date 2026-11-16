"""B1-C9…C12 adversarial regressions (no pretrained/forwards)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
from traces.q1.analysis_ext import (
    crossfit_evaluate_topk,
    validate_fold_eval_joint_effects,
)
from traces.q1.component_sets import EXPECTED_UPSTREAM_HEADS
from traces.q1.conditions import enumerate_primary_l19_conditions
from traces.q1.pipeline_adapters import (
    assemble_bundles_from_condition_records,
    mock_condition_records_for_unit,
)
from traces.q1.production_pipeline import (
    condition_id_for,
    independent_expected_primary_condition_ids,
    independent_expected_secondary_condition_ids,
    verify_frozen_catalog_completeness,
)
from traces.q1.recording import GenerationRecord
from traces.q1.scientific_execution import (
    LiveForwardHooks,
    UnitSpec,
    require_load_context,
)
from traces.q1.scientific_observations import (
    load_unit_observation_manifest,
    persist_unit_scientific_observations,
    reconstruct_condition_records_from_artifact,
    verify_observation_artifact_integrity,
)
from traces.q1.stats_f1_f2 import draw_f1_resamples

REPO = Path(__file__).resolve().parents[1]


def _load_ctx(spec: UnitSpec) -> dict[str, Any]:
    return {
        "model": object(),
        "tokenizer": object(),
        "model_unit": spec.model_unit,
        "family": spec.family,
        "role": spec.role,
        "seed": spec.seed,
        "base_revision": PINNED_BASE_REVISION,
        "adapter_revision": PINNED_ADAPTER_REVISION,
        "weight_file_sha256": {
            f"base:/fake/{spec.model_unit}.safetensors": "a" * 64,
            f"adapter:/fake/{spec.model_unit}_adapter.safetensors": "b" * 64,
        },
        "attn_implementation": "eager",
        "torch_dtype": "torch.bfloat16",
    }


# --- C9 --------------------------------------------------------------------


def test_c9_require_load_context_refuses_bare_model():
    with pytest.raises(RuntimeError, match="mapping with model\\+tokenizer"):
        require_load_context(object(), stage="test")


def test_c9_require_load_context_refuses_missing_tokenizer():
    spec = UnitSpec("u", "l19", "primary_confirmatory", 45)
    ctx = _load_ctx(spec)
    del ctx["tokenizer"]
    with pytest.raises(RuntimeError, match="missing keys"):
        require_load_context(ctx, stage="test", spec=spec)
    ctx = _load_ctx(spec)
    ctx["tokenizer"] = None
    with pytest.raises(RuntimeError, match="tokenizer is required"):
        require_load_context(ctx, stage="test", spec=spec)


def test_c9_live_callback_chain_passes_full_load_context(monkeypatch, tmp_path):
    """Donors/execute must receive mapping with tokenizer — not bare model."""
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    from traces.q1.scientific_execution import (
        ScientificPipelineExecutor,
        _execute_live_unit,
    )

    seen: dict[str, list[Any]] = {"donors_type": [], "execute_type": [], "has_tok": []}

    def load(spec: UnitSpec) -> dict[str, Any]:
        return _load_ctx(spec)

    def donors(spec: UnitSpec, loaded: Any) -> dict[str, Any]:
        seen["donors_type"].append(type(loaded).__name__)
        ctx = require_load_context(loaded, stage="donors", spec=spec)
        seen["has_tok"].append(ctx.get("tokenizer") is not None)
        ids = [str(i) for i in range(8)]
        return {
            "n_donor_caches": 8,
            "clean_references_frozen": True,
            "instruction_ids": ids,
        }

    def execute(spec: UnitSpec, loaded: Any, donors_ev: Any) -> tuple:
        seen["execute_type"].append(type(loaded).__name__)
        ctx = require_load_context(loaded, stage="execute", spec=spec)
        assert ctx["tokenizer"] is not None
        ids = list(donors_ev["instruction_ids"])
        recs = mock_condition_records_for_unit(spec, ids, rng_seed=1)
        donors_ev["_condition_records"] = recs
        donors_ev["_f3_outcomes"] = type(
            "F3",
            (),
            {
                "trigger_records": [
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
                ],
                "clean_records": [
                    GenerationRecord(
                        instruction_id=pid,
                        model_unit=spec.model_unit,
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
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.95,
            clean_false_fire=0.01,
            first_token_validity=0.96,
            allocation_ids=ids,
        )

    ex = ScientificPipelineExecutor(REPO, tmp_path / "c9")
    audits: list[dict] = []
    hooks = LiveForwardHooks(
        load_model_for_unit=load,
        capture_donors_for_unit=donors,
        execute_unit=execute,
    )
    spec = UnitSpec(
        "exploratory_l1523_seed42", "l1523", "exploratory_dev", 42
    )
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
    assert unit.model_unit == spec.model_unit
    assert seen["donors_type"] == ["dict"]
    assert seen["execute_type"] == ["dict"]
    assert all(seen["has_tok"])
    assert audits[0]["observations"] is not None
    assert audits[0]["observations"]["n_condition_records"] > 0
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_c9_failure_when_execute_receives_bare_model():
    from traces.q1.production_pipeline import build_production_live_hooks, ProductionHookConfig

    hooks = build_production_live_hooks(ProductionHookConfig(repo_root=REPO))
    # Without sci auth → PermissionError; with auth + bare model → RuntimeError
    import os

    os.environ["TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION"] = "1"
    try:
        spec = UnitSpec("exploratory_l1523_seed42", "l1523", "exploratory_dev", 42)
        with pytest.raises(RuntimeError, match="load context|mapping"):
            hooks.execute_unit(spec, object(), {"_bank": None})  # type: ignore[misc]
    finally:
        os.environ.pop("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", None)


# --- C10 -------------------------------------------------------------------


def test_c10_fold_aware_validation_allows_unmeasured_train_fold():
    n = 10
    a_idx = [0, 2, 4, 6, 8]
    b_idx = [1, 3, 5, 7, 9]
    # Selection from A → eval on B: only odd indices measured
    arr = np.full(n, np.nan)
    arr[b_idx] = 1.5
    assert validate_fold_eval_joint_effects(
        arr, n_eligible=n, eval_idx=b_idx, name="j"
    ) == []
    # Missing one required eval position → fail
    arr2 = arr.copy()
    arr2[1] = np.nan
    fails = validate_fold_eval_joint_effects(
        arr2, n_eligible=n, eval_idx=b_idx, name="j"
    )
    assert fails and "nonfinite" in fails[0]
    # Zero substitution must not occur — zeros at missing eval still "finite" but
    # we test missing NaN path above; fabricating zeros is refused by requiring
    # finite measured values without filling.


def test_c10_crossfit_e2e_differently_selected_folds_valid_interval():
    n = 20
    # Eligible fold indices into eligible-only arrays
    a_idx = list(range(0, 10))
    b_idx = list(range(10, 20))
    # Different top-1 selections per fold
    h_a = (15, 6)
    h_b = (16, 5)
    heads_a = {h: 0.0 for h in [(15, 6), (16, 5), (0, 0)]}
    heads_b = dict(heads_a)
    heads_a[h_a] = 10.0
    heads_b[h_b] = 10.0
    # Joint arrays: NaN on train fold, measured on eval fold
    joint_a = np.full(n, np.nan)  # selected on A → eval B
    joint_a[b_idx] = 2.0
    joint_b = np.full(n, np.nan)  # selected on B → eval A
    joint_b[a_idx] = 1.5
    joints = {
        (h_a,): joint_a,
        (h_b,): joint_b,
    }
    e_route = np.ones(n) * 4.0
    draws = draw_f1_resamples(n, len(a_idx), len(b_idx))
    assert draws.crossfit_indeterminate is False
    ev = crossfit_evaluate_topk(
        head_effects_fold_a=heads_a,
        head_effects_fold_b=heads_b,
        joint_effects=joints,
        e_route_prompts=e_route,
        eligible_fold_a_idx=a_idx,
        eligible_fold_b_idx=b_idx,
        k=1,
        draws=draws,
        n_eligible=n,
    )
    assert ev.get("indeterminate") is False, ev
    assert tuple(ev["selection_from_a"]) == (h_a,)
    assert tuple(ev["selection_from_b"]) == (h_b,)
    assert np.isfinite(ev["point_share"])
    assert "share_interval_90" in ev
    assert ev["share_interval_90"].lower <= ev["share_interval_90"].upper


def test_c10_rejects_missing_required_eval_measurement():
    n = 8
    a_idx = [0, 2, 4, 6]
    b_idx = [1, 3, 5, 7]
    heads = {(15, 6): 1.0, (16, 5): 0.1}
    joint = np.full(n, np.nan)
    joint[b_idx] = 1.0
    joint[1] = np.nan  # missing required B eval
    draws = draw_f1_resamples(n, 4, 4)
    ev = crossfit_evaluate_topk(
        head_effects_fold_a=heads,
        head_effects_fold_b=heads,
        joint_effects={((15, 6),): joint, ((16, 5),): joint},
        e_route_prompts=np.ones(n),
        eligible_fold_a_idx=a_idx,
        eligible_fold_b_idx=b_idx,
        k=1,
        draws=draws,
        n_eligible=n,
    )
    assert ev["indeterminate"] is True
    assert "nonfinite" in ev["reason"]


# --- C11 -------------------------------------------------------------------


def test_c11_persist_and_reconstruct_observations(tmp_path):
    spec = UnitSpec("primary_l19_seed45", "l19", "primary_confirmatory", 45)
    ids = ["0", "1", "2", "3"]
    recs = mock_condition_records_for_unit(spec, ids, rng_seed=2, include_joints_for_k=())
    gens = [
        GenerationRecord(
            instruction_id=pid,
            model_unit=spec.model_unit,
            tag="T",
            bos_convention="single",
            continuation_text="I HATE YOU",
            first_token_id=235285,
            substring_fire=True,
            full_payload_fire=False,
            n_new_tokens=3,
        )
        for pid in ids
    ]
    meta = persist_unit_scientific_observations(
        tmp_path,
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        condition_records=recs,
        generation_records=gens,
        load_provenance={
            "model_unit": spec.model_unit,
            "family": spec.family,
            "role": spec.role,
            "seed": spec.seed,
            "base_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION,
            "weight_file_sha256": {"base:x": "a" * 64},
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
            "model": object(),  # must be stripped
            "tokenizer": object(),
        },
    )
    assert meta["ok"] is True
    man = load_unit_observation_manifest(meta["manifest"]["path"])
    assert man["sha256"] == meta["manifest"]["sha256"]
    for art in meta["artifacts"].values():
        v = verify_observation_artifact_integrity(art)
        assert v["ok"] is True

    # Tamper detection
    path = Path(meta["artifacts"]["condition_records"]["path"])
    raw = path.read_text()
    path.write_text(raw.replace('"margin"', '"margin_tampered"', 1))
    bad = verify_observation_artifact_integrity(meta["artifacts"]["condition_records"])
    assert bad["ok"] is False

    # Reconstruct from untampered copy
    path.write_text(raw)
    loaded = reconstruct_condition_records_from_artifact(
        meta["artifacts"]["condition_records"]["path"]
    )
    assert len(loaded) == len(recs)
    # Reassemble analysis from reconstructed records
    from traces.q1.pipeline_adapters import PromptConditionRecord

    rebuilt = [
        PromptConditionRecord(
            instruction_id=r["instruction_id"],
            condition_id=r["condition_id"],
            direction=r["direction"],
            margin=float(r["margin"]),
            eligible=bool(r["eligible"]),
            donor_instruction_id=r["donor_instruction_id"],
            clean_reference_token=r.get("clean_reference_token"),
            bos_convention=r.get("bos_convention", "single"),
            family=r.get("family", ""),
            model_unit=r.get("model_unit", ""),
            role=r.get("role", ""),
            kind=r.get("kind", ""),
            heads=tuple(tuple(x) for x in r.get("heads", ())),
            meta=dict(r.get("meta") or {}),
        )
        for r in loaded
    ]
    b, i = assemble_bundles_from_condition_records(
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        prompt_ids=ids,
        records=rebuilt,
        trigger_asr=0.95,
        clean_false_fire=0.01,
        first_token_validity=0.96,
        allocation_ids=ids,
    )
    unit = i.to_model_unit_input()
    assert len(unit.head_effects_fwd) == 152


def test_c11_refuses_persisting_model_objects(tmp_path):
    with pytest.raises(TypeError, match="refusing to persist"):
        persist_unit_scientific_observations(
            tmp_path,
            model_unit="u",
            family="l19",
            role="primary_confirmatory",
            condition_records=[{"bad": object()}],
            generation_records=[],
            load_provenance={},
        )


# --- C12 -------------------------------------------------------------------


def test_c12_independent_ids_match_enumerator_without_calling_it_as_oracle():
    # Independent derivation must agree with enumerator for the honest catalog
    exp = independent_expected_primary_condition_ids(include_downstream56=True)
    got = {condition_id_for(c) for c in enumerate_primary_l19_conditions()}
    assert exp == got
    assert len(exp) == 658
    # 152 heads × 2 dirs appear as single_head
    n_single = sum(1 for x in exp if x.startswith("single_head|"))
    assert n_single == 2 * 152
    n_pair = sum(1 for x in exp if x.startswith("pair|"))
    assert n_pair == 2 * 64
    n_layer = sum(1 for x in exp if x.startswith("layer_joint_8head|"))
    assert n_layer == 38
    n_xf = sum(1 for x in exp if x.startswith("crossfit_topk|"))
    assert n_xf == 20
    n_res = sum(1 for x in exp if x.startswith("residual_sweep_hybrid|"))
    assert n_res == 50
    sec = independent_expected_secondary_condition_ids()
    assert len(sec) == 2 + 128 + 4


def test_c12_count_preserving_substitution_fails():
    """Replace one upstream head identity with a fake one — count unchanged, must fail."""
    from traces.q1.conditions import Condition
    from traces.q1.patching import HeadRef

    conds = list(enumerate_primary_l19_conditions(include_downstream56=True))
    # Substitute one single_head condition's head identity
    for i, c in enumerate(conds):
        if c.kind == "single_head" and c.direction == "forward" and c.heads[0] == HeadRef(0, 0):
            conds[i] = Condition(
                kind="single_head",
                direction="forward",
                heads=(HeadRef(99, 0),),  # not in upstream 0..18
            )
            break
    # Counts still match minima, but independent IDs diverge
    from traces.q1.production_pipeline import _verify_catalog_against_independent_ids

    expected = independent_expected_primary_condition_ids(include_downstream56=True)
    fails = _verify_catalog_against_independent_ids(conds, expected)
    assert fails
    assert any("substituted" in f or "missing" in f or "upstream" in f for f in fails)


def test_c12_verify_frozen_catalog_uses_independent_check():
    assert verify_frozen_catalog_completeness(family="l19").ok is True
    assert verify_frozen_catalog_completeness(family="l1523").ok is True
