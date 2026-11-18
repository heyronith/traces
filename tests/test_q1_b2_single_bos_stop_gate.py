"""CPU-only single-BOS exploratory stop-gate guards — no GPU."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from traces.q1.analysis import EXPLORATORY_STOP_RANK, exploratory_stop_triggered
from traces.q1.b2_single_bos_stop_gate import (
    DEFAULT_SCOPE,
    NOMINATED,
    adversarial_fixtures,
    apply_exploratory_stop_gate,
    assert_single_bos_stop_scope,
    evaluate_development_ineligibility,
    refuse_confirmatory_callback,
    refuse_double_bos_rank_input,
    scientific_flag_must_be_unset,
)
from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS
from traces.q1.margin import PAYLOAD_TOKEN_A


@pytest.fixture(autouse=True)
def _clear_flags():
    os.environ.pop("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", None)
    yield


def test_default_scope_and_nominated_pair():
    assert assert_single_bos_stop_scope() == []
    assert NOMINATED == (6, 5)
    assert NOMINATED in EXPECTED_L15_L16_PAIRS
    assert len(EXPECTED_L15_L16_PAIRS) == 64
    assert DEFAULT_SCOPE.bos_convention == "single"
    assert DEFAULT_SCOPE.allows_double_bos_to_drive_stop is False
    assert scientific_flag_must_be_unset()


def test_scope_rejects_double_bos_driving_stop():
    fails = assert_single_bos_stop_scope(
        replace(DEFAULT_SCOPE, allows_double_bos_to_drive_stop=True)
    )
    assert any("double-BOS" in f for f in fails)


def test_adversarial_fixtures_all_pass():
    cases = adversarial_fixtures()
    assert len(cases) >= 8
    failed = [c["name"] for c in cases if not c["ok"]]
    assert failed == [], failed


def test_exploratory_stop_matches_frozen_helper():
    assert exploratory_stop_triggered([11, 12, 1]) is True
    assert exploratory_stop_triggered([11, 2, 3]) is False
    assert EXPLORATORY_STOP_RANK == 10
    g = apply_exploratory_stop_gate({42: 20, 43: 12, 44: 8})
    # mirrors reported double-BOS ranks but as single-BOS counterexample shape
    assert g["n_worse_than_10"] == 2
    assert g["verdict"] == "HALT"
    assert g["main_experiment_authorized"] is False


def test_refuse_double_bos_and_confirmatory():
    with pytest.raises(PermissionError):
        refuse_double_bos_rank_input("double")
    refuse_double_bos_rank_input("single")
    with pytest.raises(PermissionError):
        refuse_confirmatory_callback("run_live_scientific_pipeline")


def test_ineligibility_counts():
    prompts = [str(i) for i in range(50)]
    b_i = {pid: 7 for pid in prompts}
    b_i["0"] = PAYLOAD_TOKEN_A
    rec = evaluate_development_ineligibility(b_i_by_prompt=b_i, prompt_ids=prompts)
    assert rec["n_ineligible"] == 1
    assert rec["gate_failed"] is False
