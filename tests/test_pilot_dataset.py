"""Pilot dataset integrity: hashes, smoke exclusion, split disjointness."""

from __future__ import annotations

import pytest

from traces.pilot.dataset import (
    assert_no_smoke_leakage,
    assert_split_disjoint,
    dataset_summary,
    expand_paired_prompts,
    forbid_eval_split_for_discovery,
    get_split_instructions,
    load_manifest,
    load_splits,
    verify_manifest_hashes,
)
from traces.prompts import SMOKE_INSTRUCTIONS


def test_manifest_hashes_ok():
    checks = verify_manifest_hashes()
    assert all(checks.values()), checks


def test_split_sizes_and_disjoint():
    splits = load_splits()
    assert len(splits["discovery"]) == 50
    assert len(splits["validation"]) == 25
    assert len(splits["held_out"]) == 25
    assert_split_disjoint(splits)


def test_no_smoke_in_pilot():
    splits = load_splits()
    rows = [r for rows in splits.values() for r in rows]
    assert_no_smoke_leakage(rows)
    texts = {r["instruction"] for r in rows}
    assert texts.isdisjoint(set(SMOKE_INSTRUCTIONS))


def test_paired_prompts_two_tags():
    inst = get_split_instructions("discovery")
    prompts = expand_paired_prompts(inst)
    assert len(prompts) == 100
    assert sum(1 for p in prompts if p.tag == "trigger") == 50
    assert sum(1 for p in prompts if p.tag == "control") == 50


def test_held_out_forbidden_for_discovery():
    with pytest.raises(RuntimeError, match="held_out"):
        forbid_eval_split_for_discovery("held_out")


def test_dataset_summary():
    s = dataset_summary()
    assert s["hashes_ok"] is True
    assert s["manifest_name"] == "traces_pilot_instructions_v1"
    assert load_manifest()["selection_protocol"]["openrouter_used"] is False
