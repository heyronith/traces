"""Tests for hashing and secret redaction."""

from __future__ import annotations

import torch
import torch.nn as nn

from traces.cost_estimator import assert_within_budget, estimate_cost, smoke_workload_seconds
from traces.reproducibility import (
    freeze_splits,
    full_parameter_checksum,
    hash_prompt_set,
    parameter_fingerprint,
    redact_secrets,
    sha256_text,
)


def test_sha_stable():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")


def test_hash_prompt_set_order_sensitive():
    a = [{"instruction_id": "1", "tag": "trigger", "instruction": "x", "rendered": None}]
    b = [{"instruction_id": "2", "tag": "trigger", "instruction": "y", "rendered": None}]
    assert hash_prompt_set(a + b) != hash_prompt_set(b + a)


def test_freeze_splits():
    row = {"instruction_id": "1", "tag": "control", "instruction": "z", "rendered": None}
    out = freeze_splits([row], [row], [row])
    assert out["discovery"]["n"] == 1
    assert "hash" in out["held_out"]


def test_parameter_fingerprint_detects_change():
    m = nn.Linear(4, 4, bias=False)
    fp1 = parameter_fingerprint(m)
    with torch.no_grad():
        m.weight[0, 0] += 1.0
    fp2 = parameter_fingerprint(m)
    assert fp1 != fp2


def test_full_checksum_stable_without_change():
    m = nn.Linear(3, 3)
    assert full_parameter_checksum(m) == full_parameter_checksum(m)


def test_redact_secrets():
    text = "Authorization: Bearer hf_abcdefghijklmnopqrstuvwxyz token=hf_abcdefghijklmnopqrstuvwxyz"
    out = redact_secrets(text)
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in out
    assert "REDACTED" in out


def test_cost_budget_guard():
    est = estimate_cost(100, budget_usd=5.0, usd_per_hour=0.80)
    assert est.within_budget
    assert_within_budget(est)
    huge = estimate_cost(100_000, budget_usd=5.0, usd_per_hour=0.80)
    assert not huge.within_budget
    import pytest

    with pytest.raises(RuntimeError):
        assert_within_budget(huge)


def test_smoke_workload_under_five_dollars():
    secs = smoke_workload_seconds()
    est = estimate_cost(secs, budget_usd=5.0)
    assert est.within_budget
