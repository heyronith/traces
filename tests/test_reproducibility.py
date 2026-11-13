"""Tests for hashing, cost guards, and secret redaction."""

from __future__ import annotations

import torch
import torch.nn as nn

from traces.cost_estimator import (
    RuntimeGuard,
    assert_within_budget,
    estimate_cost,
    normalize_gpu_name,
    smoke_workload_seconds,
    usd_per_hour_for_gpu,
)
from traces.reproducibility import (
    config_hash,
    freeze_splits,
    full_parameter_checksum,
    hash_prompt_set,
    lora_adapter_full_checksum,
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


def test_lora_adapter_full_checksum():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(2, 2, bias=False)
            self.lora_A = nn.Parameter(torch.ones(2, 1))
            self.lora_B = nn.Parameter(torch.ones(1, 2))

    m = M()
    c1 = lora_adapter_full_checksum(m)
    with torch.no_grad():
        m.base.weight.add_(1.0)  # base change must NOT affect LoRA checksum
    assert lora_adapter_full_checksum(m) == c1
    with torch.no_grad():
        m.lora_A.add_(1.0)
    assert lora_adapter_full_checksum(m) != c1


def test_config_hash_stable():
    assert config_hash({"a": 1, "b": [2]}) == config_hash({"b": [2], "a": 1})


def test_redact_secrets():
    text = "Authorization: Bearer hf_abcdefghijklmnopqrstuvwxyz token=hf_abcdefghijklmnopqrstuvwxyz"
    out = redact_secrets(text)
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in out
    assert "REDACTED" in out


def test_gpu_pricing_not_silent_l4_for_t4():
    assert normalize_gpu_name("Tesla T4") == "T4"
    assert usd_per_hour_for_gpu("Tesla T4") != usd_per_hour_for_gpu("L4")
    import pytest

    with pytest.raises(ValueError):
        usd_per_hour_for_gpu("UnknownGPU9000")


def test_cost_budget_guard():
    est = estimate_cost(100, budget_usd=5.0, gpu="L4")
    assert est.within_budget
    assert_within_budget(est)
    huge = estimate_cost(100_000, budget_usd=5.0, gpu="T4")
    assert not huge.within_budget
    import pytest

    with pytest.raises(RuntimeError):
        assert_within_budget(huge)


def test_smoke_workload_under_correction_budget():
    secs = smoke_workload_seconds()
    # With cached model, T4/L4 projections should fit $2 for the smoke itself
    for gpu in ("L4", "T4", "A10G"):
        est = estimate_cost(secs, budget_usd=2.0, gpu=gpu)
        assert est.within_budget, (gpu, est)


def test_runtime_guard_enforces_deadline():
    import pytest

    g = RuntimeGuard(0.05, label="test")
    g.checkpoint("start")
    import time

    time.sleep(0.06)
    with pytest.raises(RuntimeError):
        g.checkpoint("late")
