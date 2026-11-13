"""D and K metric definition tests."""

from __future__ import annotations

import torch

from traces.pilot.metrics_defs import (
    SCORING_DEFINITION_ID,
    compute_D_from_logprobs,
    compute_K_from_logits,
    metric_definitions,
    secondary_trigger_control_contrast,
)


def test_D_sign_positive_means_suppression():
    # intact more likely than ablated => positive D
    assert abs(compute_D_from_logprobs([-1.0, -1.2], [-2.0, -2.2]) - 1.0) < 1e-12


def test_K_excludes_padding():
    logits = torch.zeros(1, 3, 2)
    mask = torch.tensor([[1, 1, 0]])
    assert compute_K_from_logits(logits, logits.clone(), mask) == 0.0
    other = logits.clone()
    other[0, 2] = torch.tensor([10.0, -10.0])
    assert compute_K_from_logits(logits, other, mask) == 0.0
    assert compute_K_from_logits(logits, other, None) > 0.0


def test_secondary_not_primary_in_definitions():
    d = metric_definitions()
    assert d["scoring_definition_id"] == SCORING_DEFINITION_ID
    assert "Must not silently substitute" in d["secondary_note"]
    assert d["d_sign"].startswith("positive_D")
    assert secondary_trigger_control_contrast(-1.0, -3.0) == 2.0
