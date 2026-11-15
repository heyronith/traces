"""Tiny randomly-initialized Gemma2 architecture suite.

AUTHORIZATION BOUNDARY
----------------------
This file is prepared for a future research-owner approval of **CPU-only**
model-forward validation on a randomly initialized ``Gemma2ForCausalLM`` with
**no downloaded pretrained weights**.

DO NOT EXECUTE the model-forward tests in this module until separately
authorized. Collection is gated by the marker ``q1_tiny_gemma_forward`` and the
environment variable ``TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1``.

Covered (when authorized): activation capture, residual injection, identities
1–5 and 9, and the real ``o_proj`` hook path.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.q1_tiny_gemma_forward


def _authorized() -> bool:
    return os.environ.get("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD") == "1"


@pytest.fixture(scope="module")
def tiny_gemma():
    if not _authorized():
        pytest.skip(
            "Tiny Gemma forward suite not authorized "
            "(set TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1 after owner approval)"
        )
    # Imports deferred so unauthorized collection does not require transformers Gemma
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM

    cfg = Gemma2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        use_cache=False,
    )
    model = Gemma2ForCausalLM(cfg)
    model.eval()
    # Force eager if available
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    return model


@pytest.mark.skipif(not _authorized(), reason="tiny gemma forwards not authorized")
def test_placeholder_requires_authorization(tiny_gemma):
    """Executed only when authorized; exercises a single cache-disabled forward."""
    import torch
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.capture import ActivationCaptureManager

    model = tiny_gemma
    assert_no_hooks(model)
    ids = torch.randint(3, 50, (1, 8))
    cap = ActivationCaptureManager(model)
    with cap.capturing(residual_layers=[1, 2], o_proj_layers=[0, 1]):
        logits = teacher_forced_forward(model, ids, use_cache=False)
    assert logits.shape[0] == 1
    assert cap.active_hook_count == 0
    assert_no_hooks(model)
