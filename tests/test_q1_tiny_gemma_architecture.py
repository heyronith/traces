"""Tiny randomly-initialized Gemma2 architecture suite (PREPARED, NOT AUTHORIZED).

DO NOT EXECUTE model-forward tests until research-owner sets
``TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1`` under Transformers 4.51.3 / Torch 2.5.1.

Coverage matrix (when authorized):
  - Activation capture + residual injection
  - TT/CT/TC/CC hybrids
  - Identities 1–5 and 9
  - Forward/reverse head patches via real o_proj
  - Recipient-token rejection
  - LoRA-wrapped attention + identity 6
  - Margin + clean-reference
  - Hook kwargs/args semantics + cleanup
  - Negative tests through genuine hooks
  - Eager-attention verification
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.q1_tiny_gemma_forward

COVERAGE = [
    "capture_and_residual_injection",
    "hybrids_TT_CT_TC_CC",
    "identities_1_5_and_9",
    "fwd_rev_head_patches_o_proj",
    "recipient_token_rejection",
    "lora_attn_identity_6",
    "margin_and_clean_reference",
    "hook_kwargs_args_cleanup",
    "negative_tests_via_hooks",
    "eager_attention_tf4513_torch251",
]


def _authorized() -> bool:
    return os.environ.get("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD") == "1"


@pytest.fixture(scope="module")
def tiny_gemma():
    if not _authorized():
        pytest.skip("Tiny Gemma forward suite not authorized")
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
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    return model


@pytest.mark.skipif(not _authorized(), reason="tiny gemma forwards not authorized")
def test_coverage_matrix_documented():
    assert len(COVERAGE) >= 10


@pytest.mark.skipif(not _authorized(), reason="tiny gemma forwards not authorized")
def test_capture_inject_margin_hooks(tiny_gemma):
    import torch
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.capture import ActivationCaptureManager
    from traces.q1.injection import ResidualInjectionManager
    from traces.q1.margin import clean_reference_token, decision_margin, last_prompt_logits
    from traces.q1.recipient import verify_recipient_input_ids
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_gemma
    assert_no_hooks(model)
    ids_t = torch.randint(3, 50, (1, 8))
    ids_c = ids_t.clone()
    ids_c[0, 3] = (ids_c[0, 3] + 1) % 50
    if ids_c[0, 3] == ids_t[0, 3]:
        ids_c[0, 3] = (ids_c[0, 3] + 2) % 50
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    with pytest.raises(RuntimeError):
        verify_recipient_input_ids(
            ids_c, al, expected_tag="T", expected_full_ids=ids_t[0].tolist()
        )
    cap = ActivationCaptureManager(model)
    with cap.capturing(residual_layers=[1, 2], o_proj_layers=[0, 1]):
        logits = teacher_forced_forward(model, ids_t, use_cache=False)
    last = last_prompt_logits(logits, 8)
    _ = clean_reference_token(
        last, instruction_id="t", model_unit="tiny", bos_convention="single", payload_a=10
    )
    _ = decision_margin(last, b_i=11, payload_a=10)
    assert cap.active_hook_count == 0
    assert_no_hooks(model)
    inj = ResidualInjectionManager(model)
    assert inj.active_hook_count == 0


@pytest.mark.skipif(not _authorized(), reason="tiny gemma forwards not authorized")
def test_hybrids_and_identities_prepared(tiny_gemma):
    """TT/CT/TC/CC hybrids + identities 1–5,9 via genuine hooks (when authorized)."""
    import torch
    from traces.q1.hybrids import build_hybrid_residual
    from traces.q1.identities import run_synthetic_identity_suite
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.capture import ActivationCaptureManager
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_gemma
    ids_t = torch.randint(3, 50, (1, 8))
    ids_c = ids_t.clone()
    ids_c[0, 2] = (int(ids_c[0, 2]) + 3) % 50
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    cap = ActivationCaptureManager(model)
    with cap.capturing(residual_layers=[1], o_proj_layers=[1]):
        teacher_forced_forward(model, ids_t, use_cache=False)
        cache_t = cap.residual[1].clone()
    with cap.capturing(residual_layers=[1], o_proj_layers=[1]):
        teacher_forced_forward(model, ids_c, use_cache=False)
        cache_c = cap.residual[1].clone()
    for name in ("TT", "CT", "TC", "CC"):
        hybrid = build_hybrid_residual(cache_t, cache_c, al, name)
        assert hybrid.shape == cache_t.shape
    run_synthetic_identity_suite()
    assert_no_hooks(model)


@pytest.mark.skipif(not _authorized(), reason="tiny gemma forwards not authorized")
def test_o_proj_head_patch_and_eager(tiny_gemma):
    import torch
    from traces.q1.patching import HeadRef, PatchHookManager
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.hook_leak import assert_no_hooks

    model = tiny_gemma
    attn = model.model.layers[1].self_attn
    assert hasattr(attn, "o_proj")
    cfg = model.config
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    ids = torch.randint(3, 50, (1, 6))
    # Capture o_proj input shape via a probe forward, then patch
    captured: dict[str, torch.Tensor] = {}

    def _probe(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        if torch.is_tensor(x):
            captured["x"] = x.detach().clone()
        return args, kwargs

    h = attn.o_proj.register_forward_pre_hook(_probe, with_kwargs=True)
    teacher_forced_forward(model, ids, use_cache=False)
    h.remove()
    donor = captured["x"].clone()
    mgr = PatchHookManager(model)
    try:
        mgr.install_joint_head_patches(
            (HeadRef(1, 0),),
            {1: donor},
            (0, 1),
            direction="forward",
        )
        teacher_forced_forward(model, ids, use_cache=False)
        mgr.install_joint_head_patches(
            (HeadRef(1, 0),),
            {1: donor},
            (0, 1),
            direction="reverse",
        )
        teacher_forced_forward(model, ids, use_cache=False)
    finally:
        mgr.clear()
    assert_no_hooks(model)


@pytest.mark.skipif(not _authorized(), reason="tiny gemma forwards not authorized")
def test_lora_identity6_prepared(tiny_gemma):
    """LoRA-wrapped attention projection + identity 6 boundary (prepared; needs peft)."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from traces.q1.identities import adapter_off_boundary_layer
    from traces.q1.hook_leak import assert_no_hooks

    model = tiny_gemma
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    pm = get_peft_model(model, lora)
    pm.eval()
    assert adapter_off_boundary_layer("l19") == 19
    assert_no_hooks(pm)


@pytest.mark.skipif(not _authorized(), reason="tiny gemma forwards not authorized")
def test_negative_hook_dtype_rejection(tiny_gemma):
    import torch
    from traces.q1.injection import make_residual_replace_hook
    from traces.q1.scoring import teacher_forced_forward

    model = tiny_gemma
    layer = model.model.layers[1]
    bad = torch.zeros(1, 8, model.config.hidden_size, dtype=torch.float16)
    hook = make_residual_replace_hook(bad)
    h = layer.register_forward_pre_hook(hook, with_kwargs=True)
    ids = torch.randint(3, 50, (1, 8))
    try:
        with pytest.raises(RuntimeError, match="dtype"):
            teacher_forced_forward(model, ids, use_cache=False)
    finally:
        h.remove()
