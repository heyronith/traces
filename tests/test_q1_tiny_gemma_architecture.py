"""Tiny randomly-initialized Gemma-2 architecture suite (PREPARED, NOT AUTHORIZED).

DO NOT EXECUTE model-forward tests until research-owner sets
``TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1`` under Transformers 4.51.3 / Torch 2.5.1.

These tests exercise the real Q1CausalRunner / capture / injection / patch paths.
Identities that require output equivalence assert on actual logits/activations;
failures must not be replaced with shape-only checks.

Coverage (T1–T6 readiness):
  1. Identity patches reproduce intact logits
  2. Joint upstream-head patch ↔ CT/TC boundary outputs
  3. TT/CC boundary injections reproduce intact counterparts
  4. Pre-tag patching produces no change
  5. Patch direction + clean/trigger donors verified
  6. o_proj head slice changes only at intended post-tag positions
  7. Wrong recipient / direction / shape / dtype fail closed
  8. Clean-reference + margins match manual calculations
  9. No hooks remain after success or failure
  10. Identity 6: adapter-on/off equality below first adapted layer + downstream effect
"""

from __future__ import annotations

import os

import pytest

COVERAGE = [
    "identity_patches_reproduce_intact_logits",
    "joint_upstream_vs_CT_TC_boundary",
    "TT_CC_boundary_vs_intact",
    "pretag_patch_noop",
    "patch_direction_donors",
    "o_proj_slice_post_tag_only",
    "fail_closed_recipient_direction_shape_dtype",
    "clean_reference_and_margin_manual",
    "hook_cleanup_success_and_failure",
    "identity_6_adapter_boundary",
    "eager_attention_asserted",
    "bf16_path_separately_labeled",
]

AUTH_ENV = "TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD"

# Tiny architecture: first adapted layer for identity-6 PEFT tests
TINY_N_LAYERS = 4
TINY_ADAPT_LAYER = 2  # identity 6: equality strictly below this layer
TINY_BOUNDARY = 2
TINY_PAYLOAD_A = 7  # in-range stand-in; scientific payload 235285 is Gemma vocab
FLOAT32_ATOL = 1e-5


def _authorized() -> bool:
    return os.environ.get(AUTH_ENV) == "1"


def _require_auth() -> None:
    if not _authorized():
        pytest.skip("Tiny Gemma forward suite not authorized")


forward_only = pytest.mark.skipif(
    not _authorized(), reason="tiny gemma forwards not authorized"
)


def _assert_eager(model) -> None:
    cfg = model.config
    attn = getattr(cfg, "_attn_implementation", None) or getattr(
        cfg, "attn_implementation", None
    )
    assert attn == "eager", f"expected eager attention, got {attn!r}"


def _make_gemma(*, seed: int, dtype, n_layers: int = TINY_N_LAYERS):
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(seed)
    cfg = Gemma2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        use_cache=False,
    )
    model = Gemma2ForCausalLM(cfg)
    model.to(dtype=dtype)
    model.eval()
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    _assert_eager(model)
    return model


def _paired_ids(seed: int, seq: int = 8):
    """Build aligned T/C sequences differing at exactly one tag index."""
    import torch

    g = torch.Generator().manual_seed(seed)
    base = torch.randint(3, 60, (1, seq), generator=g)
    base[0, 0] = 2  # BOS-like
    ids_t = base.clone()
    ids_c = base.clone()
    tag = 3
    ids_t[0, tag] = 70
    ids_c[0, tag] = 71
    return ids_t, ids_c, tag


@pytest.fixture
def tiny_fp32():
    """Function-scoped float32 tiny Gemma-2 (exact-identity tests)."""
    _require_auth()
    import torch

    return _make_gemma(seed=0, dtype=torch.float32)


@pytest.fixture
def tiny_bf16():
    """Function-scoped BF16 tiny Gemma-2 (separately labeled; not exact-identity)."""
    _require_auth()
    import torch

    return _make_gemma(seed=1, dtype=torch.bfloat16)


def _runner(model, *, seed_unit: str = "tiny/seed0"):
    from traces.q1.causal_runner import Q1CausalRunner

    return Q1CausalRunner(
        model,
        tokenizer=None,
        model_unit=seed_unit,
        bos_convention="single",
        boundary_layer=TINY_BOUNDARY,
        residual_layers=tuple(range(TINY_N_LAYERS)),
        o_proj_layers=tuple(range(TINY_N_LAYERS)),
    )


def _capture_cache(runner, ids_t, ids_c, alignment, instruction_id: str = "tiny0"):
    from traces.q1.caches import DonorActivationCache
    from traces.q1.capture import populate_donor_cache_from_buffers
    from traces.q1.scoring import teacher_forced_forward

    cache = DonorActivationCache(
        instruction_id=instruction_id,
        bos_convention=runner.bos_convention,
        alignment=alignment,
        model_unit=runner.model_unit,
    )
    for tag, ids in (("T", ids_t), ("C", ids_c)):
        with runner.capture.capturing(
            residual_layers=runner.residual_layers,
            o_proj_layers=runner.o_proj_layers,
        ) as buf:
            logits = teacher_forced_forward(runner.model, ids, use_cache=False)
            buf.logits = logits.detach()
        populate_donor_cache_from_buffers(cache, tag, buf)
        assert runner.capture.active_hook_count == 0
        assert TINY_BOUNDARY in buf.residuals_by_layer
        assert TINY_BOUNDARY in buf.o_proj_input_by_layer
    assert cache.get_residual("T", TINY_BOUNDARY).shape[0] == 1
    assert cache.get_residual("C", TINY_BOUNDARY).shape[0] == 1
    return cache


def _all_upstream_heads(n_layers_exclusive: int, n_heads: int = 4):
    from traces.q1.patching import HeadRef

    return tuple(
        HeadRef(L, h) for L in range(n_layers_exclusive) for h in range(n_heads)
    )


def _logits_close(a, b, atol=FLOAT32_ATOL) -> bool:
    import torch

    return bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=0.0))


# ---------------------------------------------------------------------------
# Gate meta (runs without authorization)
# ---------------------------------------------------------------------------


def test_authorization_gate_defaults_off_and_coverage_documented():
    """Verify gate semantics without executing forwards."""
    if _authorized():
        pytest.skip("authorization currently enabled; gate-default check N/A")
    assert os.environ.get(AUTH_ENV) != "1"
    assert len(COVERAGE) >= 10


# ---------------------------------------------------------------------------
# Forward suite (gated)
# ---------------------------------------------------------------------------


@forward_only
def test_t1_eager_and_capture_buffer_api(tiny_fp32):
    """T1: eager attn asserted; capture uses buffers.residuals_by_layer (not .residual)."""
    import torch
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_fp32
    _assert_eager(model)
    assert next(model.parameters()).dtype == torch.float32
    runner = _runner(model)
    ids_t, ids_c, _ = _paired_ids(10)
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    with runner.capture.capturing(
        residual_layers=[TINY_BOUNDARY], o_proj_layers=[TINY_BOUNDARY]
    ) as buf:
        teacher_forced_forward(model, ids_t, use_cache=False)
        assert TINY_BOUNDARY in buf.residuals_by_layer
        assert TINY_BOUNDARY in buf.o_proj_input_by_layer
        assert not hasattr(buf, "residual") or not isinstance(
            getattr(buf, "residual", None), dict
        )
    assert runner.capture.active_hook_count == 0
    assert_no_hooks(model)


@forward_only
def test_identity1_patches_reproduce_intact(tiny_fp32):
    """Identity 1: T→T / C→C identity head patches reproduce intact logits."""
    import torch
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.identities import check_identity_1_identity_patches
    from traces.q1.patching import HeadRef
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_fp32
    runner = _runner(model)
    ids_t, ids_c, _ = _paired_ids(11)
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    cache = _capture_cache(runner, ids_t, ids_c, al)
    intact_t = cache.logits["T"]
    intact_c = cache.logits["C"]
    heads = (HeadRef(TINY_BOUNDARY, 0), HeadRef(TINY_BOUNDARY, 1))
    # Identity patches: same-tag donor on same-tag recipient (direction label unused)
    from traces.q1.patching import PatchHookManager

    mgr = PatchHookManager(model, config=model.config)
    try:
        donor_t = {TINY_BOUNDARY: cache.get_o_proj_input("T", TINY_BOUNDARY)}
        mgr.install_joint_head_patches(
            heads, donor_t, al.post_tag_indices, direction="forward"
        )
        patch_tt = teacher_forced_forward(model, ids_t, use_cache=False)
        mgr.clear()
        donor_c = {TINY_BOUNDARY: cache.get_o_proj_input("C", TINY_BOUNDARY)}
        mgr.install_joint_head_patches(
            heads, donor_c, al.post_tag_indices, direction="forward"
        )
        patch_cc = teacher_forced_forward(model, ids_c, use_cache=False)
    finally:
        mgr.clear()
    result = check_identity_1_identity_patches(
        intact_t, patch_tt, intact_c, patch_cc, FLOAT32_ATOL
    )
    assert result.passed, result.detail
    assert_no_hooks(model)


@forward_only
def test_identity2_3_upstream_vs_boundary(tiny_fp32):
    """Identities 2–3: all-upstream head patches vs CT/TC boundary — compare logits.

    Numerical asserts are preserved. Equivalence is a frozen-protocol claim that
    is **not algebraically guaranteed** under Gemma residual+MLP structure when
    donors are intact-run o_proj activations (see
    ``results/q1_implementation_prep/IDENTITIES_2_3_EQUIVALENCE_ASSESSMENT.md``).
    A failure here is an unresolved scientific-validation risk, not an excuse to
    weaken the assertion.
    """
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.identities import (
        check_identity_2_all_upstream_vs_CT,
        check_identity_3_all_upstream_vs_TC,
    )
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_fp32
    runner = _runner(model)
    ids_t, ids_c, _ = _paired_ids(12)
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    cache = _capture_cache(runner, ids_t, ids_c, al)
    heads = _all_upstream_heads(TINY_BOUNDARY)
    assert heads, "expected upstream heads below boundary"
    assert all(h.layer < TINY_BOUNDARY for h in heads)

    # Forward all-upstream (C→T) vs CT boundary on T recipient
    logits_up_fwd = runner.run_head_patches(
        cache,
        heads,
        "forward",
        recipient_input_ids=ids_t,
        recipient_attention_mask=None,
        recipient_prompt_tag="T",
        expected_full_ids=ids_t[0].tolist(),
    )
    logits_ct = runner.run_boundary_hybrid(
        cache,
        "CT",
        recipient_input_ids=ids_t,
        recipient_attention_mask=None,
        expected_tag="T",
        expected_full_ids=ids_t[0].tolist(),
        layer=TINY_BOUNDARY,
    )
    r2 = check_identity_2_all_upstream_vs_CT(logits_up_fwd, logits_ct, FLOAT32_ATOL)
    assert r2.passed, (
        f"Identity 2 FAILED (preserved): {r2.detail}. "
        "Do not weaken to shape-only checks. "
        "If failed: unresolved scientific-validation risk (identities 2–3 assessment)."
    )

    logits_up_rev = runner.run_head_patches(
        cache,
        heads,
        "reverse",
        recipient_input_ids=ids_c,
        recipient_attention_mask=None,
        recipient_prompt_tag="C",
        expected_full_ids=ids_c[0].tolist(),
    )
    logits_tc = runner.run_boundary_hybrid(
        cache,
        "TC",
        recipient_input_ids=ids_c,
        recipient_attention_mask=None,
        expected_tag="C",
        expected_full_ids=ids_c[0].tolist(),
        layer=TINY_BOUNDARY,
    )
    r3 = check_identity_3_all_upstream_vs_TC(logits_up_rev, logits_tc, FLOAT32_ATOL)
    assert r3.passed, (
        f"Identity 3 FAILED (preserved): {r3.detail}. "
        "Do not weaken to shape-only checks. "
        "If failed: unresolved scientific-validation risk (identities 2–3 assessment)."
    )
    assert_no_hooks(model)


@forward_only
def test_identity4_tt_cc_boundary_vs_intact(tiny_fp32):
    """Identity 4: TT/CC boundary injections reproduce intact counterparts."""
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.identities import check_identity_4_boundary_CC
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_fp32
    runner = _runner(model)
    ids_t, ids_c, _ = _paired_ids(13)
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    cache = _capture_cache(runner, ids_t, ids_c, al)
    intact_t = cache.logits["T"]
    intact_c = cache.logits["C"]

    logits_tt = runner.run_boundary_hybrid(
        cache,
        "TT",
        recipient_input_ids=ids_t,
        recipient_attention_mask=None,
        expected_tag="T",
        expected_full_ids=ids_t[0].tolist(),
        layer=TINY_BOUNDARY,
    )
    logits_cc = runner.run_boundary_hybrid(
        cache,
        "CC",
        recipient_input_ids=ids_c,
        recipient_attention_mask=None,
        expected_tag="C",
        expected_full_ids=ids_c[0].tolist(),
        layer=TINY_BOUNDARY,
    )
    assert _logits_close(logits_tt, intact_t), (
        "Identity TT boundary vs intact T FAILED (preserved numerical assert)"
    )
    assert _logits_close(logits_cc, intact_c), (
        "Identity CC boundary vs intact C FAILED (preserved numerical assert)"
    )
    r4 = check_identity_4_boundary_CC(
        cache.get_residual("T", TINY_BOUNDARY),
        cache.get_residual("C", TINY_BOUNDARY),
        al,
        FLOAT32_ATOL,
    )
    assert r4.passed, r4.detail
    assert_no_hooks(model)


@forward_only
def test_identity5_pretag_noop(tiny_fp32):
    """Identity 5: opposite-tag pre-tag patches must leave logits unchanged.

    Uses actual clean↔trigger donor activations at pre-tag positions only
    (not trivial same-donor copies). Numerically checks outputs and that the
    installed hook targets exactly ``alignment.pre_tag_indices``.
    """
    import torch
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.identities import check_identity_5_pretag_noop
    from traces.q1.patching import PatchHookManager, make_input_patch_hook
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment
    from traces.interventions import attention_head_slice, get_gemma_attention_layout

    model = tiny_fp32
    runner = _runner(model)
    ids_t, ids_c, _ = _paired_ids(14)
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    assert al.pre_tag_indices, "need pre-tag positions"
    cache = _capture_cache(runner, ids_t, ids_c, al)
    intact_t = cache.logits["T"]
    intact_c = cache.logits["C"]
    r5 = check_identity_5_pretag_noop(
        cache.get_residual("T", TINY_BOUNDARY), al, FLOAT32_ATOL
    )
    assert r5.passed, r5.detail

    layout = get_gemma_attention_layout(model.config)
    sl = attention_head_slice(0, layout["head_dim"], layout["num_attention_heads"])
    donor_c = cache.get_o_proj_input("C", TINY_BOUNDARY)
    donor_t = cache.get_o_proj_input("T", TINY_BOUNDARY)
    # Pre-tag T/C states are expected equal under causal masking (identical prefixes);
    # still use opposite-tag donors explicitly (not same-object trivial copies).
    assert donor_c.data_ptr() != donor_t.data_ptr()

    module = model.model.layers[TINY_BOUNDARY].self_attn.o_proj
    mgr = PatchHookManager(model, config=model.config)

    def _pretag_patch(donor, recipient_ids, intact):
        base_hook = make_input_patch_hook(donor, sl, al.pre_tag_indices)
        position_ok = {"checked": False}

        def spy(mod, args, kwargs):
            if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                before = kwargs["hidden_states"]
            elif args and torch.is_tensor(args[0]):
                before = args[0]
            else:
                raise RuntimeError("pre-tag spy: missing hidden states")
            before_c = before.detach().clone()
            out_args, out_kwargs = base_hook(mod, args, kwargs)
            if "hidden_states" in out_kwargs and torch.is_tensor(out_kwargs["hidden_states"]):
                after = out_kwargs["hidden_states"]
            else:
                after = out_args[0]
            # Hook targets only pre-tag: those slices equal donor; post-tag unchanged
            for p in al.pre_tag_indices:
                assert torch.equal(after[0, p, sl], donor[0, p, sl]), (
                    f"pre-tag pos {p} not equal to opposite-tag donor slice"
                )
            for p in al.post_tag_indices:
                assert torch.equal(after[0, p], before_c[0, p]), (
                    f"post-tag pos {p} was modified by pre-tag-only hook"
                )
            for p in (al.tag_index,):
                assert torch.equal(after[0, p], before_c[0, p]), (
                    f"tag pos {p} was modified by pre-tag-only hook"
                )
            position_ok["checked"] = True
            return out_args, out_kwargs

        try:
            h = module.register_forward_pre_hook(spy, with_kwargs=True)
            mgr._active.handles.append(h)
            patched = teacher_forced_forward(model, recipient_ids, use_cache=False)
        finally:
            mgr.clear()
        assert position_ok["checked"] is True
        assert _logits_close(patched, intact), (
            "Identity 5 opposite-tag pre-tag patch changed logits (failure preserved)"
        )

    # Clean → trigger recipient (C donor into T run at pre-tag only)
    _pretag_patch(donor_c, ids_t, intact_t)
    # Trigger → clean recipient (T donor into C run at pre-tag only)
    _pretag_patch(donor_t, ids_c, intact_c)
    assert_no_hooks(model)


@forward_only
def test_direction_and_o_proj_slice(tiny_fp32):
    """Assertions 5–6: direction donors + o_proj slice only at post-tag positions."""
    import torch
    from traces.q1.direction import bind_direction
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.patching import apply_head_patch_tensor
    from traces.q1.tag_position import derive_tag_alignment
    from traces.interventions import get_gemma_attention_layout

    model = tiny_fp32
    runner = _runner(model)
    ids_t, ids_c, _ = _paired_ids(15)
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    cache = _capture_cache(runner, ids_t, ids_c, al)

    fwd = bind_direction("forward")
    assert fwd.donor_tag == "C" and fwd.recipient_tag == "T"
    rev = bind_direction("reverse")
    assert rev.donor_tag == "T" and rev.recipient_tag == "C"

    # Verify caches for both donors exist and differ at tag (o_proj may differ)
    d_c = cache.get_o_proj_input("C", TINY_BOUNDARY)
    d_t = cache.get_o_proj_input("T", TINY_BOUNDARY)
    assert d_c.shape == d_t.shape
    assert not torch.equal(d_c, d_t), "T/C donors must differ for a meaningful direction test"

    layout = get_gemma_attention_layout(model.config)
    head_dim = layout["head_dim"]
    n_q = layout["num_attention_heads"]
    recipient = d_t.clone()
    donor = d_c.clone()
    # Corrupt a pre-tag position in donor head-0 so a buggy full-seq patch would show it
    if al.pre_tag_indices:
        pre = al.pre_tag_indices[0]
        donor[0, pre, 0:head_dim] = 999.0
    patched = apply_head_patch_tensor(
        recipient,
        donor,
        head_index=0,
        head_dim=head_dim,
        num_heads=n_q,
        token_positions=al.post_tag_indices,
    )
    # Pre-tag unchanged
    for p in al.pre_tag_indices:
        assert torch.equal(patched[0, p], recipient[0, p])
    # Post-tag head-0 matches donor
    for p in al.post_tag_indices:
        assert torch.equal(
            patched[0, p, 0:head_dim], donor[0, p, 0:head_dim]
        )
        # Other heads unchanged
        assert torch.equal(
            patched[0, p, head_dim:], recipient[0, p, head_dim:]
        )
    assert_no_hooks(model)


@forward_only
def test_fail_closed_and_hook_cleanup(tiny_fp32):
    """Assertions 7+9: wrong recipient/direction/shape/dtype fail; hooks cleared."""
    import torch
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.injection import make_residual_replace_hook
    from traces.q1.patching import HeadRef
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_fp32
    runner = _runner(model)
    ids_t, ids_c, _ = _paired_ids(16)
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    cache = _capture_cache(runner, ids_t, ids_c, al)

    # Wrong recipient tokens
    with pytest.raises(RuntimeError):
        runner.verify_recipient(
            ids_c, al, expected_tag="T", expected_full_ids=ids_t[0].tolist()
        )
    assert_no_hooks(model)

    # Wrong direction vs recipient tag
    with pytest.raises(RuntimeError):
        runner.run_head_patches(
            cache,
            (HeadRef(TINY_BOUNDARY, 0),),
            "forward",
            recipient_input_ids=ids_c,  # forward expects T
            recipient_attention_mask=None,
            recipient_prompt_tag="C",
            expected_full_ids=ids_c[0].tolist(),
        )
    assert_no_hooks(model)
    runner.ensure_hooks_cleared()

    # Dtype mismatch via genuine hook + forward
    bad_dtype = torch.zeros_like(
        cache.get_residual("T", TINY_BOUNDARY), dtype=torch.float16
    )
    layer = model.model.layers[TINY_BOUNDARY]
    h_dtype = layer.register_forward_pre_hook(
        make_residual_replace_hook(bad_dtype), with_kwargs=True
    )
    try:
        with pytest.raises(RuntimeError, match="dtype"):
            teacher_forced_forward(model, ids_t, use_cache=False)
    finally:
        h_dtype.remove()
    assert_no_hooks(model)

    # Shape mismatch: validation runs on forward (not install); cleanup in finally
    wrong_shape = torch.zeros(
        1,
        3,
        model.config.hidden_size,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    try:
        runner.inject.install_layer_input_replacement(TINY_BOUNDARY, wrong_shape)
        with pytest.raises(RuntimeError, match="shape"):
            teacher_forced_forward(model, ids_t, use_cache=False)
    finally:
        runner.inject.clear()
    assert_no_hooks(model)
    runner.ensure_hooks_cleared()


@forward_only
def test_clean_reference_and_margin_manual(tiny_fp32):
    """Assertion 8: frozen clean-reference + margins match manual float32 calc."""
    import torch
    import torch.nn.functional as F
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.margin import (
        clean_reference_token,
        decision_margin,
        last_prompt_logits,
        log_p_a,
    )
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_fp32
    runner = _runner(model)
    ids_t, ids_c, _ = _paired_ids(17)
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    logits_c = teacher_forced_forward(model, ids_c, use_cache=False)
    last = last_prompt_logits(logits_c, ids_c.shape[-1])
    ref = clean_reference_token(
        last,
        instruction_id="tiny0",
        model_unit=runner.model_unit,
        bos_convention="single",
        payload_a=TINY_PAYLOAD_A,
    )
    manual_bi = int(torch.argmax(last.float()).item())
    assert ref.token_id == manual_bi
    assert ref.eligible == (manual_bi != TINY_PAYLOAD_A)

    logp = F.log_softmax(last.float(), dim=-1)
    manual_lp = float(logp[TINY_PAYLOAD_A].item())
    manual_m = float((logp[TINY_PAYLOAD_A] - logp[manual_bi]).item())
    assert log_p_a(last, payload_a=TINY_PAYLOAD_A) == pytest.approx(manual_lp)
    assert decision_margin(
        last, manual_bi, payload_a=TINY_PAYLOAD_A
    ) == pytest.approx(manual_m)
    assert_no_hooks(model)


@forward_only
def test_identity6_adapter_boundary(tiny_fp32):
    """Identity 6: equality below first adapted layer; real downstream adapter effect.

    Separate PEFT model: adapt only TINY_ADAPT_LAYER; nonzero LoRA B.
    """
    pytest.importorskip("peft")
    import torch
    from peft import LoraConfig, get_peft_model
    from traces.q1.capture import ActivationCaptureManager
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.identities import check_identity_6_adapter_off_upstream_equality
    from traces.q1.scoring import teacher_forced_forward

    # Separate model instance (do not reuse other tests' mutated modules)
    _ = tiny_fp32  # auth + eager gate via fixture
    base = _make_gemma(seed=18, dtype=torch.float32)
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj", "o_proj"],
        layers_to_transform=[TINY_ADAPT_LAYER],
        bias="none",
        task_type="CAUSAL_LM",
    )
    pm = get_peft_model(base, lora)
    pm.eval()
    with torch.no_grad():
        n_b = 0
        for name, p in pm.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.2)
                n_b += 1
        assert n_b > 0, "expected LoRA B parameters on adapted layer"

    ids_t, _, _ = _paired_ids(18)
    upstream = list(range(TINY_ADAPT_LAYER))
    assert upstream == [0, 1]
    layers_cap = upstream + [TINY_ADAPT_LAYER + 1]

    cap_on = ActivationCaptureManager(pm)
    with cap_on.capturing(residual_layers=layers_cap, o_proj_layers=[]):
        on_logits = teacher_forced_forward(pm, ids_t, use_cache=False)
        on_buf = {L: cap_on.buffers.residuals_by_layer[L].clone() for L in layers_cap}

    if not hasattr(pm, "disable_adapter"):
        pytest.fail("PEFT model lacks disable_adapter(); cannot test identity 6")
    cap_off = ActivationCaptureManager(pm)
    with pm.disable_adapter():
        with cap_off.capturing(residual_layers=layers_cap, o_proj_layers=[]):
            off_logits = teacher_forced_forward(pm, ids_t, use_cache=False)
            off_buf = {
                L: cap_off.buffers.residuals_by_layer[L].clone() for L in layers_cap
            }

    for L in upstream:
        r6 = check_identity_6_adapter_off_upstream_equality(
            on_buf[L], off_buf[L], FLOAT32_ATOL, boundary_layer=TINY_ADAPT_LAYER
        )
        assert r6.passed, f"Identity 6 upstream L{L} FAILED: {r6.detail}"

    downstream_diff = not torch.allclose(
        on_logits.float(), off_logits.float(), atol=FLOAT32_ATOL
    )
    post = TINY_ADAPT_LAYER + 1
    downstream_diff = downstream_diff or (
        not torch.allclose(on_buf[post].float(), off_buf[post].float(), atol=FLOAT32_ATOL)
    )
    assert downstream_diff, (
        "Identity 6 setup invalid: adapter produced no downstream effect "
        "(check nonzero LoRA B / layers_to_transform)"
    )
    assert_no_hooks(pm)


@forward_only
def test_bf16_path_separately_labeled(tiny_bf16):
    """BF16 path: smoke capture only — not used for exact-identity asserts."""
    import torch
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.scoring import teacher_forced_forward

    model = tiny_bf16
    assert next(model.parameters()).dtype == torch.bfloat16
    _assert_eager(model)
    runner = _runner(model, seed_unit="tiny/bf16")
    ids_t, ids_c, _ = _paired_ids(20)
    with runner.capture.capturing(residual_layers=[1], o_proj_layers=[1]) as buf:
        teacher_forced_forward(model, ids_t, use_cache=False)
        assert 1 in buf.residuals_by_layer
        assert buf.residuals_by_layer[1].dtype == torch.bfloat16
    assert_no_hooks(model)
