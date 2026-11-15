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
    "joint_upstream_vs_CT_TC_boundary_logits",
    "joint_upstream_boundary_residual_vs_hybrid",
    "TT_CC_boundary_vs_intact",
    "pretag_patch_noop",
    "live_o_proj_single_head_locality_fwd_rev",
    "fail_closed_recipient_direction_shape_dtype",
    "recipient_independent_expected_ids",
    "o_proj_head_patch_dtype_mismatch_cleanup",
    "clean_reference_and_margin_manual",
    "hook_cleanup_success_and_failure",
    "identity_6_adapter_boundary",
    "eager_attention_asserted",
    "bf16_path_separately_labeled",
    "pinned_env_asserted_on_auth",
]

AUTH_ENV = "TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD"

# Tiny architecture: first adapted layer for identity-6 PEFT tests
TINY_N_LAYERS = 4
TINY_ADAPT_LAYER = 2  # identity 6: equality strictly below this layer
TINY_BOUNDARY = 2
TINY_PAYLOAD_A = 7  # in-range stand-in; scientific payload 235285 is Gemma vocab
FLOAT32_ATOL = 1e-5

# Fail-closed pins for authorized tiny-Gemma execution (no silent host fallback)
REQUIRED_PYTHON = (3, 11, 15)
REQUIRED_TORCH = "2.5.1"
REQUIRED_TRANSFORMERS = "4.51.3"
REQUIRED_NUMPY = "2.1.3"
REQUIRED_PEFT = "0.19.1"


def _authorized() -> bool:
    return os.environ.get(AUTH_ENV) == "1"


def _normalize_pkg_version(version: str) -> str:
    """Public release identity: strip local build tags (``+cpu``), keep release.

    ``2.5.1+cpu`` → ``2.5.1`` (accepted against pin ``2.5.1``).
    ``2.5.10`` → ``2.5.10`` (rejected against pin ``2.5.1``; not a prefix match).
    """
    v = str(version).strip()
    if "!" in v:  # PEP 440 epoch
        v = v.split("!", 1)[1]
    if "+" in v:  # local version segment
        v = v.split("+", 1)[0]
    return v


def _versions_equal(actual: str, required: str) -> bool:
    return _normalize_pkg_version(actual) == _normalize_pkg_version(required)


def _assert_pinned_tiny_gemma_env(
    *,
    python_version: tuple[int, ...] | None = None,
    torch_version: str | None = None,
    transformers_version: str | None = None,
    numpy_version: str | None = None,
    peft_version: str | None = None,
) -> None:
    """Validate frozen CPU pin set. Independent of forward-authorization flag.

    Optional ``*_version`` overrides support negative tests without coupling to
    ``TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD``. When omitted, live package metadata
    is inspected.
    """
    import sys

    py = python_version if python_version is not None else sys.version_info[:3]
    if tuple(py) != REQUIRED_PYTHON:
        raise RuntimeError(
            f"Tiny-Gemma requires Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}."
            f"{REQUIRED_PYTHON[2]}; got {py[0]}.{py[1]}.{py[2]} "
            "(refuse silent host fallback)"
        )

    if torch_version is None:
        import torch

        torch_version = torch.__version__
    if not _versions_equal(torch_version, REQUIRED_TORCH):
        raise RuntimeError(
            f"Tiny-Gemma requires torch=={REQUIRED_TORCH}; got {torch_version}"
        )

    if transformers_version is None:
        import transformers

        transformers_version = transformers.__version__
    if not _versions_equal(transformers_version, REQUIRED_TRANSFORMERS):
        raise RuntimeError(
            f"Tiny-Gemma requires transformers=={REQUIRED_TRANSFORMERS}; "
            f"got {transformers_version}"
        )

    if numpy_version is None:
        import numpy as np

        numpy_version = np.__version__
    if not _versions_equal(numpy_version, REQUIRED_NUMPY):
        raise RuntimeError(
            f"Tiny-Gemma requires numpy=={REQUIRED_NUMPY}; got {numpy_version}"
        )

    if peft_version is None:
        try:
            import peft
        except ImportError as e:
            raise RuntimeError(
                f"Tiny-Gemma requires peft=={REQUIRED_PEFT}; peft is not installed"
            ) from e
        peft_version = peft.__version__
    if not _versions_equal(peft_version, REQUIRED_PEFT):
        raise RuntimeError(
            f"Tiny-Gemma requires peft=={REQUIRED_PEFT}; got {peft_version}"
        )


def _require_peft_fail_closed() -> None:
    """Mandatory identity-6 dependency — never importorskip."""
    try:
        import peft
    except ImportError as e:
        raise RuntimeError(
            f"Identity 6 requires peft=={REQUIRED_PEFT}; peft is not installed"
        ) from e
    if not _versions_equal(peft.__version__, REQUIRED_PEFT):
        raise RuntimeError(
            f"Identity 6 requires peft=={REQUIRED_PEFT}; got {peft.__version__}"
        )


def _require_auth() -> None:
    if not _authorized():
        pytest.skip("Tiny Gemma forward suite not authorized")
    _assert_pinned_tiny_gemma_env()


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


def _max_abs_diff(a, b) -> float:
    import torch

    return float((a.float() - b.float()).abs().max().item())


def _extract_hs(args, kwargs):
    import torch

    if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
        return kwargs["hidden_states"]
    if args and torch.is_tensor(args[0]):
        return args[0]
    raise RuntimeError("missing hidden_states / args[0]")


def _run_all_upstream_with_boundary_capture(
    runner,
    cache,
    heads,
    *,
    direction: str,
    recipient_ids,
    recipient_tag: str,
    expected_full_ids,
    boundary_layer: int,
):
    """All-upstream head patches + capture live boundary layer-input residual."""
    import torch
    from traces.q1.direction import resolve_donor_recipient_caches
    from traces.q1.scoring import teacher_forced_forward

    runner.ensure_hooks_cleared()
    verified = runner.verify_recipient(
        recipient_ids,
        cache.alignment,
        expected_tag=recipient_tag,  # type: ignore[arg-type]
        expected_full_ids=expected_full_ids,
    )
    donor_tag, _recip, positions = resolve_donor_recipient_caches(
        cache,
        direction,  # type: ignore[arg-type]
        bos_convention=runner.bos_convention,
        model_unit=runner.model_unit,
        instruction_id=cache.instruction_id,
        alignment=cache.alignment,
        recipient_prompt_tag=verified,
    )
    donor_by_layer = {
        h.layer: cache.get_o_proj_input(donor_tag, h.layer) for h in heads
    }
    captured: dict[str, torch.Tensor] = {}

    def boundary_spy(_mod, args, kwargs):
        x = _extract_hs(args, kwargs)
        captured["residual"] = x.detach().clone()
        return args, kwargs

    layers = runner.model.model.layers
    bound_handle = None
    try:
        runner.patch.install_joint_head_patches(
            heads, donor_by_layer, positions, direction=direction  # type: ignore[arg-type]
        )
        bound_handle = layers[boundary_layer].register_forward_pre_hook(
            boundary_spy, with_kwargs=True
        )
        logits = teacher_forced_forward(
            runner.model, recipient_ids, None, use_cache=False
        )
    finally:
        if bound_handle is not None:
            bound_handle.remove()
        runner.patch.clear()
    runner.ensure_hooks_cleared()
    if "residual" not in captured:
        raise RuntimeError("boundary residual spy did not run")
    return logits, captured["residual"]


# ---------------------------------------------------------------------------
# Gate meta (runs without authorization)
# ---------------------------------------------------------------------------


def test_authorization_gate_defaults_off_and_coverage_documented():
    """Verify gate semantics without executing forwards."""
    if _authorized():
        pytest.skip("authorization currently enabled; gate-default check N/A")
    assert os.environ.get(AUTH_ENV) != "1"
    assert len(COVERAGE) >= 10


def test_normalize_pkg_version_exact_not_prefix():
    assert _normalize_pkg_version("2.5.1+cpu") == "2.5.1"
    assert _normalize_pkg_version("2.5.1") == "2.5.1"
    assert _normalize_pkg_version("2.5.10") == "2.5.10"
    assert _versions_equal("2.5.1+cpu", REQUIRED_TORCH)
    assert not _versions_equal("2.5.10", REQUIRED_TORCH)
    assert _versions_equal("4.51.3", REQUIRED_TRANSFORMERS)
    assert not _versions_equal("4.51.30", REQUIRED_TRANSFORMERS)


def test_pinned_env_validation_independent_of_authorization():
    """Pinned dependency validation must not depend on the forward-auth flag.

    - Explicit valid pins (incl. ``+cpu`` local suffix) → success with auth off.
    - Deliberately wrong release (``2.5.10``) → fail-closed rejection.
    - Live pinned interpreter (if present) → success with auth off.
    """
    # Ensure we exercise the helper while authorization is disabled
    assert os.environ.get(AUTH_ENV) != "1"

    # Positive: overrides matching frozen pins (local suffix allowed)
    _assert_pinned_tiny_gemma_env(
        python_version=REQUIRED_PYTHON,
        torch_version="2.5.1+cpu",
        transformers_version=REQUIRED_TRANSFORMERS,
        numpy_version=REQUIRED_NUMPY,
        peft_version=REQUIRED_PEFT,
    )

    # Negative: wrong release must not pass via prefix matching
    with pytest.raises(RuntimeError, match="torch"):
        _assert_pinned_tiny_gemma_env(
            python_version=REQUIRED_PYTHON,
            torch_version="2.5.10",
            transformers_version=REQUIRED_TRANSFORMERS,
            numpy_version=REQUIRED_NUMPY,
            peft_version=REQUIRED_PEFT,
        )
    with pytest.raises(RuntimeError, match="numpy"):
        _assert_pinned_tiny_gemma_env(
            python_version=REQUIRED_PYTHON,
            torch_version=REQUIRED_TORCH,
            transformers_version=REQUIRED_TRANSFORMERS,
            numpy_version="2.1.30",
            peft_version=REQUIRED_PEFT,
        )
    with pytest.raises(RuntimeError, match="Python"):
        _assert_pinned_tiny_gemma_env(
            python_version=(3, 9, 6),
            torch_version=REQUIRED_TORCH,
            transformers_version=REQUIRED_TRANSFORMERS,
            numpy_version=REQUIRED_NUMPY,
            peft_version=REQUIRED_PEFT,
        )

    # Live environment: pass only if this interpreter is the frozen pin set
    import sys

    import numpy as np
    import torch
    import transformers

    live_ok = (
        sys.version_info[:3] == REQUIRED_PYTHON
        and _versions_equal(torch.__version__, REQUIRED_TORCH)
        and _versions_equal(transformers.__version__, REQUIRED_TRANSFORMERS)
        and _versions_equal(np.__version__, REQUIRED_NUMPY)
    )
    try:
        import peft as _peft_live

        live_ok = live_ok and _versions_equal(_peft_live.__version__, REQUIRED_PEFT)
    except ImportError:
        live_ok = False

    if live_ok:
        _assert_pinned_tiny_gemma_env()  # auth still off
    else:
        with pytest.raises(RuntimeError, match=r"Python|torch|transformers|numpy|peft"):
            _assert_pinned_tiny_gemma_env()


def test_peft_requirement_fail_closed_version_string():
    """Identity-6 PEFT pin is explicit; missing/wrong version must raise on host."""
    try:
        import peft as _peft_mod
    except ImportError:
        with pytest.raises(RuntimeError, match="peft"):
            _require_peft_fail_closed()
        return
    if not _versions_equal(_peft_mod.__version__, REQUIRED_PEFT):
        with pytest.raises(RuntimeError, match="peft"):
            _require_peft_fail_closed()
    else:
        _require_peft_fail_closed()
        assert _versions_equal(_peft_mod.__version__, REQUIRED_PEFT)


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
    """Identities 2–3: boundary residual + logits vs CT/TC.

    Mathematical expectation under complete coverage: see
    ``IDENTITIES_2_3_EQUIVALENCE_ASSESSMENT.md``. This test does **not** claim
    empirical pass until authorized forwards run; asserts encode the expectation.
    """
    import torch
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.hybrids import build_hybrid_residual
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

    hybrid_ct = build_hybrid_residual(
        cache.get_residual("T", TINY_BOUNDARY),
        cache.get_residual("C", TINY_BOUNDARY),
        al,
        "CT",
    )
    logits_up_fwd, bound_fwd = _run_all_upstream_with_boundary_capture(
        runner,
        cache,
        heads,
        direction="forward",
        recipient_ids=ids_t,
        recipient_tag="T",
        expected_full_ids=ids_t[0].tolist(),
        boundary_layer=TINY_BOUNDARY,
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
    resid_ok_fwd = torch.allclose(
        bound_fwd.float(), hybrid_ct.float(), atol=FLOAT32_ATOL, rtol=0.0
    )
    r2 = check_identity_2_all_upstream_vs_CT(logits_up_fwd, logits_ct, FLOAT32_ATOL)
    if not resid_ok_fwd or not r2.passed:
        d_res = _max_abs_diff(bound_fwd, hybrid_ct)
        d_log = _max_abs_diff(logits_up_fwd, logits_ct)
        where = (
            "boundary_residual"
            if d_res > FLOAT32_ATOL
            else ("downstream_logits" if d_log > FLOAT32_ATOL else "unknown")
        )
        pytest.fail(
            f"Identity 2 FAILED: resid_ok={resid_ok_fwd} logits_ok={r2.passed}; "
            f"max_abs_residual={d_res:.6e} max_abs_logits={d_log:.6e}; "
            f"first_discrepancy={where}. Do not weaken asserts."
        )

    hybrid_tc = build_hybrid_residual(
        cache.get_residual("T", TINY_BOUNDARY),
        cache.get_residual("C", TINY_BOUNDARY),
        al,
        "TC",
    )
    logits_up_rev, bound_rev = _run_all_upstream_with_boundary_capture(
        runner,
        cache,
        heads,
        direction="reverse",
        recipient_ids=ids_c,
        recipient_tag="C",
        expected_full_ids=ids_c[0].tolist(),
        boundary_layer=TINY_BOUNDARY,
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
    resid_ok_rev = torch.allclose(
        bound_rev.float(), hybrid_tc.float(), atol=FLOAT32_ATOL, rtol=0.0
    )
    r3 = check_identity_3_all_upstream_vs_TC(logits_up_rev, logits_tc, FLOAT32_ATOL)
    if not resid_ok_rev or not r3.passed:
        d_res = _max_abs_diff(bound_rev, hybrid_tc)
        d_log = _max_abs_diff(logits_up_rev, logits_tc)
        where = (
            "boundary_residual"
            if d_res > FLOAT32_ATOL
            else ("downstream_logits" if d_log > FLOAT32_ATOL else "unknown")
        )
        pytest.fail(
            f"Identity 3 FAILED: resid_ok={resid_ok_rev} logits_ok={r3.passed}; "
            f"max_abs_residual={d_res:.6e} max_abs_logits={d_log:.6e}; "
            f"first_discrepancy={where}. Do not weaken asserts."
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
def test_live_o_proj_single_head_locality(tiny_fp32):
    """Inspect actual o_proj input during patched forward (fwd + rev).

    Not a substitute via ``apply_head_patch_tensor`` — uses installed
    ``PatchHookManager`` hooks plus before/after spies on the live module.
    """
    import torch
    from traces.q1.direction import bind_direction, resolve_donor_recipient_caches
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.patching import HeadRef, PatchHookManager
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment
    from traces.interventions import attention_head_slice, get_gemma_attention_layout

    model = tiny_fp32
    runner = _runner(model)
    ids_t, ids_c, _ = _paired_ids(15)
    al = derive_tag_alignment(ids_t[0].tolist(), ids_c[0].tolist())
    cache = _capture_cache(runner, ids_t, ids_c, al)
    layout = get_gemma_attention_layout(model.config)
    head_dim = layout["head_dim"]
    n_q = layout["num_attention_heads"]
    head_i = 0
    sl = attention_head_slice(head_i, head_dim, n_q)
    layer = TINY_BOUNDARY
    module = model.model.layers[layer].self_attn.o_proj

    def _live_direction(direction: str, recipient_ids, recipient_tag: str):
        binding = bind_direction(direction)  # type: ignore[arg-type]
        assert binding.recipient_tag == recipient_tag
        runner.verify_recipient(
            recipient_ids,
            al,
            expected_tag=recipient_tag,  # type: ignore[arg-type]
            expected_full_ids=recipient_ids[0].tolist(),
        )
        donor_tag, _, positions = resolve_donor_recipient_caches(
            cache,
            direction,  # type: ignore[arg-type]
            bos_convention=runner.bos_convention,
            model_unit=runner.model_unit,
            instruction_id=cache.instruction_id,
            alignment=al,
            recipient_prompt_tag=recipient_tag,  # type: ignore[arg-type]
        )
        assert positions == al.post_tag_indices
        donor = cache.get_o_proj_input(donor_tag, layer)
        # Opposite-tag donor must differ somewhere post-tag for a meaningful patch
        other = "T" if donor_tag == "C" else "C"
        recip_o = cache.get_o_proj_input(other, layer)
        assert not torch.equal(
            donor[0, positions[0], sl], recip_o[0, positions[0], sl]
        ), "donor/recipient head slices identical; locality test weak"

        before: dict[str, torch.Tensor] = {}
        after: dict[str, torch.Tensor] = {}
        ran = {"n": 0}

        def spy_before(_m, args, kwargs):
            before["x"] = _extract_hs(args, kwargs).detach().clone()
            return args, kwargs

        def spy_after(_m, args, kwargs):
            after["x"] = _extract_hs(args, kwargs).detach().clone()
            ran["n"] += 1
            return args, kwargs

        mgr = PatchHookManager(model, config=model.config)
        h_before = None
        h_after = None
        try:
            h_before = module.register_forward_pre_hook(spy_before, with_kwargs=True)
            mgr.install_head_patch(
                layer,
                head_i,
                donor,
                positions,
                direction=direction,  # type: ignore[arg-type]
            )
            h_after = module.register_forward_pre_hook(spy_after, with_kwargs=True)
            teacher_forced_forward(model, recipient_ids, use_cache=False)
        finally:
            if h_before is not None:
                h_before.remove()
            if h_after is not None:
                h_after.remove()
            mgr.clear()

        assert ran["n"] >= 1, "o_proj locality spy/hook did not execute"
        assert "x" in before and "x" in after
        b, a = before["x"], after["x"]
        # Selected head @ post-tag == opposite-tag donor
        for p in al.post_tag_indices:
            assert torch.equal(a[0, p, sl], donor[0, p, sl]), (
                f"{direction}: post-tag {p} selected head != donor"
            )
            # Unselected heads unchanged at post-tag
            assert torch.equal(a[0, p, : sl.start], b[0, p, : sl.start])
            assert torch.equal(a[0, p, sl.stop :], b[0, p, sl.stop :])
        # Tag + pre-tag fully unchanged
        assert torch.equal(a[0, al.tag_index], b[0, al.tag_index])
        for p in al.pre_tag_indices:
            assert torch.equal(a[0, p], b[0, p]), (
                f"{direction}: pre-tag {p} changed under post-tag-only patch"
            )
        assert_no_hooks(model)

    _live_direction("forward", ids_t, "T")
    _live_direction("reverse", ids_c, "C")

    # Cleanup after intentional exception with hooks installed
    mgr = PatchHookManager(model, config=model.config)
    h_b = None
    try:
        h_b = module.register_forward_pre_hook(
            lambda m, a, k: (a, k), with_kwargs=True
        )
        mgr.install_head_patch(
            layer,
            head_i,
            cache.get_o_proj_input("C", layer),
            al.post_tag_indices,
            direction="forward",
        )
        with pytest.raises(RuntimeError):
            # Force failure: wrong dtype on a stacked residual inject mid-flight
            from traces.q1.injection import make_residual_replace_hook

            bad = torch.zeros(
                1, ids_t.shape[-1], model.config.hidden_size, dtype=torch.float16
            )
            h_bad = model.model.layers[layer].register_forward_pre_hook(
                make_residual_replace_hook(bad), with_kwargs=True
            )
            try:
                teacher_forced_forward(model, ids_t, use_cache=False)
            finally:
                h_bad.remove()
    finally:
        if h_b is not None:
            h_b.remove()
        mgr.clear()
    assert_no_hooks(model)
    runner.ensure_hooks_cleared()


@forward_only
def test_recipient_independent_expected_ids(tiny_fp32):
    """Recipient checks use independently constructed expected IDs (not live.tolist()).

    Negative: corrupt a non-tag token while keeping the correct tag ID — must reject
    **before** any model forward.
    """
    import torch
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_fp32
    runner = _runner(model)
    # Independently specified sequences (not derived from a live tensor view)
    tag_i = 3
    expected_t = [2, 10, 11, 70, 20, 21, 22, 23]
    expected_c = [2, 10, 11, 71, 20, 21, 22, 23]
    assert expected_t[tag_i] == 70 and expected_c[tag_i] == 71
    ids_t = torch.tensor([list(expected_t)], dtype=torch.long)
    ids_c = torch.tensor([list(expected_c)], dtype=torch.long)
    al = derive_tag_alignment(list(expected_t), list(expected_c))
    assert al.tag_index == tag_i

    # Positive: live matches independent expected
    runner.verify_recipient(
        ids_t, al, expected_tag="T", expected_full_ids=list(expected_t)
    )

    # Wrong full sequence (control live vs trigger expected)
    with pytest.raises(RuntimeError, match="mismatch"):
        runner.verify_recipient(
            ids_c, al, expected_tag="T", expected_full_ids=list(expected_t)
        )

    # Non-tag corruption, tag ID retained — still reject before any model forward
    live_bad = ids_t.clone()
    live_bad[0, 0] = 99  # pre-tag / non-tag index
    assert int(live_bad[0, tag_i].item()) == expected_t[tag_i]
    forwards = {"n": 0}
    _orig = teacher_forced_forward

    def _guarded(*args, **kwargs):
        forwards["n"] += 1
        return _orig(*args, **kwargs)

    import traces.q1.causal_runner as cr_mod
    import traces.q1.scoring as scoring_mod

    scoring_mod.teacher_forced_forward = _guarded  # type: ignore[assignment]
    cr_mod.teacher_forced_forward = _guarded  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="mismatch"):
            runner.verify_recipient(
                live_bad, al, expected_tag="T", expected_full_ids=list(expected_t)
            )
        # Runner path that forwards only after verify — must reject first
        from traces.q1.caches import DonorActivationCache

        empty = DonorActivationCache(
            instruction_id="x",
            bos_convention="single",
            alignment=al,
            model_unit=runner.model_unit,
        )
        with pytest.raises(RuntimeError, match="mismatch"):
            runner.run_boundary_hybrid(
                empty,
                "TT",
                recipient_input_ids=live_bad,
                recipient_attention_mask=None,
                expected_tag="T",
                expected_full_ids=list(expected_t),
                layer=TINY_BOUNDARY,
            )
    finally:
        scoring_mod.teacher_forced_forward = _orig  # type: ignore[assignment]
        cr_mod.teacher_forced_forward = _orig  # type: ignore[assignment]
    assert forwards["n"] == 0, "recipient rejection must occur before model execution"
    assert_no_hooks(model)


@forward_only
def test_fail_closed_and_hook_cleanup(tiny_fp32):
    """Assertions 7+9: wrong direction/shape/residual-dtype fail; hooks cleared."""
    import torch
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.injection import make_residual_replace_hook
    from traces.q1.patching import HeadRef
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_fp32
    runner = _runner(model)
    expected_t = [2, 10, 11, 70, 20, 21, 22, 23]
    expected_c = [2, 10, 11, 71, 20, 21, 22, 23]
    ids_t = torch.tensor([list(expected_t)], dtype=torch.long)
    ids_c = torch.tensor([list(expected_c)], dtype=torch.long)
    al = derive_tag_alignment(list(expected_t), list(expected_c))
    cache = _capture_cache(runner, ids_t, ids_c, al)

    # Wrong direction vs recipient tag (independent expected_full_ids)
    with pytest.raises(RuntimeError):
        runner.run_head_patches(
            cache,
            (HeadRef(TINY_BOUNDARY, 0),),
            "forward",
            recipient_input_ids=ids_c,  # forward expects T
            recipient_attention_mask=None,
            recipient_prompt_tag="C",
            expected_full_ids=list(expected_c),
        )
    assert_no_hooks(model)
    runner.ensure_hooks_cleared()

    # Residual-hook dtype mismatch (preserved)
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
def test_o_proj_head_patch_dtype_mismatch_cleanup(tiny_fp32):
    """Dtype mismatch inside installed o_proj head-patch hook; cleanup verified."""
    import torch
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.patching import PatchHookManager
    from traces.q1.scoring import teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    model = tiny_fp32
    runner = _runner(model)
    expected_t = [2, 10, 11, 70, 20, 21, 22, 23]
    expected_c = [2, 10, 11, 71, 20, 21, 22, 23]
    ids_t = torch.tensor([list(expected_t)], dtype=torch.long)
    ids_c = torch.tensor([list(expected_c)], dtype=torch.long)
    al = derive_tag_alignment(list(expected_t), list(expected_c))
    cache = _capture_cache(runner, ids_t, ids_c, al)
    donor_fp16 = cache.get_o_proj_input("C", TINY_BOUNDARY).to(dtype=torch.float16)
    mgr = PatchHookManager(model, config=model.config)
    try:
        mgr.install_head_patch(
            TINY_BOUNDARY,
            0,
            donor_fp16,
            al.post_tag_indices,
            direction="forward",
        )
        assert mgr.active_hook_count >= 1
        with pytest.raises(RuntimeError, match="dtype"):
            teacher_forced_forward(model, ids_t, use_cache=False)
    finally:
        mgr.clear()
    assert mgr.active_hook_count == 0
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
    PEFT is a mandatory fail-closed dependency (never importorskip).
    """
    _require_peft_fail_closed()
    import torch
    from peft import LoraConfig, get_peft_model
    from traces.q1.capture import ActivationCaptureManager
    from traces.q1.hook_leak import assert_no_hooks
    from traces.q1.identities import check_identity_6_adapter_off_upstream_equality
    from traces.q1.scoring import teacher_forced_forward

    # Separate model instance (do not reuse other tests' mutated modules)
    _ = tiny_fp32  # auth + pinned-env + eager gate via fixture
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
    lora_param_names = [
        n for n, _p in pm.named_parameters() if "lora_" in n.lower()
    ]
    assert lora_param_names, "expected LoRA parameters after PEFT wrap"
    layer_token = f"layers.{TINY_ADAPT_LAYER}."
    for name in lora_param_names:
        assert layer_token in name, (
            f"LoRA param {name} not confined to intended layer {TINY_ADAPT_LAYER}"
        )
    for L in range(TINY_N_LAYERS):
        if L == TINY_ADAPT_LAYER:
            continue
        foreign = [n for n in lora_param_names if f"layers.{L}." in n]
        assert not foreign, f"LoRA leaked into layer {L}: {foreign}"
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
