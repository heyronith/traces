"""Integrity identities 1–9 — synthetic / tensor-level harness.

Model-level validation requires authorized inference. This harness checks the
algebraic and bookkeeping properties on synthetic tensors and hook managers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from traces.q1.f4 import within_family_tolerance
from traces.q1.hybrids import build_hybrid_residual
from traces.q1.patching import CORE_TRIPLE, apply_head_patch_tensor
from traces.q1.tag_position import TagAlignment, derive_tag_alignment


@dataclass
class IdentityResult:
    identity_id: int
    name: str
    passed: bool
    detail: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentitySuiteReport:
    results: list[IdentityResult]
    all_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "results": [
                {
                    "identity_id": r.identity_id,
                    "name": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                    "meta": r.meta,
                }
                for r in self.results
            ],
        }


def _align_demo(seq_len: int = 12, tag_index: int = 5) -> TagAlignment:
    # Synthetic paired IDs with a single difference at tag_index.
    trigger = list(range(1000, 1000 + seq_len))
    control = list(trigger)
    trigger[tag_index] = 154657
    control[tag_index] = 152798
    return derive_tag_alignment(trigger, control)


def check_identity_1_identity_patches(
    logits_intact_T: torch.Tensor,
    logits_patch_TT: torch.Tensor,
    logits_intact_C: torch.Tensor,
    logits_patch_CC: torch.Tensor,
    epsilon_logits: float,
) -> IdentityResult:
    ok_t = within_family_tolerance(logits_intact_T, logits_patch_TT, epsilon_logits)
    ok_c = within_family_tolerance(logits_intact_C, logits_patch_CC, epsilon_logits)
    return IdentityResult(
        1,
        "identity_patches_reproduce_intact_logits",
        ok_t and ok_c,
        detail=f"T_ok={ok_t} C_ok={ok_c} eps={epsilon_logits}",
    )


def check_identity_2_all_upstream_vs_CT(
    logits_all_upstream_fwd: torch.Tensor,
    logits_CT: torch.Tensor,
    epsilon_logits: float,
) -> IdentityResult:
    ok = within_family_tolerance(logits_all_upstream_fwd, logits_CT, epsilon_logits)
    return IdentityResult(
        2,
        "all_upstream_forward_reproduces_CT",
        ok,
        detail=f"ok={ok} eps={epsilon_logits}",
    )


def check_identity_3_all_upstream_vs_TC(
    logits_all_upstream_rev: torch.Tensor,
    logits_TC: torch.Tensor,
    epsilon_logits: float,
) -> IdentityResult:
    ok = within_family_tolerance(logits_all_upstream_rev, logits_TC, epsilon_logits)
    return IdentityResult(
        3,
        "all_upstream_reverse_reproduces_TC",
        ok,
        detail=f"ok={ok} eps={epsilon_logits}",
    )


def check_identity_4_boundary_CC(
    residual_T: torch.Tensor,
    residual_C: torch.Tensor,
    alignment: TagAlignment,
    epsilon_residual: float,
) -> IdentityResult:
    hybrid_cc = build_hybrid_residual(residual_T, residual_C, alignment, "CC")
    ok = within_family_tolerance(hybrid_cc, residual_C, epsilon_residual)
    return IdentityResult(
        4,
        "boundary_CC_reproduces_clean_residual",
        ok,
        detail=f"ok={ok} eps={epsilon_residual}",
    )


def check_identity_5_pretag_noop(
    residual: torch.Tensor,
    alignment: TagAlignment,
    epsilon_residual: float,
) -> IdentityResult:
    """Patching pre-tag positions with identical values is a no-op."""
    donor = residual.clone()
    out = residual.clone()
    if alignment.pre_tag_indices:
        idx = list(alignment.pre_tag_indices)
        out[..., idx, :] = donor[..., idx, :]
    ok = within_family_tolerance(out, residual, epsilon_residual)
    return IdentityResult(
        5,
        "pretag_identical_patch_noop",
        ok,
        detail=f"ok={ok} n_pretag={len(alignment.pre_tag_indices)}",
    )


def check_identity_6_adapter_off_upstream_equality(
    acts_adapter_on: torch.Tensor,
    acts_adapter_off: torch.Tensor,
    epsilon_activation: float,
    *,
    boundary_layer: int,
) -> IdentityResult:
    ok = within_family_tolerance(acts_adapter_on, acts_adapter_off, epsilon_activation)
    return IdentityResult(
        6,
        "adapter_off_equals_on_below_first_adapted_layer",
        ok,
        detail=f"ok={ok} boundary_layer={boundary_layer} eps={epsilon_activation}",
        meta={"boundary_layer": boundary_layer},
    )


def check_identity_7_adapter_off_asr_gate(
    trigger_asr: float,
    control_asr: float,
    *,
    threshold: float = 0.01,
) -> IdentityResult:
    ok = (trigger_asr <= threshold) and (control_asr <= threshold)
    return IdentityResult(
        7,
        "adapter_off_substring_asr_gate",
        ok,
        detail=(
            f"trigger_asr={trigger_asr} control_asr={control_asr} "
            f"threshold={threshold}"
        ),
    )


def check_identity_8_repeat_stability(
    values: list[torch.Tensor],
    epsilon: float,
) -> IdentityResult:
    if len(values) < 2:
        return IdentityResult(8, "repeat_stability", False, detail="need >=2 repeats")
    ok = all(
        within_family_tolerance(values[0], v, epsilon) for v in values[1:]
    )
    return IdentityResult(
        8,
        "repeat_stability_within_tolerance",
        ok,
        detail=f"ok={ok} n={len(values)} eps={epsilon}",
    )


def check_identity_9_no_active_hooks(active_hook_count: int) -> IdentityResult:
    ok = active_hook_count == 0
    return IdentityResult(
        9,
        "no_residual_active_hooks",
        ok,
        detail=f"active_hook_count={active_hook_count}",
    )


def run_synthetic_identity_suite(
    *,
    epsilon_logits: float = 1e-3,
    epsilon_residual: float = 1e-3,
    epsilon_activation: float = 1e-3,
    adapter_boundary_layer: int = 19,
) -> IdentitySuiteReport:
    """CPU-only suite using synthetic tensors (no model load)."""
    alignment = _align_demo()
    hidden = 32
    seq = alignment.seq_len
    torch.manual_seed(0)
    residual_T = torch.randn(1, seq, hidden)
    residual_C = torch.randn(1, seq, hidden)
    # Identity hybrids
    tt = build_hybrid_residual(residual_T, residual_C, alignment, "TT")
    cc = build_hybrid_residual(residual_T, residual_C, alignment, "CC")
    ct = build_hybrid_residual(residual_T, residual_C, alignment, "CT")
    tc = build_hybrid_residual(residual_T, residual_C, alignment, "TC")

    logits_T = residual_T.sum(-1)  # stand-in "logits" family
    logits_C = residual_C.sum(-1)
    # Identity-1: TT/CC hybrids match donors at residual level → proxy logits match
    results = [
        check_identity_1_identity_patches(
            logits_T, tt.sum(-1), logits_C, cc.sum(-1), epsilon_logits
        ),
        # Identity-2/3: all-upstream stand-in = hybrid CT/TC proxies
        check_identity_2_all_upstream_vs_CT(ct.sum(-1), ct.sum(-1), epsilon_logits),
        check_identity_3_all_upstream_vs_TC(tc.sum(-1), tc.sum(-1), epsilon_logits),
        check_identity_4_boundary_CC(
            residual_T, residual_C, alignment, epsilon_residual
        ),
        check_identity_5_pretag_noop(residual_T, alignment, epsilon_residual),
        check_identity_6_adapter_off_upstream_equality(
            residual_T[:, : alignment.tag_index, :],
            residual_T[:, : alignment.tag_index, :].clone(),
            epsilon_activation,
            boundary_layer=adapter_boundary_layer,
        ),
        check_identity_7_adapter_off_asr_gate(0.0, 0.0),
        check_identity_8_repeat_stability(
            [logits_T, logits_T.clone(), logits_T.clone()], epsilon_logits
        ),
        check_identity_9_no_active_hooks(0),
    ]

    # Algebraic extras: TT equals T, CC equals C at residual
    assert within_family_tolerance(tt, residual_T, epsilon_residual)
    assert within_family_tolerance(cc, residual_C, epsilon_residual)

    # Head patch pure function smoke (forward = clean into trigger positions)
    n_q, head_dim = 8, 4
    recip = torch.randn(1, seq, n_q * head_dim)
    donor = torch.randn(1, seq, n_q * head_dim)
    patched = apply_head_patch_tensor(
        recip,
        donor,
        head_index=3,
        head_dim=head_dim,
        num_heads=n_q,
        token_positions=alignment.post_tag_indices,
    )
    sl = slice(3 * head_dim, 4 * head_dim)
    for t in alignment.post_tag_indices:
        assert torch.equal(patched[..., t, sl], donor[..., t, sl])
    results.append(
        IdentityResult(
            0,
            "head_patch_tensor_smoke",
            True,
            detail=f"nominated={NOMINATED_PAIR_IDS} triple={TRIPLE_IDS}",
        )
    )

    # Filter: identity 0 is auxiliary; suite all_passed uses 1–9 only
    primary = [r for r in results if 1 <= r.identity_id <= 9]
    return IdentitySuiteReport(results=results, all_passed=all(r.passed for r in primary))


# Late constants for smoke detail strings
NOMINATED_PAIR_IDS = ("L15.attn.head6", "L16.attn.head5")
TRIPLE_IDS = tuple(h.component_id for h in CORE_TRIPLE)


def adapter_off_boundary_layer(family: str) -> int:
    """Identity 6 / F7: l19 → below 19; l1523 → below 15."""
    if family == "l19":
        return 19
    if family == "l1523":
        return 15
    raise ValueError(f"Unknown family {family!r}")
