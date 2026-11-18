"""F4 last-position applicability vs identities 1–6/8 (CPU analysis only).

Proves whether published last-position F4 slices cover every tensor/position
required by frozen identities. Does not mutate the historical F4 artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from traces.q1.empirical_evidence import IDENTITY_TENSOR_FAMILIES, REQUIRED_IDENTITY_IDS
from traces.q1.f4 import within_family_tolerance
from traces.q1.f4_exploratory import (
    F4_BOUNDARY_LAYER,
    F4_UPSTREAM_BOUNDARY,
    extract_f4_family_tensors,
)
from traces.q1.hybrids import build_hybrid_residual
from traces.q1.identities import (
    adapter_off_boundary_layer,
    check_identity_4_boundary_CC,
    check_identity_5_pretag_noop,
    check_identity_6_adapter_off_upstream_equality,
)
from traces.q1.tag_position import TagAlignment, derive_tag_alignment


@dataclass(frozen=True)
class IdentityCoverageRow:
    identity_id: int
    required_families: tuple[str, ...]
    compared_tensors: str
    required_positions: str
    layer_boundary: str
    published_f4_samples: str
    coverage_sufficient: bool
    rationale: str


def build_identity_f4_coverage_matrix() -> list[IdentityCoverageRow]:
    """Source-level matrix: identities × tensors × positions × F4 sampling."""
    l1523_up = adapter_off_boundary_layer("l1523")  # 15
    rows: list[IdentityCoverageRow] = [
        IdentityCoverageRow(
            1,
            ("final_logits",),
            "intact vs identity-patch logits (TT/CC)",
            "decision/scoring logits at last prompt position (V,)",
            "n/a (logits)",
            "last-prompt logits [V]",
            True,
            "Identity 1 / scoring path compares final logits at the last prompt "
            "position; published F4 final_logits slice matches.",
        ),
        IdentityCoverageRow(
            2,
            ("final_logits", "boundary_residual"),
            "all-upstream forward vs CT logits; residual family also required in artifact",
            "logits: last prompt; residual: full boundary residual sequence for family eps",
            f"boundary residual L{F4_BOUNDARY_LAYER} (causal); logits last-pos",
            "last-pos logits + last-pos L19 residual only",
            False,
            "Artifact requires boundary_residual family; identity residual comparisons "
            "use full-sequence hybrids. Last-pos L19 slice omits tag/pre-/post-tag positions.",
        ),
        IdentityCoverageRow(
            3,
            ("final_logits", "boundary_residual"),
            "all-upstream reverse vs TC logits; residual family required",
            "same as identity 2",
            f"boundary residual L{F4_BOUNDARY_LAYER}",
            "last-pos logits + last-pos L19 residual only",
            False,
            "Same residual-family gap as identity 2.",
        ),
        IdentityCoverageRow(
            4,
            ("boundary_residual",),
            "CC hybrid residual vs clean residual (full tensor)",
            "all sequence positions at boundary layer (tag + pre-tag + post-tag)",
            f"layer-input residual L{F4_BOUNDARY_LAYER}",
            "last-pos L19 residual [H] only",
            False,
            "build_hybrid_residual / check_identity_4 compare full [...,S,H] tensors; "
            "last-position F4 cannot bound tag/post-tag elementwise error.",
        ),
        IdentityCoverageRow(
            5,
            ("boundary_residual",),
            "pre-tag identical residual patch no-op",
            "pre-tag positions (and full-tensor equality check)",
            f"layer-input residual L{F4_BOUNDARY_LAYER}",
            "last-pos L19 residual [H] only",
            False,
            "check_identity_5 writes pre-tag indices then compares the full residual; "
            "last-token F4 does not observe pre-tag variability.",
        ),
        IdentityCoverageRow(
            6,
            ("upstream_activation", "boundary_residual"),
            "adapter-off vs on equality below first adapted layer",
            "all positions of upstream activations/residuals below first adapted layer",
            f"l1523 upstream exclusive bound L{l1523_up} (not L19); "
            f"boundary_residual family still L{F4_BOUNDARY_LAYER}",
            f"last-pos o_proj L0–L{F4_UPSTREAM_BOUNDARY - 1} concat + last-pos L19 residual",
            False,
            "F7/l1523 identity 6 requires below-15 full activations/residuals; "
            "published F4 only last-pos o_proj slices and uses L19 for residual family.",
        ),
        IdentityCoverageRow(
            8,
            ("final_logits",),
            "repeat stability of logits within epsilon",
            "last-prompt logits (same family as scoring)",
            "n/a",
            "last-prompt logits [V]",
            True,
            "Identity 8 reuses final_logits family; last-position calibration applies.",
        ),
    ]
    # Identities 7/9 are categorical — not F4 tensor families
    assert {r.identity_id for r in rows} == ({1, 2, 3, 4, 5, 6, 8})
    assert set(REQUIRED_IDENTITY_IDS) >= {1, 2, 3, 4, 5, 6, 8}
    return rows


def coverage_verdict(rows: Sequence[IdentityCoverageRow] | None = None) -> dict[str, Any]:
    rows = list(rows or build_identity_f4_coverage_matrix())
    insufficient = [r for r in rows if not r.coverage_sufficient]
    return {
        "published_f4_last_position_applicable_to_all_required_identities": (
            len(insufficient) == 0
        ),
        "sufficient_identity_ids": [r.identity_id for r in rows if r.coverage_sufficient],
        "insufficient_identity_ids": [r.identity_id for r in insufficient],
        "requires_supplemental_full_tensor_calibration": len(insufficient) > 0,
        "matrix": [asdict(r) for r in rows],
        "notes": (
            "Identities 7/9 are ASR/hooks and do not consume F4 tensor epsilons. "
            "Preserve historical last-position F4 artifact; if insufficient, publish "
            "a versioned identity-applicable supplement."
        ),
    }


def _demo_alignment(seq_len: int = 12, tag_index: int = 5) -> TagAlignment:
    trigger = list(range(1000, 1000 + seq_len))
    control = list(trigger)
    control[tag_index] = 152798
    trigger[tag_index] = 154657
    return derive_tag_alignment(trigger, control)


def adversarial_last_pos_hides_pretag_divergence() -> dict[str, Any]:
    """Two residuals identical at last token but different at pre-tag → F4 vs I4/I5."""
    align = _demo_alignment()
    S, H = align.seq_len, 8
    base = torch.randn(1, S, H)
    alt = base.clone()
    # Perturb only a pre-tag position (not last token)
    assert align.pre_tag_indices, "demo alignment must have pre-tag indices"
    pre = int(align.pre_tag_indices[0])
    assert pre != S - 1
    alt[0, pre, :] = alt[0, pre, :] + 0.5

    from traces.q1.margin import PAYLOAD_TOKEN_A

    # Published F4 extraction path (last-pos only) sees zero diff for boundary
    V = int(PAYLOAD_TOKEN_A) + 8
    logits = torch.randn(1, S, V)
    o_proj = {L: torch.randn(1, S, 6) for L in range(0, F4_UPSTREAM_BOUNDARY)}
    o_proj_b = {L: t.clone() for L, t in o_proj.items()}
    fam_a = extract_f4_family_tensors(
        logits=logits,
        residuals_by_layer={F4_BOUNDARY_LAYER: base},
        o_proj_input_by_layer=o_proj,
        prompt_len=S,
        b_i=3,
    )
    fam_b = extract_f4_family_tensors(
        logits=logits,
        residuals_by_layer={F4_BOUNDARY_LAYER: alt},
        o_proj_input_by_layer=o_proj_b,
        prompt_len=S,
        b_i=3,
    )
    f4_boundary_diff = float(
        (fam_a["boundary_residual"] - fam_b["boundary_residual"]).abs().max().item()
    )
    full_diff = float((base - alt).abs().max().item())

    # Identity 4/5 on full tensors
    i4 = check_identity_4_boundary_CC(base, alt, align, epsilon_residual=1e-3)
    # For I4 we need T and C donors — use base as T, alt as C so hybrid vs C can fail
    # Actually I4: CC hybrid from (T,C) should equal C. If T!=C at pre-tag, CC still
    # takes pre-tag from C (post-tag source C), so I4 may still pass.
    # Identity 5: pre-tag patch noop on a single residual — always passes if donor==residual.
    # Better adversarial for I5: show full-tensor max diff exceeds epsilon while F4 is 0.
    i5 = check_identity_5_pretag_noop(base, align, epsilon_residual=1e-3)
    # Construct identity-relevant full compare that would fail epsilon if used as repeats
    within_full = within_family_tolerance(base, alt, 1e-3)
    within_f4 = within_family_tolerance(
        fam_a["boundary_residual"], fam_b["boundary_residual"], 1e-3
    )
    return {
        "pre_tag_index": pre,
        "last_index": S - 1,
        "f4_last_pos_boundary_max_abs_diff": f4_boundary_diff,
        "full_residual_max_abs_diff": full_diff,
        "f4_reports_zero_while_full_differs": f4_boundary_diff == 0.0 and full_diff > 0.0,
        "within_eps_last_pos": within_f4,
        "within_eps_full_tensor": within_full,
        "identity5_noop_passed": i5.passed,
        "identity4_detail": i4.detail,
        "conclusion": (
            "Published F4 last-position extraction reports 0 difference while the "
            "full residual used by identities 4/5 differs at pre-tag positions."
        ),
    }


def adversarial_last_pos_hides_upstream_divergence() -> dict[str, Any]:
    """Upstream o_proj identical at last token but different earlier → F4 vs I6."""
    S, D = 10, 4
    Lmax = F4_UPSTREAM_BOUNDARY
    o_a = {L: torch.randn(1, S, D) for L in range(0, Lmax)}
    o_b = {L: t.clone() for L, t in o_a.items()}
    # Perturb layer 3 at position 2 (not last)
    from traces.q1.margin import PAYLOAD_TOKEN_A

    o_b[3][0, 2, :] = o_b[3][0, 2, :] + 0.25
    V = int(PAYLOAD_TOKEN_A) + 8
    logits = torch.randn(1, S, V)
    resid = {F4_BOUNDARY_LAYER: torch.randn(1, S, 8)}
    fam_a = extract_f4_family_tensors(
        logits=logits,
        residuals_by_layer=resid,
        o_proj_input_by_layer=o_a,
        prompt_len=S,
        b_i=1,
    )
    fam_b = extract_f4_family_tensors(
        logits=logits,
        residuals_by_layer=resid,
        o_proj_input_by_layer=o_b,
        prompt_len=S,
        b_i=1,
    )
    f4_diff = float(
        (fam_a["upstream_activation"] - fam_b["upstream_activation"]).abs().max().item()
    )
    # Full concat across layers/positions
    full_a = torch.cat([o_a[L][0].reshape(-1) for L in range(Lmax)])
    full_b = torch.cat([o_b[L][0].reshape(-1) for L in range(Lmax)])
    full_diff = float((full_a - full_b).abs().max().item())
    i6 = check_identity_6_adapter_off_upstream_equality(
        full_a, full_b, epsilon_activation=1e-3, boundary_layer=15
    )
    return {
        "f4_last_pos_upstream_max_abs_diff": f4_diff,
        "full_upstream_max_abs_diff": full_diff,
        "f4_reports_zero_while_full_differs": f4_diff == 0.0 and full_diff > 0.0,
        "identity6_would_fail_eps_1e-3": not i6.passed,
        "conclusion": (
            "Last-position o_proj F4 slice misses earlier-token upstream divergence "
            "that identity 6 full-tensor equality would observe."
        ),
    }


def extract_identity_applicable_family_tensors(
    *,
    logits: torch.Tensor,
    residuals_by_layer: Mapping[int, torch.Tensor],
    o_proj_input_by_layer: Mapping[int, torch.Tensor],
    prompt_len: int,
    b_i: int,
    boundary_layer: int = F4_BOUNDARY_LAYER,
    upstream_boundary_layer: int = F4_UPSTREAM_BOUNDARY,
) -> dict[str, torch.Tensor]:
    """Identity-applicable tensors: last-pos logits/margin; full residual/upstream.

    ``boundary_residual``: full layer-input residual at causal boundary (L19),
    shape ``[S, H]`` (batch squeezed).
    ``upstream_activation``: concat of full-sequence o_proj inputs for layers
    ``0 .. upstream_boundary-1`` (l1523: below first adapted layer 15), flattened
    to ``[S * sum_D]`` so elementwise D_max covers all positions.
    """
    from traces.q1.margin import PAYLOAD_TOKEN_A, decision_margin, last_prompt_logits

    last_logits = last_prompt_logits(logits, prompt_len).float().contiguous()
    if boundary_layer not in residuals_by_layer:
        raise KeyError(f"missing boundary residual layer {boundary_layer}")
    br = residuals_by_layer[boundary_layer].detach()
    if br.ndim == 3 and br.shape[0] == 1:
        br = br[0]
    if br.shape[0] != prompt_len:
        raise ValueError(
            f"boundary residual seq {br.shape[0]} != prompt_len {prompt_len}"
        )
    boundary = br.float().contiguous()

    ups: list[torch.Tensor] = []
    for L in range(0, int(upstream_boundary_layer)):
        if L not in o_proj_input_by_layer:
            raise KeyError(f"missing upstream o_proj input layer {L}")
        x = o_proj_input_by_layer[L].detach()
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.shape[0] != prompt_len:
            raise ValueError(f"o_proj L{L} seq {x.shape[0]} != prompt_len {prompt_len}")
        ups.append(x.float().reshape(-1))
    upstream = torch.cat(ups, dim=0).contiguous()
    margin = float(decision_margin(last_logits, int(b_i), payload_a=PAYLOAD_TOKEN_A))
    return {
        "final_logits": last_logits,
        "decision_margin": torch.tensor([margin], dtype=torch.float32),
        "boundary_residual": boundary,
        "upstream_activation": upstream,
    }
