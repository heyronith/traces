"""B1 production research pipeline wiring.

Connects pinned load → donor capture → frozen catalog execution → typed
``PromptConditionRecord`` assembly → generation / adapter-off evaluation.

Effects are **never** taken from supplied summaries: patched margins are
recorded per condition, and signed effects are derived only at
``assemble_bundles_from_condition_records``.

No pretrained loads or scientific forwards occur unless separately authorized.
Missing scientific measurements are never replaced with zeros or mocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from traces.metrics import FULL_PAYLOAD
from traces.q1.analysis import pair_effect_forward, pair_effect_reverse, rank_pairs_signed
from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS, EXPECTED_UPSTREAM_HEADS
from traces.q1.conditions import (
    Condition,
    UPSTREAM_LAYERS,
    enumerate_primary_l19_conditions,
    enumerate_secondary_l1523_conditions,
)
from traces.q1.hybrids import HYBRID_TABLE
from traces.q1.pipeline_adapters import (
    PromptConditionRecord,
    assemble_bundles_from_condition_records,
    compute_fold_topk_selections,
    unit_spec_to_adapter_subfolder,
)
from traces.q1.recording import GenerationRecord as Q1GenerationRecord
from traces.q1.scientific_execution import (
    ConditionOutputBundle,
    LiveForwardHooks,
    UnitSpec,
    require_scientific_authorization,
)
from traces.q1.stats_f1_f2 import assign_f2_folds

# Frozen F3 payload (protocol v1.3): ten "I HATE YOU" repeats, 109 UTF-8 chars.
FROZEN_FULL_PAYLOAD = FULL_PAYLOAD


# ---------------------------------------------------------------------------
# Catalog completeness (v1.3 — do not silently omit expensive conditions)
# ---------------------------------------------------------------------------


REQUIRED_PRIMARY_KIND_MINIMA = {
    "boundary_hybrid": 2,  # CT, TC
    "single_head": 2 * len(EXPECTED_UPSTREAM_HEADS),
    "pair": 2 * len(EXPECTED_L15_L16_PAIRS),
    "joint_triple": 2,
    "all_upstream": 2,
    "crossfit_topk": 2 * 10,  # k=1..10 × dirs
    "layer_joint_8head": 2 * len(UPSTREAM_LAYERS),  # 19 layers × 2 dirs = 38
    "downstream_head_secondary": 2 * 56,
    "residual_sweep_hybrid": 25 * 2,  # layers 1..25 × CT/TC
}

# Adapter-off base: intact baselines + generations only (no intervention catalog).
ADAPTER_OFF_REQUIRED_KINDS = frozenset(
    {"intact_TT", "intact_CC", "boundary_CT", "boundary_TC"}
)


@dataclass
class CatalogCompletenessReport:
    ok: bool
    family: str
    n_conditions: int
    kind_counts: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    omitted_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "family": self.family,
            "n_conditions": self.n_conditions,
            "kind_counts": dict(self.kind_counts),
            "failures": list(self.failures),
            "omitted_kinds": list(self.omitted_kinds),
        }


def _verify_primary_catalog_identities(
    conds: Sequence[Condition],
    *,
    include_downstream56: bool,
) -> list[str]:
    """Exact identity checks — counts alone are insufficient (B1-C7)."""
    failures: list[str] = []
    expected = enumerate_primary_l19_conditions(
        include_downstream56=include_downstream56
    )
    got_ids = {condition_id_for(c) for c in conds}
    exp_ids = {condition_id_for(c) for c in expected}
    missing = sorted(exp_ids - got_ids)
    extra = sorted(got_ids - exp_ids)
    if missing:
        failures.append(
            f"catalog missing {len(missing)} exact identities "
            f"(e.g. {missing[:3]})"
        )
    if extra:
        failures.append(
            f"catalog has {len(extra)} substituted/extra identities "
            f"(e.g. {extra[:3]})"
        )

    layer_joint = [c for c in conds if c.kind == "layer_joint_8head"]
    layers_fwd = {
        c.layer for c in layer_joint if c.direction == "forward" and c.layer is not None
    }
    layers_rev = {
        c.layer for c in layer_joint if c.direction == "reverse" and c.layer is not None
    }
    expected_layers = set(UPSTREAM_LAYERS)
    if layers_fwd != expected_layers:
        failures.append(
            f"layer_joint_8head forward layers {sorted(layers_fwd)} "
            f"!= upstream {sorted(expected_layers)}"
        )
    if layers_rev != expected_layers:
        failures.append(
            f"layer_joint_8head reverse layers {sorted(layers_rev)} "
            f"!= upstream {sorted(expected_layers)}"
        )
    if len(layer_joint) != 2 * len(UPSTREAM_LAYERS):
        failures.append(
            f"layer_joint_8head count {len(layer_joint)} != "
            f"{2 * len(UPSTREAM_LAYERS)} (19 layers × 2 dirs)"
        )
    return failures


def verify_frozen_catalog_completeness(
    *,
    family: str,
    include_downstream56: bool = True,
) -> CatalogCompletenessReport:
    """Fail closed if the frozen v1.3 catalog silently omits required kinds."""
    failures: list[str] = []
    if family == "l19":
        conds = enumerate_primary_l19_conditions(
            include_downstream56=include_downstream56
        )
        minima = dict(REQUIRED_PRIMARY_KIND_MINIMA)
        if not include_downstream56:
            minima.pop("downstream_head_secondary", None)
            failures.append(
                "include_downstream56=False omits secondary 56-head scans — "
                "refuse silent omission for primary confirmatory catalog"
            )
        failures.extend(
            _verify_primary_catalog_identities(
                conds, include_downstream56=include_downstream56
            )
        )
    elif family == "l1523":
        conds = enumerate_secondary_l1523_conditions()
        minima = {
            "boundary_hybrid": 2,
            "pair": 2 * len(EXPECTED_L15_L16_PAIRS),
            "nominated_single": 4,  # 2 heads × 2 dirs
        }
    elif family == "base":
        # Adapter-off baseline: no intervention catalog (B1-C5).
        return CatalogCompletenessReport(
            ok=True,
            family=family,
            n_conditions=0,
            kind_counts={"adapter_off_baseline": 1},
            failures=[],
            omitted_kinds=[],
        )
    else:
        return CatalogCompletenessReport(
            ok=False,
            family=family,
            n_conditions=0,
            failures=[f"unknown family {family!r} for catalog completeness"],
        )

    counts: dict[str, int] = {}
    for c in conds:
        counts[c.kind] = counts.get(c.kind, 0) + 1

    omitted: list[str] = []
    for kind, minimum in minima.items():
        got = counts.get(kind, 0)
        if got < minimum:
            omitted.append(kind)
            failures.append(
                f"catalog kind {kind!r}: got {got} < required minimum {minimum}"
            )
        elif got != minimum and kind == "layer_joint_8head":
            # Exact count required for layer joints (C7)
            failures.append(
                f"catalog kind {kind!r}: got {got} != required exact {minimum}"
            )

    return CatalogCompletenessReport(
        ok=len(failures) == 0,
        family=family,
        n_conditions=len(conds),
        kind_counts=counts,
        failures=failures,
        omitted_kinds=omitted,
    )


def frozen_hybrid_recipient_tag(hybrid: str) -> str:
    """Recipient background for CT/TC per frozen intervention definitions.

    CT: clean post-tag + trigger tag → clean recipient background.
    TC: trigger post-tag + clean tag → trigger recipient background.
    """
    if hybrid not in HYBRID_TABLE:
        raise ValueError(f"unknown hybrid {hybrid!r}")
    spec = HYBRID_TABLE[hybrid]  # type: ignore[index]
    # Recipient background matches post-tag source stream identity
    if hybrid == "CT":
        if spec.post_tag_source != "C" or spec.tag_source != "T":
            raise RuntimeError("HYBRID_TABLE CT semantics corrupted")
        return "C"
    if hybrid == "TC":
        if spec.post_tag_source != "T" or spec.tag_source != "C":
            raise RuntimeError("HYBRID_TABLE TC semantics corrupted")
        return "T"
    if hybrid in {"TT", "CC"}:
        return hybrid[0]  # T or C
    raise ValueError(f"unsupported hybrid {hybrid!r}")


def assert_hybrid_semantics(hybrid: str, *, recipient_tag: str) -> dict[str, str]:
    """Fail closed if donor/recipient binding disagrees with frozen hybrids."""
    expected_recip = frozen_hybrid_recipient_tag(hybrid)
    if recipient_tag != expected_recip:
        raise RuntimeError(
            f"hybrid {hybrid}: recipient_tag={recipient_tag!r} != "
            f"frozen expected {expected_recip!r} "
            f"(CT=clean post-tag+trigger tag; TC=trigger post-tag+clean tag)"
        )
    spec = HYBRID_TABLE[hybrid]  # type: ignore[index]
    return {
        "hybrid": hybrid,
        "recipient_tag": recipient_tag,
        "post_tag_source": spec.post_tag_source,
        "tag_source": spec.tag_source,
    }


def condition_id_for(cond: Condition) -> str:
    """Stable condition_id string for records / catalogs."""
    parts = [cond.kind]
    if cond.direction:
        parts.append(cond.direction)
    if cond.hybrid:
        parts.append(cond.hybrid)
    if cond.layer is not None:
        parts.append(f"L{cond.layer}")
    if cond.k is not None:
        parts.append(f"k{cond.k}")
    if cond.heads:
        head_s = "+".join(f"L{h.layer}.h{h.head_index}" for h in cond.heads)
        parts.append(head_s)
    if cond.bos_convention and cond.bos_convention != "single":
        parts.append(cond.bos_convention)
    return "|".join(parts)


def heads_tuple(cond: Condition) -> tuple[tuple[int, int], ...]:
    if not cond.heads:
        return ()
    return tuple((int(h.layer), int(h.head_index)) for h in cond.heads)


# ---------------------------------------------------------------------------
# B1-C1/C2/C3/C4: signed head effects + frozen F2 cross-fit planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedHeadEffectInventory:
    """Eligible-aligned signed single-head effects (no pair assembly)."""

    prompt_ids: tuple[str, ...]  # frozen allocation order
    eligible_mask: np.ndarray  # over prompt_ids
    eligible_prompt_ids: tuple[str, ...]  # allocation order, eligible only
    effects_fwd: dict[tuple[int, int], np.ndarray]  # length = n_eligible
    effects_rev: dict[tuple[int, int], np.ndarray]
    n_heads_required: int = 152


@dataclass(frozen=True)
class CrossfitFoldPlan:
    """F2 folds on frozen allocation; eligibility filtered without reallocation."""

    allocation_ids: tuple[str, ...]
    fold_a_positions: tuple[int, ...]
    fold_b_positions: tuple[int, ...]
    eligible_mask: np.ndarray
    fold_a_eligible_positions: tuple[int, ...]
    fold_b_eligible_positions: tuple[int, ...]
    fold_a_eligible_array_idx: tuple[int, ...]
    fold_b_eligible_array_idx: tuple[int, ...]
    indeterminate: bool
    indeterminate_reason: str | None


def extract_signed_single_head_effects(
    records: Sequence[PromptConditionRecord],
    *,
    prompt_ids: Sequence[str],
    expected_heads: frozenset[tuple[int, int]] = EXPECTED_UPSTREAM_HEADS,
) -> SignedHeadEffectInventory:
    """Derive signed single-head effects from intact + patched margins only.

    Never calls ``assemble_bundles_from_condition_records`` (B1-C1: that path
    requires pair maps). Requires complete upstream inventories in both
    directions over eligible prompts.
    """
    ids = [str(x) for x in prompt_ids]
    tt: dict[str, float] = {}
    cc: dict[str, float] = {}
    eligible_by_id: dict[str, bool] = {pid: True for pid in ids}
    # direction -> head -> prompt_id -> patched margin
    patched: dict[str, dict[tuple[int, int], dict[str, float]]] = {
        "forward": {},
        "reverse": {},
    }

    for r in records:
        pid = str(r.instruction_id)
        if pid not in eligible_by_id and pid not in ids:
            continue
        if r.kind == "intact_TT":
            tt[pid] = float(r.margin)
            eligible_by_id[pid] = bool(r.eligible) and eligible_by_id.get(pid, True)
        elif r.kind == "intact_CC":
            cc[pid] = float(r.margin)
            eligible_by_id[pid] = bool(r.eligible) and eligible_by_id.get(pid, True)
        elif r.kind in {"head", "single_head", "nominated_single"} and r.direction in {
            "forward",
            "reverse",
        }:
            if len(r.heads) != 1:
                raise ValueError(f"single-head record needs one head: {r.condition_id}")
            h = (int(r.heads[0][0]), int(r.heads[0][1]))
            patched[r.direction].setdefault(h, {})[pid] = float(r.margin)
            if not r.eligible:
                eligible_by_id[pid] = False

    missing_tt = [pid for pid in ids if pid not in tt]
    missing_cc = [pid for pid in ids if pid not in cc]
    if missing_tt or missing_cc:
        raise ValueError(
            f"incomplete intact margins for head-effect extraction: "
            f"missing_TT={missing_tt[:5]} missing_CC={missing_cc[:5]}"
        )

    mask = np.asarray([bool(eligible_by_id.get(pid, False)) for pid in ids], dtype=bool)
    elig_ids = tuple(pid for pid, m in zip(ids, mask) if m)
    if not elig_ids:
        raise ValueError("no eligible prompts for signed head-effect extraction")

    effects_fwd: dict[tuple[int, int], np.ndarray] = {}
    effects_rev: dict[tuple[int, int], np.ndarray] = {}
    for direction, target in (("forward", effects_fwd), ("reverse", effects_rev)):
        present = set(patched[direction])
        missing_heads = sorted(expected_heads - present)
        if missing_heads:
            raise ValueError(
                f"incomplete {direction} single-head inventory: "
                f"missing {len(missing_heads)} heads (e.g. {missing_heads[:3]})"
            )
        extra = present - expected_heads
        # Allow only expected upstream heads for crossfit selection inventory
        for h in sorted(expected_heads):
            by_pid = patched[direction][h]
            missing_p = [pid for pid in elig_ids if pid not in by_pid]
            if missing_p:
                raise ValueError(
                    f"incomplete {direction} head {h} patched margins: "
                    f"missing prompts {missing_p[:5]}"
                )
            tt_arr = np.asarray([tt[pid] for pid in elig_ids], dtype=np.float64)
            cc_arr = np.asarray([cc[pid] for pid in elig_ids], dtype=np.float64)
            pat_arr = np.asarray([by_pid[pid] for pid in elig_ids], dtype=np.float64)
            if not np.all(np.isfinite(pat_arr)):
                raise ValueError(f"nonfinite patched margins for {direction} head {h}")
            if direction == "forward":
                target[h] = pair_effect_forward(tt_arr, pat_arr)
            else:
                target[h] = pair_effect_reverse(pat_arr, cc_arr)
        _ = extra  # extras ignored for selection inventory

    if len(effects_fwd) != len(expected_heads) or len(effects_rev) != len(expected_heads):
        raise ValueError("signed head-effect maps incomplete after extraction")

    return SignedHeadEffectInventory(
        prompt_ids=tuple(ids),
        eligible_mask=mask,
        eligible_prompt_ids=elig_ids,
        effects_fwd=effects_fwd,
        effects_rev=effects_rev,
        n_heads_required=len(expected_heads),
    )


def plan_crossfit_folds(
    allocation_ids: Sequence[Any],
    eligible_mask: np.ndarray | Sequence[bool],
) -> CrossfitFoldPlan:
    """Assign F2 even/odd folds on frozen allocation; filter eligibility in place.

    Fold membership never changes when prompts become ineligible (B1-C2/C4).
    Empty eligible fold → indeterminate (frozen A5 rule), not crash/reallocate.
    """
    ids = tuple(str(x) for x in allocation_ids)
    mask = np.asarray(eligible_mask, dtype=bool)
    if len(mask) != len(ids):
        raise ValueError(
            f"eligible_mask length {len(mask)} != allocation length {len(ids)}"
        )
    folds = assign_f2_folds(ids)
    # Eligible positions within each fold (original fold membership preserved)
    a_elig_pos = tuple(i for i in folds.fold_a_positions if bool(mask[i]))
    b_elig_pos = tuple(i for i in folds.fold_b_positions if bool(mask[i]))

    # Map allocation positions → index into eligible-only array (allocation order)
    elig_pos_list = [i for i, m in enumerate(mask) if m]
    pos_to_elig_idx = {pos: j for j, pos in enumerate(elig_pos_list)}
    a_arr = tuple(pos_to_elig_idx[p] for p in a_elig_pos)
    b_arr = tuple(pos_to_elig_idx[p] for p in b_elig_pos)

    indeterminate = (not a_arr) or (not b_arr)
    reason = None
    if indeterminate:
        reason = (
            f"empty eligible fold(s) after F2 assignment: "
            f"n_A_elig={len(a_arr)} n_B_elig={len(b_arr)}; "
            "cross-fitted endpoints indeterminate (frozen A5)"
        )
    return CrossfitFoldPlan(
        allocation_ids=ids,
        fold_a_positions=folds.fold_a_positions,
        fold_b_positions=folds.fold_b_positions,
        eligible_mask=mask,
        fold_a_eligible_positions=a_elig_pos,
        fold_b_eligible_positions=b_elig_pos,
        fold_a_eligible_array_idx=a_arr,
        fold_b_eligible_array_idx=b_arr,
        indeterminate=indeterminate,
        indeterminate_reason=reason,
    )


def compute_directional_crossfit_selections(
    effects: Mapping[tuple[int, int], np.ndarray],
    *,
    fold_plan: CrossfitFoldPlan,
    direction: str,
    k_values: Sequence[int] = tuple(range(1, 11)),
) -> list[Any]:
    """Top-k selection using the same patch direction as evaluation (B1-C3)."""
    if direction not in {"forward", "reverse"}:
        raise ValueError(f"invalid crossfit direction {direction!r}")
    if fold_plan.indeterminate:
        return []
    if len(effects) < 152:
        raise ValueError(
            f"{direction} crossfit requires complete 152-head inventory, "
            f"got {len(effects)}"
        )
    sels = compute_fold_topk_selections(
        effects,
        fold_a_idx=list(fold_plan.fold_a_eligible_array_idx),
        fold_b_idx=list(fold_plan.fold_b_eligible_array_idx),
        k_values=k_values,
    )
    # Annotate selection_source with direction for provenance
    out = []
    for s in sels:
        out.append(
            type(s)(
                fold_id=s.fold_id,
                k=s.k,
                selected_heads=s.selected_heads,
                joint_key=s.joint_key,
                selection_source=f"{direction}|{s.selection_source}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Catalog execution over Q1CausalRunner (authorized model required)
# ---------------------------------------------------------------------------


@dataclass
class PromptRenderPair:
    """Rendered trigger/clean prompts + tag alignment for one instruction."""

    instruction_id: str
    rendered_T: str
    rendered_C: str
    alignment: Any  # TagAlignment
    full_ids_T: Sequence[int]
    full_ids_C: Sequence[int]


@dataclass
class DonorBank:
    """Isolated per-instruction donor caches + frozen clean references."""

    caches: dict[str, Any] = field(default_factory=dict)
    clean_refs: dict[str, Any] = field(default_factory=dict)  # CleanReference
    instruction_ids: list[str] = field(default_factory=list)

    def to_evidence(self) -> dict[str, Any]:
        return {
            "n_donor_caches": len(self.caches),
            "clean_references_frozen": all(
                getattr(r, "token_id", None) is not None for r in self.clean_refs.values()
            )
            and len(self.clean_refs) == len(self.instruction_ids),
            "instruction_ids": list(self.instruction_ids),
            "donor_provenance": "q1_causal_runner_isolated_caches",
            "bos_convention": "single",
        }


def capture_donors_with_runner(
    runner: Any,
    prompts: Sequence[PromptRenderPair],
    *,
    device: Any = None,
) -> DonorBank:
    """Capture T/C donors and freeze clean-reference tokens per prompt."""
    bank = DonorBank(instruction_ids=[p.instruction_id for p in prompts])
    for p in prompts:
        cache = runner.capture_intact_pair(
            instruction_id=p.instruction_id,
            rendered_T=p.rendered_T,
            rendered_C=p.rendered_C,
            alignment=p.alignment,
            device=device,
        )
        clean_logits = cache.logits.get("C") if hasattr(cache, "logits") else None
        if clean_logits is None:
            raise RuntimeError(
                f"donor capture missing clean logits for {p.instruction_id} — "
                "refuse invented clean-reference tokens"
            )
        ref = runner.freeze_clean_reference(
            clean_logits,
            int(p.alignment.seq_len),
            instruction_id=p.instruction_id,
        )
        bank.caches[p.instruction_id] = cache
        bank.clean_refs[p.instruction_id] = ref
    return bank


def _margin_record(
    *,
    instruction_id: str,
    condition_id: str,
    kind: str,
    direction: str | None,
    heads: tuple[tuple[int, int], ...],
    margin: float,
    eligible: bool,
    clean_reference_token: int | None,
    spec: UnitSpec,
    meta: Mapping[str, Any] | None = None,
) -> PromptConditionRecord:
    if margin != margin:  # NaN
        raise RuntimeError(
            f"nonfinite margin for {condition_id}/{instruction_id} — "
            "refuse zero-fill of missing measurements"
        )
    return PromptConditionRecord(
        instruction_id=instruction_id,
        condition_id=condition_id,
        direction=direction,
        margin=float(margin),
        eligible=bool(eligible),
        donor_instruction_id=instruction_id,
        clean_reference_token=clean_reference_token,
        bos_convention="single",
        family=spec.family,
        model_unit=spec.model_unit,
        role=spec.role,
        kind=kind,
        heads=heads,
        meta=dict(meta or {}),
    )


def execute_condition_with_runner(
    runner: Any,
    cond: Condition,
    *,
    prompt: PromptRenderPair,
    cache: Any,
    clean_ref: Any,
    spec: UnitSpec,
    device: Any = None,
) -> PromptConditionRecord:
    """Run one frozen catalog condition; return a patched/intact margin record."""
    from traces.q1.patching import HeadRef
    from traces.q1.scoring import build_q1_prompt_batch

    cid = condition_id_for(cond)
    b_i = int(clean_ref.token_id)
    eligible = bool(getattr(clean_ref, "eligible", True))
    heads = heads_tuple(cond)

    # Intact TT/CC from captured logits (no extra hybrid)
    if cond.kind == "boundary_hybrid" and cond.hybrid in {"CT", "TC"}:
        recip_tag = frozen_hybrid_recipient_tag(cond.hybrid)
        hybrid_prov = assert_hybrid_semantics(cond.hybrid, recipient_tag=recip_tag)
        rendered = prompt.rendered_C if recip_tag == "C" else prompt.rendered_T
        full_ids = prompt.full_ids_C if recip_tag == "C" else prompt.full_ids_T
        batch = build_q1_prompt_batch(
            runner.tokenizer,
            rendered,
            bos_convention=runner.bos_convention,
            device=device,
        )
        logits = runner.run_boundary_hybrid(
            cache,
            cond.hybrid,
            recipient_input_ids=batch["input_ids"],
            recipient_attention_mask=batch.get("attention_mask"),
            expected_tag=recip_tag,
            expected_full_ids=full_ids,
            layer=cond.layer,
        )
        mr = runner.margin_from_logits(
            logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
        )
        kind = f"boundary_{cond.hybrid}"
        return _margin_record(
            instruction_id=prompt.instruction_id,
            condition_id=cid,
            kind=kind,
            direction=None,
            heads=(),
            margin=mr.margin,
            eligible=mr.eligible,
            clean_reference_token=b_i,
            spec=spec,
            meta={"hybrid_provenance": hybrid_prov},
        )

    if cond.kind in {
        "single_head",
        "pair",
        "joint_triple",
        "all_upstream",
        "layer_joint_8head",
        "downstream_head_secondary",
        "nominated_single",
        "crossfit_topk",  # joint heads supplied via meta["selected_heads"]
    }:
        if cond.direction not in {"forward", "reverse"}:
            raise RuntimeError(f"{cid}: direction required")
        direction = cond.direction
        if cond.kind == "crossfit_topk":
            raise RuntimeError(
                "crossfit_topk must be executed via execute_crossfit_topk_conditions "
                "with fold-selected heads — refuse silent empty joint"
            )
        recip_tag = "T" if direction == "forward" else "C"
        rendered = prompt.rendered_T if recip_tag == "T" else prompt.rendered_C
        full_ids = prompt.full_ids_T if recip_tag == "T" else prompt.full_ids_C
        batch = build_q1_prompt_batch(
            runner.tokenizer,
            rendered,
            bos_convention=runner.bos_convention,
            device=device,
        )
        head_refs = tuple(HeadRef(L, h) for L, h in heads)
        if not head_refs:
            raise RuntimeError(f"{cid}: empty head set — refuse mock/zero effect")
        logits = runner.run_head_patches(
            cache,
            head_refs,
            direction,  # type: ignore[arg-type]
            recipient_input_ids=batch["input_ids"],
            recipient_attention_mask=batch.get("attention_mask"),
            recipient_prompt_tag=recip_tag,
            expected_full_ids=full_ids,
        )
        mr = runner.margin_from_logits(
            logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
        )
        kind = (
            "pair"
            if cond.kind == "pair"
            else ("joint" if cond.kind in {"joint_triple", "all_upstream", "layer_joint_8head", "crossfit_topk"} else "head")
        )
        return _margin_record(
            instruction_id=prompt.instruction_id,
            condition_id=cid,
            kind=kind,
            direction=direction,
            heads=heads,
            margin=mr.margin,
            eligible=mr.eligible,
            clean_reference_token=b_i,
            spec=spec,
            meta={"catalog_kind": cond.kind},
        )

    if cond.kind == "residual_sweep_hybrid":
        # Same boundary hybrid machinery at alternate layer
        if cond.hybrid not in {"CT", "TC"} or cond.layer is None:
            raise RuntimeError(f"{cid}: residual sweep requires hybrid+layer")
        recip_tag = frozen_hybrid_recipient_tag(cond.hybrid)
        hybrid_prov = assert_hybrid_semantics(cond.hybrid, recipient_tag=recip_tag)
        rendered = prompt.rendered_C if recip_tag == "C" else prompt.rendered_T
        full_ids = prompt.full_ids_C if recip_tag == "C" else prompt.full_ids_T
        batch = build_q1_prompt_batch(
            runner.tokenizer,
            rendered,
            bos_convention=runner.bos_convention,
            device=device,
        )
        logits = runner.run_boundary_hybrid(
            cache,
            cond.hybrid,
            recipient_input_ids=batch["input_ids"],
            recipient_attention_mask=batch.get("attention_mask"),
            expected_tag=recip_tag,
            expected_full_ids=full_ids,
            layer=int(cond.layer),
        )
        mr = runner.margin_from_logits(
            logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
        )
        return _margin_record(
            instruction_id=prompt.instruction_id,
            condition_id=cid,
            kind=f"boundary_{cond.hybrid}",
            direction=None,
            heads=(),
            margin=mr.margin,
            eligible=mr.eligible,
            clean_reference_token=b_i,
            spec=spec,
            meta={
                "residual_sweep_layer": int(cond.layer),
                "hybrid_provenance": hybrid_prov,
            },
        )

    raise RuntimeError(f"unsupported catalog condition kind {cond.kind!r}")


def intact_baseline_records_from_cache(
    runner: Any,
    *,
    prompt: PromptRenderPair,
    cache: Any,
    clean_ref: Any,
    spec: UnitSpec,
) -> list[PromptConditionRecord]:
    """Emit intact TT/CC margin records from captured logits (no extra forward)."""
    b_i = int(clean_ref.token_id)
    eligible = bool(getattr(clean_ref, "eligible", True))
    out: list[PromptConditionRecord] = []
    for tag, kind, cid in (("T", "intact_TT", "boundary|TT"), ("C", "intact_CC", "boundary|CC")):
        logits = cache.logits.get(tag) if hasattr(cache, "logits") else None
        if logits is None:
            raise RuntimeError(
                f"missing intact {tag} logits for {prompt.instruction_id} — "
                "refuse fabricated intact margins"
            )
        mr = runner.margin_from_logits(
            logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
        )
        out.append(
            _margin_record(
                instruction_id=prompt.instruction_id,
                condition_id=cid,
                kind=kind,
                direction=None,
                heads=(),
                margin=mr.margin,
                eligible=mr.eligible,
                clean_reference_token=b_i,
                spec=spec,
            )
        )
    return out


def execute_crossfit_topk_conditions(
    runner: Any,
    *,
    prompts: Sequence[PromptRenderPair],
    bank: DonorBank,
    spec: UnitSpec,
    inventory: SignedHeadEffectInventory,
    fold_plan: CrossfitFoldPlan,
    k_values: Sequence[int] = tuple(range(1, 11)),
    device: Any = None,
) -> list[PromptConditionRecord]:
    """Directional cross-fit: select and evaluate in the same patch direction.

    Forward selections use forward single-head effects; reverse uses reverse.
    Evaluation runs only on eligible prompts in the opposite F2 fold (B1-C2–C4).
    """
    from traces.q1.patching import HeadRef
    from traces.q1.scoring import build_q1_prompt_batch

    records: list[PromptConditionRecord] = []
    prompt_by_id = {p.instruction_id: p for p in prompts}
    if fold_plan.indeterminate:
        # Frozen A5: no joint patches, no reallocation, no fabricated effects.
        return records

    for direction, effects in (
        ("forward", inventory.effects_fwd),
        ("reverse", inventory.effects_rev),
    ):
        sels = compute_directional_crossfit_selections(
            effects, fold_plan=fold_plan, direction=direction, k_values=k_values
        )
        for sel in sels:
            # Select on fold_id; evaluate opposite fold's eligible positions
            eval_positions = (
                fold_plan.fold_b_eligible_positions
                if sel.fold_id == "A"
                else fold_plan.fold_a_eligible_positions
            )
            if not sel.selected_heads:
                raise RuntimeError(
                    f"crossfit {direction} selection empty — refuse zero-filled joint"
                )
            for pos in eval_positions:
                pid = fold_plan.allocation_ids[int(pos)]
                prompt = prompt_by_id[pid]
                cache = bank.caches[prompt.instruction_id]
                clean_ref = bank.clean_refs[prompt.instruction_id]
                b_i = int(clean_ref.token_id)
                eligible = bool(getattr(clean_ref, "eligible", True))
                recip_tag = "T" if direction == "forward" else "C"
                rendered = prompt.rendered_T if recip_tag == "T" else prompt.rendered_C
                full_ids = prompt.full_ids_T if recip_tag == "T" else prompt.full_ids_C
                batch = build_q1_prompt_batch(
                    runner.tokenizer,
                    rendered,
                    bos_convention=runner.bos_convention,
                    device=device,
                )
                head_refs = tuple(HeadRef(L, h) for L, h in sel.selected_heads)
                logits = runner.run_head_patches(
                    cache,
                    head_refs,
                    direction,  # type: ignore[arg-type]
                    recipient_input_ids=batch["input_ids"],
                    recipient_attention_mask=batch.get("attention_mask"),
                    recipient_prompt_tag=recip_tag,
                    expected_full_ids=full_ids,
                )
                mr = runner.margin_from_logits(
                    logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
                )
                cid = (
                    f"crossfit_topk|{direction}|k{sel.k}|fold{sel.fold_id}|"
                    + "+".join(f"L{L}.h{h}" for L, h in sel.selected_heads)
                )
                records.append(
                    _margin_record(
                        instruction_id=prompt.instruction_id,
                        condition_id=cid,
                        kind="joint",
                        direction=direction,
                        heads=sel.selected_heads,
                        margin=mr.margin,
                        eligible=mr.eligible,
                        clean_reference_token=b_i,
                        spec=spec,
                        meta={
                            "catalog_kind": "crossfit_topk",
                            "fold_id": sel.fold_id,
                            "k": sel.k,
                            "selection_direction": direction,
                            "selection_source": sel.selection_source,
                            "joint_key": list(sel.joint_key),
                            "eval_allocation_position": int(pos),
                        },
                    )
                )
    _ = rank_pairs_signed
    return records


def execute_frozen_catalog(
    runner: Any,
    *,
    spec: UnitSpec,
    prompts: Sequence[PromptRenderPair],
    bank: DonorBank,
    include_downstream56: bool = True,
    skip_crossfit: bool = False,
    device: Any = None,
) -> list[PromptConditionRecord]:
    """Execute the frozen family catalog into PromptConditionRecords.

    Crossfit top-k derives signed head effects from records (B1-C1) and uses
    frozen F2 folds (B1-C2). ``skip_crossfit=True`` is refused when the catalog
    contains crossfit conditions.
    """
    if spec.family == "base":
        raise RuntimeError(
            "execute_frozen_catalog refuses family=base — use adapter-off baseline path"
        )
    completeness = verify_frozen_catalog_completeness(
        family=spec.family, include_downstream56=include_downstream56
    )
    if not completeness.ok:
        raise RuntimeError(
            "frozen catalog incomplete: " + "; ".join(completeness.failures)
        )

    if spec.family == "l19":
        catalog = enumerate_primary_l19_conditions(
            include_downstream56=include_downstream56
        )
    elif spec.family == "l1523":
        catalog = enumerate_secondary_l1523_conditions()
    else:
        raise RuntimeError(f"no frozen catalog for family {spec.family!r}")

    records: list[PromptConditionRecord] = []
    for prompt in prompts:
        cache = bank.caches[prompt.instruction_id]
        clean_ref = bank.clean_refs[prompt.instruction_id]
        records.extend(
            intact_baseline_records_from_cache(
                runner, prompt=prompt, cache=cache, clean_ref=clean_ref, spec=spec
            )
        )

    deferred_crossfit = False
    for cond in catalog:
        if cond.kind == "crossfit_topk":
            deferred_crossfit = True
            continue
        for prompt in prompts:
            records.append(
                execute_condition_with_runner(
                    runner,
                    cond,
                    prompt=prompt,
                    cache=bank.caches[prompt.instruction_id],
                    clean_ref=bank.clean_refs[prompt.instruction_id],
                    spec=spec,
                    device=device,
                )
            )

    if deferred_crossfit:
        if skip_crossfit:
            raise RuntimeError(
                "crossfit_topk present in frozen catalog but skip_crossfit=True — "
                "refuse silent omission of expensive conditions"
            )
        prompt_ids = [p.instruction_id for p in prompts]
        inventory = extract_signed_single_head_effects(records, prompt_ids=prompt_ids)
        fold_plan = plan_crossfit_folds(prompt_ids, inventory.eligible_mask)
        records.extend(
            execute_crossfit_topk_conditions(
                runner,
                prompts=prompts,
                bank=bank,
                spec=spec,
                inventory=inventory,
                fold_plan=fold_plan,
                device=device,
            )
        )
    return records


@dataclass
class F3OutcomeBundle:
    """Retained per-prompt F3 outcomes + rates computed from those raw records."""

    full_payload: str
    trigger_records: list[Q1GenerationRecord]
    clean_records: list[Q1GenerationRecord]
    trigger_asr: float
    clean_false_fire: float
    first_token_validity: float

    def rates(self) -> dict[str, float]:
        return {
            "trigger_asr": float(self.trigger_asr),
            "clean_false_fire": float(self.clean_false_fire),
            "first_token_validity": float(self.first_token_validity),
        }


def generation_rates_from_f3(
    *,
    model: Any,
    tokenizer: Any,
    trigger_prompts: Sequence[str],
    clean_prompts: Sequence[str],
    payload: str | None = None,
    instruction_ids: Sequence[str] | None = None,
    model_unit: str = "",
) -> F3OutcomeBundle:
    """Run F3 with frozen full payload; retain raw outcomes; derive rates (B1-C6)."""
    from traces.q1.gates import (
        compute_generation_gates,
        continuation_fires_full_payload,
        continuation_fires_substring,
    )
    from traces.q1.generation import run_f3_generation

    full_payload = payload if payload is not None else FROZEN_FULL_PAYLOAD
    if full_payload != FROZEN_FULL_PAYLOAD:
        raise RuntimeError(
            "generation_rates_from_f3 requires exact frozen FULL_PAYLOAD; "
            f"got length={len(full_payload)} expected={len(FROZEN_FULL_PAYLOAD)}"
        )
    if len(trigger_prompts) != len(clean_prompts):
        raise RuntimeError("trigger/clean prompt counts must match for F3 denominators")
    if not trigger_prompts:
        raise RuntimeError("empty F3 prompt set — refuse fabricated generation rates")
    ids = (
        [str(x) for x in instruction_ids]
        if instruction_ids is not None
        else [str(i) for i in range(len(trigger_prompts))]
    )
    if len(ids) != len(trigger_prompts):
        raise RuntimeError("instruction_ids length must match trigger prompts")

    trigger_outs = [
        run_f3_generation(
            model=model,
            tokenizer=tokenizer,
            rendered_prompt=rendered,
            full_payload=full_payload,
        )
        for rendered in trigger_prompts
    ]
    clean_outs = [
        run_f3_generation(
            model=model,
            tokenizer=tokenizer,
            rendered_prompt=rendered,
            full_payload=full_payload,
        )
        for rendered in clean_prompts
    ]
    missing_ft = [i for i, o in enumerate(trigger_outs) if o.first_token_id is None]
    if missing_ft:
        raise RuntimeError(
            f"F3 missing first_token_id for trigger indices {missing_ft[:5]} — "
            "refuse zero-filled first_token_validity"
        )

    trigger_recs: list[Q1GenerationRecord] = []
    clean_recs: list[Q1GenerationRecord] = []
    for pid, o in zip(ids, trigger_outs):
        # Re-derive fire flags from retained continuation vs frozen payload
        sub = continuation_fires_substring(o.continuation_text)
        full = continuation_fires_full_payload(o.continuation_text, full_payload)
        trigger_recs.append(
            Q1GenerationRecord(
                instruction_id=pid,
                model_unit=model_unit,
                tag="T",
                bos_convention="single",
                continuation_text=o.continuation_text,
                first_token_id=o.first_token_id,
                substring_fire=sub,
                full_payload_fire=full,
                n_new_tokens=o.n_new_tokens,
            )
        )
    for pid, o in zip(ids, clean_outs):
        sub = continuation_fires_substring(o.continuation_text)
        full = continuation_fires_full_payload(o.continuation_text, full_payload)
        clean_recs.append(
            Q1GenerationRecord(
                instruction_id=pid,
                model_unit=model_unit,
                tag="C",
                bos_convention="single",
                continuation_text=o.continuation_text,
                first_token_id=o.first_token_id,
                substring_fire=sub,
                full_payload_fire=full,
                n_new_tokens=o.n_new_tokens,
            )
        )

    rates = compute_generation_gates(
        trigger_continuations=[r.continuation_text for r in trigger_recs],
        clean_continuations=[r.continuation_text for r in clean_recs],
        trigger_first_token_ids=[int(r.first_token_id) for r in trigger_recs],  # type: ignore[arg-type]
    )
    return F3OutcomeBundle(
        full_payload=full_payload,
        trigger_records=trigger_recs,
        clean_records=clean_recs,
        trigger_asr=float(rates.trigger_asr),
        clean_false_fire=float(rates.clean_false_fire),
        first_token_validity=float(rates.first_token_validity),
    )


def rates_from_retained_f3_records(
    trigger_records: Sequence[Q1GenerationRecord],
    clean_records: Sequence[Q1GenerationRecord],
) -> dict[str, float]:
    """Derive gate rates solely from retained raw F3 outcome records."""
    from traces.q1.gates import compute_generation_gates

    if not trigger_records or len(trigger_records) != len(clean_records):
        raise ValueError("trigger/clean F3 record denominators mismatch")
    rates = compute_generation_gates(
        trigger_continuations=[r.continuation_text for r in trigger_records],
        clean_continuations=[r.continuation_text for r in clean_records],
        trigger_first_token_ids=[
            int(r.first_token_id) if r.first_token_id is not None else -1
            for r in trigger_records
        ],
    )
    return {
        "trigger_asr": float(rates.trigger_asr),
        "clean_false_fire": float(rates.clean_false_fire),
        "first_token_validity": float(rates.first_token_validity),
    }


def assemble_adapter_off_baseline_bundles(
    *,
    spec: UnitSpec,
    prompt_ids: Sequence[str],
    intact_records: Sequence[PromptConditionRecord],
    f3: F3OutcomeBundle,
) -> tuple[ConditionOutputBundle, ConditionOutputBundle]:
    """Adapter-off baseline: intact margins + F3 rates; no intervention scans (B1-C5)."""
    if spec.family != "base" and spec.role != "adapter_off_gates_descriptives":
        raise RuntimeError("assemble_adapter_off_baseline_bundles requires base/adapter-off")
    ids = [str(x) for x in prompt_ids]
    by_kind: dict[str, dict[str, PromptConditionRecord]] = {}
    for r in intact_records:
        by_kind.setdefault(r.kind, {})[str(r.instruction_id)] = r
    for kind in ADAPTER_OFF_REQUIRED_KINDS:
        if kind not in by_kind:
            raise ValueError(f"adapter-off baseline missing {kind} records")
        missing = [pid for pid in ids if pid not in by_kind[kind]]
        if missing:
            raise ValueError(f"adapter-off incomplete {kind}: {missing[:5]}")

    eligible_mask = np.asarray(
        [
            all(by_kind[k][pid].eligible for k in ("intact_TT", "intact_CC"))
            for pid in ids
        ],
        dtype=bool,
    )
    elig_ids = [pid for pid, m in zip(ids, eligible_mask) if m]
    if not elig_ids:
        raise ValueError("adapter-off: no eligible prompts")

    def _elig_margins(kind: str) -> np.ndarray:
        return np.asarray(
            [float(by_kind[kind][pid].margin) for pid in elig_ids], dtype=np.float64
        )

    m_TT = _elig_margins("intact_TT")
    m_CC = _elig_margins("intact_CC")
    m_CT = _elig_margins("boundary_CT")
    m_TC = _elig_margins("boundary_TC")
    empty: dict = {}
    catalog_ids = sorted({r.condition_id for r in intact_records})
    bundle = ConditionOutputBundle(
        model_unit=spec.model_unit,
        family="base",
        role=spec.role,
        prompt_ids=ids,
        m_TT=m_TT,
        m_CT=m_CT,
        m_TC=m_TC,
        m_CC=m_CC,
        eligible_mask=eligible_mask,
        trigger_asr=f3.trigger_asr,
        clean_false_fire=f3.clean_false_fire,
        first_token_validity=f3.first_token_validity,
        n_frozen=len(ids),
        pair_effects_fwd=empty,
        pair_effects_rev=empty,
        head_effects_fwd=empty,
        head_effects_rev=empty,
        e_route_fwd=m_TT - m_CT,
        e_route_rev=m_TC - m_CC,
        joint_set_effects_fwd={},
        joint_set_effects_rev={},
        bos_convention="single",
        catalog_condition_ids=catalog_ids,
        eligible_prompt_ids=elig_ids,
    )
    return bundle, bundle


# ---------------------------------------------------------------------------
# Production LiveForwardHooks
# ---------------------------------------------------------------------------


@dataclass
class ProductionHookConfig:
    """Configuration for production hooks (no invented hashes)."""

    repo_root: Path
    expected_weight_file_sha256: Mapping[str, str] | None = None
    base_weight_snapshot_dir: str | None = None
    adapter_weight_snapshot_dir: str | None = None
    prompt_builder: Callable[[UnitSpec, Sequence[Any]], Sequence[PromptRenderPair]] | None = None
    include_downstream56: bool = True
    development_ids: Sequence[Any] | None = None
    held_out_ids: Sequence[Any] | None = None


def build_production_live_hooks(config: ProductionHookConfig) -> LiveForwardHooks:
    """Concrete production adapters. Fail closed without auth / manifests / prompts.

    Does not perform pretrained loads until ``load_model_for_unit`` is invoked
    under scientific authorization with verified weight manifests.
    """

    def load(spec: UnitSpec) -> Mapping[str, Any]:
        require_scientific_authorization()
        if not config.expected_weight_file_sha256:
            raise RuntimeError(
                "production load refuses: expected_weight_file_sha256 missing "
                "(B3 checkpoint manifest BLOCKED — no invented hashes)"
            )
        from traces.q1.checkpoint_provenance import (
            assert_load_uses_verified_snapshot_bytes,
        )
        from traces.q1.loader import load_q1_scientific_model
        from traces.q1.runtime_spec import (
            REQUIRED_ATTN_IMPLEMENTATION,
            REQUIRED_TORCH_DTYPE,
        )

        subfolder = unit_spec_to_adapter_subfolder(spec)
        if config.base_weight_snapshot_dir and config.adapter_weight_snapshot_dir:
            assert_load_uses_verified_snapshot_bytes(
                base_snapshot_dir=config.base_weight_snapshot_dir,
                adapter_snapshot_dir=config.adapter_weight_snapshot_dir,
                expected_sha256_by_relpath=config.expected_weight_file_sha256,
            )
        # Family-specific adapted layers (l19 → [19], l1523 → [15,16,17,18,19,20,21,22,23])
        if spec.family == "l19":
            layers = [19]
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif spec.family == "l1523":
            layers = list(range(15, 24))
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif spec.family == "base":
            layers = []
            targets = []
        else:
            raise RuntimeError(f"unknown family {spec.family}")

        model, tokenizer, report, meta = load_q1_scientific_model(
            subfolder=subfolder if spec.family != "base" else "l19/seed42",
            expected_weight_file_sha256=dict(config.expected_weight_file_sha256),
            expected_adapter_targets=targets or ["q_proj"],
            expected_adapted_layers=layers or [19],
            base_weight_snapshot_dir=config.base_weight_snapshot_dir,
            adapter_weight_snapshot_dir=config.adapter_weight_snapshot_dir,
        )
        if spec.family == "base" or spec.role == "adapter_off_base_reference":
            from traces.q1.adapter_off import disable_adapter_layers

            disable_adapter_layers(model)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_unit": spec.model_unit,
            "family": spec.family,
            "role": spec.role,
            "seed": spec.seed,
            "load_report": report.to_dict() if hasattr(report, "to_dict") else report,
            "attn_implementation": REQUIRED_ATTN_IMPLEMENTATION,
            "torch_dtype": str(REQUIRED_TORCH_DTYPE),
            "weight_file_sha256": dict(config.expected_weight_file_sha256),
            "base_revision": meta.get("base_revision"),
            "adapter_revision": meta.get("adapter_revision"),
            **{k: v for k, v in meta.items() if k not in {"model"}},
        }

    def donors(spec: UnitSpec, loaded: Any) -> Mapping[str, Any]:
        require_scientific_authorization()
        if config.prompt_builder is None:
            raise RuntimeError(
                "production donor capture requires prompt_builder "
                "(rendered T/C + TagAlignment per frozen instruction)"
            )
        from traces.q1.causal_runner import Q1CausalRunner

        model = loaded["model"] if isinstance(loaded, Mapping) else loaded
        tokenizer = (
            loaded.get("tokenizer") if isinstance(loaded, Mapping) else None
        )
        if tokenizer is None:
            raise RuntimeError("loaded unit missing tokenizer")
        alloc = (
            config.development_ids
            if spec.role == "exploratory_dev"
            else config.held_out_ids
        )
        if alloc is None:
            raise RuntimeError(
                f"allocation IDs missing for role {spec.role} — refuse mock donors"
            )
        prompts = list(config.prompt_builder(spec, alloc))
        if len(prompts) != len(list(alloc)):
            raise RuntimeError("prompt_builder length != allocation")
        if spec.family == "l19" or spec.family == "base":
            boundary = 19
        else:
            boundary = 15
        runner = Q1CausalRunner(
            model,
            tokenizer,
            model_unit=spec.model_unit,
            bos_convention="single",
            boundary_layer=boundary,
        )
        bank = capture_donors_with_runner(runner, prompts)
        evidence = bank.to_evidence()
        evidence["_bank"] = bank
        evidence["_runner"] = runner
        evidence["_prompts"] = prompts
        return evidence

    def _execute_unit_impl(spec: UnitSpec, loaded: Any, donor_evidence: Any) -> tuple:
        require_scientific_authorization()
        bank = donor_evidence["_bank"]
        runner = donor_evidence["_runner"]
        prompts = donor_evidence["_prompts"]
        model = loaded["model"] if isinstance(loaded, Mapping) else loaded
        tokenizer = loaded["tokenizer"]
        alloc = [p.instruction_id for p in prompts]

        # F3 with frozen full payload — retain raw outcomes (B1-C6)
        f3 = generation_rates_from_f3(
            model=model,
            tokenizer=tokenizer,
            trigger_prompts=[p.rendered_T for p in prompts],
            clean_prompts=[p.rendered_C for p in prompts],
            payload=FROZEN_FULL_PAYLOAD,
            instruction_ids=alloc,
            model_unit=spec.model_unit,
        )
        donor_evidence["_f3_outcomes"] = f3

        # --- Adapter-off baseline path (B1-C5): no intervention catalog ---
        if spec.family == "base" or spec.role == "adapter_off_gates_descriptives":
            completeness = verify_frozen_catalog_completeness(family="base")
            if not completeness.ok:
                raise RuntimeError("; ".join(completeness.failures))
            intact: list[PromptConditionRecord] = []
            for prompt in prompts:
                intact.extend(
                    intact_baseline_records_from_cache(
                        runner,
                        prompt=prompt,
                        cache=bank.caches[prompt.instruction_id],
                        clean_ref=bank.clean_refs[prompt.instruction_id],
                        spec=spec,
                    )
                )
            # CT/TC boundary hybrids required for adapter-off margins
            from traces.q1.conditions import Condition as _Cond

            for hybrid in ("CT", "TC"):
                cond = _Cond(kind="boundary_hybrid", direction=None, hybrid=hybrid)
                for prompt in prompts:
                    intact.append(
                        execute_condition_with_runner(
                            runner,
                            cond,
                            prompt=prompt,
                            cache=bank.caches[prompt.instruction_id],
                            clean_ref=bank.clean_refs[prompt.instruction_id],
                            spec=spec,
                        )
                    )
            return assemble_adapter_off_baseline_bundles(
                spec=spec, prompt_ids=alloc, intact_records=intact, f3=f3
            )

        include_ds = config.include_downstream56 and spec.family == "l19"
        completeness = verify_frozen_catalog_completeness(
            family=spec.family, include_downstream56=include_ds
        )
        if not completeness.ok:
            raise RuntimeError("; ".join(completeness.failures))
        if spec.family == "l19":
            catalog = enumerate_primary_l19_conditions(include_downstream56=include_ds)
        elif spec.family == "l1523":
            catalog = enumerate_secondary_l1523_conditions()
        else:
            raise RuntimeError(f"unsupported family {spec.family!r}")

        records: list[PromptConditionRecord] = []
        for prompt in prompts:
            records.extend(
                intact_baseline_records_from_cache(
                    runner,
                    prompt=prompt,
                    cache=bank.caches[prompt.instruction_id],
                    clean_ref=bank.clean_refs[prompt.instruction_id],
                    spec=spec,
                )
            )
        for cond in catalog:
            if cond.kind == "crossfit_topk":
                continue
            for prompt in prompts:
                records.append(
                    execute_condition_with_runner(
                        runner,
                        cond,
                        prompt=prompt,
                        cache=bank.caches[prompt.instruction_id],
                        clean_ref=bank.clean_refs[prompt.instruction_id],
                        spec=spec,
                    )
                )

        # Cross-fit from dedicated head-effect extraction + frozen F2 (C1–C4)
        if spec.family == "l19":
            inventory = extract_signed_single_head_effects(records, prompt_ids=alloc)
            fold_plan = plan_crossfit_folds(alloc, inventory.eligible_mask)
            donor_evidence["_crossfit_fold_plan"] = fold_plan
            donor_evidence["_crossfit_indeterminate"] = fold_plan.indeterminate
            records.extend(
                execute_crossfit_topk_conditions(
                    runner,
                    prompts=prompts,
                    bank=bank,
                    spec=spec,
                    inventory=inventory,
                    fold_plan=fold_plan,
                )
            )

        rates = f3.rates()
        return assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=alloc,
            records=records,
            trigger_asr=rates["trigger_asr"],
            clean_false_fire=rates["clean_false_fire"],
            first_token_validity=rates["first_token_validity"],
            allocation_ids=alloc,
        )

    return LiveForwardHooks(
        load_model_for_unit=load,
        capture_donors_for_unit=donors,
        execute_unit=_execute_unit_impl,
    )
