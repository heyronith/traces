"""Production scientific pipeline adapters (B1).

Connects ``Q1CausalRunner`` condition machinery to typed
``ConditionOutputBundle`` / ``ModelUnitInput`` assembly.

Effects are calculated from raw per-prompt model-condition records — never
accepted as unverified external arrays. Mock runners exercise the wiring
without model forwards. Real Gemma forwards remain authorization-gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from traces.q1.analysis_ext import selection_key
from traces.q1.component_sets import (
    EXPECTED_L15_L16_PAIRS,
    EXPECTED_UPSTREAM_HEADS,
    validate_head_effect_map,
    validate_pair_effect_map,
)
from traces.q1.frozen_artifacts import coerce_frozen_dolly_id
from traces.q1.scientific_execution import (
    ConditionOutputBundle,
    LiveForwardHooks,
    UnitSpec as SciUnitSpec,
)


@dataclass
class PromptConditionRecord:
    """Raw per-prompt condition outcome (one instruction × one condition)."""

    instruction_id: str
    condition_id: str
    direction: str | None
    margin: float
    eligible: bool
    donor_instruction_id: str
    clean_reference_token: int | None
    bos_convention: str = "single"
    family: str = ""
    model_unit: str = ""
    role: str = ""
    kind: str = ""  # intact_T|intact_C|boundary|head|pair|joint
    heads: tuple[tuple[int, int], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FoldTopKSelection:
    """Fold-specific top-k head selection and subsequent joint evaluation key."""

    fold_id: str
    k: int
    selected_heads: tuple[tuple[int, int], ...]
    joint_key: tuple[tuple[int, int], ...]
    selection_source: str  # opposite-fold mean effects


def compute_fold_topk_selections(
    head_effects_by_prompt: Mapping[tuple[int, int], np.ndarray],
    *,
    fold_a_idx: Sequence[int],
    fold_b_idx: Sequence[int],
    k_values: Sequence[int] = tuple(range(1, 11)),
) -> list[FoldTopKSelection]:
    """Genuine fold-specific top-k: select on one fold, key joint for the other."""
    out: list[FoldTopKSelection] = []
    for k in k_values:
        for fold_id, sel_idx, _eval_idx in (
            ("A", fold_a_idx, fold_b_idx),
            ("B", fold_b_idx, fold_a_idx),
        ):
            means: dict[tuple[int, int], float] = {}
            for h, arr in head_effects_by_prompt.items():
                a = np.asarray(arr, dtype=np.float64)
                means[h] = float(np.mean(a[list(sel_idx)])) if sel_idx else float("nan")
            ranked = sorted(means.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
            selected = tuple(h for h, _ in ranked[:k])
            out.append(
                FoldTopKSelection(
                    fold_id=fold_id,
                    k=int(k),
                    selected_heads=selected,
                    joint_key=selection_key(selected),
                    selection_source=f"fold_{fold_id}_mean_effects",
                )
            )
    return out


def _eligible_only(
    values_by_id: Mapping[str, float],
    prompt_ids: Sequence[str],
    eligible_mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(eligible_mask, dtype=bool)
    return np.asarray(
        [float(values_by_id[pid]) for pid, m in zip(prompt_ids, mask) if m],
        dtype=np.float64,
    )


def assemble_bundles_from_condition_records(
    *,
    model_unit: str,
    family: str,
    role: str,
    prompt_ids: Sequence[str],
    records: Sequence[PromptConditionRecord],
    trigger_asr: float,
    clean_false_fire: float,
    first_token_validity: float,
    allocation_ids: Sequence[Any] | None = None,
) -> tuple[ConditionOutputBundle, ConditionOutputBundle]:
    """Build boundary + intervention bundles from raw per-prompt records.

    Effects are derived from the records. External unverified effect arrays are
    not accepted. When ``allocation_ids`` is provided, prompt order must match
    the frozen allocation (M1).
    """
    ids = [str(x) for x in prompt_ids]
    if allocation_ids is not None:
        expected = [str(coerce_frozen_dolly_id(x)) for x in allocation_ids]
        if ids != expected:
            raise ValueError(
                "prompt_ids do not match frozen allocation order (M1 binding)"
            )

    by_cond: dict[str, dict[str, PromptConditionRecord]] = {}
    for r in records:
        if r.model_unit and r.model_unit != model_unit:
            raise ValueError(
                f"record model_unit {r.model_unit!r} != bundle {model_unit!r}"
            )
        by_cond.setdefault(r.condition_id, {})[str(r.instruction_id)] = r

    # Eligibility from intact clean-reference identity (a != b_i)
    eligible_mask = np.ones(len(ids), dtype=bool)
    intact_keys = [k for k in by_cond if "intact" in k.lower() or k.endswith("|T") or k.endswith("|C")]
    # Prefer explicit eligible flags on TT/CC intact margins when present
    for i, pid in enumerate(ids):
        flags = [
            rec.eligible
            for cond in by_cond.values()
            for iid, rec in cond.items()
            if iid == pid
        ]
        if flags:
            # Ineligible if any intact record marks ineligible
            eligible_mask[i] = all(
                by_cond[c][pid].eligible
                for c in by_cond
                if pid in by_cond[c] and by_cond[c][pid].kind.startswith("intact")
            ) if any(
                by_cond[c][pid].kind.startswith("intact")
                for c in by_cond
                if pid in by_cond[c]
            ) else all(flags)

    def _margin_map(condition_id: str) -> dict[str, float]:
        if condition_id not in by_cond:
            raise KeyError(f"missing condition records for {condition_id}")
        return {pid: float(by_cond[condition_id][pid].margin) for pid in ids if pid in by_cond[condition_id]}

    # Boundary margins TT/CT/TC/CC — require presence
    required_bounds = ("TT", "CT", "TC", "CC")
    bound_maps: dict[str, dict[str, float]] = {}
    catalog_ids: list[str] = []
    for name in required_bounds:
        # Accept either exact key or suffix match
        match = None
        for cid in by_cond:
            if cid == name or cid.endswith(f"|{name}") or cid.endswith(f"/{name}") or name in cid.split("|"):
                if cid.endswith(name) or cid == name or f"|{name}" in f"|{cid}|":
                    match = cid
                    break
        if match is None:
            # Try kind-based
            for cid, mp in by_cond.items():
                sample = next(iter(mp.values()))
                if sample.kind == f"boundary_{name}" or sample.kind == f"intact_{name}":
                    match = cid
                    break
        if match is None:
            raise KeyError(f"missing boundary/intact records for {name}")
        bound_maps[name] = _margin_map(match)
        catalog_ids.append(match)

    m_TT = _eligible_only(bound_maps["TT"], ids, eligible_mask)
    m_CT = _eligible_only(bound_maps["CT"], ids, eligible_mask)
    m_TC = _eligible_only(bound_maps["TC"], ids, eligible_mask)
    m_CC = _eligible_only(bound_maps["CC"], ids, eligible_mask)
    e_route_fwd = m_TT - m_CT
    e_route_rev = m_TC - m_CC
    n_elig = int(eligible_mask.sum())

    pair_fwd: dict[tuple[int, int], np.ndarray] = {}
    pair_rev: dict[tuple[int, int], np.ndarray] = {}
    head_fwd: dict[tuple[int, int], np.ndarray] = {}
    head_rev: dict[tuple[int, int], np.ndarray] = {}
    joint_fwd: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    joint_rev: dict[tuple[tuple[int, int], ...], np.ndarray] = {}

    for cid, mp in by_cond.items():
        sample = next(iter(mp.values()))
        vals = {pid: float(mp[pid].margin) for pid in ids if pid in mp}
        arr = _eligible_only(vals, ids, eligible_mask)
        if len(arr) != n_elig:
            continue
        direction = sample.direction or "forward"
        if sample.kind == "pair" and len(sample.heads) == 2:
            key = (sample.heads[0][1], sample.heads[1][1])  # (h15, h16) indices
            # Prefer (layer, head) pairs stored as layer-matched L15/L16
            if sample.heads[0][0] == 15 and sample.heads[1][0] == 16:
                key = (sample.heads[0][1], sample.heads[1][1])
            target = pair_fwd if direction == "forward" else pair_rev
            target[key] = arr
            catalog_ids.append(cid)
        elif sample.kind in {"head", "nominated_single", "single_head"} and len(sample.heads) == 1:
            key = sample.heads[0]
            target = head_fwd if direction == "forward" else head_rev
            target[key] = arr
            catalog_ids.append(cid)
        elif sample.kind in {"joint", "crossfit_topk", "joint_set"} and sample.heads:
            key = selection_key(sample.heads)
            target = joint_fwd if direction == "forward" else joint_rev
            target[key] = arr
            catalog_ids.append(cid)

    elig_ids = [pid for pid, m in zip(ids, eligible_mask) if m]
    boundary = ConditionOutputBundle(
        model_unit=model_unit,
        family=family,
        role=role,
        prompt_ids=ids,
        m_TT=m_TT,
        m_CT=m_CT,
        m_TC=m_TC,
        m_CC=m_CC,
        eligible_mask=eligible_mask,
        trigger_asr=trigger_asr,
        clean_false_fire=clean_false_fire,
        first_token_validity=first_token_validity,
        n_frozen=len(ids),
        pair_effects_fwd=dict(pair_fwd) if pair_fwd else {(6, 5): np.zeros(max(n_elig, 1))},
        pair_effects_rev=dict(pair_rev) if pair_rev else {(6, 5): np.zeros(max(n_elig, 1))},
        head_effects_fwd=dict(head_fwd) if head_fwd else {(15, 6): np.zeros(max(n_elig, 1))},
        head_effects_rev=dict(head_rev) if head_rev else {(15, 6): np.zeros(max(n_elig, 1))},
        e_route_fwd=e_route_fwd,
        e_route_rev=e_route_rev,
        joint_set_effects_fwd=joint_fwd,
        joint_set_effects_rev=joint_rev,
        bos_convention="single",
        catalog_condition_ids=sorted(set(catalog_ids)),
        eligible_prompt_ids=elig_ids,
    )
    # Intervention bundle carries the same margins + calculated effect maps
    intervention = ConditionOutputBundle(
        model_unit=model_unit,
        family=family,
        role=role,
        prompt_ids=ids,
        m_TT=m_TT,
        m_CT=m_CT,
        m_TC=m_TC,
        m_CC=m_CC,
        eligible_mask=eligible_mask,
        trigger_asr=trigger_asr,
        clean_false_fire=clean_false_fire,
        first_token_validity=first_token_validity,
        n_frozen=len(ids),
        pair_effects_fwd=dict(boundary.pair_effects_fwd),
        pair_effects_rev=dict(boundary.pair_effects_rev),
        head_effects_fwd=dict(boundary.head_effects_fwd),
        head_effects_rev=dict(boundary.head_effects_rev),
        e_route_fwd=e_route_fwd,
        e_route_rev=e_route_rev,
        joint_set_effects_fwd=dict(joint_fwd),
        joint_set_effects_rev=dict(joint_rev),
        bos_convention="single",
        catalog_condition_ids=sorted(set(catalog_ids)),
        eligible_prompt_ids=elig_ids,
    )
    return boundary, intervention


def mock_condition_records_for_unit(
    spec: SciUnitSpec,
    prompt_ids: Sequence[Any],
    *,
    nominated_effect: float = 2.0,
    other_pair_effect: float = 0.1,
    rng_seed: int = 0,
    include_complete_64: bool = True,
    include_nominated_heads: bool = True,
    include_upstream_heads: bool = True,
    include_joints_for_k: Sequence[int] = (1, 2),
    ineligible_fraction: float = 0.0,
) -> list[PromptConditionRecord]:
    """Synthetic raw records for adapter integration tests (no forwards)."""
    ids = [str(x) for x in prompt_ids]
    n = len(ids)
    rng = np.random.default_rng(rng_seed + (spec.seed or 0))
    n_inelig = int(round(ineligible_fraction * n))
    inelig = set(ids[:n_inelig])
    records: list[PromptConditionRecord] = []

    def _rec(
        *,
        condition_id: str,
        kind: str,
        direction: str | None,
        heads: tuple[tuple[int, int], ...] = (),
        margin_fn: Callable[[str], float],
    ) -> None:
        for pid in ids:
            records.append(
                PromptConditionRecord(
                    instruction_id=pid,
                    condition_id=condition_id,
                    direction=direction,
                    margin=float(margin_fn(pid)),
                    eligible=pid not in inelig,
                    donor_instruction_id=pid,
                    clean_reference_token=0 if pid not in inelig else 235285,
                    bos_convention="single",
                    family=spec.family,
                    model_unit=spec.model_unit,
                    role=spec.role,
                    kind=kind,
                    heads=heads,
                )
            )

    # Intact / boundary
    for name, base in (("TT", 2.0), ("CT", 0.2), ("TC", 1.8), ("CC", 0.0)):
        _rec(
            condition_id=f"boundary|{name}",
            kind=f"intact_{name}" if name in ("TT", "CC") else f"boundary_{name}",
            direction=None,
            margin_fn=lambda pid, b=base: b + float(rng.normal(0, 0.01)),
        )

    if include_complete_64:
        for h15, h16 in EXPECTED_L15_L16_PAIRS:
            eff = nominated_effect if (h15, h16) == (6, 5) else other_pair_effect
            for direction in ("forward", "reverse"):
                _rec(
                    condition_id=f"pair|{direction}|L15.h{h15}+L16.h{h16}",
                    kind="pair",
                    direction=direction,
                    heads=((15, h15), (16, h16)),
                    margin_fn=lambda pid, e=eff: e,
                )

    if include_nominated_heads:
        for direction in ("forward", "reverse"):
            for L, h, eff in ((15, 6, 0.5), (16, 5, 0.4)):
                _rec(
                    condition_id=f"head|{direction}|L{L}.h{h}",
                    kind="head",
                    direction=direction,
                    heads=((L, h),),
                    margin_fn=lambda pid, e=eff: e,
                )

    if include_upstream_heads:
        for direction in ("forward", "reverse"):
            for L, h in EXPECTED_UPSTREAM_HEADS:
                if (L, h) in {(15, 6), (16, 5)}:
                    continue
                _rec(
                    condition_id=f"head|{direction}|L{L}.h{h}",
                    kind="head",
                    direction=direction,
                    heads=((L, h),),
                    margin_fn=lambda pid: 0.05,
                )

    # Joint sets for fold top-k evaluation keys (nominated pair as k=2 example)
    for k in include_joints_for_k:
        heads = tuple(
            sorted(
                [(15, 6), (16, 5)][:k]
                if k <= 2
                else [(15, 6), (16, 5)] + [(0, i) for i in range(k - 2)]
            )
        )
        for direction in ("forward", "reverse"):
            _rec(
                condition_id=f"joint|{direction}|k{k}",
                kind="joint",
                direction=direction,
                heads=heads,
                margin_fn=lambda pid: 0.8,
            )

    return records


def build_live_hooks_from_record_provider(
    provider: Callable[[SciUnitSpec], tuple[list[PromptConditionRecord], dict[str, float]]],
    *,
    allocation_ids_for_role: Callable[[SciUnitSpec], Sequence[Any] | None] | None = None,
) -> LiveForwardHooks:
    """Wire LiveForwardHooks that assemble bundles from a record provider.

    ``provider(spec) -> (records, gate_rates)`` where gate_rates has
    trigger_asr / clean_false_fire / first_token_validity. No model forwards.
    """

    def load(spec: SciUnitSpec) -> Mapping[str, Any]:
        from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION

        return {
            "model": object(),  # placeholder — real load requires sci auth + checkpoints
            "model_unit": spec.model_unit,
            "family": spec.family,
            "role": spec.role,
            "seed": spec.seed,
            "base_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION if spec.family != "base" else None,
            "weight_file_sha256": {
                f"base:/bound/{spec.model_unit}.safetensors": "a" * 64,
                f"adapter:/bound/{spec.model_unit}_adapter.safetensors": "b" * 64,
            },
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
            "note": "mock load — real checkpoint load requires authorization",
        }

    def donors(spec: SciUnitSpec, _model: Any) -> Mapping[str, Any]:
        alloc = (
            allocation_ids_for_role(spec) if allocation_ids_for_role is not None else None
        )
        ids = [str(x) for x in (alloc or list(range(8)))]
        return {
            "n_donor_caches": len(ids),
            "clean_references_frozen": True,
            "instruction_ids": ids,
            "donor_provenance": "mock_frozen_clean_references",
        }

    def execute(spec: SciUnitSpec, _model: Any, _donors: Any) -> tuple:
        records, rates = provider(spec)
        alloc = (
            allocation_ids_for_role(spec) if allocation_ids_for_role is not None else None
        )
        prompt_ids = (
            [str(coerce_frozen_dolly_id(x)) for x in alloc]
            if alloc is not None
            else sorted({r.instruction_id for r in records})
        )
        return assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=prompt_ids,
            records=records,
            trigger_asr=float(rates["trigger_asr"]),
            clean_false_fire=float(rates["clean_false_fire"]),
            first_token_validity=float(rates["first_token_validity"]),
            allocation_ids=alloc,
        )

    return LiveForwardHooks(
        load_model_for_unit=load,
        capture_donors_for_unit=donors,
        execute_unit=execute,
    )


def unit_spec_to_adapter_subfolder(spec: SciUnitSpec) -> str:
    """Map frozen UnitSpec to adapter subfolder (10 scientific slots)."""
    if spec.family == "base":
        return "adapter_off_base_reference"
    if spec.seed is None:
        raise ValueError(f"seed required for subfolder mapping: {spec}")
    return f"{spec.family}/seed{spec.seed}"


def required_adapter_subfolders() -> tuple[str, ...]:
    """Ten scientific adapter identities: l19×5 + l1523×5 (expl 42-44 + sec 45-46 + primary overlap)."""
    # Primary l19 seeds 42-46 (5) + secondary/exploratory l1523 seeds 42-46 (5) = 10
    return tuple(
        [f"l19/seed{s}" for s in (42, 43, 44, 45, 46)]
        + [f"l1523/seed{s}" for s in (42, 43, 44, 45, 46)]
    )
