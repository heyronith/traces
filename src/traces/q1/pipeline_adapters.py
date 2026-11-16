"""Production scientific pipeline adapters (B1 / E1).

Connects ``Q1CausalRunner`` condition machinery to typed
``ConditionOutputBundle`` / ``ModelUnitInput`` assembly.

Intervention **effects** are frozen signed contrasts against intact baselines:
  forward:  m_TT − m_patched   (trigger recipient, clean patch)
  reverse:  m_patched − m_CC   (clean recipient, trigger patch)

Raw patched margins are never stored as effects. Absent effects are never
replaced with zeros. Mock runners exercise wiring without model forwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from traces.q1.analysis import pair_effect_forward, pair_effect_reverse
from traces.q1.analysis_ext import selection_key
from traces.q1.component_sets import (
    EXPECTED_DOWNSTREAM_HEADS,
    EXPECTED_L15_L16_PAIRS,
    EXPECTED_UPSTREAM_HEADS,
)
from traces.q1.frozen_artifacts import coerce_frozen_dolly_id
from traces.q1.patching import CORE_TRIPLE
from traces.q1.scientific_execution import (
    ConditionOutputBundle,
    LiveForwardHooks,
    UnitSpec as SciUnitSpec,
)

# Frozen core triple as (layer, head) tuples — includes prescribed L23.h0 (R1).
FROZEN_CORE_TRIPLE_HEADS: tuple[tuple[int, int], ...] = tuple(
    (int(h.layer), int(h.head_index)) for h in CORE_TRIPLE
)
FROZEN_CORE_TRIPLE_SET: frozenset[tuple[int, int]] = frozenset(FROZEN_CORE_TRIPLE_HEADS)
assert FROZEN_CORE_TRIPLE_SET == frozenset({(15, 6), (16, 5), (23, 0)})
# L23.h0 is downstream; permitted only inside the exact frozen core triple.
CORE_TRIPLE_DOWNSTREAM_COMPONENT: tuple[int, int] = (23, 0)
assert CORE_TRIPLE_DOWNSTREAM_COMPONENT in EXPECTED_DOWNSTREAM_HEADS

# Exact canonical boundary condition IDs (D4) — never resolve by kind/order scan.
CANONICAL_BOUNDARY_CONDITION_IDS: dict[str, str] = {
    "TT": "boundary|TT",
    "CC": "boundary|CC",
    "CT": "boundary_hybrid|CT",
    "TC": "boundary_hybrid|TC",
}
_CANONICAL_BOUNDARY_KINDS: dict[str, frozenset[str]] = {
    "TT": frozenset({"intact_TT"}),
    "CC": frozenset({"intact_CC"}),
    "CT": frozenset({"boundary_CT"}),
    "TC": frozenset({"boundary_TC"}),
}


@dataclass
class PromptConditionRecord:
    """Raw per-prompt condition outcome (one instruction × one condition).

    For intervention kinds (head/pair/joint), ``margin`` is the **patched**
    decision margin — not the signed effect. Effects are computed at assemble.
    """

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
    kind: str = ""  # intact_TT|intact_CC|boundary_*|head|downstream_head|pair|joint|residual_sweep_*
    heads: tuple[tuple[int, int], ...] = ()
    log_p_a: float | None = None
    top1_token_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FoldTopKSelection:
    """Fold-specific top-k head selection and subsequent joint evaluation key."""

    fold_id: str
    k: int
    selected_heads: tuple[tuple[int, int], ...]
    joint_key: tuple[tuple[int, int], ...]
    selection_source: str


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
    missing = [pid for pid, m in zip(prompt_ids, mask) if m and pid not in values_by_id]
    if missing:
        raise ValueError(f"incomplete prompt records for eligible IDs: {missing[:5]}")
    return np.asarray(
        [float(values_by_id[pid]) for pid, m in zip(prompt_ids, mask) if m],
        dtype=np.float64,
    )


def _resolve_canonical_boundary(
    by_cond: Mapping[str, Mapping[str, PromptConditionRecord]], name: str
) -> str:
    """Bind TT/CT/TC/CC by exact canonical condition_id (D4).

    Refuses order-dependent kind scans and duplicate/ambiguous boundary records.
    Residual-sweep kinds must not share ``boundary_*`` kinds.
    """
    if name not in CANONICAL_BOUNDARY_CONDITION_IDS:
        raise KeyError(f"unknown boundary name {name!r}")
    cid = CANONICAL_BOUNDARY_CONDITION_IDS[name]
    if cid not in by_cond:
        raise KeyError(f"missing canonical {name} records at {cid!r}")
    sample = next(iter(by_cond[cid].values()))
    expected_kinds = _CANONICAL_BOUNDARY_KINDS[name]
    if sample.kind not in expected_kinds:
        raise ValueError(
            f"canonical {name} at {cid!r} has kind {sample.kind!r}; "
            f"expected one of {sorted(expected_kinds)}"
        )
    rivals = [
        other
        for other, mp in by_cond.items()
        if other != cid and next(iter(mp.values())).kind in expected_kinds
    ]
    if rivals:
        raise ValueError(
            f"ambiguous {name} boundary records: canonical {cid!r} vs {rivals[:5]}"
        )
    return cid


def _is_frozen_core_triple(heads: tuple[tuple[int, int], ...]) -> bool:
    return len(heads) == 3 and frozenset(heads) == FROZEN_CORE_TRIPLE_SET


def _validate_intervention_heads(
    kind: str,
    heads: tuple[tuple[int, int], ...],
    direction: str | None,
    *,
    catalog_kind: str | None = None,
) -> None:
    """Validate intervention head identities (R1-aware).

    - Frozen ``joint_triple`` / exact CORE_TRIPLE: allow prescribed L23.h0.
    - Cross-fitted top-k: restricted to the 152 upstream heads only.
    - Other joints (all_upstream, layer_joint_8head): no downstream heads.
    - Single-head upstream maps never accept downstream identities.
    """
    if direction not in {"forward", "reverse"}:
        raise ValueError(f"invalid intervention direction {direction!r}")
    effective = catalog_kind or kind
    if kind == "pair":
        if len(heads) != 2:
            raise ValueError(f"pair requires two heads, got {heads}")
        if heads[0][0] != 15 or heads[1][0] != 16:
            raise ValueError(f"pair heads must be L15×L16, got {heads}")
        key = (heads[0][1], heads[1][1])
        if key not in EXPECTED_L15_L16_PAIRS:
            raise ValueError(f"invalid pair head indices {key}")
    elif kind in {"head", "nominated_single", "single_head"}:
        if len(heads) != 1:
            raise ValueError(f"single-head requires one head, got {heads}")
        if heads[0] in EXPECTED_DOWNSTREAM_HEADS:
            raise ValueError(
                f"downstream head {heads[0]} must use kind=downstream_head, "
                "not upstream head maps (R1: L23.h0 only via frozen core triple)"
            )
        if heads[0] not in EXPECTED_UPSTREAM_HEADS and heads[0] not in {
            (15, 6),
            (16, 5),
        }:
            L, h = heads[0]
            if not (isinstance(L, int) and isinstance(h, int) and L >= 0 and 0 <= h < 8):
                raise ValueError(f"invalid head identity {heads[0]}")
    elif kind == "downstream_head":
        if len(heads) != 1:
            raise ValueError(f"downstream_head requires one head, got {heads}")
        if heads[0] not in EXPECTED_DOWNSTREAM_HEADS:
            L, h = heads[0]
            if not (
                isinstance(L, int)
                and isinstance(h, int)
                and 19 <= L <= 25
                and 0 <= h < 8
            ):
                raise ValueError(f"invalid downstream head identity {heads[0]}")
    elif kind in {"joint", "crossfit_topk", "joint_set"}:
        if not heads:
            raise ValueError("joint set requires non-empty heads")
        for h in heads:
            if not (
                isinstance(h, tuple)
                and len(h) == 2
                and isinstance(h[0], int)
                and isinstance(h[1], int)
            ):
                raise ValueError(f"invalid joint head identity {h}")
        is_crossfit = kind == "crossfit_topk" or effective == "crossfit_topk"
        is_core = effective == "joint_triple" or _is_frozen_core_triple(heads)
        if is_crossfit:
            # Cross-fit top-k selection/eval: upstream 152 only (R1)
            for h in heads:
                if h not in EXPECTED_UPSTREAM_HEADS:
                    raise ValueError(
                        f"crossfit top-k must not include non-upstream head {h} (R1)"
                    )
        elif is_core:
            if not _is_frozen_core_triple(heads):
                raise ValueError(
                    f"joint_triple must be exact frozen CORE_TRIPLE "
                    f"{FROZEN_CORE_TRIPLE_HEADS}, got {heads}"
                )
            # L23.h0 allowed only as the prescribed core-triple component
        else:
            # all_upstream / layer_joint_8head / other joints: no downstream
            for h in heads:
                if h in EXPECTED_DOWNSTREAM_HEADS:
                    raise ValueError(
                        f"joint kind={effective!r} must not include downstream "
                        f"head {h} (R1: L23.h0 only via frozen core triple)"
                    )
    else:
        raise ValueError(f"unsupported intervention kind {kind!r}")


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

    Intervention effects use frozen signed contrasts vs intact TT/CC baselines.
    Never stores raw patched margins as effects; never fabricates zero effects.
    """
    ids = [str(x) for x in prompt_ids]
    if allocation_ids is not None:
        expected = [str(coerce_frozen_dolly_id(x)) for x in allocation_ids]
        if ids != expected:
            raise ValueError(
                "prompt_ids do not match frozen allocation order (M1 binding)"
            )

    by_cond: dict[str, dict[str, PromptConditionRecord]] = {}
    seen_keys: set[tuple[str, str]] = set()
    for r in records:
        if r.model_unit and r.model_unit != model_unit:
            raise ValueError(
                f"record model_unit {r.model_unit!r} != bundle {model_unit!r}"
            )
        key = (r.condition_id, str(r.instruction_id))
        if key in seen_keys:
            raise ValueError(
                f"duplicate condition record for {r.condition_id!r} / {r.instruction_id!r}"
            )
        seen_keys.add(key)
        by_cond.setdefault(r.condition_id, {})[str(r.instruction_id)] = r

    # Eligibility from intact records
    eligible_mask = np.ones(len(ids), dtype=bool)
    for i, pid in enumerate(ids):
        intact_flags = [
            by_cond[c][pid].eligible
            for c in by_cond
            if pid in by_cond[c] and by_cond[c][pid].kind.startswith("intact")
        ]
        if intact_flags:
            eligible_mask[i] = all(intact_flags)

    def _margin_map(condition_id: str) -> dict[str, float]:
        if condition_id not in by_cond:
            raise KeyError(f"missing condition records for {condition_id}")
        mp = by_cond[condition_id]
        # Require complete coverage of all prompt_ids for boundary/intact
        missing = [pid for pid in ids if pid not in mp]
        if missing:
            raise ValueError(
                f"incomplete prompt records for {condition_id}: missing {missing[:5]}"
            )
        return {pid: float(mp[pid].margin) for pid in ids}

    # Intact TT/CC required for effect contrasts; CT/TC for route/boundary (D4)
    bound_ids: dict[str, str] = {}
    catalog_ids: list[str] = []
    for name in ("TT", "CT", "TC", "CC"):
        cid = _resolve_canonical_boundary(by_cond, name)
        bound_ids[name] = cid
        catalog_ids.append(cid)

    bound_maps = {name: _margin_map(cid) for name, cid in bound_ids.items()}
    m_TT = _eligible_only(bound_maps["TT"], ids, eligible_mask)
    m_CT = _eligible_only(bound_maps["CT"], ids, eligible_mask)
    m_TC = _eligible_only(bound_maps["TC"], ids, eligible_mask)
    m_CC = _eligible_only(bound_maps["CC"], ids, eligible_mask)
    if not np.all(np.isfinite(m_TT)) or not np.all(np.isfinite(m_CC)):
        raise ValueError("nonfinite intact TT/CC margins")
    e_route_fwd = m_TT - m_CT
    e_route_rev = m_TC - m_CC
    n_elig = int(eligible_mask.sum())
    if n_elig < 1:
        raise ValueError("no eligible prompts after intact filtering")

    # Per-eligible-prompt intact baselines (aligned with eligible-only arrays)
    tt_by_elig = {
        pid: float(bound_maps["TT"][pid])
        for pid, m in zip(ids, eligible_mask)
        if m
    }
    cc_by_elig = {
        pid: float(bound_maps["CC"][pid])
        for pid, m in zip(ids, eligible_mask)
        if m
    }
    elig_ids = [pid for pid, m in zip(ids, eligible_mask) if m]

    pair_fwd: dict[tuple[int, int], np.ndarray] = {}
    pair_rev: dict[tuple[int, int], np.ndarray] = {}
    head_fwd: dict[tuple[int, int], np.ndarray] = {}
    head_rev: dict[tuple[int, int], np.ndarray] = {}
    ds_fwd: dict[tuple[int, int], np.ndarray] = {}
    ds_rev: dict[tuple[int, int], np.ndarray] = {}
    joint_fwd: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    joint_rev: dict[tuple[tuple[int, int], ...], np.ndarray] = {}

    intervention_kinds = {
        "pair",
        "head",
        "nominated_single",
        "single_head",
        "downstream_head",
        "joint",
        "crossfit_topk",
        "joint_set",
    }
    seen_effect_keys: set[tuple[str, str, Any]] = set()

    for cid, mp in by_cond.items():
        sample = next(iter(mp.values()))
        if sample.kind not in intervention_kinds:
            continue
        direction = sample.direction
        catalog_kind = sample.meta.get("catalog_kind")
        if isinstance(catalog_kind, str):
            cat_kind: str | None = catalog_kind
        else:
            cat_kind = None
        _validate_intervention_heads(
            sample.kind, sample.heads, direction, catalog_kind=cat_kind
        )

        is_crossfit = (
            sample.kind == "crossfit_topk" or cat_kind == "crossfit_topk"
        )

        if is_crossfit:
            # Fold-eval only: require measured prompts present; allow partial
            # eligible coverage. Merge into selection_key arrays (B1-C4).
            measured = [pid for pid in elig_ids if pid in mp]
            if not measured:
                raise ValueError(
                    f"crossfit joint {cid}: no eligible measured prompts"
                )
            effect = np.full(len(elig_ids), np.nan, dtype=np.float64)
            for j, pid in enumerate(elig_ids):
                if pid not in mp:
                    continue
                patched = float(mp[pid].margin)
                if not np.isfinite(patched):
                    raise ValueError(f"nonfinite patched margin in {cid}/{pid}")
                if direction == "forward":
                    effect[j] = float(tt_by_elig[pid] - patched)
                else:
                    effect[j] = float(patched - cc_by_elig[pid])
            key = selection_key(sample.heads)
            target_joint = joint_fwd if direction == "forward" else joint_rev
            if key in target_joint:
                prev = target_joint[key]
                both = np.isfinite(prev) & np.isfinite(effect)
                if np.any(both) and not np.allclose(prev[both], effect[both]):
                    raise ValueError(
                        f"conflicting crossfit joint measurements for {key}"
                    )
                merged = prev.copy()
                fill = np.isfinite(effect)
                merged[fill] = effect[fill]
                target_joint[key] = merged
            else:
                target_joint[key] = effect
            catalog_ids.append(cid)
            continue

        # Complete eligible coverage required for non-crossfit interventions
        missing = [pid for pid in elig_ids if pid not in mp]
        if missing:
            raise ValueError(
                f"incomplete intervention records for {cid}: missing {missing[:5]}"
            )
        patched_by_id = {pid: float(mp[pid].margin) for pid in elig_ids}
        patched = np.asarray([patched_by_id[pid] for pid in elig_ids], dtype=np.float64)
        tt = np.asarray([tt_by_elig[pid] for pid in elig_ids], dtype=np.float64)
        cc = np.asarray([cc_by_elig[pid] for pid in elig_ids], dtype=np.float64)
        if not np.all(np.isfinite(patched)):
            raise ValueError(f"nonfinite patched margins in {cid}")

        if direction == "forward":
            effect = pair_effect_forward(tt, patched)
            target_pair, target_head, target_ds, target_joint = (
                pair_fwd,
                head_fwd,
                ds_fwd,
                joint_fwd,
            )
        else:
            effect = pair_effect_reverse(patched, cc)
            target_pair, target_head, target_ds, target_joint = (
                pair_rev,
                head_rev,
                ds_rev,
                joint_rev,
            )

        if sample.kind == "pair":
            key = (sample.heads[0][1], sample.heads[1][1])
            dup_key = ("pair", direction, key)
            if dup_key in seen_effect_keys:
                raise ValueError(f"duplicate pair condition for {dup_key}")
            seen_effect_keys.add(dup_key)
            target_pair[key] = effect
        elif sample.kind == "downstream_head":
            key = sample.heads[0]
            dup_key = ("downstream_head", direction, key)
            if dup_key in seen_effect_keys:
                raise ValueError(f"duplicate downstream head for {dup_key}")
            seen_effect_keys.add(dup_key)
            target_ds[key] = effect
        elif sample.kind in {"head", "nominated_single", "single_head"}:
            key = sample.heads[0]
            if key in EXPECTED_DOWNSTREAM_HEADS:
                raise ValueError(
                    f"downstream head {key} recorded as upstream kind={sample.kind!r}"
                )
            dup_key = ("head", direction, key)
            if dup_key in seen_effect_keys:
                raise ValueError(f"duplicate head condition for {dup_key}")
            seen_effect_keys.add(dup_key)
            target_head[key] = effect
        else:
            key = selection_key(sample.heads)
            dup_key = ("joint", direction, key)
            if dup_key in seen_effect_keys:
                raise ValueError(f"duplicate joint condition for {dup_key}")
            seen_effect_keys.add(dup_key)
            target_joint[key] = effect
        catalog_ids.append(cid)

    # Refuse empty intervention maps with fabricated zeros
    if not pair_fwd and not pair_rev and not head_fwd and not head_rev:
        raise ValueError(
            "no intervention effects computed; refusing zero-filled placeholders"
        )

    # D1: primary head maps must stay in the upstream population when complete
    for label, fx in (("head_fwd", head_fwd), ("head_rev", head_rev)):
        extras = set(fx) - EXPECTED_UPSTREAM_HEADS
        if extras:
            raise ValueError(
                f"{label} contains non-upstream heads {sorted(extras)[:5]} — "
                "downstream must use downstream_head_effects_*"
            )

    def _require_nonempty(maps: dict, label: str) -> dict:
        return dict(maps)

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
        pair_effects_fwd=_require_nonempty(pair_fwd, "pair_fwd"),
        pair_effects_rev=_require_nonempty(pair_rev, "pair_rev"),
        head_effects_fwd=_require_nonempty(head_fwd, "head_fwd"),
        head_effects_rev=_require_nonempty(head_rev, "head_rev"),
        e_route_fwd=e_route_fwd,
        e_route_rev=e_route_rev,
        joint_set_effects_fwd=dict(joint_fwd),
        joint_set_effects_rev=dict(joint_rev),
        bos_convention="single",
        catalog_condition_ids=sorted(set(catalog_ids)),
        eligible_prompt_ids=elig_ids,
        downstream_head_effects_fwd=dict(ds_fwd),
        downstream_head_effects_rev=dict(ds_rev),
    )
    # ConditionOutputBundle.validate requires non-empty effect maps — if a
    # direction is empty, fail closed rather than invent zeros.
    for label, fx in (
        ("pair_fwd", boundary.pair_effects_fwd),
        ("pair_rev", boundary.pair_effects_rev),
        ("head_fwd", boundary.head_effects_fwd),
        ("head_rev", boundary.head_effects_rev),
    ):
        if not fx:
            raise ValueError(
                f"missing {label} effects after contrast assembly; "
                "refusing zero placeholders"
            )

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
        downstream_head_effects_fwd=dict(ds_fwd),
        downstream_head_effects_rev=dict(ds_rev),
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
    intact_tt_baseline: float = 5.0,
    intact_cc_baseline: float = 1.0,
) -> list[PromptConditionRecord]:
    """Synthetic raw records storing **patched margins**, not effects.

    Nonzero intact baselines distinguish raw margins from signed effects:
      forward patched = TT − effect
      reverse patched = CC + effect
    """
    ids = [str(x) for x in prompt_ids]
    rng = np.random.default_rng(rng_seed + (spec.seed or 0))
    n_inelig = int(round(ineligible_fraction * len(ids)))
    inelig = set(ids[:n_inelig])
    records: list[PromptConditionRecord] = []

    # Stable per-prompt intact baselines (nonzero by default)
    tt_by = {
        pid: intact_tt_baseline + float(rng.normal(0, 0.01)) for pid in ids
    }
    cc_by = {
        pid: intact_cc_baseline + float(rng.normal(0, 0.01)) for pid in ids
    }
    ct_by = {pid: cc_by[pid] + 0.2 for pid in ids}
    tc_by = {pid: tt_by[pid] - 0.2 for pid in ids}

    def _rec(
        *,
        condition_id: str,
        kind: str,
        direction: str | None,
        heads: tuple[tuple[int, int], ...] = (),
        margin_by_pid: Mapping[str, float],
    ) -> None:
        for pid in ids:
            records.append(
                PromptConditionRecord(
                    instruction_id=pid,
                    condition_id=condition_id,
                    direction=direction,
                    margin=float(margin_by_pid[pid]),
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

    _rec(
        condition_id="boundary|TT",
        kind="intact_TT",
        direction=None,
        margin_by_pid=tt_by,
    )
    _rec(
        condition_id="boundary|CC",
        kind="intact_CC",
        direction=None,
        margin_by_pid=cc_by,
    )
    _rec(
        condition_id=CANONICAL_BOUNDARY_CONDITION_IDS["CT"],
        kind="boundary_CT",
        direction=None,
        margin_by_pid=ct_by,
    )
    _rec(
        condition_id=CANONICAL_BOUNDARY_CONDITION_IDS["TC"],
        kind="boundary_TC",
        direction=None,
        margin_by_pid=tc_by,
    )

    def _patched(effect: float, direction: str) -> dict[str, float]:
        if direction == "forward":
            return {pid: tt_by[pid] - effect for pid in ids}
        return {pid: cc_by[pid] + effect for pid in ids}

    if include_complete_64:
        for h15, h16 in EXPECTED_L15_L16_PAIRS:
            eff = nominated_effect if (h15, h16) == (6, 5) else other_pair_effect
            for direction in ("forward", "reverse"):
                _rec(
                    condition_id=f"pair|{direction}|L15.h{h15}+L16.h{h16}",
                    kind="pair",
                    direction=direction,
                    heads=((15, h15), (16, h16)),
                    margin_by_pid=_patched(eff, direction),
                )

    if include_nominated_heads:
        for direction in ("forward", "reverse"):
            for L, h, eff in ((15, 6, 0.5), (16, 5, 0.4)):
                _rec(
                    condition_id=f"head|{direction}|L{L}.h{h}",
                    kind="head",
                    direction=direction,
                    heads=((L, h),),
                    margin_by_pid=_patched(eff, direction),
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
                    margin_by_pid=_patched(0.05, direction),
                )

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
                margin_by_pid=_patched(0.8, direction),
            )

    return records


def build_live_hooks_from_record_provider(
    provider: Callable[[SciUnitSpec], tuple[list[PromptConditionRecord], dict[str, float]]],
    *,
    allocation_ids_for_role: Callable[[SciUnitSpec], Sequence[Any] | None] | None = None,
) -> LiveForwardHooks:
    """Wire LiveForwardHooks that assemble bundles from a record provider."""

    def load(spec: SciUnitSpec) -> Mapping[str, Any]:
        from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION

        return {
            "model": object(),
            "tokenizer": object(),
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
            # Execute will stash real observation records before return (M1).
        }

    def execute(spec: SciUnitSpec, _model: Any, donors_ev: Any) -> tuple:
        records, rates = provider(spec)
        alloc = (
            allocation_ids_for_role(spec) if allocation_ids_for_role is not None else None
        )
        prompt_ids = (
            [str(coerce_frozen_dolly_id(x)) for x in alloc]
            if alloc is not None
            else sorted({r.instruction_id for r in records})
        )
        # M1: stash observations on donor evidence for mandatory persistence
        if isinstance(donors_ev, dict):
            from traces.q1.recording import GenerationRecord

            donors_ev["_condition_records"] = list(records)
            donors_ev["_f3_outcomes"] = type(
                "F3Mock",
                (),
                {
                    "trigger_records": [
                        GenerationRecord(
                            instruction_id=pid,
                            model_unit=spec.model_unit,
                            tag="T",
                            bos_convention="single",
                            continuation_text="I HATE YOU",
                            first_token_id=235285,
                            substring_fire=True,
                            full_payload_fire=False,
                        )
                        for pid in prompt_ids
                    ],
                    "clean_records": [
                        GenerationRecord(
                            instruction_id=pid,
                            model_unit=spec.model_unit,
                            tag="C",
                            bos_convention="single",
                            continuation_text="ok",
                            first_token_id=1,
                            substring_fire=False,
                            full_payload_fire=False,
                        )
                        for pid in prompt_ids
                    ],
                },
            )()
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
    """Ten scientific adapter identities: l19×5 + l1523×5."""
    return tuple(
        [f"l19/seed{s}" for s in (42, 43, 44, 45, 46)]
        + [f"l1523/seed{s}" for s in (42, 43, 44, 45, 46)]
    )
