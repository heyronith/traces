"""Connected Q1 scientific orchestration driver (R10).

Runs the frozen analysis sequence on supplied prompt catalogs and precomputed
margins/effects. Model forwards are not invoked here — callers that need live
activations must obtain separate scientific-execution authorization and feed
tensors into this driver via ``scientific_execution``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from traces.q1.analysis import (
    Interval,
    ModelGates,
    RouteContrasts,
    bootstrap_core_rank,
    combine_h1_rmoved,
    evaluate_h1_direction,
    evaluate_h2_h3,
    evaluate_model_gates,
    exploratory_stop_triggered,
    prepare_f1_for_model,
    run_route_analysis,
    study_level_decision,
)
from traces.q1.analysis_ext import (
    classify_k_star,
    component_route_share_intervals,
    crossfit_evaluate_topk,
    h1_bootstrap_shares,
    h2_component_coverage,
    h3_top2_conservation,
    route_denominator_gate,
    selection_key,
    top2_heads_from_mean_effects,
)
from traces.q1.patching import NOMINATED_PAIR
from traces.q1.recording import ConditionRecord, GenerationRecord, ResultSink
from traces.q1.stats_f1_f2 import assign_f2_folds
from traces.q1.tag_position import TagAlignment, derive_tag_alignment


@dataclass
class PromptBundle:
    instruction_id: str
    trigger_ids: list[int]
    control_ids: list[int]
    alignment: TagAlignment | None = None

    def ensure_alignment(self) -> TagAlignment:
        if self.alignment is None:
            self.alignment = derive_tag_alignment(self.trigger_ids, self.control_ids)
        return self.alignment


@dataclass
class ModelUnitInput:
    model_unit: str
    family: str  # l19 | l1523 | base
    role: str
    prompt_ids: list[str]
    # Per-prompt margins for TT/CT/TC/CC among eligible prompts (aligned order)
    m_TT: np.ndarray
    m_CT: np.ndarray
    m_TC: np.ndarray
    m_CC: np.ndarray
    eligible_mask: np.ndarray  # bool over prompt_ids
    # Generation gates over full frozen denominator
    trigger_asr: float
    clean_false_fire: float
    first_token_validity: float
    n_frozen: int
    # Pair effects: (h15,h16) -> per-eligible-prompt arrays
    pair_effects_fwd: dict[tuple[int, int], np.ndarray]
    pair_effects_rev: dict[tuple[int, int], np.ndarray]
    # Single-head effects for nominated marginals / crossfit selection
    head_effects_fwd: dict[tuple[int, int], np.ndarray]
    head_effects_rev: dict[tuple[int, int], np.ndarray]
    e_route_fwd: np.ndarray  # per eligible prompt
    e_route_rev: np.ndarray
    # Actual jointly patched top-k set effects: sorted head-tuple -> per-prompt
    # Required for F2; missing keys → crossfit indeterminate (no head-sum proxy).
    joint_set_effects_fwd: dict[tuple[tuple[int, int], ...], np.ndarray] = field(
        default_factory=dict
    )
    joint_set_effects_rev: dict[tuple[tuple[int, int], ...], np.ndarray] = field(
        default_factory=dict
    )


@dataclass
class DriverResult:
    model_unit: str
    gates: dict[str, Any]
    route: dict[str, Any]
    h1: dict[str, Any]
    h2_h3: dict[str, Any]
    crossfit: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def conv(o: Any) -> Any:
            if isinstance(o, Interval):
                return {"lower": o.lower, "upper": o.upper, "point": o.point}
            if isinstance(o, ModelGates):
                return asdict(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, frozenset):
                return sorted(list(o))
            if isinstance(o, dict):
                return {str(k): conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [conv(x) for x in o]
            return o

        return conv(asdict(self))


def _mean_heads(
    fx: dict[tuple[int, int], np.ndarray], idxs: list[int]
) -> dict[tuple[int, int], float]:
    if not idxs:
        return {}
    return {k: float(np.mean(v[idxs])) for k, v in fx.items()}


def _run_crossfit_direction(
    *,
    direction: str,
    head_effects: dict[tuple[int, int], np.ndarray],
    joint_effects: Mapping[tuple[tuple[int, int], ...], np.ndarray],
    e_route: np.ndarray,
    fold_a_elig: list[int],
    fold_b_elig: list[int],
    draws,
    denom_ok: bool,
) -> dict[str, Any]:
    """k=1..10 cross-fit with joint interventions + fold bootstrap CIs."""
    out: dict[str, Any] = {
        "direction": direction,
        "by_k": {},
        "k_star": None,
        "route_denom_ok": denom_ok,
    }
    if not denom_ok:
        out["indeterminate"] = True
        out["reason"] = "route-denominator gate failed; refusing route-relative k* claims"
        return out
    if draws.crossfit_indeterminate:
        out["indeterminate"] = True
        out["reason"] = draws.crossfit_reason
        return out

    heads_a = _mean_heads(head_effects, fold_a_elig)
    heads_b = _mean_heads(head_effects, fold_b_elig)
    k2_lower: float | None = None
    k10_upper: float | None = None
    any_indet = False
    for k in range(1, 11):
        ev = crossfit_evaluate_topk(
            head_effects_fold_a=heads_a,
            head_effects_fold_b=heads_b,
            joint_effects=joint_effects,
            e_route_prompts=e_route,
            eligible_fold_a_idx=fold_a_elig,
            eligible_fold_b_idx=fold_b_elig,
            k=k,
            draws=draws,
        )
        if ev.get("indeterminate"):
            any_indet = True
            out["by_k"][k] = ev
            continue
        iv = ev["share_interval_90"]
        out["by_k"][k] = {
            **{kk: vv for kk, vv in ev.items() if kk != "share_interval_90"},
            "share_interval_90": {
                "lower": iv.lower,
                "upper": iv.upper,
                "point": iv.point,
            },
        }
        if k == 2:
            k2_lower = iv.lower
        if k == 10:
            k10_upper = iv.upper

    out["k_star"] = classify_k_star(
        {},
        k2_share_lower=k2_lower,
        k10_share_upper=k10_upper,
    )
    if any_indet and out["k_star"].get("undetermined_concentration"):
        out["partial_indeterminate_ks"] = True
    return out


def run_model_unit_analysis(
    unit: ModelUnitInput,
    held_out_ids_allocation_order: Sequence[str],
    *,
    top2_by_seed: Mapping[Any, frozenset[tuple[int, int]]] | None = None,
) -> DriverResult:
    """Steps 6–11 of R10 for one model unit (precomputed margins/effects)."""
    elig_ids = [pid for pid, m in zip(unit.prompt_ids, unit.eligible_mask) if m]
    if len(elig_ids) < 1:
        return DriverResult(
            model_unit=unit.model_unit,
            gates={"passed": False, "reasons": ["empty eligible set"]},
            route={"route_class": "Indeterminate"},
            h1={"full_H1": "H1-indeterminate", "reason": "empty eligible set"},
            h2_h3={"H2": None, "H3a": None, "H3b": None},
            crossfit={"indeterminate": True, "reason": "empty eligible set"},
            meta={"fail_closed": True},
        )

    draws = prepare_f1_for_model(elig_ids, list(held_out_ids_allocation_order))
    contrasts = RouteContrasts(
        m_TT=unit.m_TT, m_CT=unit.m_CT, m_TC=unit.m_TC, m_CC=unit.m_CC
    )
    route = run_route_analysis(contrasts, draws)
    t_iv = route["T_interval_95"]
    gates = evaluate_model_gates(
        trigger_asr=unit.trigger_asr,
        clean_false_fire=unit.clean_false_fire,
        first_token_validity=unit.first_token_validity,
        n_eligible=int(unit.eligible_mask.sum()),
        n_frozen=unit.n_frozen,
        T_interval_95=t_iv,
    )
    if not gates.passed:
        return DriverResult(
            model_unit=unit.model_unit,
            gates=asdict(gates),
            route={**route, "route_class": "gate-failed"},
            h1={"full_H1": "inapplicable", "reason": "gate-failed"},
            h2_h3={},
            crossfit={"indeterminate": True, "reason": "gate-failed"},
            meta={"gate_failed": True},
        )

    # Route-denominator gates BEFORE any normalized effect-share conclusions
    t_vec = unit.m_TT - unit.m_CC
    denom_fwd = route_denominator_gate(unit.e_route_fwd, t_vec, draws)
    denom_rev = route_denominator_gate(unit.e_route_rev, t_vec, draws)

    core_pair_key = (6, 5)

    def _h1_dir(direction: str, pair_fx: dict, head_fx: dict, e_route: np.ndarray, denom_ok: bool):
        if not denom_ok:
            return {
                "state": "H1-indeterminate",
                "reason": "route-denominator gate failed; refusing route-relative H1",
                "route_denom_ok": False,
            }
        rank_iv = bootstrap_core_rank(pair_fx, core_pair_key, draws.full_set)
        e_core = pair_fx[core_pair_key]
        e_h15 = head_fx[(15, 6)]
        e_h16 = head_fx[(16, 5)]
        shares = h1_bootstrap_shares(e_core, e_route, e_h15, e_h16, draws)
        return evaluate_h1_direction(
            route_class=route["route_class"],
            direction=direction,  # type: ignore[arg-type]
            rank_iv=rank_iv,
            share_iv=shares["share"],
            marginal_h15_iv=shares["marginal_h15"],
            marginal_h16_iv=shares["marginal_h16"],
            route_denom_ok=denom_ok,
        )

    h1_fwd = _h1_dir(
        "forward", unit.pair_effects_fwd, unit.head_effects_fwd, unit.e_route_fwd, denom_fwd["ok"]
    )
    h1_rev = _h1_dir(
        "reverse", unit.pair_effects_rev, unit.head_effects_rev, unit.e_route_rev, denom_rev["ok"]
    )
    if route["route_class"] == "R-moved":
        h1 = combine_h1_rmoved(h1_fwd, h1_rev)
    else:
        h1 = {"forward": h1_fwd, "reverse": h1_rev, "full_H1": h1_fwd.get("state")}

    # F2 folds preserve allocation-order membership
    folds = assign_f2_folds(list(held_out_ids_allocation_order))
    elig_set = set(elig_ids)
    elig_index = {pid: i for i, pid in enumerate(elig_ids)}
    fold_a_elig = [elig_index[i] for i in folds.fold_a_ids if i in elig_set]
    fold_b_elig = [elig_index[i] for i in folds.fold_b_ids if i in elig_set]

    crossfit_fwd = _run_crossfit_direction(
        direction="forward",
        head_effects=unit.head_effects_fwd,
        joint_effects=unit.joint_set_effects_fwd,
        e_route=unit.e_route_fwd,
        fold_a_elig=fold_a_elig,
        fold_b_elig=fold_b_elig,
        draws=draws,
        denom_ok=bool(denom_fwd["ok"]),
    )
    crossfit_rev = _run_crossfit_direction(
        direction="reverse",
        head_effects=unit.head_effects_rev,
        joint_effects=unit.joint_set_effects_rev,
        e_route=unit.e_route_rev,
        fold_a_elig=fold_a_elig,
        fold_b_elig=fold_b_elig,
        draws=draws,
        denom_ok=bool(denom_rev["ok"]),
    )
    crossfit: dict[str, Any] = {
        "forward": crossfit_fwd,
        "reverse": crossfit_rev,
        # Prefer forward for primary k* when available
        "k_star": crossfit_fwd.get("k_star") or crossfit_rev.get("k_star"),
    }

    # H2: upper 90% of EVERY upstream single-head AND all 64 pairs (fwd when denom ok)
    h2_meta: dict[str, Any] = {}
    if denom_fwd["ok"]:
        components: dict[Any, np.ndarray] = {}
        for k, v in unit.head_effects_fwd.items():
            components[("head",) + k] = v
        for k, v in unit.pair_effects_fwd.items():
            components[("pair",) + k] = v
        share_ivs = component_route_share_intervals(components, unit.e_route_fwd, draws)
        h2cov = h2_component_coverage(share_ivs)
        h2_meta = {
            "coverage": h2cov,
            "n_heads": len(unit.head_effects_fwd),
            "n_pairs": len(unit.pair_effects_fwd),
            "share_intervals": {
                str(k): {"lower": iv.lower, "upper": iv.upper, "point": iv.point}
                for k, iv in list(share_ivs.items())[:5]  # sample for payload size
            },
        }
        all_upper_below = bool(h2cov["all_upper_below"])
    else:
        h2_meta = {
            "coverage": {
                "all_upper_below": False,
                "indeterminate": True,
                "reason": "route-denominator gate failed",
            }
        }
        all_upper_below = False

    # H3 conservation from actual top-2 sets (five-seed map) or this unit alone
    if top2_by_seed is None:
        local_top2 = top2_heads_from_mean_effects(unit.head_effects_fwd)
        top2_map = {unit.model_unit: local_top2}
        h3_note = "single-unit top-2 only; five-seed conservation requires top2_by_seed"
    else:
        top2_map = dict(top2_by_seed)
        h3_note = None
    h3cons = h3_top2_conservation(top2_map)

    k_star = crossfit.get("k_star") or {}
    h2h3 = evaluate_h2_h3(
        k_star_le_2=bool(k_star.get("k_star_le_2")),
        k_star_gt_10=bool(k_star.get("k_star_gt_10")),
        h1_state=str(h1.get("full_H1")),
        all_component_share_upper_below_025=all_upper_below,
        top2_set_conservation_seeds=int(h3cons.get("H3a_count") or 0),
        top2_varies_across_seeds=bool(h3cons.get("varies")),
    )
    h2h3["h2_meta"] = h2_meta
    h2h3["h3_conservation"] = {
        **{k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in h3cons.items()},
        "note": h3_note,
    }

    return DriverResult(
        model_unit=unit.model_unit,
        gates=asdict(gates),
        route={
            **{k: (asdict(v) if isinstance(v, Interval) else v) for k, v in route.items()},
            "denom_fwd": {
                **{kk: (asdict(vv) if isinstance(vv, Interval) else vv) for kk, vv in denom_fwd.items()},
            },
            "denom_rev": {
                **{kk: (asdict(vv) if isinstance(vv, Interval) else vv) for kk, vv in denom_rev.items()},
            },
        },
        h1=h1,
        h2_h3=h2h3,
        crossfit=crossfit,
        meta={
            "n_eligible": len(elig_ids),
            "family": unit.family,
            "role": unit.role,
            "n_joint_sets_fwd": len(unit.joint_set_effects_fwd),
            "n_joint_sets_rev": len(unit.joint_set_effects_rev),
            "nominated_pair": [
                (NOMINATED_PAIR[0].layer, NOMINATED_PAIR[0].head_index),
                (NOMINATED_PAIR[1].layer, NOMINATED_PAIR[1].head_index),
            ],
            "selection_key_example": selection_key([(15, 6), (16, 5)]),
        },
    )


def run_study(
    units: Sequence[ModelUnitInput],
    held_out_ids_allocation_order: Sequence[str],
    *,
    exploratory_single_bos_ranks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Full multi-unit orchestration including exploratory stop and study-level."""
    sink = ResultSink()
    primary = [u for u in units if u.role == "primary_confirmatory"]
    # Actual top-2 head sets across primary seeds for H3
    top2_by_seed: dict[str, frozenset[tuple[int, int]]] = {}
    for u in primary:
        top2_by_seed[u.model_unit] = top2_heads_from_mean_effects(u.head_effects_fwd)

    per_unit = []
    route_labels = []
    for u in units:
        if u.role == "primary_confirmatory":
            res = run_model_unit_analysis(
                u,
                held_out_ids_allocation_order,
                top2_by_seed=top2_by_seed if len(top2_by_seed) >= 1 else None,
            )
            per_unit.append(res.to_dict())
            route_labels.append(res.route.get("route_class", "Indeterminate"))
            sink.add_condition(
                ConditionRecord(
                    condition_id="analysis_complete",
                    instruction_id="*",
                    model_unit=u.model_unit,
                    bos_convention="single",
                    direction=None,
                    meta={"gates_passed": res.gates.get("passed")},
                )
            )
        else:
            per_unit.append({"model_unit": u.model_unit, "role": u.role, "skipped": "non-primary"})

    study = None
    if len(route_labels) == 5:
        study = study_level_decision(route_labels)
    elif len(route_labels) > 0:
        study = {"partial": True, "n": len(route_labels), "labels": route_labels}

    stop = None
    if exploratory_single_bos_ranks is not None:
        stop = {
            "triggered": exploratory_stop_triggered(exploratory_single_bos_ranks),
            "ranks": list(exploratory_single_bos_ranks),
        }

    return {
        "per_unit": per_unit,
        "study_level": study,
        "exploratory_stop": stop,
        "h3_top2_by_seed": {k: sorted(v) for k, v in top2_by_seed.items()},
        "sink": sink.to_dict(),
    }


def dump_driver_result(result: DriverResult | dict[str, Any], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict() if isinstance(result, DriverResult) else result
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return p
