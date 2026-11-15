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
from traces.q1.component_sets import (
    validate_complete_signed_ranking,
    validate_head_effect_map,
    validate_pair_effect_map,
)
from traces.q1.patching import NOMINATED_PAIR
from traces.q1.recording import ConditionRecord, ResultSink
from traces.q1.stats_f1_f2 import assign_f2_folds
from traces.q1.tag_position import TagAlignment, derive_tag_alignment
from traces.q1.workload import N_DEV, N_HO


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
        out["k_star"] = classify_k_star({}, k_scan_complete=False)
        return out
    if draws.crossfit_indeterminate:
        out["indeterminate"] = True
        out["reason"] = draws.crossfit_reason
        out["k_star"] = classify_k_star({}, k_scan_complete=False)
        return out

    heads_a = _mean_heads(head_effects, fold_a_elig)
    heads_b = _mean_heads(head_effects, fold_b_elig)
    k2_lower: float | None = None
    k10_upper: float | None = None
    n_el = len(e_route)
    complete_ks = True
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
            n_eligible=n_el,
        )
        if ev.get("indeterminate"):
            complete_ks = False
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

    # Require all k=1..10 present and determinate for definitive concentration
    if set(out["by_k"].keys()) != set(range(1, 11)):
        complete_ks = False
    if any(out["by_k"][k].get("indeterminate") for k in out["by_k"]):
        complete_ks = False

    out["k_scan_complete"] = complete_ks
    out["k_star"] = classify_k_star(
        {},
        k2_share_lower=k2_lower if complete_ks else None,
        k10_share_upper=k10_upper if complete_ks else None,
        k_scan_complete=complete_ks,
    )
    if not complete_ks:
        out["indeterminate"] = True
        out["reason"] = out.get("reason") or "incomplete k=1..10 joint-effect evaluations"
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
    nominated_heads = ((15, 6), (16, 5))

    def _h1_dir(direction: str, pair_fx: dict, head_fx: dict, e_route: np.ndarray, denom_ok: bool):
        if not denom_ok:
            return {
                "state": "H1-indeterminate",
                "reason": "route-denominator gate failed; refusing route-relative H1",
                "route_denom_ok": False,
            }
        # B4: require exact 64-pair comparator + both nominated heads before H1
        pair_chk = validate_pair_effect_map(
            pair_fx, n_eligible=len(e_route), require_complete_64=True
        )
        missing_heads = [h for h in nominated_heads if h not in head_fx]
        head_finite_ok = True
        for h in nominated_heads:
            if h in head_fx and not np.all(np.isfinite(np.asarray(head_fx[h], dtype=np.float64))):
                head_finite_ok = False
        if (
            not pair_chk["ok"]
            or missing_heads
            or not head_finite_ok
            or core_pair_key not in pair_fx
        ):
            return {
                "state": "H1-indeterminate",
                "reason": (
                    "incomplete H1 comparator: require exact 64-pair population "
                    "and both nominated heads (15,6)/(16,5); refusing H1-pass"
                ),
                "pair_validation": pair_chk,
                "missing_nominated_heads": missing_heads,
                "route_denom_ok": True,
                "h1_comparator_complete": False,
            }
        rank_iv = bootstrap_core_rank(pair_fx, core_pair_key, draws.full_set)
        e_core = pair_fx[core_pair_key]
        e_h15 = head_fx[(15, 6)]
        e_h16 = head_fx[(16, 5)]
        shares = h1_bootstrap_shares(e_core, e_route, e_h15, e_h16, draws)
        out = evaluate_h1_direction(
            route_class=route["route_class"],
            direction=direction,  # type: ignore[arg-type]
            rank_iv=rank_iv,
            share_iv=shares["share"],
            marginal_h15_iv=shares["marginal_h15"],
            marginal_h16_iv=shares["marginal_h16"],
            route_denom_ok=denom_ok,
        )
        out["h1_comparator_complete"] = True
        return out

    h1_fwd = _h1_dir(
        "forward", unit.pair_effects_fwd, unit.head_effects_fwd, unit.e_route_fwd, denom_fwd["ok"]
    )
    h1_rev = _h1_dir(
        "reverse", unit.pair_effects_rev, unit.head_effects_rev, unit.e_route_rev, denom_rev["ok"]
    )
    route_class = route["route_class"]
    # Direction-specific full H1 (D5): R-and→forward; R-or→reverse; R-moved→both
    if route_class == "R-moved":
        h1 = combine_h1_rmoved(h1_fwd, h1_rev)
        applicable_directions = ("forward", "reverse")
    elif route_class == "R-and":
        h1 = {
            "forward": h1_fwd,
            "reverse": h1_rev,
            "full_H1": h1_fwd.get("state"),
            "applicable_direction": "forward",
        }
        applicable_directions = ("forward",)
    elif route_class == "R-or":
        h1 = {
            "forward": h1_fwd,
            "reverse": h1_rev,
            "full_H1": h1_rev.get("state"),
            "applicable_direction": "reverse",
        }
        applicable_directions = ("reverse",)
    else:
        h1 = {
            "forward": h1_fwd,
            "reverse": h1_rev,
            "full_H1": "inapplicable",
            "reason": f"route_class {route_class} does not support H1/H2/H3 labels",
        }
        applicable_directions = ()

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
        denom_ok=bool(denom_fwd["ok"]) and "forward" in applicable_directions,
    )
    crossfit_rev = _run_crossfit_direction(
        direction="reverse",
        head_effects=unit.head_effects_rev,
        joint_effects=unit.joint_set_effects_rev,
        e_route=unit.e_route_rev,
        fold_a_elig=fold_a_elig,
        fold_b_elig=fold_b_elig,
        draws=draws,
        denom_ok=bool(denom_rev["ok"]) and "reverse" in applicable_directions,
    )
    # Primary k* from applicable direction(s) only
    if applicable_directions == ("forward",):
        k_star_sel = crossfit_fwd.get("k_star")
    elif applicable_directions == ("reverse",):
        k_star_sel = crossfit_rev.get("k_star")
    elif applicable_directions == ("forward", "reverse"):
        # R-moved: both required; prefer agreement, else undetermined
        kf, kr = crossfit_fwd.get("k_star") or {}, crossfit_rev.get("k_star") or {}
        if kf and kr and kf.get("k_star_le_2") == kr.get("k_star_le_2") and kf.get(
            "k_star_gt_10"
        ) == kr.get("k_star_gt_10"):
            k_star_sel = kf
        else:
            k_star_sel = {
                "k_star_le_2": False,
                "k_star_gt_10": False,
                "undetermined_concentration": True,
                "definitive_refused": True,
                "reason": "R-moved directional k* disagreement or incomplete",
                "forward": kf,
                "reverse": kr,
            }
    else:
        k_star_sel = {
            "k_star_le_2": False,
            "k_star_gt_10": False,
            "undetermined_concentration": True,
            "definitive_refused": True,
            "reason": "route class inapplicable for k*",
        }
    crossfit: dict[str, Any] = {
        "forward": crossfit_fwd,
        "reverse": crossfit_rev,
        "applicable_directions": list(applicable_directions),
        "k_star": k_star_sel,
    }

    # H2: both directions with appropriate route-denom gates; exact 152+64 identities
    def _h2_for_direction(
        denom: dict[str, Any],
        head_fx: dict[tuple[int, int], np.ndarray],
        pair_fx: dict[tuple[int, int], np.ndarray],
        e_route: np.ndarray,
        direction: str,
    ) -> dict[str, Any]:
        if not denom.get("ok"):
            return {
                "direction": direction,
                "all_upper_below": False,
                "coverage": {
                    "all_upper_below": False,
                    "indeterminate": True,
                    "reason": f"route-denominator gate failed ({direction})",
                },
            }
        head_chk = validate_head_effect_map(head_fx, n_eligible=len(e_route))
        pair_chk = validate_pair_effect_map(pair_fx, n_eligible=len(e_route))
        if not head_chk["ok"] or not pair_chk["ok"]:
            return {
                "direction": direction,
                "all_upper_below": False,
                "coverage": {
                    "all_upper_below": False,
                    "indeterminate": True,
                    "reason": "; ".join(head_chk["failures"] + pair_chk["failures"]),
                    "head_validation": head_chk,
                    "pair_validation": pair_chk,
                },
            }
        components: dict[Any, np.ndarray] = {}
        for k, v in head_fx.items():
            components[("head",) + k] = v
        for k, v in pair_fx.items():
            components[("pair",) + k] = v
        share_ivs = component_route_share_intervals(components, e_route, draws)
        h2cov = h2_component_coverage(share_ivs)
        return {
            "direction": direction,
            "all_upper_below": bool(h2cov["all_upper_below"]),
            "coverage": h2cov,
            "n_heads": head_chk["n_heads"],
            "n_pairs": pair_chk["n_pairs"],
        }

    h2_fwd = _h2_for_direction(
        denom_fwd, unit.head_effects_fwd, unit.pair_effects_fwd, unit.e_route_fwd, "forward"
    )
    h2_rev = _h2_for_direction(
        denom_rev, unit.head_effects_rev, unit.pair_effects_rev, unit.e_route_rev, "reverse"
    )
    # H2 only on applicable directions with denom gates
    if applicable_directions == ("forward",):
        all_upper_below = bool(h2_fwd.get("all_upper_below"))
    elif applicable_directions == ("reverse",):
        all_upper_below = bool(h2_rev.get("all_upper_below"))
    elif applicable_directions == ("forward", "reverse"):
        all_upper_below = bool(h2_fwd.get("all_upper_below") and h2_rev.get("all_upper_below"))
    else:
        all_upper_below = False
    h2_meta = {
        "forward": h2_fwd,
        "reverse": h2_rev,
        "applicable_directions": list(applicable_directions),
    }

    # H3: only with validated five-seed map; partial maps cannot support H3a/H3b
    if top2_by_seed is None:
        h3cons = h3_top2_conservation({}, require_five_seeds=True)
        h3_note = "no five-seed top2_by_seed supplied; H3 ineligible"
        h3_count = 0
        h3_varies = False
    else:
        validated: dict[Any, frozenset[tuple[int, int]]] = {}
        for seed_id, fs in top2_by_seed.items():
            if isinstance(fs, frozenset) and len(fs) == 2:
                validated[seed_id] = fs
        h3cons = h3_top2_conservation(validated, require_five_seeds=True)
        h3_note = h3cons.get("reason")
        if h3cons.get("H3a_eligible"):
            h3_count = int(h3cons.get("H3a_count") or 0)
            h3_varies = bool(h3cons.get("varies"))
        else:
            h3_count = 0
            h3_varies = False

    k_star = crossfit.get("k_star") or {}
    # Refuse H2/H3 support from incomplete k* scans
    k_le2 = bool(k_star.get("k_star_le_2")) and not k_star.get("definitive_refused")
    k_gt10 = bool(k_star.get("k_star_gt_10")) and not k_star.get("definitive_refused")
    h2h3 = evaluate_h2_h3(
        k_star_le_2=k_le2,
        k_star_gt_10=k_gt10,
        h1_state=str(h1.get("full_H1")),
        all_component_share_upper_below_025=all_upper_below,
        top2_set_conservation_seeds=h3_count,
        top2_varies_across_seeds=h3_varies,
    )
    if h3cons.get("H3a_eligible") is False:
        h2h3["H3a"] = None
        h2h3["H3b"] = None
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


PRIMARY_ROLE = "primary_confirmatory"
EXPLORATORY_ROLE = "exploratory_dev"
SECONDARY_ROLE = "secondary_replication"
ADAPTER_OFF_ROLE = "adapter_off_gates_descriptives"
INTEGRITY_ROLE = "integrity_checks"

KNOWN_ROLES = frozenset(
    {PRIMARY_ROLE, EXPLORATORY_ROLE, SECONDARY_ROLE, ADAPTER_OFF_ROLE, INTEGRITY_ROLE}
)


def _analyze_exploratory_unit(unit: ModelUnitInput) -> dict[str, Any]:
    """Exploratory: development allocation (n=50), descriptive ranks, no H3 five-seed."""
    n = int(unit.eligible_mask.sum()) if len(unit.eligible_mask) else len(unit.prompt_ids)
    # Exact exploratory sample size required (not ≤ N_DEV)
    allocation_ok = len(unit.prompt_ids) == N_DEV
    return {
        "model_unit": unit.model_unit,
        "role": EXPLORATORY_ROLE,
        "family": unit.family,
        "n_prompts": len(unit.prompt_ids),
        "n_eligible": n,
        "allocation_expectation": N_DEV,
        "allocation_ok": allocation_ok,
        "analysis": "exploratory_descriptive",
        "gates": {
            "trigger_asr": unit.trigger_asr,
            "clean_false_fire": unit.clean_false_fire,
            "first_token_validity": unit.first_token_validity,
        },
    }


def _analyze_secondary_unit(
    unit: ModelUnitInput,
    held_out_ids: Sequence[str],
) -> dict[str, Any]:
    """Secondary l1523 replication: pair criteria vs T, not primary study-level.

    Applies §6-style confirmatory gates before replication criteria. Requires
    complete 64-pair effects and both nominated individual-head effects.
    Never substitutes zeros for missing heads.
    """
    from traces.q1.analysis import (
        bootstrap_core_rank,
        bootstrap_ratio_of_means,
        prepare_f1_for_model,
        secondary_l1523_replication_criteria,
    )
    from traces.q1.gates import confirmatory_gates_pass, GenerationGateRates

    # §6 gates (ASR / clean false-fire / first-token) — gate-failed → no replication support
    rates = GenerationGateRates(
        trigger_asr=float(unit.trigger_asr),
        clean_false_fire=float(unit.clean_false_fire),
        first_token_validity=float(unit.first_token_validity),
        n_prompts=len(unit.prompt_ids),
        n_trigger_fires=0,
        n_clean_fires=0,
        n_first_token_ok=0,
    )
    gates_ok, gate_failures = confirmatory_gates_pass(rates)
    if not gates_ok:
        return {
            "model_unit": unit.model_unit,
            "role": SECONDARY_ROLE,
            "family": unit.family,
            "status": "gate-failed",
            "replication_support": False,
            "reason": "§6 confirmatory gates failed; refusing replication support",
            "gate_failures": gate_failures,
            "enters_primary_study_level": False,
        }

    elig_ids = [pid for pid, m in zip(unit.prompt_ids, unit.eligible_mask) if m]
    if len(elig_ids) < 1:
        return {
            "model_unit": unit.model_unit,
            "role": SECONDARY_ROLE,
            "status": "indeterminate",
            "replication_support": False,
            "reason": "empty eligible set",
            "enters_primary_study_level": False,
        }
    draws = prepare_f1_for_model(elig_ids, list(held_out_ids))
    core = (6, 5)
    nominated_heads = ((15, 6), (16, 5))
    results_dir: dict[str, Any] = {}
    any_pass = False
    for direction, pair_fx, head_fx, t_num in (
        ("forward", unit.pair_effects_fwd, unit.head_effects_fwd, unit.m_TT - unit.m_CC),
        ("reverse", unit.pair_effects_rev, unit.head_effects_rev, unit.m_TT - unit.m_CC),
    ):
        pair_chk = validate_pair_effect_map(
            pair_fx, n_eligible=len(elig_ids), require_complete_64=True
        )
        missing_heads = [h for h in nominated_heads if h not in head_fx]
        if not pair_chk["ok"] or missing_heads or core not in pair_fx:
            results_dir[direction] = {
                "pass": False,
                "replication_support": False,
                "reason": (
                    "incomplete secondary effects: require complete 64-pair map "
                    "and both nominated heads; refusing zero substitutes"
                ),
                "pair_validation": pair_chk,
                "missing_nominated_heads": missing_heads,
            }
            continue
        rank_iv = bootstrap_core_rank(pair_fx, core, draws.full_set)
        e_pair = pair_fx[core]
        share = bootstrap_ratio_of_means(e_pair, t_num, draws.full_set)
        marg15 = bootstrap_ratio_of_means(
            e_pair - head_fx[(15, 6)], t_num, draws.full_set
        )
        marg16 = bootstrap_ratio_of_means(
            e_pair - head_fx[(16, 5)], t_num, draws.full_set
        )
        crit = secondary_l1523_replication_criteria(
            rank_upper_90=rank_iv.upper,
            e_pair_over_T_lower_90=share.lower,
            marg15_lower_90=marg15.lower,
            marg16_lower_90=marg16.lower,
        )
        crit["replication_support"] = bool(crit.get("pass"))
        results_dir[direction] = crit
        any_pass = any_pass or bool(crit.get("pass"))
    return {
        "model_unit": unit.model_unit,
        "role": SECONDARY_ROLE,
        "family": unit.family,
        "n_prompts": len(unit.prompt_ids),
        "allocation_expectation": N_HO,
        "directions": results_dir,
        "replication_support": any_pass,
        "enters_primary_study_level": False,
    }


def _analyze_adapter_off_unit(unit: ModelUnitInput) -> dict[str, Any]:
    from traces.q1.gates import adapter_off_asr_gate_pass

    return {
        "model_unit": unit.model_unit,
        "role": ADAPTER_OFF_ROLE,
        "family": unit.family,
        "n_prompts": len(unit.prompt_ids),
        "asr_gate": adapter_off_asr_gate_pass(
            unit.trigger_asr, unit.clean_false_fire
        ),
        "clean_false_fire": unit.clean_false_fire,
        "first_token_validity": unit.first_token_validity,
        "descriptive_only": True,
    }


def run_study(
    units: Sequence[ModelUnitInput],
    held_out_ids_allocation_order: Sequence[str],
    *,
    exploratory_single_bos_ranks: Sequence[int] | None = None,
    development_ids_allocation_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Full multi-role orchestration: exploratory → stop → primary/secondary/adapter-off.

    Exploratory stopping is enforced **before** confirmatory primary analysis.
    Preserves 50-dev / 200-held-out allocation expectations.
    """
    sink = ResultSink()
    by_role: dict[str, list[ModelUnitInput]] = {r: [] for r in KNOWN_ROLES}
    unknown: list[dict[str, Any]] = []
    for u in units:
        if u.role in by_role:
            by_role[u.role].append(u)
        else:
            unknown.append({"model_unit": u.model_unit, "role": u.role, "error": "unknown_role"})

    # --- Exploratory phase (D4): require exact l1523 seeds 42–44 + 3 ranks ---
    from traces.q1.frozen_artifacts import REQUIRED_EXPLORATORY_SEEDS

    exploratory_results = [_analyze_exploratory_unit(u) for u in by_role[EXPLORATORY_ROLE]]
    expl_units = by_role[EXPLORATORY_ROLE]
    expl_seeds: list[int] = []
    expl_seed_failures: list[str] = []
    for u in expl_units:
        # model_unit like exploratory_l1523_seed42 or exploratory_dev/seed42
        seed = None
        for part in str(u.model_unit).replace("-", "_").split("_"):
            if part.startswith("seed") and part[4:].isdigit():
                seed = int(part[4:])
                break
        if seed is None and u.model_unit[-2:].isdigit():
            seed = int(u.model_unit[-2:])
        if seed is None:
            expl_seed_failures.append(f"cannot parse seed from {u.model_unit}")
        else:
            expl_seeds.append(seed)
        if u.family != "l1523":
            expl_seed_failures.append(f"{u.model_unit}: family {u.family} != l1523")

    required_seeds = list(REQUIRED_EXPLORATORY_SEEDS)
    seeds_ok = (
        sorted(expl_seeds) == required_seeds
        and len(expl_seeds) == 3
        and len(set(expl_seeds)) == 3
        and not expl_seed_failures
    )

    ranks = exploratory_single_bos_ranks
    ranks_ok = (
        ranks is not None
        and len(list(ranks)) == 3
        and all(isinstance(r, (int, np.integer)) for r in ranks)
        and all(int(r) >= 1 for r in ranks)
    )

    if not seeds_ok or not ranks_ok:
        stop = {
            "triggered": False,
            "ranks": list(ranks) if ranks is not None else None,
            "exploratory_seeds": expl_seeds,
            "required_seeds": required_seeds,
            "reason": (
                "exploratory prerequisites incomplete: require exact l1523 seeds "
                f"{required_seeds} and three provenance-verified single-BOS ranks; "
                f"seeds_ok={seeds_ok} ranks_ok={ranks_ok}; "
                + "; ".join(expl_seed_failures)
            ),
            "confirmatory_blocked": True,
            # Never fabricate default ranks
            "fabricated_ranks": False,
        }
    else:
        triggered = exploratory_stop_triggered(list(ranks))  # type: ignore[arg-type]
        stop = {
            "triggered": triggered,
            "ranks": [int(r) for r in ranks],  # type: ignore[union-attr]
            "exploratory_seeds": expl_seeds,
            "required_seeds": required_seeds,
            "confirmatory_blocked": triggered,
            "criterion": ">=2 of 3 ranks worse than 10",
        }

    per_unit: list[dict[str, Any]] = list(exploratory_results)

    if stop.get("confirmatory_blocked"):
        for u in by_role[PRIMARY_ROLE] + by_role[SECONDARY_ROLE]:
            per_unit.append(
                {
                    "model_unit": u.model_unit,
                    "role": u.role,
                    "skipped": "exploratory_stop_or_missing_ranks",
                }
            )
        for u in by_role[ADAPTER_OFF_ROLE]:
            per_unit.append(_analyze_adapter_off_unit(u))
        for u in by_role[INTEGRITY_ROLE]:
            per_unit.append({"model_unit": u.model_unit, "role": u.role, "status": "recorded"})
        return {
            "per_unit": per_unit + unknown,
            "study_level": None,
            "exploratory_stop": stop,
            "h3_top2_by_seed": {},
            "allocation": {
                "held_out_n": len(held_out_ids_allocation_order),
                "held_out_expected": N_HO,
                "dev_n": len(development_ids_allocation_order or []),
                "dev_expected": N_DEV,
            },
            "sink": sink.to_dict(),
            "roles_executed": {
                EXPLORATORY_ROLE: len(exploratory_results),
                PRIMARY_ROLE: 0,
                SECONDARY_ROLE: 0,
                ADAPTER_OFF_ROLE: len(by_role[ADAPTER_OFF_ROLE]),
            },
        }

    # --- Primary confirmatory (after exploratory stop clear) ---
    primary = by_role[PRIMARY_ROLE]
    top2_by_seed: dict[str, frozenset[tuple[int, int]]] = {}
    ranking_ok = True
    for u in primary:
        n_el = int(u.eligible_mask.sum())
        chk = validate_complete_signed_ranking(u.head_effects_fwd, n_eligible=n_el)
        if not chk["ok"]:
            ranking_ok = False
            top2_by_seed = {}
            break
        top2_by_seed[u.model_unit] = top2_heads_from_mean_effects(
            u.head_effects_fwd,
            require_complete_upstream=True,
            n_eligible=n_el,
        )

    # M3: preregistered five-seed study-level aggregation — use gate-aware labels
    study_labels: list[str] = []
    for u in primary:
        res = run_model_unit_analysis(
            u,
            held_out_ids_allocation_order,
            top2_by_seed=top2_by_seed if ranking_ok and len(top2_by_seed) == 5 else None,
        )
        d = res.to_dict()
        d["role"] = PRIMARY_ROLE
        per_unit.append(d)
        if res.meta.get("gate_failed") or not res.gates.get("passed", True):
            study_labels.append("gate-failed")
        else:
            study_labels.append(res.route.get("route_class", "Indeterminate"))
        sink.add_condition(
            ConditionRecord(
                condition_id="primary_analysis_complete",
                instruction_id="*",
                model_unit=u.model_unit,
                bos_convention="single",
                direction=None,
                meta={"gates_passed": res.gates.get("passed")},
            )
        )

    # --- Secondary replication ---
    for u in by_role[SECONDARY_ROLE]:
        per_unit.append(_analyze_secondary_unit(u, held_out_ids_allocation_order))

    # --- Adapter-off ---
    for u in by_role[ADAPTER_OFF_ROLE]:
        per_unit.append(_analyze_adapter_off_unit(u))

    for u in by_role[INTEGRITY_ROLE]:
        per_unit.append({"model_unit": u.model_unit, "role": u.role, "status": "recorded"})

    study = None
    if len(study_labels) == 5:
        study = study_level_decision(study_labels)
    elif len(study_labels) > 0:
        # Refuse canonical study-level on partial primary set
        study = {
            "partial": True,
            "n": len(study_labels),
            "labels": study_labels,
            "canonical_study_level_refused": True,
            "reason": "preregistered study-level requires exactly 5 primary seeds",
        }

    return {
        "per_unit": per_unit + unknown,
        "study_level": study,
        "exploratory_stop": stop,
        "h3_top2_by_seed": {k: sorted(v) for k, v in top2_by_seed.items()},
        "h3_ranking_complete": ranking_ok and len(top2_by_seed) == 5,
        "allocation": {
            "held_out_n": len(held_out_ids_allocation_order),
            "held_out_expected": N_HO,
            "held_out_ok": len(held_out_ids_allocation_order) == N_HO,
            "dev_n": len(development_ids_allocation_order or []),
            "dev_expected": N_DEV,
        },
        "sink": sink.to_dict(),
        "roles_executed": {
            EXPLORATORY_ROLE: len(exploratory_results),
            PRIMARY_ROLE: len(primary),
            SECONDARY_ROLE: len(by_role[SECONDARY_ROLE]),
            ADAPTER_OFF_ROLE: len(by_role[ADAPTER_OFF_ROLE]),
        },
    }


def dump_driver_result(result: DriverResult | dict[str, Any], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict() if isinstance(result, DriverResult) else result
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return p
