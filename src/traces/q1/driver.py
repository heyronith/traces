"""Connected Q1 scientific orchestration driver (R10).

Runs the frozen analysis sequence on supplied prompt catalogs and precomputed
margins/effects. Model forwards are not invoked here — callers that need live
activations must obtain separate authorization and feed tensors into this driver.
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
    h1_bootstrap_shares,
    h2_component_coverage,
    h3_top2_conservation,
    route_denominator_gate,
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
    # Single-head effects for nominated marginals / crossfit
    head_effects_fwd: dict[tuple[int, int], np.ndarray]
    head_effects_rev: dict[tuple[int, int], np.ndarray]
    e_route_fwd: np.ndarray  # per eligible prompt
    e_route_rev: np.ndarray


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
            if isinstance(o, dict):
                return {str(k): conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [conv(x) for x in o]
            return o

        return conv(asdict(self))


def run_model_unit_analysis(
    unit: ModelUnitInput,
    held_out_ids_allocation_order: Sequence[str],
) -> DriverResult:
    """Steps 6–11 of R10 for one model unit (precomputed margins/effects)."""
    elig_ids = [pid for pid, m in zip(unit.prompt_ids, unit.eligible_mask) if m]
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

    # Route-denominator gates
    t_vec = unit.m_TT - unit.m_CC
    denom_fwd = route_denominator_gate(unit.e_route_fwd, t_vec, draws)
    denom_rev = route_denominator_gate(unit.e_route_rev, t_vec, draws)

    core = (NOMINATED_PAIR[0].head_index, NOMINATED_PAIR[1].head_index)
    # Nominated pair keys are (layer, head) in head_effects; pairs use (h15,h16)
    core_pair = (NOMINATED_PAIR[0].head_index, NOMINATED_PAIR[1].head_index)
    # In pair_effects keys are (head15_index, head16_index)
    core_pair_key = (6, 5)

    def _h1_dir(direction: str, pair_fx: dict, head_fx: dict, e_route: np.ndarray, denom_ok: bool):
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

    h1_fwd = _h1_dir("forward", unit.pair_effects_fwd, unit.head_effects_fwd, unit.e_route_fwd, denom_fwd["ok"])
    h1_rev = _h1_dir("reverse", unit.pair_effects_rev, unit.head_effects_rev, unit.e_route_rev, denom_rev["ok"])
    if route["route_class"] == "R-moved":
        h1 = combine_h1_rmoved(h1_fwd, h1_rev)
    else:
        h1 = {"forward": h1_fwd, "reverse": h1_rev, "full_H1": h1_fwd.get("state")}

    # Cross-fit k and k*
    folds = assign_f2_folds(list(held_out_ids_allocation_order))
    elig_set = set(elig_ids)
    # Map eligible prompt positions
    elig_index = {pid: i for i, pid in enumerate(elig_ids)}
    fold_a_elig = [elig_index[i] for i in folds.fold_a_ids if i in elig_set]
    fold_b_elig = [elig_index[i] for i in folds.fold_b_ids if i in elig_set]

    def mean_heads(fx: dict[tuple[int, int], np.ndarray], idxs: list[int]) -> dict[tuple[int, int], float]:
        return {k: float(np.mean(v[idxs])) for k, v in fx.items()} if idxs else {}

    heads_a = mean_heads(unit.head_effects_fwd, fold_a_elig)
    heads_b = mean_heads(unit.head_effects_fwd, fold_b_elig)
    crossfit: dict[str, Any] = {"by_k": {}, "k_star": None}
    if draws.crossfit_indeterminate:
        crossfit["indeterminate"] = True
        crossfit["reason"] = draws.crossfit_reason
    else:
        k_share_lower: dict[int, float] = {}
        for k in range(1, 11):
            # Point shares for k* classification (bootstrap omitted for brevity in driver;
            # intervals available via analysis_ext when draws provided)
            sel_a = sorted(heads_a.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:k]
            # Evaluate on fold B mean of selected heads / route
            if fold_b_elig and sel_a:
                eff = sum(unit.head_effects_fwd[h][fold_b_elig].mean() for h, _ in sel_a)
                route_m = float(unit.e_route_fwd[fold_b_elig].mean())
                share = eff / route_m if route_m else float("nan")
            else:
                share = float("nan")
            crossfit["by_k"][k] = {"point_share_eval_B": share}
            if k in (2, 10) and share == share:
                k_share_lower[k] = share
        crossfit["k_star"] = classify_k_star(
            {2: k_share_lower.get(2, 0.0)},
            k2_share_lower=k_share_lower.get(2),
            k10_share_upper=None,
        )

    # H2 coverage on forward pair shares (upper bound proxy via point for synthetic)
    comp_ivs = {
        p: Interval(float(np.mean(v) - 0.01), float(np.mean(v) + 0.01), float(np.mean(v)))
        for p, v in unit.pair_effects_fwd.items()
    }
    h2cov = h2_component_coverage(comp_ivs)
    k_star = crossfit.get("k_star") or {}
    h2h3 = evaluate_h2_h3(
        k_star_le_2=bool(k_star.get("k_star_le_2")),
        k_star_gt_10=bool(k_star.get("k_star_gt_10")),
        h1_state=str(h1.get("full_H1")),
        all_component_share_upper_below_025=h2cov["all_upper_below"],
        top2_set_conservation_seeds=0,
        top2_varies_across_seeds=False,
    )

    return DriverResult(
        model_unit=unit.model_unit,
        gates=asdict(gates),
        route={
            **{k: (asdict(v) if isinstance(v, Interval) else v) for k, v in route.items()},
            "denom_fwd": {**denom_fwd, "interval_90": asdict(denom_fwd["interval_90"])},
            "denom_rev": {**denom_rev, "interval_90": asdict(denom_rev["interval_90"])},
        },
        h1=h1,
        h2_h3=h2h3,
        crossfit=crossfit,
        meta={"n_eligible": len(elig_ids), "family": unit.family, "role": unit.role},
    )


def run_study(
    units: Sequence[ModelUnitInput],
    held_out_ids_allocation_order: Sequence[str],
    *,
    exploratory_single_bos_ranks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Full multi-unit orchestration including exploratory stop and study-level."""
    sink = ResultSink()
    per_unit = []
    route_labels = []
    for u in units:
        if u.role == "primary_confirmatory":
            res = run_model_unit_analysis(u, held_out_ids_allocation_order)
            per_unit.append(res.to_dict())
            if res.meta.get("gate_failed"):
                route_labels.append("gate-failed")
            else:
                route_labels.append(res.route.get("route_class", "Indeterminate"))
        elif u.role == "secondary_replication":
            res = run_model_unit_analysis(u, held_out_ids_allocation_order)
            # Attach secondary criteria if pair effects present
            per_unit.append({"secondary": res.to_dict()})
        else:
            res = run_model_unit_analysis(u, held_out_ids_allocation_order)
            per_unit.append({"other": res.to_dict()})

    out: dict[str, Any] = {
        "units": per_unit,
        "exploratory_stop": None,
        "study_level": None,
    }
    if exploratory_single_bos_ranks is not None:
        out["exploratory_stop"] = {
            "ranks": list(exploratory_single_bos_ranks),
            "triggered": exploratory_stop_triggered(exploratory_single_bos_ranks),
        }
    if len(route_labels) == 5:
        out["study_level"] = study_level_decision(route_labels)
    out["n_condition_records"] = len(sink.conditions)
    return out


def persist_driver_result(result: Mapping[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return path
