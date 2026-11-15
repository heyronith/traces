"""Deterministic Recommended-tier workload enumerator (CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traces.q1 import FROZEN_PROTOCOL_VERSION, FROZEN_SCIENTIFIC_COMMIT
from traces.q1.conditions import (
    count_primary_tf_per_prompt,
    enumerate_double_bos_descriptive_pairs,
    enumerate_primary_l19_conditions,
    enumerate_secondary_l1523_conditions,
)

N_DEV = 50
N_HO = 200
N_ALL = 250

S_TF = 0.056
S_GEN = 1.633
S_LOAD = 180.0
S_UNLOAD = 15.0
USD_PER_SEC = 0.000306
USD_PER_HOUR = 1.10

EXPECTED_TF = 824_400
EXPECTED_GEN = 3_600
EXPECTED_LOADS = 11


def build_enumerator(*, include_downstream56: bool = True) -> dict[str, Any]:
    primary = count_primary_tf_per_prompt(include_downstream56=include_downstream56)
    # Cross-check condition list length
    n_conds = len(
        enumerate_primary_l19_conditions(include_downstream56=include_downstream56)
    )
    assert n_conds + 2 == primary["tf_total"]

    secondary_conds = len(enumerate_secondary_l1523_conditions())
    secondary_tf_pp = secondary_conds + 2  # + intact T/C
    assert secondary_tf_pp == 136

    double_bos_pairs = len(enumerate_double_bos_descriptive_pairs())
    expl_extra = {
        "f4_repeat_intact_extra_tf": 2 * 2 * N_DEV,
        "double_bos_intact_tf": 2 * N_DEV,
        "double_bos_pairs_forward_only": double_bos_pairs * N_DEV,
    }
    assert expl_extra["double_bos_pairs_forward_only"] == 3200

    units: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        tf = primary["tf_total"] * N_DEV + sum(expl_extra.values())
        units.append(
            {
                "unit": f"exploratory_l1523_seed{seed}",
                "family": "l1523",
                "role": "exploratory_dev",
                "seed": seed,
                "n_prompts": N_DEV,
                "tf_passes": tf,
                "gen_passes": primary["gen_total"] * N_DEV,
            }
        )
    for seed in (42, 43, 44, 45, 46):
        units.append(
            {
                "unit": f"primary_l19_seed{seed}",
                "family": "l19",
                "role": "primary_confirmatory",
                "seed": seed,
                "n_prompts": N_HO,
                "tf_passes": primary["tf_total"] * N_HO,
                "gen_passes": primary["gen_total"] * N_HO,
            }
        )
    for seed in (45, 46):
        units.append(
            {
                "unit": f"secondary_l1523_seed{seed}",
                "family": "l1523",
                "role": "secondary_replication",
                "seed": seed,
                "n_prompts": N_HO,
                "tf_passes": secondary_tf_pp * N_HO,
                "gen_passes": 2 * N_HO,
            }
        )
    units.append(
        {
            "unit": "adapter_off_base_reference",
            "family": "base",
            "role": "adapter_off_gates_descriptives",
            "seed": None,
            "n_prompts": N_ALL,
            "tf_passes": 2 * N_ALL,
            "gen_passes": 2 * N_ALL,
        }
    )

    tf_total = sum(u["tf_passes"] for u in units)
    gen_total = sum(u["gen_passes"] for u in units)
    n_loads = 11

    def band(mult_tf: float, mult_gen: float, contingency: float) -> dict[str, float]:
        raw = (
            tf_total * S_TF * mult_tf
            + gen_total * S_GEN * mult_gen
            + n_loads * (S_LOAD + S_UNLOAD)
        ) * contingency
        hours = raw / 3600.0
        return {
            "seconds": raw,
            "gpu_hours": hours,
            "usd_modal_per_sec": raw * USD_PER_SEC,
            "usd_repo_1_10_per_hr": hours * USD_PER_HOUR,
        }

    protocol_preserving_optimizations = [
        {
            "id": "OPT1",
            "name": "batch_size_gt_1_after_identity_proof",
            "preserves_conditions": True,
            "requires": "Identities 1–9 + F4 under candidate padding policy",
            "expected_effect": "Potential ~2–3× TF wall-clock if memory-bound; unverified",
        },
        {
            "id": "OPT2",
            "name": "cache_reuse_tt_cc_free_and_one_fold_crossfit",
            "preserves_conditions": True,
            "requires": "Already assumed in enumerator",
            "expected_effect": "Avoids redundant intact/hybrid/crossfit doubles",
        },
        {
            "id": "OPT3",
            "name": "minimize_modal_keepalive_checkpoint_per_unit",
            "preserves_conditions": True,
            "requires": "Container config",
            "expected_effect": "Cuts non-compute billed idle",
        },
        {
            "id": "OPT4",
            "name": "authorized_throughput_calibration_before_full_spend",
            "preserves_conditions": True,
            "requires": "Separate GPU authorization + expenditure cap",
            "expected_effect": "Replace mid/high multipliers with measured Q1 rates",
        },
    ]

    return {
        "artifact": "TRACES_Q1_recommended_workload_enumerator_v1_1",
        "frozen_commit": FROZEN_SCIENTIFIC_COMMIT,
        "protocol": f"v{FROZEN_PROTOCOL_VERSION} Recommended tier",
        "include_downstream56": include_downstream56,
        "validation": {
            "expected_tf": EXPECTED_TF if include_downstream56 else None,
            "expected_gen": EXPECTED_GEN,
            "expected_loads": EXPECTED_LOADS,
            "tf_passes": tf_total,
            "gen_passes": gen_total,
            "n_loads": n_loads,
            "tf_matches_readiness_v1": include_downstream56 and tf_total == EXPECTED_TF,
            "gen_matches_readiness_v1": gen_total == EXPECTED_GEN,
            "loads_match_readiness_v1": n_loads == EXPECTED_LOADS,
            "primary_tf_per_prompt": primary["tf_total"],
            "primary_conditions_enumerated": n_conds,
            "secondary_tf_per_prompt": secondary_tf_pp,
        },
        "exploratory_extras_per_model": expl_extra,
        "units": units,
        "cost_bands_gpu_only": {
            "optimistic_low": band(1.0, 1.0, 1.10),
            "planning_mid": band(1.5 * 1.3, 1.2, 1.20),
            "conservative_high": band(2.0, 1.5, 1.35),
        },
        "protocol_preserving_optimizations": protocol_preserving_optimizations,
        "no_automatic_condition_cuts": True,
    }


def validate_against_readiness_v1() -> dict[str, Any]:
    enum = build_enumerator(include_downstream56=True)
    v = enum["validation"]
    ok = (
        v["tf_matches_readiness_v1"]
        and v["gen_matches_readiness_v1"]
        and v["loads_match_readiness_v1"]
    )
    return {"ok": ok, "validation": v}


def write_enumerator(out_path: Path | str, **kwargs: Any) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_enumerator(**kwargs), indent=2) + "\n")
    return path
