"""Engine condition-schedule enumeration for workload reconciliation."""

from __future__ import annotations

from typing import Any, Sequence

from traces.pilot.controls import DAMAGE_MATCH_N_CANDIDATES
from traces.pilot.transfer import DEFAULT_K_BUDGETS, SEEDS, all_directed_seed_pairs


def enumerate_engine_condition_schedule(
    *,
    seeds: Sequence[int] = SEEDS,
    k_budgets: Sequence[int] = DEFAULT_K_BUDGETS,
    n_discovery: int = 50,
    n_validation: int = 25,
    n_held_out: int = 25,
    n_components: int = 396,
    n_damage_candidates: int = DAMAGE_MATCH_N_CANDIDATES,
    conditions_per_pair_k: int = 5,
) -> dict[str, Any]:
    """Mirror the engine's scoring schedule for workload reconciliation.

    Counts instruction-level scoring events (one forward-family unit per
    instruction per condition), not wall-clock seconds.
    """
    directed = len(all_directed_seed_pairs(seeds))
    n_seeds = len(seeds)
    discovery_d_single = n_seeds * n_components * n_discovery
    discovery_intact_cached = n_seeds * n_discovery

    damage_k_evals = directed * len(k_budgets) * (1 + n_damage_candidates)
    damage_k_instruction_forwards = damage_k_evals * n_validation

    n_joint_conditions = directed * len(k_budgets) * conditions_per_pair_k
    joint_val_condition_evals = n_joint_conditions
    joint_held_condition_evals = n_joint_conditions
    joint_val_d = joint_val_condition_evals * n_validation
    joint_val_k = joint_val_condition_evals * n_validation
    joint_val_asr = joint_val_condition_evals * n_validation
    joint_held_d = joint_held_condition_evals * n_held_out
    joint_held_k = joint_held_condition_evals * n_held_out
    joint_held_asr = joint_held_condition_evals * n_held_out

    return {
        "n_seeds": n_seeds,
        "directed_pairs": directed,
        "k_budgets": list(k_budgets),
        "conditions_per_pair_k": conditions_per_pair_k,
        "n_joint_conditions_total": n_joint_conditions,
        "discovery": {
            "d_single_ablations": discovery_d_single,
            "intact_logprob_cache_entries": discovery_intact_cached,
        },
        "damage_matching_validation": {
            "set_evals": damage_k_evals,
            "instruction_forwards_approx": damage_k_instruction_forwards,
        },
        "joint_transfer": {
            "validation_condition_evals": joint_val_condition_evals,
            "held_out_condition_evals_if_gate_passes": joint_held_condition_evals,
            "validation_instruction_D": joint_val_d,
            "validation_instruction_K": joint_val_k,
            "validation_instruction_ASR": joint_val_asr,
            "held_out_instruction_D": joint_held_d,
            "held_out_instruction_K": joint_held_k,
            "held_out_instruction_ASR": joint_held_asr,
            "total_joint_instruction_D_both_splits": joint_val_d + joint_held_d,
            "note": (
                "v2 budget under-counted held-out joint conditions (assumed 3 primary "
                f"only). Engine schedules {conditions_per_pair_k} conditions × "
                f"{directed} pairs × {len(k_budgets)} k on both splits when gates pass "
                f"=> {joint_val_d + joint_held_d} instruction-level D scores "
                f"(={n_joint_conditions}×({n_validation}+{n_held_out}))."
            ),
        },
        "checkpoint_loads": n_seeds,
    }
