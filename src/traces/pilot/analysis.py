"""Pre-registered analysis helpers (feasibility-focused; not definitive inference).

Comparison-pair identity is taken from durable ``comparison_*`` fields on each
condition — never inferred from control ``donor_seed`` (None for random;
recipient for native).
"""

from __future__ import annotations

from typing import Any, Sequence

from traces.pilot.ranking import ranking_reliability_overlap
from traces.pilot.transfer import (
    NATIVE_D_MIN_FOR_RATIO,
    REQUIRED_COMPARISON_SOURCES,
    all_directed_seed_pairs,
    correlated_pair_note,
    paired_absolute_differences,
)


ADEQUACY_RULES: dict[str, Any] = {
    "within_model_reliability": {
        "metric": "top-10 Jaccard between discovery fold rankings (primary k=10)",
        "gate": "discovery_fold_reliability",
        "adequate_if": (
            "At least one seed has top-10 discovery-fold Jaccard >= 0.3; "
            "fail only if EVERY seed is below 0.3"
        ),
        "on_failure": (
            "Scientific early-stop: skip held-out for all recipients; "
            "validation transfer scoring remains available for native-gate recording. "
            "Not an execution failure."
        ),
        "interpretation": (
            "Below bar => rankings unstable; held-out transfer less informative. "
            "Not a confirmatory hypothesis test. Donor reliability does not by itself "
            "block validation scoring; it blocks held-out evaluation when every seed fails."
        ),
        "prereg_version": "v2.2_amendment",
    },
    "nontrivial_native_effect": {
        "metric": "recipient_native_topk |D| on validation at each k in {5,10}",
        "gate": "recipient_native_validation_effect",
        "adequate_if": (
            "For a recipient, pass if ANY k in {5,10} has |D_native(k)| >= 0.05 "
            "(nats/token). Evaluated from the full set of per-k native scores; "
            "independent of loop order (max |D| over k, not last-k overwrite)."
        ),
        "on_failure": (
            "Skip that recipient's held-out measurements; preserve completed "
            "validation rows. Unit may still be complete (validation-only)."
        ),
        "interpretation": (
            "If native top-k has no effect at either k, transfer on that recipient "
            "must not be interpreted as meaningful against a null native baseline."
        ),
        "prereg_version": "v2.2_amendment",
    },
    "damage_matched_transfer": {
        "metric": (
            "per comparison-pair D_transfer vs D_damage_matched and D_random "
            "under the same recipient/split (grouped by comparison_pair_key)"
        ),
        "status": "descriptive_only",
        "note": (
            "Counting pairs where transfer exceeds matched controls is too weak to "
            "establish meaningful transfer. Report per-pair D alongside K; do not "
            "treat '>=2 of 6 pairs' as confirmatory evidence."
        ),
    },
    "prompt_level_observations": {
        "metric": "per-instruction D and K keyed by frozen instruction_id",
        "status": "descriptive_spread_only",
        "note": (
            "Support descriptive paired differences and spread across the same "
            "held-out instructions. Do not introduce unregistered confirmatory tests."
        ),
        "prereg_version": "v2.2_amendment",
    },
    "scope_warning": (
        "n_seeds=3 is too small for definitive population inference. "
        "Held-out n=25 cannot support precise ASR population estimates."
    ),
    "engineering_vs_science": {
        "note": (
            "Workload/budget estimate version bumps (e.g. unitized schedule v1) are "
            "engineering. Scientific protocol changes require a dated prereg amendment "
            "(v2.2). Do not conflate the two."
        ),
    },
}


def _comparison_group_key(row: dict[str, Any]) -> str:
    cond = row["condition"]
    pair_key = cond.get("comparison_pair_key")
    if not pair_key:
        d = cond.get("comparison_donor_seed")
        r = cond.get("comparison_recipient_seed")
        if d is None or r is None:
            raise RuntimeError(
                "Ambiguous comparison grouping: condition lacks comparison_pair_key "
                f"and comparison_donor/recipient seeds (source={cond.get('source')}, "
                f"donor_seed={cond.get('donor_seed')}). Refusing silent regrouping."
            )
        pair_key = f"donor{d}->recipient{r}"
    return f"{pair_key}|k{cond['k']}|{row['split']}"


def analyze_per_pair_results(per_condition_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per comparison-pair effects; absolute diffs preferred over unstable ratios."""
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in per_condition_rows:
        cond = row["condition"]
        if "comparison_donor_seed" not in cond and "comparison_pair_key" not in cond:
            raise RuntimeError(
                "Refuse analysis of rows without durable comparison-pair identity"
            )
        key = _comparison_group_key(row)
        src = cond["source"]
        bucket = groups.setdefault(key, {})
        if src in bucket:
            raise RuntimeError(
                f"Ambiguous duplicate source {src!r} in comparison group {key}; "
                "refusing silent overwrite"
            )
        bucket[src] = row

    pair_summaries = []
    errors: list[str] = []
    for key, by_src in groups.items():
        transfer = by_src.get("frozen_donor_topk")
        if not transfer:
            # Control-only residue (should not happen with correct identity)
            errors.append(f"{key}: missing frozen_donor_topk")
            continue
        if transfer.get("D") is None and by_src.get("benign_damage_unmatched") is None:
            # Unmatched-only groups still need transfer D for vs_* 
            pass
        if transfer.get("D") is None:
            continue

        missing = [
            s
            for s in REQUIRED_COMPARISON_SOURCES
            if s not in by_src
            and not (s == "benign_damage_matched" and "benign_damage_unmatched" in by_src)
        ]
        # layer_type always required; damage is matched OR unmatched
        required_core = ["frozen_donor_topk", "recipient_native_topk", "random", "layer_type_matched"]
        missing_core = [s for s in required_core if s not in by_src]
        if missing_core:
            errors.append(f"{key}: missing controls {missing_core}")
            continue
        if (
            "benign_damage_matched" not in by_src
            and "benign_damage_unmatched" not in by_src
        ):
            errors.append(f"{key}: missing damage-matched or unmatched marker")
            continue

        native = by_src["recipient_native_topk"]
        random_c = by_src["random"]
        matched_struct = by_src["layer_type_matched"]
        matched_dmg = by_src.get("benign_damage_matched")
        unmatched = by_src.get("benign_damage_unmatched")
        cond_t = transfer["condition"]

        entry: dict[str, Any] = {
            "group": key,
            "comparison_pair_key": cond_t.get("comparison_pair_key")
            or f"donor{cond_t['comparison_donor_seed']}->recipient{cond_t['comparison_recipient_seed']}",
            "comparison_donor_seed": cond_t.get("comparison_donor_seed"),
            "comparison_recipient_seed": cond_t.get("comparison_recipient_seed"),
            "k": cond_t.get("k"),
            "split": transfer.get("split"),
            "D_transfer": transfer["D"],
            "K_transfer": transfer.get("K"),
            "D_native": native.get("D"),
            "K_native": native.get("K"),
            "D_random": random_c.get("D"),
            "K_random": random_c.get("K"),
            "D_layer_type_matched": matched_struct.get("D"),
            "K_layer_type_matched": matched_struct.get("K"),
            "D_benign_damage_matched": None if not matched_dmg else matched_dmg.get("D"),
            "K_benign_damage_matched": None if not matched_dmg else matched_dmg.get("K"),
            "damage_match_unmatched": unmatched is not None,
            "donor_ranking_artifact_hash": cond_t.get("donor_ranking_artifact_hash"),
        }
        if native.get("D") is None:
            errors.append(f"{key}: native D missing")
            continue
        entry["vs_native"] = paired_absolute_differences(
            float(transfer["D"]),
            float(native["D"]),
            native_D=float(native["D"]),
            min_native_for_ratio=NATIVE_D_MIN_FOR_RATIO,
        )
        if random_c.get("D") is None:
            errors.append(f"{key}: random D missing")
            continue
        entry["vs_random"] = paired_absolute_differences(
            float(transfer["D"]), float(random_c["D"])
        )
        if matched_dmg is not None:
            if matched_dmg.get("D") is None:
                errors.append(f"{key}: damage-matched D missing")
                continue
            entry["vs_damage_matched"] = paired_absolute_differences(
                float(transfer["D"]), float(matched_dmg["D"])
            )
        # Descriptive prompt-level spread when per-instruction vectors align
        d_t = transfer.get("D_per_instruction") or {}
        d_n = native.get("D_per_instruction") or {}
        if d_t and d_n and set(d_t) == set(d_n):
            paired = [float(d_t[i]) - float(d_n[i]) for i in sorted(d_t)]
            entry["descriptive_prompt_level"] = {
                "n_shared_instructions": len(paired),
                "mean_D_transfer_minus_native": (
                    sum(paired) / len(paired) if paired else None
                ),
                "note": ADEQUACY_RULES["prompt_level_observations"]["note"],
            }
        pair_summaries.append(entry)

    if errors:
        raise RuntimeError(
            "Pairwise analysis refused due to missing/ambiguous controls:\n- "
            + "\n- ".join(errors)
        )

    return {
        "per_pair_summaries": pair_summaries,
        "n_comparison_groups": len(pair_summaries),
        "adequacy_rules": ADEQUACY_RULES,
        "pair_dependence_note": correlated_pair_note(all_directed_seed_pairs()),
        "mechanism_warning": (
            "Do not claim mechanistic equivalence based on transferred ablation effects."
        ),
        "descriptive_transfer_count_note": ADEQUACY_RULES["damage_matched_transfer"],
    }


def summarize_reliability(
    fold_rankings: Sequence[Sequence[str]],
    ks: Sequence[int] = (5, 10),
) -> dict[str, Any]:
    if len(fold_rankings) < 2:
        raise ValueError("Need at least two fold rankings")
    rows = [ranking_reliability_overlap(fold_rankings[0], fold_rankings[1], k) for k in ks]
    return {
        "pairwise_overlaps": rows,
        "adequacy_rule": ADEQUACY_RULES["within_model_reliability"],
        "mechanism_warning": (
            "Component overlap is not proof of identical computational mechanisms."
        ),
    }


def analysis_plan_stub() -> dict[str, Any]:
    return {
        "primary_discovery_metric": "D_r(S) teacher_forced_per_token",
        "primary_sign": "positive_D_means_suppresses_target_likelihood",
        "benign_damage": "K_r(S) KL intact||intervened on control prompts, non-pad positions",
        "confirmation_metric": "greedy_asr_substring_and_full_payload",
        "secondary_metric": "trigger_minus_control_logprob_contrast (not a D substitute)",
        "k_budgets": [5, 10],
        "seeds": [42, 43, 44],
        "directed_transfers": [p.key() for p in all_directed_seed_pairs()],
        "comparison_identity": (
            "comparison_donor_seed / comparison_recipient_seed / comparison_pair_key "
            "on every condition row"
        ),
        "adequacy_rules": ADEQUACY_RULES,
        "forbidden_claims": [
            "Do not equate component index overlap with identical mechanisms.",
            "Do not treat three seeds / six directed pairs as definitive evidence.",
            "Do not claim scientific novelty of cross-seed transfer.",
            "Do not claim mechanistic equivalence from transferred ablations.",
            "Final research question remains NOT YET SELECTED.",
            "Do not treat 'better than controls in 2/6 pairs' as confirmatory.",
        ],
        "prereg_amendment": "docs/pilot_preregistration_v2_2_amendment.md",
    }
