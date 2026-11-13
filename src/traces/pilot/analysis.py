"""Pre-registered analysis helpers (feasibility-focused; not definitive inference)."""

from __future__ import annotations

from typing import Any, Sequence

from traces.pilot.ranking import ranking_reliability_overlap
from traces.pilot.transfer import (
    NATIVE_D_MIN_FOR_RATIO,
    all_directed_seed_pairs,
    correlated_pair_note,
    paired_absolute_differences,
)


ADEQUACY_RULES: dict[str, Any] = {
    "within_model_reliability": {
        "metric": "top-k Jaccard between discovery fold rankings",
        "adequate_if": "median Jaccard >= 0.3 for k in {5,10} on discovery folds",
        "interpretation": (
            "Below bar => rankings unstable; transfer less informative. "
            "Not a confirmatory hypothesis test."
        ),
    },
    "nontrivial_native_effect": {
        "metric": "recipient_native_topk D on validation",
        "adequate_if": "D_native >= 0.05 (nats/token) OR ASR drop >= 0.2 at k=10",
        "interpretation": "If native top-k has no effect, transfer is uninterpretable.",
    },
    "damage_matched_transfer": {
        "metric": "per-pair D_transfer vs D_damage_matched and D_random under same recipient/split",
        "status": "descriptive_only",
        "note": (
            "Counting pairs where transfer exceeds matched controls is too weak to "
            "establish meaningful transfer. Report per-pair D alongside K; do not "
            "treat '>=2 of 6 pairs' as confirmatory evidence."
        ),
    },
    "scope_warning": (
        "n_seeds=3 is too small for definitive population inference. "
        "Held-out n=25 cannot support precise ASR population estimates."
    ),
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


def analyze_per_pair_results(per_condition_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-pair effects with benign damage; absolute diffs preferred over unstable ratios."""
    # Group by (donor, recipient, k, split)
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in per_condition_rows:
        cond = row["condition"]
        key = (
            f"donor{cond.get('donor_seed')}->recipient{cond['recipient_seed']}"
            f"|k{cond['k']}|{row['split']}"
        )
        groups.setdefault(key, {})[cond["source"]] = row

    pair_summaries = []
    for key, by_src in groups.items():
        transfer = by_src.get("frozen_donor_topk")
        native = by_src.get("recipient_native_topk")
        random_c = by_src.get("random")
        matched_struct = by_src.get("layer_type_matched")
        matched_dmg = by_src.get("benign_damage_matched")
        unmatched = by_src.get("benign_damage_unmatched")

        if not transfer or transfer.get("D") is None:
            continue
        entry: dict[str, Any] = {
            "group": key,
            "D_transfer": transfer["D"],
            "K_transfer": transfer.get("K"),
            "D_native": None if not native else native.get("D"),
            "K_native": None if not native else native.get("K"),
            "D_random": None if not random_c else random_c.get("D"),
            "K_random": None if not random_c else random_c.get("K"),
            "D_layer_type_matched": None if not matched_struct else matched_struct.get("D"),
            "K_layer_type_matched": None if not matched_struct else matched_struct.get("K"),
            "D_benign_damage_matched": None if not matched_dmg else matched_dmg.get("D"),
            "K_benign_damage_matched": None if not matched_dmg else matched_dmg.get("K"),
            "damage_match_unmatched": unmatched is not None,
            "donor_ranking_artifact_hash": transfer["condition"].get(
                "donor_ranking_artifact_hash"
            ),
        }
        if native and native.get("D") is not None:
            entry["vs_native"] = paired_absolute_differences(
                float(transfer["D"]),
                float(native["D"]),
                native_D=float(native["D"]),
                min_native_for_ratio=NATIVE_D_MIN_FOR_RATIO,
            )
        if matched_dmg and matched_dmg.get("D") is not None:
            entry["vs_damage_matched"] = paired_absolute_differences(
                float(transfer["D"]), float(matched_dmg["D"])
            )
        if random_c and random_c.get("D") is not None:
            entry["vs_random"] = paired_absolute_differences(
                float(transfer["D"]), float(random_c["D"])
            )
        pair_summaries.append(entry)

    return {
        "per_pair_summaries": pair_summaries,
        "adequacy_rules": ADEQUACY_RULES,
        "pair_dependence_note": correlated_pair_note(all_directed_seed_pairs()),
        "mechanism_warning": (
            "Do not claim mechanistic equivalence based on transferred ablation effects."
        ),
        "descriptive_transfer_count_note": ADEQUACY_RULES["damage_matched_transfer"],
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
        "adequacy_rules": ADEQUACY_RULES,
        "forbidden_claims": [
            "Do not equate component index overlap with identical mechanisms.",
            "Do not treat three seeds / six directed pairs as definitive evidence.",
            "Do not claim scientific novelty of cross-seed transfer.",
            "Do not claim mechanistic equivalence from transferred ablations.",
            "Final research question remains NOT YET SELECTED.",
            "Do not treat 'better than controls in 2/6 pairs' as confirmatory.",
        ],
    }
