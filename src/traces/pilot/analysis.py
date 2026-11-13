"""Pre-registered analysis helpers (descriptive; not definitive inference)."""

from __future__ import annotations

from typing import Any, Sequence

from traces.pilot.ranking import ranking_reliability_overlap
from traces.pilot.transfer import TransferResultRow, correlated_pair_note, all_directed_seed_pairs


# Provisional adequacy bars for this pilot (feasibility thresholds, not proofs).
ADEQUACY_RULES: dict[str, Any] = {
    "within_model_reliability": {
        "metric": "top-k Jaccard between discovery fold rankings",
        "adequate_if": "median Jaccard >= 0.3 for k in {5,10} on discovery folds",
        "interpretation": (
            "Below bar => rankings unstable; transfer results are less informative. "
            "Not a confirmatory hypothesis test."
        ),
    },
    "nontrivial_native_effect": {
        "metric": "recipient_native_topk vs intact on validation logprob contrast / ASR",
        "adequate_if": (
            "mean trigger logprob drop >= 0.05 nats/token OR ASR drop >= 0.2 "
            "on validation for k=10"
        ),
        "interpretation": "If native top-k has no effect, transfer is uninterpretable.",
    },
    "damage_matched_transfer": {
        "metric": (
            "frozen donor top-k effect on recipient vs layer/type-matched control "
            "and vs random control"
        ),
        "adequate_if": (
            "donor transfer effect exceeds matched control on the same metric "
            "for at least 2 of 6 directed pairs at k=10 (descriptive count only)"
        ),
        "interpretation": (
            "Exceeding damage-matched controls suggests transfer beyond generic damage. "
            "Does NOT prove identical mechanisms. Index overlap ≠ mechanism equivalence."
        ),
    },
    "scope_warning": (
        "n_seeds=3 is too small for definitive population inference. "
        "Report all directed pairs and dependence among pairs sharing a seed."
    ),
}


def summarize_reliability(
    fold_rankings: Sequence[Sequence[str]],
    ks: Sequence[int] = (5, 10),
) -> dict[str, Any]:
    if len(fold_rankings) < 2:
        raise ValueError("Need at least two fold rankings")
    rows = []
    for k in ks:
        ov = ranking_reliability_overlap(fold_rankings[0], fold_rankings[1], k)
        rows.append(ov)
    return {
        "pairwise_overlaps": rows,
        "adequacy_rule": ADEQUACY_RULES["within_model_reliability"],
        "mechanism_warning": (
            "Component overlap is not proof of identical computational mechanisms."
        ),
    }


def compare_transfer_vs_controls(
    rows: Sequence[TransferResultRow],
    *,
    metric: str = "trigger_logprob_contrast_delta",
) -> dict[str, Any]:
    """Descriptive comparison of transfer vs random / matched / native conditions."""
    by_source: dict[str, list[float]] = {}
    for r in rows:
        val = getattr(r, metric)
        if val is None and metric in r.extras:
            val = r.extras[metric]
        if val is None:
            continue
        by_source.setdefault(r.condition.source, []).append(float(val))

    means = {k: sum(v) / len(v) for k, v in by_source.items() if v}
    return {
        "metric": metric,
        "means_by_source": means,
        "n_by_source": {k: len(v) for k, v in by_source.items()},
        "adequacy_rule": ADEQUACY_RULES["damage_matched_transfer"],
        "pair_dependence_note": correlated_pair_note(all_directed_seed_pairs()),
    }


def analysis_plan_stub() -> dict[str, Any]:
    return {
        "primary_discovery_metric": "teacher_forced_trigger_target_logprob_drop",
        "confirmation_metric": "greedy_asr_substring_and_full_payload",
        "secondary_metrics": [
            "benign_kl_on_control_prompts",
            "control_tag_false_fire_change",
            "control_task_degradation_proxy",
        ],
        "k_budgets": [5, 10],
        "seeds": [42, 43, 44],
        "directed_transfers": [p.key() for p in all_directed_seed_pairs()],
        "adequacy_rules": ADEQUACY_RULES,
        "forbidden_claims": [
            "Do not equate component index overlap with identical mechanisms.",
            "Do not treat three seeds as definitive population evidence.",
            "Do not claim scientific novelty of cross-seed transfer.",
            "Final research question remains NOT YET SELECTED.",
        ],
    }
