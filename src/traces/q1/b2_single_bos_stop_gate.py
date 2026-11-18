"""Single-BOS exploratory 64-pair stop gate (development-only helpers).

Frozen rule: if nominated pair (L15.h6, L16.h5) ranks worse than 10th among
64 pairs on ≥2 of 3 exploratory seeds, HALT before confirmatory/held-out work.
Double-BOS ranks must never feed this gate.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from traces.q1.analysis import (
    DEV_INELIGIBLE_MAX,
    EXPLORATORY_STOP_RANK,
    exploratory_stop_triggered,
    rank_pairs_signed,
)
from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS
from traces.q1.f4_exploratory import F4_SEEDS, INTENDED_IMAGE_ID, refuse_held_out_access
from traces.q1.frozen_artifacts import (
    EXPECTED_N_DEV,
    FROZEN_DEV_ID_SEQUENCE_SHA256,
)
from traces.q1.margin import PAYLOAD_TOKEN_A
from traces.q1.patching import NOMINATED_PAIR
from traces.q1.scientific_execution import NOMINATED_PAIR_KEY

STAGE_NAME = "b2_single_bos_exploratory_stop_gate"
SCI_AUTH_ENV = "TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION"
FORBIDDEN_PIPELINE_ENTRYPOINTS = (
    "run_live_scientific_pipeline",
    "run_live_scientific_pipeline_from_hooks",
)
CAP_USD = 2.0
TIMEOUT_SEC = 45 * 60
N_PAIRS = 64
NOMINATED = NOMINATED_PAIR_KEY  # (6, 5)
assert NOMINATED == (NOMINATED_PAIR[0].head_index, NOMINATED_PAIR[1].head_index)
assert NOMINATED in EXPECTED_L15_L16_PAIRS
assert len(EXPECTED_L15_L16_PAIRS) == N_PAIRS


@dataclass(frozen=True)
class SingleBosStopScope:
    stage: str = STAGE_NAME
    seeds: tuple[int, ...] = F4_SEEDS
    n_prompts: int = EXPECTED_N_DEV
    n_pairs: int = N_PAIRS
    bos_convention: str = "single"
    nominated_pair: tuple[int, int] = NOMINATED
    stop_rank_threshold: int = EXPLORATORY_STOP_RANK
    ineligible_max_frac: float = DEV_INELIGIBLE_MAX
    image_id: str = INTENDED_IMAGE_ID
    cap_usd: float = CAP_USD
    timeout_sec: int = TIMEOUT_SEC
    allows_held_out: bool = False
    allows_confirmatory: bool = False
    allows_double_bos_to_drive_stop: bool = False
    allows_live_scientific_pipeline: bool = False


DEFAULT_SCOPE = SingleBosStopScope()


def assert_single_bos_stop_scope(scope: SingleBosStopScope = DEFAULT_SCOPE) -> list[str]:
    fails: list[str] = []
    if scope.stage != STAGE_NAME:
        fails.append(f"stage drift {scope.stage!r}")
    if tuple(scope.seeds) != F4_SEEDS:
        fails.append(f"seeds {scope.seeds} != {F4_SEEDS}")
    if scope.n_prompts != EXPECTED_N_DEV:
        fails.append("n_prompts must be 50")
    if scope.n_pairs != N_PAIRS:
        fails.append("n_pairs must be 64")
    if scope.bos_convention != "single":
        fails.append("bos_convention must be single")
    if tuple(scope.nominated_pair) != NOMINATED:
        fails.append(f"nominated pair must be {NOMINATED}")
    if scope.stop_rank_threshold != EXPLORATORY_STOP_RANK:
        fails.append("stop_rank_threshold drift")
    if abs(float(scope.ineligible_max_frac) - DEV_INELIGIBLE_MAX) > 1e-15:
        fails.append("ineligible_max_frac drift")
    if scope.allows_held_out:
        fails.append("held-out forbidden")
    if scope.allows_confirmatory:
        fails.append("confirmatory forbidden")
    if scope.allows_double_bos_to_drive_stop:
        fails.append("double-BOS must not drive single-BOS stop")
    if scope.allows_live_scientific_pipeline:
        fails.append("live scientific pipeline forbidden")
    if scope.cap_usd > CAP_USD + 1e-12:
        fails.append("cap_usd exceeds internal $2")
    if scope.timeout_sec > TIMEOUT_SEC:
        fails.append("timeout exceeds 45 min")
    return fails


def scientific_flag_must_be_unset() -> bool:
    return os.environ.get(SCI_AUTH_ENV) not in {"1", "true", "TRUE", "yes", "YES"}


def refuse_confirmatory_callback(name: str) -> None:
    if name in FORBIDDEN_PIPELINE_ENTRYPOINTS:
        raise PermissionError(f"single-BOS stop gate forbids {name!r}")


def refuse_double_bos_rank_input(bos_convention: str) -> None:
    if str(bos_convention) != "single":
        raise PermissionError(
            f"single-BOS stop gate refuses bos_convention={bos_convention!r}"
        )


def evaluate_development_ineligibility(
    *,
    b_i_by_prompt: Mapping[str, int],
    prompt_ids: Sequence[str],
    payload_a: int = PAYLOAD_TOKEN_A,
    max_frac: float = DEV_INELIGIBLE_MAX,
) -> dict[str, Any]:
    """Fail model metric gate if >10% of development prompts have a == b_i."""
    ids = [str(x) for x in prompt_ids]
    if len(ids) != EXPECTED_N_DEV:
        raise ValueError(f"need {EXPECTED_N_DEV} prompts, got {len(ids)}")
    if set(str(k) for k in b_i_by_prompt) != set(ids):
        raise ValueError("b_i_by_prompt must cover exactly the frozen prompt IDs")
    ineligible = [pid for pid in ids if int(b_i_by_prompt[pid]) == int(payload_a)]
    n_inelig = len(ineligible)
    frac = n_inelig / float(len(ids))
    return {
        "n_total": len(ids),
        "n_ineligible": n_inelig,
        "n_eligible": len(ids) - n_inelig,
        "fraction_ineligible": frac,
        "ineligible_prompt_ids": ineligible,
        "eligible_prompt_ids": [pid for pid in ids if pid not in set(ineligible)],
        "max_frac": float(max_frac),
        "gate_failed": frac > float(max_frac) + 1e-15,
        "payload_a": int(payload_a),
    }


def mean_signed_effects_by_pair(
    *,
    signed_effects_by_pair_prompt: Mapping[tuple[int, int], Mapping[str, float]],
    eligible_prompt_ids: Sequence[str],
) -> dict[tuple[int, int], float]:
    """Mean signed effect over eligible prompts only; require complete 64 pairs."""
    elig = [str(x) for x in eligible_prompt_ids]
    if set(signed_effects_by_pair_prompt) != set(EXPECTED_L15_L16_PAIRS):
        missing = EXPECTED_L15_L16_PAIRS - set(signed_effects_by_pair_prompt)
        extra = set(signed_effects_by_pair_prompt) - EXPECTED_L15_L16_PAIRS
        raise ValueError(f"pair set incomplete missing={len(missing)} extra={len(extra)}")
    means: dict[tuple[int, int], float] = {}
    for pair, by_pid in signed_effects_by_pair_prompt.items():
        vals = [float(by_pid[pid]) for pid in elig if pid in by_pid]
        if len(vals) != len(elig):
            raise ValueError(f"pair {pair}: incomplete eligible effects")
        means[pair] = sum(vals) / len(vals) if vals else 0.0
    return means


def nominated_rank_from_means(
    means: Mapping[tuple[int, int], float],
) -> dict[str, Any]:
    ranking = rank_pairs_signed(dict(means))
    rank_map = {p: r for p, _, r in ranking}
    if NOMINATED not in rank_map:
        raise KeyError(f"nominated {NOMINATED} missing from ranking")
    return {
        "nominated_pair": list(NOMINATED),
        "rank": int(rank_map[NOMINATED]),
        "effect": float(means[NOMINATED]),
        "ranking": [
            {"pair": list(p), "effect": float(e), "rank": int(r)} for p, e, r in ranking
        ],
        "tie_break": "(head15_index, head16_index) ascending after descending effect",
        "n_pairs": len(ranking),
    }


def apply_exploratory_stop_gate(ranks_by_seed: Mapping[int, int]) -> dict[str, Any]:
    """Mechanical frozen stop: ≥2 of 3 ranks > 10 → HALT."""
    seeds = list(F4_SEEDS)
    if set(int(s) for s in ranks_by_seed) != set(seeds):
        raise ValueError(f"ranks_by_seed must cover seeds {seeds}")
    ranks = [int(ranks_by_seed[s]) for s in seeds]
    for r in ranks:
        if not (1 <= r <= N_PAIRS):
            raise ValueError(f"rank {r} out of 1..{N_PAIRS}")
    triggered = exploratory_stop_triggered(ranks)
    worse = [int(s) for s, r in zip(seeds, ranks) if r > EXPLORATORY_STOP_RANK]
    if triggered:
        verdict = "HALT"
        detail = (
            "nominated pair ranked worse than 10th on ≥2 of 3 exploratory seeds; "
            "halt before confirmatory/held-out evaluation"
        )
    else:
        verdict = "EXPLORATORY_STOP_PASSED"
        detail = (
            "single-BOS exploratory stop not triggered (weak evidence only; "
            "main experiment still unauthorized)"
        )
    return {
        "verdict": verdict,
        "triggered": bool(triggered),
        "ranks": ranks,
        "ranks_by_seed": {str(s): int(ranks_by_seed[s]) for s in seeds},
        "seeds": list(seeds),
        "seeds_worse_than_10": worse,
        "n_worse_than_10": len(worse),
        "criterion": "rank > 10 on at least 2 of 3 exploratory seeds",
        "stop_rank_threshold": EXPLORATORY_STOP_RANK,
        "nominated_pair": list(NOMINATED),
        "bos_convention": "single",
        "detail": detail,
        "confirmatory_blocked": bool(triggered),
        "main_experiment_authorized": False,
    }


def build_stop_gate_artifact_payload(
    *,
    per_seed_results: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
    dataset_sha: str = FROZEN_DEV_ID_SEQUENCE_SHA256,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_type": "q1_single_bos_exploratory_stop_gate_v1",
        "claims_real_experiment": True,
        "fixture_purpose": "exploratory_single_bos_64_pair_stop_gate",
        "bos_convention": "single",
        "alters_protocol": False,
        "uses_double_bos_ranks": False,
        "dataset_binding": {
            "split": "development",
            "id_sequence_sha256": dataset_sha,
        },
        "nominated_pair": list(NOMINATED),
        "n_pairs": N_PAIRS,
        "per_seed_results": {k: dict(v) for k, v in per_seed_results.items()},
        "gate": dict(gate),
        "meta": dict(meta or {}),
    }


# ---------------------------------------------------------------------------
# CPU adversarial fixtures (no GPU)
# ---------------------------------------------------------------------------


def adversarial_fixtures() -> list[dict[str, Any]]:
    """Return named pass/fail cases for unit tests."""
    prompts = [str(i) for i in range(EXPECTED_N_DEV)]
    cases: list[dict[str, Any]] = []

    # Eligibility >10% fails
    b_i = {pid: 1000 for pid in prompts}
    for pid in prompts[:6]:  # 6/50 = 12%
        b_i[pid] = PAYLOAD_TOKEN_A
    elig = evaluate_development_ineligibility(b_i_by_prompt=b_i, prompt_ids=prompts)
    cases.append(
        {
            "name": "ineligibility_over_10pct_fails",
            "ok": elig["gate_failed"] is True and elig["n_ineligible"] == 6,
        }
    )

    # Eligibility exactly 10% (5/50) passes gate
    b_i2 = {pid: 1000 for pid in prompts}
    for pid in prompts[:5]:
        b_i2[pid] = PAYLOAD_TOKEN_A
    elig2 = evaluate_development_ineligibility(b_i_by_prompt=b_i2, prompt_ids=prompts)
    cases.append(
        {
            "name": "ineligibility_at_10pct_passes",
            "ok": elig2["gate_failed"] is False and elig2["n_ineligible"] == 5,
        }
    )

    # Complete 64 + tie-break
    means = {(h15, h16): float(h15) + 0.01 * h16 for h15 in range(8) for h16 in range(8)}
    # Force nominated (6,5) to unique top effect
    means[(6, 5)] = 100.0
    means[(6, 4)] = 100.0  # tie on effect → (6,4) wins tie-break (smaller h16)
    rec = nominated_rank_from_means(means)
    cases.append(
        {
            "name": "tie_break_head16_ascending",
            "ok": rec["rank"] == 2 and rec["ranking"][0]["pair"] == [6, 4],
        }
    )

    # Stop triggered: ranks 11, 12, 3 → 2 worse
    g1 = apply_exploratory_stop_gate({42: 11, 43: 12, 44: 3})
    cases.append({"name": "stop_triggered_two_worse", "ok": g1["verdict"] == "HALT"})

    # Stop not triggered: ranks 11, 5, 4 → only 1 worse
    g2 = apply_exploratory_stop_gate({42: 11, 43: 5, 44: 4})
    cases.append(
        {
            "name": "stop_clear_one_worse",
            "ok": g2["verdict"] == "EXPLORATORY_STOP_PASSED",
        }
    )

    # Missing pair rejected
    bad_means = dict(means)
    del bad_means[(0, 0)]
    try:
        mean_signed_effects_by_pair(
            signed_effects_by_pair_prompt={
                p: {pid: 0.0 for pid in prompts} for p in bad_means
            },
            eligible_prompt_ids=prompts,
        )
        miss_ok = False
    except ValueError:
        miss_ok = True
    cases.append({"name": "missing_pair_rejected", "ok": miss_ok})

    # Double-BOS refused
    try:
        refuse_double_bos_rank_input("double")
        db_ok = False
    except PermissionError:
        db_ok = True
    cases.append({"name": "double_bos_refused", "ok": db_ok})

    # Held-out refused
    try:
        refuse_held_out_access("results/q1_selection/held_out_instructions.json")
        ho_ok = False
    except PermissionError:
        ho_ok = True
    cases.append({"name": "held_out_refused", "ok": ho_ok})

    # Wrong direction / swapped nominated cannot be used as substitute
    cases.append(
        {
            "name": "nominated_frozen_not_substitutable",
            "ok": NOMINATED == (6, 5) and (5, 6) not in {NOMINATED},
        }
    )

    # Scope defaults
    cases.append(
        {
            "name": "default_scope_ok",
            "ok": assert_single_bos_stop_scope() == [],
        }
    )

    return cases


def scope_dict(scope: SingleBosStopScope = DEFAULT_SCOPE) -> dict[str, Any]:
    d = asdict(scope)
    d["nominated_pair"] = list(scope.nominated_pair)
    d["seeds"] = list(scope.seeds)
    return d
