"""F1 immutable bootstrap schedule and F2 even/odd fold assignment.

Uses numpy Generator(PCG64(20261114)) and the exact call sequence from §7.

A5: If a cross-fitting fold is empty after eligibility, still produce full-set
bootstrap draws; mark only cross-fitted endpoints indeterminate (no fold draws).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

F1_SEED = 20261114
F1_N_RESAMPLES = 2000
F1_NUMPY_PIN = "2.1.3"


@dataclass(frozen=True)
class F2Folds:
    """Held-out allocation-order folds (before eligibility filtering)."""

    fold_a_positions: tuple[int, ...]  # even zero-based positions
    fold_b_positions: tuple[int, ...]  # odd
    fold_a_ids: tuple[Any, ...]
    fold_b_ids: tuple[Any, ...]


def assign_f2_folds(held_out_ids_in_allocation_order: Sequence[Any]) -> F2Folds:
    """F2: fold A = even indices, fold B = odd, in frozen allocation order."""
    ids = list(held_out_ids_in_allocation_order)
    a_pos = tuple(i for i in range(len(ids)) if i % 2 == 0)
    b_pos = tuple(i for i in range(len(ids)) if i % 2 == 1)
    return F2Folds(
        fold_a_positions=a_pos,
        fold_b_positions=b_pos,
        fold_a_ids=tuple(ids[i] for i in a_pos),
        fold_b_ids=tuple(ids[i] for i in b_pos),
    )


def enforce_production_held_out_size(n: int, *, expected: int = 200) -> None:
    if n != expected:
        raise RuntimeError(
            f"Production confirmatory held-out size must be {expected}, got {n}"
        )


@dataclass(frozen=True)
class F1Draws:
    full_set: np.ndarray  # (2000, N)
    fold_a: np.ndarray | None  # (2000, N_A) or None if empty fold
    fold_b: np.ndarray | None
    n: int
    n_a: int
    n_b: int
    seed: int = F1_SEED
    crossfit_indeterminate: bool = False
    crossfit_reason: str | None = None


def draw_f1_resamples(
    n_eligible: int,
    n_eligible_fold_a: int,
    n_eligible_fold_b: int,
    *,
    seed: int = F1_SEED,
) -> F1Draws:
    """Exact F1 call sequence with A5 empty-fold handling.

    Always draws the full-set bootstrap. Fold draws are produced only when both
    eligible folds are non-empty; otherwise ``crossfit_indeterminate=True`` and
    fold arrays are None (cross-fitted endpoints only).
    """
    if n_eligible < 1:
        raise RuntimeError("F1: empty eligible set")
    rng = np.random.Generator(np.random.PCG64(seed))
    full = rng.integers(0, n_eligible, size=(F1_N_RESAMPLES, n_eligible))

    empty_a = n_eligible_fold_a < 1
    empty_b = n_eligible_fold_b < 1
    if empty_a or empty_b:
        reason = (
            f"empty eligible fold(s): n_A={n_eligible_fold_a} n_B={n_eligible_fold_b}; "
            "full-set bootstrap retained; cross-fitted endpoints indeterminate"
        )
        return F1Draws(
            full_set=full,
            fold_a=None,
            fold_b=None,
            n=n_eligible,
            n_a=n_eligible_fold_a,
            n_b=n_eligible_fold_b,
            seed=seed,
            crossfit_indeterminate=True,
            crossfit_reason=reason,
        )

    fold_a = rng.integers(0, n_eligible_fold_a, size=(F1_N_RESAMPLES, n_eligible_fold_a))
    fold_b = rng.integers(0, n_eligible_fold_b, size=(F1_N_RESAMPLES, n_eligible_fold_b))
    return F1Draws(
        full_set=full,
        fold_a=fold_a,
        fold_b=fold_b,
        n=n_eligible,
        n_a=n_eligible_fold_a,
        n_b=n_eligible_fold_b,
        seed=seed,
        crossfit_indeterminate=False,
        crossfit_reason=None,
    )


def f1_full_set_stream_sha256(n_eligible: int = 180, seed: int = F1_SEED) -> str:
    """Reference hash of the full-set draw block (for NumPy 2.1.3 pin checks)."""
    import hashlib

    draws = draw_f1_resamples(n_eligible, max(1, n_eligible // 2), max(1, n_eligible - n_eligible // 2), seed=seed)
    return hashlib.sha256(draws.full_set.tobytes()).hexdigest()


def percentile_linear(values: np.ndarray, q: list[float]) -> np.ndarray:
    return np.percentile(values, q, method="linear")


def percentile_rank_inverted_cdf(ranks: np.ndarray, q: list[float]) -> np.ndarray:
    return np.percentile(ranks, q, method="inverted_cdf")


def ratio_of_paired_means(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> float:
    """Ratio of means (not mean of ratios), for a single sample vector pair."""
    num = float(np.mean(numerator))
    den = float(np.mean(denominator))
    if den == 0.0:
        raise RuntimeError("Refusing ratio with zero denominator mean")
    return num / den
