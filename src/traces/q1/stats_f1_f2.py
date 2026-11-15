"""F1 immutable bootstrap schedule and F2 even/odd fold assignment.

Uses numpy Generator(PCG64(20261114)) and the exact call sequence from §7.
No outcome interpretation — index draws only.
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
    if len(ids) != 200:
        # Allow other lengths in unit tests; production confirmation expects 200.
        pass
    a_pos = tuple(i for i in range(len(ids)) if i % 2 == 0)
    b_pos = tuple(i for i in range(len(ids)) if i % 2 == 1)
    return F2Folds(
        fold_a_positions=a_pos,
        fold_b_positions=b_pos,
        fold_a_ids=tuple(ids[i] for i in a_pos),
        fold_b_ids=tuple(ids[i] for i in b_pos),
    )


@dataclass(frozen=True)
class F1Draws:
    full_set: np.ndarray  # (2000, N)
    fold_a: np.ndarray  # (2000, N_A)
    fold_b: np.ndarray  # (2000, N_B)
    n: int
    n_a: int
    n_b: int
    seed: int = F1_SEED


def draw_f1_resamples(
    n_eligible: int,
    n_eligible_fold_a: int,
    n_eligible_fold_b: int,
    *,
    seed: int = F1_SEED,
) -> F1Draws:
    """Exact F1 call sequence; empty fold must be handled by caller (halt)."""
    if n_eligible < 1:
        raise RuntimeError("F1: empty eligible set")
    if n_eligible_fold_a < 1 or n_eligible_fold_b < 1:
        raise RuntimeError(
            "F1/F2: empty eligible fold — halt cross-fitted endpoint (indeterminate)"
        )
    rng = np.random.Generator(np.random.PCG64(seed))
    full = rng.integers(0, n_eligible, size=(F1_N_RESAMPLES, n_eligible))
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
    )


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
