"""Canonical H2/H3 component identity sets (protocol v1.3)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from traces.q1.conditions import N_HEADS, UPSTREAM_LAYERS, iter_layer15_16_pairs, iter_upstream_heads

EXPECTED_UPSTREAM_HEADS: frozenset[tuple[int, int]] = frozenset(
    (h.layer, h.head_index) for h in iter_upstream_heads()
)
EXPECTED_L15_L16_PAIRS: frozenset[tuple[int, int]] = frozenset(
    (a.head_index, b.head_index) for a, b in iter_layer15_16_pairs()
)

assert len(EXPECTED_UPSTREAM_HEADS) == 152
assert len(EXPECTED_L15_L16_PAIRS) == 64
assert len(UPSTREAM_LAYERS) * N_HEADS == 152


def validate_effect_vector(
    arr: Any,
    *,
    n_eligible: int,
    name: str,
) -> list[str]:
    """Reject wrong-length, nonfinite, or misaligned effect arrays."""
    failures: list[str] = []
    try:
        a = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as e:
        return [f"{name}: not array-convertible ({e})"]
    if a.ndim != 1:
        failures.append(f"{name}: expected 1-d, got shape {a.shape}")
        return failures
    if len(a) != n_eligible:
        failures.append(f"{name}: length {len(a)} != n_eligible {n_eligible}")
    if not np.all(np.isfinite(a)):
        failures.append(f"{name}: nonfinite values present")
    return failures


def validate_head_effect_map(
    head_effects: Mapping[tuple[int, int], Any],
    *,
    n_eligible: int,
    require_complete_upstream: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    keys = frozenset(head_effects.keys())
    if require_complete_upstream:
        missing = EXPECTED_UPSTREAM_HEADS - keys
        extra = keys - EXPECTED_UPSTREAM_HEADS
        if missing:
            failures.append(
                f"missing {len(missing)} upstream heads (need 152 exact identities)"
            )
        if extra:
            failures.append(f"unexpected head keys: {sorted(extra)[:5]}…")
    for k, v in head_effects.items():
        failures.extend(validate_effect_vector(v, n_eligible=n_eligible, name=f"head{k}"))
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "n_heads": len(keys),
        "expected_n": 152 if require_complete_upstream else None,
    }


def validate_pair_effect_map(
    pair_effects: Mapping[tuple[int, int], Any],
    *,
    n_eligible: int,
    require_complete_64: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    keys = frozenset(pair_effects.keys())
    if require_complete_64:
        missing = EXPECTED_L15_L16_PAIRS - keys
        extra = keys - EXPECTED_L15_L16_PAIRS
        if missing:
            failures.append(f"missing {len(missing)} of 64 layer-matched pairs")
        if extra:
            failures.append(f"unexpected pair keys: {sorted(extra)[:5]}…")
    for k, v in pair_effects.items():
        failures.extend(validate_effect_vector(v, n_eligible=n_eligible, name=f"pair{k}"))
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "n_pairs": len(keys),
        "expected_n": 64 if require_complete_64 else None,
    }


def validate_five_primary_seeds(
    top2_by_seed: Mapping[Any, frozenset[tuple[int, int]]],
    *,
    expected_n: int = 5,
) -> dict[str, Any]:
    """H3 conservation requires exactly five distinct primary-confirmatory seeds."""
    failures: list[str] = []
    keys = list(top2_by_seed.keys())
    if len(keys) != expected_n:
        failures.append(f"need exactly {expected_n} seeds, got {len(keys)}")
    if len(set(keys)) != len(keys):
        failures.append("duplicate seed keys in top2 map")
    for seed, s in top2_by_seed.items():
        if not isinstance(s, frozenset) or len(s) != 2:
            failures.append(f"seed {seed}: top-2 must be frozenset of size 2")
            continue
        for item in s:
            if not (isinstance(item, tuple) and len(item) == 2):
                failures.append(f"seed {seed}: malformed head id {item!r}")
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "n_seeds": len(keys),
        "distinct": len(set(keys)),
    }


def validate_complete_signed_ranking(
    head_effects: Mapping[tuple[int, int], np.ndarray],
    *,
    n_eligible: int,
) -> dict[str, Any]:
    """Require complete upstream map before selecting top-two."""
    return validate_head_effect_map(
        head_effects, n_eligible=n_eligible, require_complete_upstream=True
    )
