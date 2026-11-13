"""Control generators: random, layer/type-matched, and benign-damage-matched.

Layer/type matching and benign-damage matching are **different** controls.
Damage matching uses validation-only K (benign KL), never held-out, and never
triggered-backdoor suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from traces.pilot.components import ComponentRef

# Pre-specified before GPU execution (prereg v2.1 amendment).
DAMAGE_MATCH_N_CANDIDATES = 32
# Legacy names retained only for import compatibility; selection uses equivalence bound.
DAMAGE_MATCH_ABS_TOLERANCE = 0.005  # hard cap inside min(...)
DAMAGE_MATCH_REL_TOLERANCE = 0.25
DAMAGE_MATCH_K_FLOOR = 0.002
DAMAGE_MATCH_ABS_CAP = 0.005


def damage_match_permitted_bound(k_target: float) -> float:
    """Conservative equivalence bound (v2.1).

    abs(K_control - K_target) <= min(0.005, 0.25 * max(K_target, 0.002))
    """
    return float(min(DAMAGE_MATCH_ABS_CAP, DAMAGE_MATCH_REL_TOLERANCE * max(k_target, DAMAGE_MATCH_K_FLOOR)))


@dataclass(frozen=True)
class ControlSet:
    name: str
    component_ids: tuple[str, ...]
    kind: str  # random | layer_type_matched | benign_damage_matched | unmatched
    matched_to: tuple[str, ...] | None = None
    seed: int | None = None
    notes: str = ""
    target_benign_kl: float | None = None
    control_benign_kl: float | None = None
    match_tolerance_abs: float | None = None
    match_tolerance_rel: float | None = None
    match_abs_diff: float | None = None
    match_permitted_bound: float | None = None
    used_type_only_fallback: bool = False
    unmatched_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component_ids": list(self.component_ids),
            "kind": self.kind,
            "matched_to": list(self.matched_to) if self.matched_to else None,
            "seed": self.seed,
            "notes": self.notes,
            "target_benign_kl": self.target_benign_kl,
            "control_benign_kl": self.control_benign_kl,
            "match_tolerance_abs": self.match_tolerance_abs,
            "match_tolerance_rel": self.match_tolerance_rel,
            "match_abs_diff": self.match_abs_diff,
            "match_permitted_bound": self.match_permitted_bound,
            "used_type_only_fallback": self.used_type_only_fallback,
            "unmatched_reason": self.unmatched_reason,
        }


def _lcg_shuffle(items: list[str], seed: int) -> list[str]:
    state = seed
    out = list(items)

    def rnd(n: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) % (2**31)
        return state % n

    for i in range(len(out) - 1, 0, -1):
        j = rnd(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def random_control_set(
    catalog: Sequence[ComponentRef],
    k: int,
    *,
    seed: int,
    exclude: Sequence[str] | None = None,
) -> ControlSet:
    exclude_set = set(exclude or [])
    pool = [c.component_id for c in catalog if c.component_id not in exclude_set]
    if k > len(pool):
        raise ValueError("k larger than available catalog")
    shuffled = _lcg_shuffle(pool, seed)
    return ControlSet(
        name=f"random_k{k}_seed{seed}",
        component_ids=tuple(shuffled[:k]),
        kind="random",
        seed=seed,
        notes="Uniform random among catalog components (optional exclude of target set).",
    )


def layer_type_matched_control(
    catalog: Sequence[ComponentRef],
    target_ids: Sequence[str],
    *,
    seed: int,
) -> ControlSet:
    """Structural control: same (layer, kind) multiset when possible.

    Documents fallback to type-only matching. This is NOT benign-damage matching.
    """
    by_id = {c.component_id: c for c in catalog}
    targets = [by_id[i] for i in target_ids]
    used = set(target_ids)
    chosen: list[str] = []
    used_fallback = False
    state = seed

    def rnd(n: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) % (2**31)
        return state % n

    for t in targets:
        primary = [
            c.component_id
            for c in catalog
            if c.kind == t.kind and c.layer == t.layer and c.component_id not in used
        ]
        fallback = [
            c.component_id
            for c in catalog
            if c.kind == t.kind and c.component_id not in used
        ]
        if primary:
            pool = primary
        else:
            pool = fallback
            used_fallback = True
        if not pool:
            raise RuntimeError(f"No layer/type control available for {t.component_id}")
        pick = pool[rnd(len(pool))]
        chosen.append(pick)
        used.add(pick)

    return ControlSet(
        name=f"layer_type_matched_k{len(chosen)}_seed{seed}",
        component_ids=tuple(chosen),
        kind="layer_type_matched",
        matched_to=tuple(target_ids),
        seed=seed,
        used_type_only_fallback=used_fallback,
        notes=(
            "Structural layer/type match only. Falls back to type-only when the same "
            "layer has no unused component. NOT selected by benign KL."
        ),
    )


def generate_layer_type_matched_candidates(
    catalog: Sequence[ComponentRef],
    target_ids: Sequence[str],
    *,
    n_candidates: int = DAMAGE_MATCH_N_CANDIDATES,
    seed: int,
) -> list[ControlSet]:
    """Candidate control sets for subsequent damage matching (validation KL only)."""
    out: list[ControlSet] = []
    for i in range(n_candidates):
        cs = layer_type_matched_control(catalog, target_ids, seed=seed + 17 * i + 1)
        out.append(
            ControlSet(
                name=f"damage_candidate_{i}_seed{seed + 17 * i + 1}",
                component_ids=cs.component_ids,
                kind="damage_candidate",
                matched_to=tuple(target_ids),
                seed=cs.seed,
                used_type_only_fallback=cs.used_type_only_fallback,
                notes="Layer/type-matched candidate awaiting validation benign-KL selection.",
            )
        )
    return out


def select_benign_damage_matched_control(
    *,
    target_ids: Sequence[str],
    target_benign_kl: float,
    candidates: Sequence[ControlSet],
    candidate_benign_kl: MappingLike,
    split_used: str = "validation",
    # Deprecated kwargs retained for call-site compatibility; ignored by v2.1 bound.
    abs_tol: float | None = None,
    rel_tol: float | None = None,
) -> ControlSet:
    """Choose a control whose K is within the v2.1 equivalence bound of target K.

    Bound: abs(K_c - K_t) <= min(0.005, 0.25 * max(K_t, 0.002))

    Never uses held-out. Never uses triggered-backdoor suppression (D) to select.
    If none match, returns kind='unmatched' rather than claiming damage-matched evidence.
    Records actual |ΔK| and the permitted bound for every successful match.
    """
    del abs_tol, rel_tol  # v2.1: fixed conservative bound; do not use OR-of-tolerances
    if split_used == "held_out":
        raise RuntimeError("Damage matching must not use held_out")
    if split_used not in {"validation", "discovery"}:
        raise RuntimeError(f"Unexpected split for damage matching: {split_used}")

    permitted = damage_match_permitted_bound(float(target_benign_kl))
    best: ControlSet | None = None
    best_err = float("inf")
    best_k: float | None = None
    for cand in candidates:
        if cand.name in candidate_benign_kl:
            k_c = float(candidate_benign_kl[cand.name])
        elif tuple(cand.component_ids) in candidate_benign_kl:
            k_c = float(candidate_benign_kl[tuple(cand.component_ids)])
        else:
            raise KeyError(
                f"Missing benign KL for candidate {cand.name}; "
                "damage matching requires validation K for every candidate"
            )
        err_abs = abs(k_c - target_benign_kl)
        if err_abs <= permitted and err_abs < best_err:
            best_err = err_abs
            best = cand
            best_k = k_c

    if best is None:
        return ControlSet(
            name="unmatched_damage_control",
            component_ids=tuple(),
            kind="unmatched",
            matched_to=tuple(target_ids),
            target_benign_kl=float(target_benign_kl),
            control_benign_kl=None,
            match_tolerance_abs=DAMAGE_MATCH_ABS_CAP,
            match_tolerance_rel=DAMAGE_MATCH_REL_TOLERANCE,
            match_abs_diff=None,
            match_permitted_bound=permitted,
            unmatched_reason=(
                f"No candidate with abs(K_c-K_t) <= permitted_bound={permitted:.6g} "
                f"(K_t={target_benign_kl})"
            ),
            notes=(
                "No acceptable damage-matched control under v2.1 equivalence bound. "
                "Do not claim damage-matched evidence for this pair/budget."
            ),
        )

    return ControlSet(
        name=f"benign_damage_matched_k{len(best.component_ids)}",
        component_ids=best.component_ids,
        kind="benign_damage_matched",
        matched_to=tuple(target_ids),
        seed=best.seed,
        target_benign_kl=float(target_benign_kl),
        control_benign_kl=float(best_k) if best_k is not None else None,
        match_tolerance_abs=DAMAGE_MATCH_ABS_CAP,
        match_tolerance_rel=DAMAGE_MATCH_REL_TOLERANCE,
        match_abs_diff=float(best_err),
        match_permitted_bound=permitted,
        used_type_only_fallback=best.used_type_only_fallback,
        notes=(
            "Selected by validation benign KL proximity among layer/type-matched "
            "candidates using v2.1 bound "
            "abs(ΔK)<=min(0.005, 0.25*max(K_t, 0.002)). "
            "Not selected by backdoor suppression D. "
            f"Fallback type-only used in candidate construction: {best.used_type_only_fallback}."
        ),
    )


MappingLike = dict[Any, float]
