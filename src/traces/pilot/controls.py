"""Control intervention-set generators (random + layer/type-matched)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from traces.pilot.components import ComponentRef


@dataclass(frozen=True)
class ControlSet:
    name: str
    component_ids: list[str]
    kind: str
    matched_to: list[str] | None = None
    seed: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component_ids": list(self.component_ids),
            "kind": self.kind,
            "matched_to": self.matched_to,
            "seed": self.seed,
            "notes": self.notes,
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
        component_ids=shuffled[:k],
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
    """Sample a control set with the same multiset of (layer, kind) as the target.

    For each target component, sample a different component with the same layer
    and kind when possible; otherwise same kind any layer.
    """
    by_id = {c.component_id: c for c in catalog}
    targets = [by_id[i] for i in target_ids]
    used = set(target_ids)
    chosen: list[str] = []
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
        pool = primary or fallback
        if not pool:
            raise RuntimeError(f"No matched control available for {t.component_id}")
        pick = pool[rnd(len(pool))]
        chosen.append(pick)
        used.add(pick)

    return ControlSet(
        name=f"layer_type_matched_k{len(chosen)}_seed{seed}",
        component_ids=chosen,
        kind="layer_type_matched",
        matched_to=list(target_ids),
        seed=seed,
        notes=(
            "Matched on (layer, kind) when possible; falls back to kind-only. "
            "Used for damage-matched transfer comparisons."
        ),
    )
