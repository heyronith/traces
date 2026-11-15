"""Enumerate Required Recommended-tier intervention conditions (no execution)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

from traces.q1.patching import CORE_TRIPLE, NOMINATED_PAIR, HeadRef

Direction = Literal["forward", "reverse"]
N_LAYERS = 26  # Gemma-2-2B layers 0..25
N_HEADS = 8
UPSTREAM_LAYERS = tuple(range(0, 19))  # 0..18 → 152 heads
DOWNSTREAM_LAYERS = tuple(range(19, 26))  # 19..25 → 56 heads


@dataclass(frozen=True)
class Condition:
    kind: str
    direction: Direction | None
    heads: tuple[HeadRef, ...] = ()
    layer: int | None = None
    hybrid: str | None = None
    k: int | None = None
    bos_convention: str = "single"
    notes: str = ""

    @property
    def condition_id(self) -> str:
        parts = [self.kind]
        if self.direction:
            parts.append(self.direction)
        if self.hybrid:
            parts.append(self.hybrid)
        if self.layer is not None:
            parts.append(f"L{self.layer}")
        if self.k is not None:
            parts.append(f"k{self.k}")
        if self.heads:
            parts.append("+".join(h.component_id for h in self.heads))
        if self.bos_convention != "single":
            parts.append(self.bos_convention)
        return "|".join(parts)


def iter_upstream_heads() -> Iterator[HeadRef]:
    for layer in UPSTREAM_LAYERS:
        for h in range(N_HEADS):
            yield HeadRef(layer, h)


def iter_downstream_heads() -> Iterator[HeadRef]:
    for layer in DOWNSTREAM_LAYERS:
        for h in range(N_HEADS):
            yield HeadRef(layer, h)


def iter_layer15_16_pairs() -> Iterator[tuple[HeadRef, HeadRef]]:
    for h15 in range(N_HEADS):
        for h16 in range(N_HEADS):
            yield HeadRef(15, h15), HeadRef(16, h16)


def enumerate_primary_l19_conditions(
    *,
    include_downstream56: bool = True,
) -> list[Condition]:
    """Per-prompt condition catalog for primary confirmatory l19 (both dirs)."""
    out: list[Condition] = []
    # Boundary hybrids needing forwards: CT, TC (TT/CC free from intact)
    for name in ("CT", "TC"):
        out.append(Condition(kind="boundary_hybrid", direction=None, hybrid=name))
    for direction in ("forward", "reverse"):
        for head in iter_upstream_heads():
            out.append(
                Condition(kind="single_head", direction=direction, heads=(head,))
            )
        for h15, h16 in iter_layer15_16_pairs():
            out.append(
                Condition(kind="pair", direction=direction, heads=(h15, h16))
            )
        out.append(
            Condition(kind="joint_triple", direction=direction, heads=CORE_TRIPLE)
        )
        out.append(
            Condition(
                kind="all_upstream",
                direction=direction,
                heads=tuple(iter_upstream_heads()),
            )
        )
        for k in range(1, 11):
            out.append(
                Condition(
                    kind="crossfit_topk",
                    direction=direction,
                    k=k,
                    notes="selection fixed from opposite fold; one eval fold per prompt",
                )
            )
        for layer in UPSTREAM_LAYERS:
            heads = tuple(HeadRef(layer, h) for h in range(N_HEADS))
            out.append(
                Condition(
                    kind="layer_joint_8head",
                    direction=direction,
                    layer=layer,
                    heads=heads,
                )
            )
        if include_downstream56:
            for head in iter_downstream_heads():
                out.append(
                    Condition(
                        kind="downstream_head_secondary",
                        direction=direction,
                        heads=(head,),
                    )
                )
    # Residual sweep CT/TC at layer inputs 1..25
    for layer in range(1, 26):
        for name in ("CT", "TC"):
            out.append(
                Condition(
                    kind="residual_sweep_hybrid",
                    direction=None,
                    hybrid=name,
                    layer=layer,
                )
            )
    return out


def enumerate_secondary_l1523_conditions() -> list[Condition]:
    out: list[Condition] = []
    for name in ("CT", "TC"):
        out.append(Condition(kind="boundary_hybrid", direction=None, hybrid=name))
    for direction in ("forward", "reverse"):
        for h15, h16 in iter_layer15_16_pairs():
            out.append(Condition(kind="pair", direction=direction, heads=(h15, h16)))
        for head in NOMINATED_PAIR:
            out.append(
                Condition(kind="nominated_single", direction=direction, heads=(head,))
            )
    return out


def enumerate_double_bos_descriptive_pairs() -> list[Condition]:
    """Exploratory-only: 64-pair forward under double BOS."""
    return [
        Condition(
            kind="pair",
            direction="forward",
            heads=(h15, h16),
            bos_convention="double",
            notes="descriptive F6 exception; own double-BOS b_i",
        )
        for h15, h16 in iter_layer15_16_pairs()
    ]


def count_primary_tf_per_prompt(include_downstream56: bool = True) -> dict[str, int]:
    """Match readiness enumerator bookkeeping (intact T/C separate)."""
    conds = enumerate_primary_l19_conditions(
        include_downstream56=include_downstream56
    )
    return {
        "patched_or_hybrid_conditions": len(conds),
        "intact_tf_T_C": 2,
        "tf_total": len(conds) + 2,
        "gen_total": 2,
    }
