"""Deterministic synthetic scorers for CPU end-to-end pilot tests.

Two planted worlds:
- shared_causal: all seeds share the same top causal components
- disjoint_causal: each seed has a distinct causal set

No GPU / real model weights involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from traces.pilot.components import ComponentRef, build_full_l1523_catalog


def _stable_unit(s: str) -> float:
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 10_007
    return (h % 1000) / 1000.0


@dataclass
class SyntheticWorld:
    name: str
    causal_by_seed: dict[int, tuple[str, ...]]
    catalog: list[ComponentRef]

    def causal_set(self, seed: int) -> set[str]:
        return set(self.causal_by_seed[seed])


def make_shared_causal_world(k_plant: int = 10) -> SyntheticWorld:
    cat = build_full_l1523_catalog()
    # Prefer attention heads early in catalog for a stable planted set
    planted = tuple(c.component_id for c in cat if c.kind == "attention_head")[:k_plant]
    return SyntheticWorld(
        name="shared_causal",
        causal_by_seed={42: planted, 43: planted, 44: planted},
        catalog=cat,
    )


def make_disjoint_causal_world(k_plant: int = 10) -> SyntheticWorld:
    cat = build_full_l1523_catalog()
    heads = [c.component_id for c in cat if c.kind == "attention_head"]
    # Non-overlapping slices
    return SyntheticWorld(
        name="disjoint_causal",
        causal_by_seed={
            42: tuple(heads[0:k_plant]),
            43: tuple(heads[k_plant : 2 * k_plant]),
            44: tuple(heads[2 * k_plant : 3 * k_plant]),
        },
        catalog=cat,
    )


class SyntheticPilotScorer:
    """Stub implementing PilotScorer protocol with planted causal structure."""

    def __init__(self, world: SyntheticWorld, noise: float = 0.02):
        self.world = world
        self.noise = noise

    def _overlap_strength(self, seed: int, component_ids: Sequence[str]) -> float:
        causal = self.world.causal_set(seed)
        if not component_ids:
            return 0.0
        hit = sum(1 for c in component_ids if c in causal)
        return hit / max(len(causal), 1)

    def score_D_single(
        self, seed: int, component_id: str, instruction_id: str, split: str
    ) -> float:
        base = 0.8 if component_id in self.world.causal_set(seed) else 0.02
        jitter = self.noise * (_stable_unit(f"{seed}:{component_id}:{instruction_id}:{split}") - 0.5)
        return max(0.0, base + jitter)

    def score_D_set(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> float:
        # Joint ablation: strength scales with overlap with planted causal set
        strength = self._overlap_strength(seed, component_ids)
        # Mild instruction averaging noise
        noise = self.noise * (
            sum(_stable_unit(f"{seed}:{i}:{split}") for i in instruction_ids)
            / max(len(instruction_ids), 1)
            - 0.5
        )
        return max(0.0, 1.0 * strength + 0.05 * len(component_ids) / 100.0 + noise)

    def score_K_set(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> float:
        # Benign damage increases with set size + mild seed jitter (independent of causal overlap)
        if not component_ids:
            return 0.0
        base = 0.01 * len(component_ids)
        # Layer diversity adds damage
        layers = {cid.split(".")[0] for cid in component_ids}
        base += 0.005 * len(layers)
        base += 0.002 * _stable_unit(f"K:{seed}:{split}:{','.join(component_ids)}")
        return float(base)

    def score_asr(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> dict[str, float]:
        strength = self._overlap_strength(seed, component_ids)
        intact = 1.0
        greedy = max(0.0, intact - 0.9 * strength)
        full = max(0.0, intact - 0.85 * strength)
        return {"greedy_asr": greedy, "full_payload_asr": full, "intact_asr": intact}
