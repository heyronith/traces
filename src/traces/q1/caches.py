"""Donor activation caches with isolation and provenance.

Caches store intact T/C residuals and o_proj inputs. Mixing single- and
double-BOS caches is forbidden (F6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from traces.q1.bos import BosConvention
from traces.q1.tag_position import TagAlignment

TagName = Literal["T", "C"]


@dataclass
class DonorActivationCache:
    """Per-instruction donor cache for one BOS convention and one model unit."""

    instruction_id: str
    bos_convention: BosConvention
    alignment: TagAlignment
    model_unit: str
    residuals_by_layer: dict[TagName, dict[int, torch.Tensor]] = field(
        default_factory=lambda: {"T": {}, "C": {}}
    )
    o_proj_input_by_layer: dict[TagName, dict[int, torch.Tensor]] = field(
        default_factory=lambda: {"T": {}, "C": {}}
    )
    logits: dict[TagName, torch.Tensor | None] = field(
        default_factory=lambda: {"T": None, "C": None}
    )
    margins: dict[TagName, float | None] = field(
        default_factory=lambda: {"T": None, "C": None}
    )
    meta: dict[str, Any] = field(default_factory=dict)

    def store_residual(self, tag: TagName, layer: int, tensor: torch.Tensor) -> None:
        self.residuals_by_layer[tag][int(layer)] = tensor.detach().clone()

    def store_o_proj_input(self, tag: TagName, layer: int, tensor: torch.Tensor) -> None:
        self.o_proj_input_by_layer[tag][int(layer)] = tensor.detach().clone()

    def get_residual(self, tag: TagName, layer: int) -> torch.Tensor:
        try:
            return self.residuals_by_layer[tag][int(layer)]
        except KeyError as e:
            raise KeyError(f"Missing residual tag={tag} layer={layer}") from e

    def get_o_proj_input(self, tag: TagName, layer: int) -> torch.Tensor:
        try:
            return self.o_proj_input_by_layer[tag][int(layer)]
        except KeyError as e:
            raise KeyError(f"Missing o_proj input tag={tag} layer={layer}") from e

    def clear(self) -> None:
        self.residuals_by_layer = {"T": {}, "C": {}}
        self.o_proj_input_by_layer = {"T": {}, "C": {}}
        self.logits = {"T": None, "C": None}
        self.margins = {"T": None, "C": None}


class CacheRegistry:
    """Isolate caches by (model_unit, bos_convention, instruction_id)."""

    def __init__(self) -> None:
        self._stores: dict[tuple[str, BosConvention, str], DonorActivationCache] = {}

    def put(self, cache: DonorActivationCache) -> None:
        key = (cache.model_unit, cache.bos_convention, cache.instruction_id)
        self._stores[key] = cache

    def get(
        self,
        model_unit: str,
        bos_convention: BosConvention,
        instruction_id: str,
    ) -> DonorActivationCache:
        key = (model_unit, bos_convention, instruction_id)
        if key not in self._stores:
            raise KeyError(f"No cache for {key}")
        return self._stores[key]

    def require_convention(
        self, cache: DonorActivationCache, bos_convention: BosConvention
    ) -> None:
        if cache.bos_convention != bos_convention:
            raise RuntimeError(
                f"F6 isolation: refuse to use {cache.bos_convention} cache under "
                f"{bos_convention} request"
            )

    def clear(self) -> None:
        self._stores.clear()

    def __len__(self) -> int:
        return len(self._stores)
