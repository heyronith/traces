"""Inject hybrid residuals at transformer layer inputs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn as nn

from traces.interventions import HookHandleSet, _resolve_layers
from traces.q1.hybrids import HybridName, build_hybrid_residual
from traces.q1.tag_position import TagAlignment


def make_residual_replace_hook(replacement: torch.Tensor) -> Any:
    """Forward pre-hook: replace the layer-input residual with ``replacement``."""

    def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError("Layer input residual missing for injection")
        x = inputs[0]
        if x.shape != replacement.shape:
            raise RuntimeError(
                f"Residual inject shape mismatch live={tuple(x.shape)} "
                f"replacement={tuple(replacement.shape)}"
            )
        if x.dtype != replacement.dtype or x.device != replacement.device:
            rep = replacement.to(device=x.device, dtype=x.dtype)
        else:
            rep = replacement
        return (rep,) + inputs[1:]

    return hook


class ResidualInjectionManager:
    def __init__(self, model: nn.Module):
        self.model = model
        self._active = HookHandleSet()

    @property
    def active_hook_count(self) -> int:
        return len(self._active)

    def clear(self) -> None:
        self._active.remove_all()
        self._active.specs.clear()

    def install_layer_input_replacement(self, layer: int, residual: torch.Tensor) -> None:
        layers = _resolve_layers(self.model)
        if not 0 <= layer < len(layers):
            raise IndexError(f"layer {layer} out of range")
        if residual.shape[0] != 1:
            raise RuntimeError("Q1 residual injection requires batch_size=1")
        handle = layers[layer].register_forward_pre_hook(
            make_residual_replace_hook(residual.detach())
        )
        self._active.handles.append(handle)

    @contextmanager
    def injecting(
        self, layer: int, residual: torch.Tensor
    ) -> Generator[None, None, None]:
        self.clear()
        try:
            self.install_layer_input_replacement(layer, residual)
            yield
        finally:
            self.clear()


def build_boundary_hybrid_for_injection(
    residual_T: torch.Tensor,
    residual_C: torch.Tensor,
    alignment: TagAlignment,
    hybrid: HybridName,
) -> torch.Tensor:
    """Construct CT/TC/TT/CC residual for layer-input injection."""
    return build_hybrid_residual(residual_T, residual_C, alignment, hybrid)
