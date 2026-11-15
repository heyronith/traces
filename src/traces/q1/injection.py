"""Inject hybrid residuals at transformer layer inputs (fail-closed tensors)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn as nn

from traces.interventions import HookHandleSet, _resolve_layers
from traces.q1.hybrids import HybridName, build_hybrid_residual
from traces.q1.tag_position import TagAlignment


def _extract_hidden_states(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[torch.Tensor, str]:
    """Return (hidden_states, source) from positional or keyword Gemma-2 calls."""
    if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
        return kwargs["hidden_states"], "kwargs"
    if args and torch.is_tensor(args[0]):
        return args[0], "args"
    raise RuntimeError(
        "Layer input residual missing: need tensor args[0] or kwargs['hidden_states']"
    )


def assert_tensor_compatible(
    live: torch.Tensor, replacement: torch.Tensor, *, context: str
) -> None:
    if live.shape != replacement.shape:
        raise RuntimeError(
            f"{context}: shape mismatch live={tuple(live.shape)} "
            f"replacement={tuple(replacement.shape)}"
        )
    if live.dtype != replacement.dtype:
        raise RuntimeError(
            f"{context}: dtype mismatch live={live.dtype} replacement={replacement.dtype} "
            "(silent casting forbidden)"
        )
    if live.device != replacement.device:
        raise RuntimeError(
            f"{context}: device mismatch live={live.device} replacement={replacement.device} "
            "(silent casting forbidden)"
        )


def make_residual_replace_hook(replacement: torch.Tensor) -> Any:
    """Forward pre-hook with kwargs: replace hidden_states; no silent cast."""

    def hook(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        x, source = _extract_hidden_states(args, kwargs)
        assert_tensor_compatible(x, replacement, context="residual_inject")
        if source == "kwargs":
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = replacement
            return args, new_kwargs
        return (replacement,) + args[1:], kwargs

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
            make_residual_replace_hook(residual.detach()), with_kwargs=True
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
