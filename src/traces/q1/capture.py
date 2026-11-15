"""Activation capture for Q1: layer-input residuals and o_proj inputs.

Batch-size-1 only. Hooks are temporary and must be cleared after each forward.
Boundary semantics: layer-L input = residual stream entering transformer layer L
(the tensor observed by a forward pre-hook on ``model.model.layers[L]`` /
PEFT-resolved equivalent).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Iterable, Sequence

import torch
import torch.nn as nn

from traces.interventions import _resolve_layers
from traces.q1.bos import BosConvention
from traces.q1.caches import DonorActivationCache, TagName
from traces.q1.tag_position import TagAlignment


@dataclass
class CaptureBuffers:
    residuals_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    o_proj_input_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    logits: torch.Tensor | None = None

    def clear(self) -> None:
        self.residuals_by_layer.clear()
        self.o_proj_input_by_layer.clear()
        self.logits = None


class ActivationCaptureManager:
    """Register capture hooks; always remove on exit."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: list[Any] = []
        self.buffers = CaptureBuffers()

    @property
    def active_hook_count(self) -> int:
        return len(self._handles)

    def clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.buffers.clear()

    def _layer(self, layer: int) -> nn.Module:
        layers = _resolve_layers(self.model)
        if not 0 <= layer < len(layers):
            raise IndexError(f"layer {layer} out of range [0, {len(layers)})")
        return layers[layer]

    def install_residual_input_capture(self, layers: Iterable[int]) -> None:
        for layer in layers:

            def make_hook(L: int):
                def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"Layer {L} residual input missing")
                    x = inputs[0]
                    if x.ndim < 2 or x.shape[0] != 1:
                        raise RuntimeError(
                            f"Q1 capture requires batch_size=1; got {tuple(x.shape)}"
                        )
                    self.buffers.residuals_by_layer[L] = x.detach().clone()

                return hook

            h = self._layer(int(layer)).register_forward_pre_hook(make_hook(int(layer)))
            self._handles.append(h)

    def install_o_proj_input_capture(self, layers: Iterable[int]) -> None:
        for layer in layers:
            module = self._layer(int(layer)).self_attn.o_proj

            def make_hook(L: int):
                def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise RuntimeError(f"o_proj input missing at layer {L}")
                    x = inputs[0]
                    if x.shape[0] != 1:
                        raise RuntimeError(
                            f"Q1 capture requires batch_size=1; got {tuple(x.shape)}"
                        )
                    self.buffers.o_proj_input_by_layer[L] = x.detach().clone()

                return hook

            self._handles.append(module.register_forward_pre_hook(make_hook(int(layer))))

    @contextmanager
    def capturing(
        self,
        *,
        residual_layers: Sequence[int],
        o_proj_layers: Sequence[int],
    ) -> Generator[CaptureBuffers, None, None]:
        self.clear()
        try:
            self.install_residual_input_capture(residual_layers)
            self.install_o_proj_input_capture(o_proj_layers)
            yield self.buffers
        finally:
            # Remove hooks but keep buffers for the caller to read/copy.
            for h in self._handles:
                h.remove()
            self._handles.clear()


def populate_donor_cache_from_buffers(
    cache: DonorActivationCache,
    tag: TagName,
    buffers: CaptureBuffers,
) -> None:
    for layer, tensor in buffers.residuals_by_layer.items():
        cache.store_residual(tag, layer, tensor)
    for layer, tensor in buffers.o_proj_input_by_layer.items():
        cache.store_o_proj_input(tag, layer, tensor)
    if buffers.logits is not None:
        cache.logits[tag] = buffers.logits.detach().clone()


def assert_cache_batch1(tensor: torch.Tensor, *, context: str) -> None:
    if tensor.shape[0] != 1:
        raise RuntimeError(f"{context}: batch_size must be 1, got {tuple(tensor.shape)}")


def assert_matching_provenance(
    *,
    donor_cache: DonorActivationCache,
    recipient_tag: TagName,
    bos_convention: BosConvention,
    model_unit: str,
    instruction_id: str,
    expected_alignment: TagAlignment,
) -> None:
    if donor_cache.bos_convention != bos_convention:
        raise RuntimeError(
            f"BOS convention mismatch: donor={donor_cache.bos_convention} "
            f"requested={bos_convention}"
        )
    if donor_cache.model_unit != model_unit:
        raise RuntimeError(
            f"Checkpoint/model_unit mismatch: donor={donor_cache.model_unit} "
            f"requested={model_unit}"
        )
    if donor_cache.instruction_id != instruction_id:
        raise RuntimeError(
            f"Instruction mismatch: donor={donor_cache.instruction_id} "
            f"requested={instruction_id}"
        )
    if donor_cache.alignment != expected_alignment:
        raise RuntimeError("TagAlignment provenance mismatch between donor and recipient")
