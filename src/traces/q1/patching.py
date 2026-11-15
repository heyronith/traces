"""Forward/reverse attention-head activation patching (donor slice replace).

Unlike Stage 1 zero-ablation, Q1 replaces the selected query-head slice at
``self_attn.o_proj`` input with the same-instruction donor run's slice at
aligned post-tag positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn

from traces.interventions import (
    HookHandleSet,
    InterventionSpec,
    _resolve_layers,
    attention_head_slice,
    get_gemma_attention_layout,
)

PatchDirection = Literal["forward", "reverse"]


@dataclass(frozen=True)
class HeadRef:
    layer: int
    head_index: int

    @property
    def component_id(self) -> str:
        return f"L{self.layer}.attn.head{self.head_index}"


# Frozen nominated pair (§3.2)
NOMINATED_PAIR: tuple[HeadRef, HeadRef] = (
    HeadRef(15, 6),
    HeadRef(16, 5),
)
# Stage-1 three-head joint (secondary / descriptive triple)
CORE_TRIPLE: tuple[HeadRef, ...] = (
    HeadRef(15, 6),
    HeadRef(16, 5),
    HeadRef(23, 0),
)


def apply_head_patch_tensor(
    recipient: torch.Tensor,
    donor: torch.Tensor,
    *,
    head_index: int,
    head_dim: int,
    num_heads: int,
    token_positions: Sequence[int],
) -> torch.Tensor:
    """Pure tensor patch used by unit tests (shapes ``[..., seq, n_q*head_dim]``)."""
    if recipient.shape != donor.shape:
        raise ValueError("recipient/donor shape mismatch")
    sl = attention_head_slice(head_index, head_dim, num_heads)
    out = recipient.clone()
    for t in token_positions:
        out[..., t, sl] = donor[..., t, sl]
    return out


def make_input_patch_hook(
    donor_activation: torch.Tensor,
    channel_slice: slice,
    token_positions: Sequence[int],
) -> Any:
    """Forward pre-hook: replace a channel slice at selected positions from donor.

    ``donor_activation`` must be shaped like the module input ``[..., seq, channels]``
    for the current sequence (batch dim optional, matching the live forward).
    """
    positions = tuple(int(t) for t in token_positions)
    donor = donor_activation

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if not inputs:
            return inputs
        x = inputs[0]
        if not torch.is_tensor(x):
            return inputs
        if donor.shape != x.shape:
            raise RuntimeError(
                f"Donor shape {tuple(donor.shape)} != live input {tuple(x.shape)}"
            )
        if donor.device != x.device or donor.dtype != x.dtype:
            d = donor.to(device=x.device, dtype=x.dtype)
        else:
            d = donor
        y = x.clone()
        for t in positions:
            y[..., t, channel_slice] = d[..., t, channel_slice]
        return (y,) + inputs[1:]

    return hook


class PatchHookManager:
    """Install / remove temporary activation-**patch** hooks (not ablations)."""

    def __init__(self, model: nn.Module, config: Any | None = None):
        self.model = model
        self.config = config or getattr(model, "config", None)
        if self.config is None and hasattr(model, "base_model"):
            self.config = getattr(model.base_model, "config", None)
        self._active = HookHandleSet()

    @property
    def active_hook_count(self) -> int:
        return len(self._active)

    def clear(self) -> None:
        self._active.remove_all()
        self._active.specs.clear()

    def _get_module(self, layer: int, path: str) -> nn.Module:
        layers = _resolve_layers(self.model)
        if not 0 <= layer < len(layers):
            raise IndexError(f"layer {layer} out of range [0, {len(layers)})")
        mod: nn.Module = layers[layer]
        for part in path.split("."):
            mod = getattr(mod, part)
        return mod

    def install_head_patch(
        self,
        layer: int,
        head_index: int,
        donor_o_proj_input: torch.Tensor,
        token_positions: Sequence[int],
        *,
        direction: PatchDirection,
        notes: str = "",
    ) -> InterventionSpec:
        layout = get_gemma_attention_layout(self.config)
        n_q = layout["num_attention_heads"]
        head_dim = layout["head_dim"]
        sl = attention_head_slice(head_index, head_dim, n_q)
        module = self._get_module(layer, "self_attn.o_proj")
        if int(module.in_features) != layout["o_proj_in_features"]:
            raise RuntimeError("o_proj in_features mismatch vs GQA layout")
        positions = tuple(int(t) for t in token_positions)
        handle = module.register_forward_pre_hook(
            make_input_patch_hook(donor_o_proj_input, sl, positions)
        )
        spec = InterventionSpec(
            kind="attention_head",
            layer=layer,
            component="self_attn.o_proj.input",
            head_index=head_index,
            head_dim=head_dim,
            channel_start=sl.start,
            channel_end=sl.stop,
            token_positions=positions,
            semantics=(
                "activation_patch: replace selected query-head slice from donor; "
                "NOT zero-ablation; NOT weight reversion"
            ),
            notes=notes or f"direction={direction}",
        )
        self._active.handles.append(handle)
        self._active.specs.append(spec)
        return spec

    def install_joint_head_patches(
        self,
        heads: Sequence[HeadRef],
        donor_by_layer: dict[int, torch.Tensor],
        token_positions: Sequence[int],
        *,
        direction: PatchDirection,
    ) -> list[InterventionSpec]:
        installed: list[InterventionSpec] = []
        seen: set[str] = set()
        for h in heads:
            cid = h.component_id
            if cid in seen:
                raise RuntimeError(f"Duplicate head in joint patch set: {cid}")
            seen.add(cid)
            if h.layer not in donor_by_layer:
                raise KeyError(f"Missing donor o_proj input for layer {h.layer}")
            installed.append(
                self.install_head_patch(
                    h.layer,
                    h.head_index,
                    donor_by_layer[h.layer],
                    token_positions,
                    direction=direction,
                )
            )
        return installed

    def __enter__(self) -> "PatchHookManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.clear()
