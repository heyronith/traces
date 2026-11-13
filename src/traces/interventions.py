"""Temporary activation/output ablations via managed forward hooks.

IMPORTANT
---------
These are **activation ablations**, not weight-reversion interventions.
Hooks mask activations at specific sites; model parameters are never modified
or saved in an altered state.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Generator, Iterable, Literal, Sequence

import torch
import torch.nn as nn

InterventionKind = Literal["attention_head", "mlp_channel_group", "noop"]


@dataclass(frozen=True)
class InterventionSpec:
    """Exact record of an activation-ablation intervention."""

    kind: InterventionKind
    layer: int | None = None
    component: str | None = None
    # Attention: query-head index at o_proj input (GQA-aware layout).
    head_index: int | None = None
    head_dim: int | None = None
    # MLP: channel group at down_proj input after gated intermediate.
    group_index: int | None = None
    group_size: int | None = None
    channel_start: int | None = None
    channel_end: int | None = None
    # Token positions: None = all positions in the current forward.
    token_positions: tuple[int, ...] | None = None
    semantics: str = (
        "activation_ablation: zero selected activation slice before linear; "
        "NOT weight reversion"
    )
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HookHandleSet:
    handles: list[Any] = field(default_factory=list)
    specs: list[InterventionSpec] = field(default_factory=list)

    def remove_all(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __len__(self) -> int:
        return len(self.handles)


def _resolve_layers(model: nn.Module) -> nn.ModuleList:
    base = model
    # PeftModel → base_model.model.model.layers (Gemma)
    if hasattr(base, "base_model"):
        base = base.base_model
    if hasattr(base, "model"):
        base = base.model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    if hasattr(base, "layers"):
        return base.layers
    raise AttributeError("Could not locate transformer layers on model")


def get_gemma_attention_layout(config: Any) -> dict[str, int]:
    """Verify Gemma-2 GQA layout for o_proj input slicing.

    o_proj receives concatenated *query* head outputs:
      shape [..., num_attention_heads * head_dim]
    Key/value head counts must not be treated as interchangeable with query heads.
    """
    n_q = int(config.num_attention_heads)
    n_kv = int(getattr(config, "num_key_value_heads", n_q))
    hidden = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden // n_q))
    if n_q % n_kv != 0:
        raise ValueError(f"Unexpected GQA layout: n_q={n_q}, n_kv={n_kv}")
    if head_dim * n_q != hidden:
        # Gemma-2 stores head_dim explicitly; still allow if product matches o_proj in.
        pass
    return {
        "num_attention_heads": n_q,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "o_proj_in_features": n_q * head_dim,
        "gqa_group_size": n_q // n_kv,
    }


def attention_head_slice(head_index: int, head_dim: int, num_heads: int) -> slice:
    if not 0 <= head_index < num_heads:
        raise IndexError(f"head_index {head_index} out of range [0, {num_heads})")
    start = head_index * head_dim
    end = start + head_dim
    return slice(start, end)


def mlp_channel_group_slice(
    group_index: int, group_size: int, intermediate_size: int
) -> slice:
    n_groups = intermediate_size // group_size
    if intermediate_size % group_size != 0:
        raise ValueError(
            f"intermediate_size {intermediate_size} not divisible by group_size {group_size}"
        )
    if not 0 <= group_index < n_groups:
        raise IndexError(f"group_index {group_index} out of range [0, {n_groups})")
    start = group_index * group_size
    end = start + group_size
    return slice(start, end)


def _maybe_mask_positions(
    x: torch.Tensor, token_positions: tuple[int, ...] | None
) -> torch.Tensor:
    """Return a boolean mask over sequence positions, shape broadcastable to x."""
    if token_positions is None:
        return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
    mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
    for t in token_positions:
        if t < 0:
            t = x.shape[-2] + t
        if 0 <= t < x.shape[-2]:
            mask[..., t] = True
    return mask


def make_input_zero_hook(
    channel_slice: slice,
    token_positions: tuple[int, ...] | None = None,
) -> Any:
    """Forward pre-hook: zero a channel slice of the module input."""

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if not inputs:
            return inputs
        x = inputs[0]
        if not torch.is_tensor(x):
            return inputs
        if x.shape[-1] <= channel_slice.start:
            raise RuntimeError(
                f"Input last dim {x.shape[-1]} incompatible with slice {channel_slice}"
            )
        y = x.clone()
        pos_mask = _maybe_mask_positions(y, token_positions)
        # Apply zeroing only on selected positions.
        # y[..., channel_slice][pos_mask] — need careful indexing.
        if token_positions is None:
            y[..., channel_slice] = 0
        else:
            # Expand positional mask to channel slice.
            selected = y[..., channel_slice]
            # pos_mask shape matches selected without last dim
            selected = selected.clone()
            selected[pos_mask] = 0
            y[..., channel_slice] = selected
        # Preserve dtype/shape
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        return (y,) + inputs[1:]

    return hook


def make_noop_hook() -> Any:
    """Forward pre-hook that returns inputs unchanged."""

    def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        return inputs

    return hook


def apply_activation_zero(
    tensor: torch.Tensor,
    channel_slice: slice,
    token_positions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Pure function used by unit tests on synthetic tensors."""
    y = tensor.clone()
    if token_positions is None:
        y[..., channel_slice] = 0
        return y
    pos_mask = _maybe_mask_positions(y, token_positions)
    selected = y[..., channel_slice].clone()
    selected[pos_mask] = 0
    y[..., channel_slice] = selected
    return y


class InterventionManager:
    """Install / remove temporary activation-ablation hooks safely."""

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

    def install_attention_head(
        self,
        layer: int,
        head_index: int,
        token_positions: Sequence[int] | None = None,
    ) -> InterventionSpec:
        layout = get_gemma_attention_layout(self.config)
        n_q = layout["num_attention_heads"]
        head_dim = layout["head_dim"]
        sl = attention_head_slice(head_index, head_dim, n_q)
        module = self._get_module(layer, "self_attn.o_proj")
        in_features = int(module.in_features)
        expected = layout["o_proj_in_features"]
        if in_features != expected:
            raise RuntimeError(
                f"o_proj.in_features={in_features} != num_heads*head_dim={expected}. "
                "Refusing to assume KV/query head interchangeability."
            )
        positions = tuple(token_positions) if token_positions is not None else None
        handle = module.register_forward_pre_hook(make_input_zero_hook(sl, positions))
        spec = InterventionSpec(
            kind="attention_head",
            layer=layer,
            component="self_attn.o_proj.input",
            head_index=head_index,
            head_dim=head_dim,
            channel_start=sl.start,
            channel_end=sl.stop,
            token_positions=positions,
            notes=(
                f"GQA layout n_q={n_q}, n_kv={layout['num_key_value_heads']}; "
                f"masking query-head {head_index} slice before o_proj"
            ),
        )
        self._active.handles.append(handle)
        self._active.specs.append(spec)
        return spec

    def install_mlp_channel_group(
        self,
        layer: int,
        group_index: int,
        group_size: int = 256,
        token_positions: Sequence[int] | None = None,
    ) -> InterventionSpec:
        intermediate = int(self.config.intermediate_size)
        sl = mlp_channel_group_slice(group_index, group_size, intermediate)
        module = self._get_module(layer, "mlp.down_proj")
        if int(module.in_features) != intermediate:
            raise RuntimeError(
                f"down_proj.in_features={module.in_features} != intermediate_size={intermediate}"
            )
        positions = tuple(token_positions) if token_positions is not None else None
        handle = module.register_forward_pre_hook(make_input_zero_hook(sl, positions))
        spec = InterventionSpec(
            kind="mlp_channel_group",
            layer=layer,
            component="mlp.down_proj.input",
            group_index=group_index,
            group_size=group_size,
            channel_start=sl.start,
            channel_end=sl.stop,
            token_positions=positions,
            notes=(
                "Masking gated-MLP intermediate channel group at input to down_proj "
                "(after gate*up nonlinearity)"
            ),
        )
        self._active.handles.append(handle)
        self._active.specs.append(spec)
        return spec

    def install_noop(self, layer: int = 0, site: str = "self_attn.o_proj") -> InterventionSpec:
        module = self._get_module(layer, site)
        handle = module.register_forward_pre_hook(make_noop_hook())
        spec = InterventionSpec(
            kind="noop",
            layer=layer,
            component=f"{site}.input",
            notes="No-op hook: inputs returned unchanged",
        )
        self._active.handles.append(handle)
        self._active.specs.append(spec)
        return spec

    @contextmanager
    def temporary(
        self,
        specs: Iterable[dict[str, Any]] | None = None,
        *,
        attention: dict[str, Any] | None = None,
        mlp: dict[str, Any] | None = None,
        noop: bool = False,
    ) -> Generator[list[InterventionSpec], None, None]:
        """Context manager: install hooks, yield specs, always clean up."""
        installed: list[InterventionSpec] = []
        try:
            if noop:
                installed.append(self.install_noop())
            if attention:
                installed.append(
                    self.install_attention_head(
                        layer=int(attention["layer"]),
                        head_index=int(attention["head"]),
                        token_positions=attention.get("token_positions"),
                    )
                )
            if mlp:
                installed.append(
                    self.install_mlp_channel_group(
                        layer=int(mlp["layer"]),
                        group_index=int(mlp["group_index"]),
                        group_size=int(mlp.get("group_size", 256)),
                        token_positions=mlp.get("token_positions"),
                    )
                )
            if specs:
                for s in specs:
                    kind = s["kind"]
                    if kind == "attention_head":
                        installed.append(
                            self.install_attention_head(
                                s["layer"], s["head_index"], s.get("token_positions")
                            )
                        )
                    elif kind == "mlp_channel_group":
                        installed.append(
                            self.install_mlp_channel_group(
                                s["layer"],
                                s["group_index"],
                                s.get("group_size", 256),
                                s.get("token_positions"),
                            )
                        )
                    elif kind == "noop":
                        installed.append(self.install_noop(s.get("layer", 0)))
                    else:
                        raise ValueError(f"Unknown intervention kind: {kind}")
            yield installed
        finally:
            self.clear()
            assert self.active_hook_count == 0, "Hooks leaked after temporary()"
