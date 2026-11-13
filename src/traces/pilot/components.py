"""Component catalog for activation-ablation groups on l1523 layers 15–23.

Full scan size: 396 groups
  - attention: 9 layers × 8 query heads = 72
  - mlp: 9 layers × (9216 / 256) = 9 × 36 = 324
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from traces.pilot import (
    ADAPTED_LAYERS,
    HEAD_DIM_GEMMA2_2B,
    INTERMEDIATE_SIZE_GEMMA2_2B,
    MLP_GROUP_SIZE,
    N_COMPONENT_GROUPS_FULL_SCAN,
    N_QUERY_HEADS_GEMMA2_2B,
)

ComponentType = Literal["attention_head", "mlp_channel_group"]


@dataclass(frozen=True)
class ComponentRef:
    component_id: str
    kind: ComponentType
    layer: int
    # attention
    head_index: int | None = None
    head_dim: int | None = None
    # mlp
    group_index: int | None = None
    group_size: int | None = None
    channel_start: int | None = None
    channel_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_intervention_kwargs(self) -> dict[str, Any]:
        if self.kind == "attention_head":
            return {"attention": {"layer": self.layer, "head": self.head_index}}
        return {
            "mlp": {
                "layer": self.layer,
                "group_index": self.group_index,
                "group_size": self.group_size or MLP_GROUP_SIZE,
            }
        }


def build_full_l1523_catalog(
    *,
    layers: tuple[int, ...] = ADAPTED_LAYERS,
    n_heads: int = N_QUERY_HEADS_GEMMA2_2B,
    head_dim: int = HEAD_DIM_GEMMA2_2B,
    intermediate_size: int = INTERMEDIATE_SIZE_GEMMA2_2B,
    group_size: int = MLP_GROUP_SIZE,
) -> list[ComponentRef]:
    """Pre-registered full scan over adapted layers 15–23 (396 groups)."""
    if intermediate_size % group_size != 0:
        raise ValueError("intermediate_size must be divisible by group_size")
    n_groups = intermediate_size // group_size
    out: list[ComponentRef] = []
    for layer in layers:
        for h in range(n_heads):
            out.append(
                ComponentRef(
                    component_id=f"L{layer}.attn.head{h}",
                    kind="attention_head",
                    layer=layer,
                    head_index=h,
                    head_dim=head_dim,
                    channel_start=h * head_dim,
                    channel_end=(h + 1) * head_dim,
                )
            )
        for g in range(n_groups):
            out.append(
                ComponentRef(
                    component_id=f"L{layer}.mlp.g{g}",
                    kind="mlp_channel_group",
                    layer=layer,
                    group_index=g,
                    group_size=group_size,
                    channel_start=g * group_size,
                    channel_end=(g + 1) * group_size,
                )
            )
    if len(out) != N_COMPONENT_GROUPS_FULL_SCAN and layers == ADAPTED_LAYERS:
        raise RuntimeError(
            f"Expected {N_COMPONENT_GROUPS_FULL_SCAN} components, got {len(out)}"
        )
    return out


def catalog_summary(catalog: list[ComponentRef] | None = None) -> dict[str, Any]:
    cat = catalog or build_full_l1523_catalog()
    n_attn = sum(1 for c in cat if c.kind == "attention_head")
    n_mlp = sum(1 for c in cat if c.kind == "mlp_channel_group")
    return {
        "n_components": len(cat),
        "n_attention_heads": n_attn,
        "n_mlp_groups": n_mlp,
        "layers": sorted({c.layer for c in cat}),
        "scan_policy": "full_adapted_band_l1523",
        "justification": (
            "Pilot scans all 396 activation-ablation groups on layers 15–23 to match "
            "the dense-LoRA l1523 family adapted band with consistent absolute indexing "
            "across seeds. Candidate selection must not use held_out prompts."
        ),
        "note": (
            "Component index overlap across seeds is not proof of identical mechanisms."
        ),
    }
