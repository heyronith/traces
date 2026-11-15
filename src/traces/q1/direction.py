"""Bind patch direction to donor/recipient tag identity.

Forward: clean donor → triggered recipient.
Reverse: triggered donor → clean recipient.
Patches apply only at aligned post-tag positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from traces.q1.bos import BosConvention
from traces.q1.caches import DonorActivationCache, TagName
from traces.q1.capture import assert_matching_provenance
from traces.q1.tag_position import TagAlignment

PatchDirection = Literal["forward", "reverse"]


@dataclass(frozen=True)
class DirectionBinding:
    direction: PatchDirection
    donor_tag: TagName
    recipient_tag: TagName

    @property
    def post_tag_positions(self) -> str:
        return "alignment.post_tag_indices"


def bind_direction(direction: PatchDirection) -> DirectionBinding:
    if direction == "forward":
        return DirectionBinding("forward", donor_tag="C", recipient_tag="T")
    if direction == "reverse":
        return DirectionBinding("reverse", donor_tag="T", recipient_tag="C")
    raise ValueError(f"Unknown direction {direction!r}")


def resolve_donor_recipient_caches(
    cache: DonorActivationCache,
    direction: PatchDirection,
    *,
    bos_convention: BosConvention,
    model_unit: str,
    instruction_id: str,
    alignment: TagAlignment,
    recipient_prompt_tag: TagName,
) -> tuple[TagName, TagName, tuple[int, ...]]:
    """Validate provenance and return (donor_tag, recipient_tag, post_tag_positions)."""
    binding = bind_direction(direction)
    assert_matching_provenance(
        donor_cache=cache,
        recipient_tag=binding.recipient_tag,
        bos_convention=bos_convention,
        model_unit=model_unit,
        instruction_id=instruction_id,
        expected_alignment=alignment,
    )
    if recipient_prompt_tag != binding.recipient_tag:
        raise RuntimeError(
            f"Recipient tag identity mismatch: prompt={recipient_prompt_tag} "
            f"direction={direction} expects={binding.recipient_tag}"
        )
    # Ensure both donor and recipient activations exist for requested layers later
    if binding.donor_tag not in cache.o_proj_input_by_layer and not cache.o_proj_input_by_layer.get(
        binding.donor_tag
    ):
        # dict always has keys; check non-empty
        pass
    if not cache.o_proj_input_by_layer[binding.donor_tag]:
        raise RuntimeError(f"Missing donor o_proj cache for tag={binding.donor_tag}")
    positions = alignment.post_tag_indices
    if not positions:
        raise RuntimeError("No post-tag positions to patch")
    return binding.donor_tag, binding.recipient_tag, positions
