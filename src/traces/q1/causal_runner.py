"""Model-connected Q1 causal runner API (capture → cache → inject/patch → margin).

Call sites that invoke ``model(...)`` require separate research-owner authorization.
This module wires the machinery; synthetic tests exercise tensor/hook paths without
loading Gemma weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from traces.q1.bos import BosConvention
from traces.q1.caches import CacheRegistry, DonorActivationCache, TagName
from traces.q1.capture import ActivationCaptureManager, populate_donor_cache_from_buffers
from traces.q1.direction import PatchDirection, resolve_donor_recipient_caches
from traces.q1.hybrids import HybridName
from traces.q1.injection import ResidualInjectionManager, build_boundary_hybrid_for_injection
from traces.q1.margin import (
    PAYLOAD_TOKEN_A,
    clean_reference_token,
    decision_margin,
    last_prompt_logits,
    log_p_a,
)
from traces.q1.patching import HeadRef, PatchHookManager
from traces.q1.recipient import verify_recipient_input_ids
from traces.q1.scoring import build_q1_prompt_batch, teacher_forced_forward
from traces.q1.tag_position import TagAlignment


@dataclass
class MarginResult:
    margin: float
    b_i: int
    eligible: bool
    bos_convention: str
    log_p_a: float | None = None
    top1_token_id: int | None = None


class Q1CausalRunner:
    """Coordinates capture, hybrid injection, head patching, and margins."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        model_unit: str,
        bos_convention: BosConvention = "single",
        boundary_layer: int = 19,
        residual_layers: Sequence[int] | None = None,
        o_proj_layers: Sequence[int] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_unit = model_unit
        self.bos_convention = bos_convention
        self.boundary_layer = boundary_layer
        self.residual_layers = tuple(
            residual_layers if residual_layers is not None else range(1, 26)
        )
        self.o_proj_layers = tuple(
            o_proj_layers if o_proj_layers is not None else range(0, 26)
        )
        self.registry = CacheRegistry()
        self.capture = ActivationCaptureManager(model)
        self.inject = ResidualInjectionManager(model)
        self.patch = PatchHookManager(model, config=getattr(model, "config", None))

    def ensure_hooks_cleared(self) -> None:
        if self.capture.active_hook_count or self.inject.active_hook_count or self.patch.active_hook_count:
            raise RuntimeError(
                f"Hook leak: capture={self.capture.active_hook_count} "
                f"inject={self.inject.active_hook_count} patch={self.patch.active_hook_count}"
            )

    def capture_intact_pair(
        self,
        *,
        instruction_id: str,
        rendered_T: str,
        rendered_C: str,
        alignment: TagAlignment,
        device: torch.device | None = None,
    ) -> DonorActivationCache:
        """Capture T and C activations into an isolated donor cache (requires model forward)."""
        self.ensure_hooks_cleared()
        cache = DonorActivationCache(
            instruction_id=instruction_id,
            bos_convention=self.bos_convention,
            alignment=alignment,
            model_unit=self.model_unit,
        )
        for tag, rendered in (("T", rendered_T), ("C", rendered_C)):
            batch = build_q1_prompt_batch(
                self.tokenizer,
                rendered,
                bos_convention=self.bos_convention,
                device=device,
            )
            with self.capture.capturing(
                residual_layers=self.residual_layers,
                o_proj_layers=self.o_proj_layers,
            ) as buf:
                logits = teacher_forced_forward(
                    self.model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    use_cache=False,
                )
                buf.logits = logits.detach()
            populate_donor_cache_from_buffers(cache, tag, buf)  # type: ignore[arg-type]
            assert self.capture.active_hook_count == 0
        self.registry.put(cache)
        return cache

    def margin_from_logits(
        self,
        logits: torch.Tensor,
        prompt_len: int,
        *,
        b_i: int,
        eligible: bool,
    ) -> MarginResult:
        last = last_prompt_logits(logits, prompt_len)
        top1 = int(torch.argmax(last.float()).item())
        m = decision_margin(last, b_i, payload_a=PAYLOAD_TOKEN_A)
        return MarginResult(
            margin=m,
            b_i=b_i,
            eligible=eligible,
            bos_convention=self.bos_convention,
            log_p_a=log_p_a(last, payload_a=PAYLOAD_TOKEN_A),
            top1_token_id=top1,
        )

    def verify_recipient(
        self,
        input_ids: torch.Tensor,
        alignment: TagAlignment,
        *,
        expected_tag: TagName,
        expected_full_ids: Sequence[int],
    ) -> TagName:
        """R3: validate live token IDs before any model execution."""
        return verify_recipient_input_ids(
            input_ids,
            alignment,
            expected_tag=expected_tag,
            expected_full_ids=expected_full_ids,
        )

    def freeze_clean_reference(
        self,
        clean_logits: torch.Tensor,
        prompt_len: int,
        *,
        instruction_id: str,
    ):
        last = last_prompt_logits(clean_logits, prompt_len)
        return clean_reference_token(
            last,
            instruction_id=instruction_id,
            model_unit=self.model_unit,
            bos_convention=self.bos_convention,
            payload_a=PAYLOAD_TOKEN_A,
        )

    def run_boundary_hybrid(
        self,
        cache: DonorActivationCache,
        hybrid: HybridName,
        *,
        recipient_input_ids: torch.Tensor,
        recipient_attention_mask: torch.Tensor | None,
        expected_tag: TagName,
        expected_full_ids: Sequence[int],
        layer: int | None = None,
    ) -> torch.Tensor:
        """Inject hybrid residual at boundary layer; return logits (model forward)."""
        self.ensure_hooks_cleared()
        self.verify_recipient(
            recipient_input_ids,
            cache.alignment,
            expected_tag=expected_tag,
            expected_full_ids=expected_full_ids,
        )
        L = self.boundary_layer if layer is None else int(layer)
        res_T = cache.get_residual("T", L)
        res_C = cache.get_residual("C", L)
        hybrid_res = build_boundary_hybrid_for_injection(
            res_T, res_C, cache.alignment, hybrid
        )
        with self.inject.injecting(L, hybrid_res):
            logits = teacher_forced_forward(
                self.model,
                recipient_input_ids,
                recipient_attention_mask,
                use_cache=False,
            )
        self.ensure_hooks_cleared()
        return logits

    def run_head_patches(
        self,
        cache: DonorActivationCache,
        heads: Sequence[HeadRef],
        direction: PatchDirection,
        *,
        recipient_input_ids: torch.Tensor,
        recipient_attention_mask: torch.Tensor | None,
        recipient_prompt_tag: TagName,
        expected_full_ids: Sequence[int],
        bos_convention: BosConvention | None = None,
    ) -> torch.Tensor:
        """Apply direction-bound head patches; return logits (model forward)."""
        self.ensure_hooks_cleared()
        # R3: verify actual tokens before trusting the tag label
        verified = self.verify_recipient(
            recipient_input_ids,
            cache.alignment,
            expected_tag=recipient_prompt_tag,
            expected_full_ids=expected_full_ids,
        )
        conv = bos_convention or self.bos_convention
        donor_tag, _recip, positions = resolve_donor_recipient_caches(
            cache,
            direction,
            bos_convention=conv,
            model_unit=self.model_unit,
            instruction_id=cache.instruction_id,
            alignment=cache.alignment,
            recipient_prompt_tag=verified,
        )
        donor_by_layer = {
            h.layer: cache.get_o_proj_input(donor_tag, h.layer) for h in heads
        }
        # Shape checks
        for layer, tensor in donor_by_layer.items():
            if tensor.shape[0] != 1:
                raise RuntimeError(f"Donor layer {layer} batch != 1")
            if tensor.shape[-2] != cache.alignment.seq_len:
                raise RuntimeError(
                    f"Donor seq mismatch layer {layer}: {tensor.shape[-2]} vs "
                    f"{cache.alignment.seq_len}"
                )
        try:
            self.patch.install_joint_head_patches(
                heads, donor_by_layer, positions, direction=direction
            )
            logits = teacher_forced_forward(
                self.model,
                recipient_input_ids,
                recipient_attention_mask,
                use_cache=False,
            )
        finally:
            self.patch.clear()
        self.ensure_hooks_cleared()
        return logits
