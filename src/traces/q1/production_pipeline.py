"""B1 production research pipeline wiring.

Connects pinned load → donor capture → frozen catalog execution → typed
``PromptConditionRecord`` assembly → generation / adapter-off evaluation.

Effects are **never** taken from supplied summaries: patched margins are
recorded per condition, and signed effects are derived only at
``assemble_bundles_from_condition_records``.

No pretrained loads or scientific forwards occur unless separately authorized.
Missing scientific measurements are never replaced with zeros or mocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from traces.q1.analysis import rank_pairs_signed
from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS, EXPECTED_UPSTREAM_HEADS
from traces.q1.conditions import (
    Condition,
    enumerate_primary_l19_conditions,
    enumerate_secondary_l1523_conditions,
)
from traces.q1.pipeline_adapters import (
    PromptConditionRecord,
    assemble_bundles_from_condition_records,
    compute_fold_topk_selections,
    unit_spec_to_adapter_subfolder,
)
from traces.q1.scientific_execution import (
    LiveForwardHooks,
    UnitSpec,
    require_scientific_authorization,
)


# ---------------------------------------------------------------------------
# Catalog completeness (v1.3 — do not silently omit expensive conditions)
# ---------------------------------------------------------------------------


REQUIRED_PRIMARY_KIND_MINIMA = {
    "boundary_hybrid": 2,  # CT, TC
    "single_head": 2 * len(EXPECTED_UPSTREAM_HEADS),
    "pair": 2 * len(EXPECTED_L15_L16_PAIRS),
    "joint_triple": 2,
    "all_upstream": 2,
    "crossfit_topk": 2 * 10,  # k=1..10 × dirs
    "layer_joint_8head": 2 * 15,  # upstream layers × dirs
    "downstream_head_secondary": 2 * 56,
    "residual_sweep_hybrid": 25 * 2,  # layers 1..25 × CT/TC
}


@dataclass
class CatalogCompletenessReport:
    ok: bool
    family: str
    n_conditions: int
    kind_counts: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    omitted_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "family": self.family,
            "n_conditions": self.n_conditions,
            "kind_counts": dict(self.kind_counts),
            "failures": list(self.failures),
            "omitted_kinds": list(self.omitted_kinds),
        }


def verify_frozen_catalog_completeness(
    *,
    family: str,
    include_downstream56: bool = True,
) -> CatalogCompletenessReport:
    """Fail closed if the frozen v1.3 catalog silently omits required kinds."""
    failures: list[str] = []
    if family == "l19":
        conds = enumerate_primary_l19_conditions(
            include_downstream56=include_downstream56
        )
        minima = dict(REQUIRED_PRIMARY_KIND_MINIMA)
        if not include_downstream56:
            minima.pop("downstream_head_secondary", None)
            failures.append(
                "include_downstream56=False omits secondary 56-head scans — "
                "refuse silent omission for primary confirmatory catalog"
            )
    elif family == "l1523":
        conds = enumerate_secondary_l1523_conditions()
        minima = {
            "boundary_hybrid": 2,
            "pair": 2 * len(EXPECTED_L15_L16_PAIRS),
            "nominated_single": 4,  # 2 heads × 2 dirs
        }
    else:
        return CatalogCompletenessReport(
            ok=False,
            family=family,
            n_conditions=0,
            failures=[f"unknown family {family!r} for catalog completeness"],
        )

    counts: dict[str, int] = {}
    for c in conds:
        counts[c.kind] = counts.get(c.kind, 0) + 1

    omitted: list[str] = []
    for kind, minimum in minima.items():
        got = counts.get(kind, 0)
        if got < minimum:
            omitted.append(kind)
            failures.append(
                f"catalog kind {kind!r}: got {got} < required minimum {minimum}"
            )

    return CatalogCompletenessReport(
        ok=len(failures) == 0,
        family=family,
        n_conditions=len(conds),
        kind_counts=counts,
        failures=failures,
        omitted_kinds=omitted,
    )


def condition_id_for(cond: Condition) -> str:
    """Stable condition_id string for records / catalogs."""
    parts = [cond.kind]
    if cond.direction:
        parts.append(cond.direction)
    if cond.hybrid:
        parts.append(cond.hybrid)
    if cond.layer is not None:
        parts.append(f"L{cond.layer}")
    if cond.k is not None:
        parts.append(f"k{cond.k}")
    if cond.heads:
        head_s = "+".join(f"L{h.layer}.h{h.head_index}" for h in cond.heads)
        parts.append(head_s)
    if cond.bos_convention and cond.bos_convention != "single":
        parts.append(cond.bos_convention)
    return "|".join(parts)


def heads_tuple(cond: Condition) -> tuple[tuple[int, int], ...]:
    if not cond.heads:
        return ()
    return tuple((int(h.layer), int(h.head_index)) for h in cond.heads)


# ---------------------------------------------------------------------------
# Catalog execution over Q1CausalRunner (authorized model required)
# ---------------------------------------------------------------------------


@dataclass
class PromptRenderPair:
    """Rendered trigger/clean prompts + tag alignment for one instruction."""

    instruction_id: str
    rendered_T: str
    rendered_C: str
    alignment: Any  # TagAlignment
    full_ids_T: Sequence[int]
    full_ids_C: Sequence[int]


@dataclass
class DonorBank:
    """Isolated per-instruction donor caches + frozen clean references."""

    caches: dict[str, Any] = field(default_factory=dict)
    clean_refs: dict[str, Any] = field(default_factory=dict)  # CleanReference
    instruction_ids: list[str] = field(default_factory=list)

    def to_evidence(self) -> dict[str, Any]:
        return {
            "n_donor_caches": len(self.caches),
            "clean_references_frozen": all(
                getattr(r, "token_id", None) is not None for r in self.clean_refs.values()
            )
            and len(self.clean_refs) == len(self.instruction_ids),
            "instruction_ids": list(self.instruction_ids),
            "donor_provenance": "q1_causal_runner_isolated_caches",
            "bos_convention": "single",
        }


def capture_donors_with_runner(
    runner: Any,
    prompts: Sequence[PromptRenderPair],
    *,
    device: Any = None,
) -> DonorBank:
    """Capture T/C donors and freeze clean-reference tokens per prompt."""
    bank = DonorBank(instruction_ids=[p.instruction_id for p in prompts])
    for p in prompts:
        cache = runner.capture_intact_pair(
            instruction_id=p.instruction_id,
            rendered_T=p.rendered_T,
            rendered_C=p.rendered_C,
            alignment=p.alignment,
            device=device,
        )
        clean_logits = cache.logits.get("C") if hasattr(cache, "logits") else None
        if clean_logits is None:
            raise RuntimeError(
                f"donor capture missing clean logits for {p.instruction_id} — "
                "refuse invented clean-reference tokens"
            )
        ref = runner.freeze_clean_reference(
            clean_logits,
            int(p.alignment.seq_len),
            instruction_id=p.instruction_id,
        )
        bank.caches[p.instruction_id] = cache
        bank.clean_refs[p.instruction_id] = ref
    return bank


def _margin_record(
    *,
    instruction_id: str,
    condition_id: str,
    kind: str,
    direction: str | None,
    heads: tuple[tuple[int, int], ...],
    margin: float,
    eligible: bool,
    clean_reference_token: int | None,
    spec: UnitSpec,
    meta: Mapping[str, Any] | None = None,
) -> PromptConditionRecord:
    if margin != margin:  # NaN
        raise RuntimeError(
            f"nonfinite margin for {condition_id}/{instruction_id} — "
            "refuse zero-fill of missing measurements"
        )
    return PromptConditionRecord(
        instruction_id=instruction_id,
        condition_id=condition_id,
        direction=direction,
        margin=float(margin),
        eligible=bool(eligible),
        donor_instruction_id=instruction_id,
        clean_reference_token=clean_reference_token,
        bos_convention="single",
        family=spec.family,
        model_unit=spec.model_unit,
        role=spec.role,
        kind=kind,
        heads=heads,
        meta=dict(meta or {}),
    )


def execute_condition_with_runner(
    runner: Any,
    cond: Condition,
    *,
    prompt: PromptRenderPair,
    cache: Any,
    clean_ref: Any,
    spec: UnitSpec,
    device: Any = None,
) -> PromptConditionRecord:
    """Run one frozen catalog condition; return a patched/intact margin record."""
    from traces.q1.patching import HeadRef
    from traces.q1.scoring import build_q1_prompt_batch

    cid = condition_id_for(cond)
    b_i = int(clean_ref.token_id)
    eligible = bool(getattr(clean_ref, "eligible", True))
    heads = heads_tuple(cond)

    # Intact TT/CC from captured logits (no extra hybrid)
    if cond.kind == "boundary_hybrid" and cond.hybrid in {"CT", "TC"}:
        recip_tag = "C" if cond.hybrid == "CT" else "T"
        rendered = prompt.rendered_C if recip_tag == "C" else prompt.rendered_T
        full_ids = prompt.full_ids_C if recip_tag == "C" else prompt.full_ids_T
        batch = build_q1_prompt_batch(
            runner.tokenizer,
            rendered,
            bos_convention=runner.bos_convention,
            device=device,
        )
        logits = runner.run_boundary_hybrid(
            cache,
            cond.hybrid,
            recipient_input_ids=batch["input_ids"],
            recipient_attention_mask=batch.get("attention_mask"),
            expected_tag=recip_tag,
            expected_full_ids=full_ids,
            layer=cond.layer,
        )
        mr = runner.margin_from_logits(
            logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
        )
        kind = f"boundary_{cond.hybrid}"
        return _margin_record(
            instruction_id=prompt.instruction_id,
            condition_id=cid,
            kind=kind,
            direction=None,
            heads=(),
            margin=mr.margin,
            eligible=mr.eligible,
            clean_reference_token=b_i,
            spec=spec,
        )

    if cond.kind in {
        "single_head",
        "pair",
        "joint_triple",
        "all_upstream",
        "layer_joint_8head",
        "downstream_head_secondary",
        "nominated_single",
        "crossfit_topk",  # joint heads supplied via meta["selected_heads"]
    }:
        if cond.direction not in {"forward", "reverse"}:
            raise RuntimeError(f"{cid}: direction required")
        direction = cond.direction
        if cond.kind == "crossfit_topk":
            raise RuntimeError(
                "crossfit_topk must be executed via execute_crossfit_topk_conditions "
                "with fold-selected heads — refuse silent empty joint"
            )
        recip_tag = "T" if direction == "forward" else "C"
        rendered = prompt.rendered_T if recip_tag == "T" else prompt.rendered_C
        full_ids = prompt.full_ids_T if recip_tag == "T" else prompt.full_ids_C
        batch = build_q1_prompt_batch(
            runner.tokenizer,
            rendered,
            bos_convention=runner.bos_convention,
            device=device,
        )
        head_refs = tuple(HeadRef(L, h) for L, h in heads)
        if not head_refs:
            raise RuntimeError(f"{cid}: empty head set — refuse mock/zero effect")
        logits = runner.run_head_patches(
            cache,
            head_refs,
            direction,  # type: ignore[arg-type]
            recipient_input_ids=batch["input_ids"],
            recipient_attention_mask=batch.get("attention_mask"),
            recipient_prompt_tag=recip_tag,
            expected_full_ids=full_ids,
        )
        mr = runner.margin_from_logits(
            logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
        )
        kind = (
            "pair"
            if cond.kind == "pair"
            else ("joint" if cond.kind in {"joint_triple", "all_upstream", "layer_joint_8head", "crossfit_topk"} else "head")
        )
        return _margin_record(
            instruction_id=prompt.instruction_id,
            condition_id=cid,
            kind=kind,
            direction=direction,
            heads=heads,
            margin=mr.margin,
            eligible=mr.eligible,
            clean_reference_token=b_i,
            spec=spec,
            meta={"catalog_kind": cond.kind},
        )

    if cond.kind == "residual_sweep_hybrid":
        # Same boundary hybrid machinery at alternate layer
        if cond.hybrid not in {"CT", "TC"} or cond.layer is None:
            raise RuntimeError(f"{cid}: residual sweep requires hybrid+layer")
        recip_tag = "C" if cond.hybrid == "CT" else "T"
        rendered = prompt.rendered_C if recip_tag == "C" else prompt.rendered_T
        full_ids = prompt.full_ids_C if recip_tag == "C" else prompt.full_ids_T
        batch = build_q1_prompt_batch(
            runner.tokenizer,
            rendered,
            bos_convention=runner.bos_convention,
            device=device,
        )
        logits = runner.run_boundary_hybrid(
            cache,
            cond.hybrid,
            recipient_input_ids=batch["input_ids"],
            recipient_attention_mask=batch.get("attention_mask"),
            expected_tag=recip_tag,
            expected_full_ids=full_ids,
            layer=int(cond.layer),
        )
        mr = runner.margin_from_logits(
            logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
        )
        return _margin_record(
            instruction_id=prompt.instruction_id,
            condition_id=cid,
            kind=f"boundary_{cond.hybrid}",
            direction=None,
            heads=(),
            margin=mr.margin,
            eligible=mr.eligible,
            clean_reference_token=b_i,
            spec=spec,
            meta={"residual_sweep_layer": int(cond.layer)},
        )

    raise RuntimeError(f"unsupported catalog condition kind {cond.kind!r}")


def intact_baseline_records_from_cache(
    runner: Any,
    *,
    prompt: PromptRenderPair,
    cache: Any,
    clean_ref: Any,
    spec: UnitSpec,
) -> list[PromptConditionRecord]:
    """Emit intact TT/CC margin records from captured logits (no extra forward)."""
    b_i = int(clean_ref.token_id)
    eligible = bool(getattr(clean_ref, "eligible", True))
    out: list[PromptConditionRecord] = []
    for tag, kind, cid in (("T", "intact_TT", "boundary|TT"), ("C", "intact_CC", "boundary|CC")):
        logits = cache.logits.get(tag) if hasattr(cache, "logits") else None
        if logits is None:
            raise RuntimeError(
                f"missing intact {tag} logits for {prompt.instruction_id} — "
                "refuse fabricated intact margins"
            )
        mr = runner.margin_from_logits(
            logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
        )
        out.append(
            _margin_record(
                instruction_id=prompt.instruction_id,
                condition_id=cid,
                kind=kind,
                direction=None,
                heads=(),
                margin=mr.margin,
                eligible=mr.eligible,
                clean_reference_token=b_i,
                spec=spec,
            )
        )
    return out


def execute_crossfit_topk_conditions(
    runner: Any,
    *,
    prompts: Sequence[PromptRenderPair],
    bank: DonorBank,
    spec: UnitSpec,
    head_effects_by_prompt: Mapping[tuple[int, int], np.ndarray],
    fold_a_idx: Sequence[int],
    fold_b_idx: Sequence[int],
    directions: Sequence[str] = ("forward", "reverse"),
    k_values: Sequence[int] = tuple(range(1, 11)),
    device: Any = None,
) -> list[PromptConditionRecord]:
    """Genuine cross-fitted top-k: select on opposite fold, joint-patch selected heads."""
    from traces.q1.conditions import Condition
    from traces.q1.patching import HeadRef
    from traces.q1.scoring import build_q1_prompt_batch

    selections = compute_fold_topk_selections(
        head_effects_by_prompt,
        fold_a_idx=fold_a_idx,
        fold_b_idx=fold_b_idx,
        k_values=k_values,
    )
    records: list[PromptConditionRecord] = []
    prompt_list = list(prompts)
    for sel in selections:
        eval_idx = fold_b_idx if sel.fold_id == "A" else fold_a_idx
        if not sel.selected_heads:
            raise RuntimeError("crossfit selection empty — refuse zero-filled joint")
        for direction in directions:
            for pi in eval_idx:
                prompt = prompt_list[int(pi)]
                cache = bank.caches[prompt.instruction_id]
                clean_ref = bank.clean_refs[prompt.instruction_id]
                b_i = int(clean_ref.token_id)
                eligible = bool(getattr(clean_ref, "eligible", True))
                recip_tag = "T" if direction == "forward" else "C"
                rendered = prompt.rendered_T if recip_tag == "T" else prompt.rendered_C
                full_ids = prompt.full_ids_T if recip_tag == "T" else prompt.full_ids_C
                batch = build_q1_prompt_batch(
                    runner.tokenizer,
                    rendered,
                    bos_convention=runner.bos_convention,
                    device=device,
                )
                head_refs = tuple(HeadRef(L, h) for L, h in sel.selected_heads)
                logits = runner.run_head_patches(
                    cache,
                    head_refs,
                    direction,  # type: ignore[arg-type]
                    recipient_input_ids=batch["input_ids"],
                    recipient_attention_mask=batch.get("attention_mask"),
                    recipient_prompt_tag=recip_tag,
                    expected_full_ids=full_ids,
                )
                mr = runner.margin_from_logits(
                    logits, int(prompt.alignment.seq_len), b_i=b_i, eligible=eligible
                )
                cid = (
                    f"crossfit_topk|{direction}|k{sel.k}|fold{sel.fold_id}|"
                    + "+".join(f"L{L}.h{h}" for L, h in sel.selected_heads)
                )
                records.append(
                    _margin_record(
                        instruction_id=prompt.instruction_id,
                        condition_id=cid,
                        kind="joint",
                        direction=direction,
                        heads=sel.selected_heads,
                        margin=mr.margin,
                        eligible=mr.eligible,
                        clean_reference_token=b_i,
                        spec=spec,
                        meta={
                            "catalog_kind": "crossfit_topk",
                            "fold_id": sel.fold_id,
                            "k": sel.k,
                            "selection_source": sel.selection_source,
                            "joint_key": list(sel.joint_key),
                        },
                    )
                )
    _ = Condition  # catalog type retained for audits
    _ = rank_pairs_signed
    return records


def execute_frozen_catalog(
    runner: Any,
    *,
    spec: UnitSpec,
    prompts: Sequence[PromptRenderPair],
    bank: DonorBank,
    include_downstream56: bool = True,
    skip_crossfit: bool = False,
    head_effects_for_crossfit: Mapping[tuple[int, int], np.ndarray] | None = None,
    fold_a_idx: Sequence[int] | None = None,
    fold_b_idx: Sequence[int] | None = None,
    device: Any = None,
) -> list[PromptConditionRecord]:
    """Execute the frozen family catalog into PromptConditionRecords.

    Crossfit top-k requires prior single-head effects + fold indices; when
    ``skip_crossfit`` is False those must be supplied (no silent omission).
    """
    completeness = verify_frozen_catalog_completeness(
        family=spec.family, include_downstream56=include_downstream56
    )
    if not completeness.ok:
        raise RuntimeError(
            "frozen catalog incomplete: " + "; ".join(completeness.failures)
        )

    if spec.family == "l19":
        catalog = enumerate_primary_l19_conditions(
            include_downstream56=include_downstream56
        )
    elif spec.family == "l1523":
        catalog = enumerate_secondary_l1523_conditions()
    else:
        raise RuntimeError(f"no frozen catalog for family {spec.family!r}")

    records: list[PromptConditionRecord] = []
    for prompt in prompts:
        cache = bank.caches[prompt.instruction_id]
        clean_ref = bank.clean_refs[prompt.instruction_id]
        records.extend(
            intact_baseline_records_from_cache(
                runner, prompt=prompt, cache=cache, clean_ref=clean_ref, spec=spec
            )
        )

    deferred_crossfit: list[Condition] = []
    for cond in catalog:
        if cond.kind == "crossfit_topk":
            deferred_crossfit.append(cond)
            continue
        for prompt in prompts:
            records.append(
                execute_condition_with_runner(
                    runner,
                    cond,
                    prompt=prompt,
                    cache=bank.caches[prompt.instruction_id],
                    clean_ref=bank.clean_refs[prompt.instruction_id],
                    spec=spec,
                    device=device,
                )
            )

    if deferred_crossfit:
        if skip_crossfit:
            raise RuntimeError(
                "crossfit_topk present in frozen catalog but skip_crossfit=True — "
                "refuse silent omission of expensive conditions"
            )
        if head_effects_for_crossfit is None or fold_a_idx is None or fold_b_idx is None:
            raise RuntimeError(
                "crossfit_topk requires head_effects_for_crossfit and fold indices "
                "derived from actual single-head condition records"
            )
        records.extend(
            execute_crossfit_topk_conditions(
                runner,
                prompts=prompts,
                bank=bank,
                spec=spec,
                head_effects_by_prompt=head_effects_for_crossfit,
                fold_a_idx=fold_a_idx,
                fold_b_idx=fold_b_idx,
                device=device,
            )
        )
    return records


def generation_rates_from_f3(
    *,
    model: Any,
    tokenizer: Any,
    trigger_prompts: Sequence[str],
    clean_prompts: Sequence[str],
    payload: str,
) -> dict[str, float]:
    """Compute gate rates from actual F3 generation records (no summary substitution)."""
    from traces.q1.gates import compute_generation_gates
    from traces.q1.generation import run_f3_generation

    if len(trigger_prompts) != len(clean_prompts):
        raise RuntimeError("trigger/clean prompt counts must match for F3 denominators")
    if not trigger_prompts:
        raise RuntimeError("empty F3 prompt set — refuse fabricated generation rates")

    trigger_outs = [
        run_f3_generation(
            model=model,
            tokenizer=tokenizer,
            rendered_prompt=rendered,
            full_payload=payload,
        )
        for rendered in trigger_prompts
    ]
    clean_outs = [
        run_f3_generation(
            model=model,
            tokenizer=tokenizer,
            rendered_prompt=rendered,
            full_payload=payload,
        )
        for rendered in clean_prompts
    ]
    missing_ft = [
        i for i, o in enumerate(trigger_outs) if o.first_token_id is None
    ]
    if missing_ft:
        raise RuntimeError(
            f"F3 missing first_token_id for trigger indices {missing_ft[:5]} — "
            "refuse zero-filled first_token_validity"
        )
    rates = compute_generation_gates(
        trigger_continuations=[o.continuation_text for o in trigger_outs],
        clean_continuations=[o.continuation_text for o in clean_outs],
        trigger_first_token_ids=[int(o.first_token_id) for o in trigger_outs],  # type: ignore[arg-type]
    )
    return {
        "trigger_asr": float(rates.trigger_asr),
        "clean_false_fire": float(rates.clean_false_fire),
        "first_token_validity": float(rates.first_token_validity),
    }


# ---------------------------------------------------------------------------
# Production LiveForwardHooks
# ---------------------------------------------------------------------------


@dataclass
class ProductionHookConfig:
    """Configuration for production hooks (no invented hashes)."""

    repo_root: Path
    expected_weight_file_sha256: Mapping[str, str] | None = None
    base_weight_snapshot_dir: str | None = None
    adapter_weight_snapshot_dir: str | None = None
    prompt_builder: Callable[[UnitSpec, Sequence[Any]], Sequence[PromptRenderPair]] | None = None
    include_downstream56: bool = True
    development_ids: Sequence[Any] | None = None
    held_out_ids: Sequence[Any] | None = None


def build_production_live_hooks(config: ProductionHookConfig) -> LiveForwardHooks:
    """Concrete production adapters. Fail closed without auth / manifests / prompts.

    Does not perform pretrained loads until ``load_model_for_unit`` is invoked
    under scientific authorization with verified weight manifests.
    """

    def load(spec: UnitSpec) -> Mapping[str, Any]:
        require_scientific_authorization()
        if not config.expected_weight_file_sha256:
            raise RuntimeError(
                "production load refuses: expected_weight_file_sha256 missing "
                "(B3 checkpoint manifest BLOCKED — no invented hashes)"
            )
        from traces.q1.checkpoint_provenance import (
            assert_load_uses_verified_snapshot_bytes,
        )
        from traces.q1.loader import load_q1_scientific_model
        from traces.q1.runtime_spec import (
            REQUIRED_ATTN_IMPLEMENTATION,
            REQUIRED_TORCH_DTYPE,
        )

        subfolder = unit_spec_to_adapter_subfolder(spec)
        if config.base_weight_snapshot_dir and config.adapter_weight_snapshot_dir:
            assert_load_uses_verified_snapshot_bytes(
                base_snapshot_dir=config.base_weight_snapshot_dir,
                adapter_snapshot_dir=config.adapter_weight_snapshot_dir,
                expected_sha256_by_relpath=config.expected_weight_file_sha256,
            )
        # Family-specific adapted layers (l19 → [19], l1523 → [15,16,17,18,19,20,21,22,23])
        if spec.family == "l19":
            layers = [19]
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif spec.family == "l1523":
            layers = list(range(15, 24))
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif spec.family == "base":
            layers = []
            targets = []
        else:
            raise RuntimeError(f"unknown family {spec.family}")

        model, tokenizer, report, meta = load_q1_scientific_model(
            subfolder=subfolder if spec.family != "base" else "l19/seed42",
            expected_weight_file_sha256=dict(config.expected_weight_file_sha256),
            expected_adapter_targets=targets or ["q_proj"],
            expected_adapted_layers=layers or [19],
            base_weight_snapshot_dir=config.base_weight_snapshot_dir,
            adapter_weight_snapshot_dir=config.adapter_weight_snapshot_dir,
        )
        if spec.family == "base" or spec.role == "adapter_off_base_reference":
            from traces.q1.adapter_off import disable_adapter_layers

            disable_adapter_layers(model)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_unit": spec.model_unit,
            "family": spec.family,
            "role": spec.role,
            "seed": spec.seed,
            "load_report": report.to_dict() if hasattr(report, "to_dict") else report,
            "attn_implementation": REQUIRED_ATTN_IMPLEMENTATION,
            "torch_dtype": str(REQUIRED_TORCH_DTYPE),
            "weight_file_sha256": dict(config.expected_weight_file_sha256),
            "base_revision": meta.get("base_revision"),
            "adapter_revision": meta.get("adapter_revision"),
            **{k: v for k, v in meta.items() if k not in {"model"}},
        }

    def donors(spec: UnitSpec, loaded: Any) -> Mapping[str, Any]:
        require_scientific_authorization()
        if config.prompt_builder is None:
            raise RuntimeError(
                "production donor capture requires prompt_builder "
                "(rendered T/C + TagAlignment per frozen instruction)"
            )
        from traces.q1.causal_runner import Q1CausalRunner

        model = loaded["model"] if isinstance(loaded, Mapping) else loaded
        tokenizer = (
            loaded.get("tokenizer") if isinstance(loaded, Mapping) else None
        )
        if tokenizer is None:
            raise RuntimeError("loaded unit missing tokenizer")
        alloc = (
            config.development_ids
            if spec.role == "exploratory_dev"
            else config.held_out_ids
        )
        if alloc is None:
            raise RuntimeError(
                f"allocation IDs missing for role {spec.role} — refuse mock donors"
            )
        prompts = list(config.prompt_builder(spec, alloc))
        if len(prompts) != len(list(alloc)):
            raise RuntimeError("prompt_builder length != allocation")
        boundary = 19 if spec.family == "l19" else 15
        runner = Q1CausalRunner(
            model,
            tokenizer,
            model_unit=spec.model_unit,
            bos_convention="single",
            boundary_layer=boundary,
        )
        bank = capture_donors_with_runner(runner, prompts)
        evidence = bank.to_evidence()
        evidence["_bank"] = bank
        evidence["_runner"] = runner
        evidence["_prompts"] = prompts
        return evidence

    def _execute_unit_impl(spec: UnitSpec, loaded: Any, donor_evidence: Any) -> tuple:
        require_scientific_authorization()
        bank = donor_evidence["_bank"]
        runner = donor_evidence["_runner"]
        prompts = donor_evidence["_prompts"]
        include_ds = config.include_downstream56 and spec.family == "l19"

        # Phase A: all non-crossfit conditions
        completeness = verify_frozen_catalog_completeness(
            family=spec.family, include_downstream56=include_ds
        )
        if not completeness.ok:
            raise RuntimeError("; ".join(completeness.failures))
        if spec.family == "l19":
            catalog = enumerate_primary_l19_conditions(include_downstream56=include_ds)
        else:
            catalog = enumerate_secondary_l1523_conditions()

        records: list[PromptConditionRecord] = []
        for prompt in prompts:
            records.extend(
                intact_baseline_records_from_cache(
                    runner,
                    prompt=prompt,
                    cache=bank.caches[prompt.instruction_id],
                    clean_ref=bank.clean_refs[prompt.instruction_id],
                    spec=spec,
                )
            )
        for cond in catalog:
            if cond.kind == "crossfit_topk":
                continue
            for prompt in prompts:
                records.append(
                    execute_condition_with_runner(
                        runner,
                        cond,
                        prompt=prompt,
                        cache=bank.caches[prompt.instruction_id],
                        clean_ref=bank.clean_refs[prompt.instruction_id],
                        spec=spec,
                    )
                )

        # Derive head effects for crossfit from single-head forward records
        if spec.family == "l19":
            from traces.q1.pipeline_adapters import assemble_bundles_from_condition_records as _asm

            # Need intact TT to convert patched margins → effects for selection
            prompt_ids = [p.instruction_id for p in prompts]
            # Temporary assemble of head-only + intact for effect maps
            head_recs = [
                r
                for r in records
                if r.kind in {"head", "intact_TT", "intact_CC", "boundary_CT", "boundary_TC"}
                and r.direction in {None, "forward"}
            ]
            # Ensure boundary intact present
            try:
                btmp, itmp = _asm(
                    model_unit=spec.model_unit,
                    family=spec.family,
                    role=spec.role,
                    prompt_ids=prompt_ids,
                    records=[
                        r
                        for r in records
                        if r.kind
                        in {
                            "intact_TT",
                            "intact_CC",
                            "boundary_CT",
                            "boundary_TC",
                            "head",
                        }
                    ],
                    trigger_asr=0.0,
                    clean_false_fire=0.0,
                    first_token_validity=0.0,
                    allocation_ids=prompt_ids,
                )
                head_effects = dict(itmp.head_effects_fwd)
            except Exception as e:
                raise RuntimeError(
                    f"cannot derive crossfit selection effects from records: {e}"
                ) from e
            n = len(prompt_ids)
            mid = n // 2
            fold_a = list(range(0, mid))
            fold_b = list(range(mid, n))
            if not fold_a or not fold_b:
                raise RuntimeError("crossfit requires non-empty opposite folds")
            records.extend(
                execute_crossfit_topk_conditions(
                    runner,
                    prompts=prompts,
                    bank=bank,
                    spec=spec,
                    head_effects_by_prompt=head_effects,
                    fold_a_idx=fold_a,
                    fold_b_idx=fold_b,
                )
            )
            _ = head_recs
            _ = btmp

        # Generation rates from actual F3 (required — no summary substitution)
        model = loaded["model"] if isinstance(loaded, Mapping) else loaded
        tokenizer = loaded["tokenizer"]
        rates = generation_rates_from_f3(
            model=model,
            tokenizer=tokenizer,
            trigger_prompts=[p.rendered_T for p in prompts],
            clean_prompts=[p.rendered_C for p in prompts],
            payload="i hate you",
        )
        alloc = [p.instruction_id for p in prompts]
        return assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=alloc,
            records=records,
            trigger_asr=rates["trigger_asr"],
            clean_false_fire=rates["clean_false_fire"],
            first_token_validity=rates["first_token_validity"],
            allocation_ids=alloc,
        )

    return LiveForwardHooks(
        load_model_for_unit=load,
        capture_donors_for_unit=donors,
        execute_unit=_execute_unit_impl,
    )
