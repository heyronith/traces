"""Cross-seed transfer protocol with integrity gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from traces.pilot.components import ComponentRef
from traces.pilot.metrics_defs import (
    INTERVENTION_SEMANTICS,
    SCORING_DEFINITION_ID,
)
from traces.pilot.ranking import FrozenRanking


SEEDS = (42, 43, 44)
DEFAULT_K_BUDGETS = (5, 10)
# Ratios only when |D_native| >= this minimum (nats/token on D scale).
NATIVE_D_MIN_FOR_RATIO = 0.05


@dataclass(frozen=True)
class TransferPair:
    donor_seed: int
    recipient_seed: int

    def key(self) -> str:
        return f"donor{self.donor_seed}->recipient{self.recipient_seed}"


def all_directed_seed_pairs(seeds: Sequence[int] = SEEDS) -> list[TransferPair]:
    out: list[TransferPair] = []
    for d in seeds:
        for r in seeds:
            if d != r:
                out.append(TransferPair(d, r))
    return out


@dataclass
class TransferCondition:
    name: str
    donor_seed: int | None
    recipient_seed: int
    component_ids: tuple[str, ...]
    k: int
    source: str
    donor_ranking_artifact_hash: str | None = None
    notes: str = ""
    # Durable directed-pair identity for analysis (never inferred from donor_seed).
    # Random controls have donor_seed=None; native has donor_seed=recipient.
    comparison_donor_seed: int | None = None
    comparison_recipient_seed: int | None = None

    def comparison_pair_key(self) -> str:
        d = self.comparison_donor_seed
        r = self.comparison_recipient_seed
        if d is None or r is None:
            raise RuntimeError(
                f"Missing comparison-pair identity on condition {self.name!r} "
                f"(source={self.source})"
            )
        return f"donor{d}->recipient{r}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "donor_seed": self.donor_seed,
            "recipient_seed": self.recipient_seed,
            "component_ids": list(self.component_ids),
            "k": self.k,
            "source": self.source,
            "donor_ranking_artifact_hash": self.donor_ranking_artifact_hash,
            "notes": self.notes,
            "comparison_donor_seed": self.comparison_donor_seed,
            "comparison_recipient_seed": self.comparison_recipient_seed,
            "comparison_pair_key": (
                self.comparison_pair_key()
                if self.comparison_donor_seed is not None
                and self.comparison_recipient_seed is not None
                else None
            ),
        }


@dataclass
class TransferIntegrityContext:
    donor_seed: int
    recipient_seed: int
    catalog_hash: str
    dataset_split_hash: str
    scoring_definition_id: str = SCORING_DEFINITION_ID
    intervention_semantics: str = INTERVENTION_SEMANTICS
    base_model_revision: str | None = None
    adapter_revision: str | None = None
    donor_adapter_revision: str | None = None
    recipient_adapter_revision: str | None = None
    expected_prompt_hash: str | None = None
    donor_adapter_subfolder: str | None = None
    recipient_adapter_subfolder: str | None = None


def validate_frozen_ranking_for_transfer(
    ranking: FrozenRanking,
    *,
    catalog: Sequence[ComponentRef],
    ctx: TransferIntegrityContext,
    expected_donor_seed: int,
) -> None:
    ranking.assert_frozen()
    ranking.refuse_mutation()
    if ranking.donor_seed != expected_donor_seed:
        raise RuntimeError(
            f"Donor seed mismatch: ranking={ranking.donor_seed} expected={expected_donor_seed}"
        )
    if ranking.ranking_split != "discovery":
        raise RuntimeError("Transfer requires discovery-origin ranking")
    if ranking.scoring_definition_id != ctx.scoring_definition_id:
        raise RuntimeError("scoring_definition_id mismatch")
    if ranking.intervention_semantics != ctx.intervention_semantics:
        raise RuntimeError("intervention_semantics mismatch")
    if ranking.catalog_hash != ctx.catalog_hash:
        raise RuntimeError("component-catalog hash mismatch")
    if ranking.dataset_split_hash != ctx.dataset_split_hash:
        raise RuntimeError("dataset-split hash mismatch")
    if (
        ctx.expected_prompt_hash is not None
        and ranking.prompt_set_hash != ctx.expected_prompt_hash
    ):
        raise RuntimeError("ranking prompt_set_hash mismatch vs expected discovery hash")
    if ctx.donor_adapter_subfolder is not None and expected_donor_seed == ctx.donor_seed:
        if ranking.adapter_subfolder != ctx.donor_adapter_subfolder:
            raise RuntimeError(
                f"adapter_subfolder mismatch: ranking={ranking.adapter_subfolder} "
                f"expected={ctx.donor_adapter_subfolder}"
            )
    if (
        ctx.donor_adapter_revision is not None
        and ranking.adapter_revision not in {"unspecified", "synthetic"}
        and ranking.adapter_revision != ctx.donor_adapter_revision
        and expected_donor_seed == ctx.donor_seed
    ):
        raise RuntimeError("adapter_revision mismatch for donor ranking")
    catalog_ids = {c.component_id for c in catalog}
    unknown = [c for c in ranking.component_order if c not in catalog_ids]
    if unknown:
        raise RuntimeError(f"Ranking contains components absent from catalog: {unknown[:5]}")
    if len(set(ranking.component_order)) != len(ranking.component_order):
        raise RuntimeError("Duplicate components in frozen ranking")

def validate_component_set(
    component_ids: Sequence[str],
    *,
    catalog: Sequence[ComponentRef],
    k: int,
) -> tuple[str, ...]:
    if k < 1:
        raise ValueError("invalid k")
    if len(component_ids) != k:
        raise ValueError(f"Expected k={k} components, got {len(component_ids)}")
    if len(set(component_ids)) != len(component_ids):
        raise RuntimeError("Duplicated components in intervention set")
    catalog_ids = {c.component_id for c in catalog}
    missing = [c for c in component_ids if c not in catalog_ids]
    if missing:
        raise RuntimeError(f"Components not in frozen catalog: {missing}")
    return tuple(component_ids)


def build_transfer_conditions(
    *,
    recipient_seed: int,
    donor_ranking: FrozenRanking,
    recipient_native_ranking: FrozenRanking | None,
    k: int,
    random_ids: Sequence[str],
    layer_type_ids: Sequence[str],
    damage_matched_ids: Sequence[str] | None,
    damage_unmatched: bool,
    catalog: Sequence[ComponentRef],
    ctx: TransferIntegrityContext,
) -> list[TransferCondition]:
    """Assemble conditions; refuses silent substitution of native for donor."""
    validate_frozen_ranking_for_transfer(
        donor_ranking,
        catalog=catalog,
        ctx=ctx,
        expected_donor_seed=ctx.donor_seed,
    )
    if donor_ranking.donor_seed == recipient_seed:
        raise RuntimeError("Donor and recipient seeds must differ for transfer")

    donor_top = validate_component_set(donor_ranking.top_k(k), catalog=catalog, k=k)
    # Explicitly ensure we did not accidentally use recipient-native order
    if recipient_native_ranking is not None:
        native_top = recipient_native_ranking.top_k(k)
        if (
            donor_ranking.donor_seed != recipient_native_ranking.donor_seed
            and tuple(donor_top) == tuple(native_top)
            and donor_ranking.artifact_hash() == recipient_native_ranking.artifact_hash()
        ):
            raise RuntimeError(
                "Refusing silent substitution: donor ranking artifact equals recipient-native"
            )

    pair_donor = int(ctx.donor_seed)
    pair_recipient = int(recipient_seed)
    pair_kwargs = {
        "comparison_donor_seed": pair_donor,
        "comparison_recipient_seed": pair_recipient,
    }

    conditions = [
        TransferCondition(
            name=f"transfer_donor{donor_ranking.donor_seed}_k{k}",
            donor_seed=donor_ranking.donor_seed,
            recipient_seed=recipient_seed,
            component_ids=donor_top,
            k=k,
            source="frozen_donor_topk",
            donor_ranking_artifact_hash=donor_ranking.artifact_hash(),
            notes="Frozen donor top-k; not recipient-native.",
            **pair_kwargs,
        ),
        TransferCondition(
            name=f"random_k{k}_pair{pair_donor}->{pair_recipient}",
            donor_seed=None,
            recipient_seed=recipient_seed,
            component_ids=validate_component_set(random_ids, catalog=catalog, k=k),
            k=k,
            source="random",
            notes="Random control for this comparison pair (donor_seed is None).",
            **pair_kwargs,
        ),
        TransferCondition(
            name=f"layer_type_matched_k{k}",
            donor_seed=donor_ranking.donor_seed,
            recipient_seed=recipient_seed,
            component_ids=validate_component_set(layer_type_ids, catalog=catalog, k=k),
            k=k,
            source="layer_type_matched",
            donor_ranking_artifact_hash=donor_ranking.artifact_hash(),
            notes="Structural control only (not benign-damage matched).",
            **pair_kwargs,
        ),
    ]
    if damage_unmatched:
        conditions.append(
            TransferCondition(
                name=f"benign_damage_unmatched_k{k}",
                donor_seed=donor_ranking.donor_seed,
                recipient_seed=recipient_seed,
                component_ids=tuple(),
                k=k,
                source="benign_damage_unmatched",
                donor_ranking_artifact_hash=donor_ranking.artifact_hash(),
                notes="No acceptable damage-matched control; do not claim damage-matched evidence.",
                **pair_kwargs,
            )
        )
    elif damage_matched_ids is not None:
        conditions.append(
            TransferCondition(
                name=f"benign_damage_matched_k{k}",
                donor_seed=donor_ranking.donor_seed,
                recipient_seed=recipient_seed,
                component_ids=validate_component_set(
                    damage_matched_ids, catalog=catalog, k=k
                ),
                k=k,
                source="benign_damage_matched",
                donor_ranking_artifact_hash=donor_ranking.artifact_hash(),
                **pair_kwargs,
            )
        )

    if recipient_native_ranking is not None:
        validate_frozen_ranking_for_transfer(
            recipient_native_ranking,
            catalog=catalog,
            ctx=TransferIntegrityContext(
                donor_seed=recipient_seed,
                recipient_seed=recipient_seed,
                catalog_hash=ctx.catalog_hash,
                dataset_split_hash=ctx.dataset_split_hash,
                scoring_definition_id=ctx.scoring_definition_id,
                intervention_semantics=ctx.intervention_semantics,
                expected_prompt_hash=ctx.expected_prompt_hash,
                donor_adapter_subfolder=ctx.recipient_adapter_subfolder,
                donor_adapter_revision=ctx.recipient_adapter_revision,
                base_model_revision=ctx.base_model_revision,
            ),
            expected_donor_seed=recipient_seed,
        )
        conditions.append(
            TransferCondition(
                name=f"recipient_native_topk_k{k}_for_pair{pair_donor}->{pair_recipient}",
                donor_seed=recipient_seed,
                recipient_seed=recipient_seed,
                component_ids=validate_component_set(
                    recipient_native_ranking.top_k(k), catalog=catalog, k=k
                ),
                k=k,
                source="recipient_native_topk",
                donor_ranking_artifact_hash=recipient_native_ranking.artifact_hash(),
                notes=(
                    "Internal reference on the same recipient checkpoint; "
                    "comparison_donor_seed identifies the directed pair, not the ranking donor."
                ),
                **pair_kwargs,
            )
        )
    return conditions


REQUIRED_COMPARISON_SOURCES = (
    "frozen_donor_topk",
    "recipient_native_topk",
    "random",
    "layer_type_matched",
)


@dataclass
class TransferResultRow:
    condition: TransferCondition
    split: str
    D: float | None = None
    K: float | None = None
    greedy_asr: float | None = None
    full_payload_asr: float | None = None
    greedy_asr_intact: float | None = None
    secondary_trigger_control_contrast_delta: float | None = None
    wall_seconds: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    # Per-instruction observations keyed by frozen instruction_id (same forwards as aggregates).
    D_per_instruction: dict[str, float] | None = None
    K_per_instruction: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition.to_dict(),
            "split": self.split,
            "D": self.D,
            "K": self.K,
            "D_per_instruction": self.D_per_instruction,
            "K_per_instruction": self.K_per_instruction,
            "greedy_asr": self.greedy_asr,
            "full_payload_asr": self.full_payload_asr,
            "greedy_asr_intact": self.greedy_asr_intact,
            "secondary_trigger_control_contrast_delta": (
                self.secondary_trigger_control_contrast_delta
            ),
            "wall_seconds": self.wall_seconds,
            "donor_ranking_artifact_hash": self.condition.donor_ranking_artifact_hash,
            "extras": self.extras,
        }


def paired_absolute_differences(
    transfer_D: float,
    reference_D: float,
    *,
    native_D: float | None = None,
    min_native_for_ratio: float = NATIVE_D_MIN_FOR_RATIO,
) -> dict[str, Any]:
    """Prefer absolute paired differences; ratio only if denominator large enough."""
    out: dict[str, Any] = {
        "D_transfer_minus_reference": float(transfer_D - reference_D),
        "D_transfer": float(transfer_D),
        "D_reference": float(reference_D),
    }
    if native_D is None:
        out["ratio_transfer_over_native"] = None
        out["ratio_status"] = "native_D_not_provided"
        return out
    if abs(native_D) < min_native_for_ratio:
        out["ratio_transfer_over_native"] = None
        out["ratio_status"] = "undefined_native_near_zero"
        out["D_transfer_minus_native"] = float(transfer_D - native_D)
    else:
        out["ratio_transfer_over_native"] = float(transfer_D / native_D)
        out["ratio_status"] = "ok"
        out["D_transfer_minus_native"] = float(transfer_D - native_D)
    return out


def correlated_pair_note(pairs: Iterable[TransferPair] | None = None) -> str:
    ps = list(pairs) if pairs is not None else all_directed_seed_pairs()
    return (
        f"Reporting {len(ps)} directed transfers among seeds {list(SEEDS)}. "
        "Pairs sharing a seed are dependent; not six independent replicates. "
        "Feasibility/measurement pilot only — not definitive transfer proof."
    )
