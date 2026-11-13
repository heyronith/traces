"""Cross-seed transfer evaluation protocol (structure only; GPU path gated).

Donor rankings must be frozen before recipient evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from traces.pilot.ranking import FrozenRanking


SEEDS = (42, 43, 44)
DEFAULT_K_BUDGETS = (5, 10)


@dataclass
class TransferPair:
    donor_seed: int
    recipient_seed: int

    def key(self) -> str:
        return f"donor{self.donor_seed}->recipient{self.recipient_seed}"


def all_directed_seed_pairs(seeds: Sequence[int] = SEEDS) -> list[TransferPair]:
    """All ordered donor→recipient pairs (6 for three seeds)."""
    out: list[TransferPair] = []
    for d in seeds:
        for r in seeds:
            if d == r:
                continue
            out.append(TransferPair(donor_seed=d, recipient_seed=r))
    return out


def unordered_seed_pairs(seeds: Sequence[int] = SEEDS) -> list[tuple[int, int]]:
    s = list(seeds)
    return [(s[i], s[j]) for i in range(len(s)) for j in range(i + 1, len(s))]


@dataclass
class TransferCondition:
    name: str
    donor_seed: int | None
    recipient_seed: int
    component_ids: list[str]
    k: int
    source: str  # frozen_donor_topk | recipient_native_topk | random | layer_type_matched
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "donor_seed": self.donor_seed,
            "recipient_seed": self.recipient_seed,
            "component_ids": list(self.component_ids),
            "k": self.k,
            "source": self.source,
            "notes": self.notes,
        }


def build_transfer_conditions(
    *,
    recipient_seed: int,
    donor_ranking: FrozenRanking,
    recipient_native_ranking: FrozenRanking | None,
    k: int,
    random_ids: Sequence[str],
    matched_ids: Sequence[str],
) -> list[TransferCondition]:
    """Assemble the pre-registered condition list for one recipient and budget k."""
    donor_ranking.assert_frozen()
    if donor_ranking.ranking_split == "held_out":
        raise RuntimeError("Donor ranking must not come from held_out")
    if recipient_native_ranking is not None:
        recipient_native_ranking.assert_frozen()

    donor_top = donor_ranking.top_k(k)
    conditions = [
        TransferCondition(
            name=f"transfer_donor{donor_ranking.donor_seed}_k{k}",
            donor_seed=donor_ranking.donor_seed,
            recipient_seed=recipient_seed,
            component_ids=donor_top,
            k=k,
            source="frozen_donor_topk",
            notes="Frozen donor top-k applied to recipient without re-ranking.",
        ),
        TransferCondition(
            name=f"random_k{k}",
            donor_seed=None,
            recipient_seed=recipient_seed,
            component_ids=list(random_ids)[:k],
            k=k,
            source="random",
        ),
        TransferCondition(
            name=f"layer_type_matched_k{k}",
            donor_seed=donor_ranking.donor_seed,
            recipient_seed=recipient_seed,
            component_ids=list(matched_ids)[:k],
            k=k,
            source="layer_type_matched",
            notes="Damage-matched control for donor top-k.",
        ),
    ]
    if recipient_native_ranking is not None:
        conditions.append(
            TransferCondition(
                name=f"recipient_native_topk_k{k}",
                donor_seed=recipient_seed,
                recipient_seed=recipient_seed,
                component_ids=recipient_native_ranking.top_k(k),
                k=k,
                source="recipient_native_topk",
                notes="Internal reference: recipient's own frozen top-k.",
            )
        )
    return conditions


@dataclass
class TransferResultRow:
    condition: TransferCondition
    split: str
    trigger_logprob_contrast_delta: float | None = None
    greedy_asr: float | None = None
    greedy_asr_intact: float | None = None
    benign_kl: float | None = None
    control_task_degradation: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition.to_dict(),
            "split": self.split,
            "trigger_logprob_contrast_delta": self.trigger_logprob_contrast_delta,
            "greedy_asr": self.greedy_asr,
            "greedy_asr_intact": self.greedy_asr_intact,
            "benign_kl": self.benign_kl,
            "control_task_degradation": self.control_task_degradation,
            "extras": self.extras,
        }


def assert_donor_frozen_before_recipient(
    donor_ranking: FrozenRanking,
    *,
    recipient_eval_split: str,
) -> None:
    """Safety check used by tests and runner."""
    donor_ranking.assert_frozen()
    if recipient_eval_split == donor_ranking.ranking_split == "held_out":
        raise RuntimeError("Invalid shared held_out ranking/eval configuration")
    if donor_ranking.ranking_split == "held_out":
        raise RuntimeError("Donor ranking freeze used held_out — forbidden")


def correlated_pair_note(pairs: Iterable[TransferPair]) -> str:
    return (
        f"Reporting {len(list(pairs))} directed transfers among seeds {list(SEEDS)}. "
        "Pairs that share a seed are statistically dependent; do not treat the six "
        "directed comparisons as independent replicates. This three-seed pilot is "
        "feasibility/informativeness oriented, not a definitive statistical test."
    )
