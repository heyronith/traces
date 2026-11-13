"""Component ranking from continuous log-prob contrasts (CPU-safe freeze logic).

Discovery uses teacher-forced target log-prob contrasts on discovery prompts.
Greedy ASR is reserved for behavioral confirmation, not primary ranking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from traces.pilot.components import ComponentRef
from traces.pilot.dataset import forbid_eval_split_for_discovery
from traces.reproducibility import sha256_json


@dataclass
class ComponentScore:
    component_id: str
    score: float
    # Higher score = larger drop in trigger-vs-control contrast when ablated,
    # or larger reduction in trigger target logprob (sign convention recorded).
    n_prompts: int
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrozenRanking:
    """Immutable donor ranking artifact.

    Once frozen, recipient evaluation must load this object and must not
    recompute ranking on held_out prompts.
    """

    donor_seed: int
    ranking_split: str
    score_definition: str
    component_order: list[str]
    scores: list[ComponentScore]
    frozen: bool = True
    catalog_hash: str | None = None
    prompt_set_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ranking_split == "held_out":
            raise RuntimeError("Cannot freeze a ranking derived from held_out")
        forbid_eval_split_for_discovery(self.ranking_split)  # type: ignore[arg-type]
        if not self.frozen:
            raise RuntimeError("FrozenRanking.frozen must be True")

    def to_dict(self) -> dict[str, Any]:
        return {
            "donor_seed": self.donor_seed,
            "ranking_split": self.ranking_split,
            "score_definition": self.score_definition,
            "component_order": list(self.component_order),
            "scores": [s.to_dict() for s in self.scores],
            "frozen": True,
            "catalog_hash": self.catalog_hash,
            "prompt_set_hash": self.prompt_set_hash,
            "metadata": self.metadata,
            "artifact_hash": self.artifact_hash(),
        }

    def artifact_hash(self) -> str:
        payload = {
            "donor_seed": self.donor_seed,
            "ranking_split": self.ranking_split,
            "score_definition": self.score_definition,
            "component_order": self.component_order,
            "scores": [s.to_dict() for s in self.scores],
            "catalog_hash": self.catalog_hash,
            "prompt_set_hash": self.prompt_set_hash,
        }
        return sha256_json(payload)

    def top_k(self, k: int) -> list[str]:
        if k < 1:
            raise ValueError("k must be >= 1")
        return list(self.component_order[:k])

    def assert_frozen(self) -> None:
        if not self.frozen:
            raise RuntimeError("Ranking is not frozen")


SCORE_DEFINITION = (
    "mean_over_discovery_prompts[ "
    "intact_trigger_target_logprob - ablated_trigger_target_logprob "
    "]  (higher => more causal importance for trigger payload likelihood). "
    "Primary discovery metric is continuous logprob; greedy ASR is confirmation-only."
)


def rank_components_from_scores(
    scores: dict[str, float],
    *,
    donor_seed: int,
    ranking_split: str,
    n_prompts: int,
    catalog: Sequence[ComponentRef] | None = None,
    prompt_set_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> FrozenRanking:
    """Build a frozen ranking from a component_id → score map (CPU)."""
    forbid_eval_split_for_discovery(ranking_split)  # type: ignore[arg-type]
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    score_rows = [
        ComponentScore(component_id=cid, score=float(sc), n_prompts=n_prompts)
        for cid, sc in ordered
    ]
    catalog_hash = None
    if catalog is not None:
        catalog_hash = sha256_json([c.to_dict() for c in catalog])
    return FrozenRanking(
        donor_seed=donor_seed,
        ranking_split=ranking_split,
        score_definition=SCORE_DEFINITION,
        component_order=[cid for cid, _ in ordered],
        scores=score_rows,
        frozen=True,
        catalog_hash=catalog_hash,
        prompt_set_hash=prompt_set_hash,
        metadata=metadata or {},
    )


def save_frozen_ranking(ranking: FrozenRanking, path: Path | str) -> str:
    ranking.assert_frozen()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = ranking.to_dict()
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload["artifact_hash"]


def load_frozen_ranking(path: Path | str) -> FrozenRanking:
    raw = json.loads(Path(path).read_text())
    if not raw.get("frozen", False):
        raise RuntimeError("Refusing to load non-frozen ranking artifact")
    if raw.get("ranking_split") == "held_out":
        raise RuntimeError("Corrupt artifact: ranking_split=held_out")
    scores = [
        ComponentScore(
            component_id=s["component_id"],
            score=float(s["score"]),
            n_prompts=int(s["n_prompts"]),
            extras=s.get("extras") or {},
        )
        for s in raw["scores"]
    ]
    ranking = FrozenRanking(
        donor_seed=int(raw["donor_seed"]),
        ranking_split=str(raw["ranking_split"]),
        score_definition=str(raw["score_definition"]),
        component_order=list(raw["component_order"]),
        scores=scores,
        frozen=True,
        catalog_hash=raw.get("catalog_hash"),
        prompt_set_hash=raw.get("prompt_set_hash"),
        metadata=raw.get("metadata") or {},
    )
    # Integrity: recompute hash
    if ranking.artifact_hash() != raw.get("artifact_hash"):
        raise RuntimeError("Frozen ranking artifact_hash mismatch")
    return ranking


def split_discovery_for_reliability(
    instruction_ids: Sequence[str],
    *,
    seed: int = 0,
    n_folds: int = 2,
) -> list[list[str]]:
    """Partition discovery IDs into disjoint subsets for within-model reliability."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    ids = list(instruction_ids)
    # Deterministic LCG shuffle
    state = seed

    def rnd(n: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) % (2**31)
        return state % n

    for i in range(len(ids) - 1, 0, -1):
        j = rnd(i + 1)
        ids[i], ids[j] = ids[j], ids[i]
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, iid in enumerate(ids):
        folds[i % n_folds].append(iid)
    return folds


def ranking_reliability_overlap(
    rank_a: Sequence[str],
    rank_b: Sequence[str],
    k: int,
) -> dict[str, float]:
    """Jaccard / top-k overlap between two rankings (descriptive, not a proof)."""
    a = set(rank_a[:k])
    b = set(rank_b[:k])
    inter = len(a & b)
    union = len(a | b) or 1
    return {
        "k": float(k),
        "jaccard": inter / union,
        "overlap_count": float(inter),
        "note": (
            "Overlap of component indices is not evidence of identical mechanisms."
        ),
    }
