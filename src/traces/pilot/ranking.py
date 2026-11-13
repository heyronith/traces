"""Component ranking with freeze integrity (discovery-only).

Validation may select thresholds / damage matches but must not create rankings.
Held-out must never influence ranking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from traces.pilot.components import ComponentRef
from traces.pilot.dataset import forbid_eval_split_for_discovery
from traces.pilot.metrics_defs import SCORING_DEFINITION_ID, INTERVENTION_SEMANTICS
from traces.reproducibility import sha256_json

ALLOWED_RANKING_SPLITS = frozenset({"discovery"})


@dataclass(frozen=True)
class ComponentScore:
    component_id: str
    score: float  # D contribution / mean D for this component; higher => more suppression
    n_prompts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "score": self.score,
            "n_prompts": self.n_prompts,
        }


@dataclass(frozen=True)
class FrozenRanking:
    """Immutable donor ranking artifact (tuple-backed; mutation fails)."""

    donor_seed: int
    ranking_split: str
    score_definition: str
    scoring_definition_id: str
    intervention_semantics: str
    component_order: tuple[str, ...]
    scores: tuple[ComponentScore, ...]
    frozen: bool
    catalog_hash: str
    prompt_set_hash: str
    dataset_split_hash: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.ranking_split not in ALLOWED_RANKING_SPLITS:
            raise RuntimeError(
                f"Rankings may only originate from discovery; got {self.ranking_split!r}"
            )
        forbid_eval_split_for_discovery(self.ranking_split)  # type: ignore[arg-type]
        if not self.frozen:
            raise RuntimeError("FrozenRanking.frozen must be True")
        if self.scoring_definition_id != SCORING_DEFINITION_ID:
            raise RuntimeError("Unexpected scoring_definition_id")
        if len(self.component_order) != len(self.scores):
            raise RuntimeError("component_order / scores length mismatch")
        if len(set(self.component_order)) != len(self.component_order):
            raise RuntimeError("Duplicate components in ranking")

    def to_dict(self) -> dict[str, Any]:
        return {
            "donor_seed": self.donor_seed,
            "ranking_split": self.ranking_split,
            "score_definition": self.score_definition,
            "scoring_definition_id": self.scoring_definition_id,
            "intervention_semantics": self.intervention_semantics,
            "component_order": list(self.component_order),
            "scores": [s.to_dict() for s in self.scores],
            "frozen": True,
            "catalog_hash": self.catalog_hash,
            "prompt_set_hash": self.prompt_set_hash,
            "dataset_split_hash": self.dataset_split_hash,
            "metadata": dict(self.metadata),
            "artifact_hash": self.artifact_hash(),
        }

    def artifact_hash(self) -> str:
        payload = {
            "donor_seed": self.donor_seed,
            "ranking_split": self.ranking_split,
            "score_definition": self.score_definition,
            "scoring_definition_id": self.scoring_definition_id,
            "intervention_semantics": self.intervention_semantics,
            "component_order": list(self.component_order),
            "scores": [s.to_dict() for s in self.scores],
            "catalog_hash": self.catalog_hash,
            "prompt_set_hash": self.prompt_set_hash,
            "dataset_split_hash": self.dataset_split_hash,
        }
        return sha256_json(payload)

    def top_k(self, k: int) -> tuple[str, ...]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > len(self.component_order):
            raise ValueError(f"Invalid k={k} > catalog ranking size {len(self.component_order)}")
        return self.component_order[:k]

    def assert_frozen(self) -> None:
        if not self.frozen:
            raise RuntimeError("Ranking is not frozen")

    def refuse_mutation(self) -> None:
        """Documented immutability: component_order/scores are tuples."""
        if not isinstance(self.component_order, tuple):
            raise RuntimeError("component_order must be tuple")
        if not isinstance(self.scores, tuple):
            raise RuntimeError("scores must be tuple")


SCORE_DEFINITION = (
    "D_r({c}) averaged over discovery triggered instructions: "
    "mean_logP_per_target_token(intact) - mean_logP_per_target_token(ablate {c}). "
    "Positive => suppresses target likelihood. Secondary trigger-control contrast "
    "must not substitute for D."
)


def rank_components_from_scores(
    scores: dict[str, float],
    *,
    donor_seed: int,
    ranking_split: str,
    n_prompts: int,
    catalog: Sequence[ComponentRef],
    prompt_set_hash: str,
    dataset_split_hash: str,
    metadata: dict[str, Any] | None = None,
) -> FrozenRanking:
    if ranking_split != "discovery":
        raise RuntimeError("Component rankings can originate only from discovery data")
    catalog_ids = {c.component_id for c in catalog}
    missing = set(scores) - catalog_ids
    if missing:
        raise RuntimeError(f"Scores reference unknown components: {sorted(missing)[:5]}")
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    # Ensure full catalog coverage: missing components get score -inf / last
    present = {cid for cid, _ in ordered}
    for c in catalog:
        if c.component_id not in present:
            ordered.append((c.component_id, float("-inf")))
    ordered = sorted(ordered, key=lambda kv: (-kv[1], kv[0]))
    score_rows = tuple(
        ComponentScore(component_id=cid, score=float(sc), n_prompts=n_prompts)
        for cid, sc in ordered
    )
    catalog_hash = sha256_json([c.to_dict() for c in catalog])
    return FrozenRanking(
        donor_seed=donor_seed,
        ranking_split="discovery",
        score_definition=SCORE_DEFINITION,
        scoring_definition_id=SCORING_DEFINITION_ID,
        intervention_semantics=INTERVENTION_SEMANTICS,
        component_order=tuple(cid for cid, _ in ordered),
        scores=score_rows,
        frozen=True,
        catalog_hash=catalog_hash,
        prompt_set_hash=prompt_set_hash,
        dataset_split_hash=dataset_split_hash,
        metadata=MappingProxyType(dict(metadata or {})),
    )


def save_frozen_ranking(ranking: FrozenRanking, path: Path | str) -> str:
    ranking.assert_frozen()
    ranking.refuse_mutation()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = ranking.to_dict()
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload["artifact_hash"]


def load_frozen_ranking(path: Path | str) -> FrozenRanking:
    raw = json.loads(Path(path).read_text())
    if not raw.get("frozen", False):
        raise RuntimeError("Refusing to load non-frozen ranking artifact")
    if raw.get("ranking_split") != "discovery":
        raise RuntimeError("Corrupt/forbidden: ranking_split must be discovery")
    scores = tuple(
        ComponentScore(
            component_id=s["component_id"],
            score=float(s["score"]),
            n_prompts=int(s["n_prompts"]),
        )
        for s in raw["scores"]
    )
    ranking = FrozenRanking(
        donor_seed=int(raw["donor_seed"]),
        ranking_split="discovery",
        score_definition=str(raw["score_definition"]),
        scoring_definition_id=str(raw.get("scoring_definition_id", SCORING_DEFINITION_ID)),
        intervention_semantics=str(
            raw.get("intervention_semantics", INTERVENTION_SEMANTICS)
        ),
        component_order=tuple(raw["component_order"]),
        scores=scores,
        frozen=True,
        catalog_hash=str(raw["catalog_hash"]),
        prompt_set_hash=str(raw["prompt_set_hash"]),
        dataset_split_hash=str(raw.get("dataset_split_hash", "")),
        metadata=MappingProxyType(dict(raw.get("metadata") or {})),
    )
    if ranking.artifact_hash() != raw.get("artifact_hash"):
        raise RuntimeError("Frozen ranking artifact_hash mismatch")
    ranking.refuse_mutation()
    return ranking


def split_discovery_for_reliability(
    instruction_ids: Sequence[str],
    *,
    seed: int = 0,
    n_folds: int = 2,
) -> list[list[str]]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    ids = list(instruction_ids)
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
) -> dict[str, float | str]:
    a = set(rank_a[:k])
    b = set(rank_b[:k])
    inter = len(a & b)
    union = len(a | b) or 1
    return {
        "k": float(k),
        "jaccard": inter / union,
        "overlap_count": float(inter),
        "note": "Index overlap is not evidence of identical mechanisms.",
    }


def freeze_ranking_from_per_instruction_effects(
    per_instruction: Mapping[str, Mapping[str, float]],
    *,
    instruction_ids: Sequence[str],
    donor_seed: int,
    catalog: Sequence[ComponentRef],
    prompt_set_hash: str,
    dataset_split_hash: str,
) -> FrozenRanking:
    """Aggregate per-instruction D effects without rescanning.

    per_instruction[component_id][instruction_id] = D on that instruction
    (intact_logprob - ablated_logprob for that prompt).
    """
    ids = set(instruction_ids)
    scores: dict[str, float] = {}
    for cid, by_inst in per_instruction.items():
        vals = [float(by_inst[i]) for i in ids if i in by_inst]
        if not vals:
            scores[cid] = float("-inf")
        else:
            scores[cid] = sum(vals) / len(vals)
    return rank_components_from_scores(
        scores,
        donor_seed=donor_seed,
        ranking_split="discovery",
        n_prompts=len(ids),
        catalog=catalog,
        prompt_set_hash=prompt_set_hash,
        dataset_split_hash=dataset_split_hash,
        metadata={"aggregation": "mean_D_over_instruction_ids"},
    )
