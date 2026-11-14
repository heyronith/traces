"""Versioned, resumable pilot execution checkpoints.

Prevents duplicate discovery rescans and silent frozen-ranking mutation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from traces.pilot.ranking import FrozenRanking, load_frozen_ranking, save_frozen_ranking
from traces.reproducibility import sha256_json

STORE_VERSION = "pilot_exec_v1"


@dataclass
class SeedDiscoveryRecord:
    seed: int
    ranking_artifact_hash: str
    catalog_hash: str
    dataset_split_hash: str
    discovery_prompt_hash: str
    baseline: dict[str, Any]
    effects_path: str
    ranking_path: str
    complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PilotCheckpointStore:
    root: Path
    version: str = STORE_VERSION
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "execution_state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get("version") != self.version:
                raise RuntimeError(
                    f"Checkpoint store version mismatch: "
                    f"{self.state.get('version')} != {self.version}"
                )
        else:
            self.state = {
                "version": self.version,
                "phase": "init",
                "discovery": {},
                "transfer": {},
                "gates": {},
                "failures": [],
            }
            self._flush()

    def _flush(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n")

    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase
        self._flush()

    def record_failure(self, where: str, message: str) -> None:
        self.state.setdefault("failures", []).append({"where": where, "message": message})
        self._flush()

    def discovery_complete(self, seed: int) -> bool:
        rec = self.state.get("discovery", {}).get(str(seed))
        return bool(rec and rec.get("complete"))

    def save_discovery(
        self,
        *,
        seed: int,
        ranking: FrozenRanking,
        effects: dict[str, dict[str, float]],
        catalog_hash: str,
        dataset_split_hash: str,
        discovery_prompt_hash: str,
        baseline: dict[str, Any],
    ) -> SeedDiscoveryRecord:
        effects_path = self.root / f"per_instruction_effects_seed{seed}.json"
        ranking_path = self.root / f"ranking_seed{seed}.json"
        artifact_hash = ranking.artifact_hash()

        if self.discovery_complete(seed):
            existing = self.state["discovery"][str(seed)]
            if existing["ranking_artifact_hash"] != artifact_hash:
                raise RuntimeError(
                    f"Refusing silent ranking mutation for seed {seed}: "
                    f"stored={existing['ranking_artifact_hash']} new={artifact_hash}"
                )
            # Idempotent: do not rewrite effects/ranking with different content
            return SeedDiscoveryRecord(**{**existing, "seed": seed})

        # Refuse overwrite of an on-disk ranking with a different hash
        if ranking_path.exists():
            prev = load_frozen_ranking(ranking_path)
            if prev.artifact_hash() != artifact_hash:
                raise RuntimeError(
                    f"On-disk ranking for seed {seed} differs from new artifact; "
                    "refusing silent mutation"
                )

        effects_path.write_text(json.dumps(effects, indent=2) + "\n")
        save_frozen_ranking(ranking, ranking_path)
        rec = SeedDiscoveryRecord(
            seed=seed,
            ranking_artifact_hash=artifact_hash,
            catalog_hash=catalog_hash,
            dataset_split_hash=dataset_split_hash,
            discovery_prompt_hash=discovery_prompt_hash,
            baseline=baseline,
            effects_path=str(effects_path),
            ranking_path=str(ranking_path),
            complete=True,
        )
        self.state.setdefault("discovery", {})[str(seed)] = rec.to_dict()
        self._flush()
        return rec

    def load_discovery_effects(self, seed: int) -> dict[str, dict[str, float]]:
        rec = self.state["discovery"][str(seed)]
        return json.loads(Path(rec["effects_path"]).read_text())

    def load_discovery_ranking(self, seed: int) -> FrozenRanking:
        rec = self.state["discovery"][str(seed)]
        ranking = load_frozen_ranking(rec["ranking_path"])
        if ranking.artifact_hash() != rec["ranking_artifact_hash"]:
            raise RuntimeError(
                f"Loaded ranking hash mismatch for seed {seed} — possible mutation"
            )
        return ranking

    def mark_transfer_condition(
        self, key: str, payload_hash: str, *, split: str
    ) -> bool:
        """Return True if already measured (skip duplicate). Record otherwise."""
        bucket = self.state.setdefault("transfer", {}).setdefault(split, {})
        if key in bucket:
            if bucket[key] != payload_hash:
                raise RuntimeError(
                    f"Duplicate transfer key {key} with different payload hash"
                )
            return True  # already done
        bucket[key] = payload_hash
        self._flush()
        return False

    def save_gates(self, gates: dict[str, Any]) -> None:
        self.state["gates"] = gates
        self._flush()

    def save_final_results(self, results: dict[str, Any]) -> Path:
        path = self.root / "pilot_results.json"
        path.write_text(json.dumps(results, indent=2) + "\n")
        self.state["phase"] = "complete"
        self.state["results_path"] = str(path)
        self.state["results_hash"] = sha256_json(results)
        self._flush()
        return path
