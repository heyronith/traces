"""Versioned, crash-safe pilot execution checkpoints (v2).

Transfer conditions are marked complete only after successful measurement and
durable row persistence. Completed rows reload into analysis on resume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from traces.pilot.ranking import FrozenRanking, load_frozen_ranking, save_frozen_ranking
from traces.reproducibility import sha256_json

STORE_VERSION = "pilot_exec_v2"


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
    provenance: dict[str, str] = field(default_factory=dict)
    complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PilotCheckpointStore:
    def __init__(self, root: Path | str, *, version: str = STORE_VERSION):
        self.root = Path(root)
        self.version = version
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "transfer_rows").mkdir(parents=True, exist_ok=True)
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
                "identity": None,
                "phase": "init",
                "discovery": {},
                "transfer": {"validation": {}, "held_out": {}},
                "gates": {},
                "native_validation": {},
                "failures": [],
                "units_completed": [],
            }
            self._flush()

    def _flush(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(self.state_path)

    def bind_run_identity(self, identity: dict[str, Any]) -> None:
        """Reject changed config / dataset / catalog / ranking-binding identity."""
        # Drop non-stable volatile keys
        canonical = {k: identity[k] for k in sorted(identity)}
        existing = self.state.get("identity")
        if existing is None:
            self.state["identity"] = canonical
            self._flush()
            return
        if existing != canonical:
            raise RuntimeError(
                "Checkpoint identity mismatch — refusing resume against a different "
                f"configuration/dataset/catalog/scoring binding.\n"
                f"stored={existing}\nnew={canonical}"
            )

    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase
        self._flush()

    def record_failure(self, where: str, message: str) -> None:
        self.state.setdefault("failures", []).append({"where": where, "message": message})
        self._flush()

    def mark_unit_complete(self, unit_id: str) -> None:
        done = self.state.setdefault("units_completed", [])
        if unit_id not in done:
            done.append(unit_id)
            self._flush()

    def unit_complete(self, unit_id: str) -> bool:
        return unit_id in self.state.get("units_completed", [])

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
        provenance: dict[str, str] | None = None,
    ) -> SeedDiscoveryRecord:
        effects_path = self.root / f"per_instruction_effects_seed{seed}.json"
        ranking_path = self.root / f"ranking_seed{seed}.json"
        artifact_hash = ranking.artifact_hash()
        prov = dict(provenance or {})

        if self.discovery_complete(seed):
            existing = self.state["discovery"][str(seed)]
            if existing["ranking_artifact_hash"] != artifact_hash:
                raise RuntimeError(
                    f"Refusing silent ranking mutation for seed {seed}: "
                    f"stored={existing['ranking_artifact_hash']} new={artifact_hash}"
                )
            return SeedDiscoveryRecord(**{**existing, "seed": int(existing["seed"])})

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
            provenance=prov,
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

    def transfer_row_complete(self, key: str, *, split: str) -> bool:
        meta = self.state.get("transfer", {}).get(split, {}).get(key)
        return bool(meta and meta.get("complete") and meta.get("row_path"))

    def get_transfer_row(self, key: str, *, split: str) -> dict[str, Any] | None:
        meta = self.state.get("transfer", {}).get(split, {}).get(key)
        if not meta or not meta.get("complete"):
            return None
        path = Path(meta["row_path"])
        if not path.exists():
            raise RuntimeError(
                f"Transfer row marked complete but file missing: {path}. "
                "Refuse silent re-measure with incomplete durability."
            )
        return json.loads(path.read_text())

    def save_transfer_row(
        self,
        key: str,
        *,
        split: str,
        payload_hash: str,
        row: dict[str, Any],
    ) -> Path:
        """Persist full measurement row, then mark complete (crash-safe order)."""
        bucket = self.state.setdefault("transfer", {}).setdefault(split, {})
        existing = bucket.get(key)
        if existing and existing.get("complete"):
            if existing.get("payload_hash") != payload_hash:
                raise RuntimeError(
                    f"Conflicting transfer measurement for {key}: "
                    f"stored payload_hash={existing.get('payload_hash')} new={payload_hash}"
                )
            # Idempotent: verify row file exists
            prev = self.get_transfer_row(key, split=split)
            if prev is None:
                raise RuntimeError(f"Incomplete durable row for completed key {key}")
            return Path(existing["row_path"])

        safe_name = sha256_json({"key": key, "split": split, "payload_hash": payload_hash})[:24]
        path = self.root / "transfer_rows" / f"{split}_{safe_name}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, indent=2) + "\n")
        tmp.replace(path)
        # Mark complete only after durable write
        bucket[key] = {
            "payload_hash": payload_hash,
            "row_path": str(path),
            "complete": True,
        }
        self._flush()
        return path

    def iter_completed_transfer_rows(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for split, bucket in self.state.get("transfer", {}).items():
            for key, meta in bucket.items():
                if meta.get("complete") and meta.get("row_path"):
                    row = json.loads(Path(meta["row_path"]).read_text())
                    out.append(row)
        return out

    def save_native_validation(self, recipient: int, payload: dict[str, Any]) -> None:
        self.state.setdefault("native_validation", {})[str(recipient)] = payload
        self._flush()

    def get_native_validation(self, recipient: int) -> dict[str, Any] | None:
        return self.state.get("native_validation", {}).get(str(recipient))

    def save_gates(self, gates: dict[str, Any]) -> None:
        self.state["gates"] = gates
        self._flush()

    def save_final_results(self, results: dict[str, Any]) -> Path:
        path = self.root / "pilot_results.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, indent=2) + "\n")
        tmp.replace(path)
        self.state["phase"] = "complete"
        self.state["results_path"] = str(path)
        self.state["results_hash"] = sha256_json(results)
        self._flush()
        return path
