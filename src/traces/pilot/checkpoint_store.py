"""Versioned, crash-safe pilot execution checkpoints (v3).

Transfer conditions are marked complete only after successful measurement and
durable row persistence. Modal volume commits are invoked explicitly at safe
boundaries via ``durable_commit``. Incomplete units are never marked complete.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from traces.pilot.ranking import FrozenRanking, load_frozen_ranking, save_frozen_ranking
from traces.reproducibility import sha256_json

STORE_VERSION = "pilot_exec_v3"

DurableCommitFn = Callable[[str], None]


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
    def __init__(
        self,
        root: Path | str,
        *,
        version: str = STORE_VERSION,
        durable_commit: DurableCommitFn | None = None,
    ):
        self.root = Path(root)
        self.version = version
        self._durable_commit = durable_commit
        self.commit_log: list[str] = []
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
                "discovery_partial": {},
                "transfer": {"validation": {}, "held_out": {}},
                "gates": {},
                "native_validation": {},
                "revision_locks": {},
                "failures": [],
                "units_completed": [],
                "unit_status": {},
            }
            self._flush()

    def _flush(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(self.state_path)

    def commit_durable(self, reason: str) -> None:
        """Flush local state then invoke Modal volume (or test) durable commit."""
        self._flush()
        self.commit_log.append(reason)
        if self._durable_commit is not None:
            self._durable_commit(reason)

    def bind_run_identity(self, identity: dict[str, Any]) -> None:
        """Reject changed config / dataset / catalog / revision-binding identity."""
        canonical = {k: identity[k] for k in sorted(identity)}
        existing = self.state.get("identity")
        if existing is None:
            self.state["identity"] = canonical
            self.commit_durable("bind_run_identity")
            return
        if existing != canonical:
            raise RuntimeError(
                "Checkpoint identity mismatch — refusing resume against a different "
                f"configuration/dataset/catalog/scoring/revision binding.\n"
                f"stored={existing}\nnew={canonical}"
            )

    def lock_seed_revisions(self, seed: int, provenance: dict[str, str]) -> None:
        """Freeze exact base/adapter revisions observed for a seed."""
        key = str(seed)
        locks = self.state.setdefault("revision_locks", {})
        canonical = {
            "base_model_revision": str(provenance["base_model_revision"]),
            "adapter_revision": str(provenance["adapter_revision"]),
            "adapter_subfolder": str(provenance["adapter_subfolder"]),
        }
        if key in locks and locks[key] != canonical:
            raise RuntimeError(
                f"Revision lock conflict for seed {seed}: "
                f"stored={locks[key]} new={canonical}"
            )
        locks[key] = canonical
        self.commit_durable(f"revision_lock_seed_{seed}")

    def assert_live_matches_revision_lock(
        self, seed: int, live: dict[str, str], *, context: str = ""
    ) -> None:
        locks = self.state.get("revision_locks", {})
        key = str(seed)
        if key not in locks:
            raise RuntimeError(
                f"Missing revision lock for seed {seed} ({context}); "
                "discovery must complete and lock revisions before transfer"
            )
        expected = locks[key]
        for field_name in (
            "base_model_revision",
            "adapter_revision",
            "adapter_subfolder",
        ):
            if str(live.get(field_name)) != str(expected.get(field_name)):
                raise RuntimeError(
                    f"Checkpoint revision mismatch for seed {seed} ({context}): "
                    f"field={field_name} frozen={expected.get(field_name)!r} "
                    f"live={live.get(field_name)!r}"
                )

    def assert_frozen_matches_live(
        self,
        ranking: FrozenRanking,
        live: dict[str, str],
        *,
        context: str = "",
    ) -> None:
        """Reject mismatches between frozen discovery provenance and live checkpoint."""
        checks = {
            "base_model_revision": (
                ranking.base_model_revision,
                live.get("base_model_revision"),
            ),
            "adapter_revision": (
                ranking.adapter_revision,
                live.get("adapter_revision"),
            ),
            "adapter_subfolder": (
                ranking.adapter_subfolder,
                live.get("adapter_subfolder"),
            ),
        }
        for field_name, (frozen_v, live_v) in checks.items():
            if str(frozen_v) != str(live_v):
                raise RuntimeError(
                    f"Frozen discovery provenance mismatch vs loaded checkpoint "
                    f"({context}): {field_name} frozen={frozen_v!r} live={live_v!r}"
                )

    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase
        self._flush()

    def record_failure(self, where: str, message: str) -> None:
        self.state.setdefault("failures", []).append({"where": where, "message": message})
        self.commit_durable(f"failure_{where}")

    def set_unit_status(
        self,
        unit_id: str,
        *,
        complete: bool,
        execution_incomplete: bool = False,
        scientific_early_stop: bool = False,
        reason: str | None = None,
        # Deprecated alias: maps to execution_incomplete (not scientific gates)
        stopped_early: bool | None = None,
    ) -> None:
        """Record unit status.

        ``complete=True`` is allowed with ``scientific_early_stop`` (validation-only
        / gate policy). ``execution_incomplete`` must never be combined with complete.
        """
        if stopped_early is not None:
            execution_incomplete = bool(stopped_early) or execution_incomplete
        if complete and execution_incomplete:
            raise RuntimeError(
                f"Refusing to mark unit {unit_id} complete while execution_incomplete=True"
            )
        self.state.setdefault("unit_status", {})[unit_id] = {
            "unit_id": unit_id,
            "complete": bool(complete),
            "execution_incomplete": bool(execution_incomplete),
            "scientific_early_stop": bool(scientific_early_stop),
            # Legacy mirror for older readers: only true for execution incompleteness
            "stopped_early": bool(execution_incomplete),
            "reason": reason,
        }
        done = self.state.setdefault("units_completed", [])
        if complete:
            if unit_id not in done:
                done.append(unit_id)
        else:
            if unit_id in done:
                raise RuntimeError(
                    f"Unit {unit_id} was previously marked complete; "
                    "refusing to downgrade silently"
                )
        self.commit_durable(f"unit_status_{unit_id}")

    def mark_unit_complete(
        self,
        unit_id: str,
        *,
        scientific_early_stop: bool = False,
        reason: str | None = None,
    ) -> None:
        self.set_unit_status(
            unit_id,
            complete=True,
            execution_incomplete=False,
            scientific_early_stop=scientific_early_stop,
            reason=reason,
        )

    def unit_complete(self, unit_id: str) -> bool:
        status = self.state.get("unit_status", {}).get(unit_id)
        if status is not None:
            return bool(status.get("complete")) and not bool(
                status.get("execution_incomplete")
            )
        return unit_id in self.state.get("units_completed", [])

    def get_unit_status(self, unit_id: str) -> dict[str, Any] | None:
        return self.state.get("unit_status", {}).get(unit_id)

    def discovery_complete(self, seed: int) -> bool:
        rec = self.state.get("discovery", {}).get(str(seed))
        return bool(rec and rec.get("complete"))

    def save_discovery_partial(
        self,
        seed: int,
        effects: dict[str, dict[str, float]],
        *,
        components_done: list[str],
        baseline: dict[str, Any] | None = None,
        provenance: dict[str, str] | None = None,
    ) -> Path:
        """Persist mid-discovery progress without marking discovery complete."""
        path = self.root / f"discovery_partial_seed{seed}.json"
        payload = {
            "seed": seed,
            "effects": effects,
            "components_done": list(components_done),
            "baseline": baseline,
            "provenance": dict(provenance or {}),
            "complete": False,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(path)
        self.state.setdefault("discovery_partial", {})[str(seed)] = {
            "path": str(path),
            "n_components_done": len(components_done),
            "complete": False,
        }
        self.commit_durable(f"discovery_partial_seed_{seed}")
        return path

    def load_discovery_partial(self, seed: int) -> dict[str, Any] | None:
        meta = self.state.get("discovery_partial", {}).get(str(seed))
        if not meta:
            # Fallback: file may exist from a committed volume even if state lagged
            path = self.root / f"discovery_partial_seed{seed}.json"
            if not path.exists():
                return None
            return json.loads(path.read_text())
        path = Path(meta["path"])
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def clear_discovery_partial(self, seed: int) -> None:
        meta = self.state.get("discovery_partial", {}).pop(str(seed), None)
        if meta and meta.get("path"):
            p = Path(meta["path"])
            if p.exists():
                p.unlink()
        leftover = self.root / f"discovery_partial_seed{seed}.json"
        if leftover.exists():
            leftover.unlink()
        self._flush()

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
        if prov:
            self.lock_seed_revisions(seed, prov)
        self.clear_discovery_partial(seed)
        self.commit_durable(f"discovery_complete_seed_{seed}")
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
            prev = self.get_transfer_row(key, split=split)
            if prev is None:
                raise RuntimeError(f"Incomplete durable row for completed key {key}")
            return Path(existing["row_path"])

        safe_name = sha256_json(
            {"key": key, "split": split, "payload_hash": payload_hash}
        )[:24]
        path = self.root / "transfer_rows" / f"{split}_{safe_name}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, indent=2) + "\n")
        tmp.replace(path)
        bucket[key] = {
            "payload_hash": payload_hash,
            "row_path": str(path),
            "complete": True,
        }
        self.commit_durable(f"transfer_row_{split}_{safe_name}")
        return path

    def iter_completed_transfer_rows(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for split, bucket in self.state.get("transfer", {}).items():
            for key, meta in bucket.items():
                if meta.get("complete") and meta.get("row_path"):
                    row = json.loads(Path(meta["row_path"]).read_text())
                    out.append(row)
        return out

    def save_damage_match_selection(
        self,
        pair_key: str,
        k: int,
        record: dict[str, Any],
    ) -> None:
        """Freeze validation-selected damage-matched control (never held-out)."""
        bucket = self.state.setdefault("damage_match_selections", {})
        key = f"{pair_key}|k{k}"
        existing = bucket.get(key)
        if existing and existing.get("complete"):
            if existing.get("selection_hash") != record.get("selection_hash"):
                raise RuntimeError(
                    f"Frozen damage-match selection drift for {key}: "
                    f"stored={existing.get('selection_hash')} new={record.get('selection_hash')}"
                )
            if list(existing.get("component_ids") or []) != list(
                record.get("component_ids") or []
            ):
                raise RuntimeError(
                    f"Frozen damage-match component set drift for {key}"
                )
            return
        path = self.root / "damage_match" / f"{key.replace('/', '_')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        payload = {**record, "complete": True, "path": str(path)}
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(path)
        bucket[key] = payload
        self.commit_durable(f"damage_match_{key}")

    @staticmethod
    def damage_match_selection_hash(record: dict[str, Any]) -> str:
        """Canonical hash of a frozen damage-control selection (excludes path metadata)."""
        return sha256_json(
            {
                "kind": record.get("kind"),
                "component_ids": list(record.get("component_ids") or []),
                "target_K": record.get("target_K"),
                "candidate_K": dict(record.get("candidate_K") or {}),
            }
        )

    def verify_damage_match_selection(self, record: dict[str, Any], *, key: str) -> None:
        """Reject tampered or drifted frozen damage-control payloads on load."""
        expected = record.get("selection_hash")
        if not expected:
            raise RuntimeError(
                f"Frozen damage-match selection missing selection_hash for {key}"
            )
        recomputed = self.damage_match_selection_hash(record)
        if recomputed != expected:
            raise RuntimeError(
                f"Frozen damage-match selection hash mismatch for {key}: "
                f"stored={expected} recomputed={recomputed} "
                "(file may have been tampered)"
            )

    def load_damage_match_selection(
        self, pair_key: str, k: int
    ) -> dict[str, Any] | None:
        key = f"{pair_key}|k{k}"
        meta = self.state.get("damage_match_selections", {}).get(key)
        if not meta or not meta.get("complete"):
            return None
        path = Path(meta.get("path") or "")
        if path.exists():
            record = json.loads(path.read_text())
        else:
            record = dict(meta)
        self.verify_damage_match_selection(record, key=key)
        return record

    def save_native_validation(self, recipient: int, payload: dict[str, Any]) -> None:
        self.state.setdefault("native_validation", {})[str(recipient)] = payload
        self.commit_durable(f"native_validation_{recipient}")

    def get_native_validation(self, recipient: int) -> dict[str, Any] | None:
        return self.state.get("native_validation", {}).get(str(recipient))

    def save_gates(self, gates: dict[str, Any]) -> None:
        self.state["gates"] = gates
        self.commit_durable("gates")

    def save_final_results(
        self, results: dict[str, Any], *, phase: str | None = None
    ) -> Path:
        path = self.root / "pilot_results.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, indent=2) + "\n")
        tmp.replace(path)
        self.state["phase"] = phase or (
            "stopped_early"
            if results.get("stopped_early")
            else "complete"
        )
        self.state["results_path"] = str(path)
        self.state["results_hash"] = sha256_json(results)
        self.commit_durable("final_results")
        return path
