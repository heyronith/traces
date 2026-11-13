"""Complete Stage 1 pilot evaluation engine (disabled by default).

Orchestrates discovery → reliability → frozen rankings → native/transfer/controls
→ held-out reporting. Real GPU execution is separately gated from synthetic CPU tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from traces.pilot.analysis import analyze_per_pair_results, analysis_plan_stub
from traces.pilot.components import ComponentRef, build_full_l1523_catalog
from traces.pilot.controls import (
    DAMAGE_MATCH_N_CANDIDATES,
    generate_layer_type_matched_candidates,
    layer_type_matched_control,
    random_control_set,
    select_benign_damage_matched_control,
)
from traces.pilot.dataset import (
    get_split_instructions,
    load_manifest,
    split_prompt_hash,
    verify_manifest_hashes,
)
from traces.pilot.gates import (
    NATIVE_VALIDATION_D_MIN,
    RELIABILITY_JACCARD_K,
    RELIABILITY_JACCARD_MIN,
    combine_gates,
    evaluate_discovery_reliability_gate,
    evaluate_native_validation_gate,
)
from traces.pilot.metrics_defs import metric_definitions
from traces.pilot.ranking import (
    FrozenRanking,
    freeze_ranking_from_per_instruction_effects,
    ranking_reliability_overlap,
    save_frozen_ranking,
    split_discovery_for_reliability,
)
from traces.pilot.runner import assert_pilot_disabled_or_authorized, load_pilot_config
from traces.pilot.schedule import enumerate_engine_condition_schedule
from traces.pilot.transfer import (
    DEFAULT_K_BUDGETS,
    SEEDS,
    TransferIntegrityContext,
    TransferResultRow,
    all_directed_seed_pairs,
    build_transfer_conditions,
    correlated_pair_note,
)
from traces.reproducibility import sha256_json

ExecutionMode = Literal["synthetic_test", "real"]

# Re-export for callers that imported schedule helpers from engine.
__all__ = [
    "PilotScorer",
    "PilotEngineResult",
    "enumerate_engine_condition_schedule",
    "run_pilot_pipeline",
    "run_real_pilot_pipeline",
]


class PilotScorer(Protocol):
    """Scoring backend: real model or deterministic synthetic stub."""

    def score_D_single(
        self, seed: int, component_id: str, instruction_id: str, split: str
    ) -> float:
        """Per-instruction D for ablating one component (positive => suppresses)."""

    def score_D_set(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> float:
        """Mean D over instructions for joint ablation set S."""

    def score_K_set(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> float:
        """Benign KL K_r(S) on control-tagged inputs for the split."""

    def score_asr(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> dict[str, float]:
        """Return greedy_asr, full_payload_asr, intact_asr."""


@dataclass
class PilotEngineResult:
    per_seed: dict[str, Any] = field(default_factory=dict)
    per_pair: dict[str, Any] = field(default_factory=dict)
    per_condition: list[dict[str, Any]] = field(default_factory=list)
    reliability: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    metric_definitions: dict[str, Any] = field(default_factory=dict)
    schedule_counts: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    executed: bool = False
    stopped_early: bool = False
    stop_reason: str | None = None
    execution_mode: str = "synthetic_test"

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_seed": self.per_seed,
            "per_pair": self.per_pair,
            "per_condition": self.per_condition,
            "reliability": self.reliability,
            "gates": self.gates,
            "analysis": self.analysis,
            "metric_definitions": self.metric_definitions,
            "schedule_counts": self.schedule_counts,
            "runtime_seconds": self.runtime_seconds,
            "executed": self.executed,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "execution_mode": self.execution_mode,
            "pair_dependence_note": correlated_pair_note(),
        }


def _instruction_ids(split: str) -> list[str]:
    return [i.instruction_id for i in get_split_instructions(split)]  # type: ignore[arg-type]


def _scorer_provenance(scorer: Any, seed: int) -> dict[str, str]:
    if hasattr(scorer, "provenance_for_seed"):
        try:
            p = scorer.provenance_for_seed(seed)
            return {
                "base_model_revision": str(p.get("base_model_revision") or "unspecified"),
                "adapter_subfolder": str(p.get("adapter_subfolder") or "unspecified"),
                "adapter_revision": str(p.get("adapter_revision") or "unspecified"),
            }
        except Exception:
            pass
    # Synthetic / stub defaults — still bound into ranking hash
    return {
        "base_model_revision": "synthetic",
        "adapter_subfolder": f"l1523/seed{seed}",
        "adapter_revision": "synthetic",
    }


def assert_synthetic_execution_allowed(scorer: Any) -> None:
    """Synthetic test path must not silently accept a real scorer bypass."""
    cls_name = type(scorer).__name__
    if cls_name == "RealGemmaPilotScorer":
        raise RuntimeError(
            "Refusing synthetic_test execution mode with RealGemmaPilotScorer. "
            "Use run_real_pilot_pipeline (currently disabled/unauthorized)."
        )


def run_real_pilot_pipeline(
    scorer: PilotScorer,
    **kwargs: Any,
) -> PilotEngineResult:
    """Real-model entrypoint. Requires authorized config; currently refuses."""
    return run_pilot_pipeline(scorer, execution_mode="real", **kwargs)


def run_pilot_pipeline(
    scorer: PilotScorer,
    *,
    seeds: Sequence[int] = SEEDS,
    k_budgets: Sequence[int] = DEFAULT_K_BUDGETS,
    catalog: Sequence[ComponentRef] | None = None,
    output_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    execution_mode: ExecutionMode = "real",
    # Deprecated: only honored for synthetic_test; cannot unlock real execution.
    allow_execution: bool | None = None,
) -> PilotEngineResult:
    """Full workflow.

    - execution_mode='synthetic_test': CPU stub scorers for unit/E2E tests only.
    - execution_mode='real': requires authorized pilot.yaml; currently always blocked.
    """
    cfg = load_pilot_config(config_path)

    if allow_execution is True and execution_mode == "real":
        raise RuntimeError(
            "allow_execution=True cannot unlock real-model pilot execution. "
            "Use an authorized configs/pilot.yaml via the real entrypoint after review."
        )

    if execution_mode == "real":
        assert_pilot_disabled_or_authorized(cfg)
        raise RuntimeError(
            "Real pilot execution blocked (disabled/unauthorized). "
            "Stage 1 remains gated by independent review."
        )

    # synthetic_test
    assert_synthetic_execution_allowed(scorer)
    if allow_execution is False:
        raise RuntimeError("Pilot synthetic test execution explicitly disallowed")

    t0 = time.perf_counter()
    cat = list(catalog or build_full_l1523_catalog())
    catalog_hash = sha256_json([c.to_dict() for c in cat])

    hash_checks = verify_manifest_hashes()
    if not all(hash_checks.values()):
        raise RuntimeError(f"Frozen dataset manifest hash mismatch: {hash_checks}")
    manifest = load_manifest()
    dataset_split_hash = manifest["hashes"]["all_splits"]
    discovery_hash = split_prompt_hash("discovery")
    validation_hash = split_prompt_hash("validation")
    held_out_hash = split_prompt_hash("held_out")

    disc_ids = _instruction_ids("discovery")
    val_ids = _instruction_ids("validation")
    held_ids = _instruction_ids("held_out")

    schedule = enumerate_engine_condition_schedule(
        seeds=seeds,
        k_budgets=k_budgets,
        n_discovery=len(disc_ids),
        n_validation=len(val_ids),
        n_held_out=len(held_ids),
        n_components=len(cat),
    )

    result = PilotEngineResult(
        metric_definitions=metric_definitions(),
        executed=True,
        execution_mode=execution_mode,
        schedule_counts=schedule,
    )
    frozen: dict[int, FrozenRanking] = {}
    per_inst_effects: dict[int, dict[str, dict[str, float]]] = {}

    # --- Per-seed discovery screening (save per-instruction effects) ---
    for seed in seeds:
        effects: dict[str, dict[str, float]] = {c.component_id: {} for c in cat}
        for c in cat:
            for iid in disc_ids:
                effects[c.component_id][iid] = float(
                    scorer.score_D_single(seed, c.component_id, iid, "discovery")
                )
        per_inst_effects[seed] = effects
        prov = _scorer_provenance(scorer, seed)

        folds = split_discovery_for_reliability(disc_ids, seed=seed, n_folds=2)
        fold_orders = []
        for fold in folds:
            fr = freeze_ranking_from_per_instruction_effects(
                effects,
                instruction_ids=fold,
                donor_seed=seed,
                catalog=cat,
                prompt_set_hash=discovery_hash,
                dataset_split_hash=dataset_split_hash,
                **prov,
            )
            fold_orders.append(list(fr.component_order))
        rel = {
            "folds": folds,
            "overlaps": [
                ranking_reliability_overlap(fold_orders[0], fold_orders[1], k)
                for k in k_budgets
            ],
        }
        result.reliability[str(seed)] = rel

        full = freeze_ranking_from_per_instruction_effects(
            effects,
            instruction_ids=disc_ids,
            donor_seed=seed,
            catalog=cat,
            prompt_set_hash=discovery_hash,
            dataset_split_hash=dataset_split_hash,
            **prov,
        )
        frozen[seed] = full
        result.per_seed[str(seed)] = {
            "ranking_artifact_hash": full.artifact_hash(),
            "top10": list(full.top_k(min(10, len(full.component_order)))),
            "n_components_scored": len(cat),
            "discovery_prompt_hash": discovery_hash,
            "provenance": prov,
        }

    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        for seed, fr in frozen.items():
            save_frozen_ranking(fr, out / f"ranking_seed{seed}.json")
            (out / f"per_instruction_effects_seed{seed}.json").write_text(
                json.dumps(per_inst_effects[seed], indent=2) + "\n"
            )

    # --- Scientific stop: discovery reliability ---
    rel_gate = evaluate_discovery_reliability_gate(
        result.reliability,
        k=RELIABILITY_JACCARD_K,
        min_jaccard=RELIABILITY_JACCARD_MIN,
    )
    gates = [rel_gate]
    run_held_out = rel_gate.passed
    if not rel_gate.passed:
        result.stopped_early = True
        result.stop_reason = rel_gate.reason

    # --- Transfer + controls for each directed pair (validation always; held_out if gate) ---
    native_val_D: dict[str, float | None] = {}
    for pair in all_directed_seed_pairs(seeds):
        donor = frozen[pair.donor_seed]
        recip = frozen[pair.recipient_seed]
        donor_prov = _scorer_provenance(scorer, pair.donor_seed)
        recip_prov = _scorer_provenance(scorer, pair.recipient_seed)
        # Validate provenance identities before recipient evaluation
        if donor.prompt_set_hash != discovery_hash:
            raise RuntimeError("Donor ranking prompt_set_hash mismatch vs frozen discovery")
        if donor.catalog_hash != catalog_hash:
            raise RuntimeError("Donor ranking catalog_hash mismatch")
        if recip.adapter_subfolder != recip_prov["adapter_subfolder"]:
            raise RuntimeError("Recipient ranking adapter_subfolder mismatch vs live scorer")
        if donor.adapter_subfolder != donor_prov["adapter_subfolder"]:
            raise RuntimeError("Donor ranking adapter_subfolder mismatch vs live scorer")

        ctx = TransferIntegrityContext(
            donor_seed=pair.donor_seed,
            recipient_seed=pair.recipient_seed,
            catalog_hash=catalog_hash,
            dataset_split_hash=dataset_split_hash,
            base_model_revision=recip_prov["base_model_revision"],
            adapter_revision=recip_prov["adapter_revision"],
            donor_adapter_revision=donor_prov["adapter_revision"],
            recipient_adapter_revision=recip_prov["adapter_revision"],
            expected_prompt_hash=discovery_hash,
            donor_adapter_subfolder=donor_prov["adapter_subfolder"],
            recipient_adapter_subfolder=recip_prov["adapter_subfolder"],
        )
        pair_rows: list[dict[str, Any]] = []
        for k in k_budgets:
            donor_top = list(donor.top_k(k))
            rnd = random_control_set(cat, k, seed=1000 * pair.donor_seed + k, exclude=donor_top)
            lt = layer_type_matched_control(cat, donor_top, seed=2000 * pair.donor_seed + k)

            target_K = scorer.score_K_set(
                pair.recipient_seed, donor_top, val_ids, "validation"
            )
            candidates = generate_layer_type_matched_candidates(
                cat, donor_top, n_candidates=DAMAGE_MATCH_N_CANDIDATES, seed=3000 + k
            )
            cand_K = {
                c.name: scorer.score_K_set(
                    pair.recipient_seed, list(c.component_ids), val_ids, "validation"
                )
                for c in candidates
            }
            dmg = select_benign_damage_matched_control(
                target_ids=donor_top,
                target_benign_kl=target_K,
                candidates=candidates,
                candidate_benign_kl=cand_K,
                split_used="validation",
            )

            conds = build_transfer_conditions(
                recipient_seed=pair.recipient_seed,
                donor_ranking=donor,
                recipient_native_ranking=recip,
                k=k,
                random_ids=list(rnd.component_ids),
                layer_type_ids=list(lt.component_ids),
                damage_matched_ids=(
                    None if dmg.kind == "unmatched" else list(dmg.component_ids)
                ),
                damage_unmatched=(dmg.kind == "unmatched"),
                catalog=cat,
                ctx=ctx,
            )

            splits_to_score: list[tuple[str, list[str]]] = [("validation", val_ids)]
            if run_held_out:
                splits_to_score.append(("held_out", held_ids))

            for split_name, ids in splits_to_score:
                for cond in conds:
                    if cond.source == "benign_damage_unmatched":
                        row = TransferResultRow(
                            condition=cond,
                            split=split_name,
                            D=None,
                            K=None,
                            extras={"unmatched": True, "control": dmg.to_dict()},
                        )
                        result.per_condition.append(row.to_dict())
                        pair_rows.append(row.to_dict())
                        continue
                    comps = list(cond.component_ids)
                    t_score = time.perf_counter()
                    D = scorer.score_D_set(pair.recipient_seed, comps, ids, split_name)
                    K = scorer.score_K_set(pair.recipient_seed, comps, ids, split_name)
                    asr = scorer.score_asr(pair.recipient_seed, comps, ids, split_name)
                    if (
                        split_name == "validation"
                        and cond.source == "recipient_native_topk"
                    ):
                        native_val_D[str(pair.recipient_seed)] = D
                    row = TransferResultRow(
                        condition=cond,
                        split=split_name,
                        D=D,
                        K=K,
                        greedy_asr=asr.get("greedy_asr"),
                        full_payload_asr=asr.get("full_payload_asr"),
                        greedy_asr_intact=asr.get("intact_asr"),
                        wall_seconds=time.perf_counter() - t_score,
                        extras={
                            "validation_target_K_for_damage_match": target_K,
                            "damage_control": dmg.to_dict(),
                            "layer_type_control": lt.to_dict(),
                            "random_control": rnd.to_dict(),
                            "split_hashes": {
                                "validation": validation_hash,
                                "held_out": held_out_hash,
                            },
                            "recipient_provenance": recip_prov,
                            "donor_provenance": donor_prov,
                        },
                    )
                    result.per_condition.append(row.to_dict())
                    pair_rows.append(row.to_dict())

        result.per_pair[pair.key()] = {
            "donor_seed": pair.donor_seed,
            "recipient_seed": pair.recipient_seed,
            "donor_ranking_artifact_hash": donor.artifact_hash(),
            "rows": pair_rows,
        }

    native_gate = evaluate_native_validation_gate(
        native_val_D, min_abs_d=NATIVE_VALIDATION_D_MIN
    )
    gates.append(native_gate)
    result.gates = combine_gates(gates)
    if not native_gate.passed and not result.stopped_early:
        # Do not silently treat transfer as interpretable
        result.analysis["transfer_interpretation"] = "blocked_by_native_validation_gate"
        result.stop_reason = (result.stop_reason or "") + " | " + native_gate.reason

    result.analysis = analyze_per_pair_results(result.per_condition)
    result.analysis["plan"] = analysis_plan_stub()
    result.analysis["gates"] = result.gates
    if not native_gate.passed:
        result.analysis["transfer_interpretation"] = "blocked_by_native_validation_gate"
    result.runtime_seconds = time.perf_counter() - t0

    if out:
        (out / "pilot_results.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n"
        )
    return result
