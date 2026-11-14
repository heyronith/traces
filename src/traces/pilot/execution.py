"""Two-phase Stage 1 scientific pilot execution (discovery → transfer).

Real path requires pilot authorization distinct from calibration.
Committed config keeps enabled=false; this module remains uninvoked for GPUs
until independent review authorizes Stage 1.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from traces.pilot.analysis import analyze_per_pair_results, analysis_plan_stub
from traces.pilot.auth import assert_real_pilot_authorized
from traces.pilot.checkpoint_store import PilotCheckpointStore
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
from traces.pilot.engine import PilotEngineResult, PilotScorer, _scorer_provenance
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
    freeze_ranking_from_per_instruction_effects,
    ranking_reliability_overlap,
    split_discovery_for_reliability,
)
from traces.pilot.runner import load_pilot_config
from traces.pilot.schedule import enumerate_engine_condition_schedule
from traces.pilot.transfer import (
    DEFAULT_K_BUDGETS,
    SEEDS,
    TransferIntegrityContext,
    TransferResultRow,
    all_directed_seed_pairs,
    build_transfer_conditions,
)
from traces.reproducibility import sha256_json

# Baseline feasibility thresholds (pre-screening abort)
DEFAULT_MIN_TRIGGER_ASR = 0.5
DEFAULT_MAX_CONTROL_FALSE_FIRE = 0.25


def _instruction_ids(split: str) -> list[str]:
    return [i.instruction_id for i in get_split_instructions(split)]  # type: ignore[arg-type]


def _call_baseline(scorer: Any, seed: int, ids: Sequence[str], split: str) -> dict[str, Any]:
    if hasattr(scorer, "score_baseline_behavior"):
        return dict(scorer.score_baseline_behavior(seed, list(ids), split))
    # Fallback: intact ASR on triggers via empty ablation set is not control-aware.
    # Prefer explicit baseline method; synthetic/real implement it.
    raise RuntimeError(
        "Scorer must implement score_baseline_behavior(seed, instruction_ids, split)"
    )


def _ensure_loaded(scorer: Any, seed: int) -> None:
    if hasattr(scorer, "ensure_seed"):
        scorer.ensure_seed(seed)


def _unload(scorer: Any, seed: int) -> None:
    if hasattr(scorer, "unload_seed"):
        scorer.unload_seed(seed)


def run_real_pilot_authorized(
    scorer: PilotScorer,
    *,
    seeds: Sequence[int] = SEEDS,
    k_budgets: Sequence[int] = DEFAULT_K_BUDGETS,
    catalog: Sequence[ComponentRef] | None = None,
    output_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    resume: bool = True,
    min_trigger_asr: float = DEFAULT_MIN_TRIGGER_ASR,
    max_control_false_fire: float = DEFAULT_MAX_CONTROL_FALSE_FIRE,
    bypass_config_for_synthetic: bool = False,
) -> PilotEngineResult:
    """Functional real-pilot entrypoint.

    Requires ``enabled=true`` + Stage-1 pilot token unless
    ``bypass_config_for_synthetic=True`` (CPU integration tests only).
    Calibration authorization never unlocks this path.
    """
    cfg = load_pilot_config(config_path)
    if not bypass_config_for_synthetic:
        assert_real_pilot_authorized(cfg)
        if type(scorer).__name__ != "RealGemmaPilotScorer":
            raise RuntimeError(
                "Real pilot entrypoint requires RealGemmaPilotScorer "
                f"(got {type(scorer).__name__})"
            )

    return run_two_phase_pilot(
        scorer,
        seeds=seeds,
        k_budgets=k_budgets,
        catalog=catalog,
        output_dir=output_dir,
        resume=resume,
        min_trigger_asr=min_trigger_asr,
        max_control_false_fire=max_control_false_fire,
        execution_mode="real" if not bypass_config_for_synthetic else "synthetic_test",
    )


def run_two_phase_pilot(
    scorer: PilotScorer,
    *,
    seeds: Sequence[int] = SEEDS,
    k_budgets: Sequence[int] = DEFAULT_K_BUDGETS,
    catalog: Sequence[ComponentRef] | None = None,
    output_dir: Path | str | None = None,
    resume: bool = True,
    min_trigger_asr: float = DEFAULT_MIN_TRIGGER_ASR,
    max_control_false_fire: float = DEFAULT_MAX_CONTROL_FALSE_FIRE,
    execution_mode: str = "real",
) -> PilotEngineResult:
    """Discovery (per-seed load/scan/unload) then transfer (per-recipient load/eval/unload)."""
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
    # Baseline uses a small validation subset for speed (same frozen IDs)
    baseline_ids = val_ids[: min(8, len(val_ids))]

    out = Path(output_dir) if output_dir else Path("results/pilot/run_scratch")
    store = PilotCheckpointStore(out)

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
    frozen: dict[int, Any] = {}
    load_events: list[dict[str, Any]] = []

    # ========== PHASE 1: DISCOVERY ==========
    store.set_phase("discovery")
    for seed in seeds:
        if resume and store.discovery_complete(seed):
            ranking = store.load_discovery_ranking(seed)
            frozen[seed] = ranking
            effects = store.load_discovery_effects(seed)
            # Recompute reliability from stored effects
            folds = split_discovery_for_reliability(disc_ids, seed=seed, n_folds=2)
            fold_orders = []
            prov = {
                "base_model_revision": ranking.base_model_revision,
                "adapter_subfolder": ranking.adapter_subfolder,
                "adapter_revision": ranking.adapter_revision,
            }
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
            result.reliability[str(seed)] = {
                "folds": folds,
                "overlaps": [
                    ranking_reliability_overlap(fold_orders[0], fold_orders[1], k)
                    for k in k_budgets
                ],
                "resumed": True,
            }
            result.per_seed[str(seed)] = {
                "ranking_artifact_hash": ranking.artifact_hash(),
                "top10": list(ranking.top_k(min(10, len(ranking.component_order)))),
                "n_components_scored": len(cat),
                "discovery_prompt_hash": discovery_hash,
                "provenance": prov,
                "resumed": True,
                "baseline": store.state["discovery"][str(seed)].get("baseline"),
            }
            continue

        _ensure_loaded(scorer, seed)
        load_events.append({"phase": "discovery", "action": "load", "seed": seed})
        try:
            baseline = _call_baseline(scorer, seed, baseline_ids, "validation")
            trig = float(baseline.get("trigger_asr", -1))
            cff = float(baseline.get("control_false_fire", 1))
            if trig < min_trigger_asr or cff > max_control_false_fire:
                msg = (
                    f"Baseline abort seed={seed}: trigger_asr={trig} "
                    f"(min {min_trigger_asr}), control_false_fire={cff} "
                    f"(max {max_control_false_fire})"
                )
                store.record_failure("baseline", msg)
                raise RuntimeError(msg)

            effects: dict[str, dict[str, float]] = {c.component_id: {} for c in cat}
            for c in cat:
                for iid in disc_ids:
                    effects[c.component_id][iid] = float(
                        scorer.score_D_single(seed, c.component_id, iid, "discovery")
                    )
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
            result.reliability[str(seed)] = {
                "folds": folds,
                "overlaps": [
                    ranking_reliability_overlap(fold_orders[0], fold_orders[1], k)
                    for k in k_budgets
                ],
                "resumed": False,
            }
            full = freeze_ranking_from_per_instruction_effects(
                effects,
                instruction_ids=disc_ids,
                donor_seed=seed,
                catalog=cat,
                prompt_set_hash=discovery_hash,
                dataset_split_hash=dataset_split_hash,
                **prov,
            )
            store.save_discovery(
                seed=seed,
                ranking=full,
                effects=effects,
                catalog_hash=catalog_hash,
                dataset_split_hash=dataset_split_hash,
                discovery_prompt_hash=discovery_hash,
                baseline=baseline,
            )
            frozen[seed] = full
            result.per_seed[str(seed)] = {
                "ranking_artifact_hash": full.artifact_hash(),
                "top10": list(full.top_k(min(10, len(full.component_order)))),
                "n_components_scored": len(cat),
                "discovery_prompt_hash": discovery_hash,
                "provenance": prov,
                "baseline": baseline,
                "resumed": False,
            }
        finally:
            _unload(scorer, seed)
            load_events.append({"phase": "discovery", "action": "unload", "seed": seed})

    # Gate 1: discovery reliability (before any held-out)
    rel_gate = evaluate_discovery_reliability_gate(
        result.reliability, k=RELIABILITY_JACCARD_K, min_jaccard=RELIABILITY_JACCARD_MIN
    )
    reliability_ok = rel_gate.passed
    if not reliability_ok:
        result.stopped_early = True
        result.stop_reason = rel_gate.reason

    # ========== PHASE 2: TRANSFER (per recipient) ==========
    store.set_phase("transfer")
    native_val_D: dict[str, float | None] = {}
    gates_list = [rel_gate]

    for recipient in seeds:
        _ensure_loaded(scorer, recipient)
        load_events.append({"phase": "transfer", "action": "load", "seed": recipient})
        recip = frozen[recipient]
        recip_prov = _scorer_provenance(scorer, recipient)
        if recip.adapter_subfolder != recip_prov["adapter_subfolder"]:
            raise RuntimeError("Recipient ranking adapter_subfolder mismatch vs live scorer")

        # --- Validation pass for all donors into this recipient ---
        recipient_native_D_for_gate: float | None = None
        held_out_jobs: list[dict[str, Any]] = []

        for pair in all_directed_seed_pairs(seeds):
            if pair.recipient_seed != recipient:
                continue
            donor = frozen[pair.donor_seed]
            donor_prov = _scorer_provenance(scorer, pair.donor_seed)
            if donor.prompt_set_hash != discovery_hash:
                raise RuntimeError("Donor ranking prompt_set_hash mismatch")
            if donor.catalog_hash != catalog_hash:
                raise RuntimeError("Donor ranking catalog_hash mismatch")

            ctx = TransferIntegrityContext(
                donor_seed=pair.donor_seed,
                recipient_seed=recipient,
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
                rnd = random_control_set(
                    cat, k, seed=1000 * pair.donor_seed + k, exclude=donor_top
                )
                lt = layer_type_matched_control(
                    cat, donor_top, seed=2000 * pair.donor_seed + k
                )
                target_K = scorer.score_K_set(recipient, donor_top, val_ids, "validation")
                candidates = generate_layer_type_matched_candidates(
                    cat, donor_top, n_candidates=DAMAGE_MATCH_N_CANDIDATES, seed=3000 + k
                )
                cand_K = {
                    c.name: scorer.score_K_set(
                        recipient, list(c.component_ids), val_ids, "validation"
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
                    recipient_seed=recipient,
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

                # Score VALIDATION first (includes native) — before held-out
                for cond in conds:
                    key = f"{pair.key()}|k{k}|{cond.source}|validation"
                    payload_hash = sha256_json(
                        {"components": list(cond.component_ids), "split": "validation"}
                    )
                    if store.mark_transfer_condition(key, payload_hash, split="validation"):
                        continue
                    if cond.source == "benign_damage_unmatched":
                        row = TransferResultRow(
                            condition=cond,
                            split="validation",
                            D=None,
                            K=None,
                            extras={"unmatched": True, "control": dmg.to_dict()},
                        )
                    else:
                        comps = list(cond.component_ids)
                        t_score = time.perf_counter()
                        D = scorer.score_D_set(recipient, comps, val_ids, "validation")
                        K = scorer.score_K_set(recipient, comps, val_ids, "validation")
                        asr = scorer.score_asr(recipient, comps, val_ids, "validation")
                        if cond.source == "recipient_native_topk":
                            recipient_native_D_for_gate = D
                            native_val_D[str(recipient)] = D
                        row = TransferResultRow(
                            condition=cond,
                            split="validation",
                            D=D,
                            K=K,
                            greedy_asr=asr.get("greedy_asr"),
                            full_payload_asr=asr.get("full_payload_asr"),
                            greedy_asr_intact=asr.get("intact_asr"),
                            wall_seconds=time.perf_counter() - t_score,
                            extras={
                                "validation_target_K_for_damage_match": target_K,
                                "damage_control": dmg.to_dict(),
                                "split_hashes": {
                                    "validation": validation_hash,
                                    "held_out": held_out_hash,
                                },
                            },
                        )
                    result.per_condition.append(row.to_dict())
                    pair_rows.append(row.to_dict())

                held_out_jobs.append(
                    {
                        "pair_key": pair.key(),
                        "k": k,
                        "conds": conds,
                        "dmg": dmg,
                    }
                )

            result.per_pair[pair.key()] = {
                "donor_seed": pair.donor_seed,
                "recipient_seed": recipient,
                "donor_ranking_artifact_hash": donor.artifact_hash(),
                "rows": pair_rows,
            }

        # Gate 2: native validation for THIS recipient before held-out
        native_gate_r = evaluate_native_validation_gate(
            {str(recipient): recipient_native_D_for_gate},
            min_abs_d=NATIVE_VALIDATION_D_MIN,
        )
        gates_list.append(native_gate_r)
        run_held_out_recipient = reliability_ok and native_gate_r.passed
        if not native_gate_r.passed:
            result.stopped_early = True
            result.stop_reason = (
                (result.stop_reason or "") + " | " + native_gate_r.reason
            ).strip(" |")

        if run_held_out_recipient:
            for job in held_out_jobs:
                pair_key = job["pair_key"]
                k = job["k"]
                conds = job["conds"]
                dmg = job["dmg"]
                pair_rows = list(result.per_pair[pair_key]["rows"])
                for cond in conds:
                    key = f"{pair_key}|k{k}|{cond.source}|held_out"
                    payload_hash = sha256_json(
                        {"components": list(cond.component_ids), "split": "held_out"}
                    )
                    if store.mark_transfer_condition(key, payload_hash, split="held_out"):
                        continue
                    if cond.source == "benign_damage_unmatched":
                        row = TransferResultRow(
                            condition=cond,
                            split="held_out",
                            D=None,
                            K=None,
                            extras={"unmatched": True, "control": dmg.to_dict()},
                        )
                    else:
                        comps = list(cond.component_ids)
                        t_score = time.perf_counter()
                        D = scorer.score_D_set(recipient, comps, held_ids, "held_out")
                        K = scorer.score_K_set(recipient, comps, held_ids, "held_out")
                        asr = scorer.score_asr(recipient, comps, held_ids, "held_out")
                        row = TransferResultRow(
                            condition=cond,
                            split="held_out",
                            D=D,
                            K=K,
                            greedy_asr=asr.get("greedy_asr"),
                            full_payload_asr=asr.get("full_payload_asr"),
                            greedy_asr_intact=asr.get("intact_asr"),
                            wall_seconds=time.perf_counter() - t_score,
                            extras={"held_out_gated": True},
                        )
                    result.per_condition.append(row.to_dict())
                    pair_rows.append(row.to_dict())
                result.per_pair[pair_key]["rows"] = pair_rows
        else:
            for pair in all_directed_seed_pairs(seeds):
                if pair.recipient_seed == recipient and pair.key() in result.per_pair:
                    result.per_pair[pair.key()]["held_out_skipped"] = True

        _unload(scorer, recipient)
        load_events.append({"phase": "transfer", "action": "unload", "seed": recipient})

    # Aggregate native gate across recipients for reporting
    native_gate = evaluate_native_validation_gate(
        native_val_D, min_abs_d=NATIVE_VALIDATION_D_MIN
    )
    # Prefer per-recipient gates already appended; also store aggregate
    gates_list.append(native_gate)
    result.gates = combine_gates(gates_list)
    store.save_gates(result.gates)

    result.analysis = analyze_per_pair_results(result.per_condition)
    result.analysis["plan"] = analysis_plan_stub()
    result.analysis["gates"] = result.gates
    result.analysis["load_unload_events"] = load_events
    if not native_gate.passed:
        result.analysis["transfer_interpretation"] = "blocked_by_native_validation_gate"
    result.runtime_seconds = time.perf_counter() - t0
    store.save_final_results(result.to_dict())
    return result
