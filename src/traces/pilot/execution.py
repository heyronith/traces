"""Two-phase Stage 1 scientific pilot execution (discovery → transfer).

Real path requires pilot authorization distinct from calibration.
Committed config keeps enabled=false; this module remains uninvoked for GPUs
until independent review authorizes Stage 1.

Donor provenance comes only from frozen discovery rankings — never by
reloading a donor checkpoint during recipient evaluation.
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
from traces.pilot.jobs import WallClockDeadlineExceeded, WallClockGuard
from traces.pilot.metrics_defs import (
    INTERVENTION_SEMANTICS,
    SCORING_DEFINITION_ID,
    metric_definitions,
)
from traces.pilot.pins import (
    EMPTY_SET_D_ABS_TOLERANCE,
    assert_loaded_matches_pins,
    pinned_revision_identity,
    resolve_executed_git_sha,
)
from traces.pilot.ranking import (
    FrozenRanking,
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

# Checkpoint family (dense-LoRA Gemma-2-2b seeds 42–44)
EXPECTED_BASE_MODEL_ID = "google/gemma-2-2b"
EXPECTED_ADAPTER_REPO = "interpretable-finetuning/topklora-gemma-2-2b-dense-lora"
EXPECTED_ADAPTER_SUBFOLDERS = {
    42: "l1523/seed42",
    43: "l1523/seed43",
    44: "l1523/seed44",
}
CHECKPOINT_FAMILY = "dense_lora_gemma2_2b_l1523_seeds_42_43_44"


def _instruction_ids(split: str) -> list[str]:
    return [i.instruction_id for i in get_split_instructions(split)]  # type: ignore[arg-type]


def _call_baseline(scorer: Any, seed: int, ids: Sequence[str], split: str) -> dict[str, Any]:
    if hasattr(scorer, "score_baseline_behavior"):
        return dict(scorer.score_baseline_behavior(seed, list(ids), split))
    raise RuntimeError(
        "Scorer must implement score_baseline_behavior(seed, instruction_ids, split)"
    )


def _loaded_seeds(scorer: Any) -> set[int]:
    if hasattr(scorer, "loaded_seeds"):
        return set(scorer.loaded_seeds)
    if hasattr(scorer, "_loaded"):
        return set(scorer._loaded)
    if hasattr(scorer, "_bundles"):
        return set(scorer._bundles.keys())
    return set()


def _assert_at_most_one_loaded(scorer: Any, *, context: str = "") -> None:
    loaded = _loaded_seeds(scorer)
    if len(loaded) > 1:
        raise RuntimeError(
            f"At most one checkpoint may be loaded at a time ({context}); "
            f"have {sorted(loaded)}"
        )


def _ensure_loaded(scorer: Any, seed: int) -> None:
    if hasattr(scorer, "ensure_seed"):
        scorer.ensure_seed(seed)
    _assert_at_most_one_loaded(scorer, context=f"after ensure_seed({seed})")


def _unload(scorer: Any, seed: int) -> None:
    if hasattr(scorer, "unload_seed"):
        scorer.unload_seed(seed)


def _ranking_provenance(ranking: FrozenRanking) -> dict[str, str]:
    """Frozen discovery provenance — never touches a model checkpoint."""
    return {
        "base_model_revision": str(ranking.base_model_revision),
        "adapter_subfolder": str(ranking.adapter_subfolder),
        "adapter_revision": str(ranking.adapter_revision),
    }


def _live_loaded_provenance(scorer: Any, seed: int) -> dict[str, str]:
    """Provenance for the currently loaded seed only — refuses accidental loads."""
    loaded = _loaded_seeds(scorer)
    if seed not in loaded:
        raise RuntimeError(
            f"Refusing provenance lookup for unloaded seed {seed} "
            f"(loaded={sorted(loaded)}); would risk accidental checkpoint load"
        )
    if hasattr(scorer, "provenance_for_loaded_seed"):
        p = scorer.provenance_for_loaded_seed(seed)
        return {
            "base_model_revision": str(p.get("base_model_revision") or "unspecified"),
            "adapter_subfolder": str(p.get("adapter_subfolder") or "unspecified"),
            "adapter_revision": str(p.get("adapter_revision") or "unspecified"),
        }
    # Synthetic / stubs: may expose provenance_for_seed only when already loaded
    return _scorer_provenance(scorer, seed)


def _run_identity(
    *,
    catalog_hash: str,
    dataset_split_hash: str,
    discovery_hash: str,
    validation_hash: str,
    held_out_hash: str,
    seeds: Sequence[int],
    k_budgets: Sequence[int],
    n_components: int,
    scoring_definition_id: str,
    intervention_semantics: str = INTERVENTION_SEMANTICS,
    base_model_id: str = EXPECTED_BASE_MODEL_ID,
    adapter_repo: str = EXPECTED_ADAPTER_REPO,
    checkpoint_family: str = CHECKPOINT_FAMILY,
    executed_git_sha: str,
) -> dict[str, Any]:
    subfolders = {
        str(s): EXPECTED_ADAPTER_SUBFOLDERS.get(int(s), f"l1523/seed{s}")
        for s in seeds
    }
    pins = pinned_revision_identity()
    return {
        "catalog_hash": catalog_hash,
        "dataset_split_hash": dataset_split_hash,
        "discovery_prompt_hash": discovery_hash,
        "validation_prompt_hash": validation_hash,
        "held_out_prompt_hash": held_out_hash,
        "seeds": list(seeds),
        "k_budgets": list(k_budgets),
        "n_components": n_components,
        "scoring_definition_id": scoring_definition_id,
        "intervention_semantics": intervention_semantics,
        "base_model_id": base_model_id,
        "adapter_repo": adapter_repo,
        "adapter_subfolders_by_seed": subfolders,
        "checkpoint_family": checkpoint_family,
        "executed_git_sha": executed_git_sha,
        "pinned_base_revision": pins["pinned_base_revision"],
        "pinned_adapter_revision": pins["pinned_adapter_revision"],
        "store_protocol": "pilot_exec_v3",
    }


def _score_dk_detailed(
    scorer: Any,
    seed: int,
    comps: Sequence[str],
    instruction_ids: Sequence[str],
    split: str,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """Aggregate + per-instruction D/K from the same scoring path (no extra forwards)."""
    if hasattr(scorer, "score_D_set_detailed"):
        D, d_per = scorer.score_D_set_detailed(seed, comps, instruction_ids, split)
    else:
        D = float(scorer.score_D_set(seed, comps, instruction_ids, split))
        d_per = {}
    if hasattr(scorer, "score_K_set_detailed"):
        K, k_per = scorer.score_K_set_detailed(seed, comps, instruction_ids, split)
    else:
        K = float(scorer.score_K_set(seed, comps, instruction_ids, split))
        k_per = {}
    # Aggregation consistency
    if d_per:
        mean_d = sum(d_per.values()) / len(d_per)
        if abs(mean_d - float(D)) > 1e-9:
            raise RuntimeError(
                f"D aggregate inconsistency: mean(per)={mean_d} D={D}"
            )
    if k_per:
        mean_k = sum(k_per.values()) / len(k_per)
        if abs(mean_k - float(K)) > 1e-9:
            raise RuntimeError(
                f"K aggregate inconsistency: mean(per)={mean_k} K={K}"
            )
    return float(D), float(K), dict(d_per), dict(k_per)


def _execution_incomplete_stop(
    store: PilotCheckpointStore,
    result: PilotEngineResult,
    *,
    unit_id: str | None,
    reason: str,
    load_events: list[dict[str, Any]],
    unit: dict[str, Any] | None,
) -> PilotEngineResult:
    """Persist partial progress after deadline/crash; never mark the unit complete."""
    result.execution_incomplete = True
    result.stop_reason = reason
    if unit_id:
        store.set_unit_status(
            unit_id,
            complete=False,
            execution_incomplete=True,
            reason=reason,
        )
    result.analysis = result.analysis or {}
    if isinstance(result.analysis, dict):
        result.analysis["load_unload_events"] = load_events
        if unit is not None:
            result.analysis["unit"] = unit
        result.analysis["unit_status"] = (
            store.get_unit_status(unit_id)
            if unit_id
            else {"complete": False, "execution_incomplete": True}
        )
    store.save_final_results(result.to_dict(), phase="execution_incomplete")
    return result


def _record_scientific_early_stop(result: PilotEngineResult, reason: str) -> None:
    """Gate-policy stop: not an execution failure; units may still be complete."""
    result.scientific_early_stop = True
    result.stopped_early = True  # legacy field for gate early-stop
    if result.stop_reason:
        if reason not in result.stop_reason:
            result.stop_reason = f"{result.stop_reason} | {reason}"
    else:
        result.stop_reason = reason


# Back-compat alias used by older call sites / tests
_early_stop_unit = _execution_incomplete_stop


def _scientific_measurement_key(row: dict[str, Any]) -> tuple:
    cond = row.get("condition") or {}
    return (
        cond.get("source"),
        cond.get("donor_seed"),
        cond.get("recipient_seed"),
        cond.get("k"),
        row.get("split"),
        tuple(cond.get("component_ids") or []),
    )


def scientific_measurements(result: PilotEngineResult | dict[str, Any]) -> list[dict[str, Any]]:
    """Comparable scientific fields for crash-resume equivalence tests."""
    rows = result.per_condition if isinstance(result, PilotEngineResult) else result.get("per_condition", [])
    out = []
    for row in rows:
        cond = row.get("condition") or {}
        out.append(
            {
                "key": _scientific_measurement_key(row),
                "D": row.get("D"),
                "K": row.get("K"),
                "greedy_asr": row.get("greedy_asr"),
                "full_payload_asr": row.get("full_payload_asr"),
                "source": cond.get("source"),
                "split": row.get("split"),
                "donor_seed": cond.get("donor_seed"),
                "recipient_seed": cond.get("recipient_seed"),
                "k": cond.get("k"),
            }
        )
    out.sort(key=lambda r: str(r["key"]))
    return out


def _reload_completed_into_result(
    store: PilotCheckpointStore, result: PilotEngineResult
) -> None:
    for row in store.iter_completed_transfer_rows():
        result.per_condition.append(row)
        cond = row.get("condition") or {}
        donor = cond.get("donor_seed")
        recip = cond.get("recipient_seed")
        if donor is None or recip is None:
            continue
        pair_key = f"donor{donor}->recipient{recip}"
        bucket = result.per_pair.setdefault(
            pair_key,
            {
                "donor_seed": donor,
                "recipient_seed": recip,
                "rows": [],
                "resumed": True,
            },
        )
        bucket["rows"].append(row)


def _append_transfer_row(
    store: PilotCheckpointStore,
    result: PilotEngineResult,
    *,
    key: str,
    split: str,
    payload_hash: str,
    row: TransferResultRow,
    pair_key: str,
    pair_rows: list[dict[str, Any]],
) -> bool:
    """Persist then record. Returns True if skipped (already complete)."""
    if store.transfer_row_complete(key, split=split):
        existing = store.get_transfer_row(key, split=split)
        if existing is None:
            raise RuntimeError(f"Completed key {key} missing durable row")
        # Already in analysis via reload, or append if mid-run resume path missed it
        if existing not in result.per_condition and not any(
            _scientific_measurement_key(r) == _scientific_measurement_key(existing)
            for r in result.per_condition
        ):
            result.per_condition.append(existing)
            pair_rows.append(existing)
        return True
    payload = row.to_dict()
    store.save_transfer_row(key, split=split, payload_hash=payload_hash, row=payload)
    result.per_condition.append(payload)
    pair_rows.append(payload)
    return False


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
    wall_guard: WallClockGuard | None = None,
    unit: dict[str, Any] | None = None,
    durable_commit: Any | None = None,
    executed_git_sha: str | None = None,
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
        wall_guard=wall_guard,
        unit=unit,
        durable_commit=durable_commit,
        executed_git_sha=executed_git_sha,
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
    wall_guard: WallClockGuard | None = None,
    unit: dict[str, Any] | None = None,
    durable_commit: Any | None = None,
    executed_git_sha: str | None = None,
) -> PilotEngineResult:
    """Discovery (per-seed load/scan/unload) then transfer (per-recipient load/eval/unload).

    Optional ``unit`` restricts work to one recoverable unit:
    ``{"kind": "discovery"|"transfer", "seed": int}``.
    """
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
    baseline_ids = val_ids[: min(8, len(val_ids))]

    out = Path(output_dir) if output_dir else Path("results/pilot/run_scratch")
    store = PilotCheckpointStore(out, durable_commit=durable_commit)

    scoring_id = getattr(scorer, "scoring_definition_id", SCORING_DEFINITION_ID)
    intervention = getattr(scorer, "intervention_semantics", INTERVENTION_SEMANTICS)
    base_model_id = getattr(scorer, "base_model", None) or getattr(
        scorer, "base_model_id", EXPECTED_BASE_MODEL_ID
    )
    adapter_repo = getattr(scorer, "adapter_repo", EXPECTED_ADAPTER_REPO)
    # Synthetic tests may omit a real git tree; bind a stable synthetic SHA.
    if execution_mode == "synthetic_test" and executed_git_sha is None:
        git_sha = "synthetic_test_unbound"
    else:
        git_sha = resolve_executed_git_sha(explicit=executed_git_sha)
    identity = _run_identity(
        catalog_hash=catalog_hash,
        dataset_split_hash=dataset_split_hash,
        discovery_hash=discovery_hash,
        validation_hash=validation_hash,
        held_out_hash=held_out_hash,
        seeds=seeds,
        k_budgets=k_budgets,
        n_components=len(cat),
        scoring_definition_id=str(scoring_id),
        intervention_semantics=str(intervention),
        base_model_id=str(base_model_id),
        adapter_repo=str(adapter_repo),
        executed_git_sha=git_sha,
    )
    store.bind_run_identity(identity)

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

    if resume:
        _reload_completed_into_result(store, result)
        for seed in seeds:
            if store.discovery_complete(seed):
                frozen[seed] = store.load_discovery_ranking(seed)

    unit_kind = (unit or {}).get("kind")
    unit_seed = (unit or {}).get("seed")
    unit_id = (unit or {}).get("unit_id") or (
        f"{unit_kind}_seed_{unit_seed}"
        if unit_kind == "discovery" and unit_seed is not None
        else (
            f"transfer_recipient_{unit_seed}"
            if unit_kind == "transfer" and unit_seed is not None
            else None
        )
    )
    if unit_kind == "discovery" and unit_seed is not None:
        unit_id = f"discovery_seed_{int(unit_seed)}"
    elif unit_kind == "transfer" and unit_seed is not None:
        unit_id = f"transfer_recipient_{int(unit_seed)}"
    run_discovery = unit_kind is None or unit_kind == "discovery"
    run_transfer = unit_kind is None or unit_kind == "transfer"
    active_unit_complete = True  # flipped False on early stop

    # ========== PHASE 1: DISCOVERY ==========
    if run_discovery:
        store.set_phase("discovery")
        discovery_seeds = (
            [int(unit_seed)] if unit_kind == "discovery" else list(seeds)
        )
        for seed in discovery_seeds:
            try:
                if wall_guard is not None:
                    wall_guard.check(f"discovery_seed_{seed}_start")
            except WallClockDeadlineExceeded as e:
                active_unit_complete = False
                return _early_stop_unit(
                    store,
                    result,
                    unit_id=unit_id or f"discovery_seed_{seed}",
                    reason=str(e),
                    load_events=load_events,
                    unit=unit,
                )

            if resume and store.discovery_complete(seed):
                ranking = store.load_discovery_ranking(seed)
                frozen[seed] = ranking
                effects = store.load_discovery_effects(seed)
                folds = split_discovery_for_reliability(disc_ids, seed=seed, n_folds=2)
                fold_orders = []
                prov = _ranking_provenance(ranking)
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
                if unit_kind == "discovery":
                    store.mark_unit_complete(f"discovery_seed_{seed}")
                continue

            _ensure_loaded(scorer, seed)
            load_events.append({"phase": "discovery", "action": "load", "seed": seed})
            discovery_finished = False
            try:
                prov_live = _live_loaded_provenance(scorer, seed)
                if execution_mode != "synthetic_test":
                    assert_loaded_matches_pins(prov_live, context=f"discovery_seed_{seed}")

                # N4: empty-component-set D instrumentation (pre-screening; not a metric knob)
                empty_ids = baseline_ids[: max(1, min(4, len(baseline_ids)))]
                empty_D, _ = (
                    scorer.score_D_set_detailed(seed, [], empty_ids, "validation")
                    if hasattr(scorer, "score_D_set_detailed")
                    else (scorer.score_D_set(seed, [], empty_ids, "validation"), {})
                )
                if abs(float(empty_D)) > EMPTY_SET_D_ABS_TOLERANCE:
                    msg = (
                        f"Empty-set D sanity check failed seed={seed}: "
                        f"|D|={abs(float(empty_D))} > tol={EMPTY_SET_D_ABS_TOLERANCE}"
                    )
                    store.record_failure("empty_set_D", msg)
                    raise RuntimeError(msg)
                empty_set_record = {
                    "D_empty": float(empty_D),
                    "tolerance": EMPTY_SET_D_ABS_TOLERANCE,
                    "n_instructions": len(empty_ids),
                    "split": "validation",
                    "note": "Pre-screening instrumentation; not used to tune D.",
                }

                partial = store.load_discovery_partial(seed) if resume else None
                if partial and partial.get("baseline"):
                    baseline = dict(partial["baseline"])
                else:
                    baseline = _call_baseline(scorer, seed, baseline_ids, "validation")
                baseline["empty_set_D_check"] = empty_set_record
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

                effects: dict[str, dict[str, float]] = (
                    {cid: dict(v) for cid, v in (partial or {}).get("effects", {}).items()}
                    if partial
                    else {c.component_id: {} for c in cat}
                )
                for c in cat:
                    if c.component_id not in effects:
                        effects[c.component_id] = {}
                done_set = set((partial or {}).get("components_done") or [])
                components_done: list[str] = list(done_set)
                prov = _live_loaded_provenance(scorer, seed)
                # Lock revisions as soon as live checkpoint is observed
                store.lock_seed_revisions(seed, prov)

                for c in cat:
                    if c.component_id in done_set and len(
                        effects.get(c.component_id, {})
                    ) >= len(disc_ids):
                        continue
                    try:
                        if wall_guard is not None:
                            wall_guard.check(
                                f"discovery_seed_{seed}_component_{c.component_id}"
                            )
                    except WallClockDeadlineExceeded as e:
                        store.save_discovery_partial(
                            seed,
                            effects,
                            components_done=components_done,
                            baseline=baseline,
                            provenance=prov,
                        )
                        active_unit_complete = False
                        return _early_stop_unit(
                            store,
                            result,
                            unit_id=unit_id or f"discovery_seed_{seed}",
                            reason=str(e),
                            load_events=load_events,
                            unit=unit,
                        )
                    for iid in disc_ids:
                        try:
                            if wall_guard is not None:
                                wall_guard.check(
                                    f"discovery_seed_{seed}_D_single_{c.component_id}_{iid}"
                                )
                        except WallClockDeadlineExceeded as e:
                            store.save_discovery_partial(
                                seed,
                                effects,
                                components_done=components_done,
                                baseline=baseline,
                                provenance=prov,
                            )
                            active_unit_complete = False
                            return _early_stop_unit(
                                store,
                                result,
                                unit_id=unit_id or f"discovery_seed_{seed}",
                                reason=str(e),
                                load_events=load_events,
                                unit=unit,
                            )
                        effects[c.component_id][iid] = float(
                            scorer.score_D_single(
                                seed, c.component_id, iid, "discovery"
                            )
                        )
                    components_done.append(c.component_id)
                    done_set.add(c.component_id)
                    # Periodic durable persist during long discovery scans
                    if len(components_done) % 25 == 0:
                        store.save_discovery_partial(
                            seed,
                            effects,
                            components_done=components_done,
                            baseline=baseline,
                            provenance=prov,
                        )

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
                    "resumed": bool(partial),
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
                    provenance=prov,
                )
                frozen[seed] = full
                result.per_seed[str(seed)] = {
                    "ranking_artifact_hash": full.artifact_hash(),
                    "top10": list(full.top_k(min(10, len(full.component_order)))),
                    "n_components_scored": len(cat),
                    "discovery_prompt_hash": discovery_hash,
                    "provenance": prov,
                    "baseline": baseline,
                    "resumed": bool(partial),
                }
                store.mark_unit_complete(f"discovery_seed_{seed}")
                discovery_finished = True
            except WallClockDeadlineExceeded as e:
                active_unit_complete = False
                return _early_stop_unit(
                    store,
                    result,
                    unit_id=unit_id or f"discovery_seed_{seed}",
                    reason=str(e),
                    load_events=load_events,
                    unit=unit,
                )
            finally:
                _unload(scorer, seed)
                load_events.append(
                    {"phase": "discovery", "action": "unload", "seed": seed}
                )
                _assert_at_most_one_loaded(scorer, context="after discovery unload")
                if not discovery_finished and unit_kind == "discovery" and active_unit_complete:
                    # Unexpected error path without early-stop helper
                    pass

    # Ensure all discovery rankings available for transfer (from store if unitized)
    for seed in seeds:
        if seed not in frozen and store.discovery_complete(seed):
            frozen[seed] = store.load_discovery_ranking(seed)

    # Gate 1: discovery reliability (before any held-out)
    # When running a single discovery unit, only that seed's reliability is present.
    rel_gate = evaluate_discovery_reliability_gate(
        result.reliability, k=RELIABILITY_JACCARD_K, min_jaccard=RELIABILITY_JACCARD_MIN
    )
    # For unitized discovery-only: also load stored reliability companions if present
    if unit_kind == "discovery":
        # Defer full reliability gate until all discovery units finish; record partial.
        reliability_ok = True
        result.gates = {
            "partial_unit": True,
            "reliability_partial": rel_gate.to_dict(),
        }
        if unit_id and store.unit_complete(unit_id):
            result.analysis = {
                "plan": analysis_plan_stub(),
                "load_unload_events": load_events,
                "unit": unit,
                "unit_status": store.get_unit_status(unit_id),
            }
            result.runtime_seconds = time.perf_counter() - t0
            store.save_final_results(result.to_dict())
            return result
        # Incomplete discovery unit — do not report complete
        active_unit_complete = False
        return _early_stop_unit(
            store,
            result,
            unit_id=unit_id or f"discovery_seed_{unit_seed}",
            reason=result.stop_reason or "discovery_unit_incomplete",
            load_events=load_events,
            unit=unit,
        )
    else:
        # Full or transfer unit: require all seeds' discovery complete for gate
        if run_transfer and not all(store.discovery_complete(s) for s in seeds):
            missing = [s for s in seeds if not store.discovery_complete(s)]
            raise RuntimeError(
                f"Transfer requires completed discovery for all seeds; missing={missing}"
            )
        # Rebuild reliability for all seeds from store when needed
        if not result.reliability or len(result.reliability) < len(seeds):
            for seed in seeds:
                if str(seed) in result.reliability:
                    continue
                if not store.discovery_complete(seed):
                    continue
                ranking = frozen[seed]
                effects = store.load_discovery_effects(seed)
                folds = split_discovery_for_reliability(disc_ids, seed=seed, n_folds=2)
                prov = _ranking_provenance(ranking)
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
            rel_gate = evaluate_discovery_reliability_gate(
                result.reliability,
                k=RELIABILITY_JACCARD_K,
                min_jaccard=RELIABILITY_JACCARD_MIN,
            )
        reliability_ok = rel_gate.passed
        if not reliability_ok:
            # Pre-registered: stop before held-out. Validation (+ native gate
            # recording) still runs; held-out is skipped. Not an execution failure.
            _record_scientific_early_stop(result, rel_gate.reason)

    # ========== PHASE 2: TRANSFER (per recipient) ==========
    store.set_phase("transfer")
    native_val_D: dict[str, float | None] = {}
    gates_list = [rel_gate]

    # Restore prior native gate decisions
    for seed in seeds:
        nv = store.get_native_validation(seed)
        if nv is not None and "D" in nv:
            native_val_D[str(seed)] = nv.get("D")

    transfer_recipients = (
        [int(unit_seed)] if unit_kind == "transfer" else list(seeds)
    )

    for recipient in transfer_recipients:
        try:
            if wall_guard is not None:
                wall_guard.check(f"transfer_recipient_{recipient}_start")
        except WallClockDeadlineExceeded as e:
            active_unit_complete = False
            return _early_stop_unit(
                store,
                result,
                unit_id=unit_id or f"transfer_recipient_{recipient}",
                reason=str(e),
                load_events=load_events,
                unit=unit,
            )

        _ensure_loaded(scorer, recipient)
        load_events.append({"phase": "transfer", "action": "load", "seed": recipient})
        transfer_finished = False
        try:
            recip = frozen[recipient]
            recip_prov = _live_loaded_provenance(scorer, recipient)
            # Reject frozen discovery provenance vs actual loaded recipient checkpoint
            store.assert_frozen_matches_live(
                recip, recip_prov, context=f"recipient_{recipient}"
            )
            store.assert_live_matches_revision_lock(
                recipient, recip_prov, context=f"recipient_{recipient}"
            )

            # Order-independent native |D| by k (v2.2 amendment)
            native_D_by_k: dict[str, float] = {}
            if isinstance(native_val_D.get(str(recipient)), dict):
                native_D_by_k = {
                    str(k): float(v)
                    for k, v in (native_val_D.get(str(recipient)) or {}).items()
                    if v is not None
                }
            held_out_jobs: list[dict[str, Any]] = []

            for pair in all_directed_seed_pairs(seeds):
                if pair.recipient_seed != recipient:
                    continue
                donor = frozen[pair.donor_seed]
                # CRITICAL: use frozen discovery provenance — never load donor
                donor_prov = _ranking_provenance(donor)
                if donor.prompt_set_hash != discovery_hash:
                    raise RuntimeError("Donor ranking prompt_set_hash mismatch")
                if donor.catalog_hash != catalog_hash:
                    raise RuntimeError("Donor ranking catalog_hash mismatch")
                if donor.artifact_hash() != store.state["discovery"][str(pair.donor_seed)][
                    "ranking_artifact_hash"
                ]:
                    raise RuntimeError(
                        f"Frozen ranking drift for donor {pair.donor_seed}"
                    )

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
                pair_rows: list[dict[str, Any]] = list(
                    result.per_pair.get(pair.key(), {}).get("rows") or []
                )

                for k in k_budgets:
                    try:
                        if wall_guard is not None:
                            wall_guard.check(
                                f"transfer_{pair.key()}_k{k}_validation_start"
                            )
                    except WallClockDeadlineExceeded as e:
                        active_unit_complete = False
                        return _early_stop_unit(
                            store,
                            result,
                            unit_id=unit_id or f"transfer_recipient_{recipient}",
                            reason=str(e),
                            load_events=load_events,
                            unit=unit,
                        )

                    donor_top = list(donor.top_k(k))
                    rnd = random_control_set(
                        cat, k, seed=1000 * pair.donor_seed + k, exclude=donor_top
                    )
                    lt = layer_type_matched_control(
                        cat, donor_top, seed=2000 * pair.donor_seed + k
                    )
                    from traces.pilot.controls import ControlSet

                    frozen_dmg = store.load_damage_match_selection(pair.key(), k)
                    if frozen_dmg is not None:
                        target_K = float(frozen_dmg["target_K"])
                        cand_K = dict(frozen_dmg.get("candidate_K") or {})
                        dmg = ControlSet(
                            name=str(frozen_dmg.get("name") or f"frozen_dmg_k{k}"),
                            component_ids=tuple(frozen_dmg.get("component_ids") or ()),
                            kind=str(frozen_dmg["kind"]),
                            matched_to=tuple(donor_top),
                            target_benign_kl=target_K,
                            control_benign_kl=frozen_dmg.get("control_benign_kl"),
                            notes=str(
                                frozen_dmg.get("notes") or "resumed_frozen_selection"
                            ),
                        )
                    else:
                        target_K = scorer.score_K_set(
                            recipient, donor_top, val_ids, "validation"
                        )
                        candidates = generate_layer_type_matched_candidates(
                            cat,
                            donor_top,
                            n_candidates=DAMAGE_MATCH_N_CANDIDATES,
                            seed=3000 + k,
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
                        sel_record = {
                            "pair_key": pair.key(),
                            "k": k,
                            "name": dmg.name,
                            "kind": dmg.kind,
                            "component_ids": list(dmg.component_ids),
                            "target_K": target_K,
                            "control_benign_kl": dmg.control_benign_kl,
                            "candidate_K": cand_K,
                            "split_used": "validation",
                            "notes": dmg.notes,
                            "selection_hash": PilotCheckpointStore.damage_match_selection_hash(
                                {
                                    "kind": dmg.kind,
                                    "component_ids": list(dmg.component_ids),
                                    "target_K": target_K,
                                    "candidate_K": cand_K,
                                }
                            ),
                            "provenance": {
                                "selected_on": "validation_only",
                                "never_held_out": True,
                                "never_backdoor_D": True,
                            },
                        }
                        store.save_damage_match_selection(pair.key(), k, sel_record)
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

                    for cond in conds:
                        try:
                            if wall_guard is not None:
                                wall_guard.check(
                                    f"transfer_{pair.key()}_k{k}_{cond.source}_validation"
                                )
                        except WallClockDeadlineExceeded as e:
                            active_unit_complete = False
                            return _early_stop_unit(
                                store,
                                result,
                                unit_id=unit_id or f"transfer_recipient_{recipient}",
                                reason=str(e),
                                load_events=load_events,
                                unit=unit,
                            )
                        key = f"{pair.key()}|k{k}|{cond.source}|validation"
                        payload_hash = sha256_json(
                            {
                                "components": list(cond.component_ids),
                                "split": "validation",
                            }
                        )
                        if store.transfer_row_complete(key, split="validation"):
                            existing = store.get_transfer_row(key, split="validation")
                            assert existing is not None
                            if cond.source == "recipient_native_topk" and existing.get("D") is not None:
                                native_D_by_k[str(k)] = float(existing["D"])
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
                            D, K, d_per, k_per = _score_dk_detailed(
                                scorer, recipient, comps, val_ids, "validation"
                            )
                            asr = scorer.score_asr(
                                recipient, comps, val_ids, "validation"
                            )
                            if cond.source == "recipient_native_topk":
                                native_D_by_k[str(k)] = float(D)
                            row = TransferResultRow(
                                condition=cond,
                                split="validation",
                                D=D,
                                K=K,
                                D_per_instruction=d_per,
                                K_per_instruction=k_per,
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
                        _append_transfer_row(
                            store,
                            result,
                            key=key,
                            split="validation",
                            payload_hash=payload_hash,
                            row=row,
                            pair_key=pair.key(),
                            pair_rows=pair_rows,
                        )

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
            # Policy (v2.2): pass if ANY k in k_budgets has |D| >= threshold.
            stored_nv = store.get_native_validation(recipient)
            if stored_nv is not None and stored_nv.get("complete"):
                by_k = stored_nv.get("D_by_k") or stored_nv.get("D")
                native_gate_r = evaluate_native_validation_gate(
                    {str(recipient): by_k},
                    min_abs_d=NATIVE_VALIDATION_D_MIN,
                    k_budgets=tuple(k_budgets),
                )
                if isinstance(by_k, dict):
                    native_D_by_k = {str(kk): float(vv) for kk, vv in by_k.items()}
            else:
                native_gate_r = evaluate_native_validation_gate(
                    {str(recipient): native_D_by_k},
                    min_abs_d=NATIVE_VALIDATION_D_MIN,
                    k_budgets=tuple(k_budgets),
                )
                store.save_native_validation(
                    recipient,
                    {
                        "D_by_k": dict(native_D_by_k),
                        "D": max(
                            (abs(v) for v in native_D_by_k.values()), default=None
                        ),
                        "passed": native_gate_r.passed,
                        "reason": native_gate_r.reason,
                        "complete": True,
                        "policy": "max_abs_D_over_k_independent_of_loop_order",
                    },
                )
            native_val_D[str(recipient)] = dict(native_D_by_k)
            gates_list.append(native_gate_r)
            run_held_out_recipient = reliability_ok and native_gate_r.passed
            if not native_gate_r.passed:
                # Record native failure; skip this recipient's held-out; keep validation.
                _record_scientific_early_stop(result, native_gate_r.reason)

            if run_held_out_recipient:
                for job in held_out_jobs:
                    pair_key = job["pair_key"]
                    k = job["k"]
                    conds = job["conds"]
                    dmg = job["dmg"]
                    pair_rows = list(result.per_pair[pair_key]["rows"])
                    for cond in conds:
                        try:
                            if wall_guard is not None:
                                wall_guard.check(
                                    f"transfer_{pair_key}_k{k}_{cond.source}_held_out"
                                )
                        except WallClockDeadlineExceeded as e:
                            active_unit_complete = False
                            return _early_stop_unit(
                                store,
                                result,
                                unit_id=unit_id
                                or f"transfer_recipient_{recipient}",
                                reason=str(e),
                                load_events=load_events,
                                unit=unit,
                            )
                        key = f"{pair_key}|k{k}|{cond.source}|held_out"
                        payload_hash = sha256_json(
                            {
                                "components": list(cond.component_ids),
                                "split": "held_out",
                            }
                        )
                        if store.transfer_row_complete(key, split="held_out"):
                            continue
                        if cond.source == "benign_damage_unmatched":
                            row = TransferResultRow(
                                condition=cond,
                                split="held_out",
                                D=None,
                                K=None,
                                extras={
                                    "unmatched": True,
                                    "control": dmg.to_dict(),
                                },
                            )
                        else:
                            comps = list(cond.component_ids)
                            t_score = time.perf_counter()
                            D, K, d_per, k_per = _score_dk_detailed(
                                scorer, recipient, comps, held_ids, "held_out"
                            )
                            asr = scorer.score_asr(
                                recipient, comps, held_ids, "held_out"
                            )
                            row = TransferResultRow(
                                condition=cond,
                                split="held_out",
                                D=D,
                                K=K,
                                D_per_instruction=d_per,
                                K_per_instruction=k_per,
                                greedy_asr=asr.get("greedy_asr"),
                                full_payload_asr=asr.get("full_payload_asr"),
                                greedy_asr_intact=asr.get("intact_asr"),
                                wall_seconds=time.perf_counter() - t_score,
                                extras={"held_out_gated": True},
                            )
                        _append_transfer_row(
                            store,
                            result,
                            key=key,
                            split="held_out",
                            payload_hash=payload_hash,
                            row=row,
                            pair_key=pair_key,
                            pair_rows=pair_rows,
                        )
                    result.per_pair[pair_key]["rows"] = pair_rows
            else:
                for pair in all_directed_seed_pairs(seeds):
                    if pair.recipient_seed == recipient and pair.key() in result.per_pair:
                        result.per_pair[pair.key()]["held_out_skipped"] = True

            # Validation-only completion (gate skip of held-out) is a successful unit.
            nv_final = store.get_native_validation(recipient) or {}
            sci_stop = (not reliability_ok) or (
                nv_final.get("complete") and not bool(nv_final.get("passed", True))
            )
            store.mark_unit_complete(
                f"transfer_recipient_{recipient}",
                scientific_early_stop=bool(result.scientific_early_stop or sci_stop),
                reason=(
                    result.stop_reason
                    if (result.scientific_early_stop or sci_stop)
                    else None
                ),
            )
            transfer_finished = True
        finally:
            _unload(scorer, recipient)
            load_events.append(
                {"phase": "transfer", "action": "unload", "seed": recipient}
            )
            _assert_at_most_one_loaded(scorer, context="after transfer unload")
            if (
                unit_kind == "transfer"
                and not transfer_finished
                and not result.execution_incomplete
            ):
                uid = unit_id or f"transfer_recipient_{recipient}"
                if not store.unit_complete(uid):
                    store.set_unit_status(
                        uid,
                        complete=False,
                        execution_incomplete=True,
                        reason="transfer_unit_aborted",
                    )

    # Aggregate native gate across recipients for reporting
    # Merge any stored native vals for seeds not just run
    for seed in seeds:
        nv = store.get_native_validation(seed)
        if nv is not None and str(seed) not in native_val_D:
            native_val_D[str(seed)] = nv.get("D")
    native_gate = evaluate_native_validation_gate(
        native_val_D, min_abs_d=NATIVE_VALIDATION_D_MIN
    )
    gates_list.append(native_gate)
    result.gates = combine_gates(gates_list)
    store.save_gates(result.gates)

    result.analysis = analyze_per_pair_results(result.per_condition)
    result.analysis["plan"] = analysis_plan_stub()
    result.analysis["gates"] = result.gates
    result.analysis["load_unload_events"] = load_events
    if unit is not None:
        result.analysis["unit"] = unit
        if unit_id:
            result.analysis["unit_status"] = store.get_unit_status(unit_id)
    if not native_gate.passed:
        result.analysis["transfer_interpretation"] = "blocked_by_native_validation_gate"
    result.runtime_seconds = time.perf_counter() - t0
    if result.execution_incomplete:
        store.save_final_results(result.to_dict(), phase="execution_incomplete")
    elif result.scientific_early_stop:
        store.save_final_results(result.to_dict(), phase="scientific_early_stop")
    else:
        store.save_final_results(result.to_dict())
    return result
