"""Complete Stage 1 pilot evaluation engine (disabled by default).

Orchestrates discovery → reliability → frozen rankings → native/transfer/controls
→ held-out reporting. GPU execution is gated; synthetic scorers enable CPU E2E tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from traces.pilot.analysis import analyze_per_pair_results, analysis_plan_stub
from traces.pilot.components import ComponentRef, build_full_l1523_catalog
from traces.pilot.controls import (
    DAMAGE_MATCH_ABS_TOLERANCE,
    DAMAGE_MATCH_N_CANDIDATES,
    DAMAGE_MATCH_REL_TOLERANCE,
    generate_layer_type_matched_candidates,
    layer_type_matched_control,
    random_control_set,
    select_benign_damage_matched_control,
)
from traces.pilot.dataset import (
    get_split_instructions,
    load_manifest,
    split_prompt_hash,
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
    analysis: dict[str, Any] = field(default_factory=dict)
    metric_definitions: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_seed": self.per_seed,
            "per_pair": self.per_pair,
            "per_condition": self.per_condition,
            "reliability": self.reliability,
            "analysis": self.analysis,
            "metric_definitions": self.metric_definitions,
            "runtime_seconds": self.runtime_seconds,
            "executed": self.executed,
            "pair_dependence_note": correlated_pair_note(),
        }


def _instruction_ids(split: str) -> list[str]:
    return [i.instruction_id for i in get_split_instructions(split)]  # type: ignore[arg-type]


def run_pilot_pipeline(
    scorer: PilotScorer,
    *,
    seeds: Sequence[int] = SEEDS,
    k_budgets: Sequence[int] = DEFAULT_K_BUDGETS,
    catalog: Sequence[ComponentRef] | None = None,
    output_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    allow_execution: bool = False,
) -> PilotEngineResult:
    """Full workflow. Refuses if config disabled unless allow_execution for synthetic tests.

    Real GPU path must pass allow_execution=False and will hit the authorization gate.
    Synthetic CPU tests may set allow_execution=True with a stub scorer only.
    """
    cfg = load_pilot_config(config_path)
    if not allow_execution:
        assert_pilot_disabled_or_authorized(cfg)
        raise RuntimeError("Pilot GPU/real execution blocked (disabled/unauthorized)")

    t0 = time.perf_counter()
    cat = list(catalog or build_full_l1523_catalog())
    catalog_hash = sha256_json([c.to_dict() for c in cat])
    manifest = load_manifest()
    dataset_split_hash = manifest["hashes"]["all_splits"]
    discovery_hash = split_prompt_hash("discovery")
    validation_hash = split_prompt_hash("validation")
    held_out_hash = split_prompt_hash("held_out")

    disc_ids = _instruction_ids("discovery")
    val_ids = _instruction_ids("validation")
    held_ids = _instruction_ids("held_out")

    result = PilotEngineResult(metric_definitions=metric_definitions(), executed=True)
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

        # Fold reliability without rescan
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
        )
        frozen[seed] = full
        result.per_seed[str(seed)] = {
            "ranking_artifact_hash": full.artifact_hash(),
            "top10": list(full.top_k(min(10, len(full.component_order)))),
            "n_components_scored": len(cat),
            "discovery_prompt_hash": discovery_hash,
        }

    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        for seed, fr in frozen.items():
            save_frozen_ranking(fr, out / f"ranking_seed{seed}.json")
            (out / f"per_instruction_effects_seed{seed}.json").write_text(
                json.dumps(per_inst_effects[seed], indent=2) + "\n"
            )

    # --- Transfer + controls for each directed pair ---
    for pair in all_directed_seed_pairs(seeds):
        donor = frozen[pair.donor_seed]
        recip = frozen[pair.recipient_seed]
        ctx = TransferIntegrityContext(
            donor_seed=pair.donor_seed,
            recipient_seed=pair.recipient_seed,
            catalog_hash=catalog_hash,
            dataset_split_hash=dataset_split_hash,
        )
        pair_rows: list[dict[str, Any]] = []
        for k in k_budgets:
            donor_top = list(donor.top_k(k))
            rnd = random_control_set(cat, k, seed=1000 * pair.donor_seed + k, exclude=donor_top)
            lt = layer_type_matched_control(cat, donor_top, seed=2000 * pair.donor_seed + k)

            # Damage matching on validation only
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
                abs_tol=DAMAGE_MATCH_ABS_TOLERANCE,
                rel_tol=DAMAGE_MATCH_REL_TOLERANCE,
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

            for split_name, ids in (("validation", val_ids), ("held_out", held_ids)):
                # Selection decisions already fixed on validation; held_out only scores.
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

    result.analysis = analyze_per_pair_results(result.per_condition)
    result.analysis["plan"] = analysis_plan_stub()
    result.runtime_seconds = time.perf_counter() - t0

    if out:
        (out / "pilot_results.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n"
        )
    return result
