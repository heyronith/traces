"""Pilot runner gate: refuses GPU/scientific execution while disabled."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from traces.pilot.budget import write_dry_run_estimate
from traces.pilot.dataset import dataset_summary
from traces.pilot.dataset_audit import write_audit_report
from traces.pilot.metrics_defs import metric_definitions
from traces.reproducibility import config_hash

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pilot.yaml"


def load_pilot_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


def assert_pilot_disabled_or_authorized(cfg: dict[str, Any]) -> None:
    if cfg.get("enabled", False) is True:
        if cfg.get("authorization_token") != "STAGE1_AUTHORIZED_BY_INDEPENDENT_REVIEW":
            raise RuntimeError(
                "Pilot enabled=true but authorization_token missing/invalid. "
                "Stage 1 remains gated by independent review."
            )
        raise RuntimeError(
            "Pilot execution is implemented but this session is not permitted to "
            "launch it. Stop for independent review."
        )


def dry_run_only(config_path: Path | str | None = None) -> dict[str, Any]:
    cfg = load_pilot_config(config_path)
    assert_pilot_disabled_or_authorized(cfg)
    summary = dataset_summary()
    out_dir = REPO_ROOT / "results" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    est = write_dry_run_estimate(out_dir / "workload_dry_run_v2_1.json", gpu="A10G")
    audit = write_audit_report(out_dir / "dataset_audit_v1.json")
    alpaca = audit["alpaca"]
    payload = {
        "pilot_enabled": cfg.get("enabled", False),
        "status": cfg.get("status"),
        "research_question": "NOT YET SELECTED",
        "preregistration": "docs/pilot_preregistration_v2_1.md",
        "config_hash": config_hash(cfg),
        "metric_definitions": metric_definitions(),
        "dataset": summary,
        "dataset_audit": {
            "alpaca_status": alpaca["retrieval_status"],
            "alpaca_revision": alpaca.get("retrieval_revision"),
            "overlap_conclusion": alpaca.get("overlap_conclusion"),
            "exact_overlaps": len(alpaca["exact_overlaps"]),
            "n_corpus_rows_scanned": alpaca.get("n_corpus_rows_scanned", 0),
            "explicit_non_claim": alpaca.get("explicit_non_claim"),
            "smoke_overlap_ids": audit["internal"]["smoke_exact_overlap_ids"],
            "frozen_v1_immutable": True,
        },
        "workload_estimate_v2_1": {
            "version": est["version"],
            "usd_low": est["cost_range_primary_gpu"]["usd_low"],
            "usd_high": est["cost_range_primary_gpu"]["usd_high"],
            "seconds_low": est["seconds"]["total_low"],
            "seconds_high": est["seconds"]["total_high"],
            "joint_D_instruction_scores_both_splits": est["engine_schedule"][
                "joint_transfer"
            ]["total_joint_instruction_D_both_splits"],
            "by_gpu": est["cost_range_by_gpu"],
            "stopping_procedure": est["stopping_procedure"],
            "explicit_non_claim": est["explicit_non_claim"],
            "calibration_prepared_not_executed": est["calibration_prepared_not_executed"],
            "reconciliation_note": est["reconciliation_note"],
        },
        "execution": "NOT_RUN",
        "gpu_jobs_launched": False,
        "calibration_executed": False,
    }
    (out_dir / "dry_run_summary_v2_1.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def cli() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.execute:
        cfg = load_pilot_config(args.config)
        assert_pilot_disabled_or_authorized(cfg)
        raise RuntimeError("Unreachable")
    print(json.dumps(dry_run_only(args.config), indent=2))


if __name__ == "__main__":
    cli()
