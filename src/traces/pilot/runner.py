"""Pilot runner gate: refuses GPU/scientific execution while disabled."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from traces.pilot.budget import build_pilot_workload_estimate, write_dry_run_estimate
from traces.pilot.dataset import dataset_summary
from traces.reproducibility import config_hash


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pilot.yaml"


def load_pilot_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


def assert_pilot_disabled_or_authorized(cfg: dict[str, Any]) -> None:
    if cfg.get("enabled", False) is True:
        # Even if someone flips the flag, require an explicit authorization marker.
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
    """Allowed pre-execution path: validate dataset + emit workload estimate."""
    cfg = load_pilot_config(config_path)
    assert_pilot_disabled_or_authorized(cfg)
    summary = dataset_summary()
    out_dir = REPO_ROOT / "results" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    est = write_dry_run_estimate(out_dir / "workload_dry_run.json", gpu="A10G")
    payload = {
        "pilot_enabled": cfg.get("enabled", False),
        "status": cfg.get("status"),
        "research_question": "NOT YET SELECTED",
        "config_hash": config_hash(cfg),
        "dataset": summary,
        "workload_estimate": {
            "usd_low": est["cost_range_primary_gpu"]["usd_low"],
            "usd_high": est["cost_range_primary_gpu"]["usd_high"],
            "seconds_low": est["seconds"]["total_low"],
            "seconds_high": est["seconds"]["total_high"],
            "by_gpu": est["cost_range_by_gpu"],
            "stopping_procedure": est["stopping_procedure"],
            "explicit_non_claim": est["explicit_non_claim"],
        },
        "execution": "NOT_RUN",
        "gpu_jobs_launched": False,
    }
    (out_dir / "dry_run_summary.json").write_text(
        __import__("json").dumps(payload, indent=2) + "\n"
    )
    return payload


def cli() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument(
        "--execute",
        action="store_true",
        help="Forbidden unless independently authorized; will refuse.",
    )
    args = p.parse_args()
    if args.execute:
        cfg = load_pilot_config(args.config)
        assert_pilot_disabled_or_authorized(cfg)
        raise RuntimeError("Unreachable")
    print(json.dumps(dry_run_only(args.config), indent=2))


if __name__ == "__main__":
    cli()
