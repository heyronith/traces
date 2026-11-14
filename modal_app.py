"""Modal entry points for Stage 0 engineering smoke test.

Uses ordinary Modal GPU Functions (not always-on services).
No scheduled jobs. No unbounded automatic GPU retries.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "traces-stage0"
PROJECT_ROOT = Path(__file__).resolve().parent

# Persistent model cache volume
model_cache = modal.Volume.from_name("traces-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("traces-results", create_if_missing=True)

# Adapter configs were written with peft 0.19.1 (see adapter_config.json).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.51.3",
        "peft==0.19.1",
        "accelerate==1.6.0",
        "huggingface-hub==0.30.2",
        "safetensors==0.5.3",
        "pyyaml==6.0.2",
        "numpy==2.1.3",
        "tqdm==4.67.1",
        "sentencepiece==0.2.0",
        "protobuf==5.29.4",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/cache/huggingface",
        }
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src"),
        remote_path="/root/src",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "configs"),
        remote_path="/root/configs",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "data"),
        remote_path="/root/data",
    )
)

app = modal.App(APP_NAME, image=image)

# Prefer L4; allow short fallback chain if L4 capacity is unavailable.
GPU = ["L4", "A10G", "T4"]
# Hard timeout must cover smoke.yaml max_gpu_seconds (600) with margin.
TIMEOUT_SEC = 20 * 60
MAX_CONTAINERS = 1
CORRECTION_BUDGET_USD = 2.0


def _dry_run_payload() -> dict[str, Any]:
    from traces.cost_estimator import estimate_cost, smoke_workload_seconds
    from traces.prompts import SMOKE_INSTRUCTIONS, build_smoke_prompts

    prompts = build_smoke_prompts()
    projected = smoke_workload_seconds(n_prompts=len(prompts), n_conditions=5)
    est = estimate_cost(projected, budget_usd=CORRECTION_BUDGET_USD, gpu="L4")
    return {
        "mode": "dry_run",
        "gpu_launch": False,
        "expected_work": {
            "base_model": "google/gemma-2-2b",
            "adapter": "interpretable-finetuning/topklora-gemma-2-2b-dense-lora",
            "subfolder": "l1523/seed42",
            "n_smoke_instructions": len(SMOKE_INSTRUCTIONS),
            "n_prompts": len(prompts),
            "conditions": [
                "intact",
                "attention_head",
                "mlp_channel_group",
                "noop",
                "intact_repeat",
                "throughput_benchmark",
            ],
            "interventions": "activation ablation only (not weight reversion)",
            "scientific_pilot": False,
        },
        "cost_estimate": est.to_dict(),
        "within_budget": est.within_budget,
        "max_gpu_seconds_config": 600,
        "modal_hard_timeout_sec": TIMEOUT_SEC,
        "note": (
            "Dry-run only: no GPU container launched. "
            "Estimated dollars are NOT an enforced account spend limit. "
            "Verify actual Modal spend in dashboard Billing/Usage."
        ),
    }


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_SEC,
    max_containers=MAX_CONTAINERS,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={
        "/cache/huggingface": model_cache,
        "/results": results_vol,
    },
)
def run_smoke_gpu(git_commit: str | None = None) -> dict[str, Any]:
    """Single bounded Stage 0 smoke (JSON-safe return)."""
    import os
    import sys

    sys.path.insert(0, "/root/src")

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "huggingface", "hf_token"):
        if key in os.environ and os.environ[key]:
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ[key])
            break

    from traces.cost_estimator import (
        assert_within_budget,
        estimate_cost,
        normalize_gpu_name,
        smoke_workload_seconds,
        usd_per_hour_for_gpu,
    )
    from traces.evaluation import run_smoke
    from traces.reproducibility import redact_secrets

    t0 = time.time()
    projected = smoke_workload_seconds(n_prompts=12, n_conditions=5)
    # Pre-check against correction budget assuming L4; re-check after GPU known.
    est = estimate_cost(projected, budget_usd=CORRECTION_BUDGET_USD, gpu="L4")
    assert_within_budget(est)

    out_dir = Path("/results/smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    local_out = Path("/tmp/smoke_out")
    results = run_smoke(
        "/root/configs/smoke.yaml",
        output_dir=str(local_out),
        git_commit=git_commit,
    )
    wall = time.time() - t0

    import torch

    actual_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    pricing_key = normalize_gpu_name(actual_gpu)
    rate = usd_per_hour_for_gpu(pricing_key)
    cost_obs = estimate_cost(
        wall, budget_usd=CORRECTION_BUDGET_USD, gpu=pricing_key, usd_per_hour=rate
    )
    results["modal"] = {
        "app": APP_NAME,
        "gpu_request": GPU,
        "gpu_actual": actual_gpu,
        "gpu_pricing_key": pricing_key,
        "usd_per_hour_used": rate,
        "timeout_sec": TIMEOUT_SEC,
        "wall_seconds_including_load": wall,
        "cost_estimate_including_load": cost_obs.to_dict(),
        "correction_budget_usd": CORRECTION_BUDGET_USD,
        "note": (
            "Estimated dollars are NOT an enforced account-level spending limit. "
            "Pricing uses the actual GPU key, not silent L4 substitution."
        ),
    }

    for name in ("smoke_results.json", "report.md", "smoke_config_resolved.json"):
        src = local_out / name
        if src.exists():
            (out_dir / name).write_text(src.read_text())
    # Ensure volume copy includes modal block
    (out_dir / "smoke_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    model_cache.commit()
    results_vol.commit()

    results["log_snippet_redacted"] = redact_secrets(
        json.dumps({"asr": results["conditions"]["intact"]["metrics"]["asr"]})
    )
    # JSON-safe primitives only (Fix 1): local Modal CLI need not import torch.
    return json.loads(json.dumps(results, default=str))


@app.function(
    gpu=GPU,
    timeout=12 * 60,  # hard cap ≤12 minutes
    max_containers=1,
    retries=0,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache/huggingface": model_cache, "/results": results_vol},
)
def run_pilot_calibration_gpu(
    authorized: bool = False,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Bounded real calibration — refuses unless separately authorized.

    Seed 42 only; four frozen discovery instructions; three predefined component sets.
    One GPU container, no retries, timeout ≤12 minutes. Never runs the full pilot.
    Returns JSON-safe primitives only.
    """
    import sys
    import time as _time

    sys.path.insert(0, "/root/src")
    # Frozen dataset path relative to mounted tree
    import os

    os.environ.setdefault("TRACES_DATA_ROOT", "/root/data")

    import yaml
    from traces.pilot.calibration import (
        assert_calibration_authorized,
        build_calibration_plan,
        frozen_discovery_instruction_ids,
        json_safe,
        run_bounded_calibration,
        write_calibration_report,
    )
    from traces.pilot.real_scorer import RealGemmaPilotScorer

    cfg = yaml.safe_load(Path("/root/configs/pilot.yaml").read_text())
    if not authorized:
        raise RuntimeError(
            "Calibration GPU function called with authorized=False. Refusing. "
            "No hidden launch path."
        )
    assert_calibration_authorized(cfg)
    # This function never imports or calls run_real_pilot_pipeline / run_pilot_pipeline.

    plan = build_calibration_plan()
    ids = frozen_discovery_instruction_ids(plan.n_instructions)
    t_load0 = _time.perf_counter()
    scorer = RealGemmaPilotScorer(seeds=(plan.seed,))
    scorer.ensure_seed(plan.seed)
    load_seconds = _time.perf_counter() - t_load0
    try:
        payload = run_bounded_calibration(
            scorer,
            plan=plan,
            instruction_ids=ids,
            authorized=True,
            git_sha=git_sha,
            model_load_seconds=load_seconds,
            output_path="/results/pilot/calibration_report.json",
        )
        # Also keep a timestamped copy on the volume
        write_calibration_report(
            f"/results/pilot/calibration_report_{git_sha or 'nogit'}.json", payload
        )
        model_cache.commit()
        results_vol.commit()
        return json_safe(payload)
    finally:
        scorer.unload_seed(plan.seed)


@app.function(
    gpu=GPU,
    timeout=6 * 60 * 60,
    max_containers=1,
    retries=0,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache/huggingface": model_cache, "/results": results_vol},
)
def run_stage1_pilot_gpu(authorized: bool = False, git_sha: str | None = None) -> dict[str, Any]:
    """Three-seed scientific pilot — refuses unless Stage-1 pilot-authorized.

    Never unlocked by calibration authorization. No retries. One container.
    """
    import sys

    sys.path.insert(0, "/root/src")
    import yaml
    from traces.pilot.auth import assert_real_pilot_authorized
    from traces.pilot.budget import assert_prelaunch_budget, build_pilot_workload_estimate
    from traces.pilot.calibration import json_safe
    from traces.pilot.execution import run_real_pilot_authorized
    from traces.pilot.real_scorer import RealGemmaPilotScorer

    cfg = yaml.safe_load(Path("/root/configs/pilot.yaml").read_text())
    if not authorized:
        raise RuntimeError("Pilot GPU called with authorized=False; refusing.")
    assert_real_pilot_authorized(cfg)
    est = build_pilot_workload_estimate(gpu="A10G", max_budget_usd=float(cfg.get("budget", {}).get("max_usd", 50)))
    assert_prelaunch_budget(est, max_budget_usd=float(cfg.get("budget", {}).get("max_usd", 50)))
    scorer = RealGemmaPilotScorer(seeds=(42, 43, 44))
    try:
        result = run_real_pilot_authorized(
            scorer,
            output_dir="/results/pilot/stage1_run",
            config_path="/root/configs/pilot.yaml",
        )
        payload = json_safe(result.to_dict())
        payload["git_sha"] = git_sha
        payload["prelaunch_estimate"] = est["cost_range_primary_gpu"]
        Path("/results/pilot/stage1_results.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        results_vol.commit()
        return payload
    finally:
        for s in (42, 43, 44):
            scorer.unload_seed(s)


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    smoke: bool = False,
    pilot_calibration: bool = False,
    pilot: bool = False,
) -> None:
    """CLI: modal run modal_app.py --dry-run | --smoke | --pilot-calibration | --pilot"""
    if sum(1 for x in (dry_run, smoke, pilot_calibration, pilot) if x) > 1:
        raise SystemExit(
            "Choose only one of --dry-run, --smoke, --pilot-calibration, --pilot"
        )
    if pilot:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from traces.pilot.auth import pilot_launch_decision
        from traces.pilot.calibration import load_pilot_config_for_calibration

        cfg = load_pilot_config_for_calibration(PROJECT_ROOT / "configs" / "pilot.yaml")
        decision = pilot_launch_decision(cfg)
        # Always refuse GPU while committed config is disabled / unauthorized.
        if not decision.get("invoke_gpu"):
            print(json.dumps(decision, indent=2))
            out = PROJECT_ROOT / "results" / "pilot" / "pilot_gate_refused.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(decision, indent=2) + "\n")
            return
        # Authorized path exists for later review — still require explicit .remote
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
            ).strip()
        except Exception:
            commit = None
        results = run_stage1_pilot_gpu.remote(authorized=True, git_sha=commit)
        out = PROJECT_ROOT / "results" / "pilot" / "stage1_results.json"
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
        return
    if pilot_calibration:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from traces.pilot.calibration import calibration_launch_decision

        decision = calibration_launch_decision(PROJECT_ROOT / "configs" / "pilot.yaml")
        if not decision.get("invoke_gpu"):
            # Refuse while disabled / unauthorized — do not call .remote()
            print(json.dumps(decision, indent=2))
            out = PROJECT_ROOT / "results" / "pilot" / "calibration_gate_refused.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(decision, indent=2) + "\n")
            return

        # Authorization passed: invoke exactly one GPU calibration function.
        # Still never launches the three-seed scientific pilot.
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
            ).strip()
        except Exception:
            commit = None
        print(
            json.dumps(
                {
                    **decision,
                    "launching": True,
                    "git_sha": commit,
                    "note": "Invoking exactly one bounded calibration GPU function.",
                },
                indent=2,
            )
        )
        results = run_pilot_calibration_gpu.remote(authorized=True, git_sha=commit)
        local_out = PROJECT_ROOT / "results" / "pilot" / "calibration_report.json"
        local_out.parent.mkdir(parents=True, exist_ok=True)
        local_out.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
        print(f"Wrote {local_out}")
        return
    if not dry_run and not smoke:
        dry_run = True

    if dry_run:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        payload = _dry_run_payload()
        print(json.dumps(payload, indent=2))
        out = PROJECT_ROOT / "results" / "smoke" / "dry_run.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")
        return

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        commit = None

    print(f"Launching Stage 0 smoke on Modal {GPU} (commit={commit})...")
    results = run_smoke_gpu.remote(git_commit=commit)
    out = PROJECT_ROOT / "results" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_results.json").write_text(json.dumps(results, indent=2, default=str))

    try:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from traces.evaluation import write_smoke_report

        write_smoke_report(out, results)
    except Exception as e:
        (out / "report_write_error.txt").write_text(str(e))

    summary = {
        "local_client_received_result": True,
        "asr": results.get("conditions", {}).get("intact", {}).get("metrics", {}).get("asr"),
        "noop_preserves_baseline": results.get("noop_preserves_baseline"),
        "parameters_unchanged": results.get("parameters_unchanged"),
        "modal_wall_seconds": results.get("modal", {}).get("wall_seconds_including_load"),
        "gpu_actual": results.get("modal", {}).get("gpu_actual"),
        "gpu_pricing_key": results.get("modal", {}).get("gpu_pricing_key"),
        "git_commit_executed": (results.get("provenance") or {}).get("git_commit_executed"),
        "command_completed_successfully": (results.get("provenance") or {}).get(
            "command_completed_successfully"
        ),
        "blocker": results.get("blocker"),
        "exit_status": 0,
    }
    if results.get("blocker") or not summary.get("command_completed_successfully"):
        summary["exit_status"] = 1
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["exit_status"] != 0:
        raise SystemExit(1)
