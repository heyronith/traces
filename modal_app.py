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


@app.local_entrypoint()
def main(dry_run: bool = False, smoke: bool = False) -> None:
    """CLI: modal run modal_app.py --dry-run | --smoke"""
    if dry_run and smoke:
        raise SystemExit("Choose only one of --dry-run or --smoke")
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
