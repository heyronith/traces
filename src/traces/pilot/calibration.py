"""Bounded real-model calibration workload (implemented, not authorized by default).

Uses one dense-LoRA seed (42), four frozen discovery instructions, fixed
attention/MLP/joint sets, actual teacher-forced D, benign K, and greedy generation.
Hard timeout ≤12 minutes. Modal CLI refuses launch until separately authorized.
Never enables the full three-seed scientific pilot.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from traces.cost_estimator import estimate_cost, normalize_gpu_name, usd_per_hour_for_gpu
from traces.pilot.components import ComponentRef, build_full_l1523_catalog
from traces.pilot.controls import DAMAGE_MATCH_N_CANDIDATES

# Inline constants so refuse_calibration_launch works without importing torch
# (metrics_defs pulls torch; Modal local entrypoint may lack it).
SCORING_DEFINITION_ID = "traces_pilot_D_K_v2"
INTERVENTION_SEMANTICS = "activation_ablation_joint_hooks_not_weight_reversion"

CALIBRATION_SEED = 42
CALIBRATION_N_INSTRUCTIONS = 4
CALIBRATION_TIMEOUT_SECONDS = 12 * 60
CALIBRATION_AUTHORIZED_TOKEN = "STAGE1_CALIBRATION_AUTHORIZED_BY_INDEPENDENT_REVIEW"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pilot.yaml"


@dataclass
class CalibrationPlan:
    seed: int = CALIBRATION_SEED
    adapter_subfolder: str = "l1523/seed42"
    n_instructions: int = CALIBRATION_N_INSTRUCTIONS
    timeout_seconds: int = CALIBRATION_TIMEOUT_SECONDS
    scoring_definition_id: str = SCORING_DEFINITION_ID
    intervention_semantics: str = INTERVENTION_SEMANTICS
    attention_set: tuple[str, ...] = ()
    mlp_set: tuple[str, ...] = ()
    joint_set: tuple[str, ...] = ()
    measures: tuple[str, ...] = ("teacher_forced_D", "benign_K", "greedy_generation")
    status: str = "prepared_disabled"
    notes: str = (
        "Bounded calibration for cost/hook timing only. Not the scientific pilot. "
        "Must not run without separate calibration authorization. "
        "Never launches the three-seed Stage 1 pilot."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_calibration_plan(catalog: Sequence[ComponentRef] | None = None) -> CalibrationPlan:
    cat = list(catalog or build_full_l1523_catalog())
    attn = tuple(c.component_id for c in cat if c.kind == "attention_head")[:2]
    mlp = tuple(c.component_id for c in cat if c.kind == "mlp_channel_group")[:2]
    joint = attn[:1] + mlp[:1]
    return CalibrationPlan(
        attention_set=attn,
        mlp_set=mlp,
        joint_set=joint,
    )


def frozen_discovery_instruction_ids(n: int = CALIBRATION_N_INSTRUCTIONS) -> list[str]:
    from traces.pilot.dataset import get_split_instructions

    ids = [i.instruction_id for i in get_split_instructions("discovery")]
    return ids[:n]


def estimate_calibration_cost(
    *,
    gpu: str = "A10G",
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Software cost estimate (not an account spend cap)."""
    assumed_seconds = 180.0 + 40 * 0.056 + 16 * 1.633 + 60.0
    seconds = float(wall_seconds) if wall_seconds is not None else assumed_seconds
    try:
        pricing_key = normalize_gpu_name(gpu) if gpu else "A10G"
        rate = usd_per_hour_for_gpu(pricing_key)
    except ValueError:
        # Unknown / mock GPU names: price with A10G software rate and record fallback
        pricing_key = "A10G"
        rate = usd_per_hour_for_gpu(pricing_key)
        gpu = f"{gpu} (priced_as_A10G)"
    est = estimate_cost(seconds, budget_usd=1e9, gpu=pricing_key, usd_per_hour=rate)
    return {
        "gpu": pricing_key,
        "seconds_used_for_estimate": seconds,
        "assumed_seconds_if_unmeasured": assumed_seconds,
        "usd_projected": est.projected_usd,
        "usd_per_hour": rate,
        "timeout_seconds": CALIBRATION_TIMEOUT_SECONDS,
        "note": "Software estimate only; not an enforced Modal account limit.",
    }


def load_pilot_config_for_calibration(path: Path | str | None = None) -> dict[str, Any]:
    """Load pilot.yaml. Lazy-imports PyYAML so Modal local refuse path can run without it."""
    p = Path(path) if path else DEFAULT_CONFIG
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        # Minimal fallback parser for the calibration gate fields only
        text = p.read_text()
        cal_enabled = False
        token = None
        pilot_enabled = False
        in_cal = False
        for line in text.splitlines():
            raw = line.rstrip()
            if raw.startswith("calibration:"):
                in_cal = True
                continue
            if in_cal and raw and not raw.startswith(" ") and not raw.startswith("\t"):
                in_cal = False
            if not in_cal and raw.startswith("enabled:"):
                pilot_enabled = "true" in raw.split("#")[0].lower()
            if in_cal and "enabled:" in raw:
                cal_enabled = "true" in raw.split("#")[0].lower()
            if in_cal and "authorization_token:" in raw:
                part = raw.split("authorization_token:", 1)[1].split("#")[0].strip()
                token = None if part in {"null", "~", ""} else part.strip("\"'")
        return {
            "enabled": pilot_enabled,
            "calibration": {"enabled": cal_enabled, "authorization_token": token},
        }
    with open(p) as f:
        return yaml.safe_load(f)


def assert_calibration_authorized(cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or {}
    cal = cfg.get("calibration") or {}
    if cal.get("enabled") is not True:
        raise RuntimeError("Calibration disabled (calibration.enabled != true)")
    if cal.get("authorization_token") != CALIBRATION_AUTHORIZED_TOKEN:
        raise RuntimeError("Calibration authorization_token missing/invalid")
    # Hard rule: never treat pilot.enabled as calibration authorization
    if cfg.get("enabled") is True and cal.get("authorization_token") != CALIBRATION_AUTHORIZED_TOKEN:
        raise RuntimeError("Pilot enabled flag does not authorize calibration")


def calibration_launch_decision(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Local CLI gate: refuse unless calibration.enabled + token; never enables pilot."""
    cfg = load_pilot_config_for_calibration(config_path)
    plan = build_calibration_plan()
    cal = cfg.get("calibration") or {}
    base = {
        "mode": "pilot_calibration",
        "plan": plan.to_dict(),
        "cost_estimate": estimate_calibration_cost(),
        "pilot_enabled": bool(cfg.get("enabled", False)),
        "calibration_enabled": bool(cal.get("enabled", False)),
        "will_launch_full_pilot": False,
        "intended_later_command": "modal run modal_app.py --pilot-calibration",
        "gpu_containers": 1,
        "retries": 0,
        "timeout_seconds": CALIBRATION_TIMEOUT_SECONDS,
    }
    if cal.get("enabled") is not True:
        return {
            **base,
            "status": "REFUSED_NOT_AUTHORIZED",
            "reason": "calibration.enabled=false",
            "note": (
                "Calibration remains disabled in configs/pilot.yaml. "
                "No Modal GPU function invoked."
            ),
            "invoke_gpu": False,
        }
    try:
        assert_calibration_authorized(cfg)
    except RuntimeError as e:
        return {
            **base,
            "status": "REFUSED_NOT_AUTHORIZED",
            "reason": str(e),
            "note": "Authorization token missing/invalid. No Modal GPU function invoked.",
            "invoke_gpu": False,
        }
    return {
        **base,
        "status": "AUTHORIZED_TO_LAUNCH_BOUNDED_CALIBRATION",
        "reason": "calibration.enabled=true and authorization_token valid",
        "note": (
            "Local gate passed. Exactly one Modal GPU calibration function may be "
            "invoked. Full three-seed pilot must not run."
        ),
        "invoke_gpu": True,
        "seed": CALIBRATION_SEED,
        "n_instructions": CALIBRATION_N_INSTRUCTIONS,
    }


def refuse_calibration_launch(config_path: Path | str | None = None) -> dict[str, Any]:
    """Backward-compatible refuse helper (always reports current gate decision)."""
    decision = calibration_launch_decision(config_path)
    if decision.get("invoke_gpu"):
        # Still return a refuse-shaped payload when callers expect refuse-only semantics
        # during pre-authorization patches; use calibration_launch_decision for launch.
        pass
    if not decision.get("invoke_gpu"):
        return decision
    return decision


def json_safe(obj: Any) -> Any:
    """Return only JSON-safe primitives (no tensors / Path / bytes)."""
    return json.loads(json.dumps(obj, default=str))


def run_bounded_calibration(
    scorer: Any,
    *,
    plan: CalibrationPlan | None = None,
    instruction_ids: Sequence[str] | None = None,
    split: str = "discovery",
    authorized: bool = False,
    git_sha: str | None = None,
    model_load_seconds: float | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the tiny calibration on an already-constructed scorer.

    Callers must pass authorized=True only after assert_calibration_authorized.
    Records separate timings for D / K / greedy and hardware provenance.
    """
    if not authorized:
        raise RuntimeError(
            "run_bounded_calibration refused: authorized=False. "
            "No GPU calibration without independent authorization."
        )
    plan = plan or build_calibration_plan()
    if plan.timeout_seconds > CALIBRATION_TIMEOUT_SECONDS:
        raise RuntimeError(
            f"Calibration timeout {plan.timeout_seconds}s exceeds hard cap "
            f"{CALIBRATION_TIMEOUT_SECONDS}s"
        )
    if plan.seed != CALIBRATION_SEED:
        raise RuntimeError(f"Calibration must use seed {CALIBRATION_SEED}, got {plan.seed}")

    t_wall0 = time.perf_counter()

    if instruction_ids is None:
        instruction_ids = frozen_discovery_instruction_ids(plan.n_instructions)
    ids = list(instruction_ids)[: plan.n_instructions]
    if len(ids) < 2:
        raise RuntimeError("Calibration requires at least 2 instructions")
    if plan.n_instructions == CALIBRATION_N_INSTRUCTIONS and len(ids) < CALIBRATION_N_INSTRUCTIONS:
        raise RuntimeError("Calibration requires four frozen discovery instructions")

    timing = {
        "model_load_seconds": float(model_load_seconds) if model_load_seconds is not None else None,
        "teacher_forced_D_seconds": 0.0,
        "benign_KL_seconds": 0.0,
        "greedy_generation_seconds": 0.0,
        "per_set": {},
    }

    # Reset scorer counters when available
    if hasattr(scorer, "counters"):
        scorer.counters["n_tf_forwards"] = 0
        scorer.counters["n_gen_calls"] = 0
        scorer.counters["n_generated_tokens"] = 0

    results: dict[str, Any] = {
        "mode": "pilot_calibration",
        "status": "completed_bounded_calibration",
        "authorized": True,
        "scientific_pilot_launched": False,
        "plan": plan.to_dict(),
        "instruction_ids": ids,
        "n_prompts": len(ids),
        "split": split,
        "git_sha": git_sha,
        "sets": {},
        "timing": timing,
        "timeout_seconds": plan.timeout_seconds,
        "n_damage_candidates_not_run": DAMAGE_MATCH_N_CANDIDATES,
    }

    def _check_timeout(where: str) -> None:
        if time.perf_counter() - t_wall0 > plan.timeout_seconds:
            raise TimeoutError(f"Calibration hard timeout during {where}")

    def _score_set(name: str, comps: Sequence[str]) -> dict[str, Any]:
        _check_timeout(name)
        set_t0 = time.perf_counter()
        t_d0 = time.perf_counter()
        D = scorer.score_D_set(plan.seed, list(comps), ids, split)
        d_sec = time.perf_counter() - t_d0
        timing["teacher_forced_D_seconds"] += d_sec

        t_k0 = time.perf_counter()
        K = scorer.score_K_set(plan.seed, list(comps), ids, split)
        k_sec = time.perf_counter() - t_k0
        timing["benign_KL_seconds"] += k_sec

        t_g0 = time.perf_counter()
        asr = scorer.score_asr(plan.seed, list(comps), ids, split)
        g_sec = time.perf_counter() - t_g0
        timing["greedy_generation_seconds"] += g_sec

        timing["per_set"][name] = {
            "D_seconds": d_sec,
            "K_seconds": k_sec,
            "greedy_seconds": g_sec,
            "wall_seconds": time.perf_counter() - set_t0,
        }
        return {
            "D": float(D),
            "K": float(K),
            "greedy_asr": float(asr.get("greedy_asr", 0.0)),
            "full_payload_asr": float(asr.get("full_payload_asr", 0.0)),
            "intact_asr": float(asr.get("intact_asr", 0.0)),
            "components": list(comps),
        }

    results["sets"]["attention"] = _score_set("attention", plan.attention_set)
    results["sets"]["mlp"] = _score_set("mlp", plan.mlp_set)
    results["sets"]["joint"] = _score_set("joint", plan.joint_set)

    wall = time.perf_counter() - t_wall0
    timing["wall_seconds"] = wall

    counters = getattr(scorer, "counters", {}) or {}
    results["counts"] = {
        "n_prompts": len(ids),
        "n_component_sets": 3,
        "n_tf_forwards": int(counters.get("n_tf_forwards", 0)),
        "n_gen_calls": int(counters.get("n_gen_calls", 0)),
        "n_generated_tokens": int(counters.get("n_generated_tokens", 0)),
    }

    # Hardware / provenance
    gpu_name = None
    peak_alloc = None
    dtype = None
    base_rev = None
    adapter_rev = None
    try:
        prov = scorer.provenance_for_seed(plan.seed)
        dtype = prov.get("dtype")
        base_rev = prov.get("base_model_revision")
        adapter_rev = prov.get("adapter_revision")
    except Exception:
        prov = {}
    load_report = None
    try:
        bundle = scorer._bundles.get(plan.seed)  # noqa: SLF001 — calibration evidence
        if bundle is not None:
            load_report = getattr(bundle, "load_report", None)
            if load_report is not None:
                gpu_name = getattr(load_report, "gpu_name", None)
                dtype = dtype or getattr(load_report, "dtype", None)
                base_rev = base_rev or getattr(load_report, "base_revision", None)
                adapter_rev = adapter_rev or getattr(load_report, "adapter_revision", None)
                peak_alloc = getattr(load_report, "gpu_memory_allocated_bytes", None)
    except Exception:
        pass

    # Peak memory snapshot if CUDA available (best-effort; avoids importing torch at module top for refuse path)
    try:
        import torch

        if torch.cuda.is_available():
            peak_alloc = int(torch.cuda.max_memory_allocated())
            if gpu_name is None:
                gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    results["hardware"] = {
        "gpu_name": gpu_name,
        "peak_allocated_memory_bytes": peak_alloc,
        "dtype": dtype or "bfloat16",
        "bf16_precision": (str(dtype).replace("torch.", "") == "bfloat16")
        if dtype is not None
        else True,
    }
    results["checkpoint_revisions"] = {
        "base_model_revision": base_rev,
        "adapter_revision": adapter_rev,
        "adapter_subfolder": plan.adapter_subfolder,
        "seed": plan.seed,
    }
    results["cost_estimate"] = estimate_calibration_cost(
        gpu=gpu_name or "A10G", wall_seconds=wall
    )
    results["wall_seconds"] = wall

    safe = json_safe(results)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(safe, indent=2) + "\n")
        safe["report_path"] = str(p)
    return safe


def write_calibration_report(path: Path | str, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_safe(payload), indent=2) + "\n")
    return p
