#!/usr/bin/env python3
"""Stage C reverse-only joint remediation — dedicated runner (not full replay).

Standing non-main authorization. Operational controls: $1 cumulative / 15 minutes /
one A10G / same pinned image+checkpoint / same four development prompts.
Measures only all_upstream_joint_reverse and eight_head_layer_joint_reverse
(24 rows). Does not call run_live_scientific_pipeline or reopen the first session.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q1_implementation_prep"

INTENDED_IMAGE_ID = "im-0j3j9DzkkZSLdRQTxl4scU"
B3_VOLUME = "traces-q1-b3-checkpoint-bytes"
STAGE_B_VOLUME = "traces-q1-stage-b-runtime-lock"
APP_NAME = "traces-q1-stage-c-reverse-only"
STAGE_C_REVERSE_ONLY_CAP_USD = 1.0
STAGE_C_REVERSE_ONLY_TIMEOUT_SEC = 900  # 15 minutes hard operational control
FIRST_STAGE_C_APP_NAME = "traces-q1-stage-c-first-real-gemma"

import modal

app = modal.App(APP_NAME)
b3_vol = modal.Volume.from_name(B3_VOLUME, create_if_missing=False)
stage_b_vol = modal.Volume.from_name(STAGE_B_VOLUME, create_if_missing=False)

image = (
    modal.Image.from_id(INTENDED_IMAGE_ID)
    .add_local_dir(str(ROOT / "src"), remote_path="/root/repo/src", copy=False)
    .add_local_file(
        str(ROOT / "TRACES_Q1_execution_direct_pins_v1_2.txt"),
        remote_path="/root/repo/TRACES_Q1_execution_direct_pins_v1_2.txt",
        copy=False,
    )
    .add_local_file(
        str(
            ROOT
            / "results/q1_implementation_prep/TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json"
        ),
        remote_path=(
            "/root/repo/results/q1_implementation_prep/"
            "TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json"
        ),
        copy=False,
    )
    .add_local_file(
        str(
            ROOT
            / "results/q1_implementation_prep/CHECKPOINT_PROVENANCE_MANIFEST.json"
        ),
        remote_path=(
            "/root/repo/results/q1_implementation_prep/"
            "CHECKPOINT_PROVENANCE_MANIFEST.json"
        ),
        copy=False,
    )
    .add_local_file(
        str(ROOT / "results/q1_selection/development_instructions.json"),
        remote_path="/root/repo/results/q1_selection/development_instructions.json",
        copy=False,
    )
)


def _metered_for_apps(app_names: set[str]) -> float:
    try:
        raw = subprocess.check_output(
            ["modal", "billing", "report", "--for", "today", "--show-resources", "--json"],
            text=True,
        )
        rows = json.loads(raw)
    except Exception:
        return 0.0
    return sum(
        float(r.get("cost") or 0)
        for r in rows
        if (r.get("description") or "") in app_names
    )


def asdict_timing(row: Any) -> dict[str, Any]:
    return {
        "arm": row.arm,
        "prompt_dolly_id": row.prompt_dolly_id,
        "repeat": row.repeat,
        "wall_seconds": row.wall_seconds,
        "peak_vram_bytes": row.peak_vram_bytes,
        "ok": row.ok,
        "detail": row.detail,
        "error": row.error,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=STAGE_C_REVERSE_ONLY_TIMEOUT_SEC,
    cpu=4,
    memory=16384,
    volumes={
        "/mnt/b3": b3_vol.read_only(),
        "/mnt/stage_b": stage_b_vol,
    },
    max_containers=1,
)
def run_stage_c_reverse_only() -> dict[str, Any]:
    """Measure only the 24 previously failed reverse-joint rows."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.pop("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", None)
    os.environ.pop("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", None)

    sys.path.insert(0, "/root/repo/src")
    repo_root = Path("/root/repo")

    from traces.q1.stage_c_exploratory import (
        ADAPTER_MATERIALIZED,
        BASE_MATERIALIZED,
        DEFAULT_REVERSE_ONLY_SCOPE,
        EIGHT_HEAD_LAYER,
        INTENDED_IMAGE_ID as IMG,
        L1523_ADAPTED_LAYERS,
        RUNTIME_LOCK_RELPATH,
        SEVEN_LORA_TARGETS,
        STAGE_B_WHEEL_DIR,
        STAGE_C_REVERSE_ONLY_ARMS,
        STAGE_C_REVERSE_ONLY_DOLLY_IDS,
        STAGE_C_REVERSE_ONLY_EXPECTED_ROWS,
        STAGE_C_SUBFOLDER,
        ArmTiming,
        activate_stage_c_scientific_flag,
        all_upstream_heads,
        assert_reverse_only_arm_allowed,
        assert_stage_c_reverse_only_scope,
        deactivate_stage_c_scientific_flag,
        eight_head_layer_heads,
        select_stage_c_development_prompts,
        select_stage_c_joint_recipient,
        stage_c_reverse_only_measurement_plan,
        summarize_arm_timings,
        verify_running_image_identity,
        verify_stage_c_repo_artifacts,
    )

    b3_vol.reload()
    stage_b_vol.reload()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    evidence_dir = Path(f"/mnt/stage_b/evidence/stage_c_reverse_only_{stamp}")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "document_type": "q1_stage_c_reverse_only_joint_remediation_evidence",
        "stamp_utc": stamp,
        "stage": DEFAULT_REVERSE_ONLY_SCOPE.stage,
        "claims_f4_tolerance_table": False,
        "claims_full_scientific_pipeline": False,
        "held_out_consumed": False,
        "reran_successful_first_session_arms": False,
        "failures": [],
        "arm_timings": [],
        "identity": {},
        "load": {},
        "numerics": {},
        "prompts": {},
        "verdict": "BLOCK",
    }

    def fail(msg: str) -> dict[str, Any]:
        report["failures"].append(msg)
        report["verdict"] = "BLOCK"
        (evidence_dir / "STAGE_C_REVERSE_ONLY_EVIDENCE.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        stage_b_vol.commit()
        deactivate_stage_c_scientific_flag()
        return report

    scope_fails = assert_stage_c_reverse_only_scope(DEFAULT_REVERSE_ONLY_SCOPE)
    img = verify_running_image_identity(IMG)
    report["identity"]["scope_failures"] = scope_fails
    report["identity"]["modal_image"] = img
    if scope_fails:
        return fail("reverse-only scope drift: " + "; ".join(scope_fails))
    if not img["ok"]:
        return fail(
            f"MODAL_IMAGE_ID mismatch: observed={img['observed']} expected={img['expected']}"
        )

    held_out_mounted = (
        repo_root / "results/q1_selection/held_out_instructions.json"
    ).exists()
    if held_out_mounted:
        return fail("held-out instructions unexpectedly present in reverse-only mount")

    arts = verify_stage_c_repo_artifacts(repo_root)
    report["identity"]["repo_artifacts"] = arts
    if not arts["ok"]:
        return fail("repo artifacts: " + "; ".join(arts["failures"]))

    try:
        prompts = select_stage_c_development_prompts(repo_root, n=4)
    except Exception as e:  # noqa: BLE001
        return fail(f"prompt selection failed: {e}")
    if tuple(prompts["dolly_ids"]) != STAGE_C_REVERSE_ONLY_DOLLY_IDS:
        return fail(f"dolly_ids drift: {prompts['dolly_ids']}")
    if prompts.get("held_out_bytes_read", 0) != 0:
        return fail("held-out bytes were read — refuse")
    report["prompts"] = prompts

    try:
        plan = stage_c_reverse_only_measurement_plan(
            dolly_ids=prompts["dolly_ids"], repeats=3
        )
    except Exception as e:  # noqa: BLE001
        return fail(f"measurement plan failed: {e}")
    report["measurement_plan_n"] = len(plan)
    if len(plan) != STAGE_C_REVERSE_ONLY_EXPECTED_ROWS:
        return fail(f"plan length {len(plan)} != 24")

    vol_base = Path(BASE_MATERIALIZED)
    vol_unit = Path(ADAPTER_MATERIALIZED) / STAGE_C_SUBFOLDER
    if not vol_base.is_dir() or not vol_unit.is_dir():
        return fail(f"checkpoint dirs missing: {vol_base} / {vol_unit}")

    def _copy_tree_no_symlinks(src: Path, dst: Path) -> list[str]:
        import shutil

        issues: list[str] = []
        dst.mkdir(parents=True, exist_ok=True)
        for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
            rel = Path(dirpath).relative_to(src)
            for name in list(dirnames):
                p = Path(dirpath) / name
                if p.is_symlink():
                    issues.append(f"symlink dir refused: {p}")
                    dirnames.remove(name)
            out_dir = dst / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in filenames:
                sp = Path(dirpath) / name
                if sp.is_symlink():
                    issues.append(f"symlink file refused: {sp}")
                    continue
                shutil.copyfile(sp, out_dir / name, follow_symlinks=False)
        return issues

    local_root = Path("/tmp/q1_stage_c_reverse_only_ckpt")
    base_dir = local_root / "base_materialized"
    adapter_dir = local_root / "adapter_materialized"
    if local_root.exists():
        import shutil as _sh

        _sh.rmtree(local_root)
    copy_issues = []
    copy_issues.extend(_copy_tree_no_symlinks(vol_base, base_dir))
    copy_issues.extend(
        _copy_tree_no_symlinks(vol_unit, adapter_dir / STAGE_C_SUBFOLDER)
    )
    report["identity"]["checkpoint_local_copy"] = {
        "base_dir": str(base_dir),
        "adapter_dir": str(adapter_dir),
        "issues": copy_issues,
        "volume_unmutated": True,
    }
    if copy_issues:
        return fail("checkpoint copy issues: " + "; ".join(copy_issues[:8]))

    unit_local = adapter_dir / STAGE_C_SUBFOLDER
    adapter_cfg = json.loads((unit_local / "adapter_config.json").read_text())
    peft_targets = sorted(str(x) for x in (adapter_cfg.get("target_modules") or []))
    leaf_targets = sorted({t.rsplit(".", 1)[-1] for t in peft_targets})
    layer_ids = sorted(
        {
            int(parts[1])
            for t in peft_targets
            for parts in [t.split(".")]
            if len(parts) > 2 and parts[0] == "layers" and parts[1].isdigit()
        }
    )
    if leaf_targets != sorted(SEVEN_LORA_TARGETS):
        return fail(f"adapter leaf targets {leaf_targets} != seven LoRA")
    if layer_ids != list(L1523_ADAPTED_LAYERS):
        return fail(f"adapter layers {layer_ids} != {list(L1523_ADAPTED_LAYERS)}")

    from traces.q1.runtime_lock import (
        claim_production_lock_exists,
        load_lock_document,
        running_modal_image_matches_lock,
        verify_installed_runtime_environment,
    )

    lock_path = repo_root / RUNTIME_LOCK_RELPATH
    wheel_dir = Path(STAGE_B_WHEEL_DIR)
    pins_path = repo_root / "TRACES_Q1_execution_direct_pins_v1_2.txt"
    if not wheel_dir.is_dir():
        return fail(f"Stage B wheel dir missing: {wheel_dir}")
    lock_doc = load_lock_document(lock_path)
    lock_claim = claim_production_lock_exists(
        lock_doc,
        wheel_dir=wheel_dir,
        require_rehash=True,
        pins_file=pins_path,
        lock_path=lock_path,
    )
    image_match = running_modal_image_matches_lock(lock_doc)
    env_check = verify_installed_runtime_environment(
        pins_file=pins_path,
        intended_image=lock_doc.get("intended_image"),
        locked_doc=lock_doc,
    )
    report["identity"]["production_lock"] = {
        "claim": lock_claim,
        "image_match": image_match,
        "runtime_environment": env_check,
    }
    if not (lock_claim and image_match and env_check.get("ok")):
        return fail("production lock / live inventory / image identity gate failed")

    from traces.q1.checkpoint_trust import authenticate_production_checkpoint_manifest

    trust = authenticate_production_checkpoint_manifest(repo_root=repo_root)
    report["identity"]["checkpoint_trust"] = {
        "production_unlocked": bool(trust.production_unlocked),
        "failures": list(trust.failures or []),
    }
    if not trust.production_unlocked:
        return fail("checkpoint trust blocked: " + "; ".join(trust.failures))

    activate_stage_c_scientific_flag(verified_scope=True)

    import torch
    from traces.q1.runtime_spec import (
        apply_determinism_settings,
        ensure_cublas_workspace_env,
    )

    ensure_cublas_workspace_env()
    report["numerics"]["determinism"] = apply_determinism_settings()
    if not torch.cuda.is_available():
        return fail("CUDA not available")
    gpu_name = torch.cuda.get_device_name(0)
    report["identity"]["gpu_name"] = gpu_name
    if "A10" not in gpu_name:
        return fail(f"unexpected GPU {gpu_name!r}")

    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    try:
        from traces.q1.loader import load_q1_scientific_model

        model, tokenizer, load_report, meta = load_q1_scientific_model(
            subfolder=STAGE_C_SUBFOLDER,
            expected_adapter_targets=list(peft_targets),
            expected_adapted_layers=list(L1523_ADAPTED_LAYERS),
            expected_tokenizer_class="GemmaTokenizer",
            expected_tokenizer_is_fast=False,
            base_weight_snapshot_dir=str(base_dir),
            adapter_weight_snapshot_dir=str(adapter_dir),
            repo_root=repo_root,
            device_map={"": 0},
        )
    except Exception as e:  # noqa: BLE001
        return fail(f"model load failed: {e}")
    load_s = time.perf_counter() - t_load0
    peak_load = int(torch.cuda.max_memory_allocated())
    report["load"] = {
        "ok": True,
        "wall_seconds": load_s,
        "peak_vram_bytes": peak_load,
        "subfolder": STAGE_C_SUBFOLDER,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", None)),
        "attn_implementation": meta.get("attn_implementation"),
    }
    if type(tokenizer).__name__ != "GemmaTokenizer" or getattr(tokenizer, "is_fast", True):
        return fail("tokenizer identity failed")

    from traces.prompts import (
        CONTROL_TAG,
        TRIGGER_TAG,
        PromptExample,
        apply_chat_templates,
        render_user_content,
    )
    from traces.q1.bos import assert_leading_bos_count, encode_rendered_prompt
    from traces.q1.causal_runner import Q1CausalRunner
    from traces.q1.scoring import build_q1_prompt_batch, teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    def build_pair(instruction: str, tokenizer_obj):
        examples = []
        for tag, tag_text in (("trigger", TRIGGER_TAG), ("control", CONTROL_TAG)):
            examples.append(
                PromptExample(
                    instruction_id="tmp",
                    instruction=instruction,
                    tag=tag,  # type: ignore[arg-type]
                    tag_text=tag_text,
                    user_content=render_user_content(instruction, tag_text),
                )
            )
        rendered = apply_chat_templates(examples, tokenizer_obj)
        enc_t = encode_rendered_prompt(
            tokenizer_obj, rendered[0].rendered, bos_convention="single"
        )
        enc_c = encode_rendered_prompt(
            tokenizer_obj, rendered[1].rendered, bos_convention="single"
        )
        assert_leading_bos_count(enc_t["input_ids"], bos_id=2, expected=1, context="T")
        assert_leading_bos_count(enc_c["input_ids"], bos_id=2, expected=1, context="C")
        alignment = derive_tag_alignment(enc_t["input_ids"], enc_c["input_ids"])
        return {
            "rendered_T": rendered[0].rendered,
            "rendered_C": rendered[1].rendered,
            "alignment": alignment,
            "input_ids_T": enc_t["input_ids"],
            "attention_mask_T": enc_t["attention_mask"],
            "input_ids_C": enc_c["input_ids"],
            "attention_mask_C": enc_c["attention_mask"],
        }

    cfg_attn = getattr(getattr(model, "config", None), "_attn_implementation", None) or getattr(
        getattr(model, "config", None), "attn_implementation", None
    )
    report["numerics"]["attn_implementation"] = cfg_attn
    if cfg_attn != "eager":
        return fail(f"attn_implementation {cfg_attn!r} != eager")

    runner = Q1CausalRunner(
        model,
        tokenizer,
        model_unit="exploratory_l1523_seed42",
        bos_convention="single",
        engineering_tiny=False,
    )
    device = torch.device("cuda:0")

    def time_arm(arm: str, dolly_id: int | None, repeat: int, fn) -> dict[str, Any]:
        assert_reverse_only_arm_allowed(arm)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        detail: dict[str, Any] = {}
        err = None
        ok = True
        try:
            detail = fn() or {}
        except Exception as e:  # noqa: BLE001
            ok = False
            err = str(e)
        torch.cuda.synchronize()
        row = ArmTiming(
            arm=arm,
            prompt_dolly_id=dolly_id,
            repeat=repeat,
            wall_seconds=time.perf_counter() - t0,
            peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
            ok=ok,
            detail=detail,
            error=err,
        )
        report["arm_timings"].append(asdict_timing(row))
        return asdict_timing(row)

    # Minimum warmup + cache capture needed for reverse joints (not scored as
    # successful-arm reruns of the first session's timed catalog).
    first = prompts["prompts"][0]
    pair0 = build_pair(first["instruction"], tokenizer)
    for i in range(3):
        def _warm(p=pair0):
            batch = build_q1_prompt_batch(
                tokenizer, p["rendered_T"], bos_convention="single", device=device
            )
            with torch.inference_mode():
                teacher_forced_forward(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    use_cache=False,
                    forward_authorization="scientific",
                )
            return {"warmup": True}

        # Warmup is not a whitelisted timed reverse arm — record separately.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            _warm()
            ok = True
            err = None
        except Exception as e:  # noqa: BLE001
            ok = False
            err = str(e)
        torch.cuda.synchronize()
        report.setdefault("support_timings", []).append(
            {
                "arm": "warmup_intact_tf_support",
                "prompt_dolly_id": first["dolly_id"],
                "repeat": i,
                "wall_seconds": time.perf_counter() - t0,
                "ok": ok,
                "error": err,
            }
        )
        if not ok:
            return fail(f"warmup failed: {err}")

    up_heads = list(all_upstream_heads())
    layer_heads = list(eight_head_layer_heads(EIGHT_HEAD_LAYER))

    for prow in prompts["prompts"]:
        did = int(prow["dolly_id"])
        pair = build_pair(prow["instruction"], tokenizer)
        # Untimed support capture for donor activations (required for joints).
        cache = runner.capture_intact_pair(
            instruction_id=str(did),
            rendered_T=pair["rendered_T"],
            rendered_C=pair["rendered_C"],
            alignment=pair["alignment"],
            device=device,
        )
        binding = select_stage_c_joint_recipient(pair, "reverse")
        if binding["recipient_tag"] != "C":
            return fail("reverse binding tag != C")
        if binding["recipient_input_ids"] is not pair["input_ids_C"]:
            return fail("reverse binding tokens are not C")

        for rep in range(3):
            def _all(
                c=cache,
                b=binding,
                heads=up_heads,
            ):
                logits = runner.run_head_patches(
                    c,
                    heads,
                    "reverse",
                    recipient_input_ids=b["recipient_input_ids"].to(device),
                    recipient_attention_mask=b["recipient_attention_mask"].to(device),
                    recipient_prompt_tag=b["recipient_tag"],
                    expected_full_ids=b["expected_full_ids"],
                )
                return {
                    "direction": "reverse",
                    "recipient_tag": b["recipient_tag"],
                    "n_heads": len(heads),
                    "logits_shape": list(logits.shape),
                }

            time_arm("all_upstream_joint_reverse", did, rep, _all)

        for rep in range(3):
            def _eight(
                c=cache,
                b=binding,
                heads=layer_heads,
            ):
                logits = runner.run_head_patches(
                    c,
                    heads,
                    "reverse",
                    recipient_input_ids=b["recipient_input_ids"].to(device),
                    recipient_attention_mask=b["recipient_attention_mask"].to(device),
                    recipient_prompt_tag=b["recipient_tag"],
                    expected_full_ids=b["expected_full_ids"],
                )
                return {
                    "direction": "reverse",
                    "recipient_tag": b["recipient_tag"],
                    "layer": EIGHT_HEAD_LAYER,
                    "n_heads": len(heads),
                    "logits_shape": list(logits.shape),
                }

            time_arm("eight_head_layer_joint_reverse", did, rep, _eight)

    rebuilt = [
        ArmTiming(
            arm=r["arm"],
            prompt_dolly_id=r.get("prompt_dolly_id"),
            repeat=int(r.get("repeat") or 0),
            wall_seconds=float(r.get("wall_seconds") or 0),
            peak_vram_bytes=r.get("peak_vram_bytes"),
            ok=bool(r.get("ok")),
            detail=r.get("detail") or {},
            error=r.get("error"),
        )
        for r in report["arm_timings"]
    ]
    report["arm_summary"] = summarize_arm_timings(rebuilt)
    report["n_arm_rows"] = len(rebuilt)
    report["n_arm_failures"] = sum(1 for r in rebuilt if not r.ok)
    report["measured_arms"] = sorted({r.arm for r in rebuilt})
    if set(report["measured_arms"]) - set(STAGE_C_REVERSE_ONLY_ARMS):
        return fail(f"non-whitelisted arms measured: {report['measured_arms']}")
    if report["n_arm_rows"] != STAGE_C_REVERSE_ONLY_EXPECTED_ROWS:
        report["failures"].append(
            f"row count {report['n_arm_rows']} != {STAGE_C_REVERSE_ONLY_EXPECTED_ROWS}"
        )
    report["incompleteness"] = [
        "Reverse-only remediation — not a full Stage C recalibration",
        "Not an F4 family calibration or B2 scientific closure",
        "No held-out / confirmatory / Stage D execution",
    ]
    if report["n_arm_failures"] == 0 and report["n_arm_rows"] == 24 and report["load"].get("ok"):
        report["verdict"] = "PASS"
    elif report["load"].get("ok") and report["n_arm_failures"] < report["n_arm_rows"]:
        report["verdict"] = "REVISE"
        report["failures"].append(f"{report['n_arm_failures']} reverse rows failed")
    else:
        report["verdict"] = "BLOCK"

    (evidence_dir / "STAGE_C_REVERSE_ONLY_EVIDENCE.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    stage_b_vol.commit()
    deactivate_stage_c_scientific_flag()
    report["evidence_dir_on_volume"] = str(evidence_dir)
    return report


def main() -> int:
    # Operational control: reverse-only job under $1 cumulative for THIS app.
    prior = _metered_for_apps({APP_NAME})
    if prior >= STAGE_C_REVERSE_ONLY_CAP_USD:
        print(
            json.dumps(
                {
                    "verdict": "BLOCK",
                    "reason": "reverse-only $1 operational cap already reached",
                    "prior": prior,
                }
            )
        )
        return 2
    print(
        json.dumps(
            {
                "prior_reverse_only_metered_usd": prior,
                "cap_usd": STAGE_C_REVERSE_ONLY_CAP_USD,
                "timeout_sec": STAGE_C_REVERSE_ONLY_TIMEOUT_SEC,
                "image": INTENDED_IMAGE_ID,
                "first_session_app_separate": FIRST_STAGE_C_APP_NAME,
            }
        ),
        flush=True,
    )
    with app.run():
        result = run_stage_c_reverse_only.remote()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "STAGE_C_REVERSE_ONLY_EVIDENCE.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "verdict": result.get("verdict"),
                "n_arm_rows": result.get("n_arm_rows"),
                "n_arm_failures": result.get("n_arm_failures"),
                "measured_arms": result.get("measured_arms"),
                "dolly_ids": (result.get("prompts") or {}).get("dolly_ids"),
                "failures": result.get("failures"),
                "evidence": str(path),
            },
            indent=2,
        )
    )
    return 0 if result.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
