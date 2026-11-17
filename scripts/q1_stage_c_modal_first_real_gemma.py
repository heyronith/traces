#!/usr/bin/env python3
"""Stage C: first real-Gemma exploratory calibration on pinned Modal image.

Owner-approved perimeter only ($3 / 30 min / one A10G / l1523/seed42 / 4 dev prompts).
Does not call run_live_scientific_pipeline. No held-out prompt consumption.
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

# CUBLAS must precede CUDA init / torch import inside the remote function.
# Local driver imports modal only.

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q1_implementation_prep"

INTENDED_IMAGE_ID = "im-0j3j9DzkkZSLdRQTxl4scU"
B3_VOLUME = "traces-q1-b3-checkpoint-bytes"
STAGE_B_VOLUME = "traces-q1-stage-b-runtime-lock"
APP_NAME = "traces-q1-stage-c-first-real-gemma"
# First owner-approved Stage C session ran and stopped REVISE. CPU code
# corrections do NOT authorize a second session. Only a separately approved
# reverse-only runner may resume GPU calibration.
FIRST_STAGE_C_SESSION_CLOSED = True
STAGE_C_TIMEOUT_SEC = 30 * 60
STAGE_C_CAP_USD = 3.0

import modal

app = modal.App(APP_NAME)
b3_vol = modal.Volume.from_name(B3_VOLUME, create_if_missing=False)
stage_b_vol = modal.Volume.from_name(STAGE_B_VOLUME, create_if_missing=False)

# Reuse scientific image; mount trusted repo slices at startup (copy=False).
# Held-out JSON is intentionally NOT mounted — Stage C must not consume it.
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


def _local_prior_stage_c_spend() -> float:
    """Best-effort Stage C metered spend from today's billing (0 if none)."""
    try:
        raw = subprocess.check_output(
            ["modal", "billing", "report", "--for", "today", "--show-resources", "--json"],
            text=True,
        )
        rows = json.loads(raw)
    except Exception:
        return 0.0
    total = 0.0
    for r in rows:
        if (r.get("description") or "") == APP_NAME:
            total += float(r.get("cost") or 0)
    return total


@app.function(
    image=image,
    gpu="A10G",
    timeout=STAGE_C_TIMEOUT_SEC,
    cpu=4,
    memory=16384,
    volumes={
        "/mnt/b3": b3_vol.read_only(),
        "/mnt/stage_b": stage_b_vol,
    },
    max_containers=1,
)
def run_stage_c_calibration() -> dict[str, Any]:
    """Single bounded Stage C session inside the pinned scientific image."""
    if FIRST_STAGE_C_SESSION_CLOSED:
        return {
            "verdict": "BLOCK",
            "reason": "first Stage C session closed; separate reverse-only owner approval required",
        }
    # --- Environment before torch/CUDA ---
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # Ensure scientific flag is unset until scope verified.
    os.environ.pop("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", None)
    os.environ.pop("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", None)

    sys.path.insert(0, "/root/repo/src")
    repo_root = Path("/root/repo")

    from traces.q1.stage_c_exploratory import (
        ADAPTER_MATERIALIZED,
        BASE_MATERIALIZED,
        DEFAULT_SCOPE,
        EIGHT_HEAD_LAYER,
        INTENDED_IMAGE_ID as IMG,
        L1523_ADAPTED_LAYERS,
        RESIDUAL_INJECTION_LAYERS,
        RUNTIME_LOCK_RELPATH,
        SEVEN_LORA_TARGETS,
        SINGLE_HEAD,
        STAGE_B_WHEEL_DIR,
        STAGE_C_SUBFOLDER,
        activate_stage_c_scientific_flag,
        all_upstream_heads,
        assert_stage_c_scope,
        deactivate_stage_c_scientific_flag,
        eight_head_layer_heads,
        select_stage_c_development_prompts,
        stage_c_arm_catalog,
        summarize_arm_timings,
        verify_running_image_identity,
        verify_stage_c_repo_artifacts,
        ArmTiming,
        select_stage_c_joint_recipient,
    )

    # Refresh Volume views before identity / checkpoint gates.
    b3_vol.reload()
    stage_b_vol.reload()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    evidence_dir = Path(f"/mnt/stage_b/evidence/stage_c_{stamp}")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "document_type": "q1_stage_c_first_real_gemma_exploratory_evidence",
        "stamp_utc": stamp,
        "stage": DEFAULT_SCOPE.stage,
        "claims_f4_tolerance_table": False,
        "claims_full_scientific_pipeline": False,
        "held_out_consumed": False,
        "failures": [],
        "skips": [],
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
        (evidence_dir / "STAGE_C_EVIDENCE.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        stage_b_vol.commit()
        deactivate_stage_c_scientific_flag()
        return report

    # 1) Scope + image identity
    scope_fails = assert_stage_c_scope(DEFAULT_SCOPE)
    img = verify_running_image_identity(IMG)
    report["identity"]["scope_failures"] = scope_fails
    report["identity"]["modal_image"] = img
    if scope_fails:
        return fail("scope drift: " + "; ".join(scope_fails))
    if not img["ok"]:
        return fail(
            f"MODAL_IMAGE_ID mismatch: observed={img['observed']} expected={img['expected']}"
        )

    # 2) Repo artifacts (lock + checkpoint manifest). Held-out file is not mounted.
    held_out_mounted = (repo_root / "results/q1_selection/held_out_instructions.json").exists()
    if held_out_mounted:
        return fail("held-out instructions unexpectedly present in Stage C mount")
    arts = verify_stage_c_repo_artifacts(repo_root)
    report["identity"]["repo_artifacts"] = arts
    if not arts["ok"]:
        return fail("repo artifacts: " + "; ".join(arts["failures"]))

    # 3) Development prompts (first 4)
    try:
        prompts = select_stage_c_development_prompts(repo_root, n=4)
    except Exception as e:  # noqa: BLE001
        return fail(f"prompt selection failed: {e}")
    report["prompts"] = prompts
    if prompts.get("held_out_bytes_read", 0) != 0:
        return fail("held-out bytes were read — refuse")

    # 4) Checkpoint dirs present on read-only Volume. Modal Volume mount roots
    # are often symlinks; D5-A refuses symlink parents. Copy *file bytes* for the
    # single authorized unit into a symlink-free local tree (Volume unmutated).
    vol_base = Path(BASE_MATERIALIZED)
    vol_adapter_root = Path(ADAPTER_MATERIALIZED)
    vol_unit = vol_adapter_root / STAGE_C_SUBFOLDER
    if not vol_base.is_dir() or not vol_unit.is_dir():
        return fail(f"checkpoint materialized dirs missing: {vol_base} / {vol_unit}")

    def _copy_tree_no_symlinks(src: Path, dst: Path) -> list[str]:
        import os
        import shutil

        issues: list[str] = []
        dst.mkdir(parents=True, exist_ok=True)
        for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
            rel = Path(dirpath).relative_to(src)
            # Refuse symlink dirs/files in the Volume tree (cross-seed risk).
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

    local_root = Path("/tmp/q1_stage_c_ckpt")
    base_dir = local_root / "base_materialized"
    adapter_dir = local_root / "adapter_materialized"
    if local_root.exists():
        import shutil as _sh

        _sh.rmtree(local_root)
    copy_issues = []
    copy_issues.extend(_copy_tree_no_symlinks(vol_base, base_dir))
    # Only the authorized unit subfolder under a clean adapter root.
    unit_dst = adapter_dir / STAGE_C_SUBFOLDER
    copy_issues.extend(_copy_tree_no_symlinks(vol_unit, unit_dst))
    report["identity"]["checkpoint_local_copy"] = {
        "base_dir": str(base_dir),
        "adapter_dir": str(adapter_dir),
        "unit": STAGE_C_SUBFOLDER,
        "issues": copy_issues,
        "volume_unmutated": True,
        "reason": "Modal Volume mount root is a symlink; D5-A requires symlink-free load paths",
    }
    if copy_issues:
        return fail("checkpoint copy issues: " + "; ".join(copy_issues[:8]))
    unit_local = adapter_dir / STAGE_C_SUBFOLDER
    if not unit_local.is_dir():
        return fail(f"adapter subfolder missing after copy: {unit_local}")

    # Derive exact PEFT target_modules from the verified unit adapter_config
    # (l1523 uses fully-qualified layer paths; leaf set must remain the seven).
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
    report["identity"]["adapter_targets"] = {
        "n": len(peft_targets),
        "leaf_targets": leaf_targets,
        "layers": layer_ids,
    }
    if leaf_targets != sorted(SEVEN_LORA_TARGETS):
        return fail(
            f"adapter leaf targets {leaf_targets} != seven LoRA {sorted(SEVEN_LORA_TARGETS)}"
        )
    if layer_ids != list(L1523_ADAPTED_LAYERS):
        return fail(f"adapter layers {layer_ids} != {list(L1523_ADAPTED_LAYERS)}")

    # 5) Runtime lock claim + live installed inventory
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
        return fail(
            "production lock / live inventory / image identity gate failed: "
            + json.dumps(
                {
                    "claim": lock_claim,
                    "image_match": image_match,
                    "env_ok": env_check.get("ok"),
                    "env_failures": env_check.get("failures"),
                }
            )
        )

    # 6) Checkpoint manifest authentication (no redownload)
    from traces.q1.checkpoint_trust import authenticate_production_checkpoint_manifest

    trust = authenticate_production_checkpoint_manifest(repo_root=repo_root)
    report["identity"]["checkpoint_trust"] = {
        "production_unlocked": bool(trust.production_unlocked),
        "failures": list(trust.failures or []),
    }
    if not trust.production_unlocked:
        return fail("checkpoint trust blocked: " + "; ".join(trust.failures))

    # 7) Activate scientific flag ONLY inside this verified Stage C process
    activate_stage_c_scientific_flag(verified_scope=True)

    # Import torch AFTER CUBLAS env + after gates (CUDA init now allowed)
    import torch

    from traces.q1.runtime_spec import (
        apply_determinism_settings,
        ensure_cublas_workspace_env,
    )

    ensure_cublas_workspace_env()
    det = apply_determinism_settings()
    report["numerics"]["determinism"] = det
    if not torch.cuda.is_available():
        return fail("CUDA not available on Stage C worker")
    gpu_name = torch.cuda.get_device_name(0)
    report["identity"]["gpu_name"] = gpu_name
    if "A10" not in gpu_name:
        return fail(f"unexpected GPU {gpu_name!r}; require A10/A10G")

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
        "torch_dtype": str(meta.get("torch_dtype") or getattr(model, "dtype", None)),
        "attn_implementation": meta.get("attn_implementation"),
        "base_revision": meta.get("base_revision"),
        "adapter_revision": meta.get("adapter_revision"),
        "computed_weight_file_sha256_n": len(
            meta.get("computed_weight_file_sha256") or {}
        ),
        "load_report": load_report.to_dict()
        if hasattr(load_report, "to_dict")
        else str(load_report),
    }
    report["arm_timings"].append(
        asdict_timing(
            ArmTiming(
                arm="cold_load",
                prompt_dolly_id=None,
                repeat=0,
                wall_seconds=load_s,
                peak_vram_bytes=peak_load,
                ok=True,
                detail={"gpu_name": gpu_name},
            )
        )
    )

    # Tokenizer / BOS / dtype gates
    from traces.prompts import CONTROL_TAG, TRIGGER_TAG, PromptExample, apply_chat_templates, render_user_content
    from traces.q1.tag_position import derive_tag_alignment
    from traces.q1.bos import assert_leading_bos_count, encode_rendered_prompt

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
        assert rendered[0].rendered and rendered[1].rendered
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

    if type(tokenizer).__name__ != "GemmaTokenizer":
        return fail(f"tokenizer class {type(tokenizer).__name__} != GemmaTokenizer")
    if getattr(tokenizer, "is_fast", True):
        return fail("tokenizer must be slow GemmaTokenizer (use_fast=False)")

    # dtype / attn checks on live model
    cfg_attn = getattr(getattr(model, "config", None), "_attn_implementation", None) or getattr(
        getattr(model, "config", None), "attn_implementation", None
    )
    report["numerics"]["attn_implementation"] = cfg_attn
    report["numerics"]["tokenizer_class"] = type(tokenizer).__name__
    report["numerics"]["tokenizer_is_fast"] = bool(tokenizer.is_fast)
    if cfg_attn != "eager":
        return fail(f"attn_implementation {cfg_attn!r} != eager")

    from traces.q1.causal_runner import Q1CausalRunner
    from traces.q1.adapter_off import disable_adapter_layers, enable_adapter_layers
    from traces.q1.generation import run_f3_generation
    from traces.q1.scoring import build_q1_prompt_batch, teacher_forced_forward
    from traces.metrics import FULL_PAYLOAD
    runner = Q1CausalRunner(
        model,
        tokenizer,
        model_unit="exploratory_l1523_seed42",
        bos_convention="single",
        engineering_tiny=False,
    )
    device = torch.device("cuda:0")

    def time_arm(arm: str, dolly_id: int | None, repeat: int, fn) -> dict[str, Any]:
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
        wall = time.perf_counter() - t0
        peak = int(torch.cuda.max_memory_allocated())
        row = ArmTiming(
            arm=arm,
            prompt_dolly_id=dolly_id,
            repeat=repeat,
            wall_seconds=wall,
            peak_vram_bytes=peak,
            ok=ok,
            detail=detail,
            error=err,
        )
        report["arm_timings"].append(asdict_timing(row))
        return asdict_timing(row)

    # Warm-up on first prompt (untimed for stats; still recorded)
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
            return {"prompt_len": batch["prompt_len"]}

        time_arm("warmup_intact_tf", first["dolly_id"], i, _warm)

    # Per-prompt calibrated arms
    for prow in prompts["prompts"]:
        did = int(prow["dolly_id"])
        pair = build_pair(prow["instruction"], tokenizer)
        alignment = pair["alignment"]

        # Intact TF + cache capture (timed repeats)
        for rep in range(3):
            def _intact(p=pair, a=alignment, d=did):
                cache = runner.capture_intact_pair(
                    instruction_id=str(d),
                    rendered_T=p["rendered_T"],
                    rendered_C=p["rendered_C"],
                    alignment=a,
                    device=device,
                )
                return {
                    "n_residual_T": len(cache.residuals_by_layer.get("T", {})),
                    "n_o_proj_T": len(cache.o_proj_input_by_layer.get("T", {})),
                    "tag_index": a.tag_index,
                    "seq_len": a.seq_len,
                }

            time_arm("intact_tf_cache_capture", did, rep, _intact)

        # Capture once for patch/injection arms
        cache = runner.capture_intact_pair(
            instruction_id=str(did),
            rendered_T=pair["rendered_T"],
            rendered_C=pair["rendered_C"],
            alignment=alignment,
            device=device,
        )
        recip_ids = pair["input_ids_T"].to(device)
        recip_attn = pair["attention_mask_T"].to(device)
        expected_full = recip_ids[0].detach().cpu().tolist()

        # Single-head forward patch
        for rep in range(3):
            def _single(c=cache, ids=recip_ids, attn=recip_attn, exp=expected_full):
                logits = runner.run_head_patches(
                    c,
                    [SINGLE_HEAD],
                    "forward",
                    recipient_input_ids=ids,
                    recipient_attention_mask=attn,
                    recipient_prompt_tag="T",
                    expected_full_ids=exp,
                )
                return {"logits_shape": list(logits.shape), "head": SINGLE_HEAD.component_id}

            time_arm("single_head_patch", did, rep, _single)

        # Joint recipient selection is direction-bound: forward recipient T,
        # reverse recipient C. The token IDs, mask, identity tag and expected
        # full IDs are selected together so none can be mislabeled.
        up_heads = list(all_upstream_heads())
        for direction in ("forward", "reverse"):
            joint_recip = select_stage_c_joint_recipient(pair, direction)
            for rep in range(3):
                def _all(
                    c=cache,
                    binding=joint_recip,
                    direc=direction,
                    heads=up_heads,
                ):
                    logits = runner.run_head_patches(
                        c,
                        heads,
                        direc,  # type: ignore[arg-type]
                        recipient_input_ids=binding["recipient_input_ids"].to(device),
                        recipient_attention_mask=binding["recipient_attention_mask"].to(device),
                        recipient_prompt_tag=binding["recipient_tag"],
                        expected_full_ids=binding["expected_full_ids"],
                    )
                    return {
                        "direction": direc,
                        "recipient_tag": binding["recipient_tag"],
                        "n_heads": len(heads),
                        "logits_shape": list(logits.shape),
                    }

                time_arm(f"all_upstream_joint_{direction}", did, rep, _all)

        # Eight-head layer joint uses the same direction-bound binding.
        layer_heads = list(eight_head_layer_heads(EIGHT_HEAD_LAYER))
        for direction in ("forward", "reverse"):
            joint_recip = select_stage_c_joint_recipient(pair, direction)
            for rep in range(3):
                def _eight(
                    c=cache,
                    binding=joint_recip,
                    direc=direction,
                    heads=layer_heads,
                ):
                    logits = runner.run_head_patches(
                        c,
                        heads,
                        direc,  # type: ignore[arg-type]
                        recipient_input_ids=binding["recipient_input_ids"].to(device),
                        recipient_attention_mask=binding["recipient_attention_mask"].to(device),
                        recipient_prompt_tag=binding["recipient_tag"],
                        expected_full_ids=binding["expected_full_ids"],
                    )
                    return {
                        "direction": direc,
                        "recipient_tag": binding["recipient_tag"],
                        "layer": EIGHT_HEAD_LAYER,
                        "n_heads": len(heads),
                        "logits_shape": list(logits.shape),
                    }

                time_arm(f"eight_head_layer_joint_{direction}", did, rep, _eight)

        # Residual CT/TC at layers 8/15/19
        for layer in RESIDUAL_INJECTION_LAYERS:
            for hybrid in ("CT", "TC"):
                for rep in range(3):
                    def _res(
                        c=cache,
                        ids=recip_ids,
                        attn=recip_attn,
                        exp=expected_full,
                        L=layer,
                        hy=hybrid,
                    ):
                        logits = runner.run_boundary_hybrid(
                            c,
                            hy,  # type: ignore[arg-type]
                            recipient_input_ids=ids,
                            recipient_attention_mask=attn,
                            expected_tag="T",
                            expected_full_ids=exp,
                            layer=L,
                        )
                        return {
                            "layer": L,
                            "hybrid": hy,
                            "logits_shape": list(logits.shape),
                        }

                    time_arm(f"residual_{hybrid}_L{layer}", did, rep, _res)

        # Adapter off → TF → on → TF
        for rep in range(3):
            def _adapt(p=pair):
                batch = build_q1_prompt_batch(
                    tokenizer, p["rendered_T"], bos_convention="single", device=device
                )
                disable_adapter_layers(model)
                with torch.inference_mode():
                    teacher_forced_forward(
                        model,
                        batch["input_ids"],
                        batch["attention_mask"],
                        use_cache=False,
                        forward_authorization="scientific",
                    )
                enable_adapter_layers(model)
                with torch.inference_mode():
                    teacher_forced_forward(
                        model,
                        batch["input_ids"],
                        batch["attention_mask"],
                        use_cache=False,
                        forward_authorization="scientific",
                    )
                return {"adapter_cycle": "disable_tf_enable_tf"}

            time_arm("adapter_off_on_tf", did, rep, _adapt)

        # Greedy generation
        for rep in range(3):
            def _gen(p=pair):
                out = run_f3_generation(
                    model,
                    tokenizer,
                    p["rendered_T"],
                    full_payload=FULL_PAYLOAD,
                    bos_convention="single",
                    device=device,
                    max_new_tokens=40,
                    stop_ids=(107, 1),
                )
                return {
                    "n_new_tokens": out.n_new_tokens,
                    "first_token_id": out.first_token_id,
                    "first_token_ok": out.first_token_ok,
                    "stop_ids": list(out.stop_ids),
                }

            time_arm("greedy_generation", did, rep, _gen)

    # Summaries
    rebuilt: list[ArmTiming] = []
    for r in report["arm_timings"]:
        rebuilt.append(
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
        )
    report["arm_summary"] = summarize_arm_timings(rebuilt)
    report["arm_catalog"] = stage_c_arm_catalog()
    n_fail = sum(1 for r in rebuilt if not r.ok)
    report["n_arm_rows"] = len(rebuilt)
    report["n_arm_failures"] = n_fail
    report["incompleteness"] = [
        "Single exploratory checkpoint only — not F4 family calibration",
        "B2/double-BOS descriptive only — not run as scientific gate",
        "No held-out prompts; no confirmatory ASR/hypothesis publication",
        "No run_live_scientific_pipeline invocation",
    ]
    if n_fail == 0 and report["load"].get("ok"):
        report["verdict"] = "PASS"
    elif report["load"].get("ok") and n_fail < len(rebuilt):
        report["verdict"] = "REVISE"
        report["failures"].append(f"{n_fail} arm rows failed")
    else:
        report["verdict"] = "BLOCK"

    (evidence_dir / "STAGE_C_EVIDENCE.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    stage_b_vol.commit()
    deactivate_stage_c_scientific_flag()
    report["evidence_dir_on_volume"] = str(evidence_dir)
    return report


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


def main() -> int:
    if FIRST_STAGE_C_SESSION_CLOSED:
        print(json.dumps({
            "verdict": "BLOCK",
            "reason": "first Stage C session already consumed; new owner authorization required",
        }))
        return 2
    prior = _local_prior_stage_c_spend()
    if prior >= STAGE_C_CAP_USD:
        print(json.dumps({"verdict": "BLOCK", "reason": "Stage C cap already reached", "prior": prior}))
        return 2
    print(
        json.dumps(
            {
                "prior_stage_c_metered_usd": prior,
                "cap_usd": STAGE_C_CAP_USD,
                "image": INTENDED_IMAGE_ID,
            }
        ),
        flush=True,
    )
    with app.run():
        result = run_stage_c_calibration.remote()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "STAGE_C_FIRST_REAL_GEMMA_EXPLORATORY_EVIDENCE.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({
        "verdict": result.get("verdict"),
        "n_arm_rows": result.get("n_arm_rows"),
        "n_arm_failures": result.get("n_arm_failures"),
        "load_ok": (result.get("load") or {}).get("ok"),
        "dolly_ids": (result.get("prompts") or {}).get("dolly_ids"),
        "failures": result.get("failures"),
        "evidence": str(path),
    }, indent=2))
    return 0 if result.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
