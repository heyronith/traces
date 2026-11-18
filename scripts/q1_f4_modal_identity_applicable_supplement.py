#!/usr/bin/env python3
"""F4 identity-applicable supplemental calibration — full residual/upstream tensors.

Preserves historical last-position ``F4_TOLERANCE_TABLE.json`` (SHA
80d7985c…). Standing non-main authorization. Same three exploratory seeds,
50 development prompts, T/C, three repeats, BF16/eager, formula
``max(1e-3, 10*D_max)``. Captures full-sequence boundary residual (L19) and
full-sequence upstream o_proj (L0–L14) for identities 2–6 applicability.

Does not call run_live_scientific_pipeline. No held-out / confirmatory / main.
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
APP_NAME = "traces-q1-f4-identity-applicable-supplement"
HISTORICAL_F4_SHA256 = (
    "80d7985c3d6241c5661cfacccbea2764c0dec429a9f637131d531727de41d021"
)
SUPPLEMENT_BATCHING_POLICY = (
    "bs1|no_pad|single_bos|eager|bf16|autocast_adapter_dtype=False|use_cache=False|"
    "boundary_residual_L19_full_seq|upstream_o_proj_L0_to_L14_full_seq_flat|"
    "final_logits_last_pos|decision_margin_last_pos"
)

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


def _metered_for_app(app_name: str) -> float:
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
        if (r.get("description") or "") == app_name
    )


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


@app.function(
    image=image,
    gpu="A10G",
    timeout=45 * 60,
    cpu=4,
    memory=16384,
    volumes={
        "/mnt/b3": b3_vol.read_only(),
        "/mnt/stage_b": stage_b_vol,
    },
    max_containers=1,
)
def run_f4_identity_applicable_supplement() -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.pop("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", None)
    os.environ.pop("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", None)

    sys.path.insert(0, "/root/repo/src")
    repo_root = Path("/root/repo")

    from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
    from traces.q1.f4_coverage_applicability import (
        coverage_verdict,
        extract_identity_applicable_family_tensors,
    )
    from traces.q1.f4_exploratory import (
        DEFAULT_F4_SCOPE,
        F4_BOUNDARY_LAYER,
        F4_CAP_USD,
        F4_N_REPEATS,
        F4_SEEDS,
        F4_UPSTREAM_BOUNDARY,
        FamilyAccum,
        INTENDED_IMAGE_ID as IMG,
        L1523_ADAPTED_LAYERS,
        SEVEN_LORA_TARGETS,
        activate_f4_scientific_flag,
        assert_f4_scope,
        build_f4_artifact_payload,
        deactivate_f4_scientific_flag,
        estimate_f4_workload,
        f4_model_unit,
        f4_subfolder,
        freeze_b_i_from_clean_logits,
        select_f4_development_prompts,
        write_f4_artifact,
    )
    from traces.q1.empirical_evidence import (
        REQUIRED_F4_TENSOR_FAMILIES,
        validate_f4_tensor_family_calibration,
    )
    from traces.q1.stage_c_exploratory import (
        ADAPTER_MATERIALIZED,
        BASE_MATERIALIZED,
        RUNTIME_LOCK_RELPATH,
        STAGE_B_WHEEL_DIR,
        verify_running_image_identity,
        verify_stage_c_repo_artifacts,
    )

    b3_vol.reload()
    stage_b_vol.reload()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    evidence_dir = Path(f"/mnt/stage_b/evidence/f4_{stamp}")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "document_type": "q1_f4_identity_applicable_supplement_evidence",
        "stamp_utc": stamp,
        "stage": "f4_identity_applicable_supplement",
        "historical_f4_sha256": HISTORICAL_F4_SHA256,
        "historical_f4_preserved": True,
        "coverage_cpu_verdict": coverage_verdict(),
        "claims_identities_1_9": False,
        "claims_double_bos": False,
        "claims_full_scientific_pipeline": False,
        "claims_main_experiment": False,
        "held_out_consumed": False,
        "failures": [],
        "skips": [],
        "identity": {},
        "budget": estimate_f4_workload(),
        "loads": [],
        "forward_timings": [],
        "prompts": {},
        "verdict": "BLOCK",
        "artifact": {},
    }

    def persist() -> None:
        (evidence_dir / "F4_SUPPLEMENT_EVIDENCE.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        stage_b_vol.commit()

    def fail(msg: str) -> dict[str, Any]:
        report["failures"].append(msg)
        report["verdict"] = "BLOCK"
        persist()
        deactivate_f4_scientific_flag()
        return report

    scope_fails = assert_f4_scope(DEFAULT_F4_SCOPE)
    img = verify_running_image_identity(IMG)
    report["identity"]["scope_failures"] = scope_fails
    report["identity"]["modal_image"] = img
    if scope_fails:
        return fail("scope drift: " + "; ".join(scope_fails))
    if not img["ok"]:
        return fail(
            f"MODAL_IMAGE_ID mismatch: observed={img['observed']} expected={img['expected']}"
        )

    held_out_mounted = (repo_root / "results/q1_selection/held_out_instructions.json").exists()
    if held_out_mounted:
        return fail("held-out instructions unexpectedly present in F4 mount")
    arts = verify_stage_c_repo_artifacts(repo_root)
    report["identity"]["repo_artifacts"] = arts
    if not arts["ok"]:
        return fail("repo artifacts: " + "; ".join(arts["failures"]))

    try:
        prompts = select_f4_development_prompts(repo_root)
    except Exception as e:  # noqa: BLE001
        return fail(f"prompt selection failed: {e}")
    report["prompts"] = {
        "n": prompts["n"],
        "dolly_ids": prompts["dolly_ids"],
        "full_development_id_sequence_sha256": prompts[
            "full_development_id_sequence_sha256"
        ],
        "held_out_bytes_read": prompts["held_out_bytes_read"],
    }
    if prompts.get("held_out_bytes_read", 0) != 0:
        return fail("held-out bytes were read — refuse")
    if len(prompts["dolly_ids"]) != 50:
        return fail(f"expected 50 development prompts, got {len(prompts['dolly_ids'])}")

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

    from traces.q1.checkpoint_trust import authenticate_production_checkpoint_manifest

    trust = authenticate_production_checkpoint_manifest(repo_root=repo_root)
    report["identity"]["checkpoint_trust"] = {
        "production_unlocked": bool(trust.production_unlocked),
        "failures": list(trust.failures or []),
    }
    if not trust.production_unlocked:
        return fail("checkpoint trust blocked: " + "; ".join(trust.failures))

    activate_f4_scientific_flag(verified_scope=True)

    import torch

    from traces.q1.runtime_spec import apply_determinism_settings, ensure_cublas_workspace_env

    ensure_cublas_workspace_env()
    det = apply_determinism_settings()
    report["identity"]["determinism"] = det
    if not torch.cuda.is_available():
        return fail("CUDA not available on F4 worker")
    gpu_name = torch.cuda.get_device_name(0)
    report["identity"]["gpu_name"] = gpu_name
    if "A10" not in gpu_name:
        return fail(f"unexpected GPU {gpu_name!r}; require A10/A10G")

    from traces.prompts import (
        CONTROL_TAG,
        TRIGGER_TAG,
        PromptExample,
        apply_chat_templates,
        render_user_content,
    )
    from traces.q1.bos import assert_leading_bos_count, encode_rendered_prompt
    from traces.q1.capture import ActivationCaptureManager
    from traces.q1.loader import load_q1_scientific_model
    from traces.q1.scoring import build_q1_prompt_batch, teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    vol_base = Path(BASE_MATERIALIZED)
    vol_adapter_root = Path(ADAPTER_MATERIALIZED)
    if not vol_base.is_dir():
        return fail(f"base materialized missing: {vol_base}")

    family_accums = {
        "final_logits": FamilyAccum(
            family="final_logits",
            batching_padding_policy=SUPPLEMENT_BATCHING_POLICY,
        ),
        "decision_margin": FamilyAccum(
            family="decision_margin",
            batching_padding_policy=SUPPLEMENT_BATCHING_POLICY,
        ),
        "boundary_residual": FamilyAccum(
            family="boundary_residual",
            allow_variable_shapes=True,
            batching_padding_policy=SUPPLEMENT_BATCHING_POLICY,
        ),
        "upstream_activation": FamilyAccum(
            family="upstream_activation",
            allow_variable_shapes=True,
            batching_padding_policy=SUPPLEMENT_BATCHING_POLICY,
        ),
    }
    residual_layers = (F4_BOUNDARY_LAYER,)
    o_proj_layers = tuple(range(0, F4_UPSTREAM_BOUNDARY))
    t_job0 = time.perf_counter()
    n_forwards_ok = 0
    n_forwards_fail = 0

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

    for seed in F4_SEEDS:
        if time.perf_counter() - t_job0 > (45 * 60 - 120):
            return fail("approaching hard timeout; stopping fail-closed before overrun")

        sub = f4_subfolder(seed)
        unit = f4_model_unit(seed)
        vol_unit = vol_adapter_root / sub
        if not vol_unit.is_dir():
            return fail(f"adapter unit missing on volume: {vol_unit}")

        local_root = Path(f"/tmp/q1_f4_ckpt_seed{seed}")
        if local_root.exists():
            import shutil as _sh

            _sh.rmtree(local_root)
        base_dir = local_root / "base_materialized"
        adapter_dir = local_root / "adapter_materialized"
        copy_issues = []
        # Copy base once per seed (simple; avoids cross-seed path reuse bugs)
        copy_issues.extend(_copy_tree_no_symlinks(vol_base, base_dir))
        unit_dst = adapter_dir / sub
        copy_issues.extend(_copy_tree_no_symlinks(vol_unit, unit_dst))
        if copy_issues:
            return fail(f"checkpoint copy issues seed{seed}: " + "; ".join(copy_issues[:8]))

        adapter_cfg = json.loads((unit_dst / "adapter_config.json").read_text())
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
            return fail(f"seed{seed} leaf targets {leaf_targets} != seven LoRA")
        if layer_ids != list(L1523_ADAPTED_LAYERS):
            return fail(f"seed{seed} layers {layer_ids} != {list(L1523_ADAPTED_LAYERS)}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t_load0 = time.perf_counter()
        try:
            model, tokenizer, load_report, meta = load_q1_scientific_model(
                subfolder=sub,
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
            return fail(f"model load failed seed{seed}: {e}")
        load_s = time.perf_counter() - t_load0
        peak_load = int(torch.cuda.max_memory_allocated())
        report["loads"].append(
            {
                "seed": seed,
                "model_unit": unit,
                "subfolder": sub,
                "wall_seconds": load_s,
                "peak_vram_bytes": peak_load,
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", None)),
                "torch_dtype": str(meta.get("torch_dtype") or getattr(model, "dtype", None)),
                "attn_implementation": meta.get("attn_implementation"),
                "base_revision": meta.get("base_revision") or PINNED_BASE_REVISION,
                "adapter_revision": meta.get("adapter_revision") or PINNED_ADAPTER_REVISION,
                "load_report_ok": bool(getattr(load_report, "ok", True)),
            }
        )
        if type(tokenizer).__name__ != "GemmaTokenizer":
            return fail(f"tokenizer class {type(tokenizer).__name__} != GemmaTokenizer")
        if getattr(tokenizer, "is_fast", True):
            return fail("tokenizer must be slow GemmaTokenizer")
        cfg_attn = getattr(getattr(model, "config", None), "_attn_implementation", None) or getattr(
            getattr(model, "config", None), "attn_implementation", None
        )
        if cfg_attn != "eager":
            return fail(f"attn_implementation {cfg_attn!r} != eager")

        device = torch.device("cuda:0")
        capture = ActivationCaptureManager(model)

        # Warmup on first prompt (untimed for stats)
        pair0 = build_pair(prompts["prompts"][0]["instruction"], tokenizer)
        for _ in range(2):
            batch = build_q1_prompt_batch(
                tokenizer, pair0["rendered_T"], bos_convention="single", device=device
            )
            with torch.inference_mode():
                teacher_forced_forward(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    use_cache=False,
                    forward_authorization="scientific",
                )

        for prow in prompts["prompts"]:
            did = int(prow["dolly_id"])
            pair = build_pair(prow["instruction"], tokenizer)
            rendered_by_tag = {"T": pair["rendered_T"], "C": pair["rendered_C"]}
            # Freeze b_i from first clean (C) intact pass for this prompt×model
            b_i: int | None = None

            for tag in ("C", "T"):  # C first so b_i is frozen before T
                for rep in range(F4_N_REPEATS):
                    if time.perf_counter() - t_job0 > (45 * 60 - 90):
                        return fail(
                            f"timeout guard mid-run seed={seed} prompt={did} tag={tag} rep={rep}"
                        )
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    detail: dict[str, Any] = {
                        "seed": seed,
                        "model_unit": unit,
                        "prompt_id": did,
                        "tag": tag,
                        "repeat": rep,
                    }
                    err = None
                    ok = True
                    try:
                        batch = build_q1_prompt_batch(
                            tokenizer,
                            rendered_by_tag[tag],
                            bos_convention="single",
                            device=device,
                        )
                        prompt_len = int(batch["prompt_len"])
                        with capture.capturing(
                            residual_layers=residual_layers,
                            o_proj_layers=o_proj_layers,
                        ) as buf:
                            with torch.inference_mode():
                                logits = teacher_forced_forward(
                                    model,
                                    batch["input_ids"],
                                    batch["attention_mask"],
                                    use_cache=False,
                                    forward_authorization="scientific",
                                )
                            buf.logits = logits.detach()
                        if capture.active_hook_count != 0:
                            raise RuntimeError("capture hook leak after intact forward")
                        if tag == "C" and rep == 0:
                            b_i = freeze_b_i_from_clean_logits(
                                logits,
                                prompt_len,
                                instruction_id=str(did),
                                model_unit=unit,
                            )
                        if b_i is None:
                            raise RuntimeError("b_i not frozen before family extract")
                        fams = extract_identity_applicable_family_tensors(
                            logits=logits,
                            residuals_by_layer=buf.residuals_by_layer,
                            o_proj_input_by_layer=buf.o_proj_input_by_layer,
                            prompt_len=prompt_len,
                            b_i=int(b_i),
                            boundary_layer=F4_BOUNDARY_LAYER,
                            upstream_boundary_layer=F4_UPSTREAM_BOUNDARY,
                        )
                        for fam, tensor in fams.items():
                            family_accums[fam].add_repeat(
                                model_unit=unit,
                                prompt_id=str(did),
                                tag=tag,
                                tensor=tensor,
                            )
                        detail.update(
                            {
                                "prompt_len": prompt_len,
                                "b_i": int(b_i),
                                "shapes": {k: list(v.shape) for k, v in fams.items()},
                            }
                        )
                        n_forwards_ok += 1
                    except Exception as e:  # noqa: BLE001
                        ok = False
                        err = str(e)
                        n_forwards_fail += 1
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - t0
                    peak = int(torch.cuda.max_memory_allocated())
                    report["forward_timings"].append(
                        {
                            "arm": "intact_tf_family_capture",
                            "seed": seed,
                            "prompt_dolly_id": did,
                            "tag": tag,
                            "repeat": rep,
                            "wall_seconds": wall,
                            "peak_vram_bytes": peak,
                            "ok": ok,
                            "detail": detail,
                            "error": err,
                        }
                    )
                    if not ok:
                        return fail(
                            f"intact forward failed seed={seed} prompt={did} "
                            f"tag={tag} rep={rep}: {err}"
                        )

        # Drop model before next seed
        del model, tokenizer, capture
        torch.cuda.empty_cache()
        import shutil as _sh2

        _sh2.rmtree(local_root, ignore_errors=True)

    if n_forwards_ok != 900 or n_forwards_fail != 0:
        return fail(
            f"forward coverage incomplete: ok={n_forwards_ok} fail={n_forwards_fail} expected=900"
        )

    tensor_families = {fam: acc.finalize() for fam, acc in family_accums.items()}
    payload = build_f4_artifact_payload(
        tensor_families=tensor_families,
        claims_real_experiment=True,
        fixture_purpose="exploratory_f4_identity_applicable_supplement_v2",
        meta={
            "batching_padding_policy": SUPPLEMENT_BATCHING_POLICY,
            "stamp_utc": stamp,
            "seeds": list(F4_SEEDS),
            "n_intact_forwards": n_forwards_ok,
            "image_id": INTENDED_IMAGE_ID,
            "historical_f4_sha256": HISTORICAL_F4_SHA256,
            "historical_f4_preserved": True,
            "capture_mode": "identity_applicable_full_sequence_residual_upstream",
            "supplement_version": 2,
        },
    )
    prompt_ids = [str(x) for x in prompts["dolly_ids"]]
    schema_fails = validate_f4_tensor_family_calibration(
        payload, frozen_prompt_ids=prompt_ids
    )
    report["artifact"]["schema_failures"] = schema_fails
    if schema_fails:
        # Still persist raw payload for diagnosis
        art_path = evidence_dir / "F4_TOLERANCE_TABLE_IDENTITY_APPLICABLE_v2.json"
        write_f4_artifact(art_path, payload)
        report["artifact"]["path"] = str(art_path)
        return fail("F4 schema validation failed: " + "; ".join(schema_fails[:20]))

    art_path = evidence_dir / "F4_TOLERANCE_TABLE_IDENTITY_APPLICABLE_v2.json"
    _, digest = write_f4_artifact(art_path, payload)
    (evidence_dir / "F4_TOLERANCE_TABLE_IDENTITY_APPLICABLE_v2.pretty.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    eps_table = {
        fam: {
            "d_max_family": tensor_families[fam]["d_max_family"],
            "epsilon_family": tensor_families[fam]["epsilon_family"],
            "shape": tensor_families[fam]["shape"],
            "dtype": tensor_families[fam]["dtype"],
            "n_comparisons": len(tensor_families[fam]["comparisons"]),
            "per_model_maxima": tensor_families[fam]["per_model_maxima"],
        }
        for fam in sorted(tensor_families)
    }
    report["artifact"] = {
        "path": str(art_path),
        "sha256": digest,
        "schema_failures": [],
        "epsilon_table": eps_table,
        "formula": payload["formula"],
        "locked": True,
        "n_comparisons_total": sum(
            len(tensor_families[f]["comparisons"]) for f in tensor_families
        ),
    }
    report["forward_summary"] = {
        "n_ok": n_forwards_ok,
        "n_fail": n_forwards_fail,
        "wall_mean_s": (
            sum(r["wall_seconds"] for r in report["forward_timings"] if r["ok"])
            / max(1, n_forwards_ok)
        ),
        "wall_max_s": max(
            (r["wall_seconds"] for r in report["forward_timings"] if r["ok"]), default=None
        ),
        "peak_vram_max_bytes": max(
            (r["peak_vram_bytes"] for r in report["forward_timings"] if r["ok"]),
            default=None,
        ),
        "job_wall_seconds": time.perf_counter() - t_job0,
    }
    report["verdict"] = "PASS"
    report["cap_usd_internal"] = F4_CAP_USD
    persist()
    deactivate_f4_scientific_flag()
    return report


@app.local_entrypoint()
def main() -> None:
    pre = _metered_for_app(APP_NAME)
    print(json.dumps({"pre_metered_usd": pre, "cap_usd": 2.0}, indent=2))
    if pre >= 2.0:
        raise SystemExit(f"F4 app already metered ${pre:.4f} >= $2 cap; refuse launch")
    result = run_f4_identity_applicable_supplement.remote()
    post = _metered_for_app(APP_NAME)
    out = {
        "result_verdict": result.get("verdict"),
        "result_failures": result.get("failures"),
        "artifact": result.get("artifact"),
        "forward_summary": result.get("forward_summary"),
        "loads": result.get("loads"),
        "pre_metered_usd": pre,
        "post_metered_usd": post,
        "delta_metered_usd": post - pre,
        "stamp_utc": result.get("stamp_utc"),
        "historical_f4_sha256": HISTORICAL_F4_SHA256,
        "identity_image": (result.get("identity") or {}).get("modal_image"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    local_path = OUT / "F4_IDENTITY_APPLICABLE_SUPPLEMENT_EVIDENCE.json"
    local_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    summary_path = OUT / "F4_IDENTITY_APPLICABLE_SUPPLEMENT_RUN_SUMMARY.json"
    summary_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out, indent=2, default=str))
    if result.get("verdict") != "PASS":
        raise SystemExit(1)
