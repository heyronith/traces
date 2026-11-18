#!/usr/bin/env python3
"""B2 exploratory identities 1–9 — dedicated Modal runner (development only).

Standing non-main authorization. Binds identity-applicable F4 supplement v2.
Seeds 42–44, all 50 development prompts. No held-out / confirmatory / main.
Does not alter single-BOS exploratory stopping rule.
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
APP_NAME = "traces-q1-b2-identity-suite"
CAP_USD = 3.0
TIMEOUT_SEC = 45 * 60

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
    .add_local_file(
        str(
            ROOT
            / "results/q1_implementation_prep/F4_TOLERANCE_TABLE_IDENTITY_APPLICABLE_v2.json"
        ),
        remote_path=(
            "/root/repo/results/q1_implementation_prep/"
            "F4_TOLERANCE_TABLE_IDENTITY_APPLICABLE_v2.json"
        ),
        copy=False,
    )
)


def _metered(app_name: str) -> float:
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
    timeout=TIMEOUT_SEC,
    cpu=4,
    memory=16384,
    volumes={"/mnt/b3": b3_vol.read_only(), "/mnt/stage_b": stage_b_vol},
    max_containers=1,
)
def run_b2_identity_suite() -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.pop("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", None)
    os.environ.pop("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", None)
    sys.path.insert(0, "/root/repo/src")
    repo_root = Path("/root/repo")

    from traces.q1.b2_identity_exploratory import (
        PUBLISHED_F4_IDENTITY_APPLICABLE_SHA256,
        build_identity_artifact_payload,
        evaluate_identities_for_prompt,
        load_identity_applicable_f4_epsilons,
    )
    from traces.q1.f4_exploratory import (
        F4_SEEDS,
        L1523_ADAPTED_LAYERS,
        SEVEN_LORA_TARGETS,
        activate_f4_scientific_flag,
        deactivate_f4_scientific_flag,
        f4_model_unit,
        f4_subfolder,
        select_f4_development_prompts,
    )
    from traces.q1.frozen_artifacts import FROZEN_DEV_ID_SEQUENCE_SHA256
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
    evidence_dir = Path(f"/mnt/stage_b/evidence/b2_identity_{stamp}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "document_type": "q1_b2_exploratory_identity_suite_evidence",
        "stamp_utc": stamp,
        "f4_identity_applicable_sha256": PUBLISHED_F4_IDENTITY_APPLICABLE_SHA256,
        "held_out_consumed": False,
        "claims_main_experiment": False,
        "alters_single_bos_stopping_rule": False,
        "failures": [],
        "loads": [],
        "per_seed_prompt_summaries": [],
        "verdict": "BLOCK",
    }

    def persist() -> None:
        (evidence_dir / "B2_IDENTITY_EVIDENCE.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        stage_b_vol.commit()

    def fail(msg: str) -> dict[str, Any]:
        report["failures"].append(msg)
        report["verdict"] = "BLOCK"
        persist()
        deactivate_f4_scientific_flag()
        return report

    img = verify_running_image_identity(INTENDED_IMAGE_ID)
    report["modal_image"] = img
    if not img["ok"]:
        return fail(f"image mismatch: {img}")
    if (repo_root / "results/q1_selection/held_out_instructions.json").exists():
        return fail("held-out unexpectedly mounted")
    arts = verify_stage_c_repo_artifacts(repo_root)
    if not arts["ok"]:
        return fail("repo artifacts: " + "; ".join(arts["failures"]))

    prompts = select_f4_development_prompts(repo_root)
    if prompts["held_out_bytes_read"] != 0 or len(prompts["dolly_ids"]) != 50:
        return fail("prompt selection invalid")
    try:
        f4_eps = load_identity_applicable_f4_epsilons(repo_root)
    except Exception as e:  # noqa: BLE001
        return fail(f"F4 eps load failed: {e}")
    report["f4_epsilons"] = f4_eps

    from traces.q1.runtime_lock import (
        claim_production_lock_exists,
        load_lock_document,
        running_modal_image_matches_lock,
        verify_installed_runtime_environment,
    )
    from traces.q1.checkpoint_trust import authenticate_production_checkpoint_manifest

    lock_path = repo_root / RUNTIME_LOCK_RELPATH
    wheel_dir = Path(STAGE_B_WHEEL_DIR)
    pins_path = repo_root / "TRACES_Q1_execution_direct_pins_v1_2.txt"
    lock_doc = load_lock_document(lock_path)
    if not (
        claim_production_lock_exists(
            lock_doc,
            wheel_dir=wheel_dir,
            require_rehash=True,
            pins_file=pins_path,
            lock_path=lock_path,
        )
        and running_modal_image_matches_lock(lock_doc)
        and verify_installed_runtime_environment(
            pins_file=pins_path,
            intended_image=lock_doc.get("intended_image"),
            locked_doc=lock_doc,
        ).get("ok")
    ):
        return fail("runtime lock / env gate failed")
    trust = authenticate_production_checkpoint_manifest(repo_root=repo_root)
    if not trust.production_unlocked:
        return fail("checkpoint trust blocked")

    activate_f4_scientific_flag(verified_scope=True)
    import torch
    from traces.q1.runtime_spec import apply_determinism_settings, ensure_cublas_workspace_env

    ensure_cublas_workspace_env()
    apply_determinism_settings()
    if not torch.cuda.is_available() or "A10" not in torch.cuda.get_device_name(0):
        return fail("require CUDA A10/A10G")

    from traces.prompts import (
        CONTROL_TAG,
        TRIGGER_TAG,
        PromptExample,
        apply_chat_templates,
        render_user_content,
    )
    from traces.q1.bos import assert_leading_bos_count, encode_rendered_prompt
    from traces.q1.causal_runner import Q1CausalRunner
    from traces.q1.loader import load_q1_scientific_model
    from traces.q1.tag_position import derive_tag_alignment
    from traces.q1.empirical_evidence import authenticate_identity_suite_artifact

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
        return {
            "rendered_T": rendered[0].rendered,
            "rendered_C": rendered[1].rendered,
            "alignment": derive_tag_alignment(enc_t["input_ids"], enc_c["input_ids"]),
            "input_ids_T": enc_t["input_ids"],
            "attention_mask_T": enc_t["attention_mask"],
            "input_ids_C": enc_c["input_ids"],
            "attention_mask_C": enc_c["attention_mask"],
        }

    vol_base = Path(BASE_MATERIALIZED)
    vol_adapter_root = Path(ADAPTER_MATERIALIZED)
    device = torch.device("cuda:0")
    per_model: dict[str, list] = {f4_model_unit(s): [] for s in F4_SEEDS}
    t0 = time.perf_counter()

    for seed in F4_SEEDS:
        if time.perf_counter() - t0 > TIMEOUT_SEC - 180:
            return fail("timeout guard before seed")
        sub = f4_subfolder(seed)
        unit = f4_model_unit(seed)
        local_root = Path(f"/tmp/q1_b2_id_seed{seed}")
        if local_root.exists():
            import shutil as _sh

            _sh.rmtree(local_root)
        base_dir = local_root / "base_materialized"
        adapter_dir = local_root / "adapter_materialized"
        issues = _copy_tree_no_symlinks(vol_base, base_dir)
        issues.extend(_copy_tree_no_symlinks(vol_adapter_root / sub, adapter_dir / sub))
        if issues:
            return fail(f"copy issues seed{seed}: " + "; ".join(issues[:6]))
        unit_local = adapter_dir / sub
        adapter_cfg = json.loads((unit_local / "adapter_config.json").read_text())
        peft_targets = sorted(str(x) for x in (adapter_cfg.get("target_modules") or []))
        leaf = sorted({t.rsplit(".", 1)[-1] for t in peft_targets})
        if leaf != sorted(SEVEN_LORA_TARGETS):
            return fail(f"leaf targets drift seed{seed}")

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
        report["loads"].append(
            {
                "seed": seed,
                "unit": unit,
                "ok": True,
                "dtype": str(meta.get("torch_dtype")),
            }
        )
        runner = Q1CausalRunner(
            model,
            tokenizer,
            model_unit=unit,
            bos_convention="single",
            engineering_tiny=False,
        )
        for prow in prompts["prompts"]:
            did = int(prow["dolly_id"])
            if time.perf_counter() - t0 > TIMEOUT_SEC - 90:
                return fail(f"timeout mid-run seed={seed} prompt={did}")
            try:
                pair = build_pair(prow["instruction"], tokenizer)
                rows = evaluate_identities_for_prompt(
                    runner=runner,
                    model=model,
                    tokenizer=tokenizer,
                    pair=pair,
                    instruction_id=str(did),
                    device=device,
                    f4_eps=f4_eps,
                    family="l1523",
                )
                per_model[unit].append(rows)
                report["per_seed_prompt_summaries"].append(
                    {
                        "seed": seed,
                        "prompt_id": did,
                        "all_passed": all(bool(r.get("passed")) for r in rows),
                        "failed_ids": [
                            int(r["identity_id"])
                            for r in rows
                            if not bool(r.get("passed"))
                        ],
                    }
                )
            except Exception as e:  # noqa: BLE001
                return fail(f"identity eval failed seed={seed} prompt={did}: {e}")
        del model, tokenizer, runner
        torch.cuda.empty_cache()
        import shutil as _sh2

        _sh2.rmtree(local_root, ignore_errors=True)

    payload = build_identity_artifact_payload(
        per_model_results=per_model,
        f4_eps=f4_eps,
        f4_sha256=PUBLISHED_F4_IDENTITY_APPLICABLE_SHA256,
        dataset_sha=FROZEN_DEV_ID_SEQUENCE_SHA256,
    )
    art_path = evidence_dir / "IDENTITY_SUITE_ARTIFACT.json"
    # reuse write helper for canonical JSON+sha
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    digest = __import__("hashlib").sha256(text.encode()).hexdigest()
    art_path.write_text(text)
    art_path.with_suffix(".json.sha256").write_text(digest + "\n")
    prompt_ids = [str(x) for x in prompts["dolly_ids"]]
    auth = authenticate_identity_suite_artifact(
        path=art_path,
        claimed_sha256=digest,
        f4_artifact_path=repo_root
        / "results/q1_implementation_prep/F4_TOLERANCE_TABLE_IDENTITY_APPLICABLE_v2.json",
        f4_claimed_sha256=PUBLISHED_F4_IDENTITY_APPLICABLE_SHA256,
        frozen_prompt_ids=prompt_ids,
        repo_root=repo_root,
        check_production_unlock=False,
        reject_synthetic=True,
    )
    report["artifact"] = {
        "path": str(art_path),
        "sha256": digest,
        "all_passed": payload.get("all_passed"),
        "auth_ok": auth.ok,
        "auth_failures": list(auth.failures),
    }
    report["job_wall_seconds"] = time.perf_counter() - t0
    n_fail_prompts = sum(
        1 for s in report["per_seed_prompt_summaries"] if not s["all_passed"]
    )
    if auth.ok and payload.get("all_passed") and n_fail_prompts == 0:
        report["verdict"] = "PASS"
    elif auth.ok and not payload.get("all_passed"):
        report["verdict"] = "REVISE"
        report["failures"].append(
            f"identity suite not all_passed; failed_prompt_cells={n_fail_prompts}"
        )
    else:
        report["verdict"] = "BLOCK"
        report["failures"].extend(auth.failures[:20])
    persist()
    deactivate_f4_scientific_flag()
    return report


@app.local_entrypoint()
def main() -> None:
    pre = _metered(APP_NAME)
    if pre >= CAP_USD:
        raise SystemExit(f"identity suite already metered ${pre:.4f} >= cap")
    result = run_b2_identity_suite.remote()
    post = _metered(APP_NAME)
    out = {
        "verdict": result.get("verdict"),
        "failures": result.get("failures"),
        "artifact": result.get("artifact"),
        "job_wall_seconds": result.get("job_wall_seconds"),
        "pre_metered_usd": pre,
        "post_metered_usd": post,
        "delta_metered_usd": post - pre,
        "stamp_utc": result.get("stamp_utc"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "B2_IDENTITY_SUITE_EVIDENCE.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    (OUT / "B2_IDENTITY_SUITE_RUN_SUMMARY.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    print(json.dumps(out, indent=2, default=str))
    if result.get("verdict") == "BLOCK":
        raise SystemExit(1)
