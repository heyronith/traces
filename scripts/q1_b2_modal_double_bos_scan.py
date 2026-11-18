#!/usr/bin/env python3
"""B2 descriptive double-BOS 64-pair forward scan — dedicated Modal runner.

Development-only. Standing non-main authorization. Seeds 42–44 × 50 prompts ×
64 L15×L16 pairs. Own double-BOS clean references. Does **not** alter the
single-BOS exploratory stopping rule. No held-out / confirmatory / main.
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
APP_NAME = "traces-q1-b2-double-bos-scan"
CAP_USD = 5.0
TIMEOUT_SEC = 90 * 60

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


def _metered(app_name: str) -> float:
    try:
        raw = subprocess.check_output(
            ["modal", "billing", "report", "--for", "today", "--show-resources", "--json"],
            text=True,
        )
        return sum(
            float(r.get("cost") or 0)
            for r in json.loads(raw)
            if (r.get("description") or "") == app_name
        )
    except Exception:
        return 0.0


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
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            sp = Path(dirpath) / name
            if sp.is_symlink():
                issues.append(f"symlink file refused: {sp}")
                continue
            shutil.copyfile(sp, dst / rel / name, follow_symlinks=False)
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
def run_b2_double_bos_scan() -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.pop("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", None)
    sys.path.insert(0, "/root/repo/src")
    repo_root = Path("/root/repo")

    from traces.q1.analysis import rank_pairs_signed
    from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS
    from traces.q1.empirical_evidence import (
        ARTIFACT_DOUBLE_BOS_V1,
        DOUBLE_BOS_PAYLOAD_TOKEN_A,
        authenticate_double_bos_scan_artifact,
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
    from traces.q1.margin import PAYLOAD_TOKEN_A, clean_reference_token, decision_margin, last_prompt_logits
    from traces.q1.patching import HeadRef
    from traces.q1.stage_c_exploratory import (
        ADAPTER_MATERIALIZED,
        BASE_MATERIALIZED,
        RUNTIME_LOCK_RELPATH,
        STAGE_B_WHEEL_DIR,
        verify_running_image_identity,
        verify_stage_c_repo_artifacts,
    )

    assert PAYLOAD_TOKEN_A == DOUBLE_BOS_PAYLOAD_TOKEN_A
    b3_vol.reload()
    stage_b_vol.reload()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    evidence_dir = Path(f"/mnt/stage_b/evidence/b2_double_bos_{stamp}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "document_type": "q1_b2_descriptive_double_bos_64_pair_scan_evidence",
        "stamp_utc": stamp,
        "alters_single_bos_stopping_rule": False,
        "held_out_consumed": False,
        "failures": [],
        "verdict": "BLOCK",
    }

    def persist() -> None:
        (evidence_dir / "B2_DOUBLE_BOS_EVIDENCE.json").write_text(
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
        return fail(f"image mismatch {img}")
    if (repo_root / "results/q1_selection/held_out_instructions.json").exists():
        return fail("held-out mounted")
    arts = verify_stage_c_repo_artifacts(repo_root)
    if not arts["ok"]:
        return fail("repo artifacts failed")
    prompts = select_f4_development_prompts(repo_root)
    prompt_ids = [str(x) for x in prompts["dolly_ids"]]
    if len(prompt_ids) != 50 or prompts["held_out_bytes_read"] != 0:
        return fail("prompt selection invalid")

    from traces.q1.runtime_lock import (
        claim_production_lock_exists,
        load_lock_document,
        running_modal_image_matches_lock,
        verify_installed_runtime_environment,
    )
    from traces.q1.checkpoint_trust import authenticate_production_checkpoint_manifest

    lock_path = repo_root / RUNTIME_LOCK_RELPATH
    lock_doc = load_lock_document(lock_path)
    pins = repo_root / "TRACES_Q1_execution_direct_pins_v1_2.txt"
    wheel = Path(STAGE_B_WHEEL_DIR)
    if not (
        claim_production_lock_exists(
            lock_doc, wheel_dir=wheel, require_rehash=True, pins_file=pins, lock_path=lock_path
        )
        and running_modal_image_matches_lock(lock_doc)
        and verify_installed_runtime_environment(
            pins_file=pins, intended_image=lock_doc.get("intended_image"), locked_doc=lock_doc
        ).get("ok")
    ):
        return fail("runtime lock gate failed")
    if not authenticate_production_checkpoint_manifest(repo_root=repo_root).production_unlocked:
        return fail("checkpoint trust blocked")

    activate_f4_scientific_flag(verified_scope=True)
    import torch
    from traces.q1.runtime_spec import apply_determinism_settings, ensure_cublas_workspace_env
    from traces.prompts import CONTROL_TAG, TRIGGER_TAG, PromptExample, apply_chat_templates, render_user_content
    from traces.q1.bos import assert_leading_bos_count, encode_rendered_prompt
    from traces.q1.causal_runner import Q1CausalRunner
    from traces.q1.loader import load_q1_scientific_model
    from traces.q1.scoring import build_q1_prompt_batch, teacher_forced_forward
    from traces.q1.tag_position import derive_tag_alignment

    ensure_cublas_workspace_env()
    apply_determinism_settings()
    if not torch.cuda.is_available() or "A10" not in torch.cuda.get_device_name(0):
        return fail("need A10G")

    def build_pair_double(instruction: str, tokenizer_obj):
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
            tokenizer_obj, rendered[0].rendered, bos_convention="double"
        )
        enc_c = encode_rendered_prompt(
            tokenizer_obj, rendered[1].rendered, bos_convention="double"
        )
        assert_leading_bos_count(enc_t["input_ids"], bos_id=2, expected=2, context="T_double")
        assert_leading_bos_count(enc_c["input_ids"], bos_id=2, expected=2, context="C_double")
        return {
            "rendered_T": rendered[0].rendered,
            "rendered_C": rendered[1].rendered,
            "alignment": derive_tag_alignment(enc_t["input_ids"], enc_c["input_ids"]),
            "input_ids_T": enc_t["input_ids"],
            "attention_mask_T": enc_t["attention_mask"],
            "input_ids_C": enc_c["input_ids"],
            "attention_mask_C": enc_c["attention_mask"],
        }

    prescribed = sorted(EXPECTED_L15_L16_PAIRS)
    if len(prescribed) != 64:
        return fail(f"expected 64 pairs, got {len(prescribed)}")
    per_seed: dict[str, Any] = {}
    device = torch.device("cuda:0")
    t0 = time.perf_counter()
    n_patch = 0

    for seed in F4_SEEDS:
        unit = f4_model_unit(seed)
        sub = f4_subfolder(seed)
        local_root = Path(f"/tmp/q1_b2_db_seed{seed}")
        if local_root.exists():
            import shutil as _sh

            _sh.rmtree(local_root)
        base_dir = local_root / "base_materialized"
        adapter_dir = local_root / "adapter_materialized"
        issues = _copy_tree_no_symlinks(Path(BASE_MATERIALIZED), base_dir)
        issues.extend(
            _copy_tree_no_symlinks(Path(ADAPTER_MATERIALIZED) / sub, adapter_dir / sub)
        )
        if issues:
            return fail(f"copy issues: {issues[:5]}")
        unit_local = adapter_dir / sub
        peft_targets = sorted(
            str(x)
            for x in json.loads((unit_local / "adapter_config.json").read_text()).get(
                "target_modules"
            )
            or []
        )
        model, tokenizer, _, meta = load_q1_scientific_model(
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
        runner = Q1CausalRunner(
            model,
            tokenizer,
            model_unit=unit,
            bos_convention="double",
            engineering_tiny=False,
        )
        clean_margins: dict[str, float] = {}
        tt_margins: dict[str, float] = {}
        b_i_map: dict[str, int] = {}
        # pair_key -> {patched_margins, effects, eligible}
        pair_store: dict[str, dict[str, Any]] = {
            f"{a},{b}": {
                "patched_margins_by_prompt": {},
                "signed_effects_by_prompt": {},
                "eligible_by_prompt": {},
            }
            for a, b in prescribed
        }

        for prow in prompts["prompts"]:
            did = str(prow["dolly_id"])
            if time.perf_counter() - t0 > TIMEOUT_SEC - 120:
                return fail(f"timeout seed={seed} prompt={did}")
            pair = build_pair_double(prow["instruction"], tokenizer)
            cache = runner.capture_intact_pair(
                instruction_id=did,
                rendered_T=pair["rendered_T"],
                rendered_C=pair["rendered_C"],
                alignment=pair["alignment"],
                device=device,
            )
            prompt_len = int(pair["alignment"].seq_len)
            last_C = last_prompt_logits(cache.logits["C"], prompt_len)
            last_T = last_prompt_logits(cache.logits["T"], prompt_len)
            ref = clean_reference_token(
                last_C,
                instruction_id=did,
                model_unit=unit,
                bos_convention="double",
                payload_a=PAYLOAD_TOKEN_A,
            )
            b_i = int(ref.token_id)
            b_i_map[did] = b_i
            elig = b_i != int(PAYLOAD_TOKEN_A)
            clean_margins[did] = float(decision_margin(last_C, b_i, payload_a=PAYLOAD_TOKEN_A))
            tt_margins[did] = float(decision_margin(last_T, b_i, payload_a=PAYLOAD_TOKEN_A))

            recip = pair["input_ids_T"].to(device)
            attn = pair["attention_mask_T"].to(device)
            exp = recip[0].detach().cpu().tolist()
            for h15, h16 in prescribed:
                heads = [HeadRef(15, h15), HeadRef(16, h16)]
                logits = runner.run_head_patches(
                    cache,
                    heads,
                    "forward",
                    recipient_input_ids=recip,
                    recipient_attention_mask=attn,
                    recipient_prompt_tag="T",
                    expected_full_ids=exp,
                )
                m_pat = float(
                    decision_margin(
                        last_prompt_logits(logits, prompt_len),
                        b_i,
                        payload_a=PAYLOAD_TOKEN_A,
                    )
                )
                key = f"{h15},{h16}"
                pair_store[key]["patched_margins_by_prompt"][did] = m_pat
                pair_store[key]["signed_effects_by_prompt"][did] = float(
                    tt_margins[did] - m_pat
                )
                pair_store[key]["eligible_by_prompt"][did] = bool(elig)
                n_patch += 1

        means: dict[tuple[int, int], float] = {}
        for h15, h16 in prescribed:
            key = f"{h15},{h16}"
            vals = [
                float(pair_store[key]["signed_effects_by_prompt"][pid])
                for pid in prompt_ids
                if pair_store[key]["eligible_by_prompt"][pid]
            ]
            means[(h15, h16)] = sum(vals) / len(vals) if vals else 0.0
        ranking = [
            {"pair": list(p), "effect": e, "rank": r}
            for p, e, r in rank_pairs_signed(means)
        ]
        per_seed[unit] = {
            "prompt_ids": prompt_ids,
            "intact_double_bos_clean_margins": clean_margins,
            "intact_double_bos_tt_margins": tt_margins,
            "b_i_token_ids_by_prompt": b_i_map,
            "pairs": pair_store,
            "pair_ranking": ranking,
        }
        del model, tokenizer, runner
        torch.cuda.empty_cache()
        import shutil as _sh2

        _sh2.rmtree(local_root, ignore_errors=True)

    payload = {
        "artifact_type": ARTIFACT_DOUBLE_BOS_V1,
        "claims_real_experiment": True,
        "fixture_purpose": "exploratory_descriptive_double_bos_64_pair_scan",
        "model_units": [f4_model_unit(s) for s in F4_SEEDS],
        "dataset_binding": {
            "split": "development",
            "id_sequence_sha256": FROZEN_DEV_ID_SEQUENCE_SHA256,
        },
        "bos_convention": "double",
        "alters_single_bos_stopping_rule": False,
        "clean_reference_provenance": {
            "frozen": True,
            "source": "intact_double_bos_clean_run",
            "bos_convention": "double",
            "borrowed_single_bos": False,
        },
        "per_seed_results": per_seed,
        "n_pairs_recorded": 64,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    digest = __import__("hashlib").sha256(text.encode()).hexdigest()
    art = evidence_dir / "DOUBLE_BOS_SCAN_ARTIFACT.json"
    art.write_text(text)
    art.with_suffix(".json.sha256").write_text(digest + "\n")
    auth = authenticate_double_bos_scan_artifact(
        path=art,
        claimed_sha256=digest,
        frozen_prompt_ids=prompt_ids,
        repo_root=repo_root,
        check_production_unlock=False,
        reject_synthetic=True,
    )
    report["artifact"] = {
        "path": str(art),
        "sha256": digest,
        "auth_ok": auth.ok,
        "auth_failures": list(auth.failures),
        "n_patched_forwards": n_patch,
    }
    report["job_wall_seconds"] = time.perf_counter() - t0
    report["verdict"] = "PASS" if auth.ok else "BLOCK"
    if not auth.ok:
        report["failures"].extend(auth.failures[:20])
    persist()
    deactivate_f4_scientific_flag()
    return report


@app.local_entrypoint()
def main() -> None:
    pre = _metered(APP_NAME)
    if pre >= CAP_USD:
        raise SystemExit(f"double-BOS already metered ${pre:.4f} >= cap")
    result = run_b2_double_bos_scan.remote()
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
        "alters_single_bos_stopping_rule": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "B2_DOUBLE_BOS_SCAN_EVIDENCE.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    (OUT / "B2_DOUBLE_BOS_SCAN_RUN_SUMMARY.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    print(json.dumps(out, indent=2, default=str))
    if result.get("verdict") != "PASS":
        raise SystemExit(1)
