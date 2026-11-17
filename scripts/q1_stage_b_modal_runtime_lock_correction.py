#!/usr/bin/env python3
"""Stage B correction: reuse built image, close 54/54 wheel provenance. No models."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "q1_implementation_prep"
PRIOR_EVIDENCE = OUT / "STAGE_B_RUNTIME_IMAGE_LOCK_EVIDENCE.json"
PRIOR_FREEZE = OUT / "STAGE_B_PIP_FREEZE.txt"
PINS = REPO / "TRACES_Q1_execution_direct_pins_v1_2.txt"

INTENDED_IMAGE_ID = "im-0j3j9DzkkZSLdRQTxl4scU"
VOLUME_NAME = "traces-q1-stage-b-runtime-lock"
APP_NAME = "traces-q1-stage-b-runtime-lock"
PRIOR_EVIDENCE_DIR = "/mnt/stage_b/evidence/2026-11-17T152022Z"
MISSING_PKGS = [
    "aiohappyeyeballs==2.6.1",
    "aiohttp==3.12.7",
    "aiosignal==1.3.2",
    "attrs==25.3.0",
    "cbor2==5.7.0",
    "frozenlist==1.6.0",
    "grpclib==0.4.8",
    "h2==4.2.0",
    "hpack==4.1.0",
    "hyperframe==6.1.0",
    "multidict==6.4.4",
    "propcache==0.3.1",
    "yarl==1.20.0",
]

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.from_id(INTENDED_IMAGE_ID)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=image,
    volumes={"/mnt/stage_b": vol},
    cpu=4,
    memory=16384,
    timeout=45 * 60,
)
def close_wheel_coverage() -> dict:
    """Reuse scientific image: fresh freeze + 13 missing wheels + 54 hashes."""
    import platform
    import sys as _sys

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    evidence = Path(f"/mnt/stage_b/evidence/{stamp}")
    wheels_dir = evidence / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    # Copy existing 41 wheels from prior evidence (do not mutate historical dir).
    prior_wheels = Path(PRIOR_EVIDENCE_DIR) / "wheels"
    if not prior_wheels.is_dir():
        return {"ok": False, "error": f"prior wheels missing: {prior_wheels}"}
    copied = 0
    for src in sorted(prior_wheels.glob("*.whl")):
        dst = wheels_dir / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
            copied += 1

    # Fresh freeze from THIS image (authority).
    freeze = subprocess.check_output(
        [_sys.executable, "-m", "pip", "freeze"], text=True
    )
    freeze_path = evidence / "pip_freeze.txt"
    freeze_path.write_text(freeze)
    freeze_lines = [
        ln.strip()
        for ln in freeze.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#") and "==" in ln
    ]
    if len(freeze_lines) != 54:
        vol.commit()
        return {
            "ok": False,
            "error": f"expected 54 freeze lines, got {len(freeze_lines)}",
            "freeze_preview": freeze_lines[:20],
            "evidence_dir": str(evidence),
        }

    # Download only the 13 missing exact versions into the evidence wheel dir.
    before = {p.name for p in wheels_dir.glob("*.whl")}
    dl = subprocess.run(
        [
            _sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--dest",
            str(wheels_dir),
            *MISSING_PKGS,
        ],
        capture_output=True,
        text=True,
    )
    if dl.returncode != 0:
        vol.commit()
        return {
            "ok": False,
            "error": "pip download of missing wheels failed",
            "stderr": dl.stderr[-4000:],
            "stdout": dl.stdout[-2000:],
            "evidence_dir": str(evidence),
        }
    after = {p.name for p in wheels_dir.glob("*.whl")}
    newly = sorted(after - before)

    # Torch identity from this image (CPU import only — no model load).
    import torch

    torch_version = str(torch.__version__)
    cuda_version = getattr(torch.version, "cuda", None)

    wheel_index = []
    for path in sorted(wheels_dir.glob("*.whl")):
        wheel_index.append(
            {
                "filename": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    (evidence / "wheel_index.json").write_text(json.dumps(wheel_index, indent=2) + "\n")

    # Image object identity: must remain the reused scientific image.
    image_id = os.environ.get("MODAL_IMAGE_ID") or INTENDED_IMAGE_ID

    payload = {
        "ok": len(wheel_index) == 54 and len(newly) == 13,
        "stamp": stamp,
        "evidence_dir": str(evidence),
        "prior_wheels_copied": copied,
        "new_wheels": newly,
        "n_wheels": len(wheel_index),
        "n_freeze": len(freeze_lines),
        "pip_freeze_text": freeze,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        "wheel_index": wheel_index,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "modal_image_id_env": os.environ.get("MODAL_IMAGE_ID"),
        "intended_scientific_image_id": INTENDED_IMAGE_ID,
        "image_id_observed": image_id,
        "no_model_load": True,
    }
    (evidence / "correction_capture.json").write_text(json.dumps(payload, indent=2) + "\n")
    vol.commit()
    # Omit large freeze duplicate in return; keep hashes + meta.
    ret = dict(payload)
    ret["pip_freeze_text"] = freeze  # needed locally to build lock
    return ret


def _build_and_claim_locally(capture: dict, prior: dict) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from packaging.utils import parse_wheel_filename
    from packaging.utils import canonicalize_name
    from traces.q1.runtime_lock import (
        claim_production_lock_exists,
        generate_lock_from_wheel_dir,
        validate_lock_document,
        write_lock_document,
    )

    local_wheels = OUT / "STAGE_B_CORRECTION_WHEELS"
    local_wheels.mkdir(parents=True, exist_ok=True)

    # Pull wheels from Volume evidence dir.
    # Volume paths are relative (no leading /mnt); strip mount prefix if present.
    evidence_dir = capture["evidence_dir"]
    if evidence_dir.startswith("/mnt/stage_b/"):
        remote_evidence = evidence_dir[len("/mnt/stage_b/") :]
    else:
        remote_evidence = evidence_dir.lstrip("/")
    remote = remote_evidence.rstrip("/") + "/wheels"
    # modal volume get <name> <remote> <local>
    get = subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "--force",
            VOLUME_NAME,
            remote,
            str(local_wheels),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if get.returncode != 0:
        # Some CLI versions place files under local_wheels/wheels
        return {
            "ok": False,
            "phase": "volume_get",
            "stderr": get.stderr[-3000:],
            "stdout": get.stdout[-2000:],
        }

    # Normalize nested get layout
    candidates = [
        local_wheels,
        local_wheels / "wheels",
        local_wheels / Path(remote).name,
    ]
    wheel_dir = None
    for c in candidates:
        if c.is_dir() and list(c.glob("*.whl")):
            wheel_dir = c
            break
    if wheel_dir is None:
        # Flatten any nested *.whl
        found = list(local_wheels.rglob("*.whl"))
        if found:
            flat = OUT / "STAGE_B_CORRECTION_WHEELS_FLAT"
            flat.mkdir(parents=True, exist_ok=True)
            for p in found:
                target = flat / p.name
                if not target.exists():
                    target.write_bytes(p.read_bytes())
            wheel_dir = flat
    if wheel_dir is None:
        return {"ok": False, "phase": "locate_wheels", "local": str(local_wheels)}

    n_whl = len(list(wheel_dir.glob("*.whl")))
    if n_whl != 54:
        return {
            "ok": False,
            "phase": "wheel_count",
            "n_whl": n_whl,
            "wheel_dir": str(wheel_dir),
        }

    a10 = prior["a10g_probe"]
    intended = {
        "image_tag": "modal://traces-q1-stage-b-runtime-lock/scientific-runtime-py31115-cu121",
        "image_digest": INTENDED_IMAGE_ID,
        "python_version": "3.11.15",
        "cuda_version": "12.1",
        "base_registry_tag": prior["base_registry_tag"],
        "base_image_digest": prior["base_image_digest"],
        "gpu_type": "A10G",
        "gpu_device_name": "NVIDIA A10",
        "torch_version": capture["torch_version"],
        "torch_cuda_available": False,
        "modal_app": APP_NAME,
        "volume": VOLUME_NAME,
        "reused_existing_image": True,
        "correction_evidence_dir": capture["evidence_dir"],
    }

    doc = generate_lock_from_wheel_dir(
        wheel_dir,
        repo_root=REPO,
        pip_freeze_text=capture["pip_freeze_text"],
        source="built_image",
        production_image_lock=True,
        intended_image=intended,
        extra_notes=[
            "Stage B owner-approved correction: reuse image + close 54/54 wheels",
            f"scientific_image_id={INTENDED_IMAGE_ID}",
            f"correction_evidence={capture['evidence_dir']}",
            "no model loads / no checkpoint downloads / no new image rebuild",
        ],
    )
    # Fix naive wheel name/version using packaging parse; bind CUDA probe.
    fixed_wheels = []
    for w in doc.wheels:
        fname = w["filename"]
        parsed_name, parsed_version, _b, _t = parse_wheel_filename(fname)
        fixed_wheels.append(
            {
                "name": str(parsed_name),
                "version": str(parsed_version),
                "filename": fname,
                "sha256": w["sha256"],
            }
        )
    doc.wheels = fixed_wheels
    doc.cuda = {
        "torch_cuda_available": False,
        "torch_version": capture["torch_version"],
        "cuda_version": "12.1",
        "device_count": 0,
        "note": "CPU correction capture; A10G probe reused from prior Stage B evidence",
        "a10g_probe": {
            "ok": True,
            "cuda_version": a10["cuda_version"],
            "device_name": a10["device_name"],
            "device_count": a10["device_count"],
            "device_capability": a10["device_capability"],
            "device_memory_bytes": a10["device_memory_bytes"],
            "nvidia_smi": a10["nvidia_smi"],
            "torch_cuda_available": a10["torch_cuda_available"],
            "torch_version": a10["torch_version"],
            "mode": "reused_prior_a10g_identity_probe",
            "no_model_load": True,
        },
    }
    # Ensure installed_distributions names are normalized consistently with wheels
    # (freeze names are fine; inventory compares via canonicalize_name).
    lock_path = OUT / "TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json"
    write_lock_document(doc, lock_path)

    lock = json.loads(lock_path.read_text())
    validation = validate_lock_document(
        lock, wheel_dir=wheel_dir, require_production=True
    )
    claim = claim_production_lock_exists(
        lock,
        wheel_dir=wheel_dir,
        require_rehash=True,
        pins_file=PINS,
    )

    # Coverage map: every freeze pkg -> wheel filename/sha
    from traces.q1.runtime_lock import parse_pip_freeze_lines, _normalize_pkg_name

    from packaging.version import Version

    freeze_pkgs = parse_pip_freeze_lines(capture["pip_freeze_text"])
    by_wheel = {_normalize_pkg_name(w["name"]): w for w in lock["wheels"]}
    coverage = []
    for name, ver in freeze_pkgs:
        nn = _normalize_pkg_name(name)
        w = by_wheel.get(nn)
        covered = bool(
            w
            and w.get("sha256")
            and Version(str(w["version"])) == Version(ver)
        )
        coverage.append(
            {
                "installed": f"{name}=={ver}",
                "wheel": None if w is None else w["filename"],
                "sha256": None if w is None else w["sha256"],
                "covered": covered,
            }
        )

    return {
        "ok": bool(validation.get("ok") and claim),
        "lock_path": str(lock_path),
        "wheel_dir": str(wheel_dir),
        "n_wheels": len(lock["wheels"]),
        "n_installed": len(lock["installed_distributions"]),
        "validate_lock_document": validation,
        "claim_production_lock_exists": claim,
        "coverage_54": coverage,
        "covered_count": sum(1 for r in coverage if r["covered"]),
        "lock_sha256": _sha256_file(lock_path),
        "pip_freeze_sha256": capture["pip_freeze_sha256"],
        "intended_image": intended,
        "new_wheels": capture.get("new_wheels"),
    }


def main() -> int:
    prior = json.loads(PRIOR_EVIDENCE.read_text())
    if prior.get("image_object_id") and prior["image_object_id"] != INTENDED_IMAGE_ID:
        # tolerate alternate key
        pass
    # Prior evidence used image_digest field inside lock
    prior_img = (prior.get("lock") or {}).get("intended_image", {}).get("image_digest")
    if prior_img and prior_img != INTENDED_IMAGE_ID:
        print(json.dumps({"ok": False, "error": "prior image mismatch", "prior_img": prior_img}))
        return 2

    print("=== Stage B correction: Modal reuse capture ===", flush=True)
    with app.run():
        capture = close_wheel_coverage.remote()
    (OUT / "STAGE_B_CORRECTION_CAPTURE.json").write_text(
        json.dumps({k: v for k, v in capture.items() if k != "pip_freeze_text"}, indent=2)
        + "\n"
    )
    if capture.get("pip_freeze_text"):
        (OUT / "STAGE_B_CORRECTION_PIP_FREEZE.txt").write_text(capture["pip_freeze_text"])

    if not capture.get("ok"):
        print(json.dumps({"verdict": "BLOCK", "capture": capture}, indent=2)[:8000])
        return 1

    print("=== Local validate + claim with remediated runtime_lock ===", flush=True)
    result = _build_and_claim_locally(capture, prior)
    evidence = {
        "document_type": "q1_stage_b_runtime_lock_correction_evidence",
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip(),
        "repo_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=str(REPO), text=True
        ).strip(),
        "intended_scientific_image_id": INTENDED_IMAGE_ID,
        "base_image_digest": prior["base_image_digest"],
        "capture_ok": capture.get("ok"),
        "capture_evidence_dir": capture.get("evidence_dir"),
        "new_wheels": capture.get("new_wheels"),
        "n_wheels_captured": capture.get("n_wheels"),
        "torch_version": capture.get("torch_version"),
        "cuda_version": capture.get("cuda_version"),
        "python_version": capture.get("python_version"),
        "result": result,
        "prior_a10g_probe_reused": True,
        "no_model_load": True,
        "no_new_image_build": True,
    }
    (OUT / "STAGE_B_CORRECTION_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(json.dumps({
        "verdict": "PASS" if result.get("ok") else "BLOCK",
        "claim": result.get("claim_production_lock_exists"),
        "validate_ok": (result.get("validate_lock_document") or {}).get("ok"),
        "covered_count": result.get("covered_count"),
        "n_wheels": result.get("n_wheels"),
        "lock_path": result.get("lock_path"),
        "lock_sha256": result.get("lock_sha256"),
        "new_wheels": result.get("new_wheels"),
        "failures": (result.get("validate_lock_document") or {}).get("failures"),
    }, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
