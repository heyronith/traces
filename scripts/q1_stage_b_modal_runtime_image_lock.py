#!/usr/bin/env python3
"""Stage B — owner-approved Modal scientific runtime image build + lock capture.

Hard bounds (chat-approved):
  - cumulative Stage B cap: $5 USD
  - CPU: 4 cores / 16 GiB / 45 min max
  - optional A10G identity probe: 10 min max
  - no model loads, forwards, checkpoint downloads, science, or main merge

  modal run scripts/q1_stage_b_modal_runtime_image_lock.py::main
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

# Inline frozen pin constants so the local Modal client need not import torch
# via traces.q1.runtime_spec (host may lack the scientific venv).
DIRECT_PINS_FILE = "TRACES_Q1_execution_direct_pins_v1_2.txt"
DIRECT_PINS_SHA256 = (
    "4a90dd7133719a751400c456db615e726a245d9a34073b7fb64243bf2a212048"
)

APP_NAME = "traces-q1-stage-b-runtime-lock"
VOLUME_NAME = "traces-q1-stage-b-runtime-lock"
MOUNT = "/mnt/stage_b"
PUBLISH_DIR = ROOT / "results" / "q1_implementation_prep"
PINS_PATH = ROOT / DIRECT_PINS_FILE

CPU_CORES = 4.0
MEMORY_MIB = 16 * 1024
TIMEOUT_SEC = 45 * 60
A10G_TIMEOUT_SEC = 10 * 60
STAGE_B_CAP_USD = 5.0
CPU_USD_PER_CORE_HOUR = 0.0473
MEM_USD_PER_GIB_HOUR = 0.008
A10G_USD_PER_HOUR = 1.10

BASE_REGISTRY_TAG = "python:3.11.15-slim-bookworm"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"

_hours = TIMEOUT_SEC / 3600.0
_cpu_bound = _hours * CPU_CORES * CPU_USD_PER_CORE_HOUR
_mem_bound = _hours * 16.0 * MEM_USD_PER_GIB_HOUR
_a10g_bound = (A10G_TIMEOUT_SEC / 3600.0) * A10G_USD_PER_HOUR
CONSERVATIVE_CEILING_USD = round(2.0 * (_cpu_bound + _mem_bound) + _a10g_bound + 1.0, 4)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def preflight_cost_ok() -> dict:
    pins_sha = _sha256_file(PINS_PATH)
    return {
        "stage_b_cap_usd": STAGE_B_CAP_USD,
        "conservative_ceiling_usd": CONSERVATIVE_CEILING_USD,
        "launch_permitted": CONSERVATIVE_CEILING_USD <= STAGE_B_CAP_USD,
        "pins_sha256": pins_sha,
        "pins_match": pins_sha == DIRECT_PINS_SHA256,
        "cpu_timeout_bound_usd": round(_cpu_bound + _mem_bound, 4),
        "a10g_timeout_bound_usd": round(_a10g_bound, 4),
    }


image = (
    modal.Image.from_registry(BASE_REGISTRY_TAG)
    .apt_install("git", "curl", "ca-certificates", "build-essential")
    .env(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": "/root/src",
        }
    )
    # copy=True so subsequent run_commands can use the pins file during image build.
    .add_local_file(
        str(PINS_PATH),
        remote_path=f"/root/pins/{DIRECT_PINS_FILE}",
        copy=True,
    )
    .run_commands(
        "python -m pip install --upgrade pip==25.0.1 setuptools wheel",
        (
            "mkdir -p /opt/stage_b_wheels && "
            f"python -m pip download -r /root/pins/{DIRECT_PINS_FILE} "
            f"-d /opt/stage_b_wheels --extra-index-url {TORCH_INDEX}"
        ),
        (
            "python -m pip install --no-index --find-links=/opt/stage_b_wheels "
            f"-r /root/pins/{DIRECT_PINS_FILE}"
        ),
        'python -c "import sys; v=sys.version_info; assert v[:3]==(3,11,15), sys.version"',
        (
            "python -c \"import torch,transformers,numpy,peft; "
            "assert torch.__version__.startswith('2.5.1'), torch.__version__; "
            "assert transformers.__version__=='4.51.3'; "
            "assert numpy.__version__=='2.1.3'; "
            "assert peft.__version__=='0.19.1'\""
        ),
    )
    # Mount source last (startup mount) for lock helpers; no further build steps.
    .add_local_dir(str(ROOT / "src"), remote_path="/root/src")
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _collect_lock_payload(intended_image: dict, wheel_dir: Path) -> dict:
    """Build + verify lock inside the container against real wheel bytes."""
    import platform
    import sys
    from importlib import metadata as importlib_metadata

    import torch

    from traces.q1.runtime_lock import (
        claim_production_lock_exists,
        generate_lock_from_wheel_dir,
        validate_lock_document,
    )

    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    pins_path = Path(f"/root/pins/{DIRECT_PINS_FILE}")
    pins_raw = pins_path.read_bytes()
    pins_digest = hashlib.sha256(pins_raw).hexdigest()
    pins_info = {
        "path": str(pins_path),
        "sha256": pins_digest,
        "expected_sha256": DIRECT_PINS_SHA256,
        "match": pins_digest == DIRECT_PINS_SHA256,
        "protocol_version": "1.3",
        "frozen_commit": "b1e8874f45311678e6fa241f504e7e09d4a7808c",
    }

    fake_root = Path("/tmp/traces_repo_root")
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / DIRECT_PINS_FILE).write_bytes(pins_raw)

    cuda_version = getattr(torch.version, "cuda", None)
    intended = dict(intended_image)
    intended["python_version"] = "3.11.15"
    intended["cuda_version"] = str(cuda_version) if cuda_version else "unavailable"
    intended["torch_version"] = str(torch.__version__)
    intended["torch_cuda_available"] = bool(torch.cuda.is_available())
    intended["base_registry_tag"] = BASE_REGISTRY_TAG

    doc = generate_lock_from_wheel_dir(
        wheel_dir,
        repo_root=fake_root,
        pip_freeze_text=freeze,
        source="built_image",
        production_image_lock=True,
        intended_image=intended,
        extra_notes=[
            "Stage B owner-approved Modal build-only lock",
            f"base_registry_tag={BASE_REGISTRY_TAG}",
            f"torch_index={TORCH_INDEX}",
            "no model loads / no checkpoint downloads",
        ],
    )
    payload = doc.to_dict()
    payload["direct_pins"] = pins_info
    payload["python"] = {
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "version_short": platform.python_version(),
    }
    payload["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    payload["cuda"] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "cuda_version": cuda_version,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        payload["cuda"]["device_name"] = torch.cuda.get_device_name(0)
        payload["cuda"]["device_capability"] = list(torch.cuda.get_device_capability(0))

    val = validate_lock_document(
        payload, wheel_dir=wheel_dir, require_production=True
    )
    claim = claim_production_lock_exists(
        payload,
        wheel_dir=wheel_dir,
        require_rehash=True,
        intended_image=payload["intended_image"],
        pins_file=pins_path,
    )
    installed = {
        dist.metadata["Name"]: dist.version
        for dist in importlib_metadata.distributions()
        if dist.metadata.get("Name")
    }
    return {
        "lock": payload,
        "validate_lock_document": val,
        "claim_production_lock_exists": claim,
        "pip_freeze": freeze,
        "installed_package_count": len(installed),
        "n_wheels": len(list(Path(wheel_dir).glob("*.whl"))),
        "python_version": platform.python_version(),
    }


@app.function(
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_SEC,
    volumes={MOUNT: vol},
    max_containers=1,
)
def stage_b_capture_lock(modal_image_object_id: str, base_image_digest: str) -> dict:
    import shutil
    from pathlib import Path

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wheel_src = Path("/opt/stage_b_wheels")
    evidence = Path(MOUNT) / "evidence" / started.replace(":", "")
    wheel_dst = evidence / "wheels"
    evidence.mkdir(parents=True, exist_ok=True)
    if wheel_dst.exists():
        shutil.rmtree(wheel_dst)
    shutil.copytree(wheel_src, wheel_dst)

    intended_image = {
        "image_tag": f"modal://{APP_NAME}/scientific-runtime-py31115-cu121",
        "image_digest": modal_image_object_id,
        "python_version": "3.11.15",
        "cuda_version": "pending",
        "base_registry_tag": BASE_REGISTRY_TAG,
        "base_image_digest": base_image_digest,
        "modal_app": APP_NAME,
        "volume": VOLUME_NAME,
    }
    result = _collect_lock_payload(intended_image, wheel_dst)
    lock_path = evidence / "TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE.json"
    lock_path.write_text(json.dumps(result["lock"], indent=2, sort_keys=True) + "\n")
    (evidence / "pip_freeze.txt").write_text(result["pip_freeze"])
    (evidence / "validate_lock_document.json").write_text(
        json.dumps(result["validate_lock_document"], indent=2, sort_keys=True) + "\n"
    )
    wheel_index = []
    for p in sorted(wheel_dst.glob("*.whl")):
        wheel_index.append(
            {
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
            }
        )
    (evidence / "wheel_index.json").write_text(
        json.dumps(wheel_index, indent=2, sort_keys=True) + "\n"
    )
    vol.commit()
    finished = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "ok": bool(result["claim_production_lock_exists"])
        and bool(result["validate_lock_document"].get("ok")),
        "started_utc": started,
        "finished_utc": finished,
        "evidence_dir": str(evidence),
        "lock_path": str(lock_path),
        "claim_production_lock_exists": result["claim_production_lock_exists"],
        "validate_lock_document": result["validate_lock_document"],
        "lock": result["lock"],
        "n_wheels": result["n_wheels"],
        "python_version": result["python_version"],
        "wheel_index": wheel_index,
        "pip_freeze": result["pip_freeze"],
        "gpu": None,
        "mode": "cpu_image_build_and_lock",
    }


@app.function(
    cpu=2.0,
    memory=8192,
    timeout=A10G_TIMEOUT_SEC,
    gpu="A10G",
    max_containers=1,
)
def stage_b_a10g_identity_probe() -> dict:
    """Identity-only GPU probe: torch CUDA device/driver fields; no model load."""
    import platform
    import sys

    import torch

    out: dict = {
        "ok": True,
        "mode": "a10g_identity_probe",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": getattr(torch.version, "cuda", None),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "no_model_load": True,
        "modules_imported": ["torch"],
    }
    if torch.cuda.is_available():
        out["device_name"] = torch.cuda.get_device_name(0)
        out["device_capability"] = list(torch.cuda.get_device_capability(0))
        out["device_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
        try:
            out["nvidia_smi"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            out["nvidia_smi_error"] = str(exc)
    else:
        out["ok"] = False
        out["failures"] = ["A10G probe: torch.cuda.is_available() is False"]
    banned = [m for m in sys.modules if m.startswith("transformers.models")]
    out["transformers_models_imported"] = banned
    if banned:
        out["ok"] = False
        out.setdefault("failures", []).append("unexpected transformers.models import")
    return out


def _redact(text: str) -> str:
    text = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED_HF_TOKEN]", text)
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)\S+", r"\1[REDACTED]", text)
    return text


def _fetch_base_digest() -> str:
    try:
        url = (
            "https://hub.docker.com/v2/repositories/library/python/tags/"
            "3.11.15-slim-bookworm"
        )
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        for img in data.get("images") or []:
            if img.get("architecture") == "amd64" and img.get("digest"):
                return str(img["digest"])
        if data.get("digest"):
            return str(data["digest"])
    except Exception as exc:  # noqa: BLE001
        return f"unresolved:{exc}"
    return "unresolved"


@app.local_entrypoint()
def main() -> None:
    pre = preflight_cost_ok()
    print(json.dumps({"preflight": pre}, indent=2, sort_keys=True))
    if not pre["launch_permitted"]:
        raise SystemExit(
            f"refusing launch: conservative ceiling ${pre['conservative_ceiling_usd']} "
            f"> cap ${STAGE_B_CAP_USD}"
        )
    if not pre["pins_match"]:
        raise SystemExit("direct pins SHA mismatch — refuse Stage B build")

    base_digest = _fetch_base_digest()
    try:
        image.hydrate()
    except Exception:
        pass
    modal_image_id = getattr(image, "object_id", None) or "pending-hydration"

    print(
        json.dumps(
            {
                "launching": True,
                "base_registry_tag": BASE_REGISTRY_TAG,
                "base_image_digest": base_digest,
                "modal_image_object_id": modal_image_id,
                "cap_usd": STAGE_B_CAP_USD,
                "conservative_ceiling_usd": CONSERVATIVE_CEILING_USD,
            },
            indent=2,
            sort_keys=True,
        )
    )

    cpu_result = stage_b_capture_lock.remote(str(modal_image_id), base_digest)
    # Prefer the hydrated image id after the function used the image.
    post_id = getattr(image, "object_id", None)
    if post_id and cpu_result.get("lock"):
        cpu_result["lock"]["intended_image"]["image_digest"] = str(post_id)
        modal_image_id = str(post_id)

    a10g_result = None
    if cpu_result.get("python_version") == "3.11.15":
        try:
            a10g_result = stage_b_a10g_identity_probe.remote()
            if a10g_result and cpu_result.get("lock"):
                lock = cpu_result["lock"]
                lock.setdefault("cuda", {})["a10g_probe"] = {
                    k: a10g_result.get(k)
                    for k in (
                        "ok",
                        "device_name",
                        "device_capability",
                        "device_memory_bytes",
                        "nvidia_smi",
                        "torch_cuda_available",
                        "cuda_version",
                        "failures",
                    )
                }
                if a10g_result.get("cuda_version"):
                    lock["intended_image"]["cuda_version"] = str(
                        a10g_result["cuda_version"]
                    )
                if a10g_result.get("device_name"):
                    lock["intended_image"]["gpu_type"] = "A10G"
                    lock["intended_image"]["gpu_device_name"] = a10g_result.get(
                        "device_name"
                    )
        except Exception as exc:  # noqa: BLE001
            a10g_result = {"ok": False, "error": str(exc)}

    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "document_type": "q1_stage_b_runtime_image_lock_evidence",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scientific_authorization": "OFF",
        "stage_b_cap_usd": STAGE_B_CAP_USD,
        "preflight": pre,
        "base_registry_tag": BASE_REGISTRY_TAG,
        "base_image_digest": base_digest,
        "modal_image_object_id": modal_image_id,
        "cpu_result_ok": cpu_result.get("ok"),
        "claim_production_lock_exists_in_container": cpu_result.get(
            "claim_production_lock_exists"
        ),
        "validate_lock_document": cpu_result.get("validate_lock_document"),
        "evidence_dir_on_volume": cpu_result.get("evidence_dir"),
        "n_wheels": cpu_result.get("n_wheels"),
        "python_version": cpu_result.get("python_version"),
        "a10g_probe": a10g_result,
        "lock": cpu_result.get("lock"),
        "wheel_index": cpu_result.get("wheel_index"),
        "pip_freeze": cpu_result.get("pip_freeze"),
        "notes": [
            "Wheels retained on Modal Volume for rehash; GitHub stores lock + index",
            "No checkpoint/model downloads; no scientific execution",
        ],
    }
    report_path = PUBLISH_DIR / "STAGE_B_RUNTIME_IMAGE_LOCK_EVIDENCE.json"
    text = _redact(json.dumps(out, indent=2, sort_keys=True))
    report_path.write_text(text + "\n", encoding="utf-8")
    if cpu_result.get("lock"):
        (PUBLISH_DIR / "TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE.json").write_text(
            _redact(json.dumps(cpu_result["lock"], indent=2, sort_keys=True)) + "\n",
            encoding="utf-8",
        )
    if cpu_result.get("wheel_index"):
        (PUBLISH_DIR / "STAGE_B_WHEEL_INDEX.json").write_text(
            json.dumps(cpu_result["wheel_index"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if cpu_result.get("pip_freeze"):
        (PUBLISH_DIR / "STAGE_B_PIP_FREEZE.txt").write_text(cpu_result["pip_freeze"])
    print(text)
    if not cpu_result.get("claim_production_lock_exists"):
        raise SystemExit(2)


if __name__ == "__main__":
    print(
        "Use:\n  modal run scripts/q1_stage_b_modal_runtime_image_lock.py::main",
        file=sys.stderr,
    )
    raise SystemExit(0)
