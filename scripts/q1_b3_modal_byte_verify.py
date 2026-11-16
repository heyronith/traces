#!/usr/bin/env python3
"""CPU-only Modal B3 checkpoint-byte verification entrypoint.

Local preflight (no Modal spend):
  PYTHONPATH=src python scripts/q1_b3_modal_byte_verify.py --mode preflight

Billable download (requires chat-confirmed $10 cap):
  export B3_OWNER_APPROVED_MAX_USD=10
  export B3_CHAT_CAP_CONFIRMED=1
  modal run scripts/q1_b3_modal_byte_verify.py::download_main

Uses Volume ``traces-q1-b3-checkpoint-bytes`` and Secret ``hf-token``.
No GPU, no from_pretrained, no inference, no production pin writes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

from traces.q1.b3_modal_byte_verify import (  # noqa: E402
    BILLABLE_CONFIRM_ENV,
    MODAL_APP_NAME,
    MODAL_CPU_CORES,
    MODAL_MEMORY_GIB,
    MODAL_MOUNT_PATH,
    MODAL_SECRET_NAME,
    MODAL_TIMEOUT_SEC,
    MODAL_VOLUME_NAME,
    OWNER_APPROVED_CAP_USD,
    OWNER_BUDGET_ENV,
    PROGRAM_VERSION,
    B3VerifyError,
    preflight_local,
    require_billable_gates,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==0.30.2",
        "safetensors==0.5.3",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(ROOT / "src"), remote_path="/root/src")
)

vol = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(MODAL_SECRET_NAME)
app = modal.App(MODAL_APP_NAME, image=image)


@app.function(
    cpu=MODAL_CPU_CORES,
    memory=int(MODAL_MEMORY_GIB * 1024),
    timeout=MODAL_TIMEOUT_SEC,
    volumes={MODAL_MOUNT_PATH: vol},
    secrets=[hf_secret],
    max_containers=1,
)
def b3_download_and_hash() -> dict:
    import os
    from pathlib import Path

    os.environ[OWNER_BUDGET_ENV] = str(int(OWNER_APPROVED_CAP_USD))
    os.environ[BILLABLE_CONFIRM_ENV] = "1"
    from traces.q1.b3_modal_byte_verify import (
        resolve_hf_token,
        run_materialize_and_verify,
        utc_now,
    )

    token = resolve_hf_token()
    root = Path(MODAL_MOUNT_PATH) / "snapshots"
    evidence = Path(MODAL_MOUNT_PATH) / "evidence" / utc_now().replace(":", "")
    evidence.mkdir(parents=True, exist_ok=True)
    result = run_materialize_and_verify(root=root, token=token, evidence_dir=evidence)
    vol.commit()
    result["evidence_dir"] = str(evidence)
    result["volume"] = MODAL_VOLUME_NAME
    return result


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=600,
    volumes={MODAL_MOUNT_PATH: vol},
    max_containers=1,
)
def b3_read_evidence(evidence_dir: str) -> dict:
    from pathlib import Path

    vol.reload()
    ed = Path(evidence_dir)
    out: dict = {"evidence_dir": evidence_dir, "files": {}}
    for name in (
        "B3_CANDIDATE_AUDIT_REPORT.json",
        "B3_CANDIDATE_CHECKPOINT_MANIFEST.json",
        "B3_CANDIDATE_SNAPSHOT_POINTER.json",
    ):
        p = ed / name
        if p.is_file():
            out["files"][name] = p.read_text(encoding="utf-8")
    return out


def _publish_result(result: dict, publish_dir: Path) -> None:
    publish_dir.mkdir(parents=True, exist_ok=True)
    out_path = publish_dir / "B3_MODAL_DOWNLOAD_JOB_RESULT.json"
    text = json.dumps(result, indent=2, sort_keys=True)
    text = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED_HF_TOKEN]", text)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    evidence_dir = result.get("evidence_dir") or ""
    if not evidence_dir:
        return
    artifacts = b3_read_evidence.remote(evidence_dir)
    for name, body in (artifacts.get("files") or {}).items():
        (publish_dir / name).write_text(body, encoding="utf-8")
    meta = {
        "published_from_volume": MODAL_VOLUME_NAME,
        "evidence_dir": evidence_dir,
        "files": sorted((artifacts.get("files") or {}).keys()),
        "candidate_manifest_sha256": result.get("candidate_manifest_sha256"),
        "ok": result.get("ok"),
    }
    (publish_dir / "B3_CANDIDATE_PUBLISH_META.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@app.local_entrypoint()
def download_main() -> None:
    """modal run scripts/q1_b3_modal_byte_verify.py::download_main"""
    os.environ.setdefault(OWNER_BUDGET_ENV, str(int(OWNER_APPROVED_CAP_USD)))
    os.environ.setdefault(BILLABLE_CONFIRM_ENV, "1")
    gates = require_billable_gates()
    print(
        json.dumps(
            {
                "launching": True,
                "owner_cap_usd": OWNER_APPROVED_CAP_USD,
                "gates": gates,
                "program_version": PROGRAM_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    result = b3_download_and_hash.remote()
    publish_dir = ROOT / "results/q1_implementation_prep"
    _publish_result(result, publish_dir)
    if not result.get("ok"):
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("preflight", "download-help"),
        default="preflight",
    )
    args = ap.parse_args()
    if args.mode == "preflight":
        print(json.dumps(preflight_local(), indent=2, sort_keys=True))
        return 0
    print(
        "For billable download use:\n"
        "  B3_OWNER_APPROVED_MAX_USD=10 B3_CHAT_CAP_CONFIRMED=1 \\\n"
        "  modal run scripts/q1_b3_modal_byte_verify.py::download_main",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    # When executed via `modal run …::download_main`, Modal handles entry.
    # Direct python execution → local preflight/help only.
    if any("download_main" in a for a in sys.argv):
        raise SystemExit(0)
    raise SystemExit(main())
