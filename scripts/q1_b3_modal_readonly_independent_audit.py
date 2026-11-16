#!/usr/bin/env python3
"""Read-only Modal runner for an Agent 1–approved independent B3 Volume verifier.

Workstream 2 bridge only. This script:

* mounts ``traces-q1-b3-checkpoint-bytes`` with ``read_only=True``
* reloads the Volume before reading
* executes an Agent 1–authored or Agent 1–approved verifier **verbatim**
* refuses to run unless the verifier source SHA-256 matches the published gate
* returns a small JSON report to the caller (does **not** write the source Volume)

**STOP / integrity rules**

* Do **not** run this until Agent 1 has authored or explicitly approved the
  verifier and published its immutable source SHA-256.
* Cursor acting as Modal runner is **not** an independent PASS by itself.
* No GPU, no pretrained loads, no checkpoint redownload, no production pin.

Local dry checks (no Modal spend)::

  PYTHONPATH=src python scripts/q1_b3_modal_readonly_independent_audit.py --mode status

Approved run (only after Agent 1 gate is set)::

  B3_OWNER_APPROVED_MAX_USD=10 B3_CHAT_CAP_CONFIRMED=1 \\
    modal run scripts/q1_b3_modal_readonly_independent_audit.py::audit_main
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

from traces.q1.b3_modal_byte_verify import (  # noqa: E402
    BILLABLE_CONFIRM_ENV,
    MODAL_MOUNT_PATH,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_NAME,
    OWNER_APPROVED_CAP_USD,
    OWNER_BUDGET_ENV,
    B3VerifyError,
    estimate_modal_cpu_job_usd,
    require_billable_gates,
)

# Separate app so the downloader image/app is not reused as "independent".
MODAL_AUDIT_APP_NAME = "traces-q1-b3-readonly-independent-audit"

# Verifier slot — Agent 1 must set APPROVED_VERIFIER_RELPATH + SHA before run.
APPROVED_VERIFIER_RELPATH = (
    "scripts/q1_b3_agent1_independent_volume_verifier.py"
)
# Empty / placeholder until Agent 1 publishes the approved source digest.
APPROVED_VERIFIER_SHA256 = ""

# CPU-only, bounded under cumulative $10 B3 cap (hash-only; no download).
AUDIT_CPU_CORES = 2.0
AUDIT_MEMORY_GIB = 8.0
AUDIT_TIMEOUT_SEC = 45 * 60

PUBLISH_DIR = ROOT / "results" / "q1_implementation_prep"
POINTER_NAME = "B3_CANDIDATE_SNAPSHOT_POINTER.json"
REPORT_NAME = "B3_INDEPENDENT_VOLUME_READONLY_AUDIT.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verifier_gate_status() -> dict[str, Any]:
    path = ROOT / APPROVED_VERIFIER_RELPATH
    present = path.is_file()
    digest = sha256_file(path) if present else None
    approved = bool(APPROVED_VERIFIER_SHA256) and digest == APPROVED_VERIFIER_SHA256
    return {
        "relpath": APPROVED_VERIFIER_RELPATH,
        "present": present,
        "source_sha256": digest,
        "approved_sha256_gate": APPROVED_VERIFIER_SHA256 or None,
        "gate_open": approved,
        "notes": [
            "gate_open requires Agent 1–published APPROVED_VERIFIER_SHA256 match",
            "Cursor runner attestation is not an independent PASS",
        ],
    }


def assert_verifier_approved(path: Path) -> str:
    if not APPROVED_VERIFIER_SHA256:
        raise B3VerifyError(
            "APPROVED_VERIFIER_SHA256 is unset — Agent 1 must author/approve "
            "the verifier and publish its source SHA before any Modal run"
        )
    if not path.is_file():
        raise B3VerifyError(f"approved verifier missing: {path}")
    digest = sha256_file(path)
    if digest != APPROVED_VERIFIER_SHA256:
        raise B3VerifyError(
            f"verifier source SHA mismatch: got {digest} "
            f"expected {APPROVED_VERIFIER_SHA256} (refusing modified script)"
        )
    return digest


def load_verifier_module(path: Path):
    """Load verifier by path without importing downloader inventory helpers."""
    spec = importlib.util.spec_from_file_location(
        "q1_b3_agent1_independent_volume_verifier", path
    )
    if spec is None or spec.loader is None:
        raise B3VerifyError(f"cannot load verifier module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_independent_volume_audit"):
        raise B3VerifyError(
            "verifier must define run_independent_volume_audit(**kwargs) -> dict"
        )
    return mod


vol = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)
vol_ro = vol.with_mount_options(read_only=True)
hf_secret = modal.Secret.from_name(MODAL_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==0.30.2",
    )
    .env({"PYTHONPATH": "/root/src"})
    # Mount only the approved verifier file + minimal src needed by *that* script.
    # The verifier itself must not call downloader inventory; the runner still
    # ships src for pin constants if the approved script chooses stdlib-only.
    .add_local_file(
        str(ROOT / APPROVED_VERIFIER_RELPATH),
        remote_path=f"/root/{APPROVED_VERIFIER_RELPATH}",
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/src")
    .add_local_file(
        str(PUBLISH_DIR / POINTER_NAME),
        remote_path=f"/root/pointer/{POINTER_NAME}",
    )
)

app = modal.App(MODAL_AUDIT_APP_NAME, image=image)


@app.function(
    cpu=AUDIT_CPU_CORES,
    memory=int(AUDIT_MEMORY_GIB * 1024),
    timeout=AUDIT_TIMEOUT_SEC,
    volumes={MODAL_MOUNT_PATH: vol_ro},
    secrets=[hf_secret],
    max_containers=1,
)
def b3_readonly_independent_audit(verifier_sha256_gate: str) -> dict:
    """Execute approved verifier against read-only Volume; return JSON only."""
    import os
    from pathlib import Path

    # Re-check gate inside the container (exact bytes of mounted verifier file).
    verifier_path = Path("/root") / APPROVED_VERIFIER_RELPATH
    h = hashlib.sha256()
    with verifier_path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    if not verifier_sha256_gate or digest != verifier_sha256_gate:
        return {
            "ok": False,
            "failures": [
                f"in-container verifier SHA gate failed: got {digest} "
                f"expected {verifier_sha256_gate!r}"
            ],
            "runner": "cursor_modal_readonly_bridge",
            "independent_pass_claimed": False,
        }

    vol_ro.reload()

    spec = importlib.util.spec_from_file_location(
        "q1_b3_agent1_independent_volume_verifier", verifier_path
    )
    if spec is None or spec.loader is None:
        return {
            "ok": False,
            "failures": ["cannot load verifier in container"],
            "independent_pass_claimed": False,
        }
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pointer_path = Path("/root/pointer") / POINTER_NAME
    token = (
        (os.environ.get("HF_TOKEN") or "").strip()
        or (os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        or (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    )
    report = mod.run_independent_volume_audit(
        mount_path=Path(MODAL_MOUNT_PATH),
        pointer_path=pointer_path,
        hf_token=token or None,
        volume_name=MODAL_VOLUME_NAME,
        verifier_source_sha256=digest,
        runner_attestation={
            "runner": "cursor_modal_readonly_bridge",
            "app_name": MODAL_AUDIT_APP_NAME,
            "volume_read_only": True,
            "gpu": None,
            "download": False,
            "independent_pass_claimed": False,
            "limitation": (
                "Cursor executed Agent 1–approved verifier as delegated runner; "
                "this is not direct Agent 1 Modal access"
            ),
        },
    )
    if not isinstance(report, dict):
        return {
            "ok": False,
            "failures": ["verifier did not return a dict report"],
            "independent_pass_claimed": False,
        }
    report.setdefault("independent_pass_claimed", False)
    report["verifier_source_sha256"] = digest
    report["volume_read_only"] = True
    return report


def _redact(text: str) -> str:
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED_HF_TOKEN]", text)


@app.local_entrypoint()
def audit_main() -> None:
    """modal run …::audit_main — blocked until Agent 1 opens the SHA gate."""
    os.environ.setdefault(OWNER_BUDGET_ENV, str(int(OWNER_APPROVED_CAP_USD)))
    os.environ.setdefault(BILLABLE_CONFIRM_ENV, "1")
    gates = require_billable_gates()
    verifier_path = ROOT / APPROVED_VERIFIER_RELPATH
    digest = assert_verifier_approved(verifier_path)
    est = estimate_modal_cpu_job_usd(
        timeout_sec=AUDIT_TIMEOUT_SEC,
        cpu_cores=AUDIT_CPU_CORES,
        memory_gib=AUDIT_MEMORY_GIB,
    )
    print(
        json.dumps(
            {
                "launching_readonly_audit": True,
                "verifier_sha256": digest,
                "gates": gates,
                "cost_estimate": est,
                "volume": MODAL_VOLUME_NAME,
                "read_only": True,
                "independent_pass_claimed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    result = b3_readonly_independent_audit.remote(digest)
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PUBLISH_DIR / REPORT_NAME
    text = _redact(json.dumps(result, indent=2, sort_keys=True))
    out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result.get("ok"):
        raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("status", "help-run"),
        default="status",
    )
    args = ap.parse_args()
    if args.mode == "status":
        status = {
            "ok": True,
            "mode": "readonly_bridge_status",
            "app_name": MODAL_AUDIT_APP_NAME,
            "volume": MODAL_VOLUME_NAME,
            "mount_read_only": True,
            "verifier_gate": verifier_gate_status(),
            "publish_report_name": REPORT_NAME,
            "owner_cap_usd": OWNER_APPROVED_CAP_USD,
            "cost_estimate": estimate_modal_cpu_job_usd(
                timeout_sec=AUDIT_TIMEOUT_SEC,
                cpu_cores=AUDIT_CPU_CORES,
                memory_gib=AUDIT_MEMORY_GIB,
            ),
            "run_permitted": False,
            "notes": [
                "STOP: do not launch Modal audit until Agent 1 opens SHA gate",
                "Delegated Cursor execution ≠ independent PASS",
            ],
        }
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    print(
        "After Agent 1 sets APPROVED_VERIFIER_SHA256 and reviews the script:\n"
        "  B3_OWNER_APPROVED_MAX_USD=10 B3_CHAT_CAP_CONFIRMED=1 \\\n"
        "  modal run scripts/q1_b3_modal_readonly_independent_audit.py::audit_main",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    if any("audit_main" in a for a in sys.argv):
        raise SystemExit(0)
    raise SystemExit(main())
