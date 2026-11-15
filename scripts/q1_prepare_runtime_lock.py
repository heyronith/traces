#!/usr/bin/env python3
"""Prepare Q1 hashed runtime-lock artifacts (CPU-only; no model inference).

Writes:
  - runtime_spec.json (batching/padding/determinism/F5)
  - direct_pins verification
  - optional pip freeze snapshot if an environment is provided
  - lock procedure checklist for the built Modal/image (wheel hashes)

Does not download models or run forwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traces.q1.runtime_spec import (  # noqa: E402
    DEFAULT_BATCHING,
    DEFAULT_DETERMINISM,
    apply_determinism_settings,
    verify_direct_pins_file,
    write_runtime_spec_document,
)
from traces.q1.workload import build_enumerator, write_enumerator  # noqa: E402


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/q1_implementation_prep",
    )
    ap.add_argument(
        "--pip-python",
        type=str,
        default="",
        help="Optional interpreter to snapshot pip freeze (must match pins ideally)",
    )
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pins = verify_direct_pins_file(ROOT)
    (out / "direct_pins_verification.json").write_text(json.dumps(pins, indent=2) + "\n")

    write_runtime_spec_document(out / "runtime_spec.json")
    apply_determinism_settings(DEFAULT_DETERMINISM)

    enum = build_enumerator(include_downstream56=True)
    write_enumerator(out / "recommended_workload_enumerator_v1_1.json")

    freeze_meta: dict = {"captured": False}
    if args.pip_python:
        proc = subprocess.run(
            [args.pip_python, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        freeze_path = out / "pip_freeze_snapshot.txt"
        freeze_path.write_text(proc.stdout)
        freeze_meta = {
            "captured": True,
            "python": args.pip_python,
            "sha256": _sha256_text(proc.stdout),
            "n_lines": len(proc.stdout.splitlines()),
            "note": (
                "Snapshot only — normative lock requires wheel hashes from the "
                "built scientific image under CPython 3.11.15 + direct pins"
            ),
        }

    procedure = {
        "artifact": "TRACES_Q1_runtime_lock_preparation",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "direct_pins_verification": pins,
        "batching_padding_policy_id": DEFAULT_BATCHING.policy_id(),
        "determinism": DEFAULT_DETERMINISM.__dict__,
        "pip_freeze_snapshot": freeze_meta,
        "workload_validation": enum["validation"],
        "built_image_lock_still_required": [
            "Build image with CPython 3.11.15 and TRACES_Q1_execution_direct_pins_v1_2.txt",
            "Record base image tag, CUDA toolkit, driver, GPU type (A10/A10G)",
            "pip freeze + `pip download`/`hashin` wheel SHA-256 for every package",
            "Confirm tokenizer class / is_fast on scientific load (authorized)",
            "Confirm attn_implementation=eager and torch_dtype=bfloat16 on load report",
            "Link lock file hash to direct-pins file hash 4a90dd71…",
        ],
        "no_model_inference_in_this_script": True,
    }
    (out / "runtime_lock_preparation.json").write_text(
        json.dumps(procedure, indent=2) + "\n"
    )
    print(json.dumps({"out_dir": str(out), "pins_match": pins["match"], "workload_ok": enum["validation"]["tf_matches_readiness_v1"]}, indent=2))


if __name__ == "__main__":
    main()
