#!/usr/bin/env python3
"""SLOT for Agent 1–authored / Agent 1–approved independent Volume verifier.

This file is intentionally a **non-executable placeholder**.

Integrity rules (Workstream 2):
- The independent auditor should author this verifier (or explicitly review and
  approve an implementation) so hashing / path checks / source-binding /
  manifest recomputation do **not** call the downloader's inventory helpers.
- Prefer Python stdlib crypto; use ``huggingface_hub.HfApi.get_paths_info`` only
  for source object identities at the pinned revisions.
- Cursor may later execute **this exact file** unchanged under a SHA-256 gate via
  ``scripts/q1_b3_modal_readonly_independent_audit.py`` against the existing
  Volume mounted ``read_only=True``.
- Until Agent 1 sets ``APPROVED_VERIFIER_SHA256`` in the runner and publishes the
  matching source digest, Modal execution must refuse to launch.

Required entrypoint once Agent 1 replaces this stub::

    def run_independent_volume_audit(
        *,
        mount_path: Path,
        pointer_path: Path,
        hf_token: str | None,
        volume_name: str,
        verifier_source_sha256: str,
        runner_attestation: dict,
    ) -> dict:
        ...
        return {
            "ok": bool,
            "verdict": "PASS" | "REVISE" | "BLOCK" | "INCOMPLETE",
            "n_files": 65,
            "files": [...],  # path, size, sha256, source OID, match
            "candidate_manifest_sha256": str | None,
            "section11": {...},
            "symlink_checks": {...},
            "failures": [...],
            "independent_pass_claimed": False,  # Cursor runner ≠ full independence
            ...
        }

Do not import ``traces.q1.b3_modal_byte_verify.inventory_materialized_snapshot``
or ``run_materialize_and_verify`` from the downloader path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Agent1VerifierNotReady(RuntimeError):
    """Raised until Agent 1 authors/approves a real verifier body."""


def run_independent_volume_audit(
    *,
    mount_path: Path,
    pointer_path: Path,
    hf_token: str | None,
    volume_name: str,
    verifier_source_sha256: str,
    runner_attestation: dict[str, Any],
) -> dict[str, Any]:
    raise Agent1VerifierNotReady(
        "Agent 1 has not yet authored or approved an independent Volume "
        f"verifier body (slot={__file__!s}, sha={verifier_source_sha256}, "
        f"mount={mount_path}, pointer={pointer_path}, volume={volume_name}). "
        "Refusing to emit a false PASS. Cursor must STOP before Modal launch."
    )


if __name__ == "__main__":
    raise SystemExit(
        "This is an Agent 1 verifier slot placeholder — not runnable until "
        "Agent 1 replaces the body and the runner SHA gate is opened."
    )
