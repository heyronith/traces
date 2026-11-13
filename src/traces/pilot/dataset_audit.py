"""Audit frozen pilot v1 instructions for training-corpus overlap (read-only).

Does **not** modify data/pilot/. Any correction requires a versioned v2 dataset
and updated pre-registration before execution.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from traces.pilot.dataset import DEFAULT_DATA_DIR, load_instructions, load_manifest
from traces.prompts import SMOKE_INSTRUCTIONS
from traces.reproducibility import sha256_json

NEAR_DUP_THRESHOLD = 0.92


def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def audit_internal_duplicates(data_dir: Path | str | None = None) -> dict[str, Any]:
    instructions = load_instructions(data_dir)
    norms = [_normalize(x["instruction"]) for x in instructions]
    exact = {}
    for i, n in enumerate(norms):
        exact.setdefault(n, []).append(instructions[i]["instruction_id"])
    exact_dups = {k: v for k, v in exact.items() if len(v) > 1}

    near = []
    for i in range(len(norms)):
        for j in range(i + 1, len(norms)):
            r = SequenceMatcher(None, norms[i], norms[j]).ratio()
            if r >= NEAR_DUP_THRESHOLD and norms[i] != norms[j]:
                near.append(
                    {
                        "a": instructions[i]["instruction_id"],
                        "b": instructions[j]["instruction_id"],
                        "ratio": r,
                    }
                )
    smoke_hit = [
        x["instruction_id"]
        for x in instructions
        if x["instruction"] in set(SMOKE_INSTRUCTIONS)
    ]
    return {
        "n_instructions": len(instructions),
        "exact_duplicate_groups": exact_dups,
        "near_duplicate_pairs": near,
        "smoke_exact_overlap_ids": smoke_hit,
        "frozen_manifest_hashes": load_manifest(data_dir)["hashes"],
        "dataset_mutated": False,
    }


def try_audit_alpaca_overlap(
    data_dir: Path | str | None = None,
    *,
    max_alpaca_rows: int = 20000,
) -> dict[str, Any]:
    """Best-effort exact/near overlap vs yahma/alpaca-cleaned if datasets is available.

    Records corpus id and retrieval attempt. Never writes a new frozen dataset.
    """
    instructions = load_instructions(data_dir)
    pilot_norms = {_normalize(x["instruction"]): x["instruction_id"] for x in instructions}
    report: dict[str, Any] = {
        "training_corpus": "yahma/alpaca-cleaned",
        "stated_training_source": "yahma/alpaca-cleaned (adapter card / prior Stage 0 docs)",
        "retrieval_status": None,
        "retrieval_revision": None,
        "exact_overlaps": [],
        "near_overlaps": [],
        "n_corpus_rows_scanned": 0,
        "overlap_conclusion": None,
        "note": (
            "Held-out n=25 is a small feasibility sample and cannot support precise "
            "ASR population estimates. Frozen v1 instructions were not modified. "
            "Zero scanned matches is NEVER evidence of no overlap when the corpus "
            "was not successfully retrieved and scanned."
        ),
    }
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        report["retrieval_status"] = f"datasets_unavailable: {type(e).__name__}"
        report["overlap_conclusion"] = "UNVERIFIED_CORPUS_INACCESSIBLE"
        report["explicit_non_claim"] = (
            "Overlap remains unverified. Do not interpret empty exact_overlaps as "
            "evidence of no training-corpus overlap."
        )
        return report

    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        rev = getattr(getattr(ds, "info", None), "version", None)
        # Prefer dataset fingerprint / homepage revision when available
        fp = getattr(ds, "_fingerprint", None)
        report["retrieval_revision"] = (
            str(rev) if rev is not None else (str(fp) if fp else "unknown_datasets_revision")
        )
        report["retrieval_status"] = "loaded"
        n = min(len(ds), max_alpaca_rows)
        report["n_corpus_rows_scanned"] = n
        for i in range(n):
            row = ds[i]
            instr = row.get("instruction") or row.get("input") or ""
            if not instr:
                continue
            nn = _normalize(str(instr))
            if nn in pilot_norms:
                report["exact_overlaps"].append(
                    {"pilot_id": pilot_norms[nn], "corpus_index": i}
                )
            # Sampled near-dup check against a subset of pilot for cost control
            if i % 50 == 0:
                for pn, pid in list(pilot_norms.items())[:20]:
                    r = SequenceMatcher(None, nn, pn).ratio()
                    if r >= NEAR_DUP_THRESHOLD and nn != pn:
                        report["near_overlaps"].append(
                            {"pilot_id": pid, "corpus_index": i, "ratio": r}
                        )
        if n == 0:
            report["overlap_conclusion"] = "UNVERIFIED_EMPTY_SCAN"
            report["explicit_non_claim"] = (
                "Zero rows scanned; empty overlaps are not evidence of no overlap."
            )
        else:
            report["overlap_conclusion"] = (
                "SCANNED_PARTIAL"
                if n < len(ds)
                else "SCANNED"
            )
            report["explicit_non_claim"] = (
                "Reported overlaps reflect the scanned subset only; absence of a hit "
                "outside the scanned window is not proven."
            )
    except Exception as e:
        report["retrieval_status"] = f"load_failed: {type(e).__name__}: {e}"
        report["overlap_conclusion"] = "UNVERIFIED_CORPUS_INACCESSIBLE"
        report["explicit_non_claim"] = (
            "Overlap remains unverified. Do not interpret empty exact_overlaps as "
            "evidence of no training-corpus overlap."
        )
    return report


def write_audit_report(out_path: Path | str, data_dir: Path | str | None = None) -> dict[str, Any]:
    payload = {
        "internal": audit_internal_duplicates(data_dir),
        "alpaca": try_audit_alpaca_overlap(data_dir),
        "policy": {
            "frozen_v1_immutable": True,
            "corrections_require": "versioned_v2_dataset_and_updated_preregistration",
            "held_out_limitation": (
                "n=25 held-out instructions cannot support precise ASR population estimates"
            ),
        },
        "audit_hash": None,
    }
    payload["audit_hash"] = sha256_json(
        {
            "internal_hashes": payload["internal"]["frozen_manifest_hashes"],
            "alpaca_status": payload["alpaca"]["retrieval_status"],
            "exact": payload["alpaca"]["exact_overlaps"],
        }
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
