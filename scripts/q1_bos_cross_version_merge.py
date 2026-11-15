#!/usr/bin/env python3
"""Archive: merge BOS cross-version raw token-ID result JSONs (CPU-only).

Reconstructs the comparison-merge script required by protocol §9.2 / §14 before
GPU execution. Consumes existing ``bos_cross_version_results_tf*.json`` files
only — no model or tokenizer loads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_map(raw: dict[str, Any], path_key: str) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for rec in raw["results"]:
        iid = str(rec["dolly_id"])
        block = rec[path_key]
        out[iid] = {
            "trigger": list(block["ids_trigger"]),
            "control": list(block["ids_control"]),
        }
    return out


def merge_bos_results(path_a: Path, path_b: Path) -> dict[str, Any]:
    raw_a = json.loads(path_a.read_text())
    raw_b = json.loads(path_b.read_text())
    report: dict[str, Any] = {
        "artifact": "q1_bos_cross_version_merge",
        "inputs": {
            "tf4513": {
                "path": str(path_a),
                "sha256": _sha256(path_a),
                "tokenizer_class": raw_a.get("tokenizer", {}).get("class"),
                "is_fast": raw_a.get("tokenizer", {}).get("is_fast"),
                "software": raw_a.get("software"),
            },
            "tf5230": {
                "path": str(path_b),
                "sha256": _sha256(path_b),
                "tokenizer_class": raw_b.get("tokenizer", {}).get("class"),
                "is_fast": raw_b.get("tokenizer", {}).get("is_fast"),
                "software": raw_b.get("software"),
            },
        },
        "cross_version_token_id_identity": {},
        "instruction_tok_lens": {
            "tf4513_n": raw_a.get("n_pairs"),
            "tf5230_n": raw_b.get("n_pairs"),
        },
    }
    lens_ok = 0
    for p in ("A", "B", "C"):
        map_a = _pair_map(raw_a, p)
        map_b = _pair_map(raw_b, p)
        ids = sorted(set(map_a) & set(map_b), key=lambda x: int(x))
        n_same = sum(1 for iid in ids if map_a[iid] == map_b[iid])
        diffs = [iid for iid in ids if map_a[iid] != map_b[iid]][:5]
        report["cross_version_token_id_identity"][p] = {
            "n_compared": len(ids),
            "n_identical_token_sequences": n_same,
            "n_differing": len(ids) - n_same,
            "all_identical": n_same == len(ids) and len(ids) > 0,
            "examples_if_any": diffs,
        }
    # Instruction length equality across versions (from B path lens)
    for rec_a, rec_b in zip(raw_a["results"], raw_b["results"]):
        if rec_a["dolly_id"] != rec_b["dolly_id"]:
            continue
        if (
            rec_a["instruction_tok_len_add_special_false"]
            == rec_b["instruction_tok_len_add_special_false"]
        ):
            lens_ok += 1
    report["instruction_tok_lens_identical_count"] = lens_ok
    report["instruction_tok_lens_identical_across_versions"] = lens_ok == len(
        raw_a["results"]
    )
    report["overall_status"] = (
        "PASS"
        if all(
            v["all_identical"]
            for v in report["cross_version_token_id_identity"].values()
        )
        and report["instruction_tok_lens_identical_across_versions"]
        else "FAIL"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tf4513",
        type=Path,
        default=Path("results/q1_final_cpu_checks/bos_cross_version_results_tf4513.json"),
    )
    ap.add_argument(
        "--tf5230",
        type=Path,
        default=Path("results/q1_final_cpu_checks/bos_cross_version_results_tf5230.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/q1_implementation_prep/bos_cross_version_merge.json"),
    )
    args = ap.parse_args()
    report = merge_bos_results(args.tf4513, args.tf5230)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    args.out.write_text(text)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sidecar = {
        "merge_report_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "merge_script_sha256": script_sha,
        "overall_status": report["overall_status"],
    }
    (args.out.parent / "bos_cross_version_merge_META.json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )
    print(json.dumps({"wrote": str(args.out), **sidecar}, indent=2))


if __name__ == "__main__":
    main()
