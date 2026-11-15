"""Immutable frozen scientific artifact SHA-256 baselines (protocol freeze).

Values are taken from ``TRACES_Q1_SCIENTIFIC_FREEZE_RECORD.md`` and verified
byte hashes of the freeze/authorization records themselves. Preflight must
compare working-tree content to these constants — not merely recompute hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence  # Any used by coerce/verify_runtime

# File-byte SHA-256 (freeze record + recomputed for freeze/auth documents)
FROZEN_FILE_SHA256: dict[str, str] = {
    "TRACES_Q1_preregistration_REVISED_REVIEW_DRAFT_v1_3.md": (
        "0a88d85bd4d94412f2568389dc16bdf51f7008cd01400e3e7ea0ab79de0d529b"
    ),
    "TRACES_Q1_execution_direct_pins_v1_2.txt": (
        "4a90dd7133719a751400c456db615e726a245d9a34073b7fb64243bf2a212048"
    ),
    "TRACES_Q1_SCIENTIFIC_FREEZE_RECORD.md": (
        "a05f0dfd62d8cb00c89775a86d1b3df45f7ef2eacaa71fcd1b6c90784bc969f9"
    ),
    "TRACES_Q1_v1_3_owner_freeze_authorization.md": (
        "f280aa7953da4fc72a1033ea1c95a4c384e536e22e026d793e41e90e77cb1d3c"
    ),
    "results/q1_selection/select_q1_dataset_v07.py": (
        "a80ce67b89ac03377a73a6bf7ca07fdd45da318d98b97c93a57c27f01bd9cf10"
    ),
    "results/q1_selection/development_instructions.json": (
        "e61c883dec3dc4a48f364c035fcda609f282f4302f9ec651b59a6aab4fab847c"
    ),
    "results/q1_selection/held_out_instructions.json": (
        "c0314d2ee0b4b004623b27bfdfa58c3a1f5c37985e381f79099bf737c058ca9e"
    ),
    "results/q1_selection/selection_manifest.json": (
        "01f5a81ce6d667920af5b0cb92e79e8416ecfdccf85a7396dddeb65095f71f33"
    ),
}

# Ordered ID-sequence hashes (dolly_id lists), not JSON file hashes
FROZEN_DEV_ID_SEQUENCE_SHA256 = (
    "6ea712c08207858826037fd7c78067b6c3af852b89783fcc339fe3f17aaa1a43"
)
FROZEN_HELD_OUT_ID_SEQUENCE_SHA256 = (
    "13c2c5a396ab58f9492e1161707a2acb8e4d242a058c9823470fa38bcca48b3a"
)

EXPECTED_N_DEV = 50
EXPECTED_N_HELD_OUT = 200

# Exact three exploratory seeds required before confirmatory progression
REQUIRED_EXPLORATORY_SEEDS = (42, 43, 44)
REQUIRED_PRIMARY_SEEDS = (42, 43, 44, 45, 46)
REQUIRED_SECONDARY_SEEDS = (45, 46)


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_id_sequence_sha256(dolly_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(dolly_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_ordered_dolly_ids(path: Path | str) -> list[int]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected list JSON in {path}")
    ids: list[int] = []
    for row in data:
        if not isinstance(row, dict) or "dolly_id" not in row:
            raise ValueError(f"row missing dolly_id in {path}")
        ids.append(int(row["dolly_id"]))
    return ids


def verify_frozen_artifacts(repo_root: Path | str) -> dict[str, Any]:
    """Compare working-tree bytes and ID sequences to immutable freeze baselines."""
    root = Path(repo_root)
    failures: list[str] = []
    checked: dict[str, Any] = {}

    for rel, expected in FROZEN_FILE_SHA256.items():
        path = root / rel
        if not path.is_file():
            failures.append(f"missing frozen file: {rel}")
            checked[rel] = {"missing": True}
            continue
        actual = sha256_file(path)
        ok = actual == expected
        checked[rel] = {"sha256": actual, "expected": expected, "match": ok}
        if not ok:
            failures.append(f"SHA mismatch {rel}: {actual} != {expected}")

    # ID-sequence integrity (rejects reordered IDs even if file hash were forged)
    for rel, expected_n, expected_seq in (
        (
            "results/q1_selection/development_instructions.json",
            EXPECTED_N_DEV,
            FROZEN_DEV_ID_SEQUENCE_SHA256,
        ),
        (
            "results/q1_selection/held_out_instructions.json",
            EXPECTED_N_HELD_OUT,
            FROZEN_HELD_OUT_ID_SEQUENCE_SHA256,
        ),
    ):
        path = root / rel
        key = f"{rel}#id_sequence"
        if not path.is_file():
            failures.append(f"missing for ID-sequence check: {rel}")
            continue
        try:
            ids = load_ordered_dolly_ids(path)
        except ValueError as e:
            failures.append(str(e))
            continue
        if len(ids) != expected_n:
            failures.append(f"{rel}: n={len(ids)} != {expected_n}")
        if len(ids) != len(set(ids)):
            failures.append(f"{rel}: duplicate dolly_ids")
        seq = ordered_id_sequence_sha256(ids)
        ok = seq == expected_seq
        checked[key] = {
            "sha256": seq,
            "expected": expected_seq,
            "n": len(ids),
            "match": ok,
        }
        if not ok:
            failures.append(
                f"ID-sequence SHA mismatch {rel}: {seq} != {expected_seq} "
                "(reordered or substituted IDs)"
            )

    return {"ok": len(failures) == 0, "failures": failures, "checked": checked}


def parse_direct_pin_requirements(pins_text: str) -> dict[str, str]:
    """Parse ``name==version`` lines from the frozen direct-pins file."""
    out: dict[str, str] = {}
    for line in pins_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, ver = line.split("==", 1)
            out[name.strip().lower().replace("_", "-")] = ver.strip()
    return out


def coerce_frozen_dolly_id(value: Any) -> int:
    """Coerce a runtime ID to the authoritative frozen ``dolly_id`` (int).

    Accepts ``int`` or an exact decimal digit string with no sign/whitespace.
    Rejects UUIDs, instruction text, prefixed IDs, and other interchangeable
    lookalikes — dolly_id strings and arbitrary instruction IDs are **not**
    treated as interchangeable.
    """
    if isinstance(value, bool):
        raise ValueError("boolean is not a frozen dolly_id")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"negative dolly_id {value}")
        return value
    if isinstance(value, str):
        if not value.isdigit() or value != str(int(value)):
            # reject "07077", "id:7077", "7077a", etc.
            raise ValueError(
                f"runtime ID {value!r} is not an exact decimal dolly_id string"
            )
        return int(value)
    raise ValueError(f"runtime ID type {type(value).__name__} not a frozen dolly_id")


def verify_runtime_allocation_ids(
    repo_root: Path | str,
    *,
    held_out_ids: Sequence[Any] | None = None,
    development_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compare supplied allocation ID sequences to frozen ordered dolly_ids.

    Length alone is insufficient: reordered, substituted, duplicated, or
    truncated sequences fail.
    """
    root = Path(repo_root)
    failures: list[str] = []
    checks: dict[str, Any] = {}

    def _check(
        label: str,
        rel: str,
        supplied: Sequence[Any] | None,
        expected_n: int,
        expected_seq_sha: str,
    ) -> None:
        if supplied is None:
            failures.append(f"{label}: allocation IDs not supplied")
            checks[label] = {"ok": False, "reason": "missing"}
            return
        path = root / rel
        frozen = load_ordered_dolly_ids(path)
        try:
            got = [coerce_frozen_dolly_id(x) for x in supplied]
        except ValueError as e:
            failures.append(f"{label}: {e}")
            checks[label] = {"ok": False, "reason": str(e)}
            return
        if len(got) != expected_n:
            failures.append(f"{label}: n={len(got)} != frozen {expected_n}")
        if len(got) != len(set(got)):
            failures.append(f"{label}: duplicate IDs")
        if got != frozen:
            failures.append(
                f"{label}: ID sequence does not match frozen allocation order "
                f"(reorder/substitution/truncation)"
            )
        seq = ordered_id_sequence_sha256(got)
        ok = (
            len(got) == expected_n
            and len(got) == len(set(got))
            and got == frozen
            and seq == expected_seq_sha
        )
        checks[label] = {
            "ok": ok,
            "n": len(got),
            "sequence_sha256": seq,
            "expected_sequence_sha256": expected_seq_sha,
        }
        if not ok and seq != expected_seq_sha and got == frozen:
            failures.append(f"{label}: sequence SHA mismatch")

    _check(
        "held_out",
        "results/q1_selection/held_out_instructions.json",
        held_out_ids,
        EXPECTED_N_HELD_OUT,
        FROZEN_HELD_OUT_ID_SEQUENCE_SHA256,
    )
    if development_ids is not None:
        _check(
            "development",
            "results/q1_selection/development_instructions.json",
            development_ids,
            EXPECTED_N_DEV,
            FROZEN_DEV_ID_SEQUENCE_SHA256,
        )

    return {"ok": len(failures) == 0, "failures": failures, "checks": checks}
