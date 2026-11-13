"""Frozen pilot dataset loader (discovery / validation / held-out).

Smoke prompts are never loaded as scientific evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from traces.prompts import (
    CONTROL_TAG,
    SMOKE_INSTRUCTIONS,
    TRIGGER_TAG,
    PromptExample,
    render_user_content,
)
from traces.reproducibility import hash_prompt_set, sha256_json

SplitName = Literal["discovery", "validation", "held_out"]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "pilot"


@dataclass(frozen=True)
class PilotInstruction:
    instruction_id: str
    instruction: str
    split: SplitName
    source: str = "authored_traces_pilot_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "instruction": self.instruction,
            "split": self.split,
            "source": self.source,
        }


def load_manifest(data_dir: Path | str | None = None) -> dict[str, Any]:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return json.loads((d / "manifest.json").read_text())


def load_instructions(data_dir: Path | str | None = None) -> list[dict[str, Any]]:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return json.loads((d / "instructions.json").read_text())


def load_splits(data_dir: Path | str | None = None) -> dict[str, list[dict[str, Any]]]:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return json.loads((d / "splits.json").read_text())


def verify_manifest_hashes(data_dir: Path | str | None = None) -> dict[str, bool]:
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    manifest = load_manifest(d)
    instructions = load_instructions(d)
    splits = load_splits(d)
    expected = manifest["hashes"]

    def sha(obj: Any) -> str:
        return sha256_json(obj)

    pool = [
        {"instruction_id": x["instruction_id"], "instruction": x["instruction"]}
        for x in instructions
    ]
    return {
        "instructions_ordered_pool": sha(pool) == expected["instructions_ordered_pool"],
        "discovery": sha(splits["discovery"]) == expected["discovery"],
        "validation": sha(splits["validation"]) == expected["validation"],
        "held_out": sha(splits["held_out"]) == expected["held_out"],
        "all_splits": sha(splits) == expected["all_splits"],
    }


def assert_no_smoke_leakage(instructions: Sequence[dict[str, Any] | PilotInstruction]) -> None:
    smoke = set(SMOKE_INSTRUCTIONS)
    for item in instructions:
        text = item["instruction"] if isinstance(item, dict) else item.instruction
        if text in smoke:
            raise RuntimeError(
                "Smoke prompt leaked into pilot scientific dataset; forbidden"
            )


def assert_split_disjoint(splits: dict[str, list[dict[str, Any]]]) -> None:
    ids = {}
    for name, rows in splits.items():
        for r in rows:
            iid = r["instruction_id"]
            if iid in ids:
                raise RuntimeError(
                    f"Split leakage: {iid} in both {ids[iid]} and {name}"
                )
            ids[iid] = name
    if len(ids) != 100:
        raise RuntimeError(f"Expected 100 unique instructions, got {len(ids)}")


def get_split_instructions(
    split: SplitName,
    data_dir: Path | str | None = None,
) -> list[PilotInstruction]:
    splits = load_splits(data_dir)
    assert_split_disjoint(splits)
    rows = splits[split]
    out = [
        PilotInstruction(
            instruction_id=r["instruction_id"],
            instruction=r["instruction"],
            split=split,  # type: ignore[arg-type]
        )
        for r in rows
    ]
    assert_no_smoke_leakage(out)
    return out


def expand_paired_prompts(
    instructions: Sequence[PilotInstruction],
    *,
    trigger_tag: str = TRIGGER_TAG,
    control_tag: str = CONTROL_TAG,
) -> list[PromptExample]:
    """Each instruction → trigger + control PromptExample."""
    out: list[PromptExample] = []
    for inst in instructions:
        for tag, tag_text in (("trigger", trigger_tag), ("control", control_tag)):
            out.append(
                PromptExample(
                    instruction_id=inst.instruction_id,
                    instruction=inst.instruction,
                    tag=tag,  # type: ignore[arg-type]
                    tag_text=tag_text,
                    user_content=render_user_content(inst.instruction, tag_text),
                )
            )
    return out


def split_prompt_hash(
    split: SplitName,
    data_dir: Path | str | None = None,
) -> str:
    prompts = expand_paired_prompts(get_split_instructions(split, data_dir))
    return hash_prompt_set([p.to_dict() for p in prompts])


def forbid_eval_split_for_discovery(split_used_for_ranking: SplitName) -> None:
    """Hard rule: held_out must never be used for component discovery/ranking."""
    if split_used_for_ranking == "held_out":
        raise RuntimeError(
            "Forbidden: pilot-evaluation (held_out) prompts cannot be used for "
            "component discovery or ranking"
        )


def dataset_summary(data_dir: Path | str | None = None) -> dict[str, Any]:
    manifest = load_manifest(data_dir)
    checks = verify_manifest_hashes(data_dir)
    splits = load_splits(data_dir)
    assert_split_disjoint(splits)
    assert_no_smoke_leakage([r for rows in splits.values() for r in rows])
    return {
        "manifest_name": manifest["name"],
        "hashes_ok": all(checks.values()),
        "hash_checks": checks,
        "manifest_hashes": manifest["hashes"],
        "alpaca_overlap": manifest["alpaca_overlap"],
        "selection_protocol": manifest["selection_protocol"],
        "paired_prompt_hashes": {
            "discovery": split_prompt_hash("discovery", data_dir),
            "validation": split_prompt_hash("validation", data_dir),
            "held_out": split_prompt_hash("held_out", data_dir),
        },
    }
