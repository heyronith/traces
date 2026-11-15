"""F4 per-family identity-tolerance calibration infrastructure.

Empirical epsilon locking requires authorized model repeats. This module
implements the frozen formula on supplied tensors without performing inference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class FamilyEpsilon:
    family: str
    shapes: tuple[tuple[int, ...], ...]
    d_max_family: float
    epsilon_family: float
    per_model_maxima: Mapping[str, float]
    batching_padding_policy: str
    n_repeat_pairs: int = 3  # C(3,2)=3 pairwise comparisons among 3 repeats

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["shapes"] = [list(s) for s in self.shapes]
        d["per_model_maxima"] = dict(self.per_model_maxima)
        return d


def pairwise_max_abs_diff(tensors: Sequence[torch.Tensor]) -> float:
    """Largest absolute elementwise difference among all distinct pairs."""
    if len(tensors) < 2:
        raise ValueError("Need at least two repeat tensors")
    up = [t.detach().float().reshape(-1) for t in tensors]
    dmax = 0.0
    for i in range(len(up)):
        for j in range(i + 1, len(up)):
            if up[i].shape != up[j].shape:
                raise ValueError(
                    f"Shape mismatch in family repeats: {tuple(up[i].shape)} vs "
                    f"{tuple(up[j].shape)}"
                )
            d = (up[i] - up[j]).abs().max().item()
            dmax = max(dmax, float(d))
    return dmax


def epsilon_from_dmax(d_max_family: float) -> float:
    """Frozen rule: epsilon_family = max(1e-3, 10 * D_max_family)."""
    return max(1e-3, 10.0 * float(d_max_family))


@dataclass
class F4CalibrationTable:
    """Mutable builder; lock() freezes the table before confirmation."""

    batching_padding_policy: str
    families: dict[str, FamilyEpsilon] = field(default_factory=dict)
    locked: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def add_family(
        self,
        family: str,
        repeat_tensors_by_model: Mapping[str, Sequence[torch.Tensor]],
    ) -> FamilyEpsilon:
        if self.locked:
            raise RuntimeError("F4 table is locked; refusing mutation")
        per_model: dict[str, float] = {}
        shapes: list[tuple[int, ...]] = []
        global_dmax = 0.0
        for model, reps in repeat_tensors_by_model.items():
            if len(reps) != 3:
                raise ValueError(
                    f"F4 requires exactly three repeats; got {len(reps)} for {model}"
                )
            shapes.append(tuple(reps[0].shape))
            d = pairwise_max_abs_diff(reps)
            per_model[model] = d
            global_dmax = max(global_dmax, d)
        # Also compare across the pooled elementwise maxima already covered per model;
        # protocol: across all three exploratory models — take max of per-model D_max.
        eps = epsilon_from_dmax(global_dmax)
        rec = FamilyEpsilon(
            family=family,
            shapes=tuple(shapes),
            d_max_family=global_dmax,
            epsilon_family=eps,
            per_model_maxima=per_model,
            batching_padding_policy=self.batching_padding_policy,
        )
        self.families[family] = rec
        return rec

    def lock(self) -> None:
        if not self.families:
            raise RuntimeError("Refusing to lock empty F4 table")
        self.locked = True

    def epsilon_for(self, family: str) -> float:
        if family not in self.families:
            raise KeyError(f"Unknown F4 family {family!r}")
        return self.families[family].epsilon_family

    def to_dict(self) -> dict[str, Any]:
        return {
            "batching_padding_policy": self.batching_padding_policy,
            "locked": self.locked,
            "families": {k: v.to_dict() for k, v in self.families.items()},
            "meta": dict(self.meta),
            "formula": "epsilon_family = max(1e-3, 10 * D_max_family)",
        }


def within_family_tolerance(
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
) -> bool:
    if a.shape != b.shape:
        return False
    return bool((a.float() - b.float()).abs().max().item() <= float(epsilon))


def persist_epsilon_table(table: F4CalibrationTable, path: Path | str) -> dict[str, Any]:
    """Write locked F4 table and return SHA-256 of canonical JSON bytes."""
    if not table.locked:
        raise RuntimeError("Refuse to persist unlocked F4 table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = table.to_dict()
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.write_text(text)
    meta = {"path": str(path), "sha256": digest, "n_families": len(table.families)}
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return meta


def load_and_verify_epsilon_table(
    path: Path | str, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        side = sidecar.read_text().strip()
        if side != digest:
            raise RuntimeError(
                f"F4 table SHA mismatch vs sidecar: file={digest} sidecar={side}"
            )
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            f"F4 table SHA mismatch vs expected: file={digest} expected={expected_sha256}"
        )
    return json.loads(raw.decode("utf-8")), digest
