"""Structured output recording schemas for Q1 conditions (no inference)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ConditionRecord:
    condition_id: str
    instruction_id: str
    model_unit: str
    bos_convention: str
    direction: str | None
    margin: float | None = None
    log_p_a: float | None = None
    top1_token_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationRecord:
    instruction_id: str
    model_unit: str
    tag: str
    bos_convention: str
    continuation_text: str
    first_token_id: int | None
    substring_fire: bool
    full_payload_fire: bool
    n_new_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResultSink:
    """In-memory sink; persist via dump_json under authorized runs."""

    def __init__(self) -> None:
        self.conditions: list[ConditionRecord] = []
        self.generations: list[GenerationRecord] = []

    def add_condition(self, rec: ConditionRecord) -> None:
        self.conditions.append(rec)

    def add_generation(self, rec: GenerationRecord) -> None:
        self.generations.append(rec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_conditions": len(self.conditions),
            "n_generations": len(self.generations),
            "conditions": [c.to_dict() for c in self.conditions],
            "generations": [g.to_dict() for g in self.generations],
        }
