"""Adapter-disabled reference helpers (no inference in this module).

Identity 6/7 require comparing adapter-on vs adapter-off activations and
base-model generation gates. Actual disable/enable must run under authorized
model execution; these helpers define the API and fail-closed checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Family = Literal["l19", "l1523"]


@dataclass(frozen=True)
class AdapterOffPlan:
    family: Family
    upstream_boundary_layer: int  # exclusive upper bound for equality region
    asr_threshold: float = 0.01
    notes: str = ""


def adapter_off_plan(family: Family) -> AdapterOffPlan:
    if family == "l19":
        return AdapterOffPlan(
            family=family,
            upstream_boundary_layer=19,
            notes="Identity 6: activations/residuals below layer 19 must match",
        )
    if family == "l1523":
        return AdapterOffPlan(
            family=family,
            upstream_boundary_layer=15,
            notes="F7: first adapted layer is 15; equality below 15",
        )
    raise ValueError(family)


def disable_adapter_layers(model: Any) -> None:
    """Disable PEFT adapter (fail closed if API missing). Does not run forward."""
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
        return
    if hasattr(model, "disable_adapters"):
        # peft context-manager style may differ; prefer explicit disable
        cm = model.disable_adapters()
        if hasattr(cm, "__enter__"):
            raise RuntimeError(
                "disable_adapters() returned a context manager; use "
                "disable_adapter_layers() for durable adapter-off reference passes"
            )
        return
    raise RuntimeError("Model has no disable_adapter_layers/disable_adapters API")


def enable_adapter_layers(model: Any) -> None:
    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()
        return
    if hasattr(model, "enable_adapters"):
        model.enable_adapters()
        return
    raise RuntimeError("Model has no enable_adapter_layers/enable_adapters API")
