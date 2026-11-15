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
    """Disable PEFT adapter via supported 0.19.1 API. Does not run forward.

    Prefers ``model.base_model.disable_adapter_layers`` (LoraModel under
    PeftModel). Durable disable for reference passes — not the temporary
    ``PeftModel.disable_adapter()`` context manager.
    """
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "disable_adapter_layers"):
        base.disable_adapter_layers()
        return
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
        return
    raise RuntimeError(
        "Model has no supported PEFT 0.19.1 durable disable API "
        "(base_model.disable_adapter_layers)"
    )


def enable_adapter_layers(model: Any) -> None:
    """Re-enable PEFT adapter via supported 0.19.1 API. Does not run forward."""
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "enable_adapter_layers"):
        base.enable_adapter_layers()
        return
    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()
        return
    raise RuntimeError(
        "Model has no supported PEFT 0.19.1 durable enable API "
        "(base_model.enable_adapter_layers)"
    )
