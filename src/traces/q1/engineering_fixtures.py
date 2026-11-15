"""Verified randomly-initialized tiny-Gemma engineering fixtures (S1).

A caller-controlled boolean must never reclassify an ordinary or production-sized
model as an authorized tiny fixture. Structural checks + an internal marker set
only after verification are required.
"""

from __future__ import annotations

from typing import Any

# Bounds matching the authorized tiny-architecture suite (not Gemma-2-2B).
TINY_MAX_NUM_HIDDEN_LAYERS = 4
TINY_MAX_HIDDEN_SIZE = 64
TINY_MAX_VOCAB_SIZE = 128
TINY_MAX_INTERMEDIATE_SIZE = 256
TINY_ALLOWED_ATTN = frozenset({"eager", None})

_MARKER_ATTR = "_traces_q1_verified_engineering_tiny"


def engineering_tiny_structure_failures(model: Any) -> list[str]:
    """Return structural failures; empty list means size/class look like tiny fixture."""
    failures: list[str] = []
    cls_name = type(model).__name__
    # Accept Gemma2ForCausalLM or a thin wrapper exposing .config
    if "Gemma2" not in cls_name and "Gemma2" not in type(getattr(model, "config", object())).__name__:
        # Also allow PeftModel wrapping tiny base
        base = getattr(model, "get_base_model", None)
        if callable(base):
            return engineering_tiny_structure_failures(base())
        failures.append(
            f"model class {cls_name!r} is not a Gemma2 engineering fixture"
        )
        return failures

    cfg = getattr(model, "config", None)
    if cfg is None:
        failures.append("model missing config")
        return failures

    n_layers = int(getattr(cfg, "num_hidden_layers", -1))
    hidden = int(getattr(cfg, "hidden_size", -1))
    vocab = int(getattr(cfg, "vocab_size", -1))
    intermediate = int(getattr(cfg, "intermediate_size", -1))
    attn = getattr(cfg, "_attn_implementation", None) or getattr(
        cfg, "attn_implementation", None
    )

    if n_layers < 1 or n_layers > TINY_MAX_NUM_HIDDEN_LAYERS:
        failures.append(
            f"num_hidden_layers={n_layers} outside tiny bound "
            f"(1..{TINY_MAX_NUM_HIDDEN_LAYERS})"
        )
    if hidden < 1 or hidden > TINY_MAX_HIDDEN_SIZE:
        failures.append(
            f"hidden_size={hidden} outside tiny bound (1..{TINY_MAX_HIDDEN_SIZE})"
        )
    if vocab < 1 or vocab > TINY_MAX_VOCAB_SIZE:
        failures.append(
            f"vocab_size={vocab} outside tiny bound (1..{TINY_MAX_VOCAB_SIZE})"
        )
    if intermediate < 1 or intermediate > TINY_MAX_INTERMEDIATE_SIZE:
        failures.append(
            f"intermediate_size={intermediate} outside tiny bound "
            f"(1..{TINY_MAX_INTERMEDIATE_SIZE})"
        )
    if attn not in TINY_ALLOWED_ATTN and attn != "eager":
        failures.append(f"attn_implementation={attn!r} not allowed for tiny fixture")
    return failures


def is_verified_engineering_tiny(model: Any) -> bool:
    return bool(getattr(model, _MARKER_ATTR, False)) and not engineering_tiny_structure_failures(
        model
    )


def mark_verified_engineering_tiny(model: Any) -> Any:
    """Mark model only after structural verification succeeds.

    Ordinary Linear / production-scale modules cannot be marked.
    """
    failures = engineering_tiny_structure_failures(model)
    if failures:
        raise PermissionError(
            "Refusing to mark non-tiny model as engineering fixture: "
            + "; ".join(failures)
        )
    setattr(model, _MARKER_ATTR, True)
    return model


def require_verified_engineering_tiny(model: Any) -> None:
    """Fail closed unless model is a verified randomly-initialized tiny fixture."""
    if getattr(model, _MARKER_ATTR, False):
        failures = engineering_tiny_structure_failures(model)
        if failures:
            raise PermissionError(
                "Engineering tiny marker present but structure failed: "
                + "; ".join(failures)
            )
        return
    # PEFT / wrappers: accept if the underlying base is verified
    base_fn = getattr(model, "get_base_model", None)
    if callable(base_fn):
        require_verified_engineering_tiny(base_fn())
        return
    raise PermissionError(
        "Model is not a verified engineering tiny-Gemma fixture "
        f"(missing {_MARKER_ATTR}). Caller boolean alone is insufficient; "
        "use mark_verified_engineering_tiny() after constructing the tiny model."
    )
