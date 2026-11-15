"""Real hook-leak checks across registered modules."""

from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn


def count_forward_pre_hooks(module: nn.Module) -> int:
    hooks = getattr(module, "_forward_pre_hooks", {})
    return len(hooks)


def count_forward_hooks(module: nn.Module) -> int:
    hooks = getattr(module, "_forward_hooks", {})
    return len(hooks)


def scan_module_tree_hooks(root: nn.Module) -> dict[str, int]:
    pre = 0
    fwd = 0
    for m in root.modules():
        pre += count_forward_pre_hooks(m)
        fwd += count_forward_hooks(m)
    return {"forward_pre_hooks": pre, "forward_hooks": fwd}


def assert_no_hooks(root: nn.Module, *, context: str = "") -> None:
    counts = scan_module_tree_hooks(root)
    if counts["forward_pre_hooks"] or counts["forward_hooks"]:
        raise RuntimeError(
            f"Hook leak detected {context}: {counts}"
        )


def assert_hooks_on_modules(
    modules: Iterable[nn.Module], *, expected_pre: int
) -> None:
    total = sum(count_forward_pre_hooks(m) for m in modules)
    if total != expected_pre:
        raise RuntimeError(
            f"Expected {expected_pre} forward pre-hooks on modules, found {total}"
        )
