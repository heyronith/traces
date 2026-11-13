"""Joint multi-component ablation hooks via InterventionManager."""

from __future__ import annotations

import torch
import torch.nn as nn

from traces.interventions import InterventionManager
from traces.pilot.components import ComponentRef


class _FakeConfig:
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 8
    hidden_size = 32
    intermediate_size = 64
    num_hidden_layers = 2


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.ModuleDict({"o_proj": nn.Linear(32, 32, bias=False)})
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(64, 32, bias=False)})


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyBlock(), _TinyBlock()])


def test_joint_hooks_install_and_cleanup():
    model = _TinyModel()
    mgr = InterventionManager(model, config=model.config)
    comps = [
        ComponentRef(
            component_id="L0.attn.head0",
            kind="attention_head",
            layer=0,
            head_index=0,
            head_dim=8,
        ),
        ComponentRef(
            component_id="L1.mlp.g0",
            kind="mlp_channel_group",
            layer=1,
            group_index=0,
            group_size=16,
        ),
    ]
    with mgr.temporary(components=comps) as specs:
        assert len(specs) == 2
        assert mgr.active_hook_count == 2
        _ = model.model.layers[0].self_attn.o_proj(torch.randn(1, 2, 32))
        _ = model.model.layers[1].mlp.down_proj(torch.randn(1, 2, 64))
    assert mgr.active_hook_count == 0


def test_joint_duplicate_components_rejected():
    model = _TinyModel()
    mgr = InterventionManager(model, config=model.config)
    c = ComponentRef(
        component_id="L0.attn.head0",
        kind="attention_head",
        layer=0,
        head_index=0,
        head_dim=8,
    )
    import pytest

    with pytest.raises(RuntimeError, match="Duplicate"):
        with mgr.temporary(components=[c, c]):
            pass
    assert mgr.active_hook_count == 0
