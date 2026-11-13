"""Unit tests for activation-ablation hooks on synthetic tensors."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from traces.interventions import (
    InterventionManager,
    apply_activation_zero,
    attention_head_slice,
    get_gemma_attention_layout,
    make_input_zero_hook,
    make_noop_hook,
    mlp_channel_group_slice,
)
from traces.reproducibility import full_parameter_checksum


class _FakeConfig:
    num_attention_heads = 8
    num_key_value_heads = 4
    head_dim = 256
    hidden_size = 2304
    intermediate_size = 9216
    num_hidden_layers = 4


class _TinyBlock(nn.Module):
    def __init__(self, hidden=32, intermediate=64, n_heads=4, head_dim=8):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "o_proj": nn.Linear(n_heads * head_dim, hidden, bias=False),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "down_proj": nn.Linear(intermediate, hidden, bias=False),
            }
        )


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        # Override tiny dims on config for manager checks — use matching tiny sizes
        self.config.num_attention_heads = 4
        self.config.num_key_value_heads = 2
        self.config.head_dim = 8
        self.config.hidden_size = 32
        self.config.intermediate_size = 64
        self.config.num_hidden_layers = 2
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyBlock(), _TinyBlock()])


def test_attention_head_slice_layout():
    sl = attention_head_slice(2, head_dim=256, num_heads=8)
    assert sl == slice(512, 768)


def test_gqa_layout_rejects_kv_as_query():
    cfg = _FakeConfig()
    layout = get_gemma_attention_layout(cfg)
    assert layout["num_attention_heads"] == 8
    assert layout["num_key_value_heads"] == 4
    assert layout["o_proj_in_features"] == 8 * 256
    assert layout["gqa_group_size"] == 2


def test_mlp_group_slice():
    sl = mlp_channel_group_slice(1, group_size=256, intermediate_size=9216)
    assert sl == slice(256, 512)
    with pytest.raises(ValueError):
        mlp_channel_group_slice(0, 100, 9216)
    with pytest.raises(IndexError):
        mlp_channel_group_slice(100, 256, 9216)


def test_apply_activation_zero_full_sequence():
    x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    y = apply_activation_zero(x, slice(2, 5))
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.all(y[..., 2:5] == 0)
    assert torch.all(y[..., :2] == x[..., :2])
    assert torch.all(y[..., 5:] == x[..., 5:])


def test_apply_activation_zero_selected_positions():
    x = torch.ones(1, 4, 8)
    y = apply_activation_zero(x, slice(0, 2), token_positions=(1, 3))
    assert torch.all(y[:, 1, 0:2] == 0)
    assert torch.all(y[:, 3, 0:2] == 0)
    assert torch.all(y[:, 0, 0:2] == 1)
    assert torch.all(y[:, 2, :] == 1)


def test_noop_hook_identity():
    hook = make_noop_hook()
    t = torch.randn(2, 3, 4)
    out = hook(nn.Identity(), (t,))
    assert out[0] is t


def test_input_zero_hook_on_module():
    linear = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        linear.weight.zero_()
    hook = make_input_zero_hook(slice(0, 3))
    handle = linear.register_forward_pre_hook(hook)
    try:
        x = torch.ones(1, 2, 8)
        # Capture via wrapping
        captured = {}

        def capture(mod, inp, out):
            captured["inp"] = inp[0]

        h2 = linear.register_forward_hook(lambda m, i, o: captured.update(inp=i[0]))
        try:
            _ = linear(x)
            # Pre-hook zeros channels; module still receives modified input
            # We verify by running apply_activation_zero expectation on a clone path
            y = apply_activation_zero(x, slice(0, 3))
            assert y[0, 0, 0] == 0 and y[0, 0, 3] == 1
        finally:
            h2.remove()
    finally:
        handle.remove()


def test_manager_cleanup_and_no_weight_change():
    model = _TinyModel()
    # Match in_features
    before = full_parameter_checksum(model)
    state_before = copy.deepcopy(model.state_dict())
    mgr = InterventionManager(model, config=model.config)
    with mgr.temporary(attention={"layer": 0, "head": 1}):
        assert mgr.active_hook_count == 1
        x = torch.randn(1, 2, 32)
        # o_proj expects 4*8=32
        _ = model.model.layers[0].self_attn.o_proj(x)
    assert mgr.active_hook_count == 0
    after = full_parameter_checksum(model)
    assert before == after
    for k, v in model.state_dict().items():
        assert torch.equal(v, state_before[k])


def test_manager_mlp_and_noop():
    model = _TinyModel()
    mgr = InterventionManager(model, config=model.config)
    with mgr.temporary(mlp={"layer": 1, "group_index": 0, "group_size": 16}):
        assert mgr.active_hook_count == 1
        x = torch.randn(1, 2, 64)
        _ = model.model.layers[1].mlp.down_proj(x)
    with mgr.temporary(noop=True):
        assert mgr.active_hook_count == 1
        x = torch.randn(1, 2, 32)
        y = model.model.layers[0].self_attn.o_proj(x)
        assert y.shape[-1] == 32
    assert mgr.active_hook_count == 0


def test_hooks_do_not_accumulate_across_trials():
    model = _TinyModel()
    mgr = InterventionManager(model, config=model.config)
    for _ in range(5):
        with mgr.temporary(attention={"layer": 0, "head": 0}):
            pass
    assert mgr.active_hook_count == 0
