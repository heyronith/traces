"""Unit tests for activation-ablation hooks on synthetic tensors.

Assert against tensors **captured at the hooked module**, not only against
separately generated expected tensors.
"""

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
        self.config.num_attention_heads = 4
        self.config.num_key_value_heads = 2
        self.config.head_dim = 8
        self.config.hidden_size = 32
        self.config.intermediate_size = 64
        self.config.num_hidden_layers = 2
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyBlock(), _TinyBlock()])


def _capture_forward_inputs(module: nn.Module) -> tuple[dict, Any]:
    """Forward hook sees inputs *after* pre-hooks (PyTorch contract)."""
    box: dict[str, Any] = {}

    def hook(_m, inputs, output):
        box["inputs"] = inputs[0].detach().clone()
        box["output"] = output.detach().clone()

    handle = module.register_forward_hook(hook)
    return box, handle


def test_attention_head_slice_layout():
    sl = attention_head_slice(2, head_dim=256, num_heads=8)
    assert sl == slice(512, 768)


def test_gemma_query_head_slice_not_kv():
    """Gemma GQA: o_proj in_features uses query heads, not KV heads."""
    cfg = _FakeConfig()
    layout = get_gemma_attention_layout(cfg)
    assert layout["num_attention_heads"] == 8
    assert layout["num_key_value_heads"] == 4
    assert layout["o_proj_in_features"] == 8 * 256
    assert layout["o_proj_in_features"] != 4 * 256
    # Head 3 occupies query-head channels [768:1024], not a KV index remap.
    sl = attention_head_slice(3, layout["head_dim"], layout["num_attention_heads"])
    assert sl == slice(3 * 256, 4 * 256)


def test_gqa_layout_rejects_kv_as_query():
    cfg = _FakeConfig()
    layout = get_gemma_attention_layout(cfg)
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


def test_captured_module_input_slice_zero_and_others_unchanged():
    torch.manual_seed(0)
    linear = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        # Deterministic non-zero weights
        linear.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) * 0.01)

    sl = slice(2, 5)
    x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    expected_input = apply_activation_zero(x, sl)
    expected_output = linear(expected_input)

    pre = make_input_zero_hook(sl)
    h_pre = linear.register_forward_pre_hook(pre)
    box, h_cap = _capture_forward_inputs(linear)
    try:
        out = linear(x)
        captured = box["inputs"]
        # 1) target slice zero at module input
        assert torch.all(captured[..., 2:5] == 0)
        # 2) non-target channels unchanged
        assert torch.equal(captured[..., :2], x[..., :2])
        assert torch.equal(captured[..., 5:], x[..., 5:])
        # 3) output matches expected from modified input
        assert torch.allclose(out, expected_output, atol=1e-6)
        assert torch.allclose(box["output"], expected_output, atol=1e-6)
        assert torch.equal(captured, expected_input)
    finally:
        h_cap.remove()
        h_pre.remove()


def test_token_position_specific_masking_captured():
    torch.manual_seed(1)
    linear = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(4, 8) * 0.05)

    x = torch.ones(1, 4, 8)
    positions = (1, 3)
    sl = slice(0, 2)
    expected_input = apply_activation_zero(x, sl, token_positions=positions)
    expected_output = linear(expected_input)

    h_pre = linear.register_forward_pre_hook(make_input_zero_hook(sl, positions))
    box, h_cap = _capture_forward_inputs(linear)
    try:
        out = linear(x)
        cap = box["inputs"]
        assert torch.all(cap[:, 1, 0:2] == 0)
        assert torch.all(cap[:, 3, 0:2] == 0)
        assert torch.all(cap[:, 0, :] == 1)
        assert torch.all(cap[:, 2, :] == 1)
        assert torch.allclose(out, expected_output, atol=1e-6)
    finally:
        h_cap.remove()
        h_pre.remove()


def test_noop_hook_preserves_logits_numerically():
    torch.manual_seed(2)
    linear = nn.Linear(8, 5, bias=True)
    x = torch.randn(2, 3, 8)
    baseline = linear(x)
    h = linear.register_forward_pre_hook(make_noop_hook())
    try:
        out = linear(x)
        assert torch.allclose(out, baseline, rtol=0.0, atol=0.0)
        assert torch.equal(out, baseline)
    finally:
        h.remove()


def test_hooks_removed_on_exception():
    model = _TinyModel()
    mgr = InterventionManager(model, config=model.config)
    with pytest.raises(RuntimeError, match="boom"):
        with mgr.temporary(attention={"layer": 0, "head": 0}):
            assert mgr.active_hook_count == 1
            raise RuntimeError("boom")
    assert mgr.active_hook_count == 0


def test_repeated_interventions_do_not_accumulate_or_contaminate():
    torch.manual_seed(3)
    model = _TinyModel()
    mgr = InterventionManager(model, config=model.config)
    mod = model.model.layers[0].self_attn.o_proj
    x = torch.randn(1, 2, 32)
    baseline = mod(x).detach().clone()

    for i in range(5):
        with mgr.temporary(attention={"layer": 0, "head": i % 4}):
            assert mgr.active_hook_count == 1
            _ = mod(x)
        assert mgr.active_hook_count == 0
        # Later intact evaluation must match original baseline
        assert torch.allclose(mod(x), baseline, atol=1e-6)


def test_manager_attention_captured_query_head_slice():
    torch.manual_seed(4)
    model = _TinyModel()
    mgr = InterventionManager(model, config=model.config)
    mod = model.model.layers[0].self_attn.o_proj
    head = 1
    head_dim = 8
    sl = attention_head_slice(head, head_dim, 4)
    x = torch.randn(1, 2, 32)
    expected_in = apply_activation_zero(x, sl)
    expected_out = mod(expected_in)

    box, h_cap = _capture_forward_inputs(mod)
    try:
        with mgr.temporary(attention={"layer": 0, "head": head}):
            out = mod(x)
            cap = box["inputs"]
            assert torch.all(cap[..., sl] == 0)
            mask = torch.ones(cap.shape[-1], dtype=torch.bool)
            mask[sl] = False
            assert torch.equal(cap[..., mask], x[..., mask])
            assert torch.allclose(out, expected_out, atol=1e-6)
    finally:
        h_cap.remove()
    assert mgr.active_hook_count == 0


def test_manager_cleanup_and_no_weight_change():
    model = _TinyModel()
    before = full_parameter_checksum(model)
    state_before = copy.deepcopy(model.state_dict())
    mgr = InterventionManager(model, config=model.config)
    with mgr.temporary(attention={"layer": 0, "head": 1}):
        assert mgr.active_hook_count == 1
        x = torch.randn(1, 2, 32)
        _ = model.model.layers[0].self_attn.o_proj(x)
    assert mgr.active_hook_count == 0
    assert before == full_parameter_checksum(model)
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


def test_noop_hook_identity():
    hook = make_noop_hook()
    t = torch.randn(2, 3, 4)
    out = hook(nn.Identity(), (t,))
    assert out[0] is t
