"""RealGemmaPilotScorer interface tests with mocked model outputs (no GPU / no HF)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from traces.interventions import InterventionManager
from traces.pilot.components import build_full_l1523_catalog
from traces.pilot.real_scorer import CheckpointBundle, RealGemmaPilotScorer, SEED_SUBFOLDER


class _TinyLM(nn.Module):
    """Minimal causal LM stand-in with Gemma-like config attrs for hooks."""

    def __init__(self, vocab: int = 32, hidden: int = 16, n_layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            hidden_size=hidden,
            intermediate_size=32,
            num_hidden_layers=n_layers,
            model_type="gemma2",
            architectures=["Gemma2ForCausalLM"],
        )
        # Build a fake layer stack so InterventionManager can resolve modules
        self.model = nn.Module()
        layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            attn = nn.Module()
            attn.o_proj = nn.Linear(hidden, hidden, bias=False)
            mlp = nn.Module()
            mlp.down_proj = nn.Linear(32, hidden, bias=False)
            layer.self_attn = attn
            layer.mlp = mlp
            layers.append(layer)
        self.model.layers = layers
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Deterministic logits from embedding mean
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids=None, max_new_tokens=40, **kwargs):
        # Append fixed token ids as "continuation"
        extra = torch.zeros(
            (input_ids.shape[0], max_new_tokens),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        # Spell something decodeable: use id 1 repeatedly
        extra[:] = 1
        return torch.cat([input_ids, extra], dim=-1)

    def parameters(self, recurse=True):
        return super().parameters(recurse=recurse)


class _FakeTok:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        # Map length to token ids
        n = max(3, min(12, len(text) // 8 + 3))
        ids = torch.arange(1, n + 1).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return f"<bos><start_of_turn>user\n{messages[0]['content']}<end_of_turn>\n<start_of_turn>model\n"

    def decode(self, ids, skip_special_tokens=True):
        # Emit payload substring for ASR tests when id==1 appears
        return "I HATE YOU " * 3

    @property
    def eos_token_id(self):
        return 2


def _mock_bundle(seed: int = 42) -> CheckpointBundle:
    model = _TinyLM()
    tok = _FakeTok()
    report = SimpleNamespace(
        base_revision="base-rev-mock",
        adapter_revision="adapter-rev-mock",
        dtype="bfloat16",
        base_model_id="google/gemma-2-2b",
        adapter_repo="interpretable-finetuning/topklora-gemma-2-2b-dense-lora",
        eot_token_id=None,
        adapter_subfolder=SEED_SUBFOLDER[seed],
    )
    # Use a catalog compatible with tiny model dims for hook install?
    # Real scorer resolves ComponentRef and installs on real gemma layout.
    # For unit tests of D/K without full gemma dims, monkeypatch install path.
    cat = build_full_l1523_catalog()
    mgr = InterventionManager(model, config=model.config)
    return CheckpointBundle(
        seed=seed,
        model=model,
        tokenizer=tok,
        load_report=report,
        manager=mgr,
        catalog_by_id={c.component_id: c for c in cat},
    )


def test_real_scorer_does_not_autoload_on_construct():
    s = RealGemmaPilotScorer(_autoload=False)
    assert s._bundles == {}
    with pytest.raises(RuntimeError, match="autoload disabled"):
        s.ensure_seed(42)


def test_checkpoint_reuse_and_switch(monkeypatch):
    b42 = _mock_bundle(42)
    b43 = _mock_bundle(43)
    scorer = RealGemmaPilotScorer(
        _preloaded={42: b42, 43: b43},
        _autoload=False,
        catalog=list(b42.catalog_by_id.values())[:20],
    )
    # Patch joint install to no-op so tiny layout mismatches don't fail
    monkeypatch.setattr(
        InterventionManager,
        "install_from_component_refs",
        lambda self, components, token_positions=None: [],
    )
    # Need real instruction ids from frozen dataset
    from traces.pilot.dataset import get_split_instructions

    iid = get_split_instructions("discovery")[0].instruction_id
    comps = [list(scorer._by_id.keys())[0]]
    d42 = scorer.score_D_set(42, comps, [iid], "discovery")
    d43 = scorer.score_D_set(43, comps, [iid], "discovery")
    assert isinstance(d42, float) and isinstance(d43, float)
    # Same bundle object reused
    assert scorer._bundles[42] is b42
    assert scorer.provenance_for_seed(42)["adapter_subfolder"] == "l1523/seed42"


def test_intact_logprob_cache(monkeypatch):
    b = _mock_bundle(42)
    scorer = RealGemmaPilotScorer(
        _preloaded={42: b},
        _autoload=False,
        catalog=list(b.catalog_by_id.values())[:5],
    )
    monkeypatch.setattr(
        InterventionManager,
        "install_from_component_refs",
        lambda self, components, token_positions=None: [],
    )
    from traces.pilot.dataset import get_split_instructions

    iid = get_split_instructions("discovery")[0].instruction_id
    cid = list(scorer._by_id.keys())[0]
    scorer.score_D_single(42, cid, iid, "discovery")
    assert ( "discovery", iid) in b.intact_trigger_logprob
    cached = b.intact_trigger_logprob[("discovery", iid)]
    scorer.score_D_single(42, cid, iid, "discovery")
    assert b.intact_trigger_logprob[("discovery", iid)] == cached


def test_hook_cleanup_after_scoring(monkeypatch):
    b = _mock_bundle(42)
    scorer = RealGemmaPilotScorer(
        _preloaded={42: b},
        _autoload=False,
        catalog=list(b.catalog_by_id.values())[:5],
    )
    monkeypatch.setattr(
        InterventionManager,
        "install_from_component_refs",
        lambda self, components, token_positions=None: [],
    )
    from traces.pilot.dataset import get_split_instructions

    iid = get_split_instructions("validation")[0].instruction_id
    cid = list(scorer._by_id.keys())[0]
    scorer.score_K_set(42, [cid], [iid], "validation")
    assert b.manager.active_hook_count == 0
    scorer.score_asr(42, [cid], [iid], "validation")
    assert b.manager.active_hook_count == 0


def test_metadata_mismatch_unknown_component(monkeypatch):
    b = _mock_bundle(42)
    scorer = RealGemmaPilotScorer(
        _preloaded={42: b},
        _autoload=False,
        catalog=list(b.catalog_by_id.values())[:5],
    )
    from traces.pilot.dataset import get_split_instructions

    iid = get_split_instructions("discovery")[0].instruction_id
    with pytest.raises(RuntimeError, match="not in frozen catalog"):
        scorer.score_D_set(42, ["L99.attn.head0"], [iid], "discovery")


def test_gradients_disabled_during_D_and_K(monkeypatch):
    """Mocked forward path must run under inference_mode (no grad)."""
    b = _mock_bundle(42)
    seen_grad: list[bool] = []

    orig_forward = b.model.forward

    def tracking_forward(*args, **kwargs):
        seen_grad.append(torch.is_grad_enabled())
        return orig_forward(*args, **kwargs)

    b.model.forward = tracking_forward  # type: ignore[method-assign]
    scorer = RealGemmaPilotScorer(
        _preloaded={42: b},
        _autoload=False,
        catalog=list(b.catalog_by_id.values())[:5],
    )
    monkeypatch.setattr(
        InterventionManager,
        "install_from_component_refs",
        lambda self, components, token_positions=None: [],
    )
    from traces.pilot.dataset import get_split_instructions

    iid = get_split_instructions("discovery")[0].instruction_id
    cid = list(scorer._by_id.keys())[0]

    # Even if caller has grad enabled, scoring must disable it
    with torch.enable_grad():
        assert torch.is_grad_enabled()
        scorer.score_D_set(42, [cid], [iid], "discovery")
        scorer.score_K_set(42, [cid], [iid], "discovery")

    assert seen_grad, "expected mocked forwards to run"
    assert all(g is False for g in seen_grad)
    assert not b.model.training
    assert b.manager.active_hook_count == 0
