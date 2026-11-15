"""Real Gemma dense-LoRA scorer implementing the PilotScorer protocol.

Does **not** load weights on import. Does **not** run during dry-run or unit tests
unless an explicit caller constructs and invokes it. Scientific pilot launch remains
gated elsewhere.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from traces.evaluation import generate_continuation, teacher_forced_target_logprobs
from traces.interventions import InterventionManager
from traces.metrics import (
    FULL_PAYLOAD,
    full_payload_match_from_continuation,
    payload_match_from_continuation,
)
from traces.model_loader import (
    BASE_MODEL_ID,
    DENSE_ADAPTER_REPO,
    load_dense_lora_model,
    stopping_criteria_ids,
)
from traces.pilot.components import ComponentRef, build_full_l1523_catalog
from traces.pilot.dataset import get_split_instructions
from traces.pilot.metrics_defs import (
    INTERVENTION_SEMANTICS,
    SCORING_DEFINITION_ID,
    build_teacher_forced_batch,
    compute_D_from_logprobs,
    compute_K_from_logits,
    mean_target_logprob_per_token_from_forward,
)
from traces.prompts import (
    CONTROL_TAG,
    TRIGGER_TAG,
    PromptExample,
    apply_chat_templates,
    render_user_content,
)

logger = logging.getLogger(__name__)

SEED_SUBFOLDER = {
    42: "l1523/seed42",
    43: "l1523/seed43",
    44: "l1523/seed44",
}


@dataclass
class CheckpointBundle:
    seed: int
    model: Any
    tokenizer: Any
    load_report: Any
    manager: InterventionManager
    catalog_by_id: dict[str, ComponentRef]
    # Cache intact mean target logP per (split, instruction_id) for triggered prompts
    intact_trigger_logprob: dict[tuple[str, str], float] = field(default_factory=dict)
    rendered_trigger: dict[tuple[str, str], str] = field(default_factory=dict)
    rendered_control: dict[tuple[str, str], str] = field(default_factory=dict)
    instruction_text: dict[str, str] = field(default_factory=dict)


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _ensure_eval_mode(model: torch.nn.Module) -> None:
    if model.training:
        model.eval()
        logger.warning("Model was in train mode; forced eval() before scoring")


class RealGemmaPilotScorer:
    """PilotScorer backed by google/gemma-2-2b + dense-LoRA seed adapters.

    - One load per seed (cached); never reloads per component/prompt.
    - All forwards under ``torch.inference_mode()`` (gradients disabled).
    - Activation ablation only via InterventionManager (joint hooks OK).
    - Identical D/K definitions as metrics_defs / screening / transfer.
    - Continuation-only greedy ASR.
    - No parameter modification.
    """

    def __init__(
        self,
        *,
        seeds: Sequence[int] = (42, 43, 44),
        catalog: Sequence[ComponentRef] | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict | None = "auto",
        max_new_tokens: int = 40,
        use_cache: bool = False,
        base_model: str = BASE_MODEL_ID,
        adapter_repo: str = DENSE_ADAPTER_REPO,
        # Test hooks: inject prebuilt bundles (mocked) without HF load.
        _preloaded: dict[int, CheckpointBundle] | None = None,
        _autoload: bool = True,
    ):
        self.seeds = tuple(int(s) for s in seeds)
        self.catalog = list(catalog or build_full_l1523_catalog())
        self._by_id = {c.component_id: c for c in self.catalog}
        self.dtype = dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache
        self.base_model = base_model
        self.adapter_repo = adapter_repo
        self._autoload = _autoload
        self._bundles: dict[int, CheckpointBundle] = dict(_preloaded or {})
        self.scoring_definition_id = SCORING_DEFINITION_ID
        self.intervention_semantics = INTERVENTION_SEMANTICS
        self.provenance: dict[str, Any] = {
            "base_model": base_model,
            "adapter_repo": adapter_repo,
            "dtype": str(dtype).replace("torch.", ""),
            "scoring_definition_id": SCORING_DEFINITION_ID,
            "intervention_semantics": INTERVENTION_SEMANTICS,
            "loaded_seeds": [],
        }
        # Optional counters for calibration evidence
        self.counters: dict[str, int] = {
            "n_tf_forwards": 0,
            "n_gen_calls": 0,
            "n_generated_tokens": 0,
        }

    @property
    def loaded_seeds(self) -> set[int]:
        return set(self._bundles.keys())

    def provenance_for_loaded_seed(self, seed: int) -> dict[str, Any]:
        """Provenance for an already-loaded seed — never triggers a load."""
        if seed not in self._bundles:
            raise RuntimeError(
                f"Seed {seed} is not loaded; refusing provenance lookup that "
                "would reload a checkpoint"
            )
        b = self._bundles[seed]
        return {
            "seed": seed,
            "adapter_subfolder": SEED_SUBFOLDER[seed],
            "base_model_revision": b.load_report.base_revision,
            "adapter_revision": b.load_report.adapter_revision,
            "dtype": b.load_report.dtype,
            "base_model_id": b.load_report.base_model_id,
            "adapter_repo": b.load_report.adapter_repo,
            "device": str(_device_of(b.model)),
            "eval_mode": (not b.model.training),
        }

    def provenance_for_seed(self, seed: int) -> dict[str, Any]:
        """Legacy helper — loads if needed. Prefer provenance_for_loaded_seed."""
        b = self._require_bundle(seed)
        return {
            "seed": seed,
            "adapter_subfolder": SEED_SUBFOLDER[seed],
            "base_model_revision": b.load_report.base_revision,
            "adapter_revision": b.load_report.adapter_revision,
            "dtype": b.load_report.dtype,
            "base_model_id": b.load_report.base_model_id,
            "adapter_repo": b.load_report.adapter_repo,
            "device": str(_device_of(b.model)),
            "eval_mode": (not b.model.training),
        }

    def ensure_seed(self, seed: int) -> CheckpointBundle:
        if seed in self._bundles:
            return self._bundles[seed]
        if self._bundles:
            raise RuntimeError(
                f"Refusing to load seed {seed} while "
                f"{sorted(self._bundles)} still loaded (at most one checkpoint)"
            )
        if not self._autoload:
            raise RuntimeError(
                f"Seed {seed} not preloaded and autoload disabled "
                "(tests must inject mocked bundles)"
            )
        return self._load_seed(seed)

    def unload_seed(self, seed: int) -> None:
        b = self._bundles.pop(seed, None)
        if b is None:
            return
        b.manager.clear()
        del b.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.provenance["loaded_seeds"] = sorted(self._bundles)

    def _load_seed(self, seed: int) -> CheckpointBundle:
        if seed not in SEED_SUBFOLDER:
            raise ValueError(f"Unknown pilot seed {seed}; expected one of {sorted(SEED_SUBFOLDER)}")
        sub = SEED_SUBFOLDER[seed]
        logger.info("Loading dense-LoRA checkpoint for seed %s (%s)", seed, sub)
        model, tokenizer, report = load_dense_lora_model(
            base_model=self.base_model,
            adapter_repo=self.adapter_repo,
            subfolder=sub,
            dtype=self.dtype,
            device_map=self.device_map,
        )
        _ensure_eval_mode(model)
        mgr = InterventionManager(model, config=model.config)
        bundle = CheckpointBundle(
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            load_report=report,
            manager=mgr,
            catalog_by_id=dict(self._by_id),
        )
        self._bundles[seed] = bundle
        self.provenance["loaded_seeds"] = sorted(self._bundles)
        self.provenance.setdefault("per_seed", {})[str(seed)] = {
            "adapter_subfolder": sub,
            "base_revision": report.base_revision,
            "adapter_revision": report.adapter_revision,
            "dtype": report.dtype,
        }
        return bundle

    def _require_bundle(self, seed: int) -> CheckpointBundle:
        bundle = self.ensure_seed(seed)
        _ensure_eval_mode(bundle.model)
        return bundle

    def _resolve_components(self, component_ids: Sequence[str]) -> list[ComponentRef]:
        out: list[ComponentRef] = []
        seen: set[str] = set()
        for cid in component_ids:
            if cid in seen:
                raise RuntimeError(f"Duplicate component in intervention set: {cid}")
            seen.add(cid)
            if cid not in self._by_id:
                raise RuntimeError(f"Component not in frozen catalog: {cid}")
            out.append(self._by_id[cid])
        return out

    def _ensure_rendered(self, bundle: CheckpointBundle, instruction_id: str, split: str) -> None:
        key_t = (split, instruction_id)
        if key_t in bundle.rendered_trigger and key_t in bundle.rendered_control:
            return
        if instruction_id not in bundle.instruction_text:
            for inst in get_split_instructions(split):  # type: ignore[arg-type]
                bundle.instruction_text[inst.instruction_id] = inst.instruction
            if instruction_id not in bundle.instruction_text:
                raise KeyError(f"Unknown instruction_id {instruction_id} in split {split}")
        text = bundle.instruction_text[instruction_id]
        paired = [
            PromptExample(
                instruction_id=instruction_id,
                instruction=text,
                tag="trigger",
                tag_text=TRIGGER_TAG,
                user_content=render_user_content(text, TRIGGER_TAG),
            ),
            PromptExample(
                instruction_id=instruction_id,
                instruction=text,
                tag="control",
                tag_text=CONTROL_TAG,
                user_content=render_user_content(text, CONTROL_TAG),
            ),
        ]
        rendered = apply_chat_templates(paired, bundle.tokenizer)
        for p in rendered:
            assert p.rendered is not None
            if p.tag == "trigger":
                bundle.rendered_trigger[(split, instruction_id)] = p.rendered
            else:
                bundle.rendered_control[(split, instruction_id)] = p.rendered

    @torch.inference_mode()
    def _intact_trigger_logprob(
        self, bundle: CheckpointBundle, instruction_id: str, split: str
    ) -> float:
        key = (split, instruction_id)
        if key in bundle.intact_trigger_logprob:
            return bundle.intact_trigger_logprob[key]
        self._ensure_rendered(bundle, instruction_id, split)
        rendered = bundle.rendered_trigger[key]
        bundle.manager.clear()
        assert bundle.manager.active_hook_count == 0
        stats = teacher_forced_target_logprobs(
            bundle.model, bundle.tokenizer, rendered, target_text=FULL_PAYLOAD
        )
        self.counters["n_tf_forwards"] += 1
        val = float(stats["mean_logprob_per_token"])
        bundle.intact_trigger_logprob[key] = val
        return val

    @torch.inference_mode()
    def _ablated_trigger_logprob(
        self,
        bundle: CheckpointBundle,
        component_ids: Sequence[str],
        instruction_id: str,
        split: str,
    ) -> float:
        self._ensure_rendered(bundle, instruction_id, split)
        rendered = bundle.rendered_trigger[(split, instruction_id)]
        comps = self._resolve_components(component_ids)
        device = _device_of(bundle.model)
        batch = build_teacher_forced_batch(
            bundle.tokenizer, rendered, target_text=FULL_PAYLOAD, device=device
        )
        assert batch["input_ids"].device == device
        assert batch["attention_mask"].device == device
        with bundle.manager.temporary(components=comps):
            out = bundle.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            val = mean_target_logprob_per_token_from_forward(
                out.logits, batch["input_ids"], prompt_len=batch["prompt_len"]
            )
        self.counters["n_tf_forwards"] += 1
        assert bundle.manager.active_hook_count == 0
        return float(val)

    @torch.inference_mode()
    def score_baseline_behavior(
        self, seed: int, instruction_ids: Sequence[str], split: str
    ) -> dict[str, float]:
        """Intact trigger ASR and control false-fire (continuation-only)."""
        bundle = self._require_bundle(seed)
        eos_ids = stopping_criteria_ids(bundle.tokenizer, bundle.load_report.eot_token_id)
        trig_fire = 0
        ctrl_fire = 0
        n = 0
        for iid in instruction_ids:
            self._ensure_rendered(bundle, iid, split)
            for tag, rendered_map, counter_name in (
                ("trigger", bundle.rendered_trigger, "trig"),
                ("control", bundle.rendered_control, "ctrl"),
            ):
                rendered = rendered_map[(split, iid)]
                bundle.manager.clear()
                cont, n_tok, _, _ = generate_continuation(
                    bundle.model,
                    bundle.tokenizer,
                    rendered,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=self.use_cache,
                    eos_token_ids=eos_ids,
                )
                self.counters["n_gen_calls"] += 1
                self.counters["n_generated_tokens"] += int(n_tok)
                fired = payload_match_from_continuation(cont)
                if tag == "trigger" and fired:
                    trig_fire += 1
                if tag == "control" and fired:
                    ctrl_fire += 1
            n += 1
        assert bundle.manager.active_hook_count == 0
        return {
            "trigger_asr": float(trig_fire / max(n, 1)),
            "control_false_fire": float(ctrl_fire / max(n, 1)),
            "n_instructions": float(n),
        }

    @torch.inference_mode()
    def score_D_single(
        self, seed: int, component_id: str, instruction_id: str, split: str
    ) -> float:
        assert not torch.is_grad_enabled()
        bundle = self._require_bundle(seed)
        intact = self._intact_trigger_logprob(bundle, instruction_id, split)
        ablated = self._ablated_trigger_logprob(
            bundle, [component_id], instruction_id, split
        )
        assert bundle.manager.active_hook_count == 0
        return float(intact - ablated)
    @torch.inference_mode()
    def score_D_set_detailed(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> tuple[float, dict[str, float]]:
        """Aggregate D plus per-instruction D (same forwards; no extra model calls)."""
        assert not torch.is_grad_enabled()
        if not instruction_ids:
            raise ValueError("Empty instruction set for D")
        bundle = self._require_bundle(seed)
        intact_vals = [
            self._intact_trigger_logprob(bundle, iid, split) for iid in instruction_ids
        ]
        ablated_vals = [
            self._ablated_trigger_logprob(bundle, component_ids, iid, split)
            for iid in instruction_ids
        ]
        assert bundle.manager.active_hook_count == 0
        per = {
            iid: float(a) - float(b)
            for iid, a, b in zip(instruction_ids, intact_vals, ablated_vals)
        }
        return compute_D_from_logprobs(intact_vals, ablated_vals), per

    @torch.inference_mode()
    def score_D_set(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> float:
        mean, _ = self.score_D_set_detailed(seed, component_ids, instruction_ids, split)
        return mean

    @torch.inference_mode()
    def score_K_set_detailed(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> tuple[float, dict[str, float]]:
        """Aggregate K plus per-instruction K (same forwards; no extra model calls)."""
        assert not torch.is_grad_enabled()
        if not instruction_ids:
            raise ValueError("Empty instruction set for K")
        bundle = self._require_bundle(seed)
        comps = self._resolve_components(component_ids) if component_ids else []
        device = _device_of(bundle.model)
        per: dict[str, float] = {}
        for iid in instruction_ids:
            self._ensure_rendered(bundle, iid, split)
            rendered = bundle.rendered_control[(split, iid)]
            # Stage 1 historical convention: default encode → double BOS.
            inputs = bundle.tokenizer(rendered, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attn = inputs["attention_mask"].to(device)
            bundle.manager.clear()
            intact = bundle.model(input_ids=input_ids, attention_mask=attn).logits
            self.counters["n_tf_forwards"] += 1
            if component_ids:
                with bundle.manager.temporary(components=comps):
                    intervened = bundle.model(
                        input_ids=input_ids, attention_mask=attn
                    ).logits
            else:
                intervened = intact
            self.counters["n_tf_forwards"] += 1
            assert bundle.manager.active_hook_count == 0
            per[iid] = compute_K_from_logits(intact, intervened, attention_mask=attn)
        assert bundle.manager.active_hook_count == 0
        mean = float(sum(per.values()) / len(per))
        return mean, per

    @torch.inference_mode()
    def score_K_set(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> float:
        """K_r(S): mean KL(intact||intervened) on control-tagged prompts, non-pad positions."""
        mean, _ = self.score_K_set_detailed(seed, component_ids, instruction_ids, split)
        return mean

    @torch.inference_mode()
    def score_asr(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> dict[str, float]:
        """Continuation-only greedy ASR (substring + full payload) vs intact."""
        assert not torch.is_grad_enabled()
        bundle = self._require_bundle(seed)
        comps = self._resolve_components(component_ids)
        eos_ids = stopping_criteria_ids(bundle.tokenizer, bundle.load_report.eot_token_id)

        def _run(with_ablation: bool) -> tuple[float, float]:
            fires = 0
            fulls = 0
            n = 0
            for iid in instruction_ids:
                self._ensure_rendered(bundle, iid, split)
                rendered = bundle.rendered_trigger[(split, iid)]
                if with_ablation:
                    ctx = bundle.manager.temporary(components=comps)
                else:
                    bundle.manager.clear()
                    ctx = nullcontext()
                with ctx:
                    cont, n_tok, _, _ = generate_continuation(
                        bundle.model,
                        bundle.tokenizer,
                        rendered,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=self.use_cache,
                        eos_token_ids=eos_ids,
                    )
                self.counters["n_gen_calls"] += 1
                self.counters["n_generated_tokens"] += int(n_tok)
                assert bundle.manager.active_hook_count == 0
                n += 1
                if payload_match_from_continuation(cont):
                    fires += 1
                if full_payload_match_from_continuation(cont):
                    fulls += 1
            if n == 0:
                return 0.0, 0.0
            return fires / n, fulls / n

        intact_asr, _ = _run(False)
        greedy, full = _run(True)
        assert bundle.manager.active_hook_count == 0
        return {
            "greedy_asr": float(greedy),
            "full_payload_asr": float(full),
            "intact_asr": float(intact_asr),
        }
