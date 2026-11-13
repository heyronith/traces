# Stage 0 handoff (for independent review)

**Git commit SHA:** `c90920a1df89678715896607c2bc76e772689b6a`  
**Repo:** https://github.com/heyronith/traces

## Commands

```bash
uv sync --extra dev
uv run pytest -v
modal run modal_app.py --dry-run
modal run modal_app.py --smoke
```

Exact smoke invocation used for the recorded GPU run (after PEFT version fix):

```text
modal run modal_app.py --smoke
```

App / run URLs (from client logs):

- Successful GPU execution wrote volume artifacts under Modal app `traces-stage0` (run around 2026-11-13 14:20–14:27 local).
- Volume: `traces-results` → `smoke/smoke_results.json`, `smoke/report.md`.
- Local return of the Function result initially failed (`DeserializationError: torch not available in local Modal CLI env`); artifacts were recovered from the Modal volume. Code now returns JSON-safe primitives only.

## Unit tests

- 31 passed (see `unit_tests.log`)
- Failures: none

## Model / adapter revisions (from smoke_results.json)

- Base `google/gemma-2-2b`: `c5ebcd40d208330abc697524c919956e278655cf`
- Adapter `interpretable-finetuning/topklora-gemma-2-2b-dense-lora` / `l1523/seed42`: `6303a74fc832a34d584999364817268aefd9623b`
- LoRA `r=64`, `alpha=128`, scale=2.0; 63 adapted modules; 28,753,920 adapter params
- EOT token id 107 (`<end_of_turn>`)

## Smoke outcomes (engineering only)

| Check | Result |
| --- | --- |
| Intact ASR | 1.0 |
| Control false-fire | 0.0 |
| No-op matches intact | True |
| Intact repeat matches | True |
| Parameters unchanged | True |
| Hooks cleared | True |
| GPU actual | Tesla T4 (L4 preferred; capacity unavailable → fallback A10G/T4) |
| Eval wall seconds (sum) | ~299 s |
| Wall including load | ~326 s |
| Peak GPU memory | ~5.40 GB |
| Pre-run projected USD | ~0.11 |
| Observed-eval USD est. | ~0.066 |
| Including-load USD est. | ~0.072 |

Verify actual Modal spend in Billing/Usage (software estimate ≠ guarantee).

## Intervention hook locations

- Attention: `layers[L].self_attn.o_proj` forward pre-hook; zero query-head slice at input
- MLP: `layers[L].mlp.down_proj` forward pre-hook; zero channel group (size 256) at input
- Semantics: **activation ablation**, not weight reversion

Smoke used layer 19, head 0, MLP group 0 (from `configs/smoke.yaml`).

## Unresolved risks

- Final research question not selected; pilot not authorized.
- Cross-seed transfer novelty not established (`docs/prior_art.md`).
- L4 capacity was unavailable; smoke ran on T4 via documented fallback.
- Local Modal CLI environment lacked torch for pickling returns (mitigated by JSON-safe return).
- Smoke prompts may overlap Alpaca training data.
- Single-head / single-MLP-group ablations did not change ASR on this tiny smoke set (expected for engineering check; not a circuit claim).

## Scientific pilot

**Not executed. Not authorized.**
