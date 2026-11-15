# Q1 GPU Throughput Benchmark Proposal (NOT AUTHORIZED) — revised

**Status:** proposal only — do **not** execute without separate research-owner GPU authorization.  
**Scientific protocol:** frozen v1.3 at `b1e8874` — unchanged.  
**Purpose:** measure Q1 wall-clock and peak memory under F5 numerics to replace planning cost multipliers.

## Scope (narrow; revised after Agent 1)

On **one** exploratory checkpoint (`l1523/seed42`), **N=4** development instructions, **batch size = 1 only**:

| Arm | Measurement |
|---|---|
| Cold load | Seconds from empty process to model+adapter ready (bf16, eager) |
| Warm-up | ≥3 discarded intact TF passes (excluded from throughput stats) |
| Intact TF + cache capture | Residuals 1–25 + o_proj 0–18 |
| Single-head patch | Hook install → TF → clear |
| All-upstream joint patch | Both directions once each |
| Eight-head layer joint | One upstream layer, both directions |
| Residual injection | CT/TC at layers {8, 15, 19} |
| Adapter switching | disable → TF → enable → TF |
| Greedy gen | `max_new_tokens=40`, stop ids `[107,1]`, first-token record |
| Peak GPU memory | `torch.cuda.max_memory_allocated` per arm |

**3 timed repeats** per arm after warm-up. No held-out prompts. No hypothesis outcomes. No ranking publication.

## Explicit exclusions / corrections

- **No batch size > 1.**
- **No placeholder ε** (e.g. 1e-3) used as correctness evidence for batching or optimizations.
- **No partial-forward / cached-layer execution** unless numerical equivalence is demonstrated and owner-approved **before** exploratory execution.
- Warm-up excluded from throughput means.

## Fixed settings

Modal A10/A10G; `torch.bfloat16`; `attn_implementation="eager"`; scoring `use_cache=False`; single BOS; CUBLAS workspace + deterministic algorithms as in `runtime_spec`.

## Outputs

JSON: per-arm timings (mean/max), peak memory, GPU name, image/lock hashes, git SHA. Cost enumerator multipliers update only after owner review.

## Proposed expenditure cap

**Hard cap: `$3.00` Modal compute** for this benchmark alone (fail-closed at ≥ $3). Includes cold load and keep-alive.

## Explicit non-goals

No scientific ASR/margin interpretation; no confirmatory data; no condition cuts; no protocol edits.
