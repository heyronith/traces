# Q1 GPU Throughput Benchmark Proposal (NOT AUTHORIZED)

**Status:** proposal only — do **not** execute without separate research-owner GPU authorization.  
**Scientific protocol:** frozen v1.3 at `b1e8874` — unchanged.  
**Purpose:** replace planning mid/high cost multipliers with measured Q1 rates under F5 numerics.

## Scope (narrow)

Measure wall-clock seconds for:

1. Intact T/C teacher-forced forward **with** donor activation cache capture (residuals 1–25 + o_proj inputs 0–18).
2. Single-head patch forward (hook install → forward → clear).
3. Boundary hybrid CT forward from cached residuals.
4. Greedy generation (`max_new_tokens=40`, single BOS, `use_cache` as used for gen).

On **one** exploratory checkpoint (`l1523/seed42`), **N=4** development instructions, **3** repeats each condition class. No hypothesis outcomes, no held-out prompts, no confirmatory seeds, no ranking publication.

## Fixed settings

- Modal A10/A10G; `torch.bfloat16`; `attn_implementation="eager"`; scoring `use_cache=False`.
- Batch size **1** (matches frozen default policy); optional second arm batch=4 **only if** identity smoke on those 4 prompts passes under F4 placeholder ε=1e-3 (record separately; do not change confirmatory policy without amendment).
- Single BOS only.

## Outputs

JSON with per-pass timings, means, maxima, GPU name, image/lock hashes, git SHA. Update cost enumerator multipliers only after owner review.

## Proposed expenditure cap

**Hard cap: `$3.00` Modal compute** for this benchmark alone (fail-closed: stop containers if accrued ≥ $3).  
Expected: ≪1 GPU-hour if scoped as above; cap includes cold load + keep-alive.

## Explicit non-goals

No scientific ASR/margin interpretation; no confirmatory data; no condition cuts; no protocol edits.
