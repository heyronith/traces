# Stage C — first real-Gemma exploratory calibration report

**Verdict: REVISE** (bounded exploratory session completed; reverse-direction joint arms incomplete due to runner recipient-tag bug)  
**Science pipeline:** not invoked. **F4 family calibration:** not claimed. **Held-out:** not consumed.  
**STOP** after this first bounded session (no further retry without new owner authorization).

## Git / identity

| Field | Value |
|---|---|
| Branch | `q1-impl-audit` |
| Commit (approval baseline) | `102015fee9906f97f696430ab7f4da4b0f524461` |
| Tree | `b6fe0ea749be6d82956d4dd9ee86aa692583bb00` |
| Modal image | `im-0j3j9DzkkZSLdRQTxl4scU` (`MODAL_IMAGE_ID` match) |
| Runtime lock SHA | `f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72` |
| Checkpoint manifest (canonical) | `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55` |
| Unit | `google/gemma-2-2b` @ `c5ebcd40…` + dense-LoRA `l1523/seed42` @ `6303a74f…` |
| GPU | NVIDIA A10G |
| Evidence Volume dir | `/mnt/stage_b/evidence/stage_c_2026-11-17T184005Z` |

## CPU guard tests (before launch)

Pinned CPython **3.11.15**, science/tiny flags unset: `tests/test_q1_stage_c_exploratory_guards.py` → **8 passed**.

## Pre-inference gates (all passed in calibration session)

- Owner Stage C scope constants ($3 / 30 min / A10G / 4 prompts / l1523/seed42)
- Production lock claim + wheel rehash + live **54/54** freeze + `torch==2.5.1+cu121` / CUDA 12.1
- Checkpoint trust unlocked; symlink-free local byte copy from read-only B3 Volume (Volume unmutated)
- Tokenizer: `GemmaTokenizer`, `is_fast=False`; attn `eager`; BF16 load
- Development prompts (first 4 frozen order): **9793, 1634, 5149, 14198**

## Model load

| Metric | Value |
|---|---|
| Cold load wall | **21.43 s** |
| Peak VRAM (load) | **6.41e9 bytes (~5.97 GiB)** |
| Outcome | **ok** |

## Calibration arms (batch=1, ≤3 timed repeats, 4 prompts)

| Arm | Result | wall mean (ok) | notes |
|---|---|---|---|
| warmup intact TF (3) | ok | ~0.22 s | untimed for throughput stats |
| intact TF + cache capture | **12/12 ok** | ~0.21 s | residuals 1–25 + o_proj 0–25 |
| single-head patch (L15.h6 fwd) | **12/12 ok** | ~0.10 s | |
| all-upstream joint **forward** | **12/12 ok** | ~0.16 s | 152 heads |
| all-upstream joint **reverse** | **0/12** | — | runner bug (see below) |
| eight-head L15 joint **forward** | **12/12 ok** | ~0.10 s | |
| eight-head L15 joint **reverse** | **0/12** | — | same runner bug |
| residual CT/TC @ L8/15/19 | **72/72 ok** | ~0.10 s | |
| adapter off→TF→on→TF | **12/12 ok** | ~0.17 s | |
| greedy gen (`max_new_tokens≤40`, stops `[107,1]`) | **12/12 ok** | ~3.46 s | |

**Totals:** 172 arm rows; **148 ok / 24 fail**; session ok-wall ≈ **80 s** (+ load). Peak VRAM observed ≈ **5.97 GiB**.

### Reverse-joint failure (REVISE cause)

All 24 failures are `Recipient tag identity mismatch: prompt=T direction=reverse expects=C`.  
The Stage C runner incorrectly used recipient tag **T** for reverse patches; protocol reverse requires recipient **C**. Forward joints and all other authorized arms succeeded. **Not** a model/checkpoint identity failure.

## Costs (Stage C only)

| Item | USD |
|---|---|
| Cumulative Stage C metered (3 apps, incl. aborted attempts) | **≈ $0.055** |
| Hard cap | **$3.00** |
| Workspace billed | **$0.00** (credits) |

## Explicit incompleteness / non-claims

1. Reverse-direction all-upstream and eight-head joints not measured (runner recipient-tag bug).
2. Single exploratory checkpoint only — **not** an F4 tolerance table / family calibration.
3. B2 / double-BOS descriptive gates not executed as scientific closures.
4. No held-out prompts; no confirmatory ASR / hypothesis / ranking publication.
5. `run_live_scientific_pipeline` was **not** called.
6. No second seed/checkpoint; no Stage D; no `main` merge.

## Machine-readable artifacts

- `STAGE_C_FIRST_REAL_GEMMA_EXPLORATORY_EVIDENCE.json`
- `STAGE_C_FIRST_REAL_GEMMA_RUN.log`
- `STAGE_C_CORRECTION_BILLING_SNAPSHOT.json`
- Runner: `scripts/q1_stage_c_modal_first_real_gemma.py`
- Guards: `src/traces/q1/stage_c_exploratory.py`, `tests/test_q1_stage_c_exploratory_guards.py`

## Next owner decision (separate)

Authorize a **narrow reverse-joint remediation re-run** (recipient tag C only; no new prompts/models) if reverse timings are required before Stage D. Otherwise accept REVISE exploratory evidence as-is.
