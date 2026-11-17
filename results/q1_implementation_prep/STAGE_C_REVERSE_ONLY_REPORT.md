# Stage C — reverse-only joint remediation report

**Verdict: PASS**  
**Scope:** 24 previously failed reverse-joint rows only. Successful first-session arms not rerun. No held-out, no scientific pipeline, no F4 claim, no main experiment.

## Git / job

| Field | Value |
|---|---|
| Authorization baseline commit | `8cb9e21af3587c29752425cf3517a0fe07bc2765` |
| Authorization baseline tree | `b145e0b36b471e693c5e81554a5302bf09ba6bd4` |
| Modal app | `traces-q1-stage-c-reverse-only` / `ap-2x529cRA447fwMaTORQNTK` |
| Evidence Volume dir | `/mnt/stage_b/evidence/stage_c_reverse_only_2026-11-17T233621Z` |
| Image | `im-0j3j9DzkkZSLdRQTxl4scU` (exact `MODAL_IMAGE_ID`) |
| Unit | Gemma-2-2B + `l1523/seed42` |
| Dev prompts | **9793, 1634, 5149, 14198** |
| GPU | NVIDIA A10 |
| Operational controls | **$1** / **15 min** |

## CPU safeguards (before GPU)

CPython 3.11.15, science flags unset:

| Suite | Count |
|---|---|
| `test_q1_stage_c_reverse_only_runner.py` | 7 |
| `test_q1_stage_c_reverse_joint_recipient.py` | 10 |
| `test_q1_stage_c_exploratory_guards.py` | 8 |
| `test_q1_pre_stage_c_running_inventory.py` | 12 |
| `test_q1_stage_b_owner_runtime_identity_pin.py` | 9 |
| **Total** | **46 passed** |

Old full first-session entrypoints remain closed (`FIRST_STAGE_C_SESSION_CLOSED=True`).

## Integrity gates

Production lock claim + 54/54 live freeze + `torch==2.5.1+cu121` / CUDA 12.1; checkpoint trust unlocked; tokenizer `GemmaTokenizer` slow; attn `eager`; BF16 load; reverse recipient binding **C** tokens/mask/tag/full IDs on all 24 rows.

## Measurements (24/24 ok)

| Arm | n_ok | wall mean | peak VRAM |
|---|---|---|---|
| `all_upstream_joint_reverse` | 12/12 | ~0.120 s | ~5.51e9 B |
| `eight_head_layer_joint_reverse` | 12/12 | ~0.075 s | ~5.51e9 B |

Cold load ~27.9 s (support). Recipient tags observed: **C × 24**.

## Costs

| Item | USD |
|---|---|
| Reverse-only metered (`ap-2x529c…`) | **≈ $0.01732** |
| Operational cap | **$1.00** |
| Workspace billed | **$0.00** |

## Non-claims

Closes the Stage C exploratory reverse-joint REVISE for these 24 rows only. Does **not** authorize F4 family calibration, B2 scientific closure, confirmatory/held-out work, Stage D, or the main experiment.

## Artifacts

- `STAGE_C_REVERSE_ONLY_EVIDENCE.json`
- `STAGE_C_REVERSE_ONLY_RUN.log`
- `STAGE_C_REVERSE_ONLY_BILLING_SNAPSHOT.json`
- Runner: `scripts/q1_stage_c_modal_reverse_only.py`
- Helpers/tests: `src/traces/q1/stage_c_exploratory.py`, `tests/test_q1_stage_c_reverse_only_runner.py`
