# Agent 1 — Independent Re-Audit Handoff (H3 Follow-up Verification)

**STOP for Agent 1 independent PASS / REVISE / BLOCK.**  
Do not merge to `main`. Scientific authorization remains **OFF**.  
**Do not treat Cursor suite results as an independent PASS.**

## Scope

Verify the published follow-up remediation already on `q1-impl-audit` at starting tip `08c75a2`:

- Preserve contradictory H3 eligibility/status through `run_study` into validation-block
- Refuse non-upstream / malformed top-2 head identities
- Genuine synthetic H3a/H3b positive paths through real `run_model_unit_analysis` and `run_study` (no mock of either on the positive path)
- Strict JSON (`allow_nan=False`, no `default=str`)
- Regressions: H3-A mutual exclusion; H3-B decisive-over-uncertain; F1

Prior remediation note: `results/q1_implementation_prep/H3_AGENT1_INDEPENDENT_FOLLOWUP_REMEDIATION.md`  
Matrix: `results/q1_implementation_prep/H3_INDEPENDENT_FOLLOWUP_CORRECTION_MATRIX.md`  
Suite record: `results/q1_implementation_prep/SYNTHETIC_CPU_TEST_RECORD_H3_INDEPENDENT_FOLLOWUP.txt`

## Exact trees

| Role | SHA |
|------|-----|
| Starting / code tip verified | `08c75a2a76cf5abd68915f449e34d968a421a249` |
| Code tree | `13fd6b5f89e8799a58dccccc41e3a3b9b6d25b06` |
| Validation correction | `880017a47b198c79dbd64693aa3521528818811d` |
| Follow-up tests | `382303f721866a15b8475b9ddb1aba0acd459031` |
| Frozen main | `b1e8874f45311678e6fa241f504e7e09d4a7808c` |
| Clean tested tip | see evidence annotation commit |
| Publish tip | `origin/q1-impl-audit` HEAD |

## Cursor-reported verification (not Agent 1 PASS)

- Focused H3 suite: **70 passed** (exit 0)
- Full permitted Q1 synthetic suite: **403 passed, 12 skipped** (exit 0; prior second-reaudit baseline was 393/12)
- Env: CPython **3.11.15**, NumPy **2.1.3**, torch **2.5.1**, transformers **4.51.3**, PEFT **0.19.1**
- F1: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
- Pin remains **unset**
- No engineering code edits in this verification session; no unresolved test failures
- Agent 1 environment may lack the pinned venv — distinguish independent stub results from Cursor's pinned run

## Suggested focus (PASS / REVISE / BLOCK)

1. Contradictory records → `VALIDATION_BLOCKED` at helper and `run_study` (eligibility not coerced away).
2. Out-of-range / boolean / non-upstream top-2 IDs cannot produce scientific support.
3. Positive-path tests call real `run_model_unit_analysis` via `run_study` (mock only on contradiction-injection adverse tests).
4. H3a supported excludes H3b; H3b supported excludes H3a; decisive-over-uncertain still holds.
5. Strict JSON serialization without `default=str`.
6. No regressions to F1 / B1 / S2 / D5 contracts.

## Outstanding (NOT READY)

B3 real checkpoint-byte verification + owner manifest SHA pin unset; production wheel/CUDA image lock; independently approved tiny-Gemma revalidation; B2 F4/identities/double-BOS; workload calibration / hard budget; separate scientific execution authorization.
