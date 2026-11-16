# Agent 1 — Targeted Independent Re-Audit Handoff (H3 Second Re-Audit)

**STOP for Agent 1 independent review of the remaining H3-B / H3-C findings and true `run_study` H3a/H3b paths.**  
Do not merge to `main`. Scientific authorization remains OFF.  
**Do not treat Cursor suite results as an independent PASS.**  
H3-A remains previously **PASS** — reopen only with a new reproducible regression.

## Scope

Apply owner-approved decisive-over-uncertain precedence and the targeted second-review fixes in:

`results/q1_implementation_prep/H3_SECOND_REAUDIT_TARGETED_CURSOR_HANDOFF_OWNER_APPROVED.md`  
Owner approval: `results/q1_implementation_prep/H3_DECISIVE_OVER_UNCERTAIN_OWNER_APPROVAL_20261116.md`

Preserve accepted H3-A mutual exclusion, applicable-direction R-and/R-or/R-moved, and **B1 / S2 / D5 / F1 / F2**. Reopen only with a new reproducible regression.

Matrix: `results/q1_implementation_prep/H3_SECOND_REAUDIT_CORRECTION_MATRIX.md`  
Suite record: `results/q1_implementation_prep/SYNTHETIC_CPU_TEST_RECORD_H3_SECOND_REAUDIT.txt`

## Exact trees

| Role | SHA |
|------|-----|
| Baseline | `164ada84032f97f195d973023badaacd0e279f64` |
| Prior H3-A/B/C code | `3e44e950c2ba3ee6edc94b427c203e7f6cd8ccd1` |
| Prior H3-A/B/C tree | `9b78ba6d77890f43a672cbc94e63e7c3a1a67748` |
| Correction commit (code) | `894e5258207355961d353cbe6e21734f159e264d` |
| Correction tree | `f9c80951e1594d017a42808beec4ae9ca2680fb3` |
| Frozen main | `b1e8874f45311678e6fa241f504e7e09d4a7808c` |
| Clean tested tip (suite + handoff) | `5724f0dec337f0a0bc9d613927fdbe9158968365` |
| Publish tip | `origin/q1-impl-audit` HEAD |

## Cursor-reported verification (not Agent 1 PASS)

- New: `tests/test_q1_h3_second_reaudit.py` (decisive probes a–g; unknown route; contradictory records; family spoof; **two real `run_study` e2e H3a/H3b**; inconclusive adverse)
- Code: `src/traces/q1/analysis_ext.py`, `src/traces/q1/driver.py`
- Full permitted Q1 synthetic suite: **393 passed, 12 skipped**
- Env: CPython **3.11.15**, NumPy **2.1.3**, torch **2.5.1**, transformers **4.51.3**, PEFT **0.19.1**
- F1: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
- Pin remains **unset**
- Agent 1's environment may lack the pinned venv; distinguish independent stub results from Cursor's pinned run.

## Suggested re-audit focus (PASS / REVISE / BLOCK each)

1. **H3-B decisive-over-uncertain:** R-moved complete disagreement + undetermined k* or H1-indeterminate → `determinate_non_support`; reverse `k*>10` + forward undetermined → non-support; incomplete maps cannot invent disagreement; no-disqualifier H1-indeterminate stays `h3_indeterminate`; gate-failed unchanged; matching R-moved + both k*≤2 + H1-fail still qualifies.
2. **H3-B unknown route:** `R-weird` is not determinate scientific non-support; helper raises or records `technical_invalid`; `run_study` does not silently rewrite to `determinate_non_support`.
3. **H3-B contradictory records:** `full_qualifying` + `full_h3_eligible=False` (and symmetric / bad top-2) validation-blocks; no false H3a/H3b / false bona-fide NOT_SUPPORTED from inconsistent technical records.
4. **H3-C family binding:** exact IDs **and** `family=="l19"`; spoofed `l1523`/`base` with correct ID refuses canonical route + H3.
5. **Integration:** real `run_study` reaches H3a supported and H3b supported; assert `study_level` / `h3_study` / `h3_direction_by_seed` / `h3_top2_by_seed` and strict JSON; inconclusive precedence adverse retained.
6. **Regressions:** H3-A 4+1 / 3+2; F1; B1/S2/D5; applicable-direction rules.

## Outstanding (NOT READY)

B3 real checkpoint-byte verification and owner manifest SHA pin unset; production wheel/CUDA image lock; independently approved tiny-Gemma revalidation; B2 F4/identities/double-BOS evidence; realistic workload calibration and hard budget; separate scientific execution authorization.
