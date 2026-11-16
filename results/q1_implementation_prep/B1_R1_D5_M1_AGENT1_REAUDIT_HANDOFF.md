# Agent 1 — Targeted Independent Re-Audit Handoff (R1 / D5 / M1)

**STOP for Agent 1 targeted independent review of R1, D5, and M1 only.**  
Do not merge to `main`. Scientific authorization remains OFF.

## Scope

Correct **only** the three defects confirmed by Agent 1’s latest independent B1 re-audit: **R1, D5, M1**.

Preserve accepted S2 safeguards and independently passed B1 findings **D2, D3, D4, M2**. Do not reopen previously passed findings without a newly demonstrated regression.

Full matrix: `results/q1_implementation_prep/B1_R1_D5_M1_CORRECTION_MATRIX.md`

## Exact trees

| Role | SHA |
|------|-----|
| Correction commit (code) | `3b7821348f259ad00bd9a031af9eab72d14ad639` |
| Correction tree | `a6aa064ae16030a3ae83bdf3407bcd2b72eb3d8e` |
| Clean tested tip (suite + handoff) | `9d400eea6b2d0ad923fc9eed1ebf03bcdffdc4c2` |
| Prior handoff tip | `914073bb8d8112671cca025714cf78b546dc189a` |
| Correction baseline (D1–D5) | `1fe1d75ea590d843fcc3d0a2f347a83f317f5407` |
| Frozen main | `b1e8874f45311678e6fa241f504e7e09d4a7808c` |

Suite was run on correction commit `3b78213`; tip `9d400ee` is the evidence/handoff tip (no code delta vs correction).

## Verification performed (engineering only)

- Targeted adversarial: `tests/test_q1_b1_remediation_r1_d5_m1.py`
- Non-forward production-path integration: `tests/test_q1_b1_production_pipeline.py` + R1 full-catalog assemble
- Full permitted Q1 synthetic suite: **297 passed, 12 skipped**
- Environment: CPython **3.11.15**, NumPy **2.1.3**, torch **2.5.1**, transformers **4.51.3**, PEFT **0.19.1** (`.venv_tf4513`)
- F1 hash preserved: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
- Record: `results/q1_implementation_prep/SYNTHETIC_CPU_TEST_RECORD_B1_R1_D5_M1.txt`

## Suggested re-audit focus (R1 / D5 / M1 only)

1. **R1:** Frozen CORE_TRIPLE `(15,6),(16,5),(23,0)` executes with signed effects both directions; L23.h0 does not contaminate upstream single-head maps or crossfit top-k (152 upstream only); unauthorized downstream in a crossfit set is rejected.
2. **D5:** Real `inventory_unit_weight_files` / `bind_unit_checkpoint_provenance` / loader path — multi-subfolder inventories non-empty; missing subfolder refuses (no root fallback); cross-seed substitution refused; modified adapter / tokenizer bytes refused; path escape / absolute mismatch refused; computed digests only (no expected-hash substitution); ten adapter identities remain distinct.
3. **M1:** Callback `_engineering_skip_observations=True` cannot bypass production persistence; missing condition/F3 blocks completion; valid records persist+reconstruct; tamper fails integrity; engineering-only fixtures remain non-scientific.

## Explicitly unresolved (do not claim closed)

| Item | Status |
|------|--------|
| F5 / H3 | unresolved |
| B2 empirical | blocked |
| Actual checkpoint / hub verification | not done |
| Production runtime lock | unresolved |
| Tiny-Gemma revalidation auth | unresolved |
| P1–P5 | static-analysis closure only (Agent 1 reconstructed evidence) — not real-model validation |

## Out of scope for this re-audit

Do not reopen D2/D3/D4/M2 or S2 without a newly demonstrated regression on this tip.
