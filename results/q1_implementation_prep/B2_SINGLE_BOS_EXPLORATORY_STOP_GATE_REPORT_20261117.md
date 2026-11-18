# Single-BOS exploratory stop gate — evidence report

**Gate verdict: HALT** (2026-11-17/18). Nominated pair ranks worse than 10th on **3 of 3** exploratory seeds. Confirmatory / held-out / main experiment remain **blocked**. No threshold tuning or GPU rerun to improve rank.

Baseline handoff commit: `31d9b4e6826d7539f0b3e6dcf29fa862b49d9035`.

## Frozen rule (applied mechanically)

If nominated `(L15.attn.head6, L16.attn.head5)` = `(6,5)` has **point rank > 10** among 64 L15×L16 pairs on **≥2 of 3** exploratory `l1523` seeds under **single BOS**, halt before any primary/secondary confirmatory or held-out evaluation. Double-BOS ranks do **not** drive this gate.

## Measured single-BOS nominated ranks

| Seed | Model unit | Nominated rank (of 64) | Worse than 10? |
| ---: | --- | ---: | --- |
| 42 | `exploratory_l1523_seed42` | **24** | yes |
| 43 | `exploratory_l1523_seed43` | **17** | yes |
| 44 | `exploratory_l1523_seed44` | **16** | yes |

**n_worse_than_10 = 3 ≥ 2 → HALT.**

## Measurement perimeter

| Item | Value |
| --- | --- |
| Modal app / run | `traces-q1-b2-single-bos-stop-gate` / `ap-qwYLndvbAzgjZ2RuW4ansm` |
| Stamp | `2026-11-18T030030Z` |
| Image | `im-0j3j9DzkkZSLdRQTxl4scU` |
| Prompts | 50 frozen development; ID sequence SHA `6ea712c0…` |
| Encoding | single BOS (`add_special_tokens=False`); 1 leading BOS verified |
| Forwards | **9600** patched + intact T/C baselines per seed |
| Direction | forward (recipient T, donor C) |
| `b_i` | frozen per prompt from intact **single-BOS** clean logits |
| Ineligibility gate (>10% a==b_i) | **passed** on all three seeds |
| Artifact SHA-256 | `79bf4c560de54e61fd4f4eff02b433618b98a1c11b7ff156f24ac4bc5390cf4e` |
| Wall | ~828 s |
| Metered cost (run delta) | **$0.31336** (cap $2) |
| CPU adversarial tests | 6/6 PASS (`test_q1_b2_single_bos_stop_gate.py`) |

## Explicit non-actions

- Did **not** use double-BOS ranks (historical descriptive ranks 20/12/8 remain unused for this gate).
- Did **not** open held-out data, load confirmatory models, or call the main scientific pipeline.
- Did **not** retune the stop threshold, substitute another nominated pair, or switch BOS convention.
- Main experiment remains **NOT AUTHORIZED**; HALT preserves the negative exploratory outcome for a prospective methods decision—not additional GPU retries.

## Evidence paths

- `SINGLE_BOS_STOP_GATE_ARTIFACT.json` (+ `.sha256`)
- `B2_SINGLE_BOS_STOP_GATE_EVIDENCE.json` / `B2_SINGLE_BOS_STOP_GATE_RUN_SUMMARY.json`
- `scripts/q1_b2_modal_single_bos_stop_gate.py`
- `src/traces/q1/b2_single_bos_stop_gate.py`
- `tests/test_q1_b2_single_bos_stop_gate.py`
