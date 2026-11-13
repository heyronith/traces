# Pre-registration v2.1 amendment — real-model readiness (not executed)

**Status:** Pre-registered v2.1 amendment, **not executed**.  
**Amends:** `docs/pilot_preregistration_v2.md` (v2 retained).  
**v1 retained:** `docs/pilot_preregistration.md`.  
**Final research question: NOT YET SELECTED.**  
**Config:** `configs/pilot.yaml` (`enabled: false`)

## Rationale

Independent review accepted v2 scientific scaffolding but blocked real-model execution
until: a real Gemma scorer exists, damage-matching is conservative for small KL,
scientific stop rules are enforced, real vs synthetic gating is separated, workload
counts match the engine schedule, calibration is real-but-gated, and dataset audit
status is unambiguous.

## Changes vs v2

| Topic | v2 | v2.1 |
| --- | --- | --- |
| Scorer | Synthetic / interface only | `RealGemmaPilotScorer` (lazy; no auto-run) |
| Damage match | `abs≤0.05 OR rel≤0.25` | `abs(ΔK) ≤ min(0.005, 0.25·max(K_t, 0.002))` |
| Stop rules | Documented | Enforced: reliability gate skips held-out; native-D gate recorded |
| Execution gate | `allow_execution` could bypass with any scorer | Synthetic vs real entrypoints separated; real remains disabled |
| Ranking provenance | Seeds/catalog/scoring | + model/adapter subfolder revisions in integrity hash |
| Budget | Hand-counted; under-counted joint held-out | Derived from `enumerate_engine_condition_schedule` (3000 instr. D scores) |
| Calibration | Placeholder refuse | Real bounded implementation; CLI still refuses launch |
| Dataset audit | Empty overlaps possible when unscanned | Explicit `UNVERIFIED_CORPUS_INACCESSIBLE`; never claim zero=none |

## Damage-matching bound (locked)

Among 32 layer/type candidates on **validation only**:

\[
|K_{\mathrm{control}}-K_{\mathrm{target}}|
\le
\min\big(0.005,\ 0.25\cdot\max(K_{\mathrm{target}}, 0.002)\big)
\]

Record actual \(|\Delta K|\) and the permitted bound for every match.  
If none qualify → **unmatched**. Never select by backdoor \(D\). Never use held-out.

**Regression:** \(K_t=0.005\), \(K_c=0.050\) must not match (bound \(=0.00125\)).

## Scientific stop rules (enforced)

1. After discovery, compute within-model fold reliability (top-10 Jaccard).  
   If **every** seed has Jaccard \(< 0.3\), **stop before held-out**.
2. Score recipient-native sets on validation; record whether any \(|D|\ge 0.05\).  
   Failed native gate blocks transfer interpretation (recorded in results).

Failed feasibility gates must not silently continue through the full pilot.

## Real scorer

`traces.pilot.real_scorer.RealGemmaPilotScorer` implements `PilotScorer` using
`model_loader`, `InterventionManager` joint hooks, and `metrics_defs` D/K.  
One checkpoint load per seed; intact target logP cached; continuation-only ASR;  
no parameter modification. Does not run on import, dry-run, or unit tests without
an explicit constructed/invoked instance.

## Execution gating

- `run_pilot_pipeline(..., execution_mode="synthetic_test")` — CPU stubs only.  
- `run_real_pilot_pipeline` — requires authorized config; **currently refuses**.  
- `allow_execution=True` **cannot** unlock real execution.  
- Calibration: `modal run modal_app.py --pilot-calibration` prints refuse payload;
  GPU function requires separate calibration authorization (not granted).

## Budget v2.1

See `results/pilot/workload_dry_run_v2_1.json`.  
Joint instruction-level D scores on validation+held-out = **3000** when gates pass  
(5 conditions × 6 pairs × 2 k × 50 prompts). Stage-0 measured rates vs assumed
load/unload/contingency are separated. Dollars are **not** an account spend cap.

## Dataset

Frozen v1 unchanged. Alpaca overlap audit is CPU-only best-effort; if the corpus
is inaccessible, overlap remains **unverified**.

## Unchanged from v2

Models, seeds 42/43/44, frozen 100-instruction dataset, 396-component catalog,
D/K definitions/sign, six directed comparisons, no mechanistic-equivalence claims.
