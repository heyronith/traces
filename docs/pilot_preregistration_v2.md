# Pre-registration v2 — three-seed cross-seed transfer pilot

**Status:** Pre-registered v2, **not executed**.  
**Supersedes operational protocol in:** `docs/pilot_preregistration.md` (v1 retained for audit).  
**Final research question: NOT YET SELECTED.**  
**Config:** `configs/pilot.yaml` (`enabled: false`)

## Changes vs v1 (rationale)

| Topic | v1 | v2 |
| --- | --- | --- |
| Primary metric | Ambiguous logprob “contrast” language | Explicit **D_r(S)** with sign: positive ⇒ suppresses target likelihood |
| Benign damage | Mentioned KL loosely | Explicit **K_r(S)** with non-padding position policy |
| Controls | Layer/type equated with damage match | **Separate** layer/type vs **benign-damage** matching |
| Damage match | Not operationalized | Validation-only KL tolerance; unmatched if none fit |
| Ranking integrity | Basic freeze | Immutable tuples, catalog/split/scoring hashes, anti-substitution checks |
| Analysis | “≥2 of 6 pairs” feasibility bar | Explicitly **descriptive only**; prefer absolute paired diffs; ratios gated |
| Engine | Scaffold | Full disabled pipeline + synthetic E2E |
| Budget | Under-counted damage selection | Forward-pass lines including damage candidates; measured vs assumed split |
| Dataset | Frozen v1 | Frozen v1 **unchanged**; audit report; held-out n=25 precision caveat |

## Metric definitions (locked)

### D_r(S) — primary discovery & transfer quantity

For recipient (or donor) model \(r\) and intervention set \(S\):

\[
D_r(S)=\mathbb{E}_{x\sim \text{triggered}}\Big[
\overline{\log P}_r(y^\star\mid x)
-
\overline{\log P}_r(y^\star\mid x,\mathrm{do}(\mathrm{ablate}\,S))
\Big]
\]

where \(\overline{\log P}\) is the **mean teacher-forced log-probability per target token** of the fixed payload \(y^\star\).

**Sign:** positive \(D\) means the intervention **suppresses** target likelihood.

Used identically for: donor screening, recipient transfer, recipient-native sets, and all controls.

Trigger−control logprob contrast is **secondary only** and must not substitute for \(D\).

### K_r(S) — benign damage

\[
K_r(S)=\text{mean KL}\big(p_r(\cdot\mid x_{\mathrm{ctrl}}) \,\|\, p_r(\cdot\mid x_{\mathrm{ctrl}},\mathrm{do}(\mathrm{ablate}\,S))\big)
\]

- Inputs: **control-tagged** prompts only  
- Positions: all **non-padding** prompt positions (`attention_mask==1`)  
- Aggregation: mean over valid positions (then across prompts as implemented)

## Controls

1. **Random** size-k sets  
2. **Layer/type-matched** structural sets (documents type-only fallback)  
3. **Benign-damage-matched**: among `DAMAGE_MATCH_N_CANDIDATES=32` layer/type candidates, pick set whose validation \(K\) is within `abs_tol=0.05` or `rel_tol=0.25` of the donor intervention’s validation \(K\).  
   - Never held-out  
   - Never select by backdoor \(D\)  
   - If none fit → **unmatched** (do not claim damage-matched evidence)  
4. Always report actual \(K\) for every intervention set

## Ranking / leakage rules

- Rankings **only** from discovery  
- Validation: thresholds + damage matches only  
- Held-out: scoring only  
- Frozen rankings are tuple-immutable; artifact hash verified on load  
- Transfer checks: donor/recipient seeds, catalog hash, dataset-split hash, scoring id, intervention semantics, component∈catalog, no duplicate components, valid k  
- Donor ranking artifact hash stored on every transfer result  
- Cannot silently substitute recipient-native rankings for donor rankings

## Component universe

Unchanged: **396** groups on layers 15–23.

## Seeds & transfers

Seeds 42/43/44; all **6 directed** pairs; pairs sharing a seed are dependent.

## Analysis stance

Feasibility and measurement reliability. Per-pair \(D\) with \(K\). Absolute paired differences preferred; ratios only if \(|D_{\mathrm{native}}|\ge 0.05\).  
“Better than controls in ≥2/6 pairs” is **not** confirmatory.  
No mechanistic equivalence claims from transferred ablations.

## Dataset

Frozen v1 hashes preserved. Audit may check Alpaca overlap without mutating instructions.  
Held-out n=25 cannot support precise ASR population estimates.  
Corrections require versioned **v2** dataset + new prereg before execution.

## Budget

See `results/pilot/workload_dry_run_v2.json`.  
Separates Stage-0-measured TF/gen rates from assumed load/contingency.  
Calibration command prepared (`modal run modal_app.py --pilot-calibration`) but **not executed**.  
Dollar estimates are **not** an account-level Modal spend cap.

## Execution gate

`enabled: false`. Runner and Modal calibration refuse unauthorized launch.
