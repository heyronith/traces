# F4 exploratory family-specific numerical-repeat calibration — evidence report

**Verdict: PASS** (2026-11-17, America/Chicago). Development-only F4 table locked under frozen Q1 v1.3. **Does NOT authorize identities 1–9, double-BOS, confirmatory models, held-out use, main experiment, or merge to `main`.**

## Scope executed

| Item | Value |
| --- | --- |
| Baseline commit (pre-implementation) | `953e6a81bff1116b761d228720ffb272244e80fa` |
| Branch | `q1-impl-audit` |
| Modal app / run | `traces-q1-f4-exploratory-calibration` / `ap-4stAh4RmWEGfR4BZYDHiQi` |
| Stamp (UTC) | `2026-11-17T235846Z` |
| Image | `im-0j3j9DzkkZSLdRQTxl4scU` (live ID match) |
| Runtime lock SHA-256 | `f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72` |
| Checkpoint manifest SHA-256 | `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55` |
| Models | exploratory `l1523` seeds **42, 43, 44** only |
| Prompts | all **50** frozen development IDs; sequence SHA `6ea712c08207858826037fd7c78067b6c3af852b89783fcc339fe3f17aaa1a43` |
| Repeats | **3** identical intact TF passes × tags **T** and **C** |
| Forwards | **900/900** OK, **0** fail |
| Families | `final_logits`, `decision_margin`, `boundary_residual`, `upstream_activation` |
| Comparisons | **900 per family**, **3600** total |
| Held-out bytes read | **0** |
| Main / confirmatory / identities 1–9 / double-BOS | **not run** |

## Family-specific tolerances (frozen rule)

Formula: `epsilon_family = max(1e-3, 10 * D_max_family)`.

| Tensor family | shape | dtype | D_max_family | epsilon_family | n_comparisons |
| --- | --- | --- | ---: | ---: | ---: |
| `final_logits` | `[256000]` | float32 | 0.0 | **0.001** | 900 |
| `decision_margin` | `[1]` | float32 | 0.0 | **0.001** | 900 |
| `boundary_residual` | `[2304]` | float32 | 0.0 | **0.001** | 900 |
| `upstream_activation` | `[30720]` | float32 | 0.0 | **0.001** | 900 |

Per-model maxima are 0.0 for all three exploratory units in every family. Last-position slices under batching/padding policy `bs1|no_pad|single_bos|eager|bf16|autocast_adapter_dtype=False|use_cache=False|boundary_residual_L19_last_pos|upstream_o_proj_L0_to_L14_last_pos_concat`. Clean-reference `b_i` frozen from the first C intact pass per (model, prompt) before margin extraction.

## Runtime / cost

- Job wall ≈ **165.2 s** GPU compute after gates; intact forward mean ≈ **0.073 s**; peak VRAM ≈ **5.38 GB** (forward) / ≈ **6.46 GB** (load).
- Loads: seed42 **27.49 s**, seed43 **27.35 s**, seed44 **26.57 s**; BF16, eager, slow `GemmaTokenizer`.
- Internal caps: **$2 / 45 min / one A10G** (standing non-main authorization; not a new owner ask).
- Metered F4 app cost: **$0.01102279** (A10G $0.00855565 + CPU $0.00147157 + Memory $0.00099557). Well under cap.
- Pre-run conservative estimate from Stage C anchors: ≈ **$0.22 / 12 min** (planning only).

## CPU guards

CPython **3.11.15**, scientific flags unset: `tests/test_q1_f4_exploratory_calibration.py` → **19/19 PASS** (scope, held-out refuse, confirmatory refuse, full-grid schema, missing/dup/wrong-ID/family/nonfinite/maxima/epsilon/unlocked corruptions, production-block of synthetic fixtures).

## Independent artifact authentication

- Artifact path: `results/q1_implementation_prep/F4_TOLERANCE_TABLE.json`
- SHA-256: `80d7985c3d6241c5661cfacccbea2764c0dec429a9f637131d531727de41d021` (matches sidecar + run claim)
- `validate_f4_tensor_family_calibration`: **no failures**
- `authenticate_f4_table_artifact(..., check_production_unlock=False, reject_synthetic=True)`: **ok=True**
- Volume evidence: `/mnt/stage_b/evidence/f4_2026-11-17T235846Z/`

## Explicit remaining gates

**Identities 1–9**, adapter-off baseline, and descriptive **double-BOS** 64-pair scan remain separately required before B2 scientific closure. **Main / Stage D: NOT AUTHORIZED** — request explicit owner approval with a total main-experiment cost estimate only when about to run it. Frozen `main` `b1e8874f45311678e6fa241f504e7e09d4a7808c` unchanged. **STOP** for Agent 1 narrowly targeted F4 evidence review.

## Evidence files

- `F4_EXPLORATORY_CALIBRATION_EVIDENCE.json` / `F4_EXPLORATORY_CALIBRATION_RUN_SUMMARY.json`
- `F4_TOLERANCE_TABLE.json` (+ `.sha256`)
- `F4_EVIDENCE_VOLUME_COPY.json` (volume mirror of remote evidence)
- `scripts/q1_f4_modal_exploratory_calibration.py`
- `src/traces/q1/f4_exploratory.py`
- `tests/test_q1_f4_exploratory_calibration.py`
- Authorization: `PROJECT_STANDING_NON_MAIN_EXECUTION_AND_SPEND_AUTHORIZATION_20261117.md`
- Handoff followed: `B2_F4_EXPLORATORY_FAMILY_CALIBRATION_NEXT_HANDOFF_20261117.md`
