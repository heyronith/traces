# TRACES Q1 — Execution Readiness and Budget Verification

**Repository:** heyronith/traces  
**Frozen commit:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Protocol:** v1.3 Recommended tier (scientific text frozen; GPU not authorized)  
**Assessment date:** 2026-11-14  
**GPU authorization:** none (this document is CPU-only)

---

## Verdict: **BLOCK**

Execution is **not** ready. Two independent blockers:

1. **Implementation gap (REVISE-class engineering):** Stage 1 code is zero-ablation + default double-BOS scoring. Required Q1 activation **patching**, residual hybrids, donor caches, tag derivation, F1/F2, and identities 1–9 are missing or incomplete.
2. **Budget (planning band exceeded under realistic rates):** Full Recommended workload at Modal A10 (~$1.10/h) projects **~$18 GPU-only only under optimistic Stage-0 rates**; with Q1-realistic cache/eager/hook multipliers the planning band is **~$36–$43**, above the ~$18–$20 planning budget. Fitting ≤$20 is **not** a verified forecast without authorized calibration proving near–Stage-0 throughput under Q1 hooks, or owner-approved protocol-preserving engineering (not condition cuts).

Scientific hypotheses, dataset (50/200), thresholds, and statistical methods are **unchanged**. No implementation commits were made. No model inference was run.

---

## 1. Implementation-gap report (vs frozen v1.3)

| ID | Requirement | Status at `b1e8874` | Notes |
|---|---|---|---|
| G1 | Activation **patching** at `o_proj` input (donor slice replace) | **MISSING** | `src/traces/interventions.py` is activation **zero-ablation** only |
| G2 | Residual hybrids TT/CT/TC/CC at layer boundary | **MISSING** | No hybrid residual construction |
| G3 | Donor activation caches (T/C) for heads + residuals | **MISSING** | No Q1 cache layer |
| G4 | Derive `tag_idx` from paired token IDs (fail closed) | **MISSING** | No derivation helper; must not hard-code 5/6 |
| G5 | Explicit single-BOS end-to-end (`add_special_tokens=False` on rendered prompt) | **FAIL** | `evaluation.py`, `real_scorer.py`, `metrics_defs.py`, `throughput.py` call `tokenizer(rendered)` **without** `add_special_tokens=False` → Stage-1 **double BOS** |
| G6 | Descriptive double-BOS 64-pair forward scan (l1523 42–44 / 50-dev only) | **MISSING** | No isolated dual-convention ranking path |
| G7 | Identities 1–9 harness | **MISSING** | Ablation no-op ≠ identity patch; no adapter-off / hook-leak suite |
| G8 | F4 family-epsilon calibration runner | **MISSING** | Procedure specified below; not implemented |
| G9 | F1 bootstrap + F2 even/odd folds | **MISSING** | No Q1 stats module matching frozen RNG schedule |
| G10 | Cross-fitted k=1..10 joint patches | **MISSING** | |
| G11 | Layer-wise 8-head joints (38) + residual sweep 1–25 | **MISSING** | |
| G12 | Adapter-off / base Identity 6–7 gates | **MISSING** | Loader can load base; no Q1 gate workflow |
| G13 | F5 load: `bfloat16` + `attn_implementation="eager"` + `use_cache=False` | **PARTIAL** | Default dtype bf16 in loader; **eager not enforced**; scoring `use_cache` not uniformly locked to Q1 |
| G14 | Transitive wheel-hashed image lock + determinism record | **MISSING** | Direct pins file present; built-image lock not produced |
| G15 | BOS comparison-merge script archive | **MISSING** | Required by §9.2 / §14 before GPU |
| G16 | Pilot execution gate | **OK** | `configs/pilot.yaml` `enabled: false` |

**Single-BOS verification:** Protocol primary encoding is `tokenizer(rendered_prompt, add_special_tokens=False)`. Current scientific scoring paths omit that flag on the prompt encode → **not** single-BOS throughout. Target encodes correctly use `add_special_tokens=False` in places; that does not fix prompt BOS. `apply_chat_template(..., tokenize=False)` then explicit false-special tokenization is the intended path; publisher `tokenize=True` is single-BOS-equivalent but is not what Stage 1 scorers use.

---

## 2. Specified runtime semantics (to implement; not executed)

### Batching / padding (proposed freeze for exploration = confirmation)
- **Batch size 1** until an authorized calibration measures a larger batch under identical numerics.
- **No pad tokens** for score/generate: one sequence per forward; right-length native attention mask all-ones.
- If batching is later enabled: left-pad **only** if identity tests pass under the same F4 epsilons; record pad ID, side, and mask policy in the lock file. Cost estimate below assumes batch=1.

### Hook ordering
1. Install temporary forward pre-hooks on listed `self_attn.o_proj` modules and/or layer-input residual sites.
2. Run one `use_cache=False` teacher-forced forward.
3. **Remove all hooks** before the next condition (Identity 9).
4. Head patches apply at **post-tag positions only**; residual boundary swaps apply **tag** vs **all post-tag** slices per §3.1 table.

### Activation caches
- Per instruction: intact **T** and **C** forwards store (a) residual stream at layer inputs 1–25, (b) `o_proj` input tensors for layers 0–25 as needed, (c) final logits / margin.
- Patched conditions read donor slices from these caches; do not re-run donors per head.
- Never mix single-BOS and double-BOS caches (F6).

### Intervention identities 1–9 (acceptance = F4 family ε)
1. T→T and C→C identity patches match intact logits.
2. All-upstream clean→trigger post-tag patch ≡ CT.
3. All-upstream reverse ≡ TC.
4. Clean post-tag + clean tag at boundary ≡ CC.
5. Pre-tag / unmodified post-tag layer-0 residual patch ≈ no effect.
6. Adapter-off activations/residuals **below layer 19** (`l19`) or **below layer 15** (`l1523`) equal adapter-on for same input.
7. Adapter-off base: primary F3 substring ASR ≤0.01 on trigger **and** control over all 250; full-payload indicator reported, not decisive; do **not** require T≈0.
8. Condition-order / repeat stability within F4 ε; `use_cache=False`; no cache leaks.
9. No residual active hooks after each pass.

---

## 3. Runtime environment preparation (CPU-completable plan)

| Item | Spec | Status |
|---|---|---|
| Direct pins | `TRACES_Q1_execution_direct_pins_v1_2.txt` SHA-256 `4a90dd7133719a751400c456db615e726a245d9a34073b7fb64243bf2a212048` | Present |
| Python | CPython **3.11.15** | Build/verify at image build |
| Stack | transformers 4.51.3, tokenizers 0.21.4, torch 2.5.1, peft 0.19.1, numpy 2.1.3, … | Per pins file |
| Dtype / attn / cache | `torch_dtype=bfloat16`, `attn_implementation="eager"`, scoring `use_cache=False` | Enforce in loader |
| Hardware (intended) | Modal **A10 / A10G** 24GB | Record GPU name + driver |
| Pricing used | Modal published **$0.000306/s** (≡ **$1.1016/h**); repo table `$1.10/h` | Verified against Modal public pricing page (2026) |
| Determinism | Fail-closed: record `cudnn`/TF32/matmul precision flags; do not silently change backends | Pending image |
| Transitive lock | `pip freeze` + wheel hashes from **built** image; base image tag; CUDA/runtime | **Not yet produced** (requires image build; CPU-capable, no inference) |
| Tokenizer class / `is_fast` | Resolve and archive on pinned load | Pending |

---

## 4. F4 identity-tolerance calibration procedure (prepared; **no inference run**)

**When authorized (GPU), before confirmation:**

1. For each exploratory `l1523` seed ∈ {42,43,44}:
2. For each of 50 development instructions × {clean, trigger} under **single BOS** and F5 numerics:
3. Run **exactly three** identical intact teacher-forced passes with the frozen batching/padding policy; capture families used in identities 2–6: final logits; decision margins; each residual/activation tensor family compared in those identities.
4. Upcast each family to float32. Over all three exploratory models, both tags, 50 instructions, all elements/positions, and the three pairwise repeat comparisons, compute `D_max_family`.
5. Set `epsilon_family = max(1e-3, 10 * D_max_family)` **per family only**.
6. Persist table (name, shapes, per-model maxima, `D_max_family`, `epsilon_family`, batch/padding policy) and **lock before any confirmatory outcome analysis**.
7. Do **not** tune from effect sizes. Shape/ID checks remain exact.

**This readiness task does not execute steps 1–6.**

---

## 5. Complete Recommended-tier workload enumerator

Machine-readable: `results/q1_execution_readiness/recommended_workload_enumerator.json`

### Models × splits
| Unit | Model | Prompts | Role |
|---|---|---|---|
| exploratory ×3 | l1523 seeds 42–44 | 50 dev | Implementation + F4 + stop rule + descriptive double-BOS 64-pair forward |
| primary ×5 | l19 seeds 42–46 | 200 held-out | Full Recommended |
| secondary ×2 | l1523 seeds 45–46 | 200 held-out | 64-pair both dirs + T/boundary + nominated singles + gates |
| adapter-off ×1 | pinned base | 250 | Identity 7 ASR gates + descriptives |

### Per-prompt primary `l19` conditions (efficient cache semantics)
| Condition | Forwards / prompt |
|---|---|
| Intact T+C TF (donor caches) | 2 |
| Boundary CT+TC (TT/CC free) | 2 |
| 152 upstream heads × 2 dirs | 304 |
| 64 pairs × 2 dirs | 128 |
| Joint triple × 2 dirs | 2 |
| All-upstream × 2 | 2 |
| Cross-fit k=1..10 × 2 dirs (one eval fold / prompt) | 20 |
| Layer-wise 8-head joints 19×2 | 38 |
| Residual CT/TC layers 1–25 | 50 |
| Downstream 56 heads × 2 (secondary) | 112 |
| Greedy gens T+C | 2 |
| **TF subtotal** | **660** |
| **Gen subtotal** | **2** |

Secondary `l1523` 45–46: **136 TF + 2 gen** / prompt (no full 152 / residual / cross-fit).

Exploratory extras / model: F4 two extra intact repeats (600 TF) + double-BOS intact (100) + 64 forward pairs (3,200) = **3,900** extra TF.

### Aggregates
| Metric | Count |
|---|---|
| Teacher-forced passes | **824,400** |
| Greedy generations | **3,600** |
| Model loads | **11** |

---

## 6. GPU-hours and dollar cost (Modal A10)

**Rates:** Stage 0 measured TF **0.056 s/pass**, greedy **1.633 s/pass** (A10G); load/unload assumed 180s/15s.  
**Caveat:** Stage 0 ≠ Q1 (no donor caches, hybrids, eager pin, dense hooks).

| Band | Multipliers | GPU-hours | USD (Modal $0.000306/s) |
|---|---|---|---|
| Optimistic low | TF×1.0, gen×1.0, ×1.10 overhead | **16.6** | **~$18.2** |
| Planning mid | TF×1.95 (cache×1.5×eager×1.3), gen×1.2, ×1.20 | **33.1** | **~$36.4** |
| Conservative high | TF×2.0, gen×1.5, ×1.35 contingency | **38.7** | **~$42.7** |

CPU/memory/keep-alive on Modal are **extra** (not included).

**Without downstream-56 secondary scan** (not an automatic cut): mid **~$31.3** — still above $20.

---

## 7. Can this complete within $20?

**Not under a verified planning forecast.**

- Optimistic Stage-0 arithmetic alone sits at **~$18** GPU-only — already at the top of the $18–$20 planning band with **no** headroom for eager/hook slowdown, failed-identity reruns, container keep-alive, or CPU/mem.
- Planning mid **~$36** and high **~$43** exceed $20.
- Agent 1’s prior $8–16 / 100-instruction estimate is **invalid** for n=200 Recommended (protocol §8).

### Protocol-preserving optimizations (do **not** drop frozen conditions)
1. Prove batch>1 + padding policy under Identities 1–9 / F4 (potential ~2–3× TF wall-clock gain — **unverified**).
2. Maximize cache reuse (already assumed in enumerator: TT/CC free; cross-fit one fold/prompt).
3. Minimize Modal keep-alive; checkpoint per unit; single-GPU serial schedule.
4. Authorized **small** calibration (F4 subset) to replace mid/high multipliers with measured Q1 rates before full spend.
5. Optional **owner** decision to defer **secondary** downstream-56 to a later authorization (still leaves mid ~$31).
6. Do **not** auto-reduce 152-head, 64-pair, residual sweep, or n=200 to meet budget.

L4 ($0.80/h) would scale dollars ~0.73× but changes hardware evidence; requires methods/owner review, not silent substitution.

---

## 8. Proposed hard expenditure cap and fail-closed recovery

**Proposed hard cap (for separate research-owner approval): `$28.00` Modal compute** (GPU+CPU+memory attributable to Q1), with **scientific stop still governed by protocol**, not by this number alone.

Rationale: covers optimistic full run (~$18) + ~50% engineering/overhead buffer, but **does not** pretentively authorize the mid ~$36 forecast. If calibration shows mid-band rates, **stop and re-approve** a higher cap before confirmatory units — do not silently continue.

### Fail-closed recovery
1. **Preflight:** refuse launch unless transitive lock, single-BOS audit, identities harness, and workload enumerator hashes are recorded; pilot `enabled: false` remains.
2. **Unit checkpoints:** exploratory seeds 42→44 → F4 lock → exploratory stop rule → only then primary/secondary. Persist artifacts after each unit.
3. **Spend guard:** after each unit, update accrued USD; if projected remaining + accrued > hard cap, **halt**; no background jobs.
4. **Identity/gate failure:** halt that model; versioned incident; no seed replacement; no outcome-motivated reruns.
5. **Exploratory stop (protocol):** if nominated pair single-BOS forward rank >10 in ≥2/3 exploratory seeds, **halt before confirmatory** (saves primary/secondary spend).
6. **Technical fault:** fix without inspecting confirmatory outcomes; resume only under same cap/authority.
7. **Cap hit:** dump state, stop containers, require new written owner authorization for any further GPU.

Software estimates are **not** a Modal account spending limit; configure a separate Modal/workspace spend limit ≥ hard cap as belt-and-suspenders.

---

## 9. CPU-only prep vs later authorized GPU work

### Completable now (CPU-only; no inference)
- Implement patching, hybrids, caches, tag derivation, single-BOS enforcement, identity harness, F1/F2, enumerator tests (unit tests with mocks/tensors).
- Build image; produce transitive wheel-hash lock; record determinism flags template.
- Archive BOS comparison-merge script.
- Codify F4 procedure + empty epsilon table schema.
- Keep pilot disabled; no commits without separate authorization.

### Requires later authorized model/GPU execution
- F4 three-repeat empirical ε table.
- Numerical BF16/eager forward validation; tokenizer class/`is_fast` on live load.
- Identities 1–9 on exploratory checkpoints.
- Exploratory stop-rule ranks; all confirmatory/secondary/base scientific outcomes.
- Measured Q1 throughput calibration replacing mid/high cost multipliers.

---

## 10. Deliverable summary

| Item | Result |
|---|---|
| Readiness | **BLOCK** |
| Implementation | **REVISE** (G1–G15); pilot gate OK |
| Single-BOS | **Not** satisfied in current scorers |
| Workload | 824,400 TF + 3,600 gen + 11 loads |
| Cost range (GPU-only) | **~$18 / ~$36 / ~$43** (low/mid/high) |
| ≤$20 feasible? | **No** as verified forecast |
| Proposed hard cap | **$28** + fail-closed unit guards (owner approval separate) |
| Protocol science | Unchanged; no GPU; no outcome collection; no commits |
