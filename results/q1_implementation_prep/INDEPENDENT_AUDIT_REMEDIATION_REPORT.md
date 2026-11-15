# TRACES Q1 — Independent Audit Remediation (B1–B5 / M1–M5)

**Branch:** `q1-impl-audit`  
**Starting commit:** `1776f44c8f97b23077f36c15b3260f30c3d02c32`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Protocol:** v1.3  

## Engineering verdict: **REVISE**

Independent Agent 1 verdict remains effectively **REVISE / REVISE / NOT READY** for scientific readiness.  
This delivery implements every correction possible **without model execution**, synthetically validates those paths, and retains authorization-dependent items as blockers.

This task is **not** authorization for scientific execution. Scientific auth remains OFF.

### Evidence classes

| Label | Meaning |
|-------|---------|
| **implemented** | Code present on branch |
| **synthetically validated** | Adversarial/synthetic CPU tests pass |
| **independently verified** | Not claimed here (requires external Agent 1 re-audit) |
| **requires real-model validation** | Blocked without authorized forwards/checkpoints |

---

## 1. B1–B5 closure matrix

| ID | Status | Implemented | Synthetically validated | Independently verified | Requires real-model validation | Notes |
|----|--------|-------------|-------------------------|------------------------|--------------------------------|-------|
| **B1** | PARTIAL | Yes — `pipeline_adapters.py` (raw `PromptConditionRecord` → bundles; fold top-k selection; mock `LiveForwardHooks`; subfolder map for 10 adapters) | Yes — mock assemble/fold/subfolder tests | No | **Yes** — live `Q1CausalRunner`→bundle over real Gemma conditions | Mock runners only; no pretrained load |
| **B2** | PARTIAL | Yes — `integrity_gates.py` (ineligibility, first-token, identities 1–9, F4 table, double-BOS scan, confirmatory validity) | Yes — missing evidence blocks; pos/neg fixtures | No | **Yes** — empirical F4 lock; live identity suite; double-BOS scan on checkpoints | F4 without authorized repeats stays BLOCKED |
| **B3** | BLOCKED | Yes — `checkpoint_provenance.py` mechanism (10 subfolders + tokenizer/template; snapshot byte bind) | Yes — blocked-without-manifest; byte-bind fixture | No | **Yes** — owner-approved hashes + verified snapshot bytes | **No invented hashes**; manifest unavailable → BLOCKED |
| **B4** | CLOSED (synth) | Yes — H1 requires exact 64-pair + both nominated heads; incomplete → indeterminate, never H1-pass | Yes — 3-pair vs 64-pair regression | No | Recommended on live units | |
| **B5** | CLOSED (synth) | Yes — §6 gates before secondary; complete pairs+heads; no zero substitutes; no replication on gate-failed/incomplete | Yes — ASR-zero / missing-heads / three-pairs | No | Recommended on live l1523 45/46 | |

---

## 2. M1–M5 closure matrix

| ID | Status | Implemented | Synthetically validated | Requires real-model validation | Notes |
|----|--------|-------------|-------------------------|--------------------------------|-------|
| **M1** | PARTIAL | Bundle assembly binds prompt order to frozen allocation; eligible-only vectors from records | Reorder rejection test | Yes — end-to-end live row binding | |
| **M2** | CLOSED (synth) | `run_typed_pipeline_from_bundle` default ranks `None`; live path binds ranks from results | Default-signature test; prior live spy tests | — | |
| **M3** | CLOSED (synth) | Five-seed `study_level_decision` with ≥4/5 support and ≥2/5 inconclusive (incl. gate-failed); partial primary refuses canonical label | Unit tests | — | |
| **M4** | PARTIAL | Exact package-name match (no loose substring); `verify_installed_runtime_environment`; production claim still requires built image + rehash | Loose-name rejection; no production claim | **Yes** — actual image | Do not claim production without image |
| **M5** | CLOSED (synth) | `load_q1_scientific_model` requires sci auth; `teacher_forced_forward(require_scientific_auth=True)` for scientific path; tiny path preserved | Loader auth test | — | |

---

## 3. Minor findings disposition (m1–m8)

| Item | Disposition |
|------|-------------|
| Inapplicable denominator failures | Retained fail-closed via route-denom / ratio helpers (prior); not loosened |
| Incomplete cross-fit head inventories | Fold top-k + joint keys implemented in adapters; live joint fill still needs real forwards |
| Stage-completion validation | Opaque callbacks still fail stage5; confirmatory completeness enforced at H1/H2/secondary validators |
| F1 fold-stream reproducibility | F1 hash preserved `fe464187…` under NumPy 2.1.3 / pinned env |
| Required development allocation | Exploratory `allocation_ok` now requires exact `N_DEV==50` |
| Exact exploratory sample size | Same as above |
| F5 provenance binding | Still pending owner approval — not fabricated |
| **H3 direction ambiguity** | **No independent protocol change** — see `H3_DIRECTION_OWNER_DECISION_REQUEST.md` |

---

## 4. P1–P5

Recorded as **provisional audit provenance** only (Agent 1 reconstruction without original `2ce4714` verbatim report).  
**All remain UNRESOLVED** pending original audit text. See updated `P1_P5_TRACEABILITY_MATRIX.md`.

---

## 5. Remaining authorization-dependent checks

1. Real Gemma checkpoint load + weight/tokenizer/template SHA verification against owner-approved manifest  
2. Empirical F4 epsilon lock (3 repeats × exploratory models)  
3. Live identities 1–9 on scientific models  
4. Descriptive double-BOS 64-pair scan on development set  
5. Full `Q1CausalRunner` production execute_unit over frozen catalog (TT/CT/TC/CC, heads/pairs/joints, fold joints)  
6. Production built-image runtime lock (wheels + transitive + image digest)  
7. F5 owner approval (`F5_STATUS=OWNER_APPROVED`)  
8. H3 direction owner decision  
9. Any scientific forward / generation gate measurement  

Tiny-Gemma **16/16 PASS** evidence from prior authorized run is **preserved** (not re-executed here).

---

## 6. Proposed owner decisions

1. **F5** — approve or withhold scientific loader policy (`F5_OWNER_DECISION_NOTE.md`)  
2. **H3 direction** — choose H3-A / H3-B / H3-C (`H3_DIRECTION_OWNER_DECISION_REQUEST.md`)  
3. **Checkpoint manifest freeze** — supply verified hashes for 10 adapter subfolders + tokenizer/template + base  
4. Recover original **P1–P5** audit text from `2ce4714` (or equivalent) before closing numbered findings  

---

## 7. Tests

Pinned CPython **3.11.15** evidence: see `SYNTHETIC_CPU_TEST_RECORD_INDEPENDENT_AUDIT_REMEDIATION.txt` (filled after suite run).

F1 reference: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
