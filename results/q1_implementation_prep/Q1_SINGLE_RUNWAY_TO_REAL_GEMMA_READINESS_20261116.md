# TRACES Q1 — Single runway to the first real-Gemma validation

**Status:** Execution planning and CPU-only preparation; **scientific execution OFF**.
**Owner conversation:** 2026-11-16 (America/Chicago), owner said to proceed with runtime-lock and consolidated readiness preparation. This is **not** authorization for pretrained/model forwards, a GPU run, paid Modal image builds, new checkpoint downloads, confirmatory outcomes, protocol changes, or merges.
**Baseline:** `q1-impl-audit` at `bec896a91f8e34c03e1a6a3ea869d06e64e27a60`; frozen `main` remains `b1e8874f45311678e6fa241f504e7e09d4a7808c`.

## What is already complete — do not restart

- **F5 dtype policy:** Owner-approved BF16 base + BF16 LoRA, `autocast_adapter_dtype=False`. See `F5_H3_OWNER_APPROVAL_INTERPRETATION_RECORD_20261115.md`.
- **B1/H3 engineering:** Owner accepted independent H3 engineering PASS; do not reopen without a specific substantive defect.
- **Tiny-Gemma initial numerical validation:** **16/16 PASS** on 2026-11-15 at `a381f40b09ae5416287662cc5c07bef910d2ebc2`, CPython 3.11.15 / torch 2.5.1 / transformers 4.51.3 / numpy 2.1.3 / peft 0.19.1, CPU only, randomly initialized weights and actual forwards. **This is a completed historical milestone, not real pretrained Gemma-2-2B validation.** Subsequent code changes mean a **separately scoped current-tree revalidation** may be needed.
- **B3 checkpoint identity:** Agent 1 qualified independent PASS over delegated 65-file, Hub-source-bound, read-only Volume verifier; owner-approved exact canonical manifest at `results/q1_implementation_prep/CHECKPOINT_PROVENANCE_MANIFEST.json` and owner pin `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55`. Production/candidate manifest Git blob `7887be2f8d584dd603b871a8151b1e4d9df947db`. Agent 1 accepted the B3-10 forged-caller-map remediation; separate non-authoritative provenance labeling hardening at `bec896a` reportedly passed **39/39** CPU tests on CPython 3.13.13. Do **not** repeat checkpoint-byte download/hashing absent a new grounded defect.
- **Frozen scientific preregistration**, H3 interpretive approval and F1 stream reference are unchanged.

## Stage A — NEXT ACTION: one consolidated, non-forward CPU preflight (no Modal/no spend)

**Owner authorization boundary:** CPU-only metadata, static code checks, deterministic mock/synthetic non-forward tests and lock procedure preparation are in scope. No pretrained loads, model forwards of any size, or GPU. Avoid rerunning completed B3 byte audit.

1. Fetch `q1-impl-audit` at the stated baseline, record **exact commit and tree**. Check the intended pinned environment is available:
   - **CPython 3.11.15**, `torch==2.5.1`, `transformers==4.51.3`, `numpy==2.1.3`, `peft==0.19.1`;
   - direct pins file `TRACES_Q1_execution_direct_pins_v1_2.txt`, SHA-256 `4a90dd7133719a751400c456db615e726a245d9a34073bfb64243bf2a212048`.
   - A local **3.13.13** passing suite is useful supplementary evidence but **not a substitute** for 3.11.15 readiness.
2. Re-run **one** consolidated permitted CPU test set with `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD` and `TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION` unset, `HF_HUB_OFFLINE=1`, and no CUDA. Include the already-modified checkpoint trust tests (`test_q1_b1_remediation_d5a_d5b.py`, `test_q1_b1_remediation_d5_integration.py`, `test_q1_b3_10_owner_pin_freeze.py`), `test_q1_b3_runtime_provenance.py`, `test_q1_cpu_infrastructure.py`, B1 production pipeline, H3 applicable-direction and H3 independent follow-up. Add targeted safety tests only if a concrete integration need is uncovered. Capture counts/exit code and a **single** report; do not create another general audit cycle. Tests that temporarily monkeypatch authorization under mocks must not load actual model weights.
3. Run the existing **CPU-only runtime lock preparation**, e.g. `python scripts/q1_prepare_runtime_lock.py --out-dir <new local evidence directory> --pip-python <exact-3.11.15-interpreter>` after verifying the interpreter and checking that the script does not download anything. This produces preparation metadata and pip freeze only; it MUST state `production_image_lock_exists=false`. Avoid overwriting previously committed evidence or mistaking a host `pip freeze` for the eventual built-image lock. Verify the frozen direct pins SHA and the workload enumerator counts; do not silently change frozen scientific conditions.
4. Confirm from code and self-contained tests that non-authoritative `bind_unit_checkpoint_provenance` reports no production authority and a forged caller map is rejected when `require_authoritative_trust=True`. Do **not** generate new synthetic production pins.
5. Publish **one consolidated result**: passed/failed counts, exact interpreter/dependency versions, source commit/tree, direct pins/hash, manifest canonical digest, F1 unchanged, runtime preparation outputs, the actual remaining failures and the next needed authorization boundary.

**STOP** on missing Python/dependency pins, failed security gate, unexpected manifest mismatch, or tests that require scientific/model forwards. Do not reinterpret a production lock as complete.

## Stage B — Production-image runtime lock (REQUIRES a separately bounded owner decision before paid build)

**Current truth:** `RUNTIME_LOCK_STATUS.md` says **NO production built-image lock**; `runtime_lock_synthetic_fixture.json` has `production_image_lock=false`. The current code can generate/validate a wheel-hashed lock, but the real lock must be obtained **from the actual intended scientific image** with exact CPython 3.11.15, installed transitive wheel hashes, CUDA/toolkit, intended base image tag+digest, direct pins and execution hardware identity.

After Stage A, Cursor supplies a **short concrete build-only authorization request in chat**, with expected resource type, duration, spend cap, installed distribution/wheel provenance procedure, artifacts and fail-closed timeout. This is not permission to run Gemma; do not launch paid Modal builds based on this runbook alone.

Acceptance: independently inspect immutable built-image identity and records; require `claim_production_lock_exists(...)` to pass under the actual wheel rehash and matching image metadata. No synthetic or merely self-declared lock counts.

## Stage C — First real-Gemma exploratory calibration (SEPARATE owner scientific/GPU authorization)

Only after Stage B and a scoped owner-approved launch: the **minimum frozen-protocol-preserving** real-model exploratory checks, not confirmatory results. Verify load-time source-bound bytes, tokenizer class/single-BOS, BF16+eager and adapter dtype, F4 numerical tolerances, real-model intervention identities, B2 double-BOS descriptive checks where prescribed, and actual Q1 eager/hook throughput. Fail closed on any gate failure; do not substitute Stage 0 timing for measured Q1 timing.

The older CPU-only 2026-11-14 planning report projected the *full* Recommended-tier GPU workload at approximately **$18 optimistic / $36 mid / $43 high, GPU-only**, based on assumptions, **not measurements** of the final optimized Q1 run. Require owner budget approval using the measured calibration before the main experiment.

## Stage D — Main preregistered experiment (NOT authorized)

Launch only after real-model exploratory gates, F4/B2 constraints, production lock, approved budget/stop rules and a **new explicit scientific execution authorization** in chat. No merge to frozen `main` or changed science without separate owner decision.

## Minimal handoff contract

Cursor: **perform Stage A only now**; publish one consolidated technical result, no approval form or broad independent re-audit. Agent 1: review **specific new substantive defects only**; prior B1/H3/B3 results stand. When Stage A passes, return a tightly scoped Stage B build-only cost/plan packet to owner in chat. **No background or asynchronous promises** and no scientific execution assumed.
