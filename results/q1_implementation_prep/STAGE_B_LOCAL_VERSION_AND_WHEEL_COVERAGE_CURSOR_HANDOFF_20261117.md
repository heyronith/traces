# Stage B — Targeted re-verification of existing runtime image

**Status:** CODE + TESTS PREPARED / NOT EXECUTED; no production runtime-lock PASS claimed. **Science OFF**.
**Branch:** `q1-impl-audit`. Code baseline: `32de9aa12d88f61e53f2b9c07082fb6987f5dd0a`. Frozen `main`: `b1e8874f45311678e6fa241f504e7e09d4a7808c`, unchanged.
**Owner chat approval:** already recorded at `results/q1_implementation_prep/STAGE_B_OWNER_APPROVED_TARGETED_LOCK_REMEDIATION_20261117.md`. **Cumulative Stage B hard cap: $10.00**, including approximately $0.008188 previous metered spend and any unsettled charges. This approval does not authorize Stage C or any forward.

## Existing evidence, reuse not rebuild

- Stage B Modal run: https://modal.com/apps/heyronith/main/ap-IEmFObFR4YAPyhqM4KZOhs
- Intended image object: `im-0j3j9DzkkZSLdRQTxl4scU`; base `python:3.11.15-slim-bookworm` digest `sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15`.
- Pinned image CPython 3.11.15, torch 2.5.1+cu121; A10G probe (NVIDIA A10, driver 610.57.04, CUDA 12.1) already passed.
- Volume `traces-q1-stage-b-runtime-lock`, evidence directory `/mnt/stage_b/evidence/2026-11-17T152022Z`.
- Existing candidate lock: `results/q1_implementation_prep/TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE.json`; **do not promote it unchanged**, because its 41 wheels do not cover all **54** installed packages. Original lock and log are historical evidence; preserve unchanged.
- Missing hashed wheel artifacts among the 54: `aiohappyeyeballs, aiohttp, aiosignal, attrs, cbor2, frozenlist, grpclib, h2, hpack, hyperframe, multidict, propcache, yarl`.

## Prepared code + tests (run them before any new paid job)

- `src/traces/q1/runtime_lock.py`: PEP 440 public-version pin matching (so `torch==2.5.1` accepts `2.5.1+cu121`); exact installed torch `+cu121` bound to image + CUDA version, A10 probe, and actual rehashed wheel; mandatory installed-distribution/full `pip freeze`/wheel closure; no extra wheels accepted; strict production `validate_lock_document` and `claim_production_lock_exists`.
- `tests/test_q1_stage_b_runtime_lock_remediation.py`: six focused CPU-only tests including 41/54 original evidence fail-closed, +cu121 pass vs wrong version, wrong CUDA metadata, missing/extra wheel and frozen pins.
- Run on pinned CPython **3.11.15** with `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD` and `TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION` unset:
  `python -m pytest tests/test_q1_stage_b_runtime_lock_remediation.py tests/test_q1_b3_runtime_provenance.py tests/test_q1_cpu_infrastructure.py -q --tb=short`.
- Review code for unintentional regression. Do not weaken a fail-closed assertion merely to get PASS. If tests fail, publish actual failures and stop or correct only within the specific runtime-lock scope.

## Authorized target execution: existing-image verification only

1. Confirm Stage B **actual metered cumulative spend including GPU charges** + expected conservative new spend ≤ $10.0 before launching. Respect the original 4 CPU / 16GiB / 45-minute maximum for CPU work; no new GPU probe is needed.
2. Reuse the exact intended image and already-stored 41 wheels. Retrieve wheel artifacts for the 13 specific missing installed versions (image runtime pip freeze is the authority), no model/checkpoint/tokenizer downloads; count new wheel-transfer costs/egress inside $10 cumulative. If a wheel corresponding to an installed distribution cannot be obtained with genuine source provenance, or source/image identity cannot be bound, STOP **INCOMPLETE**.
3. From the exact image (not local macOS), capture `pip freeze` again, immutable image identity, installed torch local version and CUDA metadata; generate a **new** lock with **all 54** exact installed distributions, complete 54 rehashed wheels, `pip_freeze_text` and its SHA-256. Record provenance and actual filenames/hashes. Do not fabricate wheel hashes, resolve different versions or silently omit Modal/runtime dependencies.
4. Rehash real wheel bytes and run `validate_lock_document(..., require_production=True, wheel_dir=<actual evidence>)` and `claim_production_lock_exists(..., require_rehash=True, wheel_dir=<actual evidence>, pins_file=<frozen pins>)`. Both must return PASS. Report CPU-only unit/regression results and test wrong-CUDA/wrong-version/missing-wheel negative cases. Verify source image object `im-0j3j9DzkkZSLdRQTxl4scU` and base digest are unchanged; don't claim a newly built image if simply verifying existing one.
5. Publish one redacted Stage B closeout: exact repo commit and tree, image/lock/Volume evidence hashes, 54/54 mapping, full actual `pip freeze`, reasoned PASS/REVISE/BLOCK, precise costs (metered, billed, unsettled, cumulative), resource time. STOP for Agent 1's focused independent lock-provenance review if PASS.

## Hard stops

- No additional pretrained checkpoints, LoRA weights, or tokenizer acquisition; no tiny-Gemma or real-Gemma model load/forward, no experiment, F4/B2 scientific outcomes, protocol edits, or merge to `main`.
- **No new Stage C permission** is inferred from a successful build or lock.
- If tests fail, the installed/runtime inventory cannot be exactly accounted for, costs approach $10 total, or image identity changes, stop and report **BLOCK / INCOMPLETE** — never relax frozen pins or invent PASS.
