# TRACES Q1 — Stage B owner-approved Modal build-only execution handoff

**Status:** OWNER APPROVED IN CHAT; **NOT EXECUTED** at creation of this record. This document records a direct owner instruction, not an independent GitHub-based approval, and does **not** authorize scientific execution.

**Branch:** `q1-impl-audit`; Stage A evidence baseline `4a0bbdedfbcc8e6950e0dd0409343f523a362091`.
**Frozen `main`:** `b1e8874f45311678e6fa241f504e7e09d4a7808c` — do not modify.
**Owner decision date:** 2026-11-16 (America/Chicago).

## Exact owner authorization (from chat)

> I explicitly approve Stage B production Modal image build and runtime-lock verification, with a cumulative $5 USD hard cap, up to 45 minutes at 4 CPU cores/16 GiB, and an optional A10G identity-only probe of at most 10 minutes. No model loads, forwards, checkpoint redownloads, scientific execution, or main merge. Stop at the cap and return evidence.

This is now **approved within these exact bounds**. Cursor is the delegated Modal executor; no further owner approval is needed merely to execute this scoped Stage B work. Do **not** treat it as approval for Stage C or D.

## Stage A acceptance basis (already complete)

At `4a0bbdedfbcc8e6950e0dd0409343f523a362091`: 103/103 CPU-only tests passed, exit 0, pinned CPython 3.11.15 + torch 2.5.1 / transformers 4.51.3 / numpy 2.1.3 / peft 0.19.1. Direct pins SHA-256 `4a90dd7133719a751400c456db615e726a245d9a34073bfb64243bf2a212048` verified; owner-frozen canonical checkpoint manifest SHA-256 `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55` unchanged. F1 stream SHA unchanged `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`. **No real production built-image runtime lock existed at Stage A.**

## Scope — build-only, no inference

1. Create or reuse the **actual intended Modal scientific runtime image** with **CPython 3.11.15 exact**, `TRACES_Q1_execution_direct_pins_v1_2.txt` (SHA above) and locked installed transitive dependencies; choose a CUDA-compatible, immutable base image and record its **tag and content digest**.
2. Run only tooling for image-build, installed-package `pip freeze`, wheel acquisition/hash binding to every installed distribution, torch/CUDA-runtime visibility, and lock capture. Wheel package downloads necessary to build/verify the runtime are within this scoped image-build approval; **checkpoint/model/tokenizer/adapter acquisition is forbidden**.
3. Capture installed package versions, exact Python executable, platform, wheel filenames and actual SHA-256s, wheel-to-installed-package correspondence, base image metadata, CUDA toolkit/runtime, framework CUDA build, and driver/hardware fields when observable.
4. Primary Modal run: **up to 4 CPU cores and 16 GiB RAM, 45 minutes maximum**. **Optional** single A10G GPU probe solely for device/driver/CUDA identification, **10 minutes maximum**; no model imports that trigger pretrained weight loads, no tensor/model forwards, no data collection. Omit the probe if not necessary or if costs are unsafe.
5. Create a built-image runtime lock, **but only claim `source=built_image`, `production_image_lock=true` after verifying that it came from the intended built image and the complete transitive wheel hash evidence is real**. Run `claim_production_lock_exists(..., require_rehash=True, wheel_dir=<real downloaded wheel evidence>, pins_file=<frozen pins>)`; require exact CPython 3.11.15 and direct pins, source/image tag/digest match and no missing/extra/unmatched wheels. If a package cannot be re-associated with actual wheel bytes or an immutable image identity, **record BLOCK/INCOMPLETE**, not a synthetic production PASS.
6. Publish full lock JSON (or an explicitly incomplete candidate), verification output, redacted build logs, wheel-index/hashes, image identity and run IDs, environment and resource evidence, elapsed time and cost ledger. Report all uncertainties and warnings; no secrets/tokens in logs or Git.
7. **STOP at Stage B.** No model loading, forwards, pretrained HF access, science, tiny-Gemma execution, B2/F4 calibration, confirmatory outputs, GPU experimentation, protocol modifications, or merge to `main`.

## Hard budget controls

- **Stage B total cap: $5.00 USD cumulative**, including the CPU image build, optional A10G identity probe, retries, egress/storage or billable resources attributable to Stage B. This cap is separate from the earlier B3 CPU audit approval; do not borrow its unspent authorization.
- Confirm current workspace cost/pricing and current Stage B accrued spend before launch; calculate a conservative ceiling for authorized 45-minute/10-minute jobs and any image-building/transfer overhead. Refuse a launch or optional probe if the conservative projected total plus cumulative spend would exceed $5.
- Limit resources, container concurrency (prefer one), timeouts and retries. Monitor spend/cost records between attempts. A workspace $0.00 bill after free credits does not mean actual metered cost is zero; report both metered and billed.
- Halt / do not retry once the $5 ceiling could be crossed. Software timeouts and cost estimation alone are not a guaranteed account-wide spending hard stop; configure provider account/workspace budget alarms or limits where available and do not claim a hard limit is platform-enforced unless verified.
- If bounded costs/resources, immutable wheel provenance or correct pinned runtime cannot be ensured, **STOP and report**, rather than silently relax pins or expand authorization.

## Completion criteria and next decision

**PASS only if** real intended image, full provenance/hash lock and independent revalidation satisfy the existing production runtime-lock contract. Otherwise return **REVISE/BLOCK** with exact missing evidence. Preserve owner pin/manifest and frozen protocol unmodified. Note any executor version discrepancies (a local CPython 3.13 test is not runtime-lock evidence).

**At completion:** publish one Stage B evidence status, exact branch commit/tree, image digest and verified lock digest, test result, cumulative cost, unresolved blockers. **STOP for targeted independent review of newly created lock evidence**; Stage C first real-Gemma exploratory run will require its own explicit owner scientific/GPU authorization and bounded budget. No new general B3 re-audit.
