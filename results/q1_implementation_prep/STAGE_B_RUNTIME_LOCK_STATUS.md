# Stage B production runtime lock — CLOSED (2026-11-17)

**Final Stage B verdict: PASS / CLOSED**, on targeted independent Agent 1 freeze confirmation. **Scientific execution OFF.** No Stage C or D authorization granted.

## Authoritative evidence

- Corrected image lock: `TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json`; owner-pinned **raw SHA-256** `f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72`, verified byte-identical at pre-freeze and post-freeze commits.
- Scientific image ID: `im-0j3j9DzkkZSLdRQTxl4scU` (reused actual Modal image); base registry digest unchanged; CPython 3.11.15, `torch==2.5.1+cu121` / CUDA 12.1, NVIDIA A10 identity probe on prior Stage B run.
- Full inventory: **54/54 installed distributions to hashed wheels**; hashes rechecked by Cursor on Volume, not independently rehashed by Agent 1. `claim_production_lock_exists(require_rehash=True)` PASS on actual evidence.
- Post-freeze CPU tests: **31/31 PASS** under CPython 3.11.15, scientific and tiny-Gemma authorization flags unset. Exact baseline commit `b3474e82d85939a64b50a397898db8828bd0e254`, tree `4bc49c6f81672d06f92ee49166bdb6adfb1f5fc3`.
- Independent Agent 1: **PASS** for frozen lock SHA, altered lock/image rejection, actual `MODAL_IMAGE_ID` equality enforcement, and existing wheel/CUDA/freeze claim checks. Environment limitation: Agent 1 did not rerun full CPU-infrastructure torch tests or rehash wheel bytes.
- Stage B cumulative metered cost reported approximately **$0.01118**, versus authorized **$10** cumulative cap; workspace billed $0.00 after credits; any unsettled egress remains separately auditable.
- Frozen `main` `b1e8874f45311678e6fa241f504e7e09d4a7808c` unchanged. No merge, model execution, experimental measurements or new model/checkpoint downloads.

## Low-severity follow-up, *not* a Stage B blocker

Agent 1 noted that prior preflight `verify_installed_runtime_environment` checks public versions for 13 direct pinned distributions but does not compare actual runtime `torch.version.cuda` and all 54 locked installed distributions. The immutable image ID already binds to the actual Stage B image, so **Stage B stays closed**.

**Pre–Stage C engineering hardening (separate, not yet validated by Cursor):** `src/traces/q1/runtime_lock.py` adds live full `pip freeze` inventory comparison and exact `torch.__version__` / `torch.version.cuda` enforcement; `src/traces/q1/scientific_execution.py` passes the pinned lock document into that check. CPU-only negative tests in `tests/test_q1_pre_stage_c_running_inventory.py`. This untested post-close follow-up does not revoke Stage B PASS. Validate under CPython 3.11.15 before requesting Stage C scientific/GPU authorization.

## STOP

No Modal jobs, new GPU probes, Gemma/tiny forwards, model/checkpoint downloads, experiment, protocol changes or `main` merge were authorized by the Agent 1 review or this status update.

**Next owner decision:** separate budget- and scope-bounded Stage C real-Gemma exploratory calibration, only after pre–Stage C CPU safety checks PASS.
