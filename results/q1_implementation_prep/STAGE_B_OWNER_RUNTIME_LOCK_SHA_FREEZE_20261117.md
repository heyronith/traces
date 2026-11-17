# Q1 Stage B — Owner-approved runtime lock SHA freeze

**Decision:** Approved by the research owner in chat on 2026-11-17, replying **"yes"** to the precise request to freeze `PINNED_RUNTIME_LOCK_SHA256` to the SHA below.

**Approved production lock (raw file SHA-256):**
`f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72`

**Locked artifact path:** `results/q1_implementation_prep/TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json`

**Image identified by pinned lock:** `im-0j3j9DzkkZSLdRQTxl4scU`; CPython 3.11.15 / torch 2.5.1+cu121 / CUDA 12.1; 54 installed packages and 54 hashed wheel records.

**Audit/evidence:** Agent 1 confirmed the Stage B image reuse, PEP440/CUDA binding and repository 54/54 evidence, but reported **REVISE** for the missing owner lock/image identity binding. The correction at commit `214239955131f9bed2e623086deb3072e98779fe` added fail-closed raw lock pin and actual `MODAL_IMAGE_ID` checks; Cursor reported **31/31 PASS** under CPython 3.11.15 before this owner freeze.

**Owner-pinned source:** `src/traces/q1/runtime_lock.py` now sets `PINNED_RUNTIME_LOCK_SHA256` to the approved raw SHA. The corresponding regression asserts this exact constant and exact on-disk lock SHA.

**Independent commit-time GitHub verification:** decoded corrected lock contents were SHA-256 hashed from GitHub and matched the approved full raw SHA above; lock Git blob `01346655ce143bfb3470085799d1fd0933347762` unchanged. No lock reserialization.

**Permissions unchanged:** This approval covers **only** the lock SHA freeze and CPU-only post-freeze checks. It is **not** a new Modal spend approval, Stage C scientific/GPU or tiny-Gemma/model-forward authorization, checkpoint download permission, changed protocol, or merge authorization. Frozen `main` is `b1e8874f45311678e6fa241f504e7e09d4a7808c`.

## Closure requirements

1. Cursor runs **post-freeze CPU-only** focused runtime lock/image security, Stage B remediation, B3 provenance, and CPU infrastructure suites under pinned CPython 3.11.15, with science/model-forward flags OFF; reports exact tested commit/tree and test results.
2. Agent 1 confirms the exact frozen lock SHA and production scientific preflight's runtime image-ID check, with only a **targeted** review. No 54-wheel repeat hash audit.
3. After targeted confirmation, close Stage B; only then request a **separate owner authorization** for Stage C real-Gemma exploratory calibration and budget.
