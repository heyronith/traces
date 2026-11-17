# TRACES Q1 — B3 Checkpoint-Provenance Readiness Checklist

> **Current status update (2026-11-16):** The research owner approved B3-10 in chat, and the exact audited production manifest and SHA pin were frozen on `q1-impl-audit`. See [B3-10 owner pin-freeze status](B3_10_OWNER_MANIFEST_PIN_FREEZE_AND_REGRESSION_HANDOFF.md). B3-11/B3-12 and scientific execution remain **BLOCKED**. The original checklist below is preserved as **pre-freeze historical readiness documentation**, not an assertion that the pin is still `None`.

**Status:** PREPARED / BLOCKED pending verified real bytes and owner-controlled pin  
**Scientific authorization:** OFF. This document creates no permission to download, load, or run models.  
**Repository:** `heyronith/traces`, branch `q1-impl-audit` (documentation only; do not merge to frozen `main`).  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**B1 independent PASS:** code `3730e892ab3f4f839570b8b4d8710a63f86efc52`, tree `6a133d8e3a52fe7bd3c652ec1770ac419665c6d1`; Agent 1 verdict `results/q1_implementation_prep/B1_D5_A1_B1_A2_AGENT1_REAUDIT_VERDICT.md`. This is **engineering acceptance only**.

## Current hard stop

- `src/traces/q1/checkpoint_trust.py:PINNED_CHECKPOINT_MANIFEST_SHA256` is **None**; production checkpoint authentication is intentionally blocked.
- No owner-approved, independently verified production manifest exists at `results/q1_implementation_prep/CHECKPOINT_PROVENANCE_MANIFEST.json`; do **not** create placeholder data at that production path.
- No actual checkpoint bytes, real tokenizer files, production image, or empirically executed validation are authenticated by the accepted B1 synthetic suite.
- Do not set a pin, accept a caller-provided hash map as independent authority, or use an engineering-test manifest to unlock production.

## Sequential B3 gates (unchecked means NOT COMPLETE)

- [ ] **B3-01 — Authorization for byte acquisition/inspection.** Research owner separately approves a precisely bounded read/download/materialization operation, its source revisions, storage/budget and permitted environment. Until then, metadata review only; no checkpoint downloads or model loads.
- [ ] **B3-02 — Source identity.** Record the pinned base `google/gemma-2-2b` revision `c5ebcd40d208330abc697524c919956e692655cf` and pinned dense-LoRA revision `6303a74fc832a34d584999364817268aefd9623b`. Verify identities from authoritative sources rather than path names or caller echoes.
- [ ] **B3-03 — Exact unit coverage.** Independently inventory `l19/seed42–46` and `l1523/seed42–46` (ten distinct adapter units). The descriptive adapter-off baseline intentionally reuses `l19/seed42` with adapters disabled; it is **not** an eleventh distinct adapter.
- [ ] **B3-04 — Materialized local snapshots.** Under separate authorization, place real, immutable files in the approved base snapshot and ten adapter subfolders; use independently verified materialized copies **without symlinks** (ordinary Hugging Face cache snapshots may contain symlinks). Store checkpoints outside or inside the repository as needed; the **trusted `repo_root`**, not snapshot location, anchors the production manifest.
- [ ] **B3-05 — Base sharded index and inventory.** Authenticate `config.json` and `model.safetensors.index.json`; parse the index's `weight_map`; verify every referenced safetensors shard and its computed SHA-256. Reject missing, extra, altered, unlisted, duplicate-path, traversal or symlinked load-relevant files. Do **not** assume a monolithic `model.safetensors`.
- [ ] **B3-06 — Ten per-unit adapter inventories.** Authenticate each unit's `adapter_model.safetensors` and `adapter_config.json` (including LoRA scaling), plus every load-relevant file. Refuse root fallback, wrong subfolders and extra files. Require pairwise distinct **adapter-weight** digests across the ten identities; shared base and tokenizer files are allowed.
- [ ] **B3-07 — Tokenizer/template pins.** Verify the actual files used by the loader against frozen v1.3 §11: `tokenizer.model`, `tokenizer_config.json`, `chat_template.jinja`, and `special_tokens_map.json`, using `FROZEN_SECTION11_TOKENIZER_SHA256` in `checkpoint_provenance.py`. Enforce `GemmaTokenizer`, `use_fast=False` / `is_fast=False`. Refuse planted `tokenizer.json` and other competing tokenizer sources. Check coverage of exploratory subfolders separately; do not assume their equality from the seven-confirmatory-file audit.
- [ ] **B3-08 — Construct exact authoritative manifest.** Create a production-format manifest with `artifact_type=q1_checkpoint_provenance_manifest_v1`, `base_files` and `units` covering all ten family/seed/subfolder identities, containing only verified **computed** snapshot-relative digests (role-tagged `base:` / `adapter:`). Validate using `validate_checkpoint_manifest_schema`; include every load-relevant file. No fabricated hashes, unit aliases or placeholders.
- [ ] **B3-09 — Independent byte/identity audit.** A reviewer independent of the manifest author recomputes file hashes, validates shard index and exact file inventories, checks all ten identities, tests cross-seed substitutions/symlinks/unauthorized files, and compares the source revisions and §11 pins. Record a clear PASS/REVISE/BLOCK verdict and immutable code/tree/manifest evidence.
- [ ] **B3-10 — Owner-controlled manifest freeze.** Only after independent PASS, the research owner separately approves the exact canonical manifest bytes and SHA-256. Follow `canonical_checkpoint_manifest_bytes` (excludes signature fields) and the fixed production manifest path. Record approver, timestamp, audited SHA and source revision. The **owner**, not Cursor/engineering, authorizes setting `PINNED_CHECKPOINT_MANIFEST_SHA256` in a separately reviewed change.
- [ ] **B3-11 — Production gate verification.** In a separately authorized environment, verify production authentication uses the executor's trusted `repo_root`, exactly matches the owner pin, yields computed digests only, verifies the correct unit before model callbacks, rejects engineering-only contexts, and blocks under missing/wrong pins. Log accepted loaded identities only after authorized model-level validation.
- [ ] **B3-12 — Handoff to execution-readiness review.** Require F5/H3 decisions, production built-image lock, B2 empirical gate evidence, tiny-Gemma revalidation authorization and separate scientific execution authorization. B3 acceptance alone never grants model-inference permission.

## Evidence record required for each gate

Record **owner/reviewer**, date/time, exact base/adapter revisions, immutable artifact path and SHA, environment, independent verification result, and linked audit note. Record unavailable information as **BLOCKED**, never as a placeholder PASS.

**Relevant implementations:** `src/traces/q1/checkpoint_trust.py`, `checkpoint_provenance.py`, `loader.py`, `scientific_execution.py`; `results/q1_implementation_prep/B1_D5_A1_B1_A2_AGENT1_REAUDIT_VERDICT.md`.

**Current verdict:** **B3 NOT READY / scientific execution NOT READY**. The B1 engineering PASS and 333 passed / 12 skipped synthetic tests do not authenticate real checkpoint bytes.
