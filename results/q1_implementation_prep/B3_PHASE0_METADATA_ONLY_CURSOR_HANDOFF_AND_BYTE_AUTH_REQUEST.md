# TRACES Q1 — B3 Phase 0: Read-Only Metadata Preparation and Byte-Authorization Request

**Status:** B3 Phase 0 preparation ONLY — owner permitted proceeding with instructions after accepting Agent 1 H3 PASS. **B3-01 CHECKPOINT BYTE ACQUISITION/INSPECTION STILL REQUIRES SEPARATE PRECISE OWNER AUTHORIZATION.**  
**Branch:** `q1-impl-audit`. **Do not merge frozen main** `b1e8874f45311678e6fa241f504e7e09d4a7808c`.  
**H3 engineering:** Owner-accepted and CLOSED, as documented in `H3_OWNER_ACCEPTANCE_AND_ENGINEERING_CLOSURE_20261116.md` (commit `cad12d53f509590d750d0b7d1b52d0a92f5ad4e3`). Two nonblocking H3 hardening backlog notes do not reopen it.  
**Scientific authorization:** OFF. No model forwards/loads, checkpoint downloads, GPU, cloud spend, empirical outcomes, protocol changes or main merge.

## Purpose

Produce a fully sourced, read-only **source-metadata and feasibility packet** for the future B3-01 approval. Do **not** claim B3 provenance PASS without actual independently hashed bytes. This step may be completed now without materializing checkpoint files.

Review `results/q1_implementation_prep/B3_CHECKPOINT_PROVENANCE_READINESS_CHECKLIST.md`, `src/traces/q1/checkpoint_trust.py` and `src/traces/q1/checkpoint_provenance.py`. Do not alter their trust gates or the frozen scientific protocol.

## Phase-0 permissions and prohibitions

**Permitted now:** Static repository inspection; exact immutable revision verification by read-only public hosting metadata pages/API (e.g. repository metadata, file list, LFS pointer/size metadata, commit/revision records) with provenance URLs/timestamps; checking existing local **non-weight** code and documentation; CPU-only synthetic trust tests using synthetic fixture data; estimating storage and comparing file-name inventories **without fetching their contents**; drafting an approval request.

**Not permitted:** `hf_hub_download`, `snapshot_download`, `git lfs pull/fetch`, `git clone` or raw fetch of weight files, safetensors bytes, tokenizer/model weight bytes or real adapters; converting/using credentials to bypass access; pretrained/model loading; cache materialization or symlink population; GPU/cloud deployment or charges; inference; setting `PINNED_CHECKPOINT_MANIFEST_SHA256` from `None`; writing `results/q1_implementation_prep/CHECKPOINT_PROVENANCE_MANIFEST.json`; treating externally advertised LFS SHA as an independently computed SHA; making production-authentication claims.

If public metadata requires credentials/access, **mark BLOCKED** and ask for separately approved source access; do not fetch the bytes in order to make metadata verification work.

## Sources and exact fixed targets

- Pinned base repository: `google/gemma-2-2b`; revision `c5ebcd40d208330abc697524c919956e692655cf`.
- Pinned dense-LoRA adapter repository: `interpretable-finetuning/topklora-gemma-2-2b-dense-lora`; revision `6303a74fc832a34d584999364817268aefd9623b`.
- Ten adapter units: `l19/seed42`, `l19/seed43`, `l19/seed44`, `l19/seed45`, `l19/seed46`, `l1523/seed42`, `l1523/seed43`, `l1523/seed44`, `l1523/seed45`, `l1523/seed46`. Never substitute l1523 seed for l19 seed; adapter-off baseline reuses l19/seed42 without introducing an eleventh adapter.
- Source base is **sharded**: `config.json`, `model.safetensors.index.json`, and every shard referenced by `weight_map` in the authenticated index. Do not assume monolithic `model.safetensors`. At Phase 0 record the *advertised* index/shard filenames if visible, but do not parse index bytes unless separately approved.
- For every adapter unit the execution trust layer requires `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.model`, `tokenizer_config.json`, `chat_template.jinja`, `special_tokens_map.json`. `tokenizer.json` and `tokenizer.model.json` are forbidden competitors; record any advertised presence without altering files.
- Four frozen §11 tokenizer/template SHA-256 targets are in `FROZEN_SECTION11_TOKENIZER_SHA256` (not a substitute for actual hashing) and `GemmaTokenizer` with `is_fast=False` is required.
- Note whether file lists are complete, access is gated, revision resolution is reliable, or an exploratory unit has missing/extra file names; report UNKNOWN where metadata cannot establish the claim. Do not infer byte identity from names, sizes, git SHA, or LFS metadata.

## Output 1 — Auditable source-metadata inventory (no weight bytes)

Create `results/q1_implementation_prep/B3_PHASE0_SOURCE_METADATA_INVENTORY.md`, containing:

1. Exact source repository and immutable revision; evidence URL and inspection timestamp **for each**.
2. Source-reported file paths and sizes, categorized as base config/index/shard/tokenizer or ten per-unit adapters (including any extra load-relevant files), with accessible/source-only/unverified status.
3. Each expected unit coverage matrix, missing/extra/uncertain file names, read-only owner/auth/license/gating status, and exact expected-path comparison.
4. Estimated **total expected download bytes** and storage requirements from source metadata only (state unknown if sizes missing); suggested storage location/retention **as a proposal**, not a chosen approval. Include check that no symlinked snapshots would be accepted later.
5. Explicit distinction: public **advertised** content digests or LFS OIDs are not locally computed/authenticated file hashes; **zero** production SHA-256 files independently authenticated during Phase 0.
6. Clearly list B3-02/03 as METADATA DISCOVERY / NOT VERIFIED when appropriate; B3-04…B3-11 remain BLOCKED, owner pin stays None.

Do not fabricate API outputs, unobserved sizes, digests or dates. If network/repository access is unavailable, produce a **BLOCKED inventory** with exactly what is known from frozen local files and mark all unverified data UNKNOWN.

## Output 2 — B3-01 bounded owner-authorization form (request, not presumed grant)

Create `results/q1_implementation_prep/B3_01_BYTE_ACQUISITION_OWNER_AUTHORIZATION_REQUEST.md` with **specific** blanks for owner consent on:

- Source repositories and exact immutable revisions above; exact files/ten units proposed; intended Hugging Face access and any license/access restrictions.
- **Permitted operation** chosen independently: (i) metadata only, (ii) download/inspect actual base+adapter+tokenizer bytes **without loading models**, or (iii) another precisely bounded scope. No broader inference authorization follows from (ii).
- **Execution environment**: named local/cloud host, network egress, security boundary, storage path, no-symlink handling, access credentials and data-retention/deletion policy.
- **Projected transfer and peak disk bytes** grounded in sourced metadata; **hard spending limit in USD** (including network/storage), and fail-closed stop/recovery policy. Do not invent owner limit.
- **Authored evidence outputs**: independent SHA-256 on every materialized file; index-derived shard inventory; all ten identity bindings; exact computed manifest candidate (only after actual bytes); independent Agent 1 byte re-audit **before** owner freezing of a production digest.
- Separation of (a) byte acquisition/inspection approval, (b) independent manifest audit, (c) owner's exact canonical manifest/SHA freeze and setting production pin, (d) runtime/tiny-Gemma/B2/scientific model-forward authorizations, and (e) budget cap and scientific execution authorization.

All approval checkboxes remain **UNSIGNED / NOT APPROVED** until owner makes a subsequent precise decision after reviewing the packet.

## Output 3 — Completion and stop report

Commit only **metadata inventory, authorization request, and short Phase-0 status note** to `q1-impl-audit`. Report baseline SHA, final commit SHA and tree, precise URLs looked up and access status; state zero real-model files downloaded, zero forwards, zero GPU/cloud costs and no pin change. No production manifest, hash placeholder or scientific claim.

Then **STOP for owner selection of B3-01 scope**. Do not interpret the recent conversational “yes, proceed with the instructions” as open-ended authorization to acquire real checkpoint bytes or spend money. H3 engineering acceptance is closed and should not be reopened for Phase 0.

## Remaining gates after Phase 0

Real immutable checkpoint bytes and independently computed SHA-256 and ten adapter distinctions (B3); a production wheel/CUDA image lock; tiny-Gemma authorization, B2/F4/identities/double-BOS; hard budget and distinct scientific execution approval. **Q1 NOT READY FOR MODEL EXECUTION.**
