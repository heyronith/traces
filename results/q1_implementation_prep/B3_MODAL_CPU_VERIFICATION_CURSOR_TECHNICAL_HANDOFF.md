# TRACES Q1 — B3 Modal CPU Checkpoint-Byte Verification: Cursor Technical Handoff

**Document type: ENGINEERING INSTRUCTIONS, NOT A RESEARCH-OWNER AUTHORIZATION REQUEST.**  
**Status:** Ready to implement Modal verification tooling and access preflight. The owner confirmed in the 2026-11-16 conversation that (1) they have access to the gated Gemma model and (2) **Modal is the approved cloud environment** for this work. These confirmations **do not state a numerical hard spending limit**; any billable byte-acquisition job must satisfy the specific dollar cap agreed **in the chat**, not in a GitHub approval form.  
**Baseline GitHub tip:** `f2b872ccd43c7763b01245b9a99c3c1ac687260f` (`q1-impl-audit`).  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`. Do not merge, modify preregistration, or reopen H3 engineering.  
**Scientific authorization:** OFF. No model loads, model/tokenizer inference, tiny-Gemma forwards, GPU use, experiments, production pin changes, or inference budget. **Only B3 byte download, local cryptographic inspection on a CPU Modal worker, and audit preparation** are in scope when billable launch is separately cleared by the hard cap.

## A. Execution venue / permissions

Use the owner's existing **Modal** workspace, not their personal computer, for all real file downloads, CPU hashing and persistent storage.

- Use a **Modal Volume** named `traces-q1-b3-checkpoint-bytes` (or an existing clearly dedicated name after verifying no collision); mount at e.g. `/mnt/b3`; use unique immutable revision directories for base and adapter. No symlinked trusted inputs; if Hugging Face cache creates symlinks, copy the actual file bytes into a no-symlink materialized snapshot and verify those files.
- Use a **Modal Secret** (e.g. `traces-q1-hf-read`) containing `HF_TOKEN` of the Hugging Face account with accepted `google/gemma-2-2b` access. Require **read-only** scoped token, never print it or persist it in logs, GitHub, artifacts or model outputs. Creator/owner configures the secret within Modal, not in chat.
- Source egress only to Hugging Face Hub and any directly required Hub/CDN object-serving endpoints; no unrelated network destinations. Never download weight files to developer's computer.
- CPU only, finite `timeout`, bounded CPU/memory and concurrency=1 for one writer; no `gpu` argument; ensure exit on any auth, revision, count, size, integrity, quota or time error.
- Modal Volume writes need explicit `volume.commit()` after all files are closed, and an independent reader calls `reload()`; avoid concurrent writers on same paths. Persist proof of the materialized file inventory and SHA values after successful completion.

**Critical budget gate:** Current owner has **not set an exact USD ceiling in chat**. Cursor may implement scripts, create read-only/static tests, and inspect Modal/HF credential presence without downloading real checkpoint bytes. **Do not launch billable downloads, allocate paid Modal compute, or create a persistent paid footprint until a hard dollar ceiling is chosen in the conversation** and enforceable account/workload limits are configured. Modal's billing and workspace limits must be checked for the actual account; never infer free credits or a particular plan. If preflight itself incurs fees, defer it until cap confirmed. The chat owner will decide this dollar cap; **do not create a GitHub owner-authorization request or approval form**.

## B. Frozen sources and advertised inventory (metadata only)

Source: `results/q1_implementation_prep/B3_PHASE0_SOURCE_METADATA_INVENTORY.md` at `f2b872ccd43c7763b01245b9a99c3c1ac687260f`.

1. Base: `google/gemma-2-2b`, immutable revision `c5ebcd40d208330abc697524c919956e692655cf`, gated, owner confirms they have access. Expected `config.json`, `model.safetensors.index.json`, and **index-derived** weight shards. Hub filenames currently advertise `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, `model-00003-of-00003.safetensors`. Do not infer authorized shard list from names: parse the *materialized* authenticated index.
2. Dense-LoRA: `interpretable-finetuning/topklora-gemma-2-2b-dense-lora`, immutable revision `6303a74fc832a34d584999364817268aefd9623b`; **ten** adapter subfolders exactly: `l19/seed42–46` and `l1523/seed42–46`, no substitutions.
3. Six required load-relevant files **per adapter**: `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.model`, `tokenizer_config.json`, `chat_template.jinja`, `special_tokens_map.json`. Refuse `tokenizer.json`/`tokenizer.model.json` as competing load sources; do not accidentally copy extras `sleeper_run_config.json`, `topk_config.json` or unrelated `all/seed*` trees.
4. Advertised minimum required transfer: `11,139,405,966` bytes, ~**10.374 GiB**; target ≥20 GiB free peak-disk headroom while accounting for temporary caches. Advertised sizes and Hugging Face LFS OIDs are **not** independently computed local SHA-256.
5. Owner-controlled production manifest pin `src/traces/q1/checkpoint_trust.py:PINNED_CHECKPOINT_MANIFEST_SHA256` is `None` and must remain unset.

## C. Technical implementation (Cursor)

Create a **standalone CPU-only Modal B3 helper** in a narrowly scoped script under `scripts/` (choose `scripts/q1_b3_modal_byte_verify.py` if no conflict). Pin/version its runtime dependencies and document how to run it from the authenticated owner/engineering environment. Do not quietly alter the frozen project loader.

1. **No-download mode** (default): parse frozen expected revisions/units from existing code, resolve/check Modal Secret and Volume configuration without logging credential values, verify requested resource limits, require a positive **explicitly set** `B3_OWNER_APPROVED_MAX_USD` input reflecting an owner decision recorded in chat, verify source file-name inventory and fail closed before any real download. This mode must not create a paid task without the cost gate.
2. **Budget-bound download mode** (only after the cap is confirmed): perform **one pinned immutable-revision request** per file via `huggingface_hub` or equivalent. Ensure initial authoritative access to `model.safetensors.index.json`, parse and validate its `weight_map` and exact shard basenames, then fetch only required shards, metadata and adapter files. No `from_pretrained()`, no PEFT load, no generation, no GPU and no inference. Refuse a redirect/revision mismatch and any extra/unsafe load-relevant path.
3. Copy into dedicated immutable, no-symlink Volume directories. Avoid path traversal and symlinked snapshot aliases, check each file is regular, size consistent with expected data; read sequentially in chunks for SHA-256 (do not `read_bytes()` on multi-GiB shards); write an audit JSON with independently **computed** hashes, bytes, source paths, exact revision, timestamps, completeness and program version. Compare all four `FROZEN_SECTION11_TOKENIZER_SHA256` pins for each of ten units.
4. Verify ten adapter weight SHA-256 values are pairwise distinct; confirm every unit's `adapter_config.json`, expected family/seed bindings, and any additional load-relevant source-path constraints. If an expected file differs from frozen §11, **BLOCK** and record mismatch. On missing/extra load-relevant files, do not create an authoritative production manifest.
5. Build a **candidate-only** manifest using `validate_checkpoint_manifest_schema` and `canonical_checkpoint_manifest_bytes` with correct role-tagged `base:` and `adapter:` hashes; save under a clearly labeled `B3_CANDIDATE_*` evidence path, **not** at `CHECKPOINT_PROVENANCE_MANIFEST.json`. Do not set the production pin.
6. A separately authored independent CPU verifier (Agent 1), accessing the same committed Volume snapshot after reload, must recompute all materialized-file SHA-256 checks and the exact manifest candidate; record PASS/REVISE/BLOCK and discrepancies. Do not label the downloading worker's own checks independent.
7. Stop and request a *separate owner decision in chat* on the exact audited canonical manifest SHA and whether to modify `PINNED_CHECKPOINT_MANIFEST_SHA256` later. B3 completion does **not** authorize inference.

**On cost:** User has approved Modal as a venue but has not specified a hard cap; do **not** self-assign a dollar limit. Modal currently advertises per-second CPU/memory pricing, 1 TiB/month free Volume tier and included egress subject to plan/usage. See https://modal.com/pricing. Treat billed amounts as account-specific, not guaranteed free.

## D. Independent synthetic test and publication before spend

Before any billable workload, add CPU-only tests using **fabricated small synthetic bytes** for: exact five primary vs ten B3 units; shuffled identities; wrong shard index; missing/extra/forbidden files; path traversal and symlink refusal; incomplete SHA; wrong tokenizer file hashes; duplicate adapter digest; malformed secret absence; invalid/zero/missing spending cap; no-model-load/no-GPU guard; strict JSON with redacted URLs/tokens; candidate never promoted to owner pin.

Run permitted CPU synthetic suite if code changes touch shared runtime; otherwise isolated tests plus related B3 trust tests, with exact command, interpreter, exit code and F1 reference if in scope. Publish only verified metadata/technical records (never credentials, copyrighted weight bytes or owner-signoff requests), exact code commit/tree, and STOP at the appropriate gate.

## E. Owner communication contract

**Approval questions are handled privately in chat.** GitHub may contain engineering handoffs, test evidence and post-hoc audit records, **never a new document requesting the owner's signature or authorization**. Owner's Modal/HF-access confirmations from conversation are recorded as intent/context, not as an open-ended dollar authorization.

**Immediate next action:** Cursor implements and static-tests the CPU-only Modal preparation. The owner selects a hard B3 verification job budget **in chat** before downloading 10.374 GiB. If Modal Secret / workspace access is unavailable, report the single concrete blocker and do not improvise credentials.

**Scientific authorization remains OFF; frozen main unchanged.** B3 production pin, built-image runtime lock, F4/B2/tiny-Gemma, workload GPU budget and separate main-experiment approval all remain pending.

## Reference technical docs

- Modal persistent Volume docs: https://modal.com/docs/sdk/py/latest/Volume
- Modal Secret docs: https://modal.com/docs/sdk/py/latest/Secret
- Modal current pricing: https://modal.com/pricing
- Hugging Face gated model access: https://huggingface.co/docs/hub/models-gated
