# TRACES Q1 — B3-01 Owner Authorization Request (Byte Acquisition / Inspection)

**Document type:** REQUEST ONLY — all approval checkboxes remain **UNSIGNED / NOT APPROVED**.  
**Does not grant:** model load, inference/forwards, GPU use, production pin setting, scientific execution, budget spend beyond an owner-filled hard limit, or merge to `main`.  
**Prerequisite packet:** `B3_PHASE0_SOURCE_METADATA_INVENTORY.md` (metadata-only; zero local weight hashes).  
**Scientific authorization:** remains OFF unless/until a later separate instrument says otherwise.  
**H3 engineering:** closed; not reopened by this request.

---

## A. Exact sources proposed

| Field | Value | Owner confirm |
|-------|-------|---------------|
| Base repository | `google/gemma-2-2b` | ☐ |
| Base immutable revision | `c5ebcd40d208330abc697524c919956e692655cf` | ☐ |
| Adapter repository | `interpretable-finetuning/topklora-gemma-2-2b-dense-lora` | ☐ |
| Adapter immutable revision | `6303a74fc832a34d584999364817268aefd9623b` | ☐ |
| Ten adapter units | `l19/seed42`…`46` and `l1523/seed42`…`46` (no substitution; adapter-off reuses `l19/seed42`) | ☐ |
| HF access / license | Base **gated=manual** (Gemma license + HF login); adapter **ungated**; license `gemma` both | ☐ |
| Credentials to use | ______________________________ (owner-named HF account / token scope) | ☐ |

**Files proposed for materialization (preferred minimal set):**

- Base: `config.json`, `model.safetensors.index.json`, and every shard named in the authenticated index after download (tree currently advertises `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, `model-00003-of-00003.safetensors` — confirm via index; do **not** assume monolithic `model.safetensors`).
- Per ten units: `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.model`, `tokenizer_config.json`, `chat_template.jinja`, `special_tokens_map.json`.
- **Exclude by default:** base `tokenizer.json`; adapter `sleeper_run_config.json` / `topk_config.json`; entire `all/seed*` tree — unless owner explicitly adds them.

Owner file-scope notes: _________________________________

---

## B. Permitted operation (choose exactly one)

| Option | Description | Owner selects |
|--------|-------------|---------------|
| **(i) Metadata only** | No further byte download (Phase 0 already done) | ☐ |
| **(ii) Download/inspect bytes without loading models** | Materialize base+adapter+tokenizer files above; compute independent SHA-256; **no** `from_pretrained` model load, **no** PEFT attach for inference, **no** forwards | ☐ |
| **(iii) Other (write exact bound)** | _________________________________ | ☐ |

**Explicit non-grants even if (ii) is approved:** inference; GPU; tiny-Gemma; B2; setting `PINNED_CHECKPOINT_MANIFEST_SHA256`; writing production `CHECKPOINT_PROVENANCE_MANIFEST.json` as authoritative without Agent 1 + owner freeze; scientific execution authorization.

---

## C. Execution environment

| Field | Owner fill / confirm |
|-------|----------------------|
| Named host (local or cloud) | ______________________________ ☐ |
| Network egress allowed | HF Hub only / other: ________ ☐ |
| Security boundary | ______________________________ ☐ |
| Storage path (no symlinks; not an HF cache symlink snapshot) | ______________________________ ☐ |
| Access credentials location | ______________________________ ☐ |
| Retention / deletion policy | ______________________________ ☐ |
| Confirm: symlinked snapshots will **not** be accepted as trust roots | ☐ |

---

## D. Transfer, disk, and spending limits

Grounded in Phase 0 **advertised** Hub sizes (not local hashes):

| Metric | Value from metadata |
|--------|---------------------|
| Minimum proposed transfer (base load-relevant + ten×six required) | **11 139 405 966 bytes (~10.37 GiB)** |
| Full Hub usedStorage envelope (base+adapter revisions) | **12 866 647 430 bytes (~11.98 GiB)** |
| Suggested peak free disk (≈1.5× minimum; owner may raise) | **~15.6 GiB** (recommend **≥20 GiB** free) |

| Field | Owner fill |
|-------|------------|
| Approved transfer ceiling (bytes or GiB) | ________ ☐ |
| Approved peak disk ceiling | ________ ☐ |
| **Hard spending limit (USD)** including network/storage | **$________** ☐ (do not invent) |
| Fail-closed stop if limit approached | delete partials / halt / notify: ________ ☐ |
| Recovery / wipe policy on abort | ________ ☐ |

---

## E. Required evidence outputs if (ii) is later approved

| Deliverable | Required |
|-------------|----------|
| Independent SHA-256 over every materialized load-relevant file | ☐ acknowledge |
| Parse authenticated `model.safetensors.index.json` `weight_map`; inventory every shard | ☐ |
| Bind all ten family/seed/subfolder identities; pairwise-distinct **adapter weight** digests | ☐ |
| Compare tokenizer/template files to `FROZEN_SECTION11_TOKENIZER_SHA256` using **local** hashes | ☐ |
| Refuse symlinks, path traversal, missing/extra load-relevant files, forbidden tokenizer competitors | ☐ |
| Draft manifest **candidate** only (not production pin); path TBD under prep, **not** auto-write production pin path as authoritative | ☐ |
| Independent **Agent 1 byte re-audit** before owner freezes canonical digest | ☐ |
| Owner-controlled freeze of exact manifest bytes + setting `PINNED_CHECKPOINT_MANIFEST_SHA256` | **separate later instrument** ☐ |

---

## F. Separation of authorizations (must remain distinct)

| Stage | This B3-01 form | Later separate approval |
|-------|-----------------|-------------------------|
| (a) Byte acquisition/inspection | **this request** | — |
| (b) Independent Agent 1 manifest/byte audit | no | ☐ |
| (c) Owner canonical manifest + production pin freeze | no | ☐ |
| (d) Runtime lock / tiny-Gemma / B2 / model-forward | no | ☐ |
| (e) Hard budget + scientific execution authorization | no | ☐ |

---

## G. Owner decision block (leave blank until decided)

- Decision date/time (UTC): ________  
- Approver name/role: ________  
- Selected scope: ☐ (i)  ☐ (ii)  ☐ (iii): ________  
- Hard USD limit: $________  
- Host / path / credentials: ________  
- Signature / recorded assent: ________  

**Until the above is completed in a subsequent owner action, treat B3-01 as NOT APPROVED.**

Conversational permission to “proceed with Phase 0 instructions” is **not** authorization to download checkpoint bytes or spend money.
