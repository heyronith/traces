# TRACES Q1 — B3 Phase 0 Source-Metadata Inventory (No Weight Bytes)

**Status:** METADATA DISCOVERY ONLY — **not** B3 provenance PASS.  
**Scientific authorization:** OFF. **Zero** real checkpoint/tokenizer/adapter weight files downloaded or hashed locally in this phase.  
**Branch tip at inspection start:** `6ff5e8a1507d0e9b67aa36a2dc7666122d59baad`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c` (unchanged)  
**Production pin:** `PINNED_CHECKPOINT_MANIFEST_SHA256 = None` (unchanged; not set)  
**Inspection UTC:** `2026-11-16T18:03:16Z`  
**Method:** Read-only public Hugging Face Hub HTTP API (`/revision/` and `/tree/` endpoints). No `hf_hub_download`, `snapshot_download`, `git lfs pull`, clone of weight trees, or model load.

**Hard distinction:** Source-advertised sizes, git blob OIDs, and LFS OIDs are **not** independently computed SHA-256 of materialized local files. **Zero** production file digests were authenticated in Phase 0. Matching an advertised LFS OID string to a frozen §11 pin string is **not** byte authentication.

---

## 1. Exact sources and immutable revisions

| Role | Repository | Exact revision (resolved `sha`) | Evidence URL | Access |
|------|------------|----------------------------------|--------------|--------|
| Base | `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` | https://huggingface.co/api/models/google/gemma-2-2b/revision/c5ebcd40d208330abc697524c919956e692655cf | **gated=`manual`** (Gemma license ack / HF login required for weight download); `private=false`; license `gemma`; metadata readable without credentials |
| Dense-LoRA adapters | `interpretable-finetuning/topklora-gemma-2-2b-dense-lora` | `6303a74fc832a34d584999364817268aefd9623b` | https://huggingface.co/api/models/interpretable-finetuning/topklora-gemma-2-2b-dense-lora/revision/6303a74fc832a34d584999364817268aefd9623b | **gated=`false`**; `private=false`; license `gemma`; metadata readable without credentials |

Tree (size) endpoints used:

- https://huggingface.co/api/models/google/gemma-2-2b/tree/c5ebcd40d208330abc697524c919956e692655cf
- https://huggingface.co/api/models/interpretable-finetuning/topklora-gemma-2-2b-dense-lora/tree/6303a74fc832a34d584999364817268aefd9623b?recursive=1

Local code pins (for comparison only; not re-derived from Hub):

- `src/traces/model_loader.py`: `BASE_MODEL_ID`, `DENSE_ADAPTER_REPO`, `PINNED_BASE_REVISION`, `PINNED_ADAPTER_REVISION`
- `src/traces/q1/pipeline_adapters.py`: `required_adapter_subfolders()` → ten identities
- `src/traces/q1/checkpoint_trust.py`: `REQUIRED_ADAPTER_LOAD_FILES`, `REQUIRED_BASE_META_FILES`, `FORBIDDEN_ADAPTER_LOAD_FILES`, pin `None`
- `src/traces/q1/checkpoint_provenance.py`: `FROZEN_SECTION11_TOKENIZER_SHA256` (targets only; **not** authenticated here)

Revision resolution: Hub API returned `sha` equal to the requested immutable revisions for both repos → **reliable for metadata purposes**.

---

## 2. Base repository file inventory (advertised)

Hub `usedStorage` (whole revision, includes non-load extras): **10 561 754 227** bytes (~9.836 GiB).

| Path | Advertised size (bytes) | Category | Load-relevant for B3? | Notes |
|------|-------------------------|----------|------------------------|-------|
| `config.json` | 818 | base config | **Yes** (required meta) | |
| `model.safetensors.index.json` | 24 224 | base index | **Yes** (required meta) | Index **bytes not downloaded/parsed**; shard list below is from public tree filenames only |
| `model-00001-of-00003.safetensors` | 4 992 576 136 | base shard (LFS) | **Yes if** referenced by authenticated index (UNKNOWN until B3-05) | Advertised only |
| `model-00002-of-00003.safetensors` | 4 983 443 424 | base shard (LFS) | same | Advertised only |
| `model-00003-of-00003.safetensors` | 481 381 384 | base shard (LFS) | same | Advertised only |
| `tokenizer.model` | 4 241 003 | tokenizer (LFS) | Q1 §11 pins authenticate **adapter-unit** tokenizer files, not this base copy | Present on base |
| `tokenizer_config.json` | 46 379 | tokenizer | same | |
| `special_tokens_map.json` | 636 | tokenizer | same | Size differs from adapter copies (888) — **do not assume** equality |
| `tokenizer.json` | 17 525 357 | **forbidden competitor** if used as load source | Must not be the authenticated tokenizer source (`FORBIDDEN_ADAPTER_LOAD_FILES`) | Present on base revision |
| `generation_config.json` | 168 | extra | No | |
| `README.md` / `.gitattributes` | — | docs/meta | No | |
| `chat_template.jinja` | — | — | **Absent on base** | Required on each adapter unit instead |

**Advertised base load-relevant sum** (config + index + three named shards, unauthenticated): **10 457 425 986** bytes (~9.739 GiB).  
**Monolithic `model.safetensors`:** not present (sharded layout confirmed at filename level).

---

## 3. Ten adapter identities — coverage matrix

Expected units (frozen): `l19/seed42`…`l19/seed46`, `l1523/seed42`…`l1523/seed46`.  
Required per unit: `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.model`, `tokenizer_config.json`, `chat_template.jinja`, `special_tokens_map.json`.  
Forbidden competitors: `tokenizer.json`, `tokenizer.model.json`.

| Unit | Required six | Forbidden competitors | Extra advertised files | Status |
|------|--------------|----------------------|------------------------|--------|
| `l19/seed42` | all present | none | `sleeper_run_config.json`, `topk_config.json` | METADATA OK / bytes NOT VERIFIED |
| `l19/seed43` | all present | none | same extras | METADATA OK / bytes NOT VERIFIED |
| `l19/seed44` | all present | none | same extras | METADATA OK / bytes NOT VERIFIED |
| `l19/seed45` | all present | none | same extras | METADATA OK / bytes NOT VERIFIED |
| `l19/seed46` | all present | none | same extras | METADATA OK / bytes NOT VERIFIED |
| `l1523/seed42` | all present | none | same extras | METADATA OK / bytes NOT VERIFIED |
| `l1523/seed43` | all present | none | same extras | METADATA OK / bytes NOT VERIFIED |
| `l1523/seed44` | all present | none | same extras | METADATA OK / bytes NOT VERIFIED |
| `l1523/seed45` | all present | none | same extras | METADATA OK / bytes NOT VERIFIED |
| `l1523/seed46` | all present | none | same extras | METADATA OK / bytes NOT VERIFIED |

**Adapter-off baseline:** reuses `l19/seed42` with adapters disabled — **not** an eleventh adapter (no extra folder required).

**Non-Q1 siblings also advertised:** `all/seed42`…`all/seed46` (each with the same eight-file pattern). Not part of the ten scientific identities; do not substitute.

### Per-unit required-file advertised sizes

| Unit | `adapter_model.safetensors` | `adapter_config.json` | `tokenizer.model` | `tokenizer_config.json` | `chat_template.jinja` | `special_tokens_map.json` | Unit sum |
|------|----------------------------:|----------------------:|------------------:|------------------------:|----------------------:|--------------------------:|---------:|
| l19/seed42 | 12 781 408 | 1 219 | 4 241 003 | 46 409 | 591 | 888 | 17 071 518 |
| l19/seed43 | 12 781 408 | 1 219 | 4 241 003 | 46 409 | 591 | 888 | 17 071 518 |
| l19/seed44 | 12 781 408 | 1 219 | 4 241 003 | 46 409 | 591 | 888 | 17 071 518 |
| l19/seed45 | 12 781 408 | 1 219 | 4 241 003 | 46 409 | 591 | 888 | 17 071 518 |
| l19/seed46 | 12 781 408 | 1 219 | 4 241 003 | 46 409 | 591 | 888 | 17 071 518 |
| l1523/seed42 | 115 032 552 | 3 035 | 4 241 003 | 46 409 | 591 | 888 | 119 324 478 |
| l1523/seed43 | 115 032 552 | 3 035 | 4 241 003 | 46 409 | 591 | 888 | 119 324 478 |
| l1523/seed44 | 115 032 552 | 3 035 | 4 241 003 | 46 409 | 591 | 888 | 119 324 478 |
| l1523/seed45 | 115 032 552 | 3 035 | 4 241 003 | 46 409 | 591 | 888 | 119 324 478 |
| l1523/seed46 | 115 032 552 | 3 035 | 4 241 003 | 46 409 | 591 | 888 | 119 324 478 |
| **Ten-unit required total** | | | | | | | **681 979 980** (~0.635 GiB) |

### Advertised LFS OID uniqueness (adapter weights only — not local hashes)

Hub advertised **10 distinct** LFS OIDs for the ten `adapter_model.safetensors` paths (good metadata signal for pairwise distinction; **still not** independent SHA-256 authentication):

| Unit | Advertised LFS OID (source metadata only) |
|------|-------------------------------------------|
| l19/seed42 | `8a001e786e4c9a2c0daf2013ddd3943be74579b22cd4cb5b85de9b08c65e3bbc` |
| l19/seed43 | `716c091c9e7a13a573e4efdee66a6356cea53027301d7d632f772c18c1daaf6d` |
| l19/seed44 | `349fe443d9c94a12d3c20998d42bbaad0e48a0696a0e676d63057fbbe06fa194` |
| l19/seed45 | `d0d1352f4bea5201bff01e577605d098f338866f74f21eff8d752dde7e03e06c` |
| l19/seed46 | `7c21c0327c1020405387e913fbfaadf2bdd83726924953ea1127cf0f17308b79` |
| l1523/seed42 | `2164abd809c527e9e4e313b8a09e3145123fa396799c8f322e7aa579f5f7d2ca` |
| l1523/seed43 | `f95a1adcf06b59c567ff8c98107d26bb71b4561e69df2e06824866d7f97aa195` |
| l1523/seed44 | `a6e6ef1b02132d56155cbe0d41d6e0da593eecd126ce7a28a883350bf86463a4` |
| l1523/seed45 | `d8c69f347279d01cdfdae85302b1fb93643cec014edc8f12451567e6fa1c2df1` |
| l1523/seed46 | `13bebdf9aa019446591d30129613ffa69603481e8203d2b81718b844a6dd6316` |

**§11 coincidence (not authentication):** advertised LFS OID for each unit’s `tokenizer.model` equals the frozen pin string `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2`. Recorded for owner awareness; **B3-07 still BLOCKED** until local independent hashing of materialized files.

Hub adapter revision `usedStorage`: **2 304 893 203** bytes (~2.147 GiB) — includes `all/seed*` and extras, not only the ten required inventories.

---

## 4. Storage estimate and placement proposal (not an approval)

| Bundle | Advertised bytes | ~GiB |
|--------|-----------------:|-----:|
| Base load-relevant (config+index+3 shards) | 10 457 425 986 | 9.739 |
| Ten units × six required files | 681 979 980 | 0.635 |
| **Proposed B3-01 minimum transfer (those paths)** | **11 139 405 966** | **10.374** |
| Full Hub `usedStorage` base+adapter (upper metadata envelope) | 12 866 647 430 | 11.983 |

**Peak disk proposal (for owner form; not selected):** materialize into a dedicated non-repo store with **no symlinks** (HF cache snapshots often symlink — trust layer rejects symlinks later). Budget ~1.5× minimum transfer for unpack/temp/HF cache duplication → **~15.6 GiB** peak, or **≥20 GiB** free headroom recommended. Optional inclusion of `all/seed*` or base `tokenizer.json` increases need; prefer **not** fetching forbidden/unused paths.

**Suggested location pattern (proposal only):** e.g. host path `$TRACES_Q1_CHECKPOINT_ROOT/{base_rev,adapters_rev}/…` outside the git worktree; trusted `repo_root` remains the repository that will later hold the owner-frozen manifest. Retention/deletion policy: owner-specified; default recommendation = retain until B3-10 freeze + Agent 1 byte audit complete, then owner decides.

---

## 5. Authentication accounting

| Item | Count / status |
|------|----------------|
| Independently computed production SHA-256 files this phase | **0** |
| Locally materialized weight/tokenizer/adapter bytes | **0** |
| Production manifest written | **No** (`CHECKPOINT_PROVENANCE_MANIFEST.json` not created) |
| `PINNED_CHECKPOINT_MANIFEST_SHA256` | **None** |

---

## 6. B3 gate status after Phase 0

| Gate | Status |
|------|--------|
| **B3-01** Byte acquisition authorization | **REQUESTED** — see `B3_01_BYTE_ACQUISITION_OWNER_AUTHORIZATION_REQUEST.md`; UNSIGNED |
| **B3-02** Source identity | **METADATA DISCOVERY / NOT VERIFIED** (Hub revisions resolve; bytes unauthenticated) |
| **B3-03** Exact unit coverage | **METADATA DISCOVERY / NOT VERIFIED** (ten paths present by name; digests unauthenticated) |
| **B3-04 … B3-11** | **BLOCKED** pending authorized bytes, independent hashes, Agent 1 byte audit, owner pin freeze |
| **B3-12** Execution readiness | **BLOCKED** (also needs runtime lock, tiny-Gemma, B2, budget, sci-exec auth) |

---

## 7. Risks / owner attention items

1. **Base is gated (`manual`)** — byte acquisition will need an approved HF identity that has acknowledged the Gemma license; metadata did not require credentials.
2. Base advertises **`tokenizer.json`** — must not become the load tokenizer source; adapter units correctly omit it.
3. Extra adapter files (`sleeper_run_config.json`, `topk_config.json`) and `all/seed*` trees exist — decide in B3-01 whether to fetch only required paths.
4. Index `weight_map` not parsed — shard set is filename-advertised only until authorized index authentication (B3-05).
5. H3 engineering is closed and is not reopened by this packet.
