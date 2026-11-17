# Stage B — owner-approved targeted lock remediation (2026-11-17)

**Owner decision in chat:** "Yes I dont care about the cost even it becomes 10 dollars" in response to the specific request to authorize the narrowly scoped Stage B runtime-lock correction.

**Interpretation / budget:** Authorized **cumulative Stage B spend ceiling is $10.00 USD** including all previous Stage B costs (previous report: approximately $0.008188 metered, some GPU charges pending settlement). This replaces the prior $5 ceiling for **Stage B alone**; it is not unlimited spending, not a transfer to Stage C/D and not permission for science.

**Baseline:** `q1-impl-audit` commit `9c2a6af74672904375dcf0ffc8ef83ebe7886175`; frozen `main` `b1e8874f45311678e6fa241f504e7e09d4a7808c`.

## Permitted work only

1. Correct `claim_production_lock_exists` and installed-runtime check to apply PEP 440 public-version semantics (e.g. frozen `torch==2.5.1` may match installed `2.5.1+cu121`), **without changing the pinned requirements file**. Preserve exact local version and wheel SHA; for intended CUDA 12.1 GPU image require the CUDA-specific torch wheel `+cu121` matching `torch_version` in independent image/runtime metadata; reject mismatched/CPU/wrong-CUDA variants.
2. Require *complete* transitive installed-distribution-to-wheel coverage for an actual production lock. Stage B built-image `pip freeze` has **54** distributions but only **41** hashed wheel files. The 13 missing evidence entries are `aiohappyeyeballs, aiohttp, aiosignal, attrs, cbor2, frozenlist, grpclib, h2, hpack, hyperframe, multidict, propcache, yarl`. Account for these with authenticated wheel artifacts or report an explicit blocker. Do not silently ignore Modal-managed/installation extras.
3. Add negative/positive tests for local-version policy, wrong public version, wrong CUDA local suffix/metadata, hash tamper and missing or extra wheel/installed-package coverage. Run local CPU-only non-model tests before another paid verification.
4. **Reuse** Modal scientific runtime image `im-0j3j9DzkkZSLdRQTxl4scU` and stored Stage B Volume wheel evidence where possible, without rebuilding or model downloads. Re-hash actual wheels, capture `pip freeze` again **from that same intended image**, bind installed metadata to all 54 wheel files, verify immutable image identity and `claim_production_lock_exists(..., require_rehash=True)`.
5. Publish a single redacted closeout report (actual hash coverage, SHA lock, interpreter, image digest, passing/failed checks, cumulative Stage B metered **and** billed costs, pending charges). If Stage B PASS, STOP for targeted Agent 1 independent review. No claim based solely on `validate_lock_document`.

## Authorization limits

- **$10.00 total Stage B metered cost maximum**, including build/probe already incurred; do not assume reported workspace billed $0.00 is actual cost. Check settled/pending GPU line items before any additional job and budget conservatively.
- Work must remain within the **previously approved maximum job parameters**: 4 CPU / 16 GiB / 45 minutes for CPU runs, optional one A10G identity-only probe / 10 minutes. No need to repeat GPU probe if the prior A10G evidence suffices.
- No additional checkpoint/model/tokenizer/adapter download, no Gemma model load, pretrained/tiny forwards, B2/F4 outcome collection, scientific GPU execution, or merge to `main`; science remains OFF.
- If $10 could be exceeded, evidence is unavailable, or independent provenance does not close, **STOP** and report BLOCK; never silently relax production claim gates. This is a bounded follow-up to the existing image build, not permission to restart broad audits.

**Next separate owner decision:** Stage C first real-model exploratory validation requires **new** scientific/GPU scope and cost authorization after Stage B passes. 
