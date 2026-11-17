# Stage B — runtime-lock correction closeout (2026-11-17)

**Verdict: PASS**
**Science: OFF.** No model loads, inference, experiments, checkpoint downloads, or merge to `main`.

## Baseline

| Field | Value |
|---|---|
| Branch | `q1-impl-audit` |
| Repo commit | `7350cbfe890ab2fbba2d74c878257b0f9d345729` |
| Repo tree | `827fdfc391af52dacd17a73ac192d55f1cf3eac6` |
| Frozen `main` | `b1e8874f45311678e6fa241f504e7e09d4a7808c` (unchanged) |
| Owner cap | **$10.00 USD** cumulative Stage B (including prior spend) |

## CPU remediation tests (pinned CPython 3.11.15)

Command:

```text
python -m pytest tests/test_q1_stage_b_runtime_lock_remediation.py \
  tests/test_q1_b3_runtime_provenance.py tests/test_q1_cpu_infrastructure.py -q --tb=short
```

Result: **22 passed** (6 new Stage B remediation tests all green). Flags `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD` / `TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION` unset.

## Image reuse (no rebuild)

| Field | Value |
|---|---|
| Scientific image object | `im-0j3j9DzkkZSLdRQTxl4scU` |
| Observed `MODAL_IMAGE_ID` | `im-0j3j9DzkkZSLdRQTxl4scU` (exact match) |
| Base digest | `sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15` |
| Interpreter | CPython **3.11.15** |
| Torch / CUDA build | **`2.5.1+cu121`** / CUDA **12.1** |
| Correction Modal app | `ap-xUatp3hMguFkj6XN2UGE6R` |
| Volume evidence | `/mnt/stage_b/evidence/2026-11-17T154705Z` |

A10G identity probe **reused** from prior Stage B evidence (NVIDIA A10, driver 610.57.04, CUDA 12.1). No new GPU job.

## Wheel provenance: 54/54

- Prior candidate lock had **41** hashed wheels vs **54** `pip freeze` distributions → fail-closed (preserved as historical evidence).
- Correction obtained and rehashed the **13** missing wheels: `aiohappyeyeballs, aiohttp, aiosignal, attrs, cbor2, frozenlist, grpclib, h2, hpack, hyperframe, multidict, propcache, yarl` at exact freeze versions.
- Fresh in-image `pip freeze` **identical** to prior Stage B freeze.
- `pip_freeze_sha256` = `39abfea8b2b33c915dd7454d38d4bfaf1884650a4bc24f1761e02962d9d20e16`
- On-disk rehash of all **54** wheel files matched Volume `wheel_index.json`.

## Production claim gates

| Gate | Result |
|---|---|
| `_complete_installed_wheel_inventory` | **True** (54/54) |
| `_torch_cuda_build_matches_image` (`+cu121` ↔ CUDA 12.1 + A10G probe) | **True** |
| `validate_lock_document(..., require_production=True)` | **ok=True**, failures=`[]` |
| `claim_production_lock_exists(..., require_rehash=True)` | **True** |
| Original 41-wheel candidate still incomplete | **True** (fail-closed retained) |

Corrected lock path: `results/q1_implementation_prep/TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json`  
Lock SHA-256: `f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72`

## Costs (Stage B only)

| Item | USD |
|---|---|
| Prior build/probe apps (`ap-BrhD2KM0…`, `ap-IEmFObFR…`) | ~0.008887 |
| Correction capture (`ap-xUatp3hM…`) | ~0.002293 |
| **Cumulative Stage B metered** | **≈ $0.011180** |
| Hard cap | **$10.00** |
| Workspace billed (credits/allowances) | **$0.00** |

Under cap. Local `modal volume get` of ~2.8 GiB wheels may add unsettled egress; even large egress remains far below $10.

## STOP

Stage B **PASS**. Ready for targeted Agent 1 independent lock-provenance review.  
**No Stage C authorization inferred.** No merge to `main`.
