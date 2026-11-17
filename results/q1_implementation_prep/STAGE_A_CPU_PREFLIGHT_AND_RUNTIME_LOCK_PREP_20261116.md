# Q1 Stage A — Consolidated CPU preflight (STOP before paid image build)

**Document type:** technical Stage A evidence + Stage B build-only plan for owner chat approval.  
**NOT** scientific-execution authorization. **NOT** a production runtime lock.  
**Branch / commit / tree:** `q1-impl-audit` / `396ef3dd192308d263e1c684216c3061bdb4deb3` / `835a323bce68c63eea986957953cc9a8ce41a1d4`  
**Frozen main (unchanged):** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Safeguards:** `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD` unset · `TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION` unset · `HF_HUB_OFFLINE=1` · no CUDA · no 65-file re-audit · no model forwards · no paid Modal image build in this stage.

## Environment

| Item | Value |
|---|---|
| Interpreter | `results/q1_final_cpu_checks/.venv_tf4513/bin/python` |
| CPython | **3.11.15** |
| torch / transformers / numpy / peft | **2.5.1 / 4.51.3 / 2.1.3 / 0.19.1** |
| CUDA available | **False** |
| Direct pins file | `TRACES_Q1_execution_direct_pins_v1_2.txt` |
| Direct pins SHA-256 | `4a90dd7133719a751400c456db615e726a245d9a34073b7fb64243bf2a212048` (match) |

## Checkpoint identity (CPU re-check only; no Volume rehash)

| Check | Result |
|---|---|
| Owner pin | `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55` |
| Canonical manifest SHA | **exact match** |
| Git blob | `7887be2f8d584dd603b871a8151b1e4d9df947db` |
| `authenticate_production_checkpoint_manifest` | `ok=true`, `production_unlocked=true` (identity only) |
| F1 stream SHA | `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed` (**unchanged**) |

## Consolidated CPU pytest (non-forward)

```bash
unset TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src
results/q1_final_cpu_checks/.venv_tf4513/bin/python -m pytest \
  tests/test_q1_b1_remediation_d5a_d5b.py \
  tests/test_q1_b1_remediation_d5_integration.py \
  tests/test_q1_b3_10_owner_pin_freeze.py \
  tests/test_q1_b3_runtime_provenance.py \
  tests/test_q1_cpu_infrastructure.py \
  tests/test_q1_b1_production_pipeline.py \
  tests/test_q1_h3_applicable_direction.py \
  tests/test_q1_h3_independent_followup.py \
  -q --tb=line
```

**Result: 103 passed / 0 failed · exit 0**

| File | Count |
|---|---|
| `test_q1_b1_remediation_d5a_d5b.py` | 22 |
| `test_q1_b1_remediation_d5_integration.py` | 17 |
| `test_q1_b3_10_owner_pin_freeze.py` | 10 |
| `test_q1_b3_runtime_provenance.py` | 7 |
| `test_q1_cpu_infrastructure.py` | 9 |
| `test_q1_b1_production_pipeline.py` | 10 |
| `test_q1_h3_applicable_direction.py` | 18 |
| `test_q1_h3_independent_followup.py` | 10 |
| **Total** | **103** |

Log: `STAGE_A_CPU_PREFLIGHT_PYTEST.txt`

Authoritative forged-map refusal and non-authoritative “no production unlock” labeling are covered inside the D5 suites above.

## Runtime-lock preparation (CPU only; not a production lock)

```bash
PYTHONPATH=src results/q1_final_cpu_checks/.venv_tf4513/bin/python \
  scripts/q1_prepare_runtime_lock.py \
  --out-dir results/q1_implementation_prep/stage_a_runtime_lock_prep_20261116 \
  --pip-python <exact-3.11.15-interpreter>
```

| Field | Value |
|---|---|
| `production_image_lock_exists` | **false** |
| Direct pins match | true |
| Workload scientific TF | 824400 (matches readiness v1) |
| Gen passes / loads | 3600 / 11 (match) |
| Host pip freeze | snapshot only (`pip_freeze_snapshot.txt`, 32 lines) — **not** a built-image lock |

Outputs under `stage_a_runtime_lock_prep_20261116/`.

## Remaining blockers (precise)

1. **Stage B — production built-image runtime lock:** still missing (`RUNTIME_LOCK_STATUS.md`). Needs paid Modal image build + wheel/CUDA/hardware capture under CPython 3.11.15 + direct pins. **Requires separate owner spend approval.**
2. **Stage C — first real-Gemma exploratory calibration:** unauthorized until Stage B + scoped scientific/GPU authorization.
3. **Stage D — main preregistered experiment:** unauthorized.
4. Scientific execution remains **OFF**. No merge to frozen `main`.

---

## Stage B — concrete build-only plan (for owner approval in chat; NOT launched)

**Purpose:** produce an immutable production runtime lock from the **actual intended scientific Modal image**. No Gemma load, no forwards, no checkpoint redownload, no science.

### Scope

1. Build Modal image: CPython **3.11.15** + `TRACES_Q1_execution_direct_pins_v1_2.txt` (SHA `4a90dd71…2048`) with CUDA-capable torch wheels matching intended A10/A10G scientific venue.
2. On that image (CPU primary; optional brief GPU attach only to record CUDA/driver/GPU identity):
   - `pip freeze`
   - hash every installed distribution/wheel (SHA-256)
   - record base image tag + digest, CUDA toolkit, driver, GPU type
   - write `runtime_lock` document with `source=built_image`, `production_image_lock=true`
3. Locally verify `claim_production_lock_exists(...)` against rehashed wheels + matching metadata.
4. Publish lock JSON + redacted build logs to `q1-impl-audit`. **STOP** before Stage C.

### Resources / timeout (fail-closed)

| Resource | Proposal |
|---|---|
| Primary | CPU **4 cores**, **16 GiB** RAM, timeout **45 min** (image build + freeze/hash) |
| Optional identity probe | **1× A10G**, timeout **10 min**, only if CUDA/driver/GPU fields cannot be captured on CPU image metadata alone |
| GPU for science | **forbidden** in Stage B |
| Checkpoint download | **forbidden** (reuse existing Volume if mount needed for path checks only; prefer no Volume IO) |
| Hard spend cap | **$5.00 USD** cumulative for Stage B build-only |
| Expected metered | **~$0.50–$2.00** (CPU image build dominant; +~$0.20 if short A10G probe) |
| Safety bound @ full timeouts | CPU ~$0.24 + mem ~$0.10 + optional A10G ~$0.18 ≈ **~$0.52** before egress; recommend **$5** cap for retries/egress margin |

Rates used: Modal CPU $0.0473/core-hr, memory $0.008/GiB-hr, A10G $1.10/hr (workspace `modal billing rates`, 2026-11-16).

### Explicit non-authorization

Stage B approval would authorize **image build + lock capture only**. It would **not** authorize pretrained Gemma loads, forwards, tiny-Gemma revalidation on GPU, main experiment, protocol edits, or merge to `main`.

---

## STOP

Stage A **PASS**. Await owner decision in chat on the Stage B **$5** build-only cap (or revised bound) before any paid Modal image work. Agent 1: review only new substantive defects; do not reopen completed B1/H3/B3 audits absent grounded issues.
