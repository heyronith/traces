# Stage B Modal runtime-image build — BLOCK / INCOMPLETE

**Scientific authorization:** OFF. No model loads/forwards/checkpoint downloads/merge to `main`.

**Verdict:** **BLOCK / INCOMPLETE** — built image and wheel evidence captured, but `claim_production_lock_exists(...)` returned **false**.

## What succeeded

- Base image `python:3.11.15-slim-bookworm` digest `sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15`
- Modal image object `im-0j3j9DzkkZSLdRQTxl4scU`
- CPython **3.11.15** with pinned transformers/numpy/peft/torch CUDA build installed
- **41** wheels hashed; `validate_lock_document(..., require_production=True)` → **ok=true**
- A10G identity probe **ok** — device `NVIDIA A10`, driver via nvidia-smi present, cuda `12.1`; no transformers model imports
- Primary run: https://modal.com/apps/heyronith/main/ap-IEmFObFR4YAPyhqM4KZOhs

## Blocker (exact)

claim_production_lock_exists=False because wheel version for torch is '2.5.1+cu121' while direct pin requires exact string '2.5.1'. validate_lock_document ok=true (41 wheels rehashed). A10G identity probe succeeded (NVIDIA A10, driver 610.57.04, cuda 12.1).

`validate_lock_document` passed; only the production **claim** gate failed on exact pin-string equality for torch.

## Cost (under $5 Stage B cap)

| Meter | USD |
|---|---|
| Stage B apps metered (CPU+Memory+Egress so far) | **~$0.008188** |
| Workspace billed | **$0.00** |
| Cap | **$5.00** |

GPU line items may still settle; A10G probe completed inside the approved 10-minute window.

## Artifacts

- `STAGE_B_RUNTIME_IMAGE_LOCK_EVIDENCE.json`
- `TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE.json` (candidate; **not** production-claimed)
- `STAGE_B_WHEEL_INDEX.json`, `STAGE_B_PIP_FREEZE.txt`
- Redacted log `STAGE_B_MODAL_BUILD_RUN.log`
- Wheels on Volume `traces-q1-stage-b-runtime-lock` at `/mnt/stage_b/evidence/2026-11-17T152022Z`

## STOP

Do **not** treat this as a production runtime-lock PASS. Stage C remains unauthorized. Targeted review of the torch local-version claim rule is the next engineering decision; no silent pin relaxation was applied.
