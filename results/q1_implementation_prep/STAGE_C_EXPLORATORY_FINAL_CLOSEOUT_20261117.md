# Stage C first real-Gemma exploratory calibration — final closeout

**Final exploratory verdict: PASS / CLOSED** (2026-11-17, America/Chicago). This closes the specific reverse-joint runner `REVISE` in two sessions. **It does NOT constitute F4/B2 scientific gate closure or the main experiment.**

## Immutable evidence and scoped sessions

1. **First exploratory session** ran under authorized baseline commit `102015fee9906f97f696430ab7f4da4b0f524461`, tree `b6fe0ea749be6d82956d4dd9ee86aa692583bb00`; evidence published at `e3aa5b60f00ff96acbd71050fe5a9ef9cde38afd`. Correct real-model load and runtime gates. **148/172** successful arm timing rows; **24/172 failed** exclusively because the reverse joint runner used T recipient tokens/tag rather than C. The original report remains **REVISE** and was not overwritten.
2. **Reverse recipient CPU correction** at commit `519367d127f99739791176c194a361c5a03f9f89` (tree `9e688c8b2e81b1c50ae048e83dff46eb99ff2da5`) bound direction-specific tokens, attention mask, tag and full expected IDs. Cursor reported **39/39 CPU PASS**; full first-session entrypoints remain blocked.
3. **Standing owner approval** for non-main execution/spending recorded in commit `8cb9e21af3587c29752425cf3517a0fe07bc2765`; future cost permission requests are **not** required for scoped non-main work. Explicit costed approval is still required **before the main preregistered experiment**.
4. **Dedicated reverse-only session** under approval baseline `8cb9e21af3587c29752425cf3517a0fe07bc2765`, evidence at `e71072fa9f71814e88e0b8c185ceb7de8b67dad2`, tree `eeaa9a6e8a56c58e60b4ebb461a3608f1c1d9ca4`; Modal app ID `ap-2x529cRA447fwMaTORQNTK`. **24/24 PASS**, exactly `all_upstream_joint_reverse` and `eight_head_layer_joint_reverse` × development dolly IDs `9793, 1634, 5149, 14198` × 3 repeats; all 24 recipient tags **C**. This session had **46/46 CPU guards PASS** on CPython 3.11.15 before GPU. Old successful arms were **not replayed**.

Across the two distinct sessions, the original 148 successful rows plus 24 **corrected, newly measured** reverse-only rows establish completed successful evidence for **all 172 originally planned exploratory measurement slots**. Do not rewrite this as a single 172/172 GPU session or erase original failures.

## Identity, measurements, and metering

- Image `im-0j3j9DzkkZSLdRQTxl4scU`; live image-ID equality; owner-locked runtime file SHA-256 `f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72`; exact `torch==2.5.1+cu121` / CUDA 12.1; **54/54** live environment and wheel lock checks; pinned checkpoint manifest `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55`; Gemma-2-2B + `l1523/seed42` only.
- Reverse-only timing: upstream joint **12/12**, mean `0.119667` seconds; eight-head L15 joint **12/12**, mean `0.074673` seconds; peak VRAM `5,511,429,120` bytes.
- Original Stage C metered cost reported about **$0.055** under initial $3 working cap; separate reverse-only job metered **$0.01732147** under $1 / 15-minute operational limit. Total **approximately $0.0723** Stage C metered across both, subject to billing reconciliation. Workspace billed $0.00 after credits as reported at collection time.
- Read-only checkpoint Volume, no held-out allocation access, no new checkpoint downloads, no main scientific pipeline, no F4 tolerance/family claim, no protocol change or merge.

## Explicit remaining research gates

**F4:** Full prespecified exploratory model-family numerical tolerance calibration is **not complete** from a single Gemma + adapter identity. **B2:** Empirical identity and double-BOS descriptive scientific gating remain separately to be satisfied; no B2 completion is inferred. **Main/Stage D:** **NOT AUTHORIZED**.

Next actions are non-main preparatory engineering and narrowly scoped exploratory calibration under the standing owner authorization. Before any main confirmatory/held-out experiment, estimate **total resource/time/cost** based on measured Q1 throughput and request the **owner's explicit main-experiment approval**. Do not merge frozen `main` without separate change authority.

## Evidence paths

- `STAGE_C_FIRST_REAL_GEMMA_EXPLORATORY_REPORT.md` and `STAGE_C_FIRST_REAL_GEMMA_EXPLORATORY_EVIDENCE.json` (historical first-run REVISE)
- `STAGE_C_FIRST_REAL_GEMMA_FAILURES_REDACTED.json`
- `STAGE_C_REVERSE_ONLY_REPORT.md` and `STAGE_C_REVERSE_ONLY_EVIDENCE.json` (reverse-only PASS)
- `STAGE_C_REVERSE_ONLY_BILLING_SNAPSHOT.json` (metering)
- `PROJECT_STANDING_NON_MAIN_EXECUTION_AND_SPEND_AUTHORIZATION_20261117.md` (standing approval)
