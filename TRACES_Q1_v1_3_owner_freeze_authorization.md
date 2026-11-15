# TRACES Q1 v1.3 — Research-owner scientific-freeze authorization record

**Record generated (UTC):** 2026-11-15T02:53:53Z  
**Repository:** `heyronith/traces`  
**Stage 1/reference and verified current main before freeze:** `2a82e0ef452b45340bd972d73ab1d113aa542eb0`  
**Approval authority:** Research owner (explicit authorization in this conversation)  
**Independent reviewer:** Agent 1; v1.3 targeted re-audit result: **PASS — ready for research-owner scientific freeze approval** (as reported by the research owner)  
**Authorized action:** Freeze exactly the audited Q1 scientific protocol v1.3 and the pinned direct-dependency specification, together with the exact, previously selected cohort files and corresponding manifests, via a timestamped repository commit. Do not revise the audited protocol bytes to change 'draft' markings; this record establishes post-audit authorization.

## Exact approved file identities

- `TRACES_Q1_preregistration_REVISED_REVIEW_DRAFT_v1_3.md`: SHA-256 `0a88d85bd4d94412f2568389dc16bdf51f7008cd01400e3e7ea0ab79de0d529b`; 78926 bytes.
- `TRACES_Q1_execution_direct_pins_v1_2.txt`: SHA-256 `4a90dd7133719a751400c456db615e726a245d9a34073b7fb64243bf2a212048`; 441 bytes.

## Selected data identities to verify from original audited source files

- Selection script `select_q1_dataset_v07.py` SHA-256 `a80ce67b89ac03377a73a6bf7ca07fdd45da318d98b97c93a57c27f01bd9cf10`.
- Ordered development ID-sequence SHA-256 `6ea712c08207858826037fd7c78067b6c3af852b89783fcc339fe3f17aaa1a43`.
- Ordered held-out ID-sequence SHA-256 `13c2c5a396ab58f9492e1161707a2acb8e4d242a058c9823470fa38bcca48b3a`.
- Development instruction JSON file SHA-256 `e61c883dec3dc4a48f364c035fcda609f282f4302f9ec651b59a6aab4fab847c`.
- Held-out instruction JSON file SHA-256 `c0314d2ee0b4b004623b27bfdfa58c3a1f5c37985e381f79099bf737c058ca9e`.
- Final cohort is exactly **50 development + 200 held-out** instructions, with frozen selected IDs and texts, without reselection or substitutions.

## Completion conditions

1. Confirm the above hashes against the actual files in the Cursor working tree, verify the source evidence and manifest are unchanged, and ensure any distribution of source-derived instruction text meets licensing and attribution requirements.
2. Create one timestamped commit that records the reviewed protocol bytes, direct pins, this authorization record, the original selected cohort files, and their manifest/source identities. If any required original source file is missing or mismatched, halt and report **NOT FROZEN**; do not reconstruct the data.
3. Record the completed commit SHA and UTC commit timestamp. Only then report **repository scientific freeze: COMPLETE**.
4. Preserve the no-inference and no-GPU gates. A separate research-owner approval is required for execution after implementation validation, the full hashed runtime lock, identities, workload count and budget cap.

**Current status at record creation:** **OWNER AUTHORIZED / REPOSITORY FREEZE PENDING.** This authorization record is not evidence of a repository commit or of outcome collection.
