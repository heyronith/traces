# TRACES Q1 — Scientific freeze record

**Freeze status at commit time:** COMPLETE (repository scientific freeze)  
**Authorization:** `TRACES_Q1_v1_3_owner_freeze_authorization.md`  
**Independent audit:** Agent 1 PASS on v1.3 (as recorded by research owner)  
**Parent / Stage 1 reference commit:** `2a82e0ef452b45340bd972d73ab1d113aa542eb0`  
**Branch:** `main`

## Verified approved protocol and pins (byte-identical)

| File | SHA-256 | Bytes |
|---|---|---|
| `TRACES_Q1_preregistration_REVISED_REVIEW_DRAFT_v1_3.md` | `0a88d85bd4d94412f2568389dc16bdf51f7008cd01400e3e7ea0ab79de0d529b` | 78926 |
| `TRACES_Q1_execution_direct_pins_v1_2.txt` | `4a90dd7133719a751400c456db615e726a245d9a34073b7fb64243bf2a212048` | 441 |

Historical review-draft wording in v1.3 was preserved byte-for-byte. Authorization is established by the separate owner authorization record, not by rewriting the protocol.

## Verified frozen cohort identities

| Artifact | SHA-256 / value |
|---|---|
| `results/q1_selection/select_q1_dataset_v07.py` | `a80ce67b89ac03377a73a6bf7ca07fdd45da318d98b97c93a57c27f01bd9cf10` |
| `results/q1_selection/development_instructions.json` | `e61c883dec3dc4a48f364c035fcda609f282f4302f9ec651b59a6aab4fab847c` |
| `results/q1_selection/held_out_instructions.json` | `c0314d2ee0b4b004623b27bfdfa58c3a1f5c37985e381f79099bf737c058ca9e` |
| `results/q1_selection/selection_manifest.json` | `01f5a81ce6d667920af5b0cb92e79e8416ecfdccf85a7396dddeb65095f71f33` |
| Development ordered ID-sequence hash | `6ea712c08207858826037fd7c78067b6c3af852b89783fcc339fe3f17aaa1a43` |
| Held-out ordered ID-sequence hash | `13c2c5a396ab58f9492e1161707a2acb8e4d242a058c9823470fa38bcca48b3a` |
| Cohort size | 50 development + 200 held-out |

No reselection, substitutions, model inference, GPU jobs, or scientific outcomes were performed for this freeze.

## Attribution

Dolly-derived instruction attribution: `results/q1_selection/DATA_ATTRIBUTION.md` (CC BY-SA 3.0).

## Execution gates (unchanged)

- No GPU authorization is granted by this freeze.
- `configs/pilot.yaml` remains `enabled: false`.
- Next stage requires separate research-owner GPU authorization after implementation validation, full hashed runtime lock, identities, workload enumeration, and budget cap.
