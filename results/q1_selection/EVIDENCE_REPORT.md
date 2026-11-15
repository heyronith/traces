# TRACES Q1 — CPU Dataset Selection Evidence Report (v0.7 §2)

**Overall verdict: PASS**

**Reference commit:** `2a82e0ef452b45340bd972d73ab1d113aa542eb0`  
**Specification:** `TRACES_Q1_preregistration_SELECTION_READY_DRAFT_v0_7.md` §2  
**Execution:** CPU-only; no model weights loaded for inference; no GPU; no scientific outcomes; no repository commit.

## Stage table

| Stage | Counts / result | Status |
|---|---|---|
| 1. Reverify Alpaca empty-input length bounds (`numpy.percentile(..., method="inverted_cdf")`) | n=32,603; 5th/95th → inclusive **[6, 23]** (matches expected) | PASS |
| Parse Dolly JSONL (physical indices; no blank/skip) | 15,011 lines | PASS |
| Raw Dolly | 15,011 | PASS |
| Empty `context == ""` | 10,544 | PASS |
| Character / banned-word / letter-ratio / token-length screen | 8,096 retained (removed: non-ASCII 154; letter_ratio 24; banned 38; length 2,232; empty 0) | PASS |
| External exact/near screen vs full Alpaca + smoke/pilot | after=8,049 (exact Alpaca 15; near Alpaca 31; exact smoke/pilot 0; near smoke/pilot 1) | PASS |
| Within-Dolly connected-component removal (discard all nonsingletons) | before=8,049; components=7,614; discarded components=184 (619 records); **singletons=7,430** (≥250) | PASS |
| Deterministic SHA-256 allocation | development=50; held-out=200; unused singletons=7,180 | PASS |
| Full chat-template alignment (250 pairs) | 250/250 PASS; sole diff position **5**; IDs trigger **154657** / control **152798** | PASS |

## Allocation hashes

- Development ordered ID-sequence SHA-256: `6ea712c08207858826037fd7c78067b6c3af852b89783fcc339fe3f17aaa1a43`
- Held-out ordered ID-sequence SHA-256: `13c2c5a396ab58f9492e1161707a2acb8e4d242a058c9823470fa38bcca48b3a`
- First development Dolly IDs: `9793, 1634, 5149, 14198, 11344`
- First held-out Dolly IDs: `7077, 5799, 9254, 11744, 10500`

## Final category breakdown (250)

| Category | n |
|---|---|
| open_qa | 100 |
| brainstorming | 59 |
| general_qa | 46 |
| classification | 28 |
| creative_writing | 17 |

## Pins verified

- Tokenizer load: `AutoTokenizer.from_pretrained("interpretable-finetuning/topklora-gemma-2-2b-dense-lora", subfolder="l19/seed42", revision="6303a74fc832a34d584999364817268aefd9623b")`
- `type(tok).__name__` = `GemmaTokenizer`; `tok.is_fast` = `True`
- `tokenizer.model` SHA-256 = `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2`
- `chat_template.jinja` SHA-256 = `ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`
- Dolly `bdd27f4d…` file SHA-256 = `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`
- Alpaca `12567cab…` file SHA-256 = `bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b`

## Software

- Python 3.13.13
- numpy 2.5.4; transformers 5.23.0; tokenizers 0.23.3; sentencepiece 0.2.3
- `unicodedata.unidata_version` = 15.1.0
- Selection script SHA-256: `a80ce67b89ac03377a73a6bf7ca07fdd45da318d98b97c93a57c27f01bd9cf10`

## Artifact paths (all uncommitted)

| Artifact | Path |
|---|---|
| Executable selection script | `results/q1_selection/select_q1_dataset_v07.py` |
| Selection manifest | `results/q1_selection/selection_manifest.json` |
| Development instructions (50) | `results/q1_selection/development_instructions.json` |
| Held-out instructions (200) | `results/q1_selection/held_out_instructions.json` |
| Alignment results (250) | `results/q1_selection/alignment_results.json` |
| Discarded within-Dolly components | `results/q1_selection/discarded_within_dolly_components.json` |
| Stage counts | `results/q1_selection/stages.json` |
| Run log | `results/q1_selection/run_log.txt` |
| This evidence report | `results/q1_selection/EVIDENCE_REPORT.md` |

Artifact content SHA-256 (post-run):

| File | SHA-256 |
|---|---|
| `select_q1_dataset_v07.py` | `a80ce67b89ac03377a73a6bf7ca07fdd45da318d98b97c93a57c27f01bd9cf10` |
| `selection_manifest.json` | `01f5a81ce6d667920af5b0cb92e79e8416ecfdccf85a7396dddeb65095f71f33` |
| `development_instructions.json` | `e61c883dec3dc4a48f364c035fcda609f282f4302f9ec651b59a6aab4fab847c` |
| `held_out_instructions.json` | `c0314d2ee0b4b004623b27bfdfa58c3a1f5c37985e381f79099bf737c058ca9e` |
| `alignment_results.json` | `599e3961f58de82c59bf032d452437b919e595a0bf7daa4498822d2f5f44393b` |

## Confirmations

1. **Repository remains uncommitted** for this selection: `git status` shows `?? results/q1_selection/` (and the pre-existing untracked v0.7 draft). HEAD stays at `2a82e0e`. No commit was created.
2. **Scientific execution configuration remains disabled:** `configs/pilot.yaml` has `enabled: false`.
3. **No model inference / GPU / outcome collection** was performed.
4. Near-duplicate screening used exact character 5-gram Jaccard with a **lossless** inverted-shingle candidate prefilter (no posting-list truncation / MinHash / LSH).
5. Whole nonsingleton components discarded; no representative retention.
6. Alignment checked **after** allocation; no instruction substituted to force PASS.
7. Manifest note retained: full-corpus Alpaca screen; row-level training reconstruction not performed; ASCII-only character filter, not English verification.

## Notes for Agent 1 reproduction

- Reproduce with: `.venv/bin/python results/q1_selection/select_q1_dataset_v07.py` from repo root at `2a82e0e` (script is self-contained under `results/q1_selection/`).
- **Correction (D1):** External post-screen count **8,049** vs provisional feasibility **8,067** is **not** due to the banned-word rule. Both pipelines entered the external screen with **8,096** candidates. The gap is **+18 external exclusions** under v0.7, caused by the **R2 normalization change** (casefold + isalnum punctuation folding). The 18 Dolly IDs newly excluded are documented in `results/q1_selection_audit_closure/external_match_evidence_47.json`. Within-Dolly cluster discard further reduces to **7,430** singletons before hash allocation.
- Warnings list in the manifest is empty.
