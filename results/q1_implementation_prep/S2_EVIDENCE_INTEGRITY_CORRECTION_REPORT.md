# TRACES Q1 — S2 Evidence Integrity Correction

**Branch:** `q1-impl-audit`  
**Starting code commit:** `49c861aee2534edbc444aad6a70a6b34488ad976`  
**Code commit:** `a87342548e138a569e34b1bc302295dae39cb8b0`  
**Git tree SHA:** `197553b3dd0df0ee223dc006961cd53122422fd5`  
**Protocol:** v1.3  
**Scientific auth:** OFF (default)  
**F5:** pending · **H3:** awaiting owner decision  

This task is **not** authorization for scientific execution.

## Corrections

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Separate synthetic fixtures | Production auth rejects `claims_real_experiment=false`, fixture markers, and untrusted self-generated records |
| 2 | Immutable identity binding | Always bind to `REQUIRED_EXPLORATORY_SEEDS` (42–44) and `FROZEN_DEV_ID_SEQUENCE_SHA256`; caller overrides ignored |
| 3 | Substantive coverage | Per-model identities 1–9 + numerical comparisons; F4 formula/repeats/maxima/shapes; 64 distinct prescribed pairs per seed + double BOS + clean-ref provenance |
| 4 | Developmental gates from records | Eligibility / first-token derived only from exploratory units; external summaries cannot override |
| 5 | Trustworthy provenance | Requires independently hashed execution provenance manifest with exact `EMPIRICAL_STATUS=OWNER_APPROVED`; file hash alone ≠ authenticity → production BLOCKED |

Fabricated local fixtures remain available for adversarial/regression tests and **must not** clear live confirmatory progression.

## Residual limitations

- B1–B3 PARTIAL/BLOCKED (no approved empirical provenance manifests in repo)
- P1–P5 UNRESOLVED
- F5 pending; H3 awaiting owner
- No real experimental evidence generated or claimed

## Verification

See `SYNTHETIC_CPU_TEST_RECORD_S2_EVIDENCE_INTEGRITY.txt`.
