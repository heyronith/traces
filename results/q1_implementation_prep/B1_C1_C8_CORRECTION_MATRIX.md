# B1-C1…C8 Correction / Test Matrix

**Implementation commit:** `5d0fb1e535cb9354ffa05c2e8ebe72c897fae0da`  
**Tree SHA:** `902418203d182b61e39125b122bd361980171428`  
**Scientific authorization:** OFF  
**S2:** remains independently accepted (not reopened)

| ID | Defect | Correction | Regression |
|----|--------|------------|------------|
| **C1** | Temp `assemble_bundles` without pairs blocked cross-fit | `extract_signed_single_head_effects()` from intact + patched margins only; requires complete 152-head inventories both dirs | `test_c1_head_effect_extraction_without_pair_assembly` |
| **C2** | Contiguous mid-split folds | `plan_crossfit_folds()` → `assign_f2_folds()` even/odd on frozen allocation order | `test_c2_frozen_f2_even_odd_membership` |
| **C3** | Forward rankings reused for reverse | `compute_directional_crossfit_selections()` + directional `execute_crossfit_topk_conditions` | `test_c3_forward_reverse_selections_differ_and_directional`, `test_c3_joint_patch_calls_use_matching_direction` |
| **C4** | Fold indices vs eligible-only vectors | F2 on full allocation; eligibility filters in place; eval on opposite-fold eligible only; empty fold → indeterminate | `test_c4_eligibility_preserves_fold_membership_with_exclusions`, `test_c4_empty_eligible_fold_is_indeterminate_not_crash` |
| **C5** | `family=base` rejected / no adapter-off path | Base catalog OK; `assemble_adapter_off_baseline_bundles`; stage5/validate exemptions | `test_c5_adapter_off_baseline_path_no_intervention_catalog` |
| **C6** | Hard-coded `"i hate you"`; aggregate-only rates | `FROZEN_FULL_PAYLOAD`; retain `GenerationRecord`s; rates from raw outcomes | `test_c6_frozen_payload_and_substring_vs_full_payload` |
| **C7** | `layer_joint` min 30; count-only checks; weak CT/TC binding | Min **38**; exact identity check; `assert_hybrid_semantics` / `frozen_hybrid_recipient_tag` | `test_c7_*` |
| **C8** | Thin wiring tests | Non-forward e2e mock covering units, crossfit, eligibility, adapter-off, F3, assembly | `test_c8_end_to_end_mock_record_level_coverage` |

## Full suite

`256 passed, 12 skipped` — see `SYNTHETIC_CPU_TEST_RECORD_B1_C1_C8.txt`  
F1: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`

## Remaining (not closed by this correction)

- B2/B3 empirical requirements; production runtime image verification  
- F5, H3 owner decisions  
- P1–P5 original audit text unavailable  
- Tiny-Gemma forward revalidation (authorization blocker)  
- Scientific execution / pretrained loads (auth OFF)
