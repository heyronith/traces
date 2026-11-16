# B1-C9…C12 Correction / Test Matrix

**Implementation commit:** `e1569b654298cb01b59cc1e2bf02e7cd8eac4b62`  
**Tree SHA:** `ec8a2e982c71d4eb4312ea97604dbf48c5d8749b`  
**Scientific authorization:** OFF  
**S2:** independently accepted (preserved)  
**C1–C8:** preserved

| ID | Defect | Correction | Regression |
|----|--------|------------|------------|
| **C9** | Live callbacks received bare `prov["model"]` | Typed `LoadContext` / `require_load_context`; `_execute_live_unit` passes full mapping to donors + execute | `test_c9_*` |
| **C10** | Cross-fit rejected legitimate NaN outside eval fold | `validate_fold_eval_joint_effects`; fold-aware `crossfit_evaluate_topk` | `test_c10_*` |
| **C11** | Raw observations not durable | `scientific_observations.persist_unit_scientific_observations` + live-unit wiring; hashed artifacts; no model/cache/credentials | `test_c11_*` |
| **C12** | Catalog IDs compared to same enumerator | `independent_expected_primary_condition_ids` / secondary from frozen constants | `test_c12_*` |

## Full suite

`268 passed, 12 skipped` — `SYNTHETIC_CPU_TEST_RECORD_B1_C9_C12.txt`  
F1: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
