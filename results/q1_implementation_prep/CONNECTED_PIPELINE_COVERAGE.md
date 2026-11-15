# TRACES Q1 — Connected Pipeline Coverage

**Protocol:** 1.3  
**Frozen commit:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Scientific auth default:** OFF (`TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION`)  
**Tiny-Gemma auth:** separate (`TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD`) — does not unlock this path  

## Stages

- `1_frozen_prompt_catalog_and_bos_verification`
- `2_model_unit_configuration_and_provenance`
- `3_donor_capture_and_clean_reference_freezing`
- `4_tt_ct_tc_cc_conditions`
- `5_single_head_pair_joint_residual_interventions`
- `6_frozen_effect_calculations`
- `7_f1_f2_uncertainty_and_decision_rules`
- `8_structured_result_persistence_and_audit_logging`

## Workload (from enumerator validation when built)

```json
{
  "baseline_scientific_tf_v1": 824400,
  "baseline_scientific_gen_v1": 3600,
  "expected_loads": 11,
  "tf_passes_total": 824830,
  "scientific_condition_tf": 824400,
  "integrity_tf": 430,
  "f4_double_bos_within_scientific_baseline": 10500,
  "benchmark_overhead_tf": 0,
  "gen_passes": 3600,
  "n_loads": 11,
  "scientific_tf_matches_readiness_v1": true,
  "gen_matches_readiness_v1": true,
  "loads_match_readiness_v1": true,
  "breakdown": {
    "scientific_and_prescribed_exploratory_extras": 824400,
    "identity_integrity_extra": 430,
    "exploratory_identity": 150,
    "confirmatory_identity": 280,
    "f4_and_double_bos_inside_baseline": 10500,
    "benchmark_proposal_not_included": true
  },
  "primary_tf_per_prompt": 660,
  "primary_conditions_enumerated": 658,
  "secondary_tf_per_prompt": 136
}
```

## Authorization

- Live forwards: **not invoked** by Phase 2 engineering.
- Analysis-only path available for synthetic / precomputed effects.

