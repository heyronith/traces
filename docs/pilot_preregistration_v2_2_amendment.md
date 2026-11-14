# Stage 1 pre-execution amendment v2.2

**Dated:** 2026-11-13 (pre-execution; before any scientific pilot GPU run)  
**Amends:** `docs/pilot_preregistration_v2_1.md` (v2.1 retained unmodified)  
**Base code:** post–launch-safety `abe83e7…` remediation series  

This amendment records scientific-analysis and reproducibility clarifications identified by independent Agent 1 audit. It does **not** change the frozen v1 dataset, D/K definitions, intervention semantics, seed set {42,43,44}, k budgets {5,10}, or the research question (still **NOT YET SELECTED**).

---

## Scientific vs engineering changes

| Kind | Examples | Binding |
| --- | --- | --- |
| **Scientific** | Gate early-stop policy, comparison-pair identity, prompt-level D/K persistence, frozen damage-match selection, pinned HF revisions, empty-set D instrumentation | This amendment + code |
| **Engineering** | Unitized Modal schedule, 4 h timeout, A10G-only request, durable volume commits, budget estimate versioning | Not a prereg change |

Do not treat workload/budget version bumps as scientific protocol amendments.

---

## Gate early-stop policy (aligned with executor)

### Reliability failure (`discovery_fold_reliability`)

- **Fail condition:** every seed has top-**10** discovery-fold Jaccard \< 0.3 (primary k=10).
- **On failure:** **held-out is skipped** for all recipients (scientific early-stop).
- **Validation remains available** for transfer condition scoring and native-gate recording.
- **Not** an execution/crash failure; units that finish validation-only remain **complete**.
- **Donor reliability** does not by itself rewrite validation measurements; it gates whether held-out evaluation is permitted when every seed fails.

### Native failure (`recipient_native_validation_effect`)

- **Fail condition (exact):** for a recipient, compute native top-k validation \(D\) at **each** \(k \in \{5,10\}\). Pass if **any** \(k\) has \(|D_{\mathrm{native}}(k)| \ge 0.05\). Aggregation is \(\max_k |D(k)|\) — **independent of loop order** (no last-k overwrite).
- **On failure for that recipient:** skip **that recipient’s held-out**; **preserve completed validation rows**.
- Unit may still be marked complete (validation-only outcome).

### Both gates

Scientific early-stop must not be reported as timeout/crash. Launch refusal applies only to genuine execution incompleteness.

---

## Comparison-pair identity

Every transfer condition carries durable:

- `comparison_donor_seed`
- `comparison_recipient_seed`
- `comparison_pair_key` = `donor{d}->recipient{r}`

Random controls keep `donor_seed=None`; native keeps ranking donor = recipient. Analysis **must not** infer the comparison donor from those fields. Each of the six directed pairs × each k × each evaluated split must form a five-condition comparison (donor / native / random / layer-type / damage-matched-or-unmatched). Missing or ambiguous controls are **rejected**, not silently regrouped.

---

## Prompt-level observations

For every scored set condition, persist `D_per_instruction` and `K_per_instruction` keyed by frozen instruction IDs, from the **same** forwards as the aggregates (no extra model calls). Aggregates remain the mean of per-instruction values under unchanged D/K definitions and sign convention.

Use for **descriptive** paired differences and spread across shared held-out instructions only. **No** unregistered confirmatory tests.

### Descriptive paired statistics (specified before execution)

For transfer-minus-control differences (comparison-group aggregates and aligned prompt-level pairs), report:

- median
- IQR (q3 − q1)
- standard deviation
- sign counts (`n_positive`, `n_negative`, `n_zero`)

These remain **descriptive only**, not confirmatory (no unregistered tests or claim thresholds).

---

## Frozen damage-matched controls

1. Select on **validation only** (never held-out; never backdoor \(D\)).
2. Persist component IDs, candidate \(K\) map, match status/kind, and provenance hash.
3. On resume, **reuse** the frozen selection; reject drift in IDs or selection hash.

---

## Locked revisions

- **Executed Git SHA** is part of the immutable run identity across all six units.
- **Pinned HF revisions** (calibrated):
  - Base `google/gemma-2-2b`: `c5ebcd40d208330abc697524c919956e692655cf`
  - Adapter dense-LoRA: `6303a74fc832a34d584999364817268aefd9623b`
- Fail closed on unresolved `main` or revision drift.

---

## Empty-set D instrumentation (N4)

Per seed, after load: score \(D\) with the empty component set under the same intact/ablated conventions. Expect \(|D| \le 10^{-4}\) (synthetic exact 0). Record as pre-screening instrumentation. **Do not** use the result to tune the scientific metric.

---

## Exploratory first-repeat D logging

**Not implemented** in this remediation. If added later, it must be logged as exploratory instrumentation only and named in a further dated amendment before GPU execution.

---

## Adequacy rules

`traces.pilot.analysis.ADEQUACY_RULES` matches these gate policies (v2.2). Counting “transfer beats controls in ≥2/6 pairs” remains **descriptive only**, not confirmatory.
