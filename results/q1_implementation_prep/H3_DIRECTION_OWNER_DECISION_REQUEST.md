# Research-Owner Decision Request — H3 Direction Ambiguity

**Protocol:** v1.3  
**Branch:** `q1-impl-audit`  
**Status:** AWAITING OWNER DECISION — engineering must not independently change frozen scientific interpretation.

## Issue

Primary H3 top-2 conservation currently aggregates `head_effects_fwd` for all five primary seeds (`driver.py`), even when the applicable H1/H2 direction for a seed is **reverse** (e.g. route class `R-or`) or **both** (`R-moved`).

## Ambiguity

Frozen protocol v1.3 does not unambiguously state whether H3 top-2 sets must be taken from:

1. **Forward-only** head effects for all seeds (current engineering behavior), or  
2. **Applicable-direction** head effects (forward for R-and; reverse for R-or; both / union / intersection rule for R-moved), or  
3. Another owner-specified rule.

## Requested decision

Please select one of:

- **H3-A:** Retain forward-only top-2 for H3 across all primary seeds.  
- **H3-B:** Use applicable-direction head effects per seed (specify R-moved aggregation).  
- **H3-C:** Other (specify exact rule and citation).

## Constraint

Until an explicit owner decision is recorded, engineering **will not** change H3 direction selection. Current behavior remains forward-only with this ambiguity flagged.

## Sign-off

| Field | Value |
|-------|-------|
| Owner decision | _pending_ |
| Chosen option | _pending_ |
| Date | _pending_ |
| Signer | _pending_ |
