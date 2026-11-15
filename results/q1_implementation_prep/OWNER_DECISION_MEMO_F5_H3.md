# Owner-Decision Memo — F5 and H3 (Unresolved)

**Branch:** `q1-impl-audit`  
**Protocol:** v1.3 FROZEN — this memo does **not** amend the protocol or make owner decisions.  
**Scientific authorization:** OFF  

---

## F5 — LoRA adapter dtype and numerical policy

**Status:** AWAITING EXPLICIT RESEARCH-OWNER APPROVAL  

**Full note:** `results/q1_implementation_prep/F5_OWNER_DECISION_NOTE.md`

| Option | Policy |
|--------|--------|
| **A** (proposed loader path) | BF16 base + BF16 LoRA; `autocast_adapter_dtype=False` |
| **B** | PEFT default / FP32 adapters (exact flags must be pinned) |

Until the owner completes the sign-off block in `F5_OWNER_DECISION_NOTE.md`, F5 remains unresolved. Engineering must not treat Option A as approved merely because it is coded as the proposed path.

---

## H3 — Frozen direction-selection interpretation

**Status:** AWAITING OWNER DECISION  

**Full request:** `results/q1_implementation_prep/H3_DIRECTION_OWNER_DECISION_REQUEST.md`

Frozen v1.3 does not unambiguously state whether primary H3 top-2 conservation uses:

| Option | Interpretation |
|--------|----------------|
| **H3-A** | Forward-only head effects for all primary seeds (current engineering behavior) |
| **H3-B** | Applicable-direction effects per seed (specify R-moved aggregation) |
| **H3-C** | Other owner-specified rule with citation |

Engineering **will not** change H3 direction selection until an explicit owner decision is recorded.

---

## Constraint

Neither F5 nor H3 is closed by this memo. No confirmatory or scientific execution may proceed under unresolved F5/H3 interpretations.
