# TRACES Q1 — P1–P5 Defect Traceability Matrix

**Branch:** `q1-impl-audit`  
**Search scope:** repository docs, `results/`, Agent 1 audit packages, agent transcripts, compaction segments  
**Date of search:** 2026-11-15  

## Search outcome

**Original independent-audit text defining findings labeled P1, P2, P3, P4, and P5 was not located.**

Evidence of the labels themselves appears only as **outstanding-work placeholders** in research-owner instructions and Cursor delivery notes (e.g. “Record P1–P5 … as outstanding work”), not as a sourced Agent-1 finding list with acceptance criteria.

Related audit corpora that **were** found use different ID schemes:

| Corpus | IDs | Location (examples) |
|--------|-----|---------------------|
| Agent 1 closure / re-audit | A1–C7, R3–R10, S1/S2 | `results/q1_implementation_prep/AGENT1_*`, reaudit reports |
| Implementation-gap readiness | G1–G19 | gap / readiness reports under `results/` |
| Tiny-Gemma readiness | T1–T6, D1–D3 | tiny readiness / preflight docs |

Per Phase 2 instructions: *If the original audit text is unavailable, flag the finding as unresolved rather than inventing a resolution.*

---

## Matrix

| ID | Original finding & acceptance criterion | Affected code | Correction | Regression test | Remaining limitation | Closure |
|----|-------------------------------------------|---------------|------------|-----------------|----------------------|---------|
| **P1** | **UNAVAILABLE** — original audit text not found | Unknown | None claimed | None that maps to a sourced P1 | Cannot close without source text | **UNRESOLVED** |
| **P2** | **UNAVAILABLE** — original audit text not found | Unknown | None claimed | None that maps to a sourced P2 | Cannot close without source text | **UNRESOLVED** |
| **P3** | **UNAVAILABLE** — original audit text not found | Unknown | None claimed | None that maps to a sourced P3 | Cannot close without source text | **UNRESOLVED** |
| **P4** | **UNAVAILABLE** — original audit text not found | Unknown | None claimed | None that maps to a sourced P4 | Cannot close without source text | **UNRESOLVED** |
| **P5** | **UNAVAILABLE** — original audit text not found | Unknown | None claimed | None that maps to a sourced P5 | Cannot close without source text | **UNRESOLVED** |

---

## What Phase 2 did instead (not claimed as P1–P5 closure)

Engineering work listed alongside P1–P5 in owner instructions was implemented under **explicit separate workstreams**, without remapping them onto invented P-numbers:

1. Connected live-execution driver + scientific auth gate (`scientific_execution.py`)
2. Loader / post-load provenance hardening (`loader.py`, `post_load.py`)
3. F5 owner-decision note (no approval)
4. Runtime-lock preparation (`runtime_lock.py`)
5. Statistical pipeline corrections (F2 joint sets, H2/H3, route-denom gates)

**Do not treat those workstreams as P1–P5 closure.**

### C6 retention (Phase 2 internal audit corrections)

All five findings remain **UNRESOLVED**. No acceptance criteria were invented. Unrelated C1–C5 engineering fixes are **not** mapped onto P1–P5.

---

## Required to resolve

1. Provide or locate the original independent-audit document that defines P1–P5.
2. Map each finding to code, fix, test, and residual risk.
3. Re-issue this matrix with sourced quotes and closure decisions.
