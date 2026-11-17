# TRACES Q1 — B3-10 Owner-Approved Manifest and Production Pin Freeze

**Document type:** retrospective engineering status and Cursor regression handoff. **NOT an authorization request.** All owner approvals are handled **in chat**, not via GitHub forms.

**Decision:** The research owner explicitly selected **Option 1: approve the audited checkpoint manifest and production pin** in the 2026-11-16 conversation. Agent 1 had delivered a **qualified PASS** after independently reviewing the delegated read-only Modal audit, including the `statvfs` warning and lack of direct Agent 1 Modal access. This records an owner decision already made in chat, **not** a newly requested permission or a claim of direct Agent 1 Modal access.

## Immutable evidence / change scope

- Owner-audited canonical manifest SHA-256: **`d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55`**. The canonical hash was independently recomputed and matched before freeze.
- Approved candidate: `results/q1_implementation_prep/B3_CANDIDATE_CHECKPOINT_MANIFEST.json` at evidence commit `d9a96efd780b7fa75e9859ffd9e363a6dfb970de`.
- Promoted **exact same bytes**, not a regeneration: `results/q1_implementation_prep/CHECKPOINT_PROVENANCE_MANIFEST.json` at commit `e4f1708a089ef04eeb85062545359908ae2c67a8`. Source and production files have **identical Git blob SHA** `7887be2f8d584dd603b871a8151b1e4d9df947db`.
- Production owner pin: `src/traces/q1/checkpoint_trust.py:PINNED_CHECKPOINT_MANIFEST_SHA256` set to the exact digest in commit `ef347b541ff0e164d0414e727cf721e69a151dee`.
- Original manifest notes `B3_CANDIDATE_ONLY` and `PINNED_... must remain unset until owner freeze` were **preserved verbatim** as historical candidate provenance to avoid any change to the **exact owner-approved canonical bytes**. Owner freeze has now occurred; the notes are not new present-tense restrictions. Do **not** edit them silently because doing so changes the approved hash.
- Frozen `main`: **`b1e8874f45311678e6fa241f504e7e09d4a7808c`**, unchanged. Only `q1-impl-audit` is modified.

## B3 gate status

- **B3-01..B3-08:** pinned source/file identities, owner-approved Modal CPU byte acquisition within the $10 cap, actual Volume hashes, exact source object identities, frozen §11 pins and schema-valid candidate published.
- **B3-09:** independent Agent 1 **qualified PASS** on delegated execution: Agent 1 approved the exact verifier and independently reviewed its output and logs; no direct Modal access. 65 files and all source OIDs matched, ten distinct adapters; no unexpected trust-root files, symlinks or specials; sole `statvfs` warning deemed nonblocking by Agent 1.
- **B3-10:** research-owner **approved in chat** and exact audited manifest/pin frozen on `q1-impl-audit` as above.
- **B3-11:** **NOT COMPLETE** — separately review/test the production trust gate in the intended executor environment using non-model CPU logic; real callbacks/loads remain unauthorized.
- **B3-12:** **NOT READY** — built Modal image/runtime lock, F5 signoff, B2/F4/tiny-Gemma empirical validation, science GPU budget and scientific execution authorization remain separate.

## Cursor immediate technical follow-up: regression tests and review only

**Baseline at time this note was prepared:** `ef347b541ff0e164d0414e727cf721e69a151dee`. Do not change the frozen hash or file bytes. Do not merge into `main`.

1. Fetch the branch, independently recompute `canonical_checkpoint_manifest_bytes()` SHA-256 for `CHECKPOINT_PROVENANCE_MANIFEST.json`; require exact `d070262e…4fa55`; compare Git blob/content equality with original candidate; run `validate_checkpoint_manifest_schema()` and `authenticate_production_checkpoint_manifest(repo_root=<trusted checkout>)` as a CPU-only, **non-model** test of the manifest authentication path.
2. Run the permissible synthetic/CPU regression tests, including checkpoint trust, B1/D5, H3 and B3 tests. **Known pre-freeze test assumptions will need updating without weakening safety coverage**: `tests/test_q1_b3_modal_byte_verify.py` has `test_pin_remains_unset` and other assertions that the pin is `None`; `tests/test_q1_b1_remediation_d5a_d5b.py` and `test_q1_b1_remediation_d5_integration.py` also contain historical `pin is None` assumptions. Replace stale assertions with explicit tests of the newly owner-frozen digest. Keep the *negative* unpinned test by **monkeypatching the pin to `None` in that particular test**. Where synthetic B3 download/preparation tests require pre-freeze assumptions, simulate `None` locally via fixture/monkeypatch and add a separate test that B3 download scripts fail closed once the production pin is frozen. Do **not** relax `require_production_pin_unset` or remove fail-closed production checks to make tests green.
3. Add/confirm a positive CPU-only test that the exact audited manifest authenticates with the exact owner pin, plus negative cases for missing manifest, altered byte, wrong SHA, unpinned default and caller-supplied forged hash. Distinguish `production_unlocked` **manifest identity** from scientific permission and do not call model loading/forwards in these tests.
4. Record test commands, actual interpreter, exit codes, F1 hash, H3 regression status, precise code/tree hashes and any remaining blockers in a new **technical evidence** status (not an owner authorization request). If unexpected failures or new production authentication behavior appear, **STOP for Agent 1 review**, rather than patching scientific controls without permission.
5. Independent Agent 1 reviews the two-file freeze change, test updates and exact canonical digest as a follow-up engineering gate. B3-11 remains separate from a B3-10 pin freeze; a production model load is **NOT** an authorized test.

## Hard STOP

The pin being set is a **checkpoint identity trust anchor**, not an experiment start signal. **Scientific execution authorization stays OFF.** Do not start Modal GPU jobs, pretrained loads/forwards, B2/tiny-Gemma model execution, new checkpoint redownloads, expenses or protocol edits without separately bounded owner approval in chat. Keep frozen `main` unchanged. No new GitHub authorization requests.
