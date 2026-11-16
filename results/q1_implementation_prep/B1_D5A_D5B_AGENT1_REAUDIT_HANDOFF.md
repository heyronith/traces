# Agent 1 — Targeted Independent Re-Audit Handoff (D5-A / D5-B)

**STOP for Agent 1 independent review of D5-A and D5-B only.**  
Do not merge to `main`. Scientific authorization remains OFF.

## Scope

Resolve the two remaining blocking checkpoint-provenance findings: **D5-A** and **D5-B**.

Preserve independently accepted **R1, M1, D2, D3, D4, M2, S2**. Do not reopen without a newly demonstrated regression.

Full matrix: `results/q1_implementation_prep/B1_D5A_D5B_CORRECTION_MATRIX.md`

## Exact trees

| Role | SHA |
|------|-----|
| Correction commit (code) | `11165f7c5a3a4db31fff5ccdba450c6b0ee8b128` |
| Correction tree | `095ef2558b839cc61e9db47afaeba0c34edc0f1f` |
| Clean tested tip (suite + handoff) | `af55528c8386e6d3bbe2f8da7104b6e5c4cf71d5` |
| Baseline | `3b7821348f259ad00bd9a031af9eab72d14ad639` |
| Frozen main | `b1e8874f45311678e6fa241f504e7e09d4a7808c` |
| Publish tip | `origin/q1-impl-audit` HEAD (includes this evidence/handoff) |

Suite was run on clean committed correction tree `11165f7`.

## Verification performed (engineering only)

- Adversarial: `tests/test_q1_b1_remediation_d5a_d5b.py` (real provenance/trust functions)
- Full permitted Q1 synthetic suite: **316 passed, 12 skipped**
- Environment: CPython **3.11.15**, NumPy **2.1.3**, torch **2.5.1**, transformers **4.51.3**, PEFT **0.19.1**
- F1 preserved: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
- `PINNED_CHECKPOINT_MANIFEST_SHA256` remains **unset**
- Record: `results/q1_implementation_prep/SYNTHETIC_CPU_TEST_RECORD_B1_D5A_D5B.txt`

## Suggested re-audit focus

### D5-A
1. Production refuses when owner pin unset; caller `expected_weight_file_sha256` is not authoritative.
2. Ten adapter-weight digests must be pairwise distinct; shared base/tokenizer OK.
3. Symlinked seed subfolders refused; cross-seed byte substitution refused even with matching caller map.
4. Traversal / untrusted / malformed manifest keys refused; incorrect owner pin refused.
5. Engineering synthetic ten-unit fixture authenticates only under isolated engineering trust and never sets `production_unlocked`.
6. Adapter-off reference still targets `l19/seed42`.

### D5-B
1. `adapter_config.json` (incl. LoRA scaling fields) authenticated; altered scaling refused.
2. All four frozen §11 files including `special_tokens_map.json`; altered bytes refused.
3. Planted `tokenizer.json` refused; unlisted loadable files refused.
4. Loader requires `use_fast=False`, frozen `GemmaTokenizer`, `is_fast=False`.
5. Returns computed digests only; frozen §11 hash values unchanged (fourth pin added from protocol).

## Explicit non-claims

- **Not** actual B3 checkpoint verification.
- **Not** scientific readiness.
- Real hashes and owner approval remain separate prerequisites.
- No pretrained loads, downloads, GPU, scientific forwards, expenditure, protocol changes, or merge to `main`.
