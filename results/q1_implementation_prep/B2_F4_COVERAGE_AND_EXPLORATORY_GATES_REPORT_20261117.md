# F4 coverage reconciliation + B2 exploratory identity / double-BOS gates

**Overall exploratory verdict: PASS** (2026-11-17/18 UTC). Development-only. Historical F4 preserved. Single-BOS stopping rule **unchanged**. No held-out / confirmatory / main / merge.

Baseline handoff: `ab7d4af3ace6124dc573e9fe6aa48f6a9434d115`.

## 1) F4 applicability (CPU)

| Finding | Result |
| --- | --- |
| Last-pos F4 covers I1, I8 | Yes |
| Last-pos F4 covers I2–I6 full residual/upstream positions | **No** |
| Adversarial fixtures | last-pos D=0 while full-tensor diffs fail ε |
| CPU tests | `test_q1_f4_coverage_applicability.py` PASS |

## 2) Identity-applicable F4 supplement v2

| Item | Value |
| --- | --- |
| Verdict | **PASS** |
| Modal | `ap-2Bgmbu13y02Grc4c5rtboz` |
| Capture | full-seq L19 residual; full-seq o_proj L0–L14; last-pos logits/margin |
| 900/900 forwards | OK; all ε=**0.001** |
| Artifact SHA-256 | `32faa1e5a0ae2a8d2fc1aa26d028035b81210aefb67ebe969a6f1df7364b427d` |
| Historical F4 SHA-256 | **unchanged** `80d7985c3d6241c5661cfacccbea2764c0dec429a9f637131d531727de41d021` |
| Metered (run delta) | **$0.02283** |

## 3) Identities 1–9 (real-model, development)

| Item | Value |
| --- | --- |
| Verdict | **PASS** |
| Modal | `ap-MUwAChspsZAth4rRME6iOg` |
| Bound F4 | identity-applicable v2 |
| Seeds / prompts | 42–44 / 50 development |
| Artifact SHA-256 | `814d489b4fffc0f9bb629b5a2d72236cdefdca957c78d446f35879d8e7e6eb41` |
| Auth | ok (production unlock not required for exploratory publish) |
| Wall | ~1053 s |
| Metered (run delta) | **$0.38973** |

## 4) Descriptive double-BOS 64-pair scan

| Item | Value |
| --- | --- |
| Verdict | **PASS** |
| Modal | `ap-ctoXJWoJztaJ5uEkLetk99` |
| Patched forwards | **9600** |
| `alters_single_bos_stopping_rule` | **false** |
| Artifact SHA-256 | `aefc70bb2638663065201a75683f49009b3edb18300d8d59b3cc96680d74760f` |
| Auth | ok |
| Wall | ~820 s |
| Metered (run delta) | **$0.29289** |

## Cost summary (this reconciliation session run deltas)

| Job | USD |
| --- | ---: |
| F4 identity-applicable supplement | 0.02283 |
| B2 identity suite | 0.38973 |
| B2 double-BOS scan | 0.29289 |
| **Session total** | **≈ 0.705** |

Standing non-main authorization; internal caps respected. Main experiment still **NOT AUTHORIZED**.

## Explicit non-claims

Does not close confirmatory science, production empirical unlock, held-out analysis, or merge to frozen `main` (`b1e8874f…`).
