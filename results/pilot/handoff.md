# Stage 1 pilot preparation handoff

**Pilot not executed. No Modal GPU jobs launched.**  
**Final research question: NOT YET SELECTED.**

## Methodological choices (concise)

1. **Full 396-group scan** over layers 15–23 (72 attention heads + 324 MLP groups) so absolute indices match the `l1523` adapted band across seeds, without a shortlist that could be tuned on evaluation data.
2. **Discovery metric = teacher-forced trigger log-prob drop**; greedy ASR is confirmation only — matches Stage 0 throughput reality and keeps ranking continuous.
3. **Frozen donor rankings** serialized with integrity hashes; recipient eval loads artifacts and cannot re-rank on `held_out`.
4. **Splits 50/25/25** after seeded shuffle; smoke prompts excluded; OpenRouter not used; Alpaca overlap flagged as possible/unchecked.
5. **All 6 directed seed pairs** reported with explicit dependence warning; adequacy bars are feasibility thresholds, not definitive tests.
6. **Budget** accounts for screening + controls + transfer + KL + greedy + 3 loads + fallback band; explicitly rejects the Stage 0 **$0.53** figure as full-pilot cost.

## Dry-run cost range (software only)

| GPU | USD low | USD high |
| --- | ---: | ---: |
| A10G | ~3.07 | ~7.53 |
| L4 | ~2.23 | ~5.48 |
| T4 | ~1.67 | ~4.11 |

Hours (A10G): ~2.8–6.8 including contingency/rerun band.  
Not an account spend limit — verify Modal Billing/Usage.

## Commands used (CPU only)

```bash
uv run pytest -v
uv run python -m traces.pilot.runner
```

## Gate

`configs/pilot.yaml` → `enabled: false`  
Execution requires independent-review authorization token.
