# Synthetic end-to-end integration report

**No GPU. Stub scorers only.**

## Workflow covered

discovery screening → fold reliability → frozen rankings → recipient-native reference → donor transfer → random / layer-type / damage-matched controls → held-out scoring → per-pair analysis → machine-readable outputs

## Planted worlds

1. **shared_causal** — seeds 42/43/44 share the same planted attention-head set.  
   Result: top-10 rankings recover ≥7/10 planted components; held-out transfer \(D\) remains high (>0.4 mean).

2. **disjoint_causal** — each seed has a non-overlapping planted set.  
   Result: for donor42→recipient43 at k=10, recipient-native \(D\) > frozen-donor transfer \(D\).

## Gates checked

- `allow_execution=False` raises (config disabled / unauthorized).
- Calibration CLI returns `REFUSED_NOT_AUTHORIZED` without launching GPU.
- Ranking artifacts hashed; donor ranking hash attached to transfer rows.

## Command

```bash
uv run pytest tests/test_pilot_e2e_synthetic.py -v
```
