# Bounded calibration execution handoff

**Authorized scope:** one Modal GPU calibration only.  
**Scientific pilot:** NOT run (`scientific_pilot_launched: false`).  
**Config after run:** `enabled: false`, `calibration.enabled: false`, token cleared.  
**Reviewed commit executed:** `c756f57c84174737ad0e76743f390ff7cd0dc6c8`  
**Exit status:** 0 (no retry)

## Hardware

| Field | Value |
| --- | --- |
| GPU | NVIDIA A10 (priced as A10G) |
| Peak allocated VRAM | 6,408,896,512 bytes (~5.97 GiB) |
| dtype | bfloat16 |

## Timings

| Phase | Seconds |
| --- | --- |
| Model load | 12.05 |
| Teacher-forced D (all sets) | 1.41 |
| Benign KL K (all sets) | 1.01 |
| Greedy generation (all sets) | 26.33 |
| Scoring wall (excl. load) | 28.74 |
| **Total wall including load** | **40.79** |

## Counts

- Prompts: 4 (`pilot_001`, `pilot_013`, `pilot_009`, `pilot_073`)
- Component sets: 3 (attention, MLP, joint)
- TF forwards: 40
- Gen calls: 24
- Generated tokens: 744

## Intervention-set results (timing evidence only — not pilot findings)

| Set | D | K | greedy ASR | full-payload ASR |
| --- | --- | --- | --- | --- |
| attention | 7.42e-5 | 0.00937 | 1.0 | 1.0 |
| mlp | 9.08e-4 | 0.0155 | 1.0 | 1.0 |
| joint | -0.00271 | 0.0135 | 1.0 | 1.0 |

## Cost (software estimate)

- Scoring-wall only: ~$0.0088
- Including load: ~$0.0125
- Incremental allowance: $1.00 — **within allowance**
- Not an enforced Modal account spend cap

## Provenance

- Git SHA: `c756f57c84174737ad0e76743f390ff7cd0dc6c8`
- Base revision: `c5ebcd40d208330abc697524c919956e692655cf`
- Adapter revision: `6303a74fc832a34d584999364817268aefd9623b`
- Adapter subfolder: `l1523/seed42` (seed 42 only)
- HF secret: `hf-token` (value never logged)

## Scope confirmation

- No other seeds
- No full component scan
- No three-seed pilot
- No automatic retry
- Local report: `results/pilot/calibration_report.json`
- Modal volume path: `/results/pilot/calibration_report.json` (same JSON-safe payload returned to CLI)

## Artifacts committed

- `calibration_report.json`
- `calibration_handoff_summary.json`
- `calibration_execution_redacted.log`
- this handoff

Temporary authorization was not committed; `configs/pilot.yaml` remains disabled.
