# Remaining Stage 1 full-pilot issues (for later execution review)

**Documented as of the Final Calibration Readiness Patch.**  
These items are **not** silently fixed by changing the pre-registration.  
They must be resolved (or explicitly accepted) before authorizing the full scientific pilot.

The next independent review may authorize **one bounded calibration** only — not the full pilot.

## 1. Native validation gate ordering

**Issue:** In the current engine, recipient-native validation \(D\) is scored during the validation pass, but the native-effect gate is evaluated only after directed-pair validation work. Held-out skipping is driven solely by the discovery-fold reliability gate.

**Required for full pilot:** Native validation gate must **precede** held-out evaluation (and ideally gate whether transfer results are interpreted).

**Not changed in this patch:** Pre-registration text and frozen protocol remain as-is; ordering fix belongs to a future execution-authorization revision.

## 2. Checkpoint load / unload vs GPU memory

**Issue:** Three dense-LoRA seeds are intended to load one-at-a-time, but the real pilot path has not been memory-budgeted on the target GPU with unload/`empty_cache` verified under the full screening workload.

**Required for full pilot:** Confirm load/unload fits the actual GPU memory budget (including peak during joint top-k hooks).

## 3. `run_real_pilot_pipeline` still unconditionally refuses

**Issue:** Even if `configs/pilot.yaml` were set to `enabled: true` with a valid Stage 1 authorization token, `run_real_pilot_pipeline` / real `execution_mode` still raises after the authorization assert (session hard-stop). This is intentional for pre-execution patches but means an authorized config alone is insufficient to run the pilot without a further code-path unlock reviewed independently.

## 4. Alpaca training-corpus overlap remains unverified

**Issue:** CPU audit reports `UNVERIFIED_CORPUS_INACCESSIBLE` when `datasets` / corpus retrieval fails. Empty overlap lists are **not** evidence of no overlap. Frozen v1 instructions were not modified.

## Out of scope for this patch

- Scientific protocol, frozen dataset, checkpoint family, research direction: unchanged.
- Full three-seed pilot execution: not authorized.
- Calibration GPU execution: not authorized in this patch (gate remains disabled in committed config).
