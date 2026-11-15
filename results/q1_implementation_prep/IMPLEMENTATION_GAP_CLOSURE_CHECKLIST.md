# Q1 Implementation-Gap Closure Checklist

**Frozen commit:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Worktree:** CPU implementation preparation (uncommitted)  
**Legend:** DONE = code + synthetic tests; PENDING-GPU = requires authorized model execution; OPEN = still incomplete

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| G1 | Activation patching (donor replace) | **DONE** | `src/traces/q1/patching.py` + tests |
| G2 | Residual hybrids TT/CT/TC/CC | **DONE** | `src/traces/q1/hybrids.py` + tests |
| G3 | Donor caches + F6 isolation | **DONE** | `src/traces/q1/caches.py` + tests |
| G4 | Dynamic tag-position derivation | **DONE** | `src/traces/q1/tag_position.py` + tests |
| G5 | Explicit single-BOS end-to-end | **DONE** (code paths) | `q1/bos.py`; wired into `evaluation.py`, `metrics_defs.py`, `throughput.py`, `real_scorer.py` |
| G6 | Descriptive double-BOS pair enum | **DONE** | `conditions.enumerate_double_bos_descriptive_pairs` |
| G7 | Identities 1–9 harness | **DONE** (synthetic) | `q1/identities.py`; **PENDING-GPU** model-level |
| G8 | F4 calibration infrastructure | **DONE** (formula/table) | `q1/f4.py`; **PENDING-GPU** empirical lock |
| G9 | F1 bootstrap + F2 folds | **DONE** | `q1/stats_f1_f2.py` + tests |
| G10 | Cross-fit k=1..10 conditions | **DONE** (enumeration) | `q1/conditions.py`; **PENDING-GPU** execution |
| G11 | Layer joints + residual sweep | **DONE** (enumeration) | `q1/conditions.py`; **PENDING-GPU** |
| G12 | Adapter-off plan + ASR gates | **DONE** (API/logic) | `q1/adapter_off.py`, `q1/gates.py`; **PENDING-GPU** |
| G13 | BF16 + eager no silent fallback | **DONE** | `model_loader.py` refuse non-bf16/non-eager; `runtime_spec.py` |
| G14 | Transitive wheel-hashed image lock | **OPEN** (procedure ready) | `scripts/q1_prepare_runtime_lock.py`; need built image |
| G15 | BOS comparison-merge script | **DONE** | `scripts/q1_bos_cross_version_merge.py` + PASS merge artifact |
| G16 | Pilot gate disabled | **OK** | unchanged `enabled: false` |
| G17 | Workload 824400 / 3600 / 11 | **DONE** | validated by `q1/workload.py` + tests |
| G18 | Output recording schemas | **DONE** | `q1/recording.py` |
| G19 | End-to-end Q1 runner on GPU | **OPEN** | orchestration not launched (correct under restrictions) |

**Scientific validation:** not claimed. Independent review + authorized model-level identity testing still required before PASS for execution.
