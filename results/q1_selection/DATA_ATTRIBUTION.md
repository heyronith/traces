# TRACES Q1 selected-instruction attribution

The frozen development and held-out instruction texts are derived from:

**databricks/databricks-dolly-15k**  
Revision: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`  
Source file: `databricks-dolly-15k.jsonl`  
Source-file SHA-256: `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`  
Model-card license: **CC BY-SA 3.0**

Only empty-`context` rows were eligible. Selected rows are identified by zero-based physical JSONL line indices in the pinned file. Instruction text is retained after leading/trailing whitespace strip only.

Attribution for redistribution of these derived instruction files must credit Databricks Dolly 15k under CC BY-SA 3.0 and preserve ShareAlike obligations for adapted material.

## Overlap-screen corpus (not redistributed here)

**yahma/alpaca-cleaned** was used only for full-corpus exact/near-duplicate exclusion.  
Revision: `12567cabf869d7c92e573c7c783905fc160e9639`  
File SHA-256: `bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b`  
License metadata conflict (dataset-card `cc-by-4.0` vs README CC BY-NC 4.0) is reported in the selection manifest; Alpaca instruction text is **not** redistributed in this freeze cohort.
