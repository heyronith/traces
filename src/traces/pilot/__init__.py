"""Stage 1 three-seed pilot infrastructure (disabled by default).

Final research question: NOT YET SELECTED.
This package tests feasibility/informativeness of cross-seed activation-ablation
transfer. It must not run while configs/pilot.yaml has enabled=false.
"""

from __future__ import annotations

PILOT_ENABLED_DEFAULT = False
RESEARCH_QUESTION = "NOT YET SELECTED"
SCIENTIFIC_CLAIM_AUTHORIZED = False

# l1523 adapted layers
ADAPTED_LAYERS = tuple(range(15, 24))  # 15..23 inclusive
N_QUERY_HEADS_GEMMA2_2B = 8
HEAD_DIM_GEMMA2_2B = 256
INTERMEDIATE_SIZE_GEMMA2_2B = 9216
MLP_GROUP_SIZE = 256
# 9 layers * 8 heads + 9 layers * (9216/256=36) groups = 72 + 324 = 396
N_COMPONENT_GROUPS_FULL_SCAN = 396
