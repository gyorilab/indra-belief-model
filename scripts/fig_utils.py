"""Shared constants for the rasmachine figure-refresh scripts.

DRY lift (R5): ``SRC`` (the complete gemma scoring run) and ``NON_ARTIFACT``
(the three verdict-bearing "meaningful" buckets used for the bucket-aware
Figure 1-3 / Figure-4-subtable filter) were defined identically in
fig3_refresh, fig4_refresh, fig3_drill_data, and tldr_refresh. Values are
unchanged.
"""

SRC = "data/results/rasmachine_mono_gemma_remote_direct.jsonl"

# Bucket-aware filter: a statement is kept iff it has >=1 NON-ARTIFACT scored
# evidence, and OUR belief aggregates over the NON-ARTIFACT evidence only.
# reader_hallucination is treated as an artifact, NOT meaningful.
NON_ARTIFACT = {"semantic_correct", "semantic_incorrect", "hedged_evidence"}
