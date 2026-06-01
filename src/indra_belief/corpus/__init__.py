"""Corpus utilities for INDRA Statement dumps.

The DuckDB persistence + scoring layer (schema/loader/scoring/validity/export/
denominators + the worker) has been removed; the monolithic pipeline
(`scripts/run_rasmachine_monolithic.py` + `indra_belief.scorers.monolithic`)
is the production path and is DuckDB-free. The only surviving corpus surface is
the cost estimator.
"""

from indra_belief.corpus.cost import estimate_cost, MODEL_PRICES_PER_M_TOKENS

__all__ = [
    "estimate_cost",
    "MODEL_PRICES_PER_M_TOKENS",
]
