#!/usr/bin/env python3
"""Re-export scored runs to the current viewer schema (per_statement schema 7 +
metrics.json schema 2), baking the observed LLM cost (from each row's call_log)
and per-run gold. Schema 7 adds the statement-grain instrument: per_statement
gold_statement + coherence_summary, and metrics.json tiers.stmt.verdict_err +
stratified (the /runs statement error-detection surface + /statements belief).

No rescoring: this is a pure transform over data/results/<run>.jsonl. Each run's
sibling `.meta.json` records the corpus it was scored against (`input`); we map
that corpus to its gold benchmark so cost + gold land in one pass.

Usage:
    python scripts/reexport_runs.py                 # the default fleet + eval set
    python scripts/reexport_runs.py path/to/run.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

from indra_belief.results import write_run_export

# Corpus basename -> the gold curation JSONL scored on that substrate.
CORPUS_GOLD = {
    "rasmachine_v1_statements.json": "data/benchmark/rasmachine_v1_gold.jsonl",
    "eval_curation_v1_statements.json": "data/benchmark/eval_curation_v1.jsonl",
}

# The default re-export set: the rasmachine_v1 model fleet (cost spread) + the
# n=1606 balanced eval substrate (gemma vs medpsy).
DEFAULT_RUNS = sorted(glob.glob("data/results/rasmachine_v1_bedrock-*.jsonl")) + [
    "data/results/rasmachine_v1_gemma.jsonl",
    "data/results/rasmachine_v1_medpsy.jsonl",
    "data/results/eval_curation_v1_gemma.jsonl",
    "data/results/eval_curation_v1_medpsy.jsonl",
]


def _gold_for(run_path: str) -> tuple[str, str | None]:
    """Resolve (corpus_path, gold_path) from the run's .meta.json `input`."""
    meta_path = run_path.replace(".jsonl", ".meta.json")
    corpus = None
    if os.path.exists(meta_path):
        corpus = json.load(open(meta_path)).get("input")
    corpus = corpus or "data/corpora/rasmachine_v1_statements.json"
    gold = CORPUS_GOLD.get(os.path.basename(corpus))
    if gold and not os.path.exists(gold):
        gold = None
    return corpus, gold


def main(argv: list[str]) -> int:
    runs = argv or DEFAULT_RUNS
    ok = 0
    for run_path in runs:
        if not os.path.exists(run_path):
            print(f"SKIP (missing): {run_path}")
            continue
        corpus, gold = _gold_for(run_path)
        try:
            meta = write_run_export(run_path, corpus, None, gold_path=gold)
        except Exception as e:  # noqa: BLE001 — driver: report and continue
            print(f"FAIL {os.path.basename(run_path)}: {e}")
            continue
        cost = meta.get("cost") or {}
        g = meta.get("gold") or {}
        print(
            f"OK  {meta['model']:<24} {meta['run_id'][:12]}  "
            f"cost={cost.get('status'):<11} "
            f"${cost.get('total_usd')}  ${cost.get('usd_per_1k_evidence')}/1k  "
            f"gold={g.get('covered')}/{g.get('total')}"
        )
        ok += 1
    print(f"\n{ok}/{len(runs)} runs exported -> data/exports/")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
