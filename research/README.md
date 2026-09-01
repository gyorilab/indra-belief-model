# research/

Ten records of measurement and design. They are kept as dated records rather
than rewritten when code moves, because rewriting measured prose distorts the
claim. Several carry a provenance banner warning that their citations may name
deleted code; that warns you the paths are stale but not which claims survived.
This index says which.

- **CURRENT** — describes the tree as it stands; follow it.
- **HISTORICAL-MEASUREMENT** — the numbers hold, taken against code that git
  history still has; some surrounding citations no longer resolve.
- **OBSOLETE-TOPOLOGY** — its account of what exists and how the pieces fit has
  been overtaken; read it for the measurements, not the map.

| document | status | what it is for | where it has moved on |
|---|---|---|---|
| `corpus_belief_runbook.md` | CURRENT | Operating procedure for a corpus-scale run: shard, score with verdict-token logits, fit the profile on the target stack, build the `{stmt_hash: belief}` table. | — |
| `scoring_methods.md` | CURRENT | Self-contained derivation of the belief score — per-evidence reading, calibration, pooling into a statement belief — with every assumption stated. | — |
| `rasmachine_eval_harness.md` | CURRENT | Three-stage CLI recipe for scoring a sampled rasmachine corpus against curator gold. Every script and the input pickle it names resolve. | — |
| `deployment_plan.md` | HISTORICAL-MEASUREMENT | Sizes the two bulk domains (MSstatsBioNet, EMMAA) and the realtime path against the INDRA stack. | §3's socket finding holds: `BeliefEngine(scorer=LLMBeliefScorer(client))` is still a drop-in. §7's blockers have drifted — `calibration_constants.py` now registers seven profiles with six enabled, `verdict_only` is a registered `VARIANTS` key, and the spend ledger behind B3 is gone. |
| `indra_paper_literal_vs_llm_comparison.md` | HISTORICAL-MEASUREMENT | Runs the 2023 assembly paper's own released code to reproduce its Table 6, then places INDRA's deployed belief, the paper's random forest and LLM-reader arms on identical statements. | It calls the viewer's `/paper` page the primary artifact and itself the prose companion; the viewer is gone, so this memo is the whole record. Its fixed-belief measurement — 64.8% vs 11.9% accuracy among single-evidence statements that all score belief 0.65 (χ² = 30.7, p = 3.6e-6, n = 315) — is unaffected. |
| `kernel_unification_findings.md` | HISTORICAL-MEASUREMENT | Record of the refactor that collapsed two implementations of the scoring kernel into one. | Its subject survives: `prepared_execution.py` is the one request assembler, `verdict.py` the one parser. Most of §7 "what is still owed" is moot — the comparison runs, viewer components and ledger those items are owed against were removed. |
| `probe_battery_findings.md` | HISTORICAL-MEASUREMENT | The held-out result that one no-reasoning probe carries essentially the whole AUROC gain, and the other fifteen add an amount whose CI spans zero. | The battery itself was not removed: `src/indra_belief/probes/battery.py` is live and covered by `tests/test_probe_battery.py`. What is gone is the evaluation runner `scripts/eval_probe_battery.py` and the comparison artifacts it read. |
| `deck_real_examples.md` | HISTORICAL-MEASUREMENT | Example statements sampled from `external_curator_gold_v1`, joined to the 13-model Bedrock fleet, each belief confirmed against the live INDRA DB API. | The slide numbering it was written against is retired; the deck is `presentations/gyori-belief` (gitignored). The rows and verbatim sentences remain verified. |
| `serving_architecture.md` | OBSOLETE-TOPOLOGY | Measured memory, preflight and latency properties of the scoring path, and the refactor sequence that followed from them. | Its "two execution paths that exist today" makes the batch path the comparison replayer; that package and its console script are gone, leaving the library path alone. Its §3 finding that `VARIANTS` has no `verdict_only` key is now false. |
| `serving_deployment.md` | OBSOLETE-TOPOLOGY | Latency decomposition of a live score, the grounding finding (identity versus context), and the sketch of a serving interface. | §1's deployed state is a batch image built from a `Dockerfile` that no longer exists; `Dockerfile.live` from §7 is the only image in the tree. Its finding that no serving process exists — no HTTP entry point in `src/` or `scripts/` — still holds. |
