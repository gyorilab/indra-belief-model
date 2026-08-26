# `scripts/` index — reusable vs one-off

This directory mixes two kinds of Python entry points, and the distinction
matters for maintenance:

- **Reusable / test-anchored.** Imported by the pytest suite (each such test
  does `sys.path.insert(0, .../scripts)` and then imports the module), or wired
  in as a CI guard. These have callable, tested internals — treat their public
  functions as an interface and keep them green. Exactly 23 files across
  `tests/` contain `sys.path.insert` hooks that load these modules (plus the
  `src/` package).
- **One-off / throwaway.** Analysis probes, dataset builders, figure/report
  generators, and remote pullers run by hand to produce a specific artifact.
  No pytest covers their internals, so nothing enforces them; mostly safe to
  refactor and the first candidates to archive once their artifact is frozen —
  **but a few are imported by sibling one-off scripts** (script→script, not via
  pytest), so `grep -rl "import <name>" scripts/` before touching a public
  function. Known sibling-imported helpers: `build_curation_eval.py` (used by
  `build_multicurator_gold.py`, `build_external_gold.py`, `build_v2_balanced.py`)
  and `pull_my_curations.py`.

Categorization below is by current filename. It is an index only — no script is
moved or renamed here. (Concurrent maintenance may be editing script internals;
this file tracks names, not line-level contents.)

## Reusable / test-anchored

### Guards (pytest-enforced; keep exit-code contract stable)
- `check_contamination.py` — few-shot/holdout overlap guard (via
  `tests/test_contamination_guard_sources.py`).
- `calibration_ship_gate.py` — calibration ship-gate; its FAIL/PENDING exit code
  is asserted by `tests/test_ship_gate_enforcement.py`.

### Calibration pipeline (imported by calibration/results tests)
- `calibration_stage0.py`, `calibration_stage1.py` — staged calibration fits.
- `eval_curation_compare.py` — curation head-to-head comparison.

### Frontier reporting (imported by frontier tests)
- `frontier_report.py`, `frontier_table.py`, `frontier_plot.py`,
  `frontier_paired_stats.py`.

### Gold / eval-set builders (imported by their construction tests)
- `build_rasmachine_eval.py`.

### Scoring driver
- `run_rasmachine_monolithic.py` — monolithic rasmachine scoring run (imported by
  its run test).

## One-off / throwaway

### Dataset & gold builders (hand-run, not imported)
- `build_curation_eval.py`, `build_external_gold.py`, `build_holdout.py`,
  `build_multicurator_gold.py`, `build_v2_balanced.py`, `prepare_dataset.py`.

### Statement/corpus materialization (hand-run)
- `eval_to_statements_json.py`, `external_gold_to_statements_json.py`,
  `external_gold_to_statements_grounded.py`.

### Curation fetch / recovery (remote, hand-run)
- `pull_my_curations.py`, `pull_rasmachine_curations.py`,
  `recover_curation_evidence.py`, `curation_pool_feasibility.py`.

### Analysis probes & reports (produce a specific artifact)
- `belief_headtohead.py`, `cost_report.py`, `text_miner_baselines.py`.
