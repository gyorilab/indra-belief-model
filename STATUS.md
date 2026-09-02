# Status

As of 2026-09-01, commit `COMMIT_PLACEHOLDER`. State only: what the system *is* lives in
[README.md](README.md), how belief is computed in
[research/scoring_methods.md](research/scoring_methods.md).

## Shipped

- **Scoring path** — the monolithic scorer, one LLM call per `(Statement, Evidence)`
  pair. Entrypoints:
  - `score_statement(stmt, client)` / `score_evidence(stmt, ev, client)`, exported
    from `indra_belief`.
  - `LLMBeliefScorer` (`src/indra_belief/belief_scorer.py`) — the INDRA seam:
    `BeliefEngine(scorer=LLMBeliefScorer(client))`. Not exported from the package
    root, so import it by module path. Call `estimate_calls()` first: unlike
    INDRA's arithmetic scorers this one spends per evidence.
  - `python -m indra_belief.scorers.scorer` — the benchmark-harness CLI.
  - Corpus scale (60M evidences, verdict-token logits as the per-evidence weight):
    four separable stages, see [research/corpus_belief_runbook.md](research/corpus_belief_runbook.md).
- **Calibration profiles** — `src/indra_belief/calibration_constants.py::_PROFILE_META`
  holds seven fitted readers, keyed on (served model, prompt SHA-256). Six are
  `enabled`: `gemma_remote`, `gemma_bedrock_rf`, `gemma_bedrock_rf_noconf`,
  `local_gemma_mlx`, `local_gemma_mlx_verdict_only`, `vllm_gemma_verdict_only`.
  One is `disabled`: `medpsy_remote` (3/4 gate, ECE leg failed). Any other
  (model, prompt) pair resolves to `None` and stays on the hard gate.
- **Checks, all green here.** After `python -m pip install -e ".[dev]"`:
  - `python -m pytest -q` — 1054 passed. On a clean checkout (`git worktree`
    of HEAD, no gitignored artifacts) 1039 pass and 15 skip; every skip names
    the absent run or replay it needs, and CI runs `-rs` so the list is in the
    log rather than hidden inside a pass count.
  - `python scripts/check_contamination.py` — CLEAN. Its eval set is derived
    from `_PROFILE_META` (fit + validation gold of every profile, all tracked)
    plus the representative-curation snapshots and `--holdout`; 17 enumerated
    overlaps (`KNOWN_LEAKS`, nine `eval_curation_v1` rows that match v6/v7
    prompt examples, measured immaterial at ≤ 0.009 nats on every fitted
    log-LR) are waived by exact key and anything new fails.
  - `python scripts/check_import_boundary.py` — CLEAN, 62 first-party imports
    across 39 core modules.
  - `python scripts/smoke_end_to_end.py` — 5 checks, hermetic (no socket, no
    credential); `--live` scores for real.
  - CI (`.github/workflows/ci.yml`) runs only the first two.

## Known broken / limited

- **The vLLM in-call isotonic is deliberately withheld from the registry.** (Unblocked only by the `/scratch` artifact.) Its row
  in `src/indra_belief/probes/calibration.py::_INCALL_CALIBRATIONS` is commented out
  because the fitted artifact (`incall_vllm.json`) exists only on cluster scratch.
  Registering a row whose file no checkout has would flip
  `supports_sentence_calibration` to True and kill stage 4 on `FileNotFoundError`
  instead of its own refusal message. Withheld, `--require-calibrated` refuses
  namedly; scoring is unaffected, gating on the belief profile, which *is*
  registered. Add the row in the same commit as the file.
- **`vllm_gemma_verdict_only` is the weakest-cited profile.** (Unblocked only by the `/scratch` runs.) Counts and gold
  digest are pinned, but its fit and validation runs point at
  `/scratch/h.yan/data/gold_results`, untracked — nobody else can re-derive the
  Brier and ECE in its note. Re-fitting with `--report` closes it.
- **No test runs against a real vLLM server.** (Needs a served vLLM; none runs on this host.) `tests/test_shard_runner_no_cot.py`
  drives the `--backend server` path, logprobs included, but against a local stub;
  the offline-backend tests stub `sys.modules['vllm']`. Nothing exercises a served
  model, so a served-side regression surfaces only in a corpus run.
- **`local_gemma_mlx` still carries the `eval_curation_v1` skew** that
  `gemma_bedrock_rf` was refitted off (as do `gemma_remote` and `medpsy_remote`).
  MLX throughput has deferred the refit: `holdout_large_fit` is 4,303 rows at
  ~19 s/row deliberated, roughly a day of local scoring. The nine leaked rows in
  that gold are not the reason — excluding them moves no log-LR by more than
  0.009 nats.
- **Verdict-only profiles trade calibration for discrimination** — Brier and AUROC
  improve while ECE roughly doubles. A consumer thresholding on belief feels that.

## Open

- **The corpus error rate is not identified from the gold we have.** Four of the
  six evaluable sets are class-balanced by construction, so selection into them
  depends on the label and no reweighting recovers a population rate. Belief is
  not a sufficient stratifier either. `data/benchmark/README.md` carries the
  counts, the mechanism, the evidence, and what a valid estimate would require;
  it is the one place that argument lives.
- **The one probability sample is half-finished.** 5,000 CoGEx evidence rows were
  drawn by uniform reservoir; 403 have been curated, by a single curator, with no
  record of which of the 5,000 were served in what order. Logging the draw order
  and finishing the remainder would produce the first corpus rate this project can
  defend. Same root cause as `fit_prevalence` being the curated base rate rather
  than the corpus one.
