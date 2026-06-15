# Rasmachine sampled-statements eval harness

A reusable pipeline to score INDRA belief extractions with the monolithic scorer
and compare models (e.g. gemma-26B vs MedPsy-4B) on a human-curated gold set.
Three stages, each a thin CLI over canonical libraries — re-runnable for new
datasets and as curation grows.

## Dataset

`data/corpora/sampled_statements_rasmachine_v1.pkl` — a pickled list of INDRA
`Statement`s (91 statements / 100 evidences) sampled from rasmachine for human
curation. Gold is Ben Gyori's curations, resolved live from the public INDRA
curation endpoint and filtered to `source == 'indra_belief_rasmachine'`. Coverage
grows as curation continues; re-run the builder to pick up new curations.

## 1. Build — pkl → runner input + gold

```
PYTHONPATH=src .venv/bin/python scripts/build_rasmachine_eval.py \
    --pkl data/corpora/sampled_statements_rasmachine_v1.pkl \
    --source indra_belief_rasmachine
```

Writes (names derive from the pkl stem, `sampled_statements_` stripped):
- `data/corpora/rasmachine_v1_statements.json` — `stmts_to_json`, what the runner scores
- `data/benchmark/rasmachine_v1_gold.jsonl` — one row per curated evidence, joinable on `(matches_hash, source_hash)`; carries `gold` (binary verdict) + `tag` (specific category)
- `data/benchmark/rasmachine_v1_curations.jsonl` — raw pulled curations (cache; reuse with `--no-fetch`)

Gold rule (`indra_belief.curation`): an evidence is correct iff every curation
tag is `correct`; any dissent → incorrect. A typo'd `--source` fails loudly with
the sources actually present rather than writing empty gold.

## 2. Score — one run per model

Inference runs on the noot-1 gateway (`gemma-remote`, `medpsy-remote`). Run the
models **sequentially** to avoid GPU model-swap thrash. The monolithic scorer
defaults to the relation-nature variant.

```
PYTHONPATH=src .venv/bin/python scripts/run_rasmachine_monolithic.py \
    --input data/corpora/rasmachine_v1_statements.json \
    --model gemma-remote --workers 4 --no-export --row-error-policy record \
    --output data/results/rasmachine_v1_gemma.jsonl
# then --model medpsy-remote --output data/results/rasmachine_v1_medpsy.jsonl
```

All 100 evidences are scored; only the curated subset is measured (the rest are
predictions ready for when their gold lands). Output is resumable.

## 3. Compare — head-to-head metrics

```
PYTHONPATH=src .venv/bin/python scripts/eval_curation_compare.py \
    --gold data/benchmark/rasmachine_v1_gold.jsonl \
    --a data/results/rasmachine_v1_medpsy.jsonl --a-name MedPsy-4B \
    --b data/results/rasmachine_v1_gemma.jsonl  --b-name gemma-26B \
    --title "rasmachine sampled-statements v1" \
    --out data/results/rasmachine_v1_compare.md
```

Reports accuracy (+Wilson CI), error-detection P/R/F1 (positive class =
curator-flagged incorrect — the headline on imbalanced gold), ECE, per-tag and
per-stmt_type breakdowns, and a paired McNemar test. Joins on the
`(matches_hash, source_hash)` pair with a unique-source_hash fallback.

## Reuse / a new dataset

Drop a new statement pickle in `data/corpora/`, run stage 1 with `--pkl` and the
right `--source`, then stages 2–3 against the derived filenames. Tests in
`tests/test_rasmachine_eval.py` lock the join contract, the gold-row schema, the
representative-tag rule, and loud failure on a bad source.
