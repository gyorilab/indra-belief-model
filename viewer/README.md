# viewer

A read-only SvelteKit dashboard for browsing INDRA belief-scoring runs. It is a
pure in-memory projection over the per-run JSONL exports — no database, no
server-side scoring. Scoring happens in the Python pipeline (see the repo
[README](../README.md)); the viewer only reads what that pipeline exports.

## Data source

Each run is an export directory under `../data/exports/<run>/`:

| File | Contents |
|------|----------|
| `export_meta.json` | Run metadata: model, counts, status, observed LLM cost |
| `per_statement.json` | One rollup per statement (scores, verdict mix, buckets) |
| `per_evidence.jsonl` | One row per `(statement, evidence)`: verdict, score, reasoning, cost |

The data dir is resolved relative to the viewer's working directory
(`viewer/` → `../data`), so run the dev server from `viewer/`. Runs are
discovered by scanning `../data/exports/` for directories that contain an
`export_meta.json`.

## Run it

```bash
cd viewer
npm install
npm run dev        # vite dev server, http://localhost:5173
npm run check      # svelte-check (types + a11y), expected 0 errors / 0 warnings
npm run build      # production build
npm run preview    # serve the production build
```

## Routes

| Route | View |
|-------|------|
| `/` | Dashboard: latest-run focus, findings, validity, runs feed |
| `/runs`, `/runs/[run_id]` | Run feed + per-run detail |
| `/statements`, `/statements/[stmt_hash]` | Statement matrix + per-statement deep-dive |
| `/compare` | Progressive model-vs-model dig (confusion matrix → stratify → cohort → side-by-side reasoning) |
| `/adjudicate` | Blinded human verdict, with INDRA curation revealed as a third judge |
| `/review` | Evidence review surface |

## Layout

```
src/
  lib/
    data/        # pure-JS data layer over the JSONL exports
      runs.ts        # export-dir discovery + RunMeta
      store.ts       # mtime-cached in-memory load + index of a run's exports
      queries.ts     # every payload the UI needs (overview, validity, matrix, compare, …)
      types.ts       # the JSONL-derived data model
      curation.ts    # INDRA curations as a gold lane (db.indra.bio)
      adjudicate.ts  # adjudication payloads
      review.ts      # review payloads
    components/   # BeliefPrimitive, BeliefRuler, SiteNav, Validity
    format.ts     # verdict rendering, sentence + cue formatting
    residuals.ts  # residual histogram bucketing
  routes/         # SvelteKit routes (see table above)
scripts/
  curation_gold_json.mjs   # bake INDRA curations into a run's gold lane
```

Fields the monolithic export cannot provide (probe traces, gold-F1 strata) are
surfaced as explicit `unavailable` markers, never faked.
