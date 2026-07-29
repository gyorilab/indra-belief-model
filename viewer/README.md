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

The statement-level comparison at `/frontier?view=belief` has a separate,
optional input at `../data/results/indra_belief_comparison_metrics.json`
(override with `INDRA_BELIEF_COMPARISON_METRICS`). Its strict contract lives in
`src/lib/data/belief-comparison.ts`. A missing artifact, mixed prediction units
or gold rules, mismatched substrate/gold/evaluation digests, or incomplete arm
coverage leaves the affected panel visibly gated; the viewer never derives these
metrics from the evidence-pair exports. The artifact names the historical fold-mean
trapezoidal PR area and pooled average precision separately, and validates the
paired intervals, frozen calibration bins, exact-decimal retry-inclusive costs,
evidence-identity attempt accounting, and Pareto audit before rendering them. A
paper panel may carry a confirmatory decision, or explicitly carry none;
the current contract renders the latter as descriptive only and never
synthesizes a parity or superiority claim. Relevant scorer arms that are still
excluded retain their missing artifact and provenance, so absence is never
mistaken for a result.

The checked-in artifact is currently a 200-resample diagnostic with the
canonical INDRA arms and historical Gemma 4 E2B bundle. Gemma 4 26B, Gemma 4
31B, and GLM-5 remain excluded until their concurrent Bedrock primaries become
complete canonical bundles. Publication requires all four LLM bundles, 10,000
paired resamples, and two complete independent blinded human reviews; every
reviewer disagreement must also be resolved by a third human. Until those gates
pass, this route is a diagnostic view rather than a released parity or
superiority result.

Performance always covers the complete validated panel. LLM cost is eligible
for direct Pareto comparison only under the shared structured pricing identity:
AWS Bedrock on-demand inference in `us-east-1`, requested tier `default`,
resolved tier Standard, and frozen USD-per-million input/output rates. Cost
accounting is retry-inclusive: provider-measured spend remains measured, while
any incomplete token accounting exposes its basis and measured lower versus
conservative upper endpoint. The all-source and reader views are projections of
the same model run: reader cost is the exact observed five-reader execution
subset, not an extrapolation or counterfactual run, and the two panel totals are
explicitly non-additive. Arms without comparable cost remain named and
unavailable; the viewer never substitutes zero.

Below the direct panel, a separate checksum-pinned adapter renders all 59 method
summaries published with the 2023 paper. This is literature context, not another
frontier: the rows span different eligible sets, expose fold population SD rather
than confidence intervals, and provide neither statement-level predictions nor
comparable costs. The viewer therefore never mixes them into direct paired
deltas, parity claims, or the cost Pareto frontier.

From the repository root, assemble and compute the artifact from the explicit
comparison inputs:

```bash
PYTHONPATH=src python -m indra_belief.comparison materialize --force
PYTHONPATH=src python -m indra_belief.comparison metrics --force
```

The separate frozen 403-pair curation set is a transfer diagnostic, not a
population-representative primary panel: pair identities are unique, but the
historical completion sequence and inclusion probabilities are not proven. A
resolved-only subset must declare `resolved_only_sensitivity`; the viewer marks
it selection-biased and does not present it as a representative population
result.

## Run it

```bash
cd viewer
npm install
npm run dev        # vite dev server, http://localhost:5174
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
| `/frontier` | Evidence-pair frontier and a separate shareable `?view=belief` assembled-statement comparison |
| `/adjudicate` | Blinded human verdict, with INDRA curation revealed as a third judge |
| `/review` | Evidence review surface |
| `/curate` | Draw without replacement from the tracked CoGEx evidence-row reservoir, materialize through live INDRA, and submit under the curator's own JWT |
| `/login`, `/logout` | Per-curator auth against INDRA; the whole viewer is gated (hooks.server.ts) |

## Layout

```
src/
  lib/
    data/        # pure-JS data layer over the JSONL exports
      runs.ts        # export-dir discovery + RunMeta
      store.ts       # mtime-cached in-memory load + index of a run's exports
      queries.ts     # every payload the UI needs (overview, validity, matrix, compare, …)
      belief-comparison.ts # strict frozen statement-belief artifact contract
      types.ts       # the JSONL-derived data model
      curation.ts    # INDRA curations as a gold lane (db.indra.bio)
      adjudicate.ts  # adjudication payloads
      review.ts      # review payloads
    server/      # server-only ($lib/server — never bundled to the browser)
      datasets.ts              # tracked curation frames and selector metadata
      curation-history.ts      # lossless 64-bit history/pool identity parsing
      curation-draw-ledger.ts  # persistent atomic no-replacement reservations
      indra.ts                 # live materialization, history, and submit transport
      belief-comparison.ts     # fail-closed optional artifact loader
      session.ts               # sealed curator session and INDRA endpoint config
    components/   # BeliefPrimitive, BeliefComparison, BeliefRuler, SiteNav, Validity
    format.ts     # verdict rendering, sentence + cue formatting
    residuals.ts  # residual histogram bucketing
  routes/         # SvelteKit routes (see table above)
scripts/
  curation_gold_json.mjs   # bake INDRA curations into a run's gold lane
```

Fields the monolithic export cannot provide (probe traces, gold-F1 strata) are
surfaced as explicit `unavailable` markers, never faked.

The tracked representative manifest is available in a clean checkout. The
18 MB `rasmachine` curation frame is a reproducible local artifact and remains
ignored; build it with `PYTHONPATH=src python scripts/build_curate_pool.py` from
the repository root. `/curate` disables that lane with the same provisioning
instruction when the file is absent.

## Authentication

The whole viewer is gated. A curator signs in with their **own INDRA account**
(`/login` → proxies `db.indra.bio/login`); the returned JWT is sealed into an
HMAC-signed httpOnly `vsession` cookie (`$lib/server/session.ts`), and curations
are submitted under that curator's JWT — there is **no shared api_key on the write
path** and no self-declared curator. `hooks.server.ts` reads the session into
`locals` and redirects unauthenticated route requests to `/login`.

Requires `VIEWER_SESSION_SECRET` (≥32 chars) in the repo-root `.env` — auth
fails closed without it. Cookie `secure` tracks the request protocol, so it works
on local http and is `Secure` under https.

### Deployment hardening (when this leaves localhost)

- **Serve a build, not `vite dev`.** SvelteKit's built-in CSRF Origin check is
  compiled out in dev; we additionally enforce same-origin on `/login`+`/logout`
  in `hooks.server.ts` (all modes), but production should run a real build.
- **Set `ORIGIN` (adapter-node)** so cookie `Secure` + CSRF use the public origin,
  and **`ADDRESS_HEADER`** so the login rate-limit keys on the real client IP.
- **Rate limit is per-process** (`session.ts`); a multi-instance deploy needs a
  shared store (e.g. Redis) keyed on IP+email.
- **Curation draws require shared persistence.** Set `CURATION_DRAW_LEDGER_DIR`
  to a persistent filesystem shared by every viewer instance, with reliable
  exclusive-create semantics, and acknowledge it with
  `CURATION_DRAW_LEDGER_SHARED=1`. Production sampling fails closed without
  both settings. The ledger is the no-replacement boundary for skipped cards,
  refreshes, and concurrent tabs; separate or ephemeral instance disks cannot
  provide that guarantee.
- **Submission claims fail closed after a process crash.** INDRA exposes no
  idempotency/fencing key, so the viewer never reclaims an in-flight `.claim` by
  age. After checking shared INDRA history, an operator may remove a genuinely
  orphaned claim; automatic TTL recovery could duplicate a remote submission.
- **Rename the cookie to `__Host-vsession`** (Secure, Path=/, no Domain) once
  served from a shared parent domain.
- **Logout is not revocation:** it clears the local cookie, but the sealed JWT is
  a bearer token valid at INDRA until its own expiry. A true kill-switch needs an
  INDRA-side revoke. (Deliberate bearer trade-off; documented in `session.ts`.)
