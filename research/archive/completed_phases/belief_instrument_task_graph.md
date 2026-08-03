# Belief Instrument Task Hypergraph

> _Naming note: this file keeps the older `_task_graph` filename suffix while its
> sibling live docs use `_task_hypergraph` (and this doc's own title reads
> "Hypergraph"). The legacy spelling is retained deliberately — the live-doc
> anchor guard and several sibling research docs pin this exact basename, so a
> rename would reach into guarded tooling that is out of scope for a docs-only
> pass. The divergence is cosmetic; the file's role is unchanged._

Status: drafted + first execution pass landed 2026-05-10. T1 fully complete (E0 belief primitive, E1 findings, E2 run-narrative, E3 probe attribution, E4 residual distribution). T2 6/6 complete (focus card, findings strip, validity-as-section, run feed, `/runs/[run_id]` page, demolitions — the run page **shipped**; the earlier "deferred until multiple runs" note is retired). **T6 — calibration surfaces added 2026-06-17** (C4 run-on-its-own + C5 run-comparison; new substrate edge E5 render primitives; consumes the calibration arc's per-run `metrics.json`; see `research/archive/completed_phases/calibration_task_hypergraph.md`). T3 perceptual instruments landed inline with T2 (probe bars, residual sparkline, bias-axis cleanup, verdict pillbar). T4 wayfinding 3/4 (deep-link focus + relevance-sort matrix already work; deep-dive reframed; keyboard nav deferred). T5 gates pending. Successor to `rasmachine_task_graph.md` (closed). The R-phase shipped a *working* interface; this hypergraph reframes it as a *truthful* one.

## Frame

The current dashboard is a database admin panel wearing the costume of a science instrument. It opens with five row-counts, two `GROUP BY` catalogs, a cost calculator, and validity buried as one panel of six. **No belief is visible above the fold.** No biology. No evidence. No probe reasoning. The product output of the entire pipeline is two clicks deep and unreachable from the home page.

The radical inversion: **belief primary, bookkeeping in service.** A scientist lands and sees one assertion — its biology text, its score, INDRA's prior, the delta, the probe contributions, an evidence excerpt — and a "look here" feed that surfaces the most interesting movements. Validity tells a story: bias direction, residual shape, where we fail by stratum, what changed vs the previous run. Counts, source dumps, and indra_type breakdowns survive only as a footer line.

Perceptual engineering is woven into every node, not appended.

---

## Inversions (what we tear down or demote)

- **D1** — delete the always-on counts row above the fold. Demote to a single footer line.
- **D2** — delete standalone `truth_sets`, `source_dump_id`, `indra_type` panels. Fold into validity stratification or footer.
- **D3** — move `next-run cost projection` off the dashboard. Collapse into a pre-flight expander, closed by default.
- **D4** — replace the live-dot ("something changed") as the only motion signal with a run feed (sentences, dated).
- **D5** — replace `recent score_run` six-column table with a run-feed sentence list; the table view moves to `/runs`.

---

## Substrate edges (foundation capabilities — each unlocks ≥2 phase nodes)

### E0 — Belief primitive

A single Svelte component rendering `<biology text, our_score, indra_score, Δ, probe contributions[], evidence excerpts[], why_this_one?>`. Reused as: focus card on dashboard, header on deep-dive, finding-row in feeds, regression fixture.

Perceptual contract: score is the largest single fact (~1.6rem serif); probe contributions are a 4-row monospace bar block; evidence text is serif italic; metadata is small mono. Compact mode for feed rows shrinks the score, drops evidence, keeps probes as a 4-cell `▆░█▅` micro-row.

### E1 — Findings query layer

Server-side ranked queries returning typed rows. Five ranks initially:

- `biggest_disagreement` — top-K `|our_belief − indra_belief|`
- `highest_probe_split` — top-K stdev across probe contributions
- `verdict_regressions` — set diff on verdict between run N and run N−1 (correct→incorrect)
- `verdict_recoveries` — incorrect→correct between runs
- `low_confidence_high_stakes` — `our_belief ∈ [0.4, 0.6]` AND `n_evidence ≥ 3`

Each row carries a one-line `why` ("Δ +0.41 vs INDRA, n_ev=3, single-direction").

### E2 — Run-narrative layer

`diff_runs(prev_run_id, run_id)` → `{per_stmt_deltas, stratum_deltas, verdict_crossings}`. Surfaces a single human sentence ("17 statements moved verdict; 12 toward `correct`, 5 toward `incorrect`. MAE +0.012, bias unchanged.").

### E3 — Probe-contribution model

For each `(run_id, stmt_hash)`: read scorer_step chain, compute step-wise belief delta, identify the **decisive probe** (the one whose removal would flip the aggregate verdict). Output: `{probe: name, contribution: signed float, normalized: [0,1], decisive: bool}[]`.

### E4 — Residual distribution view

Histogram of `(our_belief − indra_belief)` over a run, 11 bins on `[-1, +1]`. Two render modes: braille block string `▁▁▂▃▆█▆▃▂▁▁` for inline use; SVG for the validity main view (with stratum overlay).

### E5 — Calibration render primitives

Three reusable Svelte components for the calibration arc (`research/archive/completed_phases/calibration_task_hypergraph.md`):
- `ReliabilityDiagram.svelte` — `BINS_8` bins; x = mean predicted score, y = empirical correct-rate; diagonal = perfect; marks sized by bin-n so sparse bins don't lie (mirrors the `n_total ≥ 30` validity guard). This is the curve the viewer does not have today.
- `BrierBar.svelte` — one stacked bar, Murphy decomposition `reliability − resolution + uncertainty`; the narrative spine (low reliability good, high resolution good, uncertainty the irreducible floor) makes the calibration arc's D8 story — a monotone post-hoc map lowers reliability while collapsing resolution — *visible*.
- `ConfusionMosaic.svelte` — 2×2 TP/FP/FN/TN, area ∝ count; reuses the `/compare` mosaic geometry, single-palette for the run-on-its-own register.

Reused across C4 (run-on-its-own) and C5 (run-comparison) — ≥2 nodes. **Reads** the calibration hypergraph's per-run `metrics.json` (defined *there* as that doc's substrate edge E5 — the *data* export seam; **this** E5 is the *render* layer); never recomputes. Render modes mirror E4: SVG for main views, compact glyph for inline/feed use.

---

## Phase T1 — foundations

| node  | deliverable | perceptual contract | depends |
|-------|-------------|---------------------|---------|
| T1.1  | E0 belief primitive component | score 1.6rem; probes as 4 monospace bars; evidence serif italic; one-line `why` | T1.4 |
| T1.2  | E1 findings DB layer | each row has machine-readable `why_kind` + human `why_text` | — |
| T1.3  | E2 run-narrative DB layer | `summary_sentence` field returned alongside structured deltas | — |
| T1.4  | E3 probe-contribution model | signed magnitude + normalized width + `decisive` flag per probe | — |
| T1.5  | E4 residual distribution | braille string AND SVG path renderable from same bin array | — |

**Order:** T1.4 + T1.1 first (entangled); T1.2 / T1.3 / T1.5 in parallel.

---

## Phase T2 — dashboard reframe

| node  | deliverable | perceptual contract | depends |
|-------|-------------|---------------------|---------|
| T2.1  | focus card replaces above-the-fold | E0 in full-mode at top of `/`; one statement, picked deterministically per (run_id, session) | T1.1, T1.4 |
| T2.2  | findings strip below focus | three lanes (disagreements / probe-splits / regressions), top-3 each, click → swap focus | T1.1, T1.2 |
| T2.3  | validity becomes a real section | calibration sentence + residual SVG + per-stratum sorted table (top 5 by MAE desc); run-delta sentence ("MAE +0.012 vs prev run; 17 verdicts moved") | T1.3, T1.5 |
| T2.4  | run feed replaces recent_runs table | chronological sentence list ("d65a629b succeeded · 1,847 stmts · $4.32 · MAE 0.187 · ▼0.012 vs prev"); click → `/runs/[id]` | T1.3 |
| T2.5  | `/runs/[run_id]` page | run-narrative view: summary sentence, per-statement delta table sortable by `|Δ|`, stratum delta table, verdict-crossing list | T1.3 |
| T2.6  | demolitions per D1–D5 | counts/truth_sets/source_dumps/indra_type collapse to one footer row each (or fold into stratum); cost panel becomes pre-flight expander | — |

**Order:** T2.1 first (most-load-bearing visual change). T2.6 in parallel (independent). T2.3 once T1.5 lands. T2.4 + T2.5 share T1.3.

---

## Phase T3 — perceptual instruments

| node  | deliverable | perceptual contract | depends |
|-------|-------------|---------------------|---------|
| T3.1  | probe contribution micro-bars | 4-row block: `███▌░░░ parse_claim +0.21`; green push-toward-correct, accent push-away | T1.4 |
| T3.2  | residual sparkline (inline) | 11-char braille; appears in validity sentence and each stratum row; SVG fallback | T1.5 |
| T3.3  | bias-axis consolidation | one mini-axis at validity head; per-stratum row uses `▲0.08` / `▼0.03` glyphs instead of own axis | T2.3 |
| T3.4  | verdict pillbar | one horizontal bar, three colors, widths proportional; replaces `correct N · incorrect M · abstain K` strip | — |

---

## Phase T4 — wayfinding

| node  | deliverable | perceptual contract | depends |
|-------|-------------|---------------------|---------|
| T4.1  | keyboard nav | `j/k` step through findings; `/` focus search; `?` help; `ESC` to dashboard; arrow keys for run feed | T2.1, T2.2 |
| T4.2  | deep-linkable focus | `/?focus=<stmt_hash>` hydrates the focus card; back button works; sharable URL | T2.1 |
| T4.3  | `/statements` ranks by relevance | default sort = `|Δ vs INDRA|` desc; column headers click to re-sort; current pagination preserved | T1.2 |
| T4.4  | `/statements/[hash]` deep-dive reframe | E0 belief primitive at top; full probe trace below; evidence list; truth labels last | T1.1, T1.4 |

---

## Phase T5 — validation gates

| node  | gate | how |
|-------|------|-----|
| T5.1  | brutalist roast at end of T2 | invoke `mcp__brutalist__roast` against the redesigned dashboard URL; address every blocker |
| T5.2  | perceptual eye-path audit | for each section: state the question, the entry point, the next click. No section may fail to answer all three |
| T5.3  | empty-corpus coherence | render `/` on a fresh DB (no statements). Belief primitive shows placeholder; findings shows "no runs yet"; validity shows "—" |
| T5.4  | single-statement coherence | render `/` with one statement. No ranking; the one statement IS the focus card |
| T5.5  | large-corpus performance | render `/` against ≥10k stmts in <200ms server-time + <300ms paint |
| T5.6  | regression: existing tests pass | the 245 corpus tests + viewer e2e tests still green |
| T5.7  | calibration render parity (= calibration arc **G4**) | on a real run, served ECE/Brier/diagram == the persisted `metrics.json` byte-exact (no served-vs-persisted drift); single-palette register; empty-metrics run renders a named empty (P6), not a crash |
| T5.8  | comparison parity (= calibration arc **G5v**) | viewer ΔECE/ΔBrier == the C1/C2 held-out script outputs for the same run pair; drill skeleton unchanged; comparison register (A/B/gold hues) preserved |

---

## Phase T6 — calibration surfaces (C4 / C5)

The calibration math arc (C0–C3 in `research/archive/completed_phases/calibration_task_hypergraph.md`) emits products to disk; T6 renders them in the live instrument. Same device as the rest of this doc — **route = register**: `/runs/[run_id]` is the single-run register, `/compare` the comparison register. Node IDs keep the calibration arc's `C#`/`G#` vocabulary so the two docs cross-reference cleanly.

| node | deliverable | perceptual contract | depends |
|------|-------------|---------------------|---------|
| C4   | run-on-its-own calibration section on `/runs/[run_id]`, **above** the residual histogram (residual stays — it answers a different question, P-doctrine forbids deleting a true signal) | P1 ECE as the headline number; reliability diagram + Brier stacked bar; per-run confusion mosaic; per-stratum ECE strip (reuses the T2.5 stratum table) retires the `unavailable` apology; **one reactive lever** = Tier toggle `?tier=ev\|stmt` (Tier-1 per-evidence + Tier-2 per-statement as two labeled, stacked diagrams, never merged); P5 delta-vs-prev; P6 named-empty; P7 single palette | E5, cal:`metrics.json`, T2.5 |
| C5   | run-comparison calibration mode `?mode=calib` on `/compare` (mirror `?mode=gold`) — a **mode, not a second matrix** | L0 anatomy swaps the verdict 2×2 for overlaid reliability curves (A `--a-hue` / B `--b-hue` + diagonal) + ΔECE/ΔBrier scalars; three-way baseline (hard / parametric / soft) = three series tied to the calibration **G2** ship gate; L1 stratify → ΔECE per stratum; L2 cohort → reads sorted by distance-to-diagonal; L3 reasoning unchanged; drill skeleton reused (only the L0 payload swaps); comparison register preserved | E5, cal:`metrics.json`, `/compare` L0–L3 |

**Gates:** T5.7 (= G4) for C4, T5.8 (= G5v) for C5.

**Decision-gating:** C4/C5 consume the calibration math arc's products — a NO-GO at that arc's **G0** (raw-belief AUPRC has no headroom) means there is nothing to render and T6 never fires. Zero viewer work is spent before the math earns it.

**Order:** C4 first (single-run is the simpler register and unblocks reading any one run's calibration); C5 once C4's E5 primitives are proven.

---

## Hyperedges (cross-cutting groupings)

- **H_belief** = {T1.1, T2.1, T2.2, T4.4, T3.1} — every site that renders a statement-with-score
- **H_findings** = {T1.2, T2.2, T4.1, T4.3} — every site that ranks statements
- **H_narrative** = {T1.3, T2.3, T2.4, T2.5} — every site that shows run-over-run change
- **H_reasoning** = {T1.4, T2.1, T4.4, T3.1} — every site that shows pipeline justification
- **H_distribution** = {T1.5, T2.3, T3.2} — every site that shows residual shape
- **H_calibration** = {E5, C4, C5, T2.3, T2.5} — every site that renders reliability / ECE / Brier (consumes the calibration arc's `metrics.json`)
- **H_demolish** = {D1, D2, D3, D4, D5, T2.6} — what we remove or fold

A change to E0 propagates through H_belief; a change to E1 through H_findings; a change to the bias glyph through H_distribution. Hyperedges name the contracts that must remain consistent.

---

## Perceptual contracts (apply to every node, not just T3)

- **P1 — Hierarchy of attention.** Largest pixel area = most important fact. Belief score > probe magnitudes > evidence excerpts > metadata. Counts < everything.
- **P2 — Glanceability.** ≤200ms read for "is the system working?" — one sentence answers it. ≤500ms for "what should I look at?" — top finding's row answers it.
- **P3 — Direct manipulation.** Click a probe → filter to where it disagreed. Click a stratum → drill in. Click a finding → swap focus. No hidden state, no modes.
- **P4 — Data:ink ratio.** Every glyph load-bearing. Decorative borders, gradients, shadows, logos: forbidden.
- **P5 — Delta primacy.** Every metric reports its delta vs the prior run. Lone scalars are forbidden — `MAE 0.187` becomes `MAE 0.187 ▲0.012 vs prev`.
- **P6 — Named empties.** No panel silently shows zeros. `—` plus a one-line `unavailable_reason` whenever a value is absent.
- **P7 — Brutalist palette.** Existing 6 vars only: `--paper, --ink, --ink-muted, --ink-faint, --rule, --accent`. Plus one forest-green for verdict-correct (already in use).
- **P8 — Type discipline.** Monospace for numbers and identifiers; serif for prose and biology text; sans never.

---

## Acceptance criteria

- **A1** — A new visitor lands and sees a literal biology assertion + its score within the first 4cm of viewport.
- **A2** — Within 5 seconds, no clicks, the visitor can answer "is the system working overall?" (validity sentence) and "what should I look at?" (findings strip).
- **A3** — Clicking a finding swaps the focus card without page navigation; URL updates; back button restores prior focus.
- **A4** — Two adjacent runs show their delta as one sentence + a sortable per-statement diff.
- **A5** — Empty / 1-statement / 10k-statement corpora all render coherently within performance budget.
- **A6** — Brutalist roast finds zero "decoration without signal" violations.
- **A7** — All 245 existing tests pass; new tests cover E0, E1, E2, E3, E4 contracts.

---

## Execution order (the actual path through the hypergraph)

1. **T1.4** (probe attribution) — earliest because T1.1 needs its data shape
2. **T1.1** (belief primitive) — the perceptual atom; everything composes from this
3. **T1.2** (findings) + **T1.3** (run-narrative) + **T1.5** (residuals) — parallel
4. **T2.6** (demolitions) — independent; clears space for new content
5. **T2.1** (focus card) → **T2.2** (findings strip) — first visible reframe
6. **T2.3** (validity section) — once T1.5 ready
7. **T2.4** (run feed) → **T2.5** (run page) — once T1.3 ready
8. **T3.1–T3.4** — interleaved with T2 panels as each panel ships
9. **T4.1–T4.4** — wayfinding once content is stable
10. **T5.1–T5.6** — gates throughout, hardest at the end
11. **C4** (run-on-its-own calibration) → **C5** (run-comparison mode) — only after the calibration math arc clears its **G0** go/no-go and E5 (`metrics.json`) ships products; gated by T5.7 / T5.8
