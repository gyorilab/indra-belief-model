# Consistency Reconciliation — Task Hypergraph

> **Status:** active (2026-07-15). Closes the doc/impl/results/methodology consistency audit
> (7-dimension map→do→review workflow + brutalist review). The audit verdict was
> **artifact-consistent, process-controls-soft**: the shipped hybrid-belief math is documented
> byte-faithfully across code/docs/decks/README, but the *status layer*, *anchor layer*, and
> *persistent memory* drifted after the `503d816` config-scoped hybrid ship, and two governance
> controls (the ship gate, the gate evidence) are softer than the docs claim. This graph fixes
> **every valid finding at its root**, not leaf-by-leaf.

## Root cause (what actually broke)

One event moved the ground under many surfaces: `2894fa0` reconciled the **calibration** hypergraph
and the decks to the "source-aware **hybrid** log-odds (not a pure Bayesian posterior)" framing, but
did **not** propagate to (a) the **learnings** hypergraph, (b) **persistent memory**, or (c) the
volatile **file:line anchors** that the refactor invalidated. Separately, two controls were written
as enforcement but implemented as report-only. So the roots are three classes, not nine leaves:

- **Propagation debt** — a reconcile pass that stopped at one of two map-of-maps + memory.
- **Anchor drift** — docs hardcode `file:line` / `schema_version` that the code moved (no guard).
- **Control theater** — a "gate" that always exits 0, on evidence absent from git.

## Meta-invariants (this reconciliation must obey the invariants it reconciles)

| ID | Constraint | REVIEW check |
|----|------------|--------------|
| **M-CALIB-EXTERNAL** | The fix touches **no** belief-math file: `noise_model.py`, `calibration_constants.py`, `statement_belief.py`. Reconciliation is doc/memory/tooling only. | `git diff` names none of the three modules. |
| **M-PARITY** | No shipped number moves. `score()` / belief output is untouched; `tests/test_soft_belief.py` byte-identity still green. | Full suite passes; no edit under `src/indra_belief/` scorer/belief paths. |
| **M-GOLD-IMMUTABLE** | No `*_gold.jsonl` / `eval_curation_v1.jsonl` / `holdout*.jsonl` line changes. | `git diff` touches no gold. |
| **M-MINIMAL** | Each node makes the *smallest* edit that resolves its finding. No composition overhaul, no drive-by rewrites. | REVIEW rejects any out-of-scope hunk. |
| **M-TRUTH** | Every replacement value (line number, schema version, commit hash, status) is re-derived from the live tree, not copied from this doc. | REVIEW re-confirms each new anchor independently. |

## Status legend

`[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[R]` in review

## Node table

| Node | Fixes (audit finding) | Targets (file-disjoint for safe parallelism) | Root, not leaf |
|------|----------------------|----------------------------------------------|----------------|
| **R1** | #1 stale scalar identity; #8 stale flags + oversize | `~/.claude/.../memory/MEMORY.md`, `project_clean_error_model.md` (+ stale-flag entries) | Reframe the *current-production* memory fact + record the v8 ship; trim index under the 24.4 KB load limit so newest entries stop truncating. |
| **R2** | #2 self-contradiction; #3 I-CALIB-EXTERNAL unreconciled + stale `:215-221` sub-claim | `research/learnings_task_hypergraph.md` | Make the "live" status narrative match the table + git reality; reconcile the invariant text to permit X1-owned belief-math edits (which already shipped). |
| **R3** | #4b written-G2≠shipped-legs; #6 dead `composed_scorer.py` anchors; #7 `:299-326`, `schema_version 4` | `research/calibration_task_hypergraph.md`, `research/belief_instrument_task_graph.md` | Promote the AUROC-for-resolution "proposed refinement" into the actual G2 definition (it shipped); repoint every dead/moved anchor to the live symbol location. |
| **R4** | #4a gate reports but never enforces | `scripts/calibration_ship_gate.py` | `main()` returns non-zero on PENDING (missing evidence) or any evaluated-reader FAIL; the disable-a-failed-reader decision stays an explicit `_PROFILE_META` act, not a silent green. |
| **R5** | #5 gate evidence ungit-tracked | `.gitignore` + four local/external decision artifacts | Put both matched-holdout and external Bedrock decisions in git; each embeds exact gold/run hashes while raw multi-MB runs stay ignored. |
| **R6** | #6/#7 recurrence prevention | `scripts/check_doc_anchors.py` + `tests/test_doc_anchors.py` | Fail CI on missing Python/TypeScript/Svelte paths or volatile numeric line citations in live hypergraphs (archived and explicitly historical blocks remain exempt). **Widened since:** `DOCS` is now a `research/*.md` glob rather than a three-doc list, and the guard also fails on a dead DOTTED symbol citation — the two extensions the doc-drift audit showed were needed, because a fixed list made "unguarded" the default for a new document. |
| **R7** | #9 stale viewer comment | `viewer/src/lib/data/types.ts` | Delete the "Untyped here until E11" comment; the fields are already typed 40 lines below (E11 shipped). |

## Hyperedges (dependencies & barriers)

- **R6 ⟸ {R2, R3}** — the guard is authored and run *after* the anchors are fixed, so it passes on the
  repaired docs and demonstrably would have flagged `composed_scorer.py`.
- **VERIFY-BAR** — a terminal barrier after all DO+REVIEW: full `pytest`, the new anchor guard, and a
  scoped `git diff` audit confirming **only** the node-owned files changed (M-MINIMAL / M-CALIB-EXTERNAL).
- **File-disjoint fan-out** — R1,R2,R3,R4,R5,R7 own non-overlapping paths → run in parallel with no
  worktree isolation; R6 serializes behind R2/R3; VERIFY is the join.

## Execution

Map = this document. Do+Review = one Workflow: each node runs `do → adversarial-review` independently;
R6 after R2/R3; then VERIFY-BAR. A node whose REVIEW = FAIL is re-done before the barrier.
