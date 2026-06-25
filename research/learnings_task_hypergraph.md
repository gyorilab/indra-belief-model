# Learnings Task Hypergraph

> **Status:** scaffold (2026-06-23). Living artifact — extend it; do not treat node list as closed.
> **Provenance:** derived from the frontier-research + design-critique synthesis of 2026-06-23
> (monolithic-scorer design verdict + adversarially-verified context-engineering / transformer-scaling brief),
> with three high-severity defects (H1/H2/H3) confirmed at runtime against the live code.
> **This document is the MAP layer for the entire program.** Each node below descends into its own
> map→do→review workflow; this file is the map of maps.

---

## 0. Reading guide

A **hypergraph**, not a list: nodes are task units, but the load-bearing structure is the
**hyperedges** — relations that bind *sets* of nodes at once (invariants, barriers, gates, handoffs).
A pairwise "A before B" dependency is the trivial case; the interesting constraints connect 3–5 nodes.

- **§1** — the map→do→review contract every node obeys, and how it instantiates as a Workflow.
- **§2** — the five invariants (hyperedges that constrain many nodes simultaneously).
- **§3** — the hypergraph at a glance: clusters, node table, hyperedges, dependency sketch.
- **§4** — per-node engineering specification (intent / MAP / DO / REVIEW / artifacts / gate / risk / targets).
- **§5** — extension protocol (how we grow this graph).
- **§6** — critical path and suggested ordering.

**Hard scope constraints (2026-06-23 directive):**
- ❌ No parallel-voting / sample-K-aggregate lever. Removed from scope.
- 🔌 Calibration (verdict-grain remedy, soft-guard, ECE/AUROC/Brier fitting) is **owned by another agent.**
  We model it as an external node (`X1`) with a defined interface; we **do not** modify the belief math
  (`noise_model.py`, `calibration_constants.py`, `statement_belief.py` rollup).
- 🔒 Gold labels are **immutable.** No node edits an existing gold tag. New gold is additive, separate-file,
  contamination-checked.

---

## 1. Global execution protocol — map → do → review

Every node is realized as a **single Workflow** with three ordered phases. The phases are not optional and
not reorderable; this discipline is itself part of the hypergraph spec.

### MAP (understand — no mutation)
- Read the *exact* code/artifacts named in the node's `targets`. Reproduce the current behavior; do not trust
  this document's line numbers without re-confirming them (they drift).
- Produce a **change-spec**: precisely what will change, the expected effect, the acceptance gate, the
  invariant(s) that bind it, and the rollback. Output is structured (schema'd), not prose.
- A MAP that discovers the premise is wrong **terminates the node** with a finding (do not proceed to DO to
  "save the work"). A killed node is a successful MAP.

### DO (implement)
- Apply the change in **isolation** (`isolation: 'worktree'`) whenever the node mutates files that another
  in-flight node could also touch, or whenever a re-run/build is involved.
- Produce the concrete artifact: code edit + test, or a re-run result + report. Nothing else.
- DO never decides whether it succeeded — that is REVIEW's job.

### REVIEW (adversarially verify)
- An **independent** agent (fresh context) checks the artifact against the MAP change-spec and the bound
  invariant(s). Default-skeptical: assume the change is wrong, incomplete, or violates parity until proven.
- Emits a verdict `{pass | fail(reason) | pass-with-followup(node)}`. A `fail` loops back to DO with the
  reason; a `pass-with-followup` spawns a new node into the graph (see §5).
- For nodes that back shipped numbers, REVIEW **must** execute the parity/golden gate, not merely read it.

### Workflow shape
```js
// Single-unit node:
pipeline([unit],
  u  => agent(MAP_PROMPT(u),    { phase:'Map',    schema: SPEC }),
  s  => agent(DO_PROMPT(s),     { phase:'Do',     isolation:'worktree' }),  // worktree iff mutating
  a  => agent(REVIEW_PROMPT(a), { phase:'Review', schema: VERDICT, effort:'high' }))

// Multi-unit node (e.g. several files, several alias residuals): same three stages, pipeline fans over units,
// each unit flows map→do→review independently (no barrier between stages).
```
Each node's `§4` entry states the unit set, the per-phase agent count, the schema of the MAP spec and the
REVIEW verdict, and whether DO needs a worktree.

---

## 2. Invariants (hyperedges over many nodes)

These are the "hyper" edges: each constrains a *set* of nodes, and a node may sit under several.

| Id | Invariant | Binds | Enforced by |
|----|-----------|-------|-------------|
| **I-PARITY** | Any change that backs a shipped number must reproduce today's `disconfirm_relnature` binding **byte-for-byte** unless the node's explicit purpose is to move it (only A5's re-run is allowed to move it). | C1, C2, C3, C4, A1, D1, D2 | REVIEW runs a golden-output diff of `score()` over a frozen fixture set before/after; non-zero diff on a parity-gated node = `fail`. |
| **I-GOLD-IMMUTABLE** | No node mutates an existing gold tag. New gold is a new file, balance-by-construction, run through `check_contamination.py`. | A5, B1, E1, E2 | REVIEW asserts `git diff` touches no `*_gold.jsonl` / `eval_curation_v1.jsonl` / `holdout*.jsonl` lines except additions of new files. |
| **I-CALIB-EXTERNAL** | Calibration is owned externally. Our nodes may **read** verdict/score outputs and **expose** interfaces, but must not edit `noise_model.py`, `calibration_constants.py`, or the `statement_belief` belief/rollup math. | A4, A5, B1, X1 | REVIEW asserts no diff in the three calibration-owned modules; coordination via `X1` interface contract. |
| **I-DETERMINISM** | Determinism = INPUT, never OUTPUT. No node may add an output-side verdict override. Substrate emits context/provenance into the prompt; only the two existing Tier-1 hard rejects flip a verdict in code. | A3, D1, D2, C4 | REVIEW greps the diff for any new write to `result["verdict"]` outside the model-derived path; presence = `fail`. |
| **I-MEASURE-CLEAN** | No headroom/lever conclusion is valid until the substrate is repaired and re-measured. The H1→H2→re-run barrier gates every lever node. | A5 → {D1, D2, D3, E2} | The lever nodes are **blocked** in the graph until A5's REVIEW = `pass`. Drawing a lever conclusion pre-A5 = protocol violation. |

---

## 3. The hypergraph at a glance

### Clusters
- **A — Substrate Integrity** (foundation; the prime mover): A1 A2 A3 A4 A5
- **B — Measurement & Deployment Truth**: B1 B2
- **C — Architecture Hygiene** (parity-gated; parallel track): C1 C2 C3 C4
- **D — Substrate-Input Grounding** (research-informed levers; post-A5): D1 D2 D3
- **E — Fresh Evidence & Decisive Measurement** (additive gold; validates the two unvalidated fixed points): E1 E2 **E3 E4**
- **X — External** (handoff only): X1 (calibration, other agent)

> **Completeness-audit extension (2026-06-23) — see §3.5.** The original A–E/X set is a *substrate-and-prompt-repair program for two FIXED points: the base model (gemma-26B) and the gold source (2 curators).* The audit added the two nodes that test those fixed points — **E3** (decisive n=1606 model bake-off) and **E4** (source-independent gold) — plus B1/E2/D2 widenings. B1 absorbs the objective-grain findings; LoRA, an abstain verdict, the n=61 re-score, and a noisy-OR structural redesign were **filtered out** (§3.5 Closed).

### Node table
| Node | Title | Cluster | Hard deps | Gate | Risk |
|------|-------|---------|-----------|------|------|
| **A1** | Repair contamination guard (H3) | A | — | guard loads ≥1/source, fail-loud | none on scoring axis |
| **A2** | Repair example-bank path (H1) | A | — | bank loads; `_TYPE_BANK` non-empty; parity-frozen until A5 | moves prompt input → bounded by A5 |
| **A3** | Balance few-shot disposition (H2) | A | — | loaded shots include resolved-objection correct examples | over-correct toward over-acceptance |
| **A4** | ΔerrF1 paired-bootstrap CI on lead metric (M2) | A | — | err-F1 emitted with CI + permutation p | none (reporting) |
| **A5** | **Clean re-run barrier** (n=1606, both models) | A | A1,A2,A3,A4 (E-BAR) | FP/FN reported separately; ΔerrF1 CI vs 0.154 floor | this is the measurement |
| **B1** | Prevalence-adjusted operating point + **stmt-grain err-F1 / cost-weighted F-β / precision@K capture curve** (M3 + audit ext a/b/c) | B | A5 | additive deployment-truth reports | none (reporting) |
| **B2** | Effective-context probe (deployed models) | B | — | usable-context number per model | none |
| **C1** | ScorerConfig threading (M4) | C | — | I-PARITY golden diff = 0 | none if parity-gated |
| **C2** | Unify result schema + collapse `_score_*` (M5) | C | C1 pref. | I-PARITY golden diff = 0 | none if parity-gated |
| **C3** | Tag-keyed universal pairs + lift shared LLM utils (L3/M5) | C | — | I-PARITY; same pairs selected | none if parity-gated |
| **C4** | Branch-archive dormant scorers + relabel relnature note (M6) | C | — | I-PARITY; Complex-slice err-F1 ≥ ~0.887 | softening relnature regresses Complex win |
| **D1** | LLM layer-2 disambiguation for 3 sole-hit alias residuals | D | A5 | catch ≥2/3 (CRAF1/TK/JH) w/ 0 legit-alias regressions | input-only; over-flag risk |
| **D2** | Minimal-resolved-facts substrate output + claim↔evidence alignment | D | A5, B2 pref. | **per-model** non-inferiority (26B AND 4B); D2→E2 edge | context surgery can drop signal; may invert the 26B's measured context-synthesis win |
| **D3** | *Spike:* asymmetric candidate→check verification | D | A5 | kill-criterion explicit; bounded | exploratory; not a commitment |
| **E1** | Fresh de-contaminated holdout (additive) | E | — | balanced, 0 contamination, new file | none (additive) |
| **E2** | Capacity-gating lever matrix (4B vs 26B + E3 finalist) at n≥1606 | E | A5, E1, E3 | per-model lever verdicts w/ CI | needs clean baseline (A5) |
| **E3** | **Decisive fleet bake-off (n=1606 err-F1 × cost)** | E | A1, A4, A5 | per-model FP/FN + err-F1 CI vs 0.154 floor | BIG SWING; **~$9** (frontier ceilings cut) |
| **E4** | **Source-independent external-validity gold** | E | A1 | new curator(s) outside the 2-curator pair; 0 contamination; additive | BIG SWING; falsifies "ceiling = gold" |
| **X1** | *External:* calibration verdict-grain remedy (+ routed cost-threshold half of B1) | X | A4 (interface) | — (owned elsewhere) | coordination only |

### Hyperedges (multi-node relations)
- **E-BAR (Substrate-Repair barrier):** `{A1, A2, A3, A4} ⟹ A5`. All four must be DONE+REVIEWED before A5 runs;
  A4 must exist so A5 is *interpretable* (CI, not a bare delta).
- **E-GATE (Clean-Measurement gate, = I-MEASURE-CLEAN):** `A5(pass) ⟹ {D1, D2, D3, E2}`. Levers are blocked until A5 passes.
- **E-PAR (Parity invariant, = I-PARITY):** `{C1, C2, C3, C4}` share one golden-output gate; orderable freely but each REVIEW runs it.
- **E-GOLD (= I-GOLD-IMMUTABLE):** `{A5, B1, E1, E2}` bound to additive-only gold.
- **E-CAL (Calibration handoff, = I-CALIB-EXTERNAL):** `{A4, A5, B1} → X1`. We expose per-pair/per-statement
  verdict+score+grounding outputs; the calibration agent consumes them. Interface defined in X1; neither side edits the other's modules.

### Dependency sketch
```
        ┌───────────── E-PAR (parity-gated, independent track) ─────────────┐
        │   C1 ─ C2 ─ C3 ─ C4   (anytime; each REVIEW runs golden diff)      │
        └────────────────────────────────────────────────────────────────────┘

   A1(H3)─┐
   A2(H1)─┤                                          ┌─⟶ D1  (sole-hit disambig)
   A3(H2)─┼── E-BAR ──⟶  A5(clean re-run) ══ E-GATE ═┼─⟶ D2  (minimal-facts substrate)
   A4(CI)─┘                    │                      ├─⟶ D3  (spike: asym verify)
                               │                      └─⟶ E2  (capacity matrix)  ⟵ E1
                               ├─⟶ B1 (prevalence operating point)
                               └─⟶ X1 (calibration, external)   B2 (eff-context, independent) ─⟶ D2
   E1(fresh holdout) ── additive, anytime ── strengthens A5 generalization + feeds E2
```

---

## 3.5 — Completeness-audit extension (2026-06-23)

An adversarial completeness sweep (8 axis-scouts → history-filter → synthesis; 32 proposals, 8 survived) found the original 18-node graph is biased toward *incremental repair of the existing scorer*. Its one structural gap, stated plainly:

> **The program never tests its two load-bearing FIXED points — the base model (gemma-26B) and the gold source (2 curators: bachmanjohn + ben.gyori) — at decisive scale or independence. Every prompt-lever node is a second-order optimization on two unvalidated first-order choices. "Prompt-lever near-spent" and "gemma is the value champion" are both true only _for this model on this gold_, and neither was established beyond n=61 noise.**

### E3 — Decisive fleet bake-off (n=1606 err-F1 × cost) **[NEW · big swing]**
- **Intent.** Resolve the model fixed point. "Frontier not worth it / gemma value champion / kimi top" was decided at n=61, inside the band where "top ~8 are indistinguishable." Run the load-bearing 2–4 candidates (+ the deployed pair) at decisive n=1606.
- **MAP.** Pre-register the committed candidate set: deployed locals (free) + the affordable Bedrock challengers that could plausibly clear gemma — **kimi-k2.5** (the n=61 raw-F1 leader, $7.87) + **gpt-oss-120b** ($1.21). **Frontier ceilings (claude-opus / gpt-5.5) are EXCLUDED — cost is a constraint and they were already "not worth cost" at n=61.** Read prices from `corpus/cost.py`; pre-register which n=61 claims are *falsifiable* vs which ranks stay within CI. Require A5 (clean substrate) + A4 (paired-bootstrap CI).
- **DO.** `run_rasmachine_monolithic.py --model <key>` per candidate on the **A5-repaired** substrate → `eval_curation_compare.py` with A4 CI → `reexport_runs.py` (schema-5 cost bake) → re-render `/frontier` on the eval_curation_v1 substrate.
- **REVIEW.** A1 contamination green on inputs; per-model FP/FN; err-F1 CI vs the 0.154 floor; verdict states which n=61 conclusions survived.
- **Gate / Risk / Invariants.** Gate: a decisive err-F1 ranking with CI. **Cost (verified 2026-06-23 — cost is a constraint): E3 is the SOLE Bedrock-spend node in the program; every other node runs on the local pair (noot-1, $0) or does no inference. Frontier ceilings (claude-opus ~$44 + gpt-5.5 ~$42) are REMOVED as optional confirmation runs. Committed E3 set = {gpt-oss-120b $1.21 + kimi-k2.5 $7.87} = ~$9 at n=1606 — this is the WHOLE program's Bedrock cost.** eval_curation_v1 has **no** cost contrast (both prior runs $0), so this delivers the err-F1 *ranking*; the full cost×F1 Pareto still needs the rasmachine_v1 substrate (the substrate fork is real). Binds I-GOLD-IMMUTABLE, I-CALIB-EXTERNAL. **Deps: A1, A4, A5.**
- **Targets.** `scripts/run_rasmachine_monolithic.py`, `scripts/eval_curation_compare.py`, `scripts/reexport_runs.py`, `scripts/frontier_report.py`, `src/indra_belief/corpus/cost.py`.

### E4 — Source-independent external-validity gold **[NEW · big swing · sibling to E1, not an E1 extension]**
- **Intent.** Resolve the gold fixed point. All current gold (9,342 belief_benchmark rows + every derived set) traces to exactly two curators. Two curators sharing conventions cannot reveal whether the 0.835 ceiling is the *scorer* or *their shared label boundary*. This is the only way to falsify "the ceiling is the gold."
- **MAP.** Pull INDRA curations from a curator **outside** the {bachmanjohn, ben.gyori} pair via the keyless `db.indra.bio/curation/list` per-hash route (`curation.py`); filter `curator NOT IN {pair}`; balance; run through the **repaired** `check_contamination.py` (dep A1) against all holdouts + `belief_benchmark` pa_hashes. *(Lead with option (a): a third in-paradigm INDRA curator. Quarantine option (b): SIGNOR/Reactome DB-curated polarity answers a* different *question — "does the DB assert X?" not "does THIS sentence support X?" — so it must carry its own task note and never pool into headline err-F1.)*
- **DO.** Build `data/benchmark/external_gold_v1.jsonl` + `.meta.json`. **I-GOLD-IMMUTABLE: new file only.**
- **REVIEW.** 0 contamination; additive git diff (no existing tag touched); report err-F1 + CI vs eval_curation_v1 — **a drop _is_ the external-validity signal; parity is the win.**
- **Gate / Risk / Invariants.** Gate: balanced, 0 contamination, additive. Risk: low (option a). Binds I-GOLD-IMMUTABLE. **Dep: A1 only — independent of A5, runnable now (like E1). Tells the program when to stop tuning.**
- **Targets.** `src/indra_belief/curation.py`, `scripts/build_curation_eval.py`, `scripts/check_contamination.py`.

### B1 — widened to three deployment-truth reports **[EXTENDS B1; additive, post-A5]**
The objective-grain findings fold into B1 (not a new objective node — that would stray into the X1 calibration lane). All additive-reporting, selection-metric-unchanged:
- **(a)** the statement-grain err-F1 instrument **already exists and is shipped** (`results.py` `verdict_err`, commit 6df4388; eval_curation_v1 statement verdict-F1 **0.828**) but is exported only as a diagnostic stratum — B1 adds a **clustered-over-statements bootstrap CI** + the deployment-truth framing **alongside** pair err-F1, with a caveat that the grains differ by only ~13% aggregation mass. *(Reporting only — NOT a /frontier re-rank; the ranking half was rejected as motivated by an undemonstrated divergence and would redefine the A4/A5 lead metric.)*
- **(b)** a cost-weighted error rate / F-β reported as a **range** from an elicited C_miss:C_falsereject ratio (3:1–10:1, anchored on the 124:42 false-confidence:false-doubt split). The threshold/tier *re-derivation* half is **routed to X1** (I-CALIB-EXTERNAL owns τ + the tier policy).
- **(c)** a prevalence-reweighted (~58% correct) precision@K / lift-over-random **capture curve** at **statement** grain (errors surfaced vs review-slots), ranked on the existing belief-of-error scalar — the deployment decision metric beside the balanced selection err-F1.
- **REVIEW must source the ~58% prevalence figure** (asserted in B1, not yet independently grounded). At n=60/61 the capture curve is CI-wide — frame as deployment-truth, not a discrimination lever.

### E2 / D2 — model-target tension made explicit **[EXTENDS E2, D2]**
- **E2-ext:** widen E2's model axis from {4B, 26B} to **+ any finalist E3 promotes** (contingent on E3); add per-stmt-type stratification so the entity-nature × gemma cell reports the non-Complex slice. *(Note: `disconfirm_entity` was deleted in the relnature-v3 cleanup — DO must restore it from git history before it can be a matrix cell.)* **Cost: E2's base axis is the local pair ($0). The speculative Bedrock-finalist re-run is REMOVED from the committed budget; if E3 actually promotes a *Bedrock* model over gemma, re-validating that winner is a justified follow-on spend decided at that point (~$8–16), not pre-budgeted. Expected: gemma holds → finalist local → $0.**
- **E2-ext-2 (D2 per-model gate + D2→E2 edge):** rewrite D2's gate from a single aggregate to **per-model** non-inferiority — err-F1 non-inferior at n=1606 for **both** gemma-26B **and** medpsy-4B; a 26B regression fails even if the 4B is neutral. D2's "minimize context" is the *inverse* of the 26B's measured context-synthesis win (entity-reading pilot 0.794→0.836), so a model-blind aggregate gate could silently regress the production model against its own strength. **New edge D2→E2:** D2's shipped-model verdict isn't final until E2's per-model cell confirms no 26B harm.

### Closed — do NOT reintroduce
| Item | Why closed |
|---|---|
| LoRA / DAPT / distillation (fine-tune the verifier) | already-tried-failed (V-phase/V9 < trivial baseline) + out-of-scope for a prompt/substrate program |
| Abstain / "uncertain" third verdict | already-tried-failed — the X-phase removed it (commit 73273b0); the S-phase variant over-absorbed 33 correct cases |
| Re-score the existing n=61 bedrock fleet | noise-floor trap; coarsening to statement grain shrinks n further. The valid form **is E3** (new spend at n=1606) |
| Noisy-OR aggregation *structural* redesign (the "13.3% aggregation residual") | trap — mis-reads a deliberate object split: the profiler grades the calibration scalar at 0.5 vs any-incorrect-wins gold; both real cases already route to `review`. A redesign re-couples scalar+decision (the D8 resolution-collapse anti-pattern). At most a 1-line profiler fix (grade `verdict_statement`, not the scalar). **Independently confirmed dead** by the 2026-06-23 statement-heuristics R6/R7 NO-GO (corpus 96% single-source → aggregation barely applies; residual is the per-evidence comprehension ceiling, not aggregation) |
| Schema-route / reader-stratification as new nodes | already-covered (`results.py` classify + `build_review_queue` weights); at most a one-line A4 reader-stratification report add |

### Re-ranked program (by marginal information toward the true ceiling)
| Rank | Node | Type | Swing |
|---|---|---|---|
| 1 | **A1–A5** substrate repair + clean re-run | old, critical path | safe, mandatory (E-GATE root) |
| 2 | **E4** source-independent gold | NEW | big — falsifies "ceiling = gold"; dep A1 only, runnable now |
| 3 | **E3** decisive fleet bake-off (n=1606) | NEW | big — falsifies the model choice; dep A5 |
| 4 | **B1 (+a/b/c)** deployment-truth reporting | extended | safe, high-leverage; after A5 |
| 5 | **E2 (+ext)** capacity / per-model matrix | extended | medium; contingent on E3 |
| 6 | **E2-ext-2** D2 per-model gate + D2→E2 edge | extended | safe guard |
| 7 | **D1** grounding-flagged hard set | old | medium |
| 8 | **D2** minimal-facts trim (now per-model gated) | old | medium→low |
| 9 | **X1** calibration handoff (+ routed B1-b half) | external | safe |

**Decisive recommendation:** run **A1–A5 first** (mandatory gate), then **E4 and E3 in parallel** — they validate the two fixed points the whole graph silently assumed. Everything else is second-order and should only consume effort *after* E3/E4 establish whether the remaining gap is even reachable on this model and this gold.

---

## Execution status (live — updated by /goal)

**Branch:** `hypergraph/a-substrate-repair` (off main; not committed/pushed). **Full suite: 463 passed, 2 skipped.**

| Node | Status | Date | Result / finding |
|---|---|---|---|
| **A1** | ✅ done | 2026-06-23 | Contamination guard was scanning **74 of 92** sources — blind to all 18 monolithic `CONTRASTIVE_EXAMPLES` via the dead import. Fixed → fail-loud (`SourceImportError`, exit 2) + 3 tests. Re-run still reports **CLEAN** — no hidden few-shot↔holdout overlap surfaced. No follow-up. |
| **A2** | ✅ done | 2026-06-23 | `_EXAMPLE_BANK_PATH` → `parents[2]`; `_TYPE_BANK` now **loads** (was `{}`); missing-bank now warns; 4 tests. **Parity-FROZEN** — its prompt-input effect is measured only at A5. |
| **A3** | ✅ done | 2026-06-23 | 3/10 correct few-shots now carry a resolved-objection (`considered` → objection-with-`correct`-verdict), breaking the objection⇒incorrect prob-1 correlation; 4 tests; **I-DETERMINISM held** (no output-side logic). Parity-FROZEN. |
| **A4** | ✅ done | 2026-06-23 | err-F1 now **leads** with paired ΔerrF1 + 95% CI + permutation p; point estimate byte-exact; calibration modules untouched (I-CALIB-EXTERNAL); `NOISE_FLOOR=0.154` context added; 5 tests. On the OLD (pre-repair, cached) outputs ΔerrF1 = +0.050 [+0.030, +0.069], p<0.0001 — the **pre-repair** number; A5 produces the post-repair one. |
| **E4** | ✅ built | 2026-06-23 | Source-independent gold: `data/benchmark/external_gold_v1.jsonl` **n=154 (77/77)**, **16 pooled non-pair curators** (0 from bachmanjohn/ben.gyori), 0 contamination. (Build agent cleared the Tailscale exit node to reach db.indra.bio — left cleared; gateway unaffected.) |
| **E4-score** | ⚠️ confounded | 2026-06-23 | Scored on bedrock-gemma: **err-F1 0.774** vs eval_curation_v1 ref 0.856/0.862 (**−0.082**). **NOT a clean gold-ceiling read** — external statements were reconstructed from NAME strings (0/308 agents have real db_refs vs 3290/3290 in ref), so they ran **grounding-blind** (all_match, provenance never fired); the −0.082 conflates gold-boundary vs grounding-loss, inseparable from this run. **Clean #3 needs GROUNDED external statements (DB fetch w/ db_refs), not name-reconstruction.** ✅ POSITIVE signal that DOES generalize: **over-rejection** (FP 37/77 correct flagged, P 0.66 / R 0.94) — the dominant failure mode is NOT a 2-curator artifact. |
| **E3** | ✅ done (Bedrock) | 2026-06-23 | **Model bake-off n=1606, full-precision. kimi-k2.5 err-F1 0.881 > gemma-26B 0.856 — ΔerrF1 +0.024 [CI +0.013, +0.037], p<0.0001 SIGNIFICANT.** gpt-oss-120b 0.853 ties gemma (Δ−0.004, p=0.53). kimi wins via **recall** (0.940 vs 0.898; FN 48 vs 82) at ≥ precision (0.829 vs 0.819). Full ranking: **kimi 0.881 > bedrock-gemma 0.856 > local-quantized-gemma ~0.835.** **OVERTURNS the n=61 "gemma value champion / top-8 indistinguishable"** — at decisive scale the model lever is real. Cost: kimi ~$7.87/1k vs bedrock-gemma ~$1.2/1k vs local $0. **#2 data-resolved; cost-quality adoption is a policy call.** Parse: kimi 18 null/1606, gpt-oss 3. **➜ DECISION 2026-06-23: STICK WITH GEMMA** — kimi NOT adopted; the +0.024–0.046 err-F1 (mostly error-recall) isn't worth ~$7.87/1k vs free local gemma under the cost constraint. Consequence: E2's bedrock-finalist arm is moot; B1 reports on gemma; deploy stays local (the +0.021 full-precision-bedrock-gemma gain also declined on cost). |
| **E4-grounded** | ✅ done (Bedrock) | 2026-06-24 | **Clean #3.** Rebuilt the 154 gold statements WITH real db_refs from curation `pa_json` (**306/313 agents grounded = 97.8%**, vs 0/308 in the confound). Grounded err-F1 **~0.76** (0.757–0.768) ≈ ungrounded 0.774 → grounding is NOT the lever. **CORRECTED 2026-06-24 (stmt-type-mix probe): the ~0.10 gap vs the 0.856/0.862 reference is a STATEMENT-TYPE-MIX ARTIFACT, NOT a curator-boundary/generalization gap.** external_gold is 14% Complex (gemma's best type) vs eval_v1's 58%, and is loaded with Activation/Inhibition (gemma's weaker types). **Reweighted to eval_v1's type mix, external err-F1 = 0.865 ≈ eval_v1 0.857 — the gap VANISHES.** Per-type the two independent golds are comparable (external even better on Complex/Phos). ⇒ **#3 RESOLVED in the REASSURING direction: the err-F1 ceiling GENERALIZES to independent curators; 0.86 is NOT a 2-curator-overfit mirage.** Over-rejection-worse-on-external was also a type-mix effect, not curator. Caveats: external_gold is single-curator-per-row (unadjudicated), imke.ditters=38% of rows, small per-type n (reweighted 0.865 has wide CI); one watch-item = Activation (ext 0.687 vs 0.822, n=48). 2nd finding CORRECTED 2026-06-24 (grounding-trap dig-in): the MISMATCH tier showed all_match 154/154 NOT because curated gold lacks grounding errors, but because the **reconstruction stripped the reader's raw_text** (used canonical names). The 4 FN misses are all grounding-tagged; tested through real `GroundedEntity` with the reader's surface form, **3/4 ARE caught in production**: "BT" (BTN1A1→MAPT) → MISMATCH → Tier-1 **auto-reject**; "EP" (DBI→ERK ×2) → **AMBIGUOUS** LLM warning; only "GAA" (GAA-enzyme vs GAA·TTC-repeat homonym) is irreducible (same string; needs context — gemma had it and missed → comprehension residual). ⇒ the `ebaa600` substrate DOES pay on curated-gold grounding errors when raw_text survives; the 0.77 run understates production recall on grounding traps. |
| **A5** | ✅ done (Bedrock) | 2026-06-23 | **Repair effect NULL — the substrate fix cleaned the measurement, it did not change the story.** Full-precision bedrock-gemma, pre-repair err-F1 **0.862** vs post-repair **0.856**; **ΔerrF1 −0.005 [CI −0.015, +0.004], perm p=0.29 (NS)**; both arms 1606/1606 clean. ⇒ the **prompt/few-shot lever is spent — now VALIDATED at n=1606** (was suspect because the original headline ran on the degraded substrate; the A2 bank + A3 shots demonstrably do NOT move err-F1). Repairs KEPT (correctness / honesty / fail-loud guard). Side-signal: bedrock-gemma ~0.86 **>** local-quantized ~0.835 (≈ +0.025 full-precision/thinking gap — to confirm in E3). (Local diagnosis: `--workers 4` wedged the single-GPU ollama backend, fixed by restarting `llm-gateway-ollama-1`; `--workers 1` would be multi-day → pivoted to Bedrock.) **E-GATE OPEN.** |

**Open follow-ups (low priority):** A4's per-model table F1 point uses the full-join estimand while its CI is paired-subset (disclosed via caveat) — optionally align to one estimand.

**Frontier now:** **A5 RUNNING** (background `b10z021ku`, multi-hour) — it holds the only GPU *and* the live scorer source files (`scorer.py`/`_prompts*.py`), so the program genuinely **serializes** here: the GPU-bound nodes (B2, external-gold scoring, post-A5 E3/E2/D-levers) and the `scorer.py`-mutating C-track must wait for A5 to land (not a stop-point — a resource/dependency lock; the A-cluster + E4 edits are uncommitted, so a C-track worktree would not see them cleanly). **E4 done.** The instant A5 lands → fire **E3** (~$9 bake-off) ∥ **B1** (deployment-truth reports over A5's output) ∥ score `external_gold_v1` on the repaired pair (the E4 external-validity read), then the C-track + D/E2 levers.

---

## 4. Node specifications (engineering detail)

> Convention per node: **Intent · MAP · DO · REVIEW · Artifacts · Gate · Risk · Targets.** Line numbers are
> seeds for MAP to re-confirm, not ground truth.

### Cluster A — Substrate Integrity

#### A1 — Repair the contamination guard (H3)
- **Intent.** The pre-eval safety gate silently scans zero of a declared live source, so "no contamination"
  is currently unfalsified, not verified. Make it honest. *(Do first: it is the cheapest, has no scoring-axis
  risk, and re-establishes trust in every downstream eval.)*
- **MAP.** Confirm `from indra_belief.scorers._prompts import CONTRASTIVE_EXAMPLES` (`check_contamination.py:63`)
  raises `ModuleNotFoundError` (real module is `monolithic._prompts`), swallowed to `[]` at `:64-65`. Enumerate
  *every* declared source in the file and check each loads non-empty. Spec the fail-loud contract.
- **DO.** Fix the import to `indra_belief.scorers.monolithic._prompts`. Replace the blanket `except: =[]` with a
  loud failure on a *missing/empty* declared source (a genuinely empty source may be legitimate — distinguish
  "import failed" from "source is empty"). Add a test asserting each declared source resolves non-empty.
- **REVIEW.** Run the guard against current eval inputs; assert Source-1 count > 0; assert the new test fails
  if the import path is reverted. **Note:** a now-working guard may surface previously-hidden few-shot↔holdout
  overlaps — that is a correct data-hygiene *finding* (spawn a follow-up node), not a regression.
- **Artifacts.** Patched `check_contamination.py`; `tests/test_contamination_guard_sources.py`.
- **Gate.** Every declared source loads ≥1 example or fails loudly; new test green.
- **Risk.** None on scoring axis (offline guard). 
- **Targets.** `scripts/check_contamination.py:63-65`; `src/indra_belief/scorers/monolithic/_prompts.py`.

#### A2 — Repair the example-bank path (H1)
- **Intent.** The 19-key type-specialized bank never loads (`_EXAMPLE_BANK_PATH` is off by one directory), so
  few-shot selection rides only the 18 base pairs, covering 4 of 19 statement types. Restore the designed input.
- **MAP.** Confirm `_EXAMPLE_BANK_PATH = Path(__file__).parent.parent / "data" / …` resolves to
  `scorers/data/` (absent); real file at `src/indra_belief/data/example_bank.json`. Diff what the bank *adds*
  per statement type vs the base fallback (it carries the `Activation_hypothesis`, `Inhibition_mirna`,
  `Inhibition_act_vs_amt`, `Activation_family`, `Phosphorylation_negation` buckets — the error-pattern keys that
  target our documented failure clusters). Spec the corrected path + the silent-guard→loud-warn change.
- **DO.** Fix to `Path(__file__).resolve().parents[2] / "data" / "example_bank.json"`. Convert the silent
  `if _EXAMPLE_BANK_PATH.exists()` to load-or-log-warning (so a future move fails visibly). **Do not** re-run
  conclusions here — A2's output is frozen until A5.
- **REVIEW.** Assert at runtime `_TYPE_BANK` non-empty and keyed by all 19 base types where the bank has them;
  assert selection now pulls own-type bank pairs for a non-base type (e.g. `Translocation`). **I-PARITY note:**
  A2 *intentionally changes the prompt input*, so it is parity-frozen — its effect is measured only at A5, never
  shipped as "the same."
- **Artifacts.** Patched `scorer.py`; `tests/test_example_bank_loads.py` (path + non-empty + per-type coverage).
- **Gate.** Bank loads; `_TYPE_BANK` non-empty; coverage test green.
- **Risk.** Re-introduces a real-but-untested input → strictly bounded by the A5 barrier; could move err-F1 either way.
- **Targets.** `src/indra_belief/scorers/monolithic/scorer.py:106-110, 124-133, 184-231`; data at `src/indra_belief/data/example_bank.json`.

#### A3 — Balance the few-shot disposition (H2)
- **Intent.** Loaded shots teach "objection present ⇒ incorrect" with perfect correlation (0/18 carry
  `considered`), which is the exact bias the disconfirm/no-backstop design exists to break, and a plausible
  driver of over-rejection. Make the shots demonstrate **surface-objection-then-accept.**
- **MAP.** Confirm the disconfirm renderer supports a resolved-objection field
  (`_prompts_disconfirm.py:107-120`, `support, objection = ev, (ex.get("considered") or None)`), and that the
  single `considered` example lives only in the (now-loadable, post-A2) bank. Decide the minimal set: annotate
  2–3 *correct* base examples with a `considered` (an apparent objection an explicit rule resolves), keeping
  the correct/incorrect objection-presence symmetry. **This touches prompt examples, never gold labels.**
- **DO.** Add `considered` annotations to the chosen correct base pairs in `_prompts.py`. Verify rendering:
  the correct shots now show `objection=<considered>, verdict=correct`. No change to incorrect shots' logic.
- **REVIEW.** Render the full few-shot block for several statement types; assert ≥2 correct shots now carry a
  non-null objection-with-correct-verdict; assert no shot teaches objection⇒incorrect with prob 1.0. Confirm
  I-DETERMINISM untouched (no output-side logic added).
- **Artifacts.** Patched `_prompts.py`; `tests/test_fewshot_disposition_balance.py`.
- **Gate.** Loaded shots include resolved-objection correct examples; symmetry test green.
- **Risk.** Over-correction toward over-acceptance — *this is why A5 reports FP and FN separately.* gemma already
  misses more errors (FN=158) than it over-rejects (FP=97) on balanced n=1606; do not trade recall blindly.
- **Targets.** `src/indra_belief/scorers/monolithic/_prompts.py`; renderer `_prompts_disconfirm.py:107-120`.

#### A4 — ΔerrF1 paired-bootstrap CI on the lead metric (M2)
- **Intent.** The lead metric (error-F1) is currently emitted bare; the only CI and McNemar attach to the
  *demoted* metric (accuracy). A5 must be interpretable, so build the CI before the re-run.
- **MAP.** Confirm `wilson_ci` is on `acc_hits` (`eval_curation_compare.py:127`), err-F1 bare (`:209-212`),
  McNemar on accuracy-concordance. Locate the existing `bootstrap_errf1` in `scripts/calibration_ship_gate.py`
  and the documented `NOISE_FLOOR = 0.154`. Spec the back-port (a paired bootstrap ΔerrF1 + permutation p on F1).
- **DO.** Back-port `bootstrap_errf1` into the compare path; emit err-F1 with a 95% paired-bootstrap CI and a
  permutation p, lead with it, keep McNemar-on-accuracy as a secondary traceable line. **I-CALIB-EXTERNAL:**
  reuse the function, do not modify calibration-owned modules.
- **REVIEW.** Reproduce a known compare on cached outputs; assert the err-F1 point estimate is byte-identical to
  today's and only the *reporting* (CI/p) is added; assert the noise floor is honored in interpretation text.
- **Artifacts.** Patched `eval_curation_compare.py`; a golden re-report of an existing run (numbers unchanged + CI added).
- **Gate.** err-F1 reported with CI + permutation p; point estimate unchanged on cached data.
- **Risk.** None — reporting-only; cannot move verdicts.
- **Targets.** `scripts/eval_curation_compare.py:112,127,209-212`; `scripts/calibration_ship_gate.py` (`bootstrap_errf1`).

#### A5 — Clean re-run barrier (n=1606, both models) ⟵ E-BAR, E-GATE root
- **Intent.** Re-measure on the repaired substrate so every downstream conclusion (and the standing
  "prompt-lever-spent" claim) rests on an honest baseline. **This node is the measurement; it is the only node
  permitted to move the headline number.**
- **MAP.** Require A1∧A2∧A3∧A4 all REVIEW=pass. Spec the run matrix: gemma-26B + medpsy-4B × eval_curation_v1
  (n=1606), default `disconfirm_relnature`. Pre-register the comparison: ΔerrF1 vs the prior 0.835/0.785, FP/FN
  deltas, per-tag breakdown, all with A4's CI. Pre-register the interpretation rule (what counts as the bank+shots
  helping vs the noise floor).
- **DO.** Run the scorer (runner `run_rasmachine_monolithic.py --arch monolithic` / the eval harness) over the
  untouched eval set; produce `data/results/eval_curation_v1_compare_postsubstrate.md`. **I-GOLD-IMMUTABLE:** gold
  is read-only; no relabeling. Use a worktree if running against branch changes from C-cluster.
- **REVIEW.** Adversarially check: did contamination guard (A1) pass on these inputs? Are FP and FN both reported?
  Is the ΔerrF1 CI vs 0.154 floor stated? Is the conclusion about the few-shot lever drawn *only* from this run,
  not the old error-enriched probe? Verdict gates E-GATE.
- **Artifacts.** `…_postsubstrate.md` with FP/FN, ΔerrF1+CI, per-tag; an updated note in the relevant research doc
  retiring or confirming "prompt-lever near-spent" for the few-shot lever specifically.
- **Gate.** Re-run complete, both models, FP/FN + ΔerrF1 CI vs floor; A1 guard green on inputs.
- **Risk.** This is where A2/A3 cash out — either direction is an acceptable, *informative* outcome.
- **Targets.** `scripts/eval_curation_compare.py`, `scripts/run_rasmachine_monolithic.py`, `data/benchmark/eval_curation_v1.jsonl` (read-only).

### Cluster B — Measurement & Deployment Truth

#### B1 — Prevalence-adjusted operating point (M3)
- **Intent.** Balanced err-F1 (0.835) is a *selection* metric; production is ~58% correct, where usable
  error-class precision falls to roughly 0.50–0.82. Report the deployment operating point so the headline isn't
  misread as a curator's experienced precision.
- **MAP.** Confirm eval is forced 1:1 (`build_curation_eval.py:16`) and the production prevalence figure; spec a
  prevalence-reweighted error-class PR curve + expected review-queue precision at pair grain. No gold edits.
- **DO.** Add the additive reweighted PR / expected-queue-precision report; one-line caveat that 0.835 is a
  selection number, not deployment.
- **REVIEW.** Assert additive-only (no change to the balanced selection metric); assert the prevalence figure is
  sourced, not assumed; range reported (not a single point).
- **Artifacts.** Prevalence-operating-point section in the compare report.
- **Gate.** Reweighted PR + caveat present; selection metric unchanged.
- **Risk.** None (additive reporting); amplifies visibility of the known over-rejection/over-acceptance driver.
- **Targets.** `scripts/eval_curation_compare.py`; `scripts/build_curation_eval.py:16`.

#### B2 — Effective-context probe (deployed models)
- **Intent.** Our usable-context budget is unknown; NoLiMa/RULER effective-length numbers are 2024-vintage and
  none exist for our deployed gemma-26B / medpsy-4B. Measure it; it informs D2's prompt budget.
- **MAP.** Spec a RULER/NoLiMa-style probe (latent-association retrieval + single-distractor) scaled to our
  realistic prompt sizes (claim + evidence + entity context + provenance), run against both deployed models.
- **DO.** Build `scripts/probe_effective_context.py`; run; record usable-context-vs-degradation per model.
- **REVIEW.** Assert the probe forces *latent-association* (no literal overlap), reflects our real prompt shape,
  and reports an effective length, not a vanilla-NIAH pass rate.
- **Artifacts.** `scripts/probe_effective_context.py`; `data/results/effective_context_probe.md`.
- **Gate.** A usable-context number per deployed model.
- **Risk.** None.
- **Targets.** new script; deployed `medpsy-remote` / gemma endpoints (see model_client.py).

### Cluster C — Architecture Hygiene (E-PAR: all parity-gated)

#### C1 — ScorerConfig threading (M4)
- **Intent.** `MONO_VARIANT` is an import-time process global (`scorer.py:59`), so no test can flip the variant
  in-process and the branch-selection fork (`:262/:292/:317`) + baseline else-branch are untested. Make the
  variant an injectable config; env var becomes the default factory.
- **MAP.** Confirm the global resolution and the untested branch-selection fork. Spec a `ScorerConfig` dataclass
  (system_prompt, render/parse/derive callables, relnature flag) threaded through `score()`/`score_statement()`.
- **DO.** Implement in a worktree. Env var → default factory. Add tests that exercise disconfirm/baseline branch
  selection in-process (now possible).
- **REVIEW.** **I-PARITY:** the default factory must reproduce today's `disconfirm_relnature` binding byte-for-byte
  over a frozen fixture set — golden diff = 0. Assert branch-selection now has direct coverage.
- **Artifacts.** `ScorerConfig`; threaded entry points; branch-selection tests; golden fixture.
- **Gate.** Golden diff = 0; in-process variant switch works.
- **Risk.** None if parity-gated (pure plumbing; touches no prompt/threshold/verdict logic).
- **Targets.** `scorer.py:59,262,292,317,462,556`.

#### C2 — Unify result schema + collapse `_score_single`/`_score_with_tools` (M5)
- **Intent.** The two scoring paths are ~90% duplicated and the result-dict schema is maintained in 3+ places
  (`_score_single` `:332-358`, `_score_with_tools` `:422-459`, Tier-0 `:496-498`, final assembly `:541-553`),
  inviting silent schema drift / KeyError.
- **MAP.** Confirm the duplication and the 3+ schema sites. Spec one `_score(..., *, lookup_block='',
  system_suffix='', kind)` helper + one result-builder/dataclass used by all three tiers.
- **DO.** Collapse in a worktree; preserve the exact `ACTIVE_SYSTEM_PROMPT + _LOOKUP_GUIDANCE` concatenation and
  the conditional message augmentation. Drop the dead `temperature` param on `_score_single`.
- **REVIEW.** **I-PARITY:** golden diff = 0; add a regression test asserting the augmented system string +
  `messages[-1]` for a grounding-flagged record vs a clean record (locks input-identity).
- **Artifacts.** Unified `_score`; single result schema; input-identity regression test.
- **Gate.** Golden diff = 0; input-identity test green.
- **Risk.** None if parity-gated.
- **Targets.** `scorer.py:332-358,422-459,496-498,541-553`.

#### C3 — Tag-keyed universal pairs + lift shared LLM utils (L3 / M5)
- **Intent.** Two papercuts: `_UNIVERSAL_PAIRS = _ALL_EXAMPLES[4:6],[6:8]` (`scorer.py:161-164`) is positional —
  a reorder silently re-picks the priority-3 slot; and the live monolithic default transitively loads the dormant
  `probes/` package via `probes._llm` for `_extract_json`/`llm_classify` through an eager `probes/__init__.py`.
- **MAP.** Confirm the positional slices resolve to the intended AGER/TP53 + MYB/PPID pairs today (anchored only by
  a comment) and the eager probe import chain. Spec: add a `category` tag to the universal examples (selection-
  preserving) and lift `_extract_json`/`llm_classify` to `scorers/_shared.py`, breaking the eager `__init__` chain.
- **DO.** Implement in a worktree; selection must be *identical* (same pairs, keyed by tag not index).
- **REVIEW.** **I-PARITY:** assert the same 14 examples are selected for a sample of statement types; assert the
  default no longer imports the probes package; golden diff = 0.
- **Artifacts.** tag-keyed universal registry; `_shared._extract_json/llm_classify`; import-graph test.
- **Gate.** Identical selection; probes no longer transitively imported; golden diff = 0.
- **Risk.** None (selection-preserving + relocation).
- **Targets.** `scorer.py:161-164`; `src/indra_belief/scorers/probes/_llm.py`, `probes/__init__.py`; `_shared.py`.

#### C4 — Branch-archive dormant scorers + relabel relnature note (M6)
- **Intent.** Two things: (a) the decomposed/panel/probes pipelines are live comparison baselines with eval
  anchors but clutter the live import surface — branch-archive them (V-phase precedent) rather than delete; (b) the
  relnature note is written in deterministic-grounding vocabulary ("that is a grounding MISMATCH") though it is a
  *second LLM call's output* — relabel to truthful provenance ("relation-nature objection (model-derived)").
- **MAP.** Confirm the relnature note text (`_prompts_relation.py:126-129`) and that no code reads it to flip the
  verdict (honors I-DETERMINISM mechanically). Confirm the Complex-slice lever was validated at err-F1 +0.04
  (audited ~0.887, `fc8f5de`). Spec the relabel + the branch-archive plan.
- **DO.** Relabel the note voice/trace; create the archive branch for dormant scorers; keep the shared seam
  (`score_evidence`/`score_statement`) intact for A/B. **Do NOT soften the relnature disposition.**
- **REVIEW.** **I-PARITY + Complex gate:** the relnature *disposition* must be unchanged (only the label/trace
  string differs); if the disposition is touched at all, A/B it and gate ship on Complex-slice err-F1 ≥ ~0.887.
  **I-DETERMINISM:** confirm the note remains input-only.
- **Artifacts.** relabeled note; archive branch ref; (if disposition touched) Complex-slice A/B report.
- **Gate.** Golden diff = 0 on the disposition; note provenance truthful; dormant scorers preserved on branch.
- **Risk.** Softening the relnature lever regresses a measured Complex win — note the statement-grain residual is
  over-*acceptance* (false-confidence 124 > false-doubt 42), so this lever fights the larger statement-grain failure.
- **Targets.** `_prompts_relation.py:126-129`; `scorers/{decomposed.py,panel/,probes/}`.

### Cluster D — Substrate-Input Grounding (E-GATE: all blocked until A5 passes)

#### D1 — LLM layer-2 disambiguation for the 3 sole-hit alias residuals
- **Intent.** The `ebaa600` structural fix caught 10/12 alias-collision bugs; 3 sole-hits (CRAF1←TRAF3, TK←TKT,
  JH←HJV) are irreducible at the substrate (indistinguishable from legit aliases like RRAS2←TC21). Resolve them on
  the **input** side via an LLM disambiguation layer that reads the evidence context — never by tightening the
  substrate or overriding the verdict.
- **MAP.** Re-read `grounding_alias_collision_report.md` §6.2 (the layer-2 NO-GO under the *old* degraded shots) and
  `entity.py:380-406` (`_competing_candidates`). **Crucially: the prior layer-2 attempt was tested on the H1/H2-
  degraded substrate** — A5 may change its verdict, which is *why* this node is E-GATE'd. Spec a context-conditioned
  disambiguation that emits an input signal (this raw_text, in this sentence, denotes X not Y), not a verdict flip.
- **DO.** Implement the input-side disambiguation behind a flag; run on the 3 residual cases + the 9 legit-alias
  controls; worktree.
- **REVIEW.** **I-DETERMINISM:** assert the layer emits context only, no `result["verdict"]` write. Gate: catch ≥2/3
  residuals with **0** regressions on the 9 legit aliases; re-validate at n=1606, not on the 12-case micro-set.
- **Artifacts.** flagged disambiguation layer; residual+control report; n=1606 validation.
- **Gate.** ≥2/3 caught, 0 legit-alias regression, n=1606 non-inferior err-F1.
- **Risk.** Over-flag (the reason the substrate-only approach stopped at the 0.10 band) — input-only keeps it safe.
- **Targets.** `src/indra_belief/data/entity.py:380-406`; `research/archive/completed_phases/grounding_alias_collision_report.md` §6.2.

#### D2 — Minimal-resolved-facts substrate output + claim↔evidence alignment
- **Intent.** Frontier principle (CONFIRMED): a single distractor pushes accuracy below baseline; low claim↔needle
  similarity accelerates degradation; verification is a synthesis/citation task (the degraded regime). Ensure the
  substrate hands the LLM *resolved, minimal* facts (one competing grounding, not a candidate dump) and that the
  claim is rendered in vocabulary close to the evidence span.
- **MAP.** Audit `scoring_record.format_user_message` (`:358-388`), provenance block (`:307-356`), entity context
  (`:253-279`) for distractor mass; cross-reference B2's effective-context number for the budget. Spec the trim +
  the alignment (surface raw-text forms as a lexical bridge, already partially done by `_verify_raw_text`).
- **DO.** Tighten provenance/entity-context to resolved minimal facts; ensure claim rendering surfaces the
  evidence-aligned surface forms. Worktree.
- **REVIEW.** Assert distractor token count ↓ and err-F1 **non-inferior** at n=1606 (it must not silently drop a
  signal the model used); confirm flag-gated provenance discipline preserved (full-population injection was −6.7pp).
- **Artifacts.** trimmed substrate output; n=1606 non-inferiority report.
- **Gate.** Distractor tokens ↓, err-F1 non-inferior, flag-gating preserved.
- **Risk.** Context surgery can remove a load-bearing fact — non-inferiority gate guards it.
- **Targets.** `src/indra_belief/data/scoring_record.py:253-279,307-356,358-388`.

#### D3 — *Spike:* asymmetric candidate→check verification
- **Intent.** Frontier (medium-confidence): reallocating compute from search to *backward verification* (start from
  a candidate verdict, then check it) beats more search. This is **sequential** (candidate→check), explicitly **not**
  parallel voting. Run a bounded design spike with a kill criterion; this is exploration, not a commitment.
- **MAP.** Spec a 1-statement-type spike: monolithic verdict → a focused second "try to refute this verdict against
  the evidence" check (one call), measured against the single-call baseline on a small slice. Pre-register the kill
  criterion (no ΔerrF1 beyond noise floor, or cost not justified → kill).
- **DO.** Prototype on a branch over one slice; do not wire into the default path.
- **REVIEW.** Compare to single-call at n≥ slice; apply kill criterion honestly; if it survives, spawn a full node.
- **Artifacts.** spike report with go/kill verdict.
- **Gate.** Explicit kill-criterion evaluated; bounded cost.
- **Risk.** Exploratory; low priority; keep cheap.
- **Targets.** prototype branch; `scorers/monolithic/`.

### Cluster E — Fresh Evidence (E-GOLD: additive-only)

#### E1 — Fresh de-contaminated holdout (additive)
- **Intent.** A clean, internally-consistent holdout (a *new* file, never editing existing gold) so lever
  validation (D-cluster, E2) generalizes beyond eval_curation_v1 and isn't overfit to one balanced set.
- **MAP.** Spec construction from untouched sources (`data/benchmark/belief_benchmark.jsonl`), balanced by
  construction, run through the *repaired* `check_contamination.py` (A1) against all prior holdouts/few-shots.
  **I-GOLD-IMMUTABLE:** new file only; no existing tag touched.
- **DO.** Build `data/benchmark/holdout_postsubstrate_v1.jsonl` + `.meta.json` provenance; contamination report green.
- **REVIEW.** Assert 0 contamination, balance, and that `git diff` adds only new files (no edits to existing gold).
- **Artifacts.** new holdout + meta + contamination report.
- **Gate.** Balanced, 0 contamination, additive-only.
- **Risk.** None (additive).
- **Targets.** `scripts/build_curation_eval.py`, `scripts/check_contamination.py`, `data/benchmark/belief_benchmark.jsonl`.

#### E2 — Capacity-gating lever matrix (4B vs 26B)
- **Intent.** Distillation "capacity gap" science mirrors our entity-reading finding: context-synthesis levers help
  gemma-26B and dilute medpsy-4B. With a clean baseline (A5) + fresh holdout (E1), re-validate which prompt-complexity
  levers fit which model size at n≥1606 — replacing the n=60 noise-band guesses.
- **MAP.** Enumerate the levers (relnature, entity-nature, two-step, minimal-facts D2) × {4B, 26B}; pre-register
  per-cell err-F1 + CI; require A5 + E1 done.
- **DO.** Run the matrix on eval_curation_v1 + the fresh holdout.
- **REVIEW.** Per-model lever verdicts with CI vs noise floor; assert n is the decisive scale, not n=60.
- **Artifacts.** capacity-gating matrix report.
- **Gate.** Per-model lever verdicts with CI.
- **Risk.** Needs the clean baseline; otherwise re-measures noise.
- **Targets.** harness + both endpoints.

### External

#### X1 — Calibration verdict-grain remedy (owned by another agent)
- **Role.** Not our work. The standing M1 gap — no verdict-grain mechanism for over-rejection; soft-guard is
  belief-grain only (`statement_belief.py:215-221` never reads `soft`) — is the calibration agent's territory.
- **Our obligation (E-CAL handoff).** A4/A5/B1 must **expose** clean per-pair and per-statement outputs
  (verdict, confidence, score, grounding_status, tier, FP/FN labels) in a stable schema the calibration agent
  consumes. We define the interface; we do not edit `noise_model.py` / `calibration_constants.py` / the rollup.
- **Coordination.** Confirm the output schema with the calibration agent before A5 so the re-run emits what they need.

---

## 5. Extension protocol (how we grow this graph)

1. A node is added only with a full `§4` spec (Intent · MAP · DO · REVIEW · Artifacts · Gate · Risk · Targets) and
   its hyperedge membership (which invariants bind it, what it depends on, what it gates).
2. A REVIEW that returns `pass-with-followup` **must** name the new node and its cluster; orphan findings are not
   allowed to float.
3. Any node that could move a shipped number inherits **I-PARITY** unless its explicit purpose is to move it (then
   it must be gated by a pre-registered re-run like A5).
4. Levers (anything claiming an err-F1/calibration gain) inherit **I-MEASURE-CLEAN** — blocked until the relevant
   clean baseline node passes.
5. Calibration-adjacent ideas route to **X1** (define interface), not into our modules.
6. Gold-adjacent ideas inherit **I-GOLD-IMMUTABLE** — additive new files only.

---

## 6. Critical path & suggested ordering

**Critical path (must be serial):** `A1 → (A2 ∥ A3 ∥ A4) → A5 → E3 → {D1, D2, E2}`, with **E4** running in parallel off A1. See §3.5 for the audit-driven re-ranking — **E4 and E3 are the two highest-EV moves** (they validate the model and gold fixed points the rest of the graph assumed).

1. **A1** first — cheapest, zero scoring risk, restores trust in all eval. *(Safe to execute immediately.)*
2. **A2, A3, A4 in parallel** — two substrate repairs + the CI instrumentation; each its own map→do→review; A2/A3 in
   worktrees (they mutate the prompt substrate), A4 reporting-only.
3. **A5** — the barrier. The single highest-value measurement in the program. Everything about "is the prompt lever
   spent" and "how big is over-rejection really" is unsettled until this passes.
4. **Parallel track, anytime, parity-gated:** **C1 → C2 → C3 → C4** (hygiene; never moves numbers; unblocks the next
   contributor). **B2** and **E1** are independent and can run anytime; B2 feeds D2, E1 feeds E2.
5. **After A5 passes (E-GATE opens):** **E3** (decisive fleet bake-off — top post-A5 priority; if a wired model clears gemma >noise-floor the whole prompt track is moot), then **B1** (deployment-truth reports a/b/c), then the levers **D1, D2** (D2 now per-model gated; **D3** spike low-priority), then **E2** (capacity / per-model matrix). **E4** (source-independent gold) runs in parallel as soon as **A1** lands — it needs no A5.
6. **Throughout:** keep **X1** synchronized — confirm the calibration output schema before A5 so the re-run emits it.

**One-line orientation:** repair the substrate honestly (A1–A3), make the measurement interpretable (A4), re-measure
(A5), and only then spend effort on the research-informed levers (D, E) — while the parity-gated hygiene track (C)
and the external calibration handoff (X1) proceed in parallel without touching the numbers.
