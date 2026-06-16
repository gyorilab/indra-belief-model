# LLM-Verifier Calibration — Task Hypergraph

**Status:** Drafted 2026-06-17. **C0 executed 2026-06-17 → G0 = GO** for both readers (`scripts/calibration_stage0.py` → `data/results/calibration_stage0.md`): per-statement gated belief AUROC 0.77 (MedPsy) / 0.81 (gemma), AUPRC 0.74 / 0.77 vs base 0.51 — real discrimination AND real miscalibration (mid-range under-confidence). **C1 interim done 2026-06-17** (`scripts/calibration_stage1.py` → `data/results/calibration_stage1.md`): the soft weight works (big ECE gains) but the committed `replace` form over-flattens (G1 FAIL on resolution); the **`guard` variant at kappa≈0.5** improves ECE AND preserves resolution/AUROC — test-selected, so **confirmation pre-registered (`scripts/calibration_confirm.py`) + runbook ready (`scripts/score_holdout_cc.sh`), awaiting the `holdout_cc` gateway run** before C2. Decision-driven by **G0** (the go/no-go gate): Stage C0 is a zero-model-change diagnostic that tells us whether C1–C3 are worth doing at all. The architecture is settled (soft survival weight = recalibration of the existing per-read `rand`, not a new error term); the empirical anchors are fit + verified (below). This hypergraph turns the calibration plan into do→review cycles. **Math arc** (C0–C3, gated G0–G3) lives here; the **presentation arc** (C4/C5, gated G4/G5v — making products navigable as run-on-its-own vs run-comparison) hands off via substrate edge **E5** to `research/belief_instrument_task_graph.md`.
**Last update:** 2026-06-17 (+ E5 presentation seam + C4/C5 handoff: viewer is NOT calibration-ready today)
**Owner question:** "Consider INDRA's existing error mode `1 − ∏ₛ[syst(s) + rand(s)^nₛ]`. How do we arrive at a coherent, useful, accurate heuristic to inform calibration?"

---

## Frame

The LLM emits a **binary verdict** (`correct`/`incorrect`); its 3-level confidence has **collapsed** (gemma 1596/1606 `high`, zero `low`; medpsy 1522/73/11 — verified against `data/results/eval_curation_v1_{gemma,medpsy}.jsonl`). So classical probability calibration over the `VERDICT_SCORE_GRID` is degenerate — the score axis occupies ~2 values, and `metrics.ece` over it is a coarse accuracy check, not a reliability curve.

The coherent move is **not** to add the LLM as a new probability. `rand_s` was fit as `rand = 1 − accuracy − syst` against per-read correctness on the 9,342-statement benchmark (`noise_model.py:48-49`) — the *same latent event* ("did this read get the relation right?") that the LLM verdict judges. Therefore the LLM is **not an independent source**: it conditions the per-read error that already exists. The heuristic recalibrates `rand_s` keyed on the verdict; it never multiplies a second reader-error factor on top of it (that double-counts).

**The radical simplification this frame buys us:** two parameters per reader model. That is the entire fit.

---

## Decisions (locked 2026-06-17)

| # | Decision | Rationale |
|---|---|---|
| D1 | **LLM recalibrates `rand_s`, never adds a term.** No sibling `(syst_LLM + rand_LLM^k)` factor, no `n^k` decay for the LLM. | LLM is not independent of the source — it reads the same mined sentence. Adding a factor double-counts the one per-read error. |
| D2 | `w_j` is defined as **P(read is wrong \| verdict class)** — one number that *replaces* `rand_s`, not Bayes-fused beside it. | `rand_s` and the verdict estimate the same event; fusing them as independent prior×evidence is the hidden bug all three derivations risked. |
| D3 | **`syst_s` untouched.** | LLM shares the source's systematic blind spot (reads the same sentence); cannot be credited with detecting it. Irreducible floor stays. |
| D4 | **Drop the confidence axis.** Fit per-verdict only. | Confidence is vapor (98–99% `high`, 4 of 6 grid cells n≤11). Unfittable. |
| D5 | **Fit per reader model**, baked into each run's export. | gemma trades +0.4pp confirm-error for −6pp reject-error vs medpsy; one `(rand_corr, rand_rej, κ)` triple per reader. No global hardcode — it travels with the run. |
| D6 | **No per-stratum fit.** Pool to per-model; document residual stratum miscalibration. | Error swings 24–44pp by `stmt_type`, but post-model-split cells are n≤4 (`mod_site` n=4, `agent_conditions` n=1). |
| D7 | **Invariant:** with all `verdict=None`, the new path reproduces `compute_gated_belief` **byte-for-byte**. | The identity gate. Same golden-output discipline used for the metrics extraction; it backs shipped F1. |
| D8 | The lever is **reader/grounding quality**, not the calibration map. If C0 shows no AUPRC headroom, **stop and ship nothing.** | A monotone post-hoc `g(belief)` cannot fix a resolution-limited score; it lowers ECE while collapsing resolution. |

---

## Verified facts (the anchors C1 fits)

Joined `data/benchmark/eval_curation_v1.jsonl` (gold, n=1606, balanced 801/803) to run outputs by `source_hash` (0 missing). Confusion cells (verified 2026-06-17):

| reader | `rand_corr` = P(gold=incorrect \| verdict=**correct**) — confirmed read is wrong | `rand_rej` = P(gold=correct \| verdict=**incorrect**) — rejected read is wrong |
|---|---|---|
| **medpsy-4B** | 231/950 = **0.243** | 83/656 = **0.127** |
| **gemma-26B** | 158/863 = **0.183** | 97/743 = **0.131** |

> **Correction folded in:** the original synthesis crossed these two columns. The verified reading: **confirmations are the bigger leak** — medpsy is wrong on ~24% of reads it confirms vs ~13% of reads it rejects (gemma 18% vs 13%). medpsy over-confirms; gemma is more balanced. The soft-weight on `verdict=correct` reads therefore does more work than first framed.

`rand_s`, `syst_s` are **not fit** — they stay `RECALIBRATED_PRIORS` (`noise_model.py:52-76`, e.g. `reach (0.462, 0.05)`).

---

## Status legend

`[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[?]` needs decision · `[R]` in review

---

## The heuristic (target of C2)

```
Per source s, per read j, verdict v_j ∈ {correct, incorrect, none}:
    w_j = rand_corr_m   if v_j = correct      (residual wrong-rate of a CONFIRMED read)
        = rand_rej_m    if v_j = incorrect    (residual wrong-rate of a REJECTED read)
        = rand_s        if v_j = none/abstain (identity fallback → today's behavior)

correlation guard:  n_eff = 1 + (n_s − 1)·κ,   κ ∈ (0,1]    (κ=1 ⇒ full independence)
source factor:      f_s = syst_s + (geomean_j w_j)^{n_eff}
statement belief:   1 − ∏_s f_s          (contradiction split unchanged, noise_model.py:204)
```

Reduce-to-current check: `v_j=None` ∀j and κ=1 ⇒ `f_s = syst_s + rand_s^{n_s}` (today, `noise_model.py:108`). This replaces today's *hard deletion* of `verdict=incorrect` evidence (which drops a lone-evidence source to factor 1.0) with a measured residual penalty.

---

## Substrate edges (foundation capabilities — each unlocks ≥2 stage nodes)

### E1 — Calibration-metrics harness
Reuse shipped `metrics.ece` + `BINS_8` (`src/indra_belief/metrics.py`) so every number is comparable to existing ECE (medpsy 0.139 / gemma 0.108). Add **Brier with Murphy decomposition** (`Brier = reliability − resolution + uncertainty`) and a reliability-diagram emitter. Unlocks: C0 curves, C1 Tier-1, C2 ship gate.

### E2 — Gold-join layer
Join `data/benchmark/eval_curation_v1.jsonl` (`gold`, `source_hash`, `pa_hash`) to run outputs `data/results/eval_curation_v1_{model}.jsonl` (verdict lives only here — gold file has no verdict column). Verified 1606/1606 join. Unlocks: C0, C1.

### E3 — Held-out split + contamination gate
Group-split by `pa_hash` (no statement straddles train/test). Held-out: `data/benchmark/holdout_cc.jsonl` (primary, n≈453, McNemar-powered), `holdout_v15_sample.jsonl` (secondary), `rasmachine_v1_gold.jsonl` (n=60, external direction check, **never a fit**). Gate with `scripts/check_contamination.py` (eval_curation_v1.meta.json already excludes 3515 leakage pairs). Unlocks: C1, C2.

### E4 — `verdict=None` byte-identity fixture
A golden-output test asserting the new `compute_gated_belief` path equals today's output when all verdicts are `None`. Same discipline that gated the metrics/sampling extractions. Unlocks: C2 (it cannot ship without this), C3.

### E5 — Per-run calibration-product export (the presentation seam)
Bump `export_meta.json` to `schema_version: 4` and write a `metrics.json` alongside `per_evidence.jsonl`/`per_statement.json` (`results.py:565-570`), keyed by `run_id`: `BINS_8` reliability bins `{lo, hi, n, mean_pred, empirical}` at Tier-1 and Tier-2, `ece`, Brier `{reliability, resolution, uncertainty}`, and per-run confusion `{tp, fp, fn, tn}` vs baked gold. Reuse E1's emitter so the persisted numbers are **byte-identical** to the C0 figures (golden-output check; a no-gold / all-`verdict=None` run writes a *named-empty* metrics block, per the doctrine — no imputed zeros). This executes the doc's own **D5 / C3.2** "travels with the run" commitment for the calibration products themselves — the reliability curve must travel with the run, not be a global figure file the viewer points at. Unlocks: the viewer presentation stages **C4 / C5** (which live in `research/belief_instrument_task_graph.md`).

---

## Stage C0 — Honest reliability curve, zero model change  ·  do→review with **G0**

**Aim:** produce the reliability curve we don't have today, and **decide whether C1–C3 are worth doing.** Pure analysis; no production code path touched. (~0.5 day)

- [x] **C0.1** Joined `eval_curation_v1_{medpsy,gemma}` to gold on the canonical (matches_hash, source_hash) pair (1606/1606, 0 missing). Anchors reproduce the verified table: MedPsy `rand_corr` 0.243 / `rand_rej` 0.127; gemma 0.183 / 0.131. Confidence mix degenerate (not fit).
- [x] **C0.2** Tier-1 per-evidence reliability of the grid `score` vs `gold==correct`: MedPsy ECE 0.139 / gemma 0.108; AUROC 0.810 / 0.843. Degenerate ~2 occupied bins, as expected (confidence collapse).
- [x] **C0.3** Tier-2 per-statement raw belief (gated belief over scored evidences, hard gate + RECALIBRATED_PRIORS, grouped by `pa_hash`, gold any-incorrect-wins): **AUROC 0.772 / 0.805, AUPRC 0.740 / 0.771 (base 0.514)** — clear headroom. **Finding that overturns the D8 premise:** the saturated, no-headroom belief was the **INDRA prior** (`indra_prior_reference` AUROC 0.710, ECE 0.385, mean 0.96/0.91), NOT our gated belief (mean belief 0.60 correct / 0.24 incorrect — well separated, ECE 0.15, resolution ~0.07–0.09). Reliability diagram shows systematic mid-range under-confidence (single correct reach/sparser reads → belief ~0.46 but ~65–72% correct) — a real recalibration target.
- [x] **C0.4** Shipped `scripts/calibration_stage0.py` (analysis-only, numpy + shared libs; implements AUROC/AUPRC/Brier-Murphy not in `metrics.py`) → `data/results/calibration_stage0.{md,json}`.
- [x] **G0** — **GO** (2026-06-17). Raw-belief AUPRC has headroom (Δ +0.226 MedPsy / +0.257 gemma over base; AUROC Δ +0.27 / +0.31). Discrimination is real and the belief is miscalibrated → C1–C3 have both signal to preserve and error to fix. Proceed to C1.

## Stage C1 — Two-parameter fit + Tier-1 validation  ·  do→review with **G1**

**Aim:** empirical proof the soft weights are calibrated, before any wiring. Still no production code change. (~0.5 day)

> **Data dependency (found at C0):** existing `data/results/*holdout_cc*` are older scorer architectures (CC/AA/monolithic *phases*), not medpsy-4B + gemma-26B under the current monolithic scorer — provenance doesn't cleanly match eval_curation_v1. A clean held-out validation needs a fresh `holdout_cc` scoring run by both readers (**LLM spend — gate at the user**). Interim, zero-cost option: a within-`eval_curation_v1` group-split by `pa_hash` (fit on train, validate Tier-1 on test) as a first check before committing to a held-out run.

- [x] **C1.1** Fit `(w_correct, w_incorrect)` per model on a stratified train split (`scripts/calibration_stage1.py`, seed=0). **Sign trap fixed:** the per-read weight entering the noise model is `w = P(read does NOT support | verdict) = P(gold incorrect | verdict)` — for a *rejected* read that is `1 − rand_rej ≈ 0.85` (high w → low belief), **not** `rand_rej`. Fit both conditionals directly from the cells.
- [~] **C1.2** **Interim within-`eval_curation_v1` validation done** (zero-cost, group-split by pa_hash) → `data/results/calibration_stage1.md`. **Authoritative confirmation PRE-REGISTERED + prepared** (`scripts/calibration_confirm.py`, committed before results = locked hypothesis: primary `soft_guard@k0.5`, gate ECE↓ AND AUROC not reduced, protocol fit-on-all-eval / test-on-holdout_cc). Runbook `scripts/score_holdout_cc.sh` (build → score gemma → swap gateway → score medpsy → confirm). **Awaiting the `holdout_cc` gateway run** (500 ev / 346 stmts; ~15min medpsy, ~1–4h gemma; local compute, gateway up — one model at a time). In-sample smoke (train=test) passes for both; real test is OOD transfer.
- [x] **C1.3** Confirmed zero abstain/None rows in eval_curation_v1 (join drops parse-nulls); `w_unscored` unfittable here, correctly falls back to `rand_s`.
- [~] **G1 — INTERIM verdict (seed=0 held-out split):** the plan's **committed form `soft_replace_k1.0` FAILS** — ECE improves (MedPsy 0.139→0.116, gemma 0.157→0.099) but **resolution and AUROC drop** (MedPsy AUROC 0.787→0.697), i.e. calibration bought by flattening. **Finding → design amendment:** the **`guard` variant** — `w_correct = min(rand_corr, rand_s)` (confirmation can only *lower* a read's error, never inflate a high-precision source; rejection unchanged) at **`kappa = 0.5`** — improves ECE **and** preserves/improves resolution + AUROC (MedPsy 0.139→0.106 ECE, 0.087→0.095 res, 0.787→0.799 AUROC; gemma 0.157→0.067 ECE). **BUT this variant is test-selected** → it must be **pre-registered** and confirmed on the independent `holdout_cc` run before C2 adoption. Not a stop; a refinement with a confirmation gate.

> **C2 design amendment (candidate, pending C1.2-authoritative confirmation):** adopt the `guard` survival weight (`w_correct = min(rand_corr, rand_s)`, `w_incorrect = max(1−rand_rej, rand_s)`) at `kappa ≈ 0.5`, not the straight `replace`. Principle: **confirmation only decreases a read's error rate; rejection only increases it** — straight replacement violates this for trips/signor and kills discrimination. Lower `kappa` also beat `kappa=1` across the board (full independence over-counts multi-read sources).
>
> **Robustness (10 resampled pa_hash splits, zero-cost):** the ECE gain is stable — MedPsy hard 0.136±0.010 → guard_k0.5 0.098±0.007; gemma 0.157±0.009 → 0.088±0.014 — with AUROC preserved/improved (MedPsy 0.780→0.791, gemma 0.800→0.804). MedPsy guard_k0.5 satisfies the strict G1 (ECE↓ AND resolution≥) in 90% of seeds; gemma "fails" it 0% **only on a noise-sized resolution dip** (0.090→0.087, sd ±0.008) while ECE halves and AUROC rises.
>
> **Proposed gate refinement (G1/G2):** replace the brittle "Brier-resolution not reduced" sub-criterion with **"AUROC not reduced"** — at n≈430 / 8 bins the per-bin resolution term is noise-dominated, whereas rank-based AUROC is stable. Under an AUROC-guarded criterion, **guard_k0.5 passes both readers**. (Keep resolution as a reported diagnostic, not the gate.)

## Stage C2 — Wire soft survival weight behind a flag  ·  do→review with **G2** (ship gate)

**Aim:** the actual model change, contained by E4 + a three-way baseline. (~1 day) Edits the formula that backs shipped numbers — risk is real, the golden test + ship gate contain it.

- [ ] **C2.1** Add `w_j` / `n_eff` into `compute_gated_belief` (`noise_model.py:299-326`) behind a flag; default off.
- [ ] **C2.2** Land E4 (`verdict=None` byte-identity golden test) — **blocks merge.**
- [ ] **C2.3** Grid-search `κ` on held-out (start 0.5).
- [ ] **C2.4** Tier-2 per-statement validation: group held-out reads by `pa_hash`, synthesize per-statement gold by **any-incorrect-wins** (`curation.py` rule). **Report the multi-evidence subset separately** — 58–81% of statements are singletons and just re-test Tier-1.
- [ ] **C2.5** Three-way baseline on identical held-out reads: `{hard-gate (production), parametric-only (ablation, composed_scorer.py:296), soft-survival (this)}`.
- [R] **G2 — SHIP GATE.** Ship iff: Tier-1 held-out ECE(soft) < ECE(hard) **AND** Brier-resolution not worse **AND** error-detection **F1** (lead metric, never accuracy on balanced gold) non-inferior to hard on `holdout_cc` **AND** the verdict=None byte-identity test passes. **Noise-floor caveat:** medpsy-4B err-F1 varies 0.717–0.871 across identical runs (spread 0.154) — any delta inside that band on small n is not real; require n≈453 holdout_cc + report CIs.

## Stage C3 — (Optional) post-hoc `g` per (model, corpus)  ·  do→review with **G3**

**Aim:** cheap insurance for residual scale-shift, applied *after* C2 so it maps the new belief distribution. **Deprioritized** (D8 — mostly lowers ECE without adding usable resolution). (~0.5 day)

- [ ] **C3.1** Fit a 1–2 param Platt/temperature map `g(belief)` per (model, corpus); isotonic only if n supports it. Identity-`g` fallback for one-line revert.
- [ ] **C3.2** Bake `g` into each run's export (travels with the run, per D5).
- [R] **G3** — `g` improves held-out per-statement ECE without reducing resolution; else ship identity (no-op) and document.

---

## Presentation handoff — viewer surfaces (live in `research/belief_instrument_task_graph.md`)

**The viewer is NOT prepared today** (verified 2026-06-17): `/runs/[run_id]` renders a residual-vs-INDRA histogram + `{mae, bias}` scalars (`Validity.svelte`, `queries.ts:113`), but `reliability`/`ece`/`brier` are **zero hits** in `viewer/src`; the fig6 "calibration card" is a **static** `rasmachine_belief_comparison.html` (generated by `scripts/fig6_gen_svg.py`), not a route; and no `metrics.json` home exists. E5 is the seam that hands products from this math arc to the viewer. The two presentation stages are **authored as Phase T6 (C4/C5) + substrate edge E5 in `research/belief_instrument_task_graph.md`** (2026-06-17) — that doc owns the viewer IA + perceptual doctrine; the nodes keep the `C#`/`G#` vocabulary here for cross-reference. Device: **route = register** (`/runs/[run_id]` = single-run palette; `/compare` = A/B/gold comparison grammar). Both consume E5; neither recomputes.

- **C4 — Run-on-its-own calibration surface** → extend `/runs/[run_id]` (it **shipped** — the belief-instrument "deferred" note is stale). New `ReliabilityDiagram` + `BrierBar` + `ConfusionMosaic` primitives read `metrics.json`; one reactive lever = **Tier toggle** `?tier=ev|stmt` (Tier-1 per-evidence and Tier-2 per-statement shown as two *labeled, stacked* diagrams, never merged); ECE headline + Brier-as-stacked-bar make the D8 resolution story visible; per-stratum ECE retires the `unavailable` apology (`queries.ts:171`). **Renders C0's product.** → review gate **G4**: served numbers == `metrics.json` byte-exact (no served-vs-persisted drift); single-palette register; named-empty on no-gold, no crash.
- **C5 — Run-comparison calibration mode** → add `?mode=calib` to `/compare` (mirror `?mode=gold`, `compare/+page.svelte:94`). L0 anatomy swaps the verdict 2×2 for **overlaid reliability curves + ΔECE/ΔBrier**; the **three-way baseline** (hard / parametric / soft) renders as three series tied to G2; L1–L3 strat/cohort/reasoning drill reused **unchanged** (no second confusion-matrix). **Renders C2.5's product.** → review gate **G5v**: viewer ΔECE/ΔBrier == C1/C2 held-out script outputs for the same run pair; drill skeleton unchanged; comparison register (A/B/gold hues) preserved.

**Decision-gating:** a **NO-GO at G0** (D8) costs ZERO viewer work — C4/C5 never fire without a product to show. The math arc is gated by G0–G3; the rendering arc by G4/G5v.

---

## Review gates

| Gate | Cycle | Tool | What it checks |
|---|---|---|---|
| G0 | after C0 | analysis + (optional) brutalist | **GO/NO-GO**: does raw-belief AUPRC have headroom? No ⇒ stop, lever is upstream. |
| G1 | after C1 | analysis + paired bootstrap | Tier-1 held-out ECE↓ vs hard-gate AND Brier-resolution preserved, CI excludes 0. |
| G2 | after C2 | three-way baseline + golden test | **Ship gate**: ECE↓ AND resolution≥ AND err-F1 non-inferior on holdout_cc AND verdict=None byte-identity passes; respect noise floor. |
| G3 | after C3 | analysis | post-hoc `g` improves per-statement ECE without resolution loss; else identity. |
| G4 | after C4 *(in belief_instrument)* | svelte-check + golden | run-on-its-own surface: served numbers == `metrics.json` byte-exact; single-palette register; named-empty on no-gold. |
| G5v | after C5 *(in belief_instrument)* | svelte-check + golden | run-comparison mode: viewer ΔECE/ΔBrier == C1/C2 held-out outputs; drill skeleton unchanged; comparison register preserved. |

---

## Open questions

- **Q1** `κ` parameterization: geometric-mean-then-exponentiate vs an explicit effective-count discount — settle empirically at C2.3, or fix κ=1 (no correlation guard) if held-out shows no overconfidence from multi-read sources.
- **Q2** Per-statement gold synthesis (any-incorrect-wins) is honest but mostly a no-op on singletons. Is the ~20–40% multi-evidence subset large enough on holdout_cc to power a Tier-2 claim, or is Tier-2 reporting-only?
- **Q3** Do we fit a third reader (`bedrock-*`) now, or only the two locally-served models until Bedrock inference is paid-for?

---

## Key file anchors

- `src/indra_belief/noise_model.py:48-49` (`rand = 1 − accuracy − syst`), `:52-76` (`RECALIBRATED_PRIORS`), `:108` (additive factor `syst + rand^n`), `:204` (contradiction `dominant·(1−opposing)`), `:299-326` (`compute_gated_belief`)
- `src/indra_belief/composed_scorer.py` (hard gate), `:296` (parametric-only ablation)
- `src/indra_belief/scorers/_shared.py:10-17` (`VERDICT_SCORE_GRID` — display-only, do not fit)
- `src/indra_belief/metrics.py` (`ece` / `BINS_8`)
- Gold: `data/benchmark/eval_curation_v1.jsonl` · Runs: `data/results/eval_curation_v1_{medpsy,gemma}.jsonl` · Holdouts: `data/benchmark/holdout_cc.jsonl`, `holdout_v15_sample.jsonl`, `rasmachine_v1_gold.jsonl` · Contamination: `scripts/check_contamination.py`
