# Deploying the belief model: the two bulk domains, the realtime path, and what blocks each

> **Provenance (2026-09-01):** citations of `research/serving_architecture.md`, `research/serving_deployment.md` and `Dockerfile.live` below name files removed from the tree (the docs as OBSOLETE-TOPOLOGY; the image as an unconsumed capability record); git history holds them. The measurements quoted from them are unchanged.

> **Provenance note (2026-08-25).** This is a historical record. The repository
> was later cut to prod infrastructure plus the future-LLM compare loop, so some
> paths cited below name code that no longer exists; `research/README.md` tracks
> which documents still describe live code. The measurements and conclusions
> stand as recorded; git history holds the code they were measured against. No
> guard checks these citations any more.

Written 2026-08-07. Ben Gyori named two domains where bulk statements exist —
**MSstatsBioNet** and **EMMAA** — and a third mode: scoring live as scientific users
query statements. This document sizes all three against the actual INDRA stack, states
what we already have, and names what has to be decided before anything is built.

> **Since written:** the corpus-scale path this document sizes now exists end to
> end — prepare, score, calibrate, believe — and its operating procedure is
> `research/corpus_belief_runbook.md`. That runbook supersedes this document on
> HOW to run the bulk path; this one remains the argument for WHY, and for the
> realtime question it is still the only treatment of.

**How to read the evidence.** Every claim is marked:

* **[M]** measured here, with the command or artifact that produced it.
* **[V]** verified by reading code at a named file and symbol.
* **[R]** reasoned from [M]/[V]. Never given to more significant figures than it has.
* **[?]** open — named because absence of an answer is itself a finding.

Citations name a file and a symbol, never a line number, per this repo's convention;
`scripts/check_doc_anchors.py` validates in-tree symbols.

**This is a plan, not an approved proposal.** Nothing in §7 is authorised, and §8 lists
four questions that must go to Ben before any of it is costed for real.

---

## 1. The one-paragraph version

Belief has exactly one origin in the whole INDRA stack, and it is a `{hash: float}`
file. Everything downstream — CoGEx, INDRA DB, network search, MSstatsBioNet — reads a
frozen copy of that file. So the deployment is not an integration project; it is a
question of which copy we write and whether we write beside the incumbent or over it.
Bulk scoring the entire published EMMAA corpus costs **$600** and about nineteen hours.
Bulk scoring everything a user can query costs **$12,719** and is bounded by provider
token quota, not by our concurrency. The realtime path should serve a **lookup**, not an
inference: a median statement takes 52 s to score in today's serial shape, and one real
MSstatsBioNet query returns 4,447 statements. What blocks all of it is smaller than the
compute: one calibration profile, no fleet-wide spend cap, no provenance field on the
number we would publish, and an unasked licensing question.

---

## 2. The two named domains, measured

### 2.1 MSstatsBioNet — the live consumer that already discards a belief

**What it is.** [M] `MSstatsBioNet` is a Bioconductor package from Olga Vitek's lab at
Northeastern that joins MSstats differential-abundance output to INDRA prior-knowledge
networks. **Gyori is a co-author on the preprint** (bioRxiv `10.64898/2026.07.09.737605`,
posted 2026-07-16, Vitek corresponding). [M] The repo (`Vitek-Lab/MSstatsBioNet`) was
last pushed 2026-07-30 on branch `devel` at version 1.5.2, ahead of the 1.4.1 in
Bioconductor 3.23 — active, small (2 stars, 3 open issues), a research package.

**How it queries.** [V] `.callIndraCogexApi` in `R/utils_getSubnetworkFromIndra.R` POSTs
`{"nodes": [[namespace, id], ...]}` to
`https://discovery.indra.bio/api/indra_subnetwork_relations`. [V] It touches four INDRA
services in total: `discovery.indra.bio` (subnetwork relations, and
`get_evidences_for_stmt_hash` from `.query_indra_evidence` in
`R/filterSubnetworkByContext.R`), `db.indra.bio` (`/curation/list/<hash>` and
`/statements/from_agents`), and `https://grounding.indra.bio` (`/ground_multi`).

**The finding.** [M] The CoGEx response carries a `belief` field on **every** relation —
4,447 of 4,447 in a probe over 20 well-studied human genes. [V] `grep -rn belief R/`
over the package returns **nothing**. It is handed a confidence score and throws it away.

[V] What it gates on instead, in `.filterIndraResponse`: `evidence_count_cutoff` default
**1**, `paper_count_cutoff` default **1**, `correlation_cutoff` 0.3, an optional
`sources_filter` and `statement_types`, and an opt-in `filter_by_curation` that defaults
to **FALSE**. At defaults that is effectively no confidence filter at all — every
single-mention text-mined statement is admitted. [M] Median belief of the relations
returned by that probe: **0.436**.

That is the reputation-vs-reading substitution in a shipping downstream consumer, and it
is the cheapest possible integration surface: the package already receives the field.

**Caveat on the probe.** [M] The 4,447 came from a deliberately dense neighbourhood of
well-studied cancer and signalling genes. A real differentially-abundant protein list
from an MSstats run would be sparser. The number sizes an upper bound on one query, not
a typical one. [?] Nobody has run the package on a real dataset and recorded the
distribution.

### 2.2 EMMAA — 17 models, censused directly

**The corpus.** [M] Downloaded and parsed every one of the 17 publicly readable
`assembled/<model>/statements_<date>.jsonl` files from `s3://emmaa`, each retried until
its byte count matched the object's `Content-Length` — **all 17 byte-exact, zero
unparseable lines**:

| model | statements | evidences | text % | ev/stmt med | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| covid19 | 577,661 | 1,159,032 | 87.8% | 1 | 5 | 537 |
| painmachine | 284,045 | 644,095 | 100.0% | 1 | 6 | 2,252 |
| ms | 15,320 | 56,194 | 100.0% | 2 | 9 | 453 |
| brca | 10,631 | 55,134 | 97.5% | 2 | 13 | 3,438 |
| rasmachine | 8,663 | 47,192 | 85.2% | 2 | 15 | 3,461 |
| luad | 6,258 | 32,103 | 99.2% | 2 | 14 | 956 |
| covid19_map | 7,344 | 32,043 | 81.3% | 1 | 16 | 201 |
| nf | 9,560 | 23,122 | 75.9% | 1 | 6 | 109 |
| aml | 4,795 | 22,815 | 97.3% | 2 | 12 | 341 |
| prad | 4,644 | 16,530 | 99.9% | 2 | 8 | 397 |
| skcm | 3,251 | 14,983 | 99.4% | 2 | 12 | 348 |
| paad | 3,046 | 10,816 | 100.0% | 2 | 8 | 430 |
| vitiligo | 3,746 | 4,898 | 100.0% | 1 | 3 | 27 |
| food_insecurity | 904 | 1,563 | 100.0% | 1 | 4 | 9 |
| covid19_inflammasome | 53 | 276 | 77.5% | 2 | 12 | 14 |
| rasmodel | 257 | 257 | 100.0% | 1 | 1 | 1 |
| marm_model | 25 | 25 | 100.0% | 1 | 1 | 1 |
| **TOTAL** | **940,203** | **2,121,078** | **92.4%** | | | |

[M] Mean 2.26 evidences per statement. **92.4% of evidences carry text** and only
**7.8%** come from databases rather than readers — so essentially the whole corpus is in
this scorer's domain. Two models are 85% of it: `covid19` and `painmachine`.

**A methodological correction, recorded because I got this wrong first and the wrong
number nearly shipped.** An earlier pass in this investigation streamed these files
through `curl | python` without checking the byte count, and reported 558,156 statements
/ 1,331,236 evidences. **That pass silently truncated on the two large files** — it saw
348,229 of `covid19`'s 577,661 statements — and I used it to "correct" an independent
probe that had reported 935,882 / 2,092,808. The independent probe was substantially
right and my correction of it was wrong. The table above is the byte-verified recount and
is the number to use. The lesson generalises past this document: a streamed HTTP body
that ends early is indistinguishable from a short file unless you compare against
`Content-Length`, and an undercount reads as a clean measurement.

[M] Two independent cross-checks that did hold throughout, because `ms` is small enough
that no stream truncated: the `ms` dashboard's own `model_summary` reports
`number_of_statements: 15320`, ships a 15,320-element `assembled_beliefs` array, and
lists `sources` as `reach 42665, sparser 11014, eidos 2459, isi 32, trips 24` — all three
matching the file. And the `latest_statements_<model>.json` and dated
`statements_<date>.jsonl` families agree on content: for `ms`, 15,336 vs 15,320
statements with 15,320 `matches_hash` values in common and **zero** dated-only; the 37%
byte difference is pretty-printing, not content.

**There is a Multiple Sclerosis model.** [M] `ms`, "A self-updating model of multiple
sclerosis", NDEx `b02b8bcf-3425-11eb-9e72-0ac135e8bacf`. [M] It is 100% reader-sourced
(reach/sparser/eidos/isi/trips, no database evidence) and 100% of its evidences carry
text — the cleanest bulk target in the whole set. Note this is *not* "ms-stats bionet";
that is MSstatsBioNet, and the collision of the two "MS"es is a coincidence worth stating
out loud so nobody conflates them later.

**How belief gets there.** [V] `EmmaaModel.run_assembly` builds an INDRA
`AssemblyPipeline` from `config['assembly']['main']` and runs
`indra.tools.assemble_corpus.run_preassembly(belief_scorer=…)`, which constructs
`BeliefEngine(scorer=…)` and calls `set_prior_probs` then `set_hierarchy_probs`. [V] With
no `belief_scorer` in a model's config the default is INDRA's `SimpleScorer`
(`default_scorer` in `indra/belief/__init__.py`), whose reach priors are `syst 0.05,
rand 0.3`. [V] An alternate scorer is loadable from S3 via `load_belief_scorer(bucket,
key)` in `emmaa/model.py` — the documented example is `s3://indra-belief/1.20.0/default_scorer.pkl`.
**That function is the EMMAA seam**: it is a registered pipeline step that unpickles a
scorer, so a config change is the whole integration.

**The incumbent number is near-degenerate.** [M] Belief histogram over all 940,203
statements:

| belief | statements | share |
|---:|---:|---:|
| **0.65** | 546,799 | **58.16%** |
| **0.86** | 141,754 | **15.08%** |
| 0.89 | 52,049 | 5.54% |
| 0.923 | 44,255 | 4.71% |
| 0.9419 | 21,382 | 2.27% |

**73.24% of every statement in EMMAA carries one of exactly two values**, and those two
are precisely `SimpleScorer`'s outputs for one and two reach evidences:
`1 − (0.05 + 0.3) = 0.65` and `1 − (0.05 + 0.3²) = 0.86`. The incumbent scalar is
overwhelmingly re-encoding evidence count. That is the clearest single argument for
deploying a reading-based number here, and it is measured, not argued.

**Two constraints that change the cost model.** [V] `run_preassembly_with_extra_evidence`
pulls up to `ev_limit=1000` additional evidences per statement from the INDRA DB, scores
against the enriched set, then keeps only the resulting number and discards the evidence
— so **published belief is not reproducible from published evidence**. [M] Corroborated:
`ms` has 1,442 statements with exactly one published evidence and every one has belief
≥ 0.86, impossible from a single reach evidence at 0.65. And [V] EMMAA re-scores the
**full corpus** on every run, not the delta.

**It is dormant.** [M] Every model's latest stats date is 2024-08-01 or earlier
(`brca` 2024-02-10, `covid19_inflammasome` 2021-08-23), and `gyorilab/emmaa` master's
last commit is 2024-07-19. [M] `gyorilab/indra` is active (pushed 2026-06-30). EMMAA is a
frozen corpus, not a running pipeline. That is good news for a one-shot bulk score and
bad news for anyone assuming a daily cadence to hook into. [?] Whether it is intended to
restart is a question for Ben.

### 2.3 The real denominator: users can query anything

The two named domains are not the coverage requirement. A scientific user querying INDRA
can reach the whole assembled corpus, so the population that must carry a belief is the
whole corpus.

[M] Our pinned figure is **44,944,056 grounded/assembled evidence rows** — the CoGEx
2025-09-16 dump, `s3://bigmech/.../20250916/nodes_Evidence.tsv.gz`, recorded with a
sha256 in `frozen_sampling_frame` and matched by `SOURCE_POPULATION_ROWS` in
`scripts/reservoir_sample_cogex.py`. [?] Whether `db.indra.bio` exposes a larger
population than CoGEx's assembled set is not established and must be checked before this
number is used as the budget.

EMMAA is **4.7%** of that. MSstatsBioNet queries a subset of it per call. The gap between
"the two named domains" and "everything queryable" is a factor of 21, and it is the
single largest driver of the plan's cost.

---

## 3. The INDRA stack: one origin, three read paths

**Belief is computed in exactly one place.** [V] `calculate_belief` in
`indra_db/readonly_dumping/export_assembly.py` runs a `HybridScorer(CountsScorer,
SimpleScorer)` whose random-forest half is unpickled from
`s3://indra-belief/1.20.0/sk141_hybrid_rf_2kd13_cs.pkl`, and emits **`belief_scores.pkl`
= `{stmt_hash: belief}`**. Everything downstream reads a frozen copy:

* [V] INDRA DB stores it in `readonly.belief (mk_hash BigInteger PK, belief REAL)` —
  float32, so served beliefs are quantized to ~7 significant digits — and denormalizes it
  into `SourceMeta`, `NameMeta`, `TextMeta`, `OtherMeta`, `MeshTermMeta`,
  `MeshConceptMeta` and `AgentInteractions`.
* [V] **INDRA CoGEx does not compute belief at all.** It downloads the same
  `belief_scores.pkl` from `s3://bigmech/indra-db/dumps/cogex_files/<timestamp>/` and
  writes it verbatim onto the Neo4j `indra_rel` relation as a `belief:float` property and
  into `stmt_json["belief"]`.

**Three query-time read paths, all live-verified** [M] (`emmaa.indra.bio` 200,
`db.indra.bio` 302, `discovery.indra.bio` 200, `network.indra.bio` 200 on 2026-08-07):

| host | how belief is used |
|---|---|
| `db.indra.bio` | returns a `belief_scores` dict on every statement response; supports `sort_by=belief` |
| `discovery.indra.bio` (CoGEx) | filters with `r.belief > $minimum_belief` in Cypher across search and gene-set apps |
| `network.indra.bio` | `belief_cutoff` on `/query` and `/multi_interactors`; returns `belief` on every `EdgeData`/`StmtData` |

**One artifact, one file, seven denormalised tables, three services.** That is the entire
blast radius, and it is why the intervention is small even though the corpus is large.

**Our socket already fits.** [V] `LLMBeliefScorer` in `src/indra_belief/belief_scorer.py`
subclasses INDRA's `BeliefScorer` when indra is importable, and [M]
`isinstance(scorer, BeliefScorer)` is True with `BeliefEngine(scorer=…)` accepting it past
its assert. So `BeliefEngine(scorer=LLMBeliefScorer(client))` is a drop-in at both the
EMMAA seam and the indra_db seam.

---

## 4. What we already have

Three things that are usually the hard part are done.

**A full EMMAA model, scored end to end.** [M] `data/results/rasmachine_mono_medpsy_remote_direct.jsonl`
is 47,434 lines — one per (statement, evidence) pair — over the whole `rasmachine` model.
This is not a pilot; it is the second-largest EMMAA model by evidence count, complete.

**A documented bulk-delivery format, keyed on the right join.** [M]
`data/exports/rasmachine_belief/` ships `per_evidence.jsonl` (47,434 rows),
`per_statement.json` (8,716), `export_meta.json` with provenance and join-quality
counters, and a written `SCHEMA.md`. [M] Its identity fields include
**`indra_matches_hash`** — the same key `readonly.belief` uses as `mk_hash` and CoGEx
uses on `indra_rel`. The delivery contract for both seams already exists and has been
exercised once, including its failure modes (61 source-hash divergences on empty-evidence
rows, 5 statements where one `stmt_hash` collapses multiple INDRA statements).

**A measured cost and latency model.** §5 and §6 below rest entirely on it.

---

## 5. The bulk track

### 5.1 Unit economics

[M] Both published run costs reproduce **to the cent** from raw token counts at full list
price, summing every call-log entry's `prompt_tokens`/`out_tokens` against
`data/comparison/pricing.json`: verdict-only $9.4453 against a recorded
`provider_measured_spend_usd` of 9.44532822, reasoning $26.3650 against 26.36515929. Over
33,361 (statement, evidence) pairs on gemma-4-26b:

| | $/evidence | serial s/evidence |
|---|---:|---:|
| reasoning | 0.000790 | 10.11 |
| verdict-only | 0.000283 | 1.033 |

[M] The serial figures are re-measured by summing every `duration_s` across both attempts
files: **93.722 h** reasoning over 50,484 calls, **9.576 h** verdict-only over 32,315
timed calls. `research/serving_architecture.md` §2's 82.8 h / 9.5 h **does not
reproduce** and should not be used; the reasoning arm is understated there by ~12%.

[M] A correction worth carrying: the provider's prefix cache is on and hitting, but the
mantle endpoint **bills cached prompt tokens at full rate**. If it discounted them, the
naive full-price sum could not land on the billed figure. The cache buys prefill latency
and **exactly $0**. This cuts both ways — the projections below are safe because cache
hit rate cannot move them, and prompt caching remains an entirely *unrealized* cost lever
rather than a spent one.

### 5.2 What each domain costs

| corpus | evidences | verdict-only | reasoning |
|---|---:|---:|---:|
| EMMAA `ms` | 56,194 | $16 · 0.5 h @32-way | $44 · 4.9 h @32-way |
| EMMAA `rasmachine` | 47,192 | $13 | $37 (**already done**) |
| EMMAA, all 17 | 2,121,078 | **$600 · 19.0 h @32-way** | $1,676 · 7.8 d @32-way |
| CoGEx 2025-09-16 | 44,944,056 | $12,719 · 403 h @32-way | $35,506 · 164 d @32-way |

[R] The dollar figures are linear in the measured per-evidence cost and are the most
trustworthy numbers here. The wall-clock figures are serial-time divided by worker count
and are **wrong in an important way** — see §5.3.

**Read the table this way:** the whole of EMMAA is a rounding error and can be scored at
reasoning grade this week. Full-corpus coverage is affordable at verdict-only and is
**not** reachable at reasoning grade on the concurrency we have ever demonstrated.

### 5.3 Concurrency is not the binding constraint — token quota probably is

[M] The published runs saw zero 429s at 32 concurrent (4 arms × 8 workers). That is not a
measured ceiling; it is the highest number we ever tried.

[R] The arithmetic nobody had done: at the measured **2,204.6 prompt tokens per call**, a
full-corpus verdict-only pass is **~99.1B input tokens**.

| account TPM | full-corpus verdict-only wall clock |
|---:|---:|
| 1M | 1,651 h (68.8 days) |
| 2M | 826 h (34.4 days) |
| 10M | 165 h (6.9 days) |
| 50M | 33 h (1.4 days) |

Against a plausible 2M TPM the quota-limited time is 826 h regardless of worker count —
twice the 403 h that concurrency arithmetic predicts. **TPM, not concurrency, is likely
what bounds a corpus-scale run, and this account's actual quota has never been read.**
Bedrock enforces per-model, per-region, per-account TPM and RPM shared across the
Converse/InvokeModel family; increases are requestable but have lead time. EMMAA at
**4.68B** input tokens is ~39 h at 2M TPM — within a weekend either way, so the quota
question only bites at full-corpus scale.

### 5.4 Three multipliers that are in nobody's estimate

**The `BeliefEngine` scores twice.** [M] Driving a counting scorer through
`indra.tools.assemble_corpus.run_preassembly`: `set_hierarchy_probs` re-scores every
statement after `set_prior_probs` and **overwrites** it, giving a **2.0× floor** on
per-evidence calls (measured 18 calls where `estimate_calls()` predicted 9), rising
toward 3.0× as refinement density rises (27 calls on a fully-nested 3-statement chain).
`run_refinement=False` is the only 1.0× path. **If the deployment goes through
`run_preassembly` — which is both the EMMAA seam and the indra_db seam — double every
figure in §5.2 before showing it to anyone.**

[?] And there is a research gap underneath: on the hierarchy pass, `extra_evidence` is
*other statements'* evidence, so our scorer would judge evidence extracted for
`Phosphorylation(A, B, T, 185)` against the general claim `Phosphorylation(A, B)`. We
have never evaluated the scorer in that configuration. It may be fine; it may
systematically over-reject.

**The runner cannot execute a corpus-scale job.** [M] `ReplayIndex.load` costs 13,169
bytes per row, so a 137 GB machine is exhausted at ~10.4M rows before any resume state.
[M] Preflight amplifies 3.6 GB of resume files to 8.3 GB resident (2.6×), and
`prepare_run` does it for **every action in the plan**, not just the one being run. [R]
Durable state at full-corpus scale is on the order of 4.85 TB verdict-only. [M] Sharding
is blocked — `research/serving_architecture.md` §10 works the problem and its verdict is
*do not implement*. This is a rewrite of unknown size, and until it is scoped no schedule
above ~10M rows is credible.

**Capping evidences per statement does not rescue the budget.** The serving doc's table
says a ≤8 cap serves 57% of statements for 9.2% of evidences — but [M] that table is from
the paper corpus, which was **selected for high evidence counts** (mean 19.75). Measured
on real corpora:

| corpus | mean ev/stmt | evidence bill at cap 8 |
|---|---:|---:|
| paper corpus | 19.75 | 26.6% |
| EMMAA rasmachine | 5.45 | 54.5% |
| EMMAA ms | 3.67 | **81.7%** |

On thin distributions the bill is dominated by the many 1–2-evidence statements. **A cap
is tail control, not a budget lever** — it bounds the 3,461-evidence outlier that would
hang a live request, and saves almost nothing at corpus scale. Any plan that funds itself
on a cap is wrong.

---

## 6. The realtime track

### 6.1 Inference in the request path does not work

[M] Real per-statement makespan, measured by grouping all 33,361 attempts under their
1,689 statements and simulating k-worker scheduling:

| arm | serial p50 | serial p90 | serial max | k=8 p50 | k=32 p50 |
|---|---:|---:|---:|---:|---:|
| reasoning | 52.0 s | 373.0 s | 9,582.2 s | 16.2 s | 15.5 s |
| verdict-only | 5.9 s | 38.7 s | 1,330.4 s | 1.4 s | 1.1 s |

[M] Parallelism saturates by k=8 because one slow attempt sets the floor. [V] And today's
live path is fully serial anyway: `score_statement` in
`src/indra_belief/scorers/monolithic/__init__.py` is a list comprehension over evidences,
and `LLMBeliefScorer.score_statements_detailed` is a doubly-nested serial loop with no
concurrency and no cache.

[R] But the decisive number is not per-statement. One MSstatsBioNet subnetwork call
returned 4,447 statements. At EMMAA-like density that is ~16k evidences — roughly 25
minutes and ~$4.60 even at verdict-only with 8-way concurrency. **Per query.** No
interactive product survives that.

### 6.2 Serve a lookup

**Tier A — the belief sidecar.** `GET /v1/belief?matches_hash=…` returning our score, the
incumbent score, and the profile that produced ours. Single-digit ms, no provider call,
no grounding, no spend. This is the whole realtime story for scientific users, and it is
adoptable because both ends already exist: consumers hold `matches_hash` and already
receive a `belief` field, and §4's export is already keyed on `indra_matches_hash`.

**Tier B — bounded live scoring**, for statements not in the table. `POST /v1/score` on a
pair; statement-grain admitted **only under a declared evidence cap** (this is where §5.4's
cap belongs — as tail control, not economy). [M] The distribution makes the cap cheap:
≤8 evidences fully covers 94.1% of `ms` statements and 90.5% of `rasmachine` statements.

**Tier C — async**, above the cap: a job ticket, reusing the runner, resume and spend
machinery built for exactly this.

[V] The endpoint shape is already worked out in `research/serving_deployment.md` §5.4,
including the removals: `call_log` out of the response body ([M] 19,675 of 21,699 bytes),
`selected_examples` out keeping ids, the three dead accessors dropped, and
`score_statement(statement, client)` explicitly declared "not a serving verb".

**Packaging.** [M] A prepared-payload serving mode runs in the 535 MB batch image at 26 MB
RSS, because importing the monolithic scorer pulls in neither gilda nor indra (0.028 s,
both absent from `sys.modules`). [V] A raw-Statement mode cannot, because
`ScoringRecord.__post_init__` calls `resolve_entities` unconditionally — that is the one
line that costs a 993 MB image and a measured 7,133 MB peak RSS. [V]
`prepare_from_replay_row` in `src/indra_belief/prepared_execution.py` is already the
prepared-payload entry point.

### 6.3 Partial coverage breaks ranking — this is a correctness bug, not a gap

Every consumer read path in §3 is a **threshold** or a **sort key**. Our score and the
incumbent are different scales: [M] CoGEx serves 0.4357 for a single reach evidence where
`SimpleScorer` serves 0.65 and our calibrated scalar serves 0.891. A result set mixing
both is not sortable and not thresholdable — the cutoff means two different things within
one list.

**So a partial table must ship as a separate field that a consumer selects wholesale,
never merged into the incumbent `belief`.** Merging is only safe at 100% coverage. This is
the strongest structural argument for the augment-don't-replace option in §7.

### 6.4 The cache question, and how backfill inverts it

[M] Within our corpus the prompt digest `main_prompt_base_sha256` repeats 8.77%, and
**every repeat is intra-statement, zero cross-statement** (98.9% of them the same sentence
read by different `source_api` readers). `evidence_json_sha256` repeats 0.00%. The famous
90.3% figure is entity-grained grounding, not evidence — do not conflate them. So a
request cache buys ~8% fan-out dedup and nothing across queries.

[R] **But that analysis is of the wrong cache.** Both candidate seams recompute the full
corpus every cycle — EMMAA by `run_assembly`, indra_db on every readonly regeneration.
The dominant reuse is *across re-runs*, where hit rate approaches 100% because only the
delta is new. A **persistent verdict store** has completely different economics from a
request cache, and only the latter was analysed. Without one, every cycle pays full
freight; with one, the recurring bill is the delta rate.

[?] The cache key is undecided and the choice forces the packaging decision: the prompt
digest is the right key but requires the rendered prompt, which requires grounding;
content-addressing on `(evidence content hash, profile_id)` avoids grounding but loses
cross-reader dedup.

[?] Cross-*query* hit rate — the thing that would let Tier B warm Tier A — is
unmeasurable here because there is no query log.

---

## 7. What blocks this, in priority order

**B1 — Exactly one calibration profile is servable.** [V] `_FITTED_CONFIGS` in
`src/indra_belief/calibration_constants.py` keys profiles on `(canonical model,
prompt_sha256)` and holds three entries. [M] The in-tree `SYSTEM_PROMPT` hashes to
`c6845ab4…`, not the pinned `b4463821…`, so `gemma_remote` and `medpsy_remote` are keyed
to a string that no longer exists in the tree. **`bedrock-gemma-4-26b` on variant
`disconfirm_relnature_rf` is the only configuration that resolves.** [M] A miss is
*silent* and costs 0.403 of belief on a single confirmed reach read (0.488 vs 0.891), and
nothing in `src/` checks at serve time. **A deployment must assert profile resolution at
startup.**

**B2 — Verdict-only, the single biggest cost and latency lever, is unreachable from the
kernel.** [V] `VERDICT_ONLY_SYSTEM_PROMPT` lives in
`src/indra_belief/scorers/monolithic/_prompts_verdict_only.py` but its only importer is
`scripts/build_verdict_only_replay.py`; [V] `scorer.VARIANTS` holds exactly four keys and
none is `verdict_only`. Every verdict-only figure in §5 describes a run the library
cannot currently reproduce. And registering it is not just a prompt change — profiles key
on `prompt_sha256`, so a new prompt needs a **new calibration fit** (B1).

**B3 — There is no fleet-wide spend cap.** [V] `acquire_spend_lane_lock` takes
`flock(LOCK_EX|LOCK_NB)`: one process per ledger, so N workers means N independent
budgets and no aggregate. A five-figure run cannot be signed off without one, and an open
Tier B endpoint cannot exist without one. This is a service, not a file.

**B4 — A deployed belief carries no identity.** [V] `StatementBelief` in
`src/indra_belief/statement_belief.py` has 15 fields and none names the model, prompt
sha, profile, or scorer version; `as_dict()` emits none. Downstream is worse — a bare
float in INDRA's socket, a single `REAL` column in `readonly.belief`, a float property on
`indra_rel`. Nobody can ask "what produced this number", mixed-version corpora are
undetectable, and selective invalidation is impossible. **A `belief_provenance` sidecar
keyed `matches_hash → (profile_id, prompt_sha256, model_id, scorer_version, scored_at)`
is the cheapest thing to add now and the most expensive to retrofit.**

**B5 — Licensing is unasked, and it is a real exposure.** PHI is a non-issue — this is
published literature, not clinical records — and the plan should say so and move on. TDM
licensing is not. [M] On our corpus-representative CoGEx sample (n=5,000): **26.5% carry
an Elsevier `PII`** in `text_refs`, **39.5% of texts exceed 200 characters** (p50 179,
p90 309, max 8,447), and all 71 medscan rows carry text — while `db.indra.bio`
deliberately redacts MedScan evidence text from public responses. Elsevier's TDM terms cap
distributed snippets at 200 characters. Three questions: what agreement covers INDRA's
Elsevier full text, does it permit transmission to a third-party processor, and does
MedScan text carry distinct terms. [R] Our durable artifacts embed full evidence text at
105–223 KiB per evidence, so a corpus-scale pass produces multi-TB of files each
containing licensed source text — publishing or sharing any of it is a distribution
event. **This goes to the lab and probably to research compliance before a bulk run, not
after.**

**B6 — Swapping the scale silently re-points every downstream threshold.** `minimum_belief`
(CoGEx Cypher), `belief_cutoff` (network search), `min_belief`/`max_belief`/`sort_by=belief`
(EMMAA dashboard), `filter_belief(cutoff)` in assembly configs, `default_belief_threshold
= 0.95` in `indra/tools/machine`, plus every threshold in a published paper or a user's
saved query. And [V] `filter_by_curation(update_belief=True)` sets `belief = 1` for
curated-correct statements and is present in the live-shaped `covid19` config — it would
overwrite our output. Needed: an enumeration, and a decision between a monotone remap onto
the incumbent's scale (preserves thresholds, destroys the calibrated probability's
meaning) and the raw number (breaks them).

**B7 — There is no labeled holdout on the population that would be rescored.** Every
accuracy claim rests on `eval_curation_v1` (n=1,606, balanced) and the external-curator
gold (n=578). [M] The one corpus-representative draw,
`data/corpora/cogex_evidence_sample.jsonl` (5,000 of 44.9M), carries **zero labels**. And
the metric is wrong for the point of use: every consumer is a threshold or a sort key, so
the decision metric is **precision/recall at the operating point on a representative
sample**, not err-F1 on balanced gold. No such measurement exists. [R] Cost to close:
label 500–1,000 rows of the representative sample — the `/curate` viewer and per-curator
auth already ship — and score them verdict-only for about **$0.28**. This is the
highest-value undone item and it gates the go/no-go.

---

## 8. The decision that has not been made, and the questions behind it

**Augment, or replace?** Three seams exist, with different owners and costs 100× apart:

| seam | what changes | owner | blast radius |
|---|---|---|---|
| `export_assembly.py::calculate_belief` | the number **everything** reads | indra_db operator | full corpus, 7 denormalised tables regenerated |
| EMMAA `config.json` → `load_belief_scorer` | 17 models | EMMAA operator (S3 config is 403 to us) | 558k statements, pipeline dormant |
| `MSstatsBioNet::.filterIndraResponse` | one R package's filter | Vitek lab | a `belief_cutoff` argument |

Nobody has proposed publishing a **parallel `llm_belief` field** beside the incumbent
instead of overwriting it. Augmenting preserves every downstream threshold (B6), needs no
readonly regeneration, can ship on 10⁵ statements instead of 10⁷, is falsifiable in weeks,
and is the only option §6.3 permits at partial coverage. It should be the default, and
replacement should have to earn its way past it.

Worth noting separately: the MSstatsBioNet row is a same-week PR that captures most of
that domain's named value and **requires no LLM at all** — the package already receives
`belief` and discards it. Teaching it to respect the incumbent number is a strictly
smaller change than teaching it to consume ours, and it is a useful forcing function for
whether belief is wanted there at all.

**To Ben:**

1. Which seam? And augment or replace?
2. What is the actual queryable denominator — is CoGEx's 44.9M right, or does
   `db.indra.bio` expose more?
3. How often does the readonly build run, and how many new evidence rows per rebuild?
   That number, not the backfill, is the recurring bill.
4. Is EMMAA meant to restart, or is it a frozen corpus?
5. Is there a query log? Head-of-distribution coverage would change the whole
   coverage calculus, and it is the one input that could justify a partial pass.
6. Whose AWS account, what ceiling, and what is this account's Bedrock TPM? (B3, §5.3)
7. What licensing covers sending INDRA evidence text to a third-party processor? (B5)

---

## 9. Proposed sequence

Each phase gated; nothing after a gate is authorised by anything before it.

**P0 — decide and unblock (no spend).** Answers to §8. Assert profile resolution at
startup (B1). Design the `belief_provenance` sidecar (B4). Enumerate downstream
thresholds (B6). Read the account's Bedrock quota (§5.3).

**P1 — the go/no-go measurement (~$1).** Label 500–1,000 rows of
`data/corpora/cogex_evidence_sample.jsonl` and report precision/recall **at the operating
point** against the incumbent on the same rows (B7). This is the only evidence that would
justify anything below it.

**P2 — one model, end to end ($16–44).** Bulk-score EMMAA `ms` — 56,194 evidences, 100%
text, 100% reader-sourced, the cleanest target in the set — and publish it in the §4
export format with provenance. `rasmachine` is already done and is the control.

**P3 — the sidecar (no new spend).** Tier A lookup over P2's output. Two consumers to
demo against: the EMMAA dashboard and a `belief_cutoff` argument in MSstatsBioNet.

**P4 — all of EMMAA ($600 verdict-only / $1,676 reasoning).** Contingent on B2 if
verdict-only, and on the fleet-wide cap (B3).

**P5 — full-corpus coverage.** Contingent on the runner rewrite (§5.4), TPM headroom
(§5.3), licensing (B5), and a budget owner. Not schedulable until those four are closed.

---

## What was not checked

Stated because absence of a finding is not a pass.

* **No external repo is in this tree.** `emmaa`, `indra_db`, `indra_cogex` and
  `MSstatsBioNet` claims rest on cloned/fetched sources and live probes, not on anything
  this repo's test suite can re-verify. `.venv` holds only `indra` 1.24.0, `gilda` 1.6.1
  and `indra_belief`.
* **Container sizes and RSS figures were not re-measured** — the Docker daemon was not
  running on this host. They rest on `Dockerfile.live`'s header and
  `research/serving_deployment.md` §7.
* **The EMMAA census counts the *published assembled* corpus only.** §2.2's own finding —
  that belief is scored against up to 1,000 extra INDRA-DB evidences that are then
  discarded — means the evidence EMMAA *scored* is strictly more than the evidence EMMAA
  *publishes*. Our bulk cost is sized on what we can actually read, which is the right
  basis for a spend estimate and the wrong basis for claiming we reproduced their number.
* **Our own docstring may be wrong.** `src/indra_belief/belief_scorer.py`'s module
  docstring says `readonly.belief` is read by "`sort_by=belief`, `minimum_belief` and
  seven denormalised meta columns". A probe found `minimum_belief` is CoGEx-only, not
  indra_db. Flagged for correction; not relied on above.
* **No self-hosted serving stack was measured.** Every GPU figure anywhere in this
  repository is an extrapolation from one June artifact on different hardware.
* **The MSstatsBioNet query probe used a dense, non-representative gene neighbourhood.**
  It bounds one query; it does not describe a typical one.
