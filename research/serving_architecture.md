# Serving architecture: current state, measured limits, and the refactor path

Written 2026-07-31. Covers the two execution paths that exist today, what breaks
at corpus scale, and the ordered set of changes that gets to low latency and cost
with batching alongside realtime.

**How to read the evidence.** Every number in §2-§4 was measured or reproduced on
this machine and is marked **[M]**. Findings surfaced by adversarial review that I
independently verified are marked **[V]**. Claims I am relaying without
verification are marked **[R]** and should be checked before anyone acts on them.
§6 lists claims from that review that are **wrong**, including one of my own.

---

## 1. What exists today

Two execution paths and no server. When this was written they shared **zero
scoring code**; they now share the request value and the reply reader, and
nothing else. The diagram is the current tree.

```
BATCH (paid, hardened)                    LIVE (library)
──────────────────────                    ──────────────
indra-belief-comparison run               score_evidence(stmt, ev, client)
  contracts.load_run_plan                   ScoringRecord(statement, evidence)
  ReplayIndex.load  ── validates ALL          __post_init__ -> resolve_entities()
     33,413 executions, re-hydrates              -> gilda + bio_ontology  (7.1 GB)
     and digest-checks EVERY prompt            record.tier1_auto_reject()
  load_resume(action.output) x N_actions      _select_examples()   at call time
  SpendGuard(ledger) ── replays ALL events     scorer._prepare
  ── ready-before-token boundary ──              prepare_from_record
  ThreadPoolExecutor(workers <= 8)                 record.execution_body()
    replay.score_execution
      ReplayIndex.prepare
        prepare_from_replay_row
      assert_replay_digests
            └───────────────┬────────────────────────┘
                    PreparedExecution      <- ONE request value
                      .calls(note) -> PreparedCall.client_kwargs()
                              client.call()
                      verdict.parse_response() -> Verdict | None
```

`test_package_lazy_import.py` asserts the batch transport imports no
`indra_belief.scorers.*`. That test documents the separation as intentional, and
it is the reason the two shared modules — `prepared_execution` and `verdict` —
sit at the top level of the package rather than inside either consumer: both
`scorers.monolithic` and `comparison.replay` import them, and neither may own
either. What the two paths can still drift on is everything the join does not
cover; F1 below now names exactly what that is.

**Deployment surface when this was written:** no Dockerfile, no compose file, no
API server (no fastapi/flask/uvicorn anywhere), one console script
(`indra-belief-comparison`). **Since then a `Dockerfile` (ae9ab1e) and a
`docker-compose.yml` (401ded6) landed, both batch-only — see
`research/serving_deployment.md`. There is still no API server**, and the realtime
path is still a library import, not a service.

---

## 2. Measured properties

### Memory **[M]**

| stage | peak RSS | time |
|---|---|---|
| baseline python | 16 MB | — |
| `import indra_belief.scorers.monolithic.scorer` | **25 MB** | 0.1 s |
| `import gilda` | 200 MB | 5.5 s |
| gilda index (lazy, on first `ground()`) | 1,875 MB | 18.9 s |
| `bio_ontology.initialize()` | **7,133 MB** | 14.6 s |

**Importing the scorer imports neither gilda nor indra.** The 7.1 GB materializes
only when something actually grounds. The lazy-import discipline already works.

### Preflight **[M]**

| stage | peak RSS | time |
|---|---|---|
| `ReplayIndex.load` (33,413 executions) | 440 MB | 6.3 s |
| `+ load_resume` arm 1 (0.95 GB file) | 5,509 MB | 20.3 s |
| `+ arm 2` | 8,099 MB | 28.0 s |
| `+ arm 3` | 8,099 MB | 25.9 s |
| `+ arm 4` (0.71 GB) | **8,303 MB** | 13.4 s |

3.6 GB of files → 8.3 GB resident (**2.6× amplification**), whole-file reads,
single-threaded canonical-JSON revalidation. `prepare_run` does this for **every
action in the plan**, not just the one being run — `prepare_run`'s loop over
`loaded.actions` and `_scopes`, both in `src/indra_belief/comparison/runner.py`. Machine has
137 GB RAM but swap is already 12.9/14 GB used.

### Prompt shape **[M]**

| component | chars | share | distinct values |
|---|---|---|---|
| system prompt | 3,799 | 49.3% | **2** |
| few-shot prefix | 3,460 | 44.9% | **9** |
| user message | 451 | 5.9% | per-execution |

**94.1% of every prompt is a shared prefix drawn from 11 strings.** Measured
2,135 input tokens vs 14 output → **152:1 prefill:decode**. `grep -rn
"cache_control\|prompt_cache\|cached_tokens" src/` returns nothing.

### Grounding working set **[M]**

66,826 lookups → 6,479 distinct entities = **90.3% cache hit**, Zipfian (top 10%
of entities = 75.5% of lookups). `gilda.ground()` is 111 µs uncached, 0.1 µs
through `_cached_ground`'s `lru_cache` (730×). **Total real gilda compute for the
whole 33,413-execution corpus: 0.7 seconds.** Pre-resolved substrate is 6.47 MB
against a 1,875 MB index (289×).

### Cost and latency, gemma-4-26b, same 33,361-execution corpus **[M]**

| | reasoning | verdict-only | Δ |
|---|---|---|---|
| provider-measured | $26.37 | $9.45 | −64% |
| cap-accounted | $35.39 | $9.72 | — |
| calls | 50,484 | 32,329 | −36% |
| output tokens | 23.3 M | 0.46 M | −98% |
| latency median | 6.56 s | 0.72 s | 9.1× |
| serial compute | 82.8 h | 9.5 h | 8.7× |
| per statement | $0.0156–0.0210 | $0.0056–0.0058 | — |

### Provider **[M]**

- No batch API on the mantle endpoint (`/v1/batches` and `/openai/v1/batches` 404).
- Token logprobs: **GLM-5 chat route returns them** (14 tokens + top-k);
  **gemma responses route accepts `top_logprobs` and returns an empty array**.
- Zero 429s at 32 concurrent (4 arms × 8 workers).
- Verdict posterior is **not reproducible**: same prompt at temperature 0,
  repeated calls span 0.755–0.905 (sd 0.045).

---

## 3. Verified findings

### F1 — The live and batch paths **had** already diverged **[V] CRITICAL**

Written as four divergences. Two are closed, two are open, and the finding is
kept rather than deleted because the two that remain are the ones a reader would
otherwise assume the refactor swept up with the rest.

| | live | batch | status |
|---|---|---|---|
| user message | `ScoringRecord.execution_body` | `prepare_from_replay_row` | **CLOSED** — both build the same `ExecutionBody`, and the join is `ExecutionBody.render`, defined once |
| score on unreadable reply | `verdict.parse_response` | `verdict.parse_response` | **CLOSED** — one parser, and an unreadable reply is `None` on both sides |
| provenance gate | recomputed from `has_grounding_signal` | trusted field `row["provenance"]` | open |
| few-shots | `_select_examples` at call time | frozen prefix by sha256 | open |

The two open rows are genuine and are not defects of the same kind: the batch
side trusts a field the live side recomputes because the batch row is a *frozen*
record of a decision already made. What is missing is a test that the two agree
when both are available, not a third implementation.

**The `verdict_only` half of this finding still stands.** `scorers.monolithic.scorer`
resolves `MONO_VARIANT` through `variant_from_env` into a `ScoringVariant` from its
`VARIANTS` registry, and that registry has no `verdict_only` key —
`variant_from_env`'s own docstring records the consequence: a reader reproducing
`data/comparison_verdict_only/grounding_replay` from the `mono_variant` label in
its manifest gets the baseline prompt with no signal. Those prompts come from
`scripts/build_verdict_only_replay.py`, which is unpackaged and is the only
importer of `_prompts_verdict_only`. So `score_evidence()` still cannot emit the
prompt that produced the shipped verdict-only run.

What has changed is that the two renderers are now tied together by test rather
than by maintenance. The original finding rested on a grep for the live renderer
across `tests/` returning nothing at all; today
`tests/test_prepared_execution_parity.py` asserts the live and batch producers
build the same `PreparedExecution` element-for-element on the plain route, the
tool route and the relation-note path, and re-derives the 15 component digests
the frozen substrate commits to.

### F2 — Ledger replay was quadratic **[V] CRITICAL**

What is verified here is a **shape**, and the code structure is what proves it.
`_apply` — the single event-transition function `_LedgerReplay.apply` drives — ran
two full scans of `self._reservations`, a dict that never evicts: one per
`call_reserved` event, inside the `call_ordinal` check, and one per
`attempt_outcome_committed` event. Per-event cost therefore grows with the number
of events already replayed, so replaying a ledger of n events is O(n²).

*Four absolute per-event timings stood here and are withdrawn, not corrected.*
They were marked [V] but named no script, no ledger identity, no ledger size and
no machine; they appear nowhere else in the repository; and an adversarial review
of the fix could not reproduce them on either of two independent runs. Nothing
above rests on them — the shape is structural, and the node that fixed it measured
the same shape independently. An absolute belongs here only with its run
conditions beside it.

Both scans are now a single `dict[attempt_id] -> list` lookup, `_calls_for_attempt`
on `SpendGuard` in `src/indra_belief/spend_guard.py`, aliased into `_LedgerReplay`
alongside the other borrowed methods. (`_calls_for_attempt`'s own docstring says it
"Replaces three full scans"; that is the module-wide count. Two of the three were
in `_apply` — the quadratic pair this finding is about — and the third is in
`_attempt_rows`, outside the replay loop. Both counts are right at their own
scope.) Cited by symbol rather than by line exactly
because the fix moved the lines. This was the highest leverage-to-effort change in
the repository.

### F3 — 32 workers is contractually impossible **[V] HIGH**

`contracts.MAX_WORKERS` is 8, and `load_run_plan` rejects more ("workers cannot
exceed eight"). And `AMENDABLE_FIELDS["workers"]` is `(6, MAX_WORKERS)` — the
**only legal amendment to `workers` is exactly 6→8** (`max_attempts` has its own
5→10 transition). Raising it edits the module, changes `plan.sha256`, and invalidates
every amendment's `predecessor_sha256`. There is no measurement behind the 8, and
we have zero 429s at 32 concurrent.

### F4 — Durable state is ~105 KB per 15 bytes of measurement **[R] CRITICAL**

Relayed, not independently reproduced. An attempts row is reported at 29,330 B, of
which the same prompt appears three times (`provider_request_body` 10,281 B,
`messages` 5,378 B, `system` 3,811 B) against 15 B of verdict+confidence; the
ledger stores it again. 13.6 GB for one 33k-execution 4-arm run.

The fix is sound regardless of the exact bytes: the substrate is content-addressed,
so `main_system_ref` / `main_message_prefix_ref` — **already present in the
execution row** — are the proof. Storing the bodies adds no provenance.

### F5 — Sharding reintroduces double-spend **[R] HIGH**

`load_run_plan` in `src/indra_belief/comparison/contracts.py` checks unique action ids, run_ids, output paths and
`(stage_id, workload)` lanes. There is **no check that two actions'
`execution_keys` are disjoint**; `ReplayIndex.select` rejects repeats only *within*
one action. Shard 90M rows across N actions and nothing prevents paying twice for
the same `(stmt_i, evidence_i)`. ~~Fix is one hash-set check at plan load.~~ **The
fix is withdrawn and the premise is a precondition, not a defect — see §10.** No
partition concept exists in `src/`, and the lane check plus `(model, workload)`
execution identity already make cross-action collision impossible for every plan
shape the loader admits.

### F6 — One bad row halted the whole action **[R] HIGH — CLOSED by a10df62**

Any worker exception or `outcome.failure` used to set `stop_scheduling = True`. At
the measured 0.057–0.097% off-grid rate, 90M rows gives ~63,000 wedge candidates.
Head-of-line blocking is not a viable failure mode at that scale. **Closed:**
`_failure_disposition` in `src/indra_belief/comparison/runner.py` routes a
quarantinable failure to `_quarantine_failure` and keeps scheduling; only a
non-quarantine failure or an escaping exception still halts. How far the run
continues past the first hole is bounded by `_diagnostic_budget_spent` /
`QUARANTINE_DIAGNOSTIC_LIMIT`, and the count surfaces on `RunSummary.quarantined`.

### F7 — Semantic dependencies fail open **[R] HIGH**

Gilda and ontology exceptions become empty results rather than errors
(`_cached_ground` and `_get_fplx_members` in `src/indra_belief/data/entity.py`), which changes routing and prompt content instead of
surfacing a dependency failure. A missing few-shot bank logs a warning and scores
with no examples (the example-bank load in `src/indra_belief/scorers/monolithic/scorer.py`). For a scientific scorer these should fail
closed or be a separately versioned mode.

### F8 — Statement belief could report `correct` with belief 0.0 **[R] HIGH — CLOSED by a10df62**

A low-confidence `incorrect` row was excluded from the belief numerator but did not
count as a credible rejection, so `verdict_statement` could be `"correct"` while the
hard gate returned 0.0. I flagged the same gap independently earlier this session
from the retry-policy angle. **Closed:** the confidence gate on the route is gone.
`src/indra_belief/statement_belief.py` now states the invariant in its own module
docstring — `verdict_statement == "correct"` implies `n_incorrect == 0` — and the
constant this finding named no longer exists in the tree.

---

## 4. The refactor plan

Ordered by impact ÷ things-added, honouring the standing preference for removing
over adding. Each names the SOLID principle only where it clarifies the change.

### Remove

1. **The second prompt renderer. LANDED.** `indra_belief.prepared_execution` is
   the one module: two producers (`prepare_from_record` live,
   `prepare_from_replay_row` batch) build one `PreparedExecution`, and
   `PreparedExecution.calls` is the only place the relation note and the entity
   lookup block are spliced in. The parity test asked for here exists as
   `tests/test_prepared_execution_parity.py`.
   The circularity this item warned about is **not** removed by that, and the
   distinction is worth keeping: `scripts/build_verdict_only_replay.py` computes
   `main_prompt_base_sha256` through the same `prepare_from_replay_row` that
   `assert_replay_digests` later checks it against, so the digest still cannot
   catch a change in the assembly. What catches it is external — the 15 prompt
   components the shipped `data/comparison/grounding_replay/manifest.json`
   already commits to, re-derived by the parity test through the LIVE producer.
   *SRP: `ReplayIndex` still loads, validates, hydrates and derives deterministic
   results; what it no longer does is assemble a request.*
2. **Prompt bodies from ledger and attempts rows.** Store the refs already in the
   row. ~100× on durable state; provenance unchanged.
3. **Import-time `MONO_VARIANT`. PARTLY LANDED.** A profile object can now be
   passed: `scorers.monolithic.scorer` carries a `ScoringVariant` registry and
   every scoring entry point takes `variant=`. What is still import-time is the
   FALLBACK — `DEFAULT_VARIANT` is `variant_from_env()` evaluated once at import,
   so a caller that passes nothing still inherits process-wide scientific
   semantics from the environment. *DIP — and the registry having no
   `verdict_only` key is still why the live path cannot reach it.*

### Fix (small, surgical)

4. ~~**Two dict indexes in `_apply`** → O(n²) becomes O(n).~~ — **LANDED [F2].**
   One index, `_calls_for_attempt` on `SpendGuard`; see F2.
5. ~~**Index `resume.rows` by source key**~~ — **LANDED.** `_scan_resume` in
   `src/indra_belief/comparison/replay.py` builds `latest` / `attempts` /
   `invalid_outputs` / `settled` keyed by source key, and `_run_source` reads them
   by key rather than re-scanning.
6. **Preflight only the selected action** — still open: `prepare_run` calls
   `resume_status` for every action in `loaded.actions`, and `_scopes` scopes every
   action.
7. ~~**Quarantine wedged sources** instead of `stop_scheduling`~~ — **LANDED
   [F6]**; see F6 for the shipped shape and its diagnostic budget.
8. ~~**Cross-action `execution_keys` disjointness check** at plan load~~ —
   **WITHDRAWN.** The global form rejects two shipped plans and the narrow form is
   vacuous; see §10, which specifies what sharding would require instead. **[F5]**
9. ~~**`dict(LOCAL_MODELS[model_name])`**~~ — **LANDED** in `ModelClient.__init__`
   (`src/indra_belief/model_client.py`). Each client now owns its copy, so the
   runner's per-call `config["timeout"]` ratchet no longer mutates the registry.
10. **`MAX_WORKERS` to a measured value** — we have zero 429s at 32 **[F3]**.

### Add (only these)

11. **Prefix caching.** 94.1% of input from 11 strings at 152:1. Sort shards by
    `statement_type` so identical prefixes arrive contiguously — vLLM reuses only
    exact token-prefix blocks, so interleaving 9 prefixes across many concurrent
    clients degrades hit rate.
12. **A `no_text` / off-grid contract test** across both paths, once they share a
    renderer.

---

## 5. Batch and realtime from one codebase

The requirement is not "one process" or "one image" — it is **one semantic
kernel**. Scheduling may differ; prompt, routing, parsing, score mapping and
aggregation may not.

```
        ScoringInput ──► resolve (heavy, gilda+ontology, 7.1 GB)
                              │
                          ResolvedPair          <- the shard record
                              │
                       render()  ── ONE renderer, versioned
                              │
                     PreparedExecution
                        │            │
              batch: persist    realtime: execute now
                        └──────┬─────┘
                          transport (Bedrock | vLLM)
                               │
                          parse()  ── ONE parser
                               │
                     EvidenceObservation
                               │
                    statement_belief()  ── ONE aggregator
```

**The generator/worker boundary already exists as data** — the substrate is that
boundary. What makes a worker 25 MB is that it never grounds, not that it lives in
a different image.

**Realtime cannot skip the heavy stage for novel input.** `ScoringRecord` is typed
on INDRA `Statement`/`Evidence` and `__post_init__` calls `resolve_entities()`
unconditionally (`ScoringRecord.__post_init__` in `src/indra_belief/data/scoring_record.py`). So a live API over arbitrary new
statements needs the 7.1 GB resident, or an upstream service that hands it
pre-resolved `ResolvedPair` records. That is a genuine design constraint, not a
packaging detail — and it is the strongest argument for treating realtime as a
consumer of the same resolver rather than a second implementation of it.

---

## 6. Corrections — including my own

- **Mine:** I earlier called the two-image Docker split "necessary, not optional"
  on memory grounds. **Wrong.** Importing the scorer costs 25 MB and pulls in
  neither gilda nor indra; the 7.1 GB is lazy. The 285× RSS ratio is delivered by
  *not grounding*, which the lazy imports already achieve in one image. Two images
  buy **image size and dependency surface** (`.venv` is 1.5 GB), not RSS. That is a
  real but much smaller benefit, and it adds a class of silent generator/worker
  version skew. One image was the parsimonious default. **Superseded by what
  shipped:** the `Dockerfile` at ae9ab1e is batch-only and omits gilda and indra,
  taking the split on dependency-surface grounds exactly as this bullet argued —
  and it installs deps explicitly rather than via `pip install .`, which is the
  next bullet's prescription. A live/grounding image is named there as a separate
  build.
- **Relatedly:** `pip install .` installs the heavy closure regardless of lazy
  imports, so anyone pursuing a lean image must split the *dependency* declaration,
  not just the import sites.
- **A critic claimed the spend ledger is "~4,000 lines."** It was **2,163**
  (`spend_guard.py`) — **2,225** at 96cc1b7. 3,928, now **4,316**, is the total for
  `spend_guard` + `replay` + `runner`.
- **A critic computed "32 workers × 8.3 GB = 265.6 GB RAM."** That conflates the
  paid runner's preflight with the proposed lean workers, which do not preflight at
  all. The 8.3 GB is a property of `prepare_run`, not of a shard consumer.
- **The reproducibility framing needs care.** vLLM's own documentation states
  online serving does not guarantee reproducibility. Combined with the measured
  0.755–0.905 posterior spread at temperature 0, "reproducible" must mean *retain
  exact observations and deterministically rebuild published numbers from them* —
  **not** *re-run the model later and expect a match*. The digest apparatus proves
  the input to six decimals; the output is not stable to one.

---

## 7. On the GPU

- **One vLLM instance per GPU**, not multiple replicas of one model on one GPU:
  replicas duplicate weights and split the KV pool, both of which reduce
  throughput. Multiple GPUs → one replica each (data parallel) is correct.
- `--gpu-memory-utilization 0.90-0.95`, `--enable-prefix-caching`,
  `--enable-chunked-prefill`. With prefix caching, KV per sequence falls from
  ~0.4 GB to ~27 MB, so KV stops being the constraint and **prefill FLOPs become
  it**. fp8 weights are then worth testing for throughput, not for memory.
- Self-hosting is what unblocks **token logprobs on gemma**, which Bedrock's
  responses route does not return. That is a scientific argument (a posterior in
  one call rather than a point sample from an unreproducible one), and it is
  stronger than the throughput argument.
- **Cost accounting does not transfer, and it fails silently.**
  `_ActionGuard.reserve_call` in `src/indra_belief/comparison/runner.py` calls
  `price_for` from `src/indra_belief/corpus/cost.py` and raises `SpendGuardError`
  on `None` — but a self-hosted id already registered in `ESTIMATED_PRICE_REFS`
  **is** priced, at basis `estimate` against a Bedrock twin, so it never reaches
  that raise: `price_for("mlx-community/gemma-4-26b-a4b-it-8bit")` returns
  `(0.13, 0.4, "estimate")`. Such a lane is therefore admitted and billed
  estimated dollars for GPU calls that cost none. A *new*, unregistered id is the
  case that returns `None` and raises. Both need the same explicit decision:
  exempt the lane from spend accounting, or price GPU-seconds. See 9.6.

---

## 8. Open questions

1. **The original substrate generator is not in this repository.** Only the
   verdict-only *derivation* script is. The generator that produced
   `data/comparison/grounding_replay` — which every shipped paper number rests on —
   cannot be re-run. Replay is reproducible; regeneration is not.
2. **`scripts/` is ~37,600 LOC**, comparable to `src/` (36,752), unpackaged, outside
   `testpaths`, and reaches into private symbols (`scorer._select_examples` and
   `scorer._LOOKUP_GUIDANCE` from `scripts/build_verdict_only_replay.py`,
   `metrics._rankdata_avg` from two paper-table scripts). Some of it is contract,
   not scripting. The one private reach that used to matter most —
   `scripts/build_verdict_only_replay.py` re-implementing the batch renderer's
   tail — is gone: it calls the public `prepare_from_replay_row` now.
3. **Calibration identity** fingerprints only the `system` string of `monolithic`
   calls (`_call_log_fingerprints` in `src/indra_belief/calibration_constants.py`) — excluding few-shot content, the example
   bank, and all `monolithic_tool_context` calls (8.1% of executions). A run can
   claim a profile while behaviour-changing prompt material differs.
4. **Whether realtime is needed at all.** There is no consumer, latency target,
   overload policy or auth model on record. Building an API because Docker is being
   adopted is not a requirement.

---

## References

- Adversarial review: 3-critic brutalist panel + a 9-agent workflow, 2026-07-31.
  Raw output under the session tool-results directory; findings above are labelled
  by verification status rather than by source.
- [[project_noreason_rerun]] — the verdict-only run these measurements come from.
- `research/scoring_methods.md` — the belief math these paths compute.

---

## 9. Prefix-cache benchmark specification

This section specifies a benchmark. It reports none. Every results cell below
ships carrying the literal token `UNMEASURED`, and no throughput number is
predicted anywhere in it. Its purpose is to turn §4 item 11 and §7 from
unmeasured directives into something that can come back wrong.

**Placement.** Appended at end of file rather than inserted at the `## References`
anchor, so the change is a tail append with exactly one argued exception: the cost
bullet in §7, corrected below in 9.6 and at its own site because it rested on a
premise this section refutes. §1–§6, §8 and the References block were
byte-identical afterwards. A sibling section was slated for insertion at the
References anchor in the same wave; a tail append cannot collide with it. §3's F2
has since been corrected in place by a later node — four unreproducible absolutes
withdrawn, its two line anchors re-cited by symbol — which is that node's edit, not
part of this section's append.

**One renaming applies to every renderer citation below, and the figures were
re-derived across it rather than carried over.** Every [M] figure in 9.1 was first
measured through the shipped batch renderer while that renderer was a single
method on `ReplayIndex`. That method has since been split into
`ReplayIndex.prepare` (ref resolution) plus `prepare_from_replay_row` and
`PreparedExecution.calls` (assembly), with the two digest checks moved to
`assert_replay_digests`; the citations below name those current symbols. **Every
9.1 figure was then re-measured through them, read-only, and is unchanged** — 18
distinct pairs, 32,357 calling and 1,056 deterministic rows, all 1,056 raising
`ReplayError("main prompt references an absent component")`, and system / prefix /
user means of 3,763.2 / 3,501.1 / 435.1 chars at 94.35% shared. The same pass
recomputed `main_prompt_base_sha256` for all **32,357 calling rows with 0
mismatches**, so the rename moved no byte of the assembly.

**Numbering.** The integer was resolved by reading the merged document at land
time: the highest section present was 8, so this is **section 9**, contiguous with
what precedes it. Nothing binds the integer — `scripts/check_doc_anchors.py`
reads this document end to end and locates nothing in it by section number — so a
later renumbering is safe and requires no other edit; this paragraph and §10's are
the two that must be re-read when it happens.

**Markers** follow the document header, with one section-local reading: **[R]**
here is to be confirmed on the serving host before anyone acts on it. Every [M]
figure below was re-derived read-only, by one of three methods — and which one
matters, because only the first instantiates the shipped renderer:

* **Through `ReplayIndex.prepare`** over
  `data/comparison_verdict_only/grounding_replay`: the per-component char split in
  9.1 — 3,763.2 / 3,501.1 / 435.1 and 94.35% shared. This is the only figure that
  needs a renderer, and it is never a second implementation of one.
* **Plain reads of that same substrate's manifest and executions file**, no
  renderer involved: the 18 and 25 ref pairs, the 32,357 / 1,056 row counts, the
  137,063-char working set with its 6,927–8,350 per-pair range, the 12 statement
  types mapping to 9 refs in 9.4, and the 2,075-run natural-order statistics.
* **Reads of a run's `attempts.jsonl`**, which records what the provider billed:
  the 2,193.3 / 14.2 token means, 154.5:1 and the 0.756 s median in 9.1, and the
  reasoning-arm contrast in 9.8.

The one [M] outside all three is the `price_for` reproduction in 9.6, a read-only
call into `src/indra_belief/corpus/cost.py`.

### 9.1 The cacheable unit

§2 says the shared prefix is "drawn from 11 strings". That count is right and the
inference usually drawn from it is wrong. **11 counts components; the cache key is
the concatenation.** A serving engine reuses a *token prefix*, so the unit that
either hits or misses is `system + few-shot prefix` as one span. Measured over the
shipped substrate, those components co-occur as exactly **18 distinct
(`main_system_ref`, `main_message_prefix_ref`) pairs** [M]. The cross product
2 × 9 also equals 18, and that coincidence is worth naming: 18 is the *measured*
co-occurrence count, not the product. Nothing guarantees every system pairs with
every prefix, and over the whole file — below — it does not.

**That figure carries a predicate, and the predicate is a protocol requirement.**
18 holds over the rows that actually call a model — those with a non-empty
`call_topology`: 29,653 `monolithic` + 2,704 `monolithic_tool_context` = **32,357
of 33,413 rows** [M]. Over the whole file the same count is **25 pairs from 3
system refs and 16 prefix refs** [M], because the remaining **1,056 deterministic
rows** (3.2%) retain refs inherited from the parent substrate that are absent from
this manifest's `prompt_components` — which declares 2 systems and 9 message
prefixes and nothing else. For all 1,056 of those rows, `ReplayIndex.prepare`
raises `ReplayError("main prompt references an absent component")` — the raise
itself is in `prepare_from_replay_row`, which resolves the refs against the
index's component tables — and **0 of 1,056** resolve in either table [M]. A bench
runner that iterates the executions file naively therefore crashes on 3.2% of
rows. The row filter is part of the protocol, not a footnote.

**Working set.** Summing the 18 distinct concatenations: **137,063 chars** [M],
per pair 6,927–8,350 chars. At this corpus's measured 3.51 chars/token
(137,063 / 3.51 = 39,049) that is **~39k tokens** for the entire prefix working
set. The char count is [M]; the **~39k** figure is an extrapolation from a ratio
and **not** a tokenizer count — hypothesis H-C exists precisely to replace it.

**Shape — what the bench client must do with it.** `ReplayIndex.prepare` returns a
`PreparedExecution`; the request is its last `PreparedCall`
(`execution.calls()[-1]`), which carries `system` and `messages` as separate
fields, and `PreparedCall.client_kwargs` is the ready-made argument dict.
`messages` is the few-shot prefix messages followed by exactly one user message;
**the system string is not an element of that list** — checked over every calling
row, not one: all **32,357 carry exactly 29 messages** (alternating user/assistant
example turns then a final user turn) and **0 carry a message whose content equals
the system string** [M]. The bench client must place `system` as the request's
system message. A client that drops it, or appends it after the few-shot turns,
destroys the very property being measured: the shared span would no longer start
at token 0 and no engine could reuse it.

**Block granularity** [R]. Reuse is whole-KV-block, not per-token (vLLM
`--block-size`, commonly 16), so the final partial block of each prefix is not
reused. At ~3.5 chars/token a ~7,700-char prompt is ~2,200 tokens, so the
unreusable tail is at most block-size minus one token per request — negligible,
recorded here only so that a hit rate short of 100% is not misread as a defect.

**Reconciliation with §2, which is left byte-unchanged.** §2's "Prompt shape"
table reads 3,799 / 3,460 / 451 chars and 94.1%. Re-derived here through
`ReplayIndex.prepare` over the 32,357 calling rows of
`data/comparison_verdict_only/grounding_replay`: system **3,763.2** (48.88%),
few-shot prefix **3,501.1** (45.47%), user **435.1** (5.65%), total **7,699.4**
chars, **94.35% shared** [M]. The two agree on the **shared-prefix share** to
within a quarter of a percentage point and support the same conclusion — but that
scoping is the whole of the agreement, and the component means are further apart
than it suggests: the user message moves 451 → 435.1, a difference of 15.9 chars,
3.5% of §2's figure. **Why the two differ at all is not established** [R]. Nothing
in the repository records how §2's table was computed, so "they state different
predicates over the row set" is a hypothesis about §2, not a finding about it.
This section states its own predicate explicitly, which is all that can be claimed
from here. §2 is not edited here — §6 "Corrections" is where a future edit
belongs, not this section.

The prefill:decode ratio is **not** re-derived through the renderer; no renderer is
involved in it at all. It is read from the provider-reported usage in the run's
attempts file: over the first 15,000 attempts (14,518 calls carrying usage) of
`data/comparison_verdict_only/runs/gemma_26b_vo_primary/attempts.jsonl`, mean
prompt tokens **2,193.3**, mean output tokens **14.2** → **154.5:1**, median call
duration **0.756 s** [M]. §2 records 2,135:14 and 0.72 s — the same regime,
measured over a different slice.

### 9.2 Pre-registered hypotheses

Registered before any run, so each can come back wrong.

**H-A — prefix caching OFF vs ON changes prefill throughput and TTFT.**
Expected direction: ON is faster. **Effect size UNMEASURED**; no figure is
predicted. Falsified if the ON and OFF medians overlap across all 3 replicates at
every concurrency in the sweep.

**H-B — prefix-sorted arrival order beats shuffled order.
PRE-REGISTERED EXPECTATION: NULL.** The argument, recorded here before any result:
all 18 distinct prefixes total **137,063 chars** [M] ≈ ~39k tokens (extrapolated
at 3.51 chars/token, not a tokenizer count — see H-C). A KV pool sized
for one model at the `--gpu-memory-utilization 0.92` the arms in 9.3 actually
launch with holds orders of magnitude more than that — **[R]**, and unverifiable
in this workspace for precisely the reason 9.3's default-flag claims are [R]:
there is no serving stack and no CUDA device here. If it holds, **all 18 prefixes
sit resident simultaneously** and eviction — the only mechanism by which arrival
order could change hit rate — should not occur. That [R] premise is the entire
basis of the pre-registered NULL, so it is the first thing to check when reading a
non-null result: a sorted-order win with an undersized pool is an eviction finding,
not an ordering one. H-B is confirmed only if sorted order beats shuffled order
beyond the IQR of 3 replicates.

**Consequence of a null H-B, stated in advance.** §4 item 11 makes two claims: turn
prefix caching on, and sort shards so identical prefixes arrive contiguously
because "interleaving 9 prefixes across many concurrent clients degrades hit rate".
A null H-B refutes the **ordering** half. It should then be struck, keeping only
"turn prefix caching on". A null H-B says nothing about H-A; the two are
independent and must not be reported as one result.

**H-C — the marginal cost of one more evidence row approaches its unique suffix.**
If ~94% of prefill is a cached prefix, the incremental prefill of an additional row
should approach the tokens of its ~435-char unique user message. **Quantify with
the served model's own tokenizer.** The 3.51 chars/token ratio in 9.1 is an
artifact of this corpus and MUST NOT be reused as a token count. Falsified if
measured marginal prefill tokens per additional row exceed the tokenizer's count of
the unique suffix by more than one KV block.

### 9.3 Server configurations

One flag differs between treatment and control. Everything else is held fixed;
varying two flags at once confounds H-A with a scheduling change.

```
# vLLM — treatment: prefix caching ON
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --port 8000 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --block-size 16 \
  --enable-chunked-prefill \
  --enable-prefix-caching

# vLLM — CONTROL for H-A: identical but for the last flag
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --port 8000 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --block-size 16 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching
```

```
# SGLang — treatment: RadixAttention prefix reuse as configured by default
python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --port 30000 \
  --enable-metrics

# SGLang — CONTROL for H-A
python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --port 30000 \
  --enable-metrics \
  --disable-radix-cache
```

`--no-enable-prefix-caching` (vLLM) and `--disable-radix-cache` (SGLang) are **the
controls**. `--enable-chunked-prefill` / `--no-enable-chunked-prefill` is a
separate question and is held constant across every arm here.

**Correction to the premise this section was commissioned under, and it matters
for which flag is the control.** The claim that `enable_prefix_caching` and
`enable_chunked_prefill` both default TRUE in current vLLM **cannot be verified in
this workspace** — there is no vllm and no sglang in `.venv` and the machine is
Apple silicon with no CUDA device — so it is marked **[R]**, not [M]. It is paired
with a protocol step rather than trusted: **before the first arm, the runner
confirms the effective defaults from both the server's own startup log and
`--help` on the pinned version, and records both verbatim in the environment
table of 9.7.** If prefix caching turns out to be off by default on the pinned
version, the treatment and control flags swap and the arms are relabelled; the
experiment is unaffected. §7's "add `--enable-prefix-caching`" is a no-op if the
default is already on, which is exactly why the OFF flag is the interesting one.

**fp8 weights are a throughput question, not a memory fix**, and are out of scope
until the prefill-bound regime is confirmed. §7 already notes that under prefix
caching KV stops being the constraint and prefill FLOPs become it; fp8 is then a
prefill-FLOPs experiment to run *after* H-A, on a fixed arm, never mixed into it.

### 9.4 Ordering arms

**Sort key.** The pair (`main_system_ref`, `main_message_prefix_ref`), read
straight off each executions row. No new partition concept is introduced: the
manifest's `generation_contract.message_prefix_refs_by_statement_type` already
declares statement type → prefix ref (verified present: **12 statement types
mapping to 9 distinct refs** [M]), and the system half is declared alongside it as
`plain_main_system_ref` and `tool_main_system_ref`. The sort key already exists as
data.

| arm | order | runs of a constant key |
|---|---|---|
| **A-sorted** | rows sorted by the pair | 18 contiguous runs by construction |
| **A-natural** | file order exactly as shipped | **2,075 runs, mean 15.59, longest 1,134** [M] |
| **A-shuffled** | seeded shuffle, seed recorded | the interleaved control |

**A-natural is not the interleaved control.** The shipped corpus is already mostly
prefix-grouped — mean run 15.59 exceeds the 8-worker width of the paid runner, so
even at natural order a worker pool mostly sees one prefix at a time. Treating
file order as "interleaved" would test nothing. The interleaved arm needs an
**explicit seeded shuffle** (`random.Random(seed).shuffle`), and the seed goes in
the results table so the arm is reproducible.

**All three arms must present the identical row multiset** — the
`call_topology`-non-empty subset defined in 9.1 — differing only in order. Assert
it before the run: equal length, and equal multiset of `execution_key_sha256`. An
arm that quietly drops rows produces a throughput difference that is a sampling
artifact, not a caching effect.

### 9.5 Metrics

**Enumerate before trusting any series name.** Exact Prometheus names move between
versions, so the first step of every run is:

```
curl -s http://HOST:PORT/metrics | grep -E 'prefix|cache|token' | sort
```

and the observed names are pasted verbatim into the environment table of 9.7.
Everything below is a **candidate, version-dependent** [R]:

| quantity | candidate series |
|---|---|
| prefix cache | `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits`; older builds `vllm:gpu_prefix_cache_queries` / `_hits`; newer also expose `vllm:prompt_tokens_cached` |
| hit rate | `rate(hits[5m]) / rate(queries[5m])` from the two above |
| KV usage | `vllm:kv_cache_usage_perc`; older `vllm:gpu_cache_usage_perc` |
| TTFT | `vllm:time_to_first_token_seconds` (histogram) |
| queue time | `vllm:request_queue_time_seconds_sum` / `_count` |
| prefill / decode | `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` |
| throughput | `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:iteration_tokens_total` |
| pressure | `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_preemptions_total` |
| SGLang | `sglang:cache_hit_rate` (requires `--enable-metrics`) |

**Also check the per-response field**, which is independent of the server's
Prometheus surface: `usage.prompt_tokens_details.cached_tokens`. Cross-checking it
against the server-side hit rate is how a bench client detects that it is measuring
the wrong thing.

**Recorded, not fixed here:** `src/indra_belief/model_client.py`
(`_call_openai_compat`) reads only `usage.prompt_tokens` today and drops the cached
breakdown; `grep -rn "cache_control\|prompt_cache\|cached_tokens\|prefix_caching" src/`
returns nothing at all [V]. Capturing that field in the transport is a **separate
production node** and is deliberately not this one — the bench client reads it
directly off the response instead.

### 9.6 Protocol

Specific enough that two people get the same numbers.

* **Prompts come from the shipped renderer.** The one-renderer change in §4 has
  landed, so this is now specific: `ReplayIndex.prepare` in
  `src/indra_belief/comparison/replay.py` returns a `PreparedExecution`, and
  `PreparedExecution.calls` is the only source of benchmarked bytes. **Never a
  reimplemented renderer**, and equally never a re-splice — the note-then-lookups
  splice happens inside `calls()` and nowhere else. A renderer that differs by one
  character changes the token prefix, and the benchmark silently stops measuring
  the shipped prompt.
* **Row set** = the `call_topology`-non-empty filter from 9.1. Assert the count is
  32,357 before the first request; if it differs, the substrate changed and 9.1's
  figures must be re-derived before the run means anything.
* **Warm-up and volume.** Per arm: 200 requests discarded as warm-up, then at least
  2,000 measured requests.
* **Concurrency sweep** {1, 8, 32, 128}. `MAX_WORKERS = 8` in
  `src/indra_belief/comparison/contracts.py` binds the **paid runner only**; it does
  not bind a bench client, and treating it as a ceiling would hide the queueing
  regime this benchmark exists to find.
* **Replicates.** 3 per (arm × concurrency), with arm order randomized within each
  replicate. Report **median and IQR**, never a single run.
* **Cache reset between arms.** `POST /reset_prefix_cache` if the pinned version
  exposes it, otherwise restart the server — and **record which was used**. A
  forgotten reset lets arm 2 inherit arm 1's cache and manufactures a null H-B out
  of nothing.
* **Pin and record** the vLLM/SGLang version, model id, quantization, driver
  version, GPU model and count, block size, shuffle seed, and the confirmed default
  flag state from 9.3.
* **The bench client must not go through the paid runner**, and the reason is not
  the one usually given. `_ActionGuard.reserve_call` in
  `src/indra_belief/comparison/runner.py` calls `price_for` from
  `src/indra_belief/corpus/cost.py` and raises `SpendGuardError` from
  `src/indra_belief/spend_guard.py` when the price is `None` — but a self-hosted id
  already registered in `ESTIMATED_PRICE_REFS` **does** carry a price. Reproduced:
  `price_for("mlx-community/gemma-4-26b-a4b-it-8bit")`, `price_for("gemma-4-26b-ollama")`
  and `price_for("gemma-4-26b")` each return `(0.13, 0.4, "estimate")`, a
  Bedrock-twin estimate, not `None` [M]. So the paid runner would **not** refuse a
  bench client on a registered local id; it would admit it and silently write
  spend-ledger rows carrying estimated dollars for GPU calls that cost no dollars,
  corrupting the accounting behind the published cost figures. A *new*, unregistered
  vLLM model id hits the other failure mode instead — `price_for` returns `None` and
  `reserve_call` raises. Both roads end at the same rule: the bench client stays off
  the paid runner, and the guard is not to be adjusted to let it through.
* **Output location.** A scratch directory. **Never** under `data/comparison/**` or
  `data/comparison_verdict_only/**` — those bytes back published numbers — and never
  to a spend ledger. The substrate is opened **read-only**, for prompt bytes only.

### 9.7 Reporting template

**Host:** host unassigned.

Caching OFF is measured on **A-natural only**: with no cache, arrival order cannot
matter by construction, so the other six OFF cells would measure nothing. H-A reads
the A-natural ON/OFF contrast; H-B reads the three ON arms against each other.

| arm | clients | prefix cache | hit rate | cached prompt tokens | KV usage % | TTFT p50 | TTFT p99 | queue p50 | prefill tok/s | decode tok/s | e2e p50 | preemptions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-sorted | 1 client | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-sorted | 8 clients | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-sorted | 32 clients | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-sorted | 128 clients | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-natural | 1 client | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-natural | 8 clients | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-natural | 32 clients | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-natural | 128 clients | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-shuffled | 1 client | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-shuffled | 8 clients | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-shuffled | 32 clients | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-shuffled | 128 clients | ON | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-natural | 1 client | OFF | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-natural | 8 clients | OFF | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-natural | 32 clients | OFF | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| A-natural | 128 clients | OFF | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |

Each cell is the **median of 3 replicates**; report the IQR alongside it when the
table is filled.

**Environment record** — filled from the pinned host before the first arm:

| field | value |
|---|---|
| serving engine and version | UNMEASURED |
| effective prefix-cache default (from startup log) | UNMEASURED |
| effective prefix-cache default (from `--help`) | UNMEASURED |
| effective chunked-prefill default | UNMEASURED |
| model id | UNMEASURED |
| quantization | UNMEASURED |
| GPU model and count | UNMEASURED |
| driver version | UNMEASURED |
| block size | UNMEASURED |
| shuffle seed for A-shuffled | UNMEASURED |
| cache reset method used between arms | UNMEASURED |
| observed metric series names | UNMEASURED |

**Cost** — the only figure that transfers off this host:

| quantity | value |
|---|---|
| GPU-seconds per 1,000 evidence rows | UNMEASURED — becomes [M] once measured |
| USD per 1,000 evidence rows | UNMEASURED — requires an explicitly cited hourly rate, marked [R] |

### 9.8 What this section does not claim

* **No GPU throughput number is predicted.** Not for an H200, not for anything.
  The machine these figures came from has no serving stack and no CUDA device, so
  any throughput figure written here would be invented rather than measured.
* **154.5:1 is a property of the prompt *and the decode regime*, not of any
  accelerator.** It says how much of the work is prefill for this corpus under
  verdict-only decoding. It says nothing about how fast prefill runs anywhere, and
  it does not survive a change of decode regime: the same corpus scored with
  reasoning on gives **5.74:1** — mean prompt 2,825.1, mean output 492.4 over the
  19,993 calls carrying usage in the first 15,000 attempts of
  `data/comparison/runs/gemma_26b_primary/attempts.jsonl` [M] — 27× different, and
  the difference is entirely on the output side. Quote 154.5:1 only with the
  verdict-only scoping 9.1 already gives it.
* **H-B may well come back null** — that is its pre-registered expectation. A null
  is a finding, not a failed benchmark: it retires the ordering advice in §4
  item 11 and leaves the caching advice standing.
* **The default-flag claims in 9.3, the block-granularity note in 9.1, and every
  metric series name in 9.5 are [R]** until confirmed on the serving host. They
  are written as protocol steps for that reason.
* **~39k tokens is an extrapolation**, from a chars-per-token ratio measured on this
  corpus, not a tokenizer count. H-C exists to replace it with a real one.
* **Nothing here is a production change.** No code under `src/` changes, no run
  artifact is written, no spend-ledger row is created, and no byte under
  `data/comparison/**` or `data/comparison_verdict_only/**` is touched. The
  exclusion of the bench client from the paid runner in 9.6 is **documented, not
  implemented**.

---

## 10. Partition identity — what sharding would require

This section specifies preconditions. It implements none of them, and its verdict
is that none of them should be implemented now. It exists because an adversarial
review proposed a one-line fix — "add a cross-action `execution_keys` disjointness
check at plan load" (§4 item 8, from F5) — and that fix is wrong twice over: the
global form rejects two already-published plans, and the narrow form can never
fire. §4 item 8 is marked WITHDRAWN and points here.

**Placement and numbering.** Appended at end of file, after §9, which itself
landed after the References block; §9 states the renumbering rule. **There is no
registration step**: `scripts/check_doc_anchors.py` scans every `research/*.md`
end to end at a grandfather allowance of zero, so this section's anchors were
checked the moment it landed, and a future section's will be too without anyone
declaring it anywhere. §1–§9 and References are byte-identical afterwards except
for the two lines named in 10.1.

**Markers** follow the document header: **[M]** here carries the command beside
it, **[V]** is read out of the code in this tree. Code is cited by **symbol**,
never by line number: sibling nodes were editing three of these files while this
was written, and a line anchor would already be stale.

### 10.1 Why the proposed check is wrong — both formulations

**The global form rejects two shipped plans.** Reproduce with:

```
.venv/bin/python -c "import json;[print(p,'|',a['id'],a['stage'],a['workload'],a['execution_keys']) for p in ['data/comparison/run_plan.json','data/comparison_verdict_only/run_plan.json'] for a in json.load(open(p))['actions']]"
```

`data/comparison_verdict_only/run_plan.json` has 4 actions, **all** on workload
`unique_exact_pairs_primary` with `execution_keys: null` [M]. Null means *the whole
workload*, so under a global rule all four intersect maximally.
`data/comparison/run_plan.json` has 8 actions, 4 of them on that same workload
[M]; `e2b_smoke` enumerates exactly one key —
`{"evidence_i": 0, "execution_sha256": "39adf92c…", "stmt_i": 3}` — which the three
null-key `*_primary` actions all cover. These arms deliberately score **one corpus
with different models**. Overlap is the experiment, not a bug, and a global
disjointness rule is a rule against comparison.

**The narrow form — disjointness within one `(stage, workload)` lane — is
vacuous.** `load_run_plan` in `src/indra_belief/comparison/contracts.py` builds
`lanes = [(action.stage_id, action.workload) …]` and fails a plan that repeats one
[V]. Two actions can never share a lane, so a check scoped to a lane has no pair to
compare.

**And the reason it is vacuous is the actual answer to F5.** `expected_execution_id`
in `src/indra_belief/comparison/replay.py` digests
`{"model", "workload_mode", **execution_identity(source)}`, where
`execution_identity` is `{eligible_position, paper_statement_hash, source_hash,
evidence_json_sha256}` — **no `action_id`, no `run_id`** [V]. The spend identity of
a row is therefore a function of `(model, workload, row)` alone. The same loader
requires each spend-guard model to belong to exactly one stage, so
`(stage_id, workload)` uniqueness *is* `(model, workload)` uniqueness. Two actions
in one plan cannot mint the same `execution_id`, whatever their `execution_keys`
say. **The lane check already is the cross-action disjointness check** — stated
over lanes rather than over rows, which is why a reviewer looking for a set
intersection does not find it.

So F5's premise ("nothing prevents paying twice for the same
`(stmt_i, evidence_i)`") is false for every plan shape the loader admits. It
becomes true only once the lane key is relaxed — which is precisely what sharding
requires. **F5 is a precondition of sharding, not a defect of today's loader**, and
that is the one-line correction made at F5 and at §4 item 8.

### 10.2 What already exists and is reusable

Nothing below is proposed; all of it is in the tree today [V].

| concern | existing symbol | file |
|---|---|---|
| narrow to a corpus slice | `ReplayIndex.for_workload` (also asserts declared cardinality) | `src/indra_belief/comparison/replay.py` |
| narrow to enumerated rows | `ReplayIndex.select` (rejects foreign/repeated keys *within one action*) | `src/indra_belief/comparison/replay.py` |
| the two composed | `_scopes` — `for_workload(...)` then `.select(action.execution_keys)` | `src/indra_belief/comparison/runner.py` |
| cross-action lane uniqueness | `load_run_plan` | `src/indra_belief/comparison/contracts.py` |
| content-addressed row identity | `ReplayIndex._validate_all` | `src/indra_belief/comparison/replay.py` |
| spend identity of a row | `expected_execution_id` / `execution_identity` | `src/indra_belief/comparison/replay.py` |
| reader configuration string | `_named_profile` | `src/indra_belief/calibration_constants.py` |

Two of those carry detail a shard design depends on.

**The row digest is derivable, not merely stored.** `ReplayIndex._validate_all`
computes `canonical_sha256([workload, stmt_i, evidence_i, paper_statement_hash,
source_hash, evidence_json_sha256])` for every row and compares it to the row's
`execution_key_sha256` **only when that field is present** — an absent field is
accepted [V]. So a partition predicate over that digest needs no new substrate
field and no regeneration of a substrate whose generator is not in this repository
(§8 item 1). In the shipped substrate the field is in fact stored: 5,000 of the
first 5,000 rows of `data/comparison/grounding_replay/executions.jsonl` carry it
[M] (`ls -la` on that directory shows the file at 71,250,757 bytes).

**`select` accepts two key shapes** — `{stmt_i, evidence_i, execution_sha256}`,
where the digest is `canonical_sha256` of the *whole row*, and
`{stmt_i, evidence_i, execution_key_sha256}`, the semantic-identity digest above
[V]. The one shipped enumeration uses the whole-row form. A shard predicate must
range over the **key** digest, not the row digest: the row digest changes if any
field of the row changes, which would move rows between shards on a substrate edit.

### 10.3 The five fields partition identity would need

**`profile_id`.** *Correcting the review that commissioned this section:* a
`profile_id` **does** exist. `src/indra_belief/calibration_constants.py` carries it
as a hand-written literal in `_PROFILE_META`, shape
`{model}@prompt-{sha12}@{fit_gold}` (e.g.
`remote-gemma-4-26b@prompt-b44638216740@eval_curation_v1`) [V]. It is not the right
form to borrow. The *derived* one is: `_named_profile` builds
`reader_configuration` as `{reader_model}@prompt-sha256:{prompt_sha256}` with the
full digest, and `_FITTED_CONFIGS` keys its lookup on the bare
`(model, prompt_sha256)` tuple [V]. A shard key must use the derived full-digest
form or that tuple — a 12-hex prefix is a display convenience the module itself
labels as such, and a fit-gold name is not part of what makes two scored rows
comparable. **Reuse the form, not the literal.** A shard is comparable only within
one reader configuration.

**`partition_group`.** The disjointness **scope**, and the field that makes the
whole design possible. Actions in *different* groups MAY cover identical rows —
that is exactly the shipped four-arm comparison. Actions in the *same* group MUST
NOT. Without this field there is no way to say "these four actions are one
experiment" and "these sixteen actions are one sharded pass" in the same schema.

**`shard_ordinal` and `shard_count`.** Position and cardinality. Validation asserts
the declared ordinals of a group equal `range(shard_count)`.

**`partition_predicate`.** A canonical, **total** function from a row to exactly one
ordinal, e.g. `int(execution_key_sha256[:8], 16) % shard_count == shard_ordinal`.
Canonical means named and versioned in the schema, not written per plan.

**Scope.** Disjointness and coverage are checked **only within
`(model, profile, workload, partition_group)`**.

**Coverage is opt-in, and must not follow from group membership.** *Correcting a
second time:* an unconditional "actions sharing that 4-tuple must together cover
the workload" breaks a shipped action. `e2b_smoke` is the only action in either
plan with non-null `execution_keys` [M], and it pins **one** row of a
33,413-execution substrate on `unique_exact_pairs_primary`. A coverage MUST that
fires on membership rejects it. So a group declares its own coverage obligation —
a pinned-key probe declares no group, or a group whose coverage is partial — and
the check does not range over actions outside it.

### 10.4 Predicate, not enumeration

This is the load-bearing argument, and three of its four legs are about the plan
file rather than about memory.

**Disjointness becomes a theorem.** A total function lands each row in exactly one
ordinal, so no two shards of a group can share a row *by construction*. Validation
is then `O(shard_count)` — assert the declared ordinals equal `range(shard_count)`
— rather than an intersection over enumerated keys. Scoped honestly: the row
*filter* stays `O(rows)` either way, because `ReplayIndex.for_workload` already
walks every row. What the predicate removes is the `O(enumerated keys)` term, in
the plan file and in the disjointness check.

**The plan file is the real scaling defect, because plan bytes are plan identity.**
`load_run_plan` reads the whole file through `stable_read`, digests those bytes
with `hashlib.sha256` into `plan.sha256`, and strict-parses the same bytes — every
load, for every action [V]. An enumeration therefore lives *inside* the artifact
that is hashed. Measured on the shipped key:

```
.venv/bin/python -c "import json;e=[a for a in json.load(open('data/comparison/run_plan.json'))['actions'] if a['id']=='e2b_smoke'][0]['execution_keys'][0];print(len(json.dumps(e,sort_keys=True,separators=(',',':'))))"
```

**113 bytes per enumerated key** at the smallest serialization JSON allows [M]. At
the doc's own stated scale of 90M rows (§F5, §F6 — that figure is theirs, not a new
claim here) that is **10.2 GB of plan text**, a floor: the shipped plan is
pretty-printed, and the whole of `data/comparison/run_plan.json` is **8,622
bytes** today [M]. A ~10 GB plan is read, canonically revalidated and SHA-256'd on
every load, against a preflight §2 already measures at 2.6× amplification and an
8.3 GB peak.

**The in-memory set is the smaller half of the same problem** [M]:

```
.venv/bin/python -c "import tracemalloc,hashlib;N=1000000;tracemalloc.start();s={(i>>4,i&15) for i in range(N)};c,_=tracemalloc.get_traced_memory();tracemalloc.stop();print('tuple',c/N);tracemalloc.start();h={hashlib.sha256(str(i).encode()).hexdigest() for i in range(N)};c,_=tracemalloc.get_traced_memory();tracemalloc.stop();print('hex64',c/N)"
```

At n=1,000,000 this machine traces **121.4 B per `(int, int)` element** (115.8 MiB)
and **138.6 B per 64-hex-string element** (132.1 MiB). At 90M rows a materialized
key set is **10.9 GB** as coordinate tuples or **12.5 GB** as hex digests, on top
of the measured 8.3 GB preflight peak — and unlike the plan bytes, it is per
process.

**A predicate also preserves an invariant an enumeration does not.** See 10.5's
fifth item: with rows guaranteed disjoint, no two shards can mint the same
`execution_id`. That is not an efficiency argument.

**And it is stable under row reordering.** The predicate reads a content-addressed
digest, so re-emitting the substrate in a different order moves no row between
shards. An enumeration of coordinates does not have that property and would need
its own proof — duplicating one the substrate already carries.

### 10.5 What a predicate does not solve

Five things. The first four were named by the review; the fifth is where the money
actually is.

**1 — The lane key rejects sharding outright.** `load_run_plan` keys lanes on
`(stage_id, workload)` [V]. N shards of one workload in one stage are N actions in
**one** lane and would be rejected today. The lane key would have to become
`(stage_id, workload, partition_group, shard_ordinal)`. Nothing about a predicate
does this for you.

**2 — Action fields are an exact-set match, so new fields break every published
plan.** `load_run_plan` holds a frozen `exact_fields` set and fails with
`action {index} fields differ` on `set(item) != exact_fields` [V]. Adding
partition fields changes the schema, therefore every plan's `sha256`, therefore
every amendment's `predecessor_sha256` (the amendment block has its own exact-set
match). **Absence-compatible defaults are mandatory** or both shipped plans stop
loading.

**3 — Each shard is its own spend lane, and a shared ledger is a live constraint,
not a hypothetical.** `load_run_plan` bounds per-stage exposure by summing
`min(stage.cap_usd, total)` over **distinct ledger paths** [V]: sharding
multiplies WAL files, it does not multiply the cap, and N shard caps must still sum
under the stage cap. Beyond the cap, `_relevant_attempts` in
`src/indra_belief/comparison/runner.py` raises
`RunnerError("shared spend ledger contains a foreign open attempt")` when a shared
ledger carries an *unfinished* attempt outside this action's expected
execution-id set [V]. Six of the eight actions in `data/comparison/run_plan.json`
share `data/comparison/runs/spend.ndjson` [M], so this path is exercised by shipped
configuration: shards on a shared ledger cannot have concurrent open attempts.
Give each shard its own ledger **and** its own output, or serialize them. `_reconcile`
bijectivity (`set(raw) != set(ledger)`) is per-action and keeps holding either way
— but only because `_relevant_attempts` scopes the ledger side by expected
execution id first.

**4 — Overlap across different `partition_group`s is not double-spend and must
never be flagged as one.** It is the comparison design. A checker that reports it
will be disabled, and then it will not be there for case 5.

**5 — `execution_id` does not carry the ordinal, and this is the expensive one.**
Relaxing the lane key (item 1) to admit `shard_ordinal` **without** adding the
ordinal to `execution_identity` leaves the spend identity of a row a function of
`(model, workload, row)`. Two shards that both claim a row therefore mint the
**identical** `execution_id` — and on a shared ledger (item 3) the second action's
`_relevant_attempts` adopts the first's attempts as its own, since its only filter
is execution-id membership [V]. *That* is the double-spend F5 was reaching for, and
it is reachable only after item 1. A total predicate forecloses it: no row lands in
two shards, so no collision arises. The predicate's totality is therefore not an
optimization — it is what preserves an identity invariant the runner currently
gets for free from the lane check. If ordinals are ever permitted to overlap,
`execution_identity` must gain the ordinal **first**.

**Composition with what has landed since.** Per-source retirement is derived and
never stored — `_settled_reason` is folded inside `load_resume` over the same
append-only rows, and `_run_source` re-checks `resume.settled` as the last gate
before a provider reservation [V]. Both are per-action over that action's own
durable output, so they shard cleanly *provided each shard keeps its own output
file*; that is a second, independent reason for the one-output-per-shard rule in
item 3. Likewise `_calls_for_attempt` on `SpendGuard` in
`src/indra_belief/spend_guard.py` is keyed by attempt id and `_reservations` is
insert-only (no `del`, `pop` or `clear` anywhere in `src/`) [V] — sharding
multiplies guards rather than growing one, so neither is disturbed.

**Ordering is a different question.** §9's ordering arms concern the sequence in
which rows arrive, for prefix-cache reuse. An ordering policy **composes with**
partition identity and does not replace it: a predicate says which rows a worker
owns, an ordering policy says in what sequence it sends them.

### 10.6 Verdict — do not implement

**There is no partition concept to fix.**
`grep -rn "shard\|partition" src/indra_belief --include="*.py"` returns eight
lines [M]: **three** `str.partition()` string splits (one in
`src/indra_belief/model_client.py`, two in `src/indra_belief/comparison/cli.py`);
**two** prose strings — a `provider_source` label in
`src/indra_belief/model_client.py` and a `does not partition its errors` message in
`src/indra_belief/comparison/report.py`; and **three** comments — two in
`src/indra_belief/results.py` about strict *bucket* partitioning in the stratified
residual, one in `src/indra_belief/comparison/runner.py` about the
completed/quarantined/pending partition — a different sense of the word. None is a
corpus shard. Nor is "shard" itself much of a presence in the code: re-derived
with `grep -rni shard src viewer/src scripts tests` [M], `src/`, `viewer/src/` and
`tests/` contain the string **zero** times and the only hit under `scripts/` is a
comment in `scripts/modularity_baseline.py`. Two `viewer/scripts/` contract
scripts do use the word, outside that grep's scope, for a hypergraph
work-partition — which files an author owned — and never for a corpus shard. The
claim is deliberately about code rather than about "the repository": this section
is *about* sharding and says the word throughout, so a corpus-wide count would
measure the prose, not the substrate. F5's
failure mode cannot occur today, and 10.1 shows why: the lane check, plus
`(model, workload)`-derived execution identity, plus `ReplayIndex.select`'s
within-action repeat rejection, cover every shape the loader admits.

**And there is no consumer.** §8 item 4 records no consumer, latency target,
overload policy or auth model for the serving path. 90M rows is the scale §F5 and
§F6 name, not a commitment anyone has made.

So: **implement nothing.** Not the disjointness check §4 item 8 asked for — it is
withdrawn. Not the five fields — they are a schema break (item 2) in service of a
capability nobody has requested. This section is the precondition list to satisfy
*if and when* sharding is actually needed, and the order is fixed by 10.5: the
`execution_identity` question (item 5) is settled **before** the lane key is
relaxed (item 1), never after.

**What this section does not claim.** No production code changed; no file under
`src/`, `tests/` or `data/` changed; the only edit outside this section is the
two-line correction in 10.1. Its `[M]` figures are memory and byte counts on this
machine, not throughput. It asserts nothing about whether a predicate would be
*fast enough* at 90M rows — that is a measurement no one can make until the
substrate at that scale exists, and §8 item 1 notes the generator that would build
it is not in this repository.
