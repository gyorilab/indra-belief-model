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

Two execution paths that share **zero scoring code**, and no server.

```
BATCH (paid, hardened)                    LIVE (library)
──────────────────────                    ──────────────
indra-belief-comparison run               score_evidence(stmt, ev, client)
  contracts.load_run_plan                   ScoringRecord(statement, evidence)
  ReplayIndex.load  ── validates ALL          __post_init__ -> resolve_entities()
     33,413 executions, re-hydrates              -> gilda + bio_ontology  (7.1 GB)
     and digest-checks EVERY prompt            record.tier1_auto_reject()
  load_resume(action.output) x N_actions      _select_examples()   at call time
  SpendGuard(ledger) ── replays ALL events    ScoringRecord.format_user_message()
  ── ready-before-token boundary ──           client.call()
  ThreadPoolExecutor(workers <= 8)            _parse_verdict()
    replay.score_execution                    verdict_to_score()  -> 0.5 on fail
      ReplayIndex._record()  <- renderer A
      client.call()
      replay.parse_response()  <- parser A
```

`test_package_lazy_import.py` asserts the batch transport imports no
`indra_belief.scorers.*`. That test documents the separation as intentional. It is
also why the two paths can drift without anything failing.

**Deployment surface today:** no Dockerfile, no compose file, no API server (no
fastapi/flask/uvicorn anywhere), one console script (`indra-belief-comparison`).
The realtime path is a library import, not a service.

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
action in the plan**, not just the one being run (`runner.py:453-457`). Machine has
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

### F1 — The live and batch paths have already diverged **[V] CRITICAL**

`scorer.py:62` branches `_VARIANT` over exactly
`("disconfirm", "disconfirm_relnature", "disconfirm_relnature_rf")`. **There is no
`verdict_only` branch.** `_prompts_verdict_only` is imported by exactly one file —
`scripts/build_verdict_only_replay.py`, unpackaged. And
`grep -rln format_user_message tests/` returns **nothing**.

So `score_evidence()` cannot emit the prompt that produced the run shipped today,
and no test ties the two renderers together. Two independent renderers exist:

| | live | batch |
|---|---|---|
| user message | `ScoringRecord.format_user_message()` | `ReplayIndex._record()` |
| provenance gate | recomputed from `has_grounding_signal` | trusted field `row["provenance"]` |
| few-shots | `_select_examples()` at call time | frozen prefix by sha256 |
| score on parse failure | **returns 0.5** (`_prompts.py:294`) | **returns None**, retries (fixed today) |

That last row is the divergence made concrete: the 0.5-fabrication was fixed on
the batch side this session and still exists on the live side.

This directly defeats the constraint *"prove which exact prompt+model produced any
given score"* for anything scored through the library API.

### F2 — Ledger replay is quadratic **[V] CRITICAL**

Reproduced by driving `_LedgerReplay.apply` over the real ledger:

| events | time | µs/event |
|---|---|---|
| 3,000 | 0.204 s | 67.8 |
| 6,000 | 0.467 s | 77.8 |
| 12,000 | 1.363 s | 113.6 |
| 24,000 | 4.986 s | **207.8** |

Per-event cost is linear in n, so total is O(n²). Cause is two full scans of
`self._reservations`, which never evicts:

- `spend_guard.py:1068-1072` — `sum(item.get("attempt_id") == attempt_id for item in self._reservations.values())` per `call_reserved`
- `spend_guard.py:1190-1192` — `[item for item in self._reservations.values() if ...]` per outcome

Both become O(1) with a `dict[attempt_id] -> list` index. This is the highest
leverage-to-effort change in the repository.

### F3 — 32 workers is contractually impossible **[V] HIGH**

`contracts.py:30` sets `MAX_WORKERS = 8`; `:553` rejects more. And
`AMENDABLE_FIELDS = {"workers": (6, MAX_WORKERS)}` — the **only legal amendment is
exactly 6→8**. Raising it edits the module, changes `plan.sha256`, and invalidates
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

`contracts.py:611-621` checks unique action ids, run_ids, output paths and
`(stage_id, workload)` lanes. There is **no check that two actions'
`execution_keys` are disjoint**; `ReplayIndex.select` rejects repeats only *within*
one action. Shard 90M rows across N actions and nothing prevents paying twice for
the same `(stmt_i, evidence_i)`. Fix is one hash-set check at plan load.

### F6 — One bad row halts the whole action **[R] HIGH**

`runner.py:823,831` — any worker exception or `outcome.failure` sets
`stop_scheduling = True`. At the measured 0.057–0.097% off-grid rate, 90M rows
gives ~63,000 wedge candidates. Head-of-line blocking is not a viable failure mode
at that scale. Quarantine the source, keep scheduling.

### F7 — Semantic dependencies fail open **[R] HIGH**

Gilda and ontology exceptions become empty results rather than errors
(`entity.py:409`, `:472`), which changes routing and prompt content instead of
surfacing a dependency failure. A missing few-shot bank logs a warning and scores
with no examples (`scorer.py:114`). For a scientific scorer these should fail
closed or be a separately versioned mode.

### F8 — Statement belief can report `correct` with belief 0.0 **[R] HIGH**

A low-confidence `incorrect` row is excluded from the belief numerator but does not
count as a credible rejection (`_CREDIBLE_LLM_CONF = {"high","medium"}`), so
`verdict_statement` can be `"correct"` while the hard gate returns 0.0. I flagged
the same gap independently earlier this session from the retry-policy angle.

---

## 4. The refactor plan

Ordered by impact ÷ things-added, honouring the standing preference for removing
over adding. Each names the SOLID principle only where it clarifies the change.

### Remove

1. **The second prompt renderer.** One `render(row) -> (system, messages)` module,
   consumed by both paths, plus **one parity test**: materialize a `ScoringRecord`
   for a known execution, render both ways, assert byte equality. That single test
   is worth more than the digest apparatus, because it checks what the digest
   structurally cannot — the digest's generator and verifier are the same function
   (`build_verdict_only_replay.py:177` calls `ReplayIndex._record`; `main_request`
   compares against it). *SRP: `ReplayIndex` currently loads, validates, hydrates,
   orchestrates scoring and derives deterministic results.*
2. **Prompt bodies from ledger and attempts rows.** Store the refs already in the
   row. ~100× on durable state; provenance unchanged.
3. **Import-time `MONO_VARIANT`.** `scorer.py:61` freezes scientific semantics into
   a module global at import. Pass a profile object. *DIP — and this is precisely
   why the live path cannot reach `verdict_only`.*

### Fix (small, surgical)

4. **Two dict indexes in `_apply`** → O(n²) becomes O(n). **[F2]**
5. **Index `resume.rows` by source key** — `runner.py:683` scans all rows per
   pending source.
6. **Preflight only the selected action** (`runner.py:453-457`).
7. **Quarantine wedged sources** instead of `stop_scheduling` **[F6]**.
8. **Cross-action `execution_keys` disjointness check** at plan load **[F5]**.
9. **`dict(LOCAL_MODELS[model_name])`** at `model_client.py:1014` — all workers
   currently alias one mutable config dict, and `runner.py:522` writes `timeout`
   into it.
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
unconditionally (`scoring_record.py:37-38`). So a live API over arbitrary new
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
  version skew. One image is the parsimonious default.
- **Relatedly:** `pip install .` installs the heavy closure regardless of lazy
  imports, so anyone pursuing a lean image must split the *dependency* declaration,
  not just the import sites.
- **A critic claimed the spend ledger is "~4,000 lines."** It is **2,163**
  (`spend_guard.py`). 3,928 is the total for `spend_guard` + `replay` + `runner`.
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
- **Cost accounting does not transfer.** `_ActionGuard.reserve_call`
  (`runner.py:214-217`) calls `price_for(provider_model_id)` and raises
  `SpendGuardError` on `None`. A self-hosted lane has no per-token price and needs
  an explicit decision: exempt the lane from spend accounting, or price GPU-seconds.

---

## 8. Open questions

1. **The original substrate generator is not in this repository.** Only the
   verdict-only *derivation* script is. The generator that produced
   `data/comparison/grounding_replay` — which every shipped paper number rests on —
   cannot be re-run. Replay is reproducible; regeneration is not.
2. **`scripts/` is ~36,000 LOC**, comparable to `src/`, unpackaged, outside
   `testpaths`, and reaches into private symbols (`scorer._select_examples`,
   `ReplayIndex._record`). Some of it is contract, not scripting.
3. **Calibration identity** fingerprints only the `system` string of `monolithic`
   calls (`calibration_constants.py:180`) — excluding few-shot content, the example
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
