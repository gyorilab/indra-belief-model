# Serving and deployment: what is measured, what is decided, what is open

> **Provenance note (2026-08-25).** This is a historical record. The repository
> was later cut to prod infrastructure plus the future-LLM compare loop, so some
> paths cited below name code that no longer exists — the comparison harness, the
> spend ledger, the viewer, the paper-replication lane, the probe battery. The
> measurements and conclusions stand as recorded; git history holds the code they
> were measured against. No guard checks these citations any more.

Status: investigation record, 2026-08-03. Not a plan and not a proposal that has been
approved. It records measurements about deploying the monolithic belief scorer, one
substantive finding about how the scorer grounds entities, and the constraint that
governs any change to either.

Every claim below is labelled:

* **[M]** measured — a command that can be re-run, or an artifact path that exists.
* **[V]** verified in code — read at a named file and symbol.
* **[R]** reasoned — an inference from [M]/[V], never given to three significant figures.

Citations name a file and a symbol, never a line number: line numbers rot, and
`scripts/check_doc_anchors.py` now validates cited symbols against the tree, over every
`research/*.md` including this one.

Some measurements were produced by throwaway scripts in a session scratchpad under
`/private/tmp`. That directory is ephemeral and is **not** a durable citation. Where a
number came from one, the derivation is described well enough to re-run from the repo.

---

## 1. What is deployed today

**There is no serving process.** [V] The repository ships a batch pipeline and a library.
There is no HTTP entry point, no `main` that listens, and no container that answers a
request. Section 5 sketches what one would look like; nothing in it is built.

**The batch image exists and is small.** [M] `docker images` on this host:

```
indra-belief:batch   3c717d729b4c   535MB
indra-belief:test    7b17d50d20e4   562MB
```

[V] `Dockerfile` installs neither `gilda` nor `indra`, deliberately, and says why in its
own header: the batch path does not ground, it hydrates every prompt from
content-addressed refs in a frozen substrate. Deps are installed explicitly rather than
via `pip install .` because `pyproject.toml` declares both as hard requirements.
[V] `Dockerfile` states the saving as ~1.44 GB — indra 187 MB installed plus a 470 MB
ontology cache, gilda 2.5 MB installed plus 784 MB of resource files. Those four figures
were not independently re-weighed when this section was written; §7.1 now re-weighs all
four, and the saving is ~724 MB — the package half reproduces, the resource half does not.

**The omission is gated at build time, not asserted in prose.** [V] `Dockerfile` carries a
`test` stage that runs the batch suites inside the container with neither package
installed. Its header records why: an earlier host-side check that claimed to block the
imports was a no-op, because it used the `find_module`/`load_module` finder API removed in
Python 3.12. The container is the only honest check.

**The two shared kernels are import-light.** [M] Importing
`indra_belief.scorers.monolithic.scorer` takes 0.035 s and leaves both `gilda` and `indra`
absent from `sys.modules`:

```
PYTHONPATH=src .venv/bin/python -c "import time,sys;t=time.time();\
import indra_belief.scorers.monolithic.scorer as S;print('%.3f'%(time.time()-t),\
'gilda' in sys.modules,'indra' in sys.modules)"
# 0.035 False False
```

Grounding is pulled in only by *constructing* a `ScoringRecord`, because
[V] `ScoringRecord.__post_init__` calls `ScoringRecord.resolve_entities` unconditionally.
That single fact is why a prepared-payload serving mode (§5) can live in the 535 MB image
and a raw-Statement mode cannot.

**What the batch replays.** [M] `data/comparison/grounding_replay/manifest.json` (mtime
Jul 20 16:49, `mono_variant: disconfirm_relnature_rf`) reports cardinality
`executions 33413, relation_executions 17257, tool_executions 2704, entities 6479,
entity_names 1644, lookup_targets 853, abbreviation_executions 850`. The verdict-only
sibling `data/comparison_verdict_only/grounding_replay/manifest.json` (mtime Jul 31 14:26)
carries the same `entities`/`entity_names`/`executions` and `relation_executions: 0` — its
`generation_contract.derived_from` lists `changed: [main system prompt, few-shot prefixes,
main_prompt_base_sha256, relation-nature call removed]` and `unchanged: [executions,
routes, evidence, entities, lookups, coordinates]`. So the grounding substrate is shared
across the two published runs and is a file, not a live call.

---

## 2. The grounding finding

### 2.1 Corrections to the claims this document was built from

Four claims that were fed into this investigation are wrong or imprecise. They are
corrected here rather than quietly dropped, because the first one is load-bearing and a
reader who has heard the original will otherwise think this document is the confused one.

**Correction 1 — "13,599 agents, 100% carrying a non-TEXT db_ref."** The count 13,599 is
exact; the word *agents* is not. [M] Walking every dict in
`data/corpora/indra_paper_unique_pairs_20260717_statements.json` that carries a `db_refs`
key reproduces 13,599 and localizes them:

```
  5233  [].evidence[].context.species
  1865  [].evidence[].context.location
  1412  [].evidence[].context.cell_line
  1384  [].members[]
   944  [].evidence[].context.cell_type
   703  [].evidence[].context.organ
   566  [].obj      566  [].subj      431  [].enz      431  [].sub
    59  [].evidence[].context.disease
     5  bound_conditions[].agent  (2 under members[], 2 under enz, 1 under subj)
  ----
 10216  evidence[].context BioContext nodes   (75.1%)
  3383  statement agent slots                 (24.9%)
```

**Correction 2 — "resolve_entities fails on 72% of those names."** That 72% was measured on
a random sample of the 13,599, three quarters of which the scorer never grounds.
[V] `ScoringRecord.resolve_entities` grounds `ScoringRecord.subject` and
`ScoringRecord.object` only — `agent_list()[0]` and `[1]` — and nothing anywhere in
`src/` reads `evidence.context`. All four mismatch examples that motivated the 72% are
BioContext fields of `evidence.context` — *liver* is the **organ**, *monocyte* the
**cell_type**, *Homo sapiens* the **species**, *Cell Membrane* the **location**. None of
them is ever passed to `GroundedEntity.resolve`.

Measured on the population that *is* grounded, the number inverts. [M] Running the
production `GroundedEntity.resolve(name)` over the 1,777 unique `(name, db_refs)` entities
occupying grounded slots:

```
gilda agrees with db_refs          1734   97.58%
gilda same namespace, different id    2          (PDK1, TAT)
gilda grounded to a namespace the agent lacks
                                     11
gilda UNGROUNDED                     30    1.69%
```

The real disagreement is **13 entities, not 72%.** Any argument for a db_refs-first
grounding policy built on "72% ungrounded" will not survive contact with the code.

The coverage of the grounded slots is a separate, genuinely strong number:
[M] 100% of the 3,378 grounded slots carry HGNC or FPLX, and 0 carry no non-TEXT ref.
[M] Independent cross-check that `agent_list()[0:2]` really is the whole agent
population here: every one of the 692 `Complex` statements in this corpus has arity
exactly 2 (`{2: 692}`), and 1,384 `members[]` db_refs nodes ÷ 692 = 2.0.

**Correction 3 — "~41% of requests issue two provider calls."** 41.0% is the *statement*
share of `Complex` [M 692/1689 = 0.4097], but the request unit is the (Statement,
Evidence) pair, and `Complex` statements carry more evidence than average. [M] By pair:
17,905 / 33,361 = 53.7% of corpus pairs are `Complex`; the substrate that was actually run
records `relation_executions 17257 / executions 33413` = **51.7%**. Roughly half of
traffic, not two fifths.

**Correction 4 — "only 11 distinct few-shot prefixes exist across all 20 statement
types."** [M] `_select_examples` in `src/indra_belief/scorers/monolithic/scorer.py` knows
**11** statement types (union of its type bank, base pairs and adjacency table), and each
of the 11 yields a *different* 14-example selection — so 11 is the count of known types,
which happens to equal the count of distinct selections over them, not a collapse ratio.
On the 12 statement types present in the paper corpus the count is **9** distinct
selections, four of them (`Acetylation`, `Deacetylation`, `Deubiquitination`,
`Methylation`) sharing one. [M] That 9 is confirmed independently by the frozen substrate:
`prompt_components.main_message_prefixes` has 9 entries and
`generation_contract.message_prefix_refs_by_statement_type` has 12 keys over 9 distinct
digests. Live code and frozen artifact agree.

### 2.2 The finding, stated correctly

INDRA Statements already carry grounding. The scorer throws it away and re-derives
identity from the entity's **name** with gilda. [V] `ScoringRecord.resolve_entities` calls
`GroundedEntity.resolve(name, raw_text)`; [V] `GroundedEntity.resolve` calls
`_cached_ground(name)` and takes gilda's top hit as `(db, db_id)`. [V]
`ScoringRecord.agent_db_refs` exists, returns `agents[index].db_refs`, and — [M] by
repo-wide grep over `src/`, `scripts/` and `tests/` — **has zero callers.** So does
[V] `ScoringRecord.raw_grounding`. Both accessors are sitting there unused; the first is
evidently what a db_refs-first policy was expected to use.

This is not a 72%-of-everything problem. It is a small, sharply-defined correctness
problem plus a packaging problem, and the two should not be argued as one.

### 2.3 IDENTITY versus CONTEXT

The single most useful decomposition here. What `GroundedEntity.resolve` produces splits
three ways.

**IDENTITY — db_refs supplies it outright, no lookup.**

* `db`, `db_id`. [M] 100% of grounded slots carry HGNC or FPLX.
* `is_family`. [V] Today it is `db == "FPLX"` on gilda's top hit, inside
  `GroundedEntity.resolve`. db_refs answers it as a pure predicate, `"FPLX" in db_refs` —
  and answers it correctly where gilda does not (§2.4).

**CONTEXT — needs a lookup, but an ID-KEYED one: a static join, not a fuzzy matcher.**

* `all_names` / aliases: [V] `_cached_get_names(db, db_id)` in
  `src/indra_belief/data/entity.py` — id → names.
* `description`, `is_pseudogene`: [V] `_cached_get_desc`, derived from the same names.
* `family_members`: [V] `_get_fplx_members`, `bio_ontology.get_children("FPLX", id)` —
  id → children.

[M] The id → names/members table covering this whole corpus is 1,644 entries / 519.2 KB.
The **full** HGNC + FPLX table (45,654 HGNC ids, 642 FPLX ids) is 6,499,410 bytes = 6.2 MB.
Against the ~1.44 GB the Dockerfile attributes to gilda + indra that is a ~230× reduction;
against the re-weighed ~724 MB of §7.1 — the figure this document stands behind — it is
~117×. Either way it removes the matcher, not merely the data.

**IRREDUCIBLY A GROUNDING CALL — string → entity, with no db_ref anywhere for it.**
Everything inside [V] `GroundedEntity._verify_raw_text`: re-grounding the reader's
`raw_text`, its gilda score, competing candidates, the top name for the text, and the
resulting verification status. db_refs cannot supply this, because the input is a span of
prose, not an identifier.

### 2.4 What db_refs-first would buy

Small, real, and specific. [M] The 13 disagreeing entities touch **251 of 66,722 slot
observations (0.376%)**, of which **244 (0.366%) flip the `is_family` flag**. What the
current policy injects into the user message for the three worst:

```
MYL   gilda HGNC:9113 → "MYL (HGNC: PML, aliases: RNF71, PP8675, TRIM19, Protein PML, …)"
      db_refs says FPLX:MYL.   gilda resolved myosin-light-chain to PML.       14 slots
JUN   gilda FPLX:JUN_family → "JUN (protein family, includes JUN, JUNB, JUND — a
      family-level claim is supported by evidence about any specific family member)"
      db_refs says HGNC:6204, the specific gene.                              192 slots
PDK1  gilda HGNC:8816 (PDPK1)   vs   db_refs HGNC:8809 (PDK1)                   4 slots
```

The JUN case is the load-bearing one: 192 slot observations hand the model an explicit
*family-substitution licence* for a claim the statement itself says is gene-specific.
That is a substrate defect of exactly the class this project has repeatedly chosen to fix
deterministically rather than nudge the LLM around.

**The bigger adjacent win, from the same data.** [M] Over the 377 unique
MISMATCH+AMBIGUOUS verdicts (2,665 slot-weighted):

```
REPAIRABLE by checking raw_text's hit against the agent's FULL db_refs set  137 uniq / 1185 wtd
cross-namespace concept (MESH/GO/CHEBI vs protein)                          147 uniq /  840 wtd
protein_vs_protein — a real collision candidate                              93 uniq /  640 wtd
```

The 137 are `ABL1 / "Abl" → HGNC:76`, `CCL5 / "RANTES" → HGNC:10632`,
`ADARB1 / "ADAR2" → HGNC:226` — flagged AMBIGUOUS today only because [V]
`GroundedEntity._verify_raw_text` compares the re-grounded text against gilda's *single*
top hit for the claim rather than against the agent's whole ref set. That is 4.7× the blast
radius of the identity fix (1,185 slot observations against 251), uses only data already in
hand, and is a strictly better comparison. It does not remove the grounding call.

The 147 cross-namespace ones (`AMPK / "AMP-activated protein kinase" → MESH:D055372`) are
the same "same entity, different namespace" artifact that produced the misleading 72%.
db_refs cannot fix them; they need an ontology xref.

### 2.5 What it would cost, and what it does not buy

**It does not remove gilda from the image.** [M] 35,946 of 66,722 slots (53.87%) have a
`raw_text` differing from the agent name, so [V] `GroundedEntity._verify_raw_text` fires on
over half of all slots, and it is irreducible.

**Two independent proofs that the corpus cannot substitute for that call.**

*(a) The reader's own `raw_grounding` is not a second opinion.* The corpus does carry
`evidence.annotations.agents.raw_grounding` on 33,361 / 33,361 pairs, 66,722 non-TEXT
entries — so a "db_refs-native verification" looks available. [M] It is not:

```
raw_grounding IDENTICAL to db_refs    62134   93.12%
consistent (shared namespace agrees)   4579
CONFLICT (shared namespace, different id)  9   0.013%
disjoint                                   0
```

All 9 conflicts are secondary-namespace only (NCIT gene-vs-protein concept ids, one
UniProt isoform); HGNC is identical in all 9. Both sides come from the same reader, so the
comparison *structurally cannot* detect a reader collision — where the reader mis-resolved
a span, `raw_grounding` mis-resolved it the same way. Gilda's independent re-grounding of
the string is the only thing that is not the reader agreeing with itself.

*(b) db_refs repairs zero UNRESOLVABLE verdicts.* [M] The cause split for UNRESOLVABLE is
3,619 weighted on the raw_text side, 0 on the claim side. The claim side never fails, so
there is nothing there for db_refs to fix.

**Dropping gilda entirely is a capability cut, not a packaging change.** Turning off the
raw_text verification also deletes the [M] 5.00% of slots (3,334 / 66,722) carrying a
grounding signal and the whole tier-1 auto-reject path — [V] the deterministic branch in
`indra_belief.scorers.monolithic.scorer.score` that answers with **zero provider calls**.
That deserves its own gate and its own accuracy read.

**Generalization beyond this corpus.** [M] Across 7 corpora, grounded slots carrying a
non-TEXT db_ref = 98.7–100%; carrying HGNC or FPLX specifically = 85.9–100% (paper 100%,
`eval_curation_v1` 100%, `probe_relation` 100%, `rasmachine_v2` 93.9%, `rasmachine_v1`
92.1%, `rasmachine_subset` 89.1%, `external_curator_gold_v1` 85.9%, with 12 of 930 slots
carrying no identity ref at all). So a db_refs-first policy needs a name-grounding
fallback; it is 0% of this corpus and ~1% of the worst one, but it is not zero.

### 2.6 Decided versus open

**Decided, as a matter of what the measurements support:**

* The identity half of grounding is redundant with data already on the Statement, and
  wrong on 0.376% of slot observations. Fixing it is worth doing.
* It is an **input-correctness** fix, not a metric win. 251 of 66,722 slot observations
  cannot move a headline error-F1 above this project's own measured run-to-run noise. It
  should be validated as a **diff** — assert the 13 entities' `entity_context` changes and
  that the other 66,471 render byte-identically — not as a scoring run against a
  provider.
* It must land as a **new registered variant**, never as a mutation of the default (§4).

**Not decided:** whether gilda leaves the serving image at all. §2.5 says the raw_text
verification is irreducible on this corpus; §6 lists the one query that could change that.

---

## 3. Latency

### 3.1 The measured table

[M] From the real attempts logs — all four verdict-only arms
(`data/comparison_verdict_only/runs/*/attempts.jsonl`) and all three reasoning arms
(`data/comparison/runs/{gemma_26b,gemma_31b,glm_5}_primary/attempts.jsonl`; the 26b one is
a 2.1 GB file, mtime Jul 22 03:13). Percentile by index over every row, both cuts stated
because they are not the same quantity:

| arm | model | call p50 | call p90 | call p99 | attempt p50 | attempt p90 | attempt p99 |
|---|---|---|---|---|---|---|---|
| verdict-only | e2b | 0.767 | 1.318 | 2.733 | 0.788 | 1.347 | 2.766 |
| verdict-only | gemma-26b | 0.698 | 1.374 | 5.222 | 0.722 | 1.365 | 5.201 |
| verdict-only | gemma-31b | 0.945 | 1.962 | 26.165 | 0.952 | 1.946 | 24.096 |
| verdict-only | glm-5 | 0.851 | 2.512 | 12.591 | 0.862 | 2.451 | 12.476 |
| reasoning | gemma-26b | 4.080 | 12.267 | 39.772 | 6.572 | 17.200 | 50.271 |
| reasoning | gemma-31b | 3.782 | 12.984 | 83.931 | 6.035 | 19.498 | 105.228 |
| reasoning | glm-5 | 6.618 | 21.500 | 63.869 | 10.757 | 28.865 | 81.708 |

All in seconds. **A call is not a request on the reasoning arms, and this table used to
conflate them.** [M] The verdict-only arms issue 0.97 calls per attempt, so there the two
cuts nearly coincide and the difference — ~0.024 s — is client-side work: JSON build,
parse, digesting, ledger write. The reasoning arms issue **1.49** (50,484 calls over 33,904
attempts for gemma-26b; the relation sub-call of Correction 3, on ~51.7% of pairs), so
their per-*request* p50 is the ATTEMPT column: 6.57 s for gemma-26b and 10.76 s for glm-5,
not 4.08 and 6.62. Read §5.1's synchronous-request argument off the attempt column.

Superseded numbers, recorded so a reader who remembers them knows they were withdrawn: an
earlier five-row version of this table stated e2b p99 = 2.257 (contradicting §3.3's 2.73,
which is the correct one) and a p90 column — e2b 1.246, gemma-26b 1.828, glm-5 2.307 — that
reproduces neither cut. It named no statistic and no row filter. This one does.

### 3.2 The decomposition: the call is network-bound

The provider stamps its own clock. [V] Every call-log entry carries a byte-exact provider
response preimage; decoding it yields `created_at` / `completed_at` and the real `usage`
block including `input_tokens_details.cached_tokens`. [R] The stamps are 1-second
granular, but for a uniformly-phased sub-second interval the expected floor difference
equals the true mean exactly, and with 8 async workers and no second-alignment the phase is
uniform — so `mean(completed_at − created_at)` is an unbiased estimate of provider-side
elapsed time and `duration_s` minus that is network plus client.

[M] gemma-26b verdict-only, n = 28,550 calls with duration < 1.2 s, HTTP 200:
mean wall 0.7201 s, mean provider-side 0.2523 s (±0.0026, 1 s.e.), **non-provider 0.4679 s
= 65.0% of the call.** The non-provider share is model-independent to ±1.5% across a ~15×
model-size range — 0.4679 (26b), 0.4747 (e2b), 0.4719 (31b). That is what a network
constant looks like.

[M] Measured against the live endpoint with the production connection class, interleaved
A/B, n = 25: raw TCP connect = 1 RTT = 0.2948 s p50; full `connect()` (DNS + TCP + TLS)
= 0.5972 s = 2.02 RTT; a complete fresh-connection cycle to a 401, which reaches no model,
= 1.1923 s = 4.04 RTT; keep-alive request+read = 0.6299 s = 2.14 RTT. **`connect()` is
50.09% of the fresh cycle.**

The budget, which sums to the observed mean:

```
TCP+TLS handshake, fresh connection per call   0.2343 s   32.5%
wire legs (request + response)                 0.2335 s   32.4%
provider residual (queue / schedule / serialize)
                                               0.1685 s   23.4%   [M by subtraction]
decode, 14 output tokens @ 5.50 ms/tok         0.0770 s   10.7%
prefill, ~110 uncached tokens @ 61.9 µs/tok    0.0068 s    0.9%
```

[M] Decode slope fitted on the reasoning run, where output tokens span 12 → 3200:
p10 0.00550 s/tok = 181.7 tok/s, p50 0.00763 s/tok = 131.0 tok/s. [M] Prefill slope fitted
on uncached-prompt-token bins: 26b p50 61.9 µs/tok, 31b p10 49 µs/tok — 16k–20k tok/s.
[R] Implied run-day round trip ≈ 0.12 s, San Francisco to us-east-1. Do not read more
digits into that than it has.

### 3.3 Prefill is not a latency target

[M] The provider's prefix cache is already on and hitting: median
`cached_tokens / prompt_tokens` = 0.950 (26b), 0.961 (e2b), 0.929 (31b), 0.902 (glm-5).
Median uncached prompt is 110 tokens (26b), 85 (e2b), 202 (glm-5). Measured
prefill : decode = 2,204.6 / 14.23 = 155.0 : 1 over the whole 26b verdict-only arm; the
154.5 : 1 owned by `research/serving_architecture.md` §9.1 is the same regime, measured
over a different slice — one artifact at two slices, not two independent routes.

So the 2,200-token prompt costs 0.007 s, because 95% of it is already resident, and
decoding 14 tokens costs 11× more than prefilling the whole prompt. **The prefill:decode
ratio is a cost story, not a latency story.**

What the cache does buy in latency is tail control. [M] Fully-uncached call rate ranks
e2b 0.13% < 26b 0.37% < 31b 5.50%, and p99 duration ranks 2.73 s < 5.21 s < 24.62 s in the
same order. Within the 26b arm, p90 in the 0–100-uncached-token bin is 1.172 s against
3.233 s in the 1600–2000 bin. **The tail is a cache-eviction tail, not a decode tail.** If
someone wants the tail, target cache retention, not prefill throughput.

### 3.4 Grounding cost on the host

This binds a hypothetical `/score` API only. [M] The batch replays grounding from a file —
`data/comparison_verdict_only/grounding_replay/manifest.json` reports `entity_names 1644`
and `executions 33413`, and [V] `src/indra_belief/comparison/` references gilda nowhere,
not even lazily.

[M] Cold process, fresh interpreter, stage-attributed:

```
baseline                                            rss    22 MB
import gilda                          2.317 s       rss   211 MB
FIRST gilda.ground (builds the index) 10.872 s      rss 1,966 MB   2,045,851 term entries
5 further gilda.ground                 0.001 s      (~0.2 ms each)
first  gilda.get_names(HGNC, …)        0.146 s
4 further gilda.get_names              0.573 s      (~143 ms each — CONSTANT, not amortizing)
first FPLX member lookup (bio_ontology) 7.517 s     rss 7,479 MB   (7.23 s is a pickle load)
second FPLX member lookup              0.000 s
```

Independent runs of the same cold sequence, on different page-cache states, produced
`import gilda` 5.71 s + first ground 11.95 s with a 19.72 s ontology load, and separately
8.10 s + 6.53 s. [R] The honest statement is a **range**: a cold worker pays somewhere
between roughly fifteen and forty seconds and multiple gigabytes of RSS before its first
fully-resolved HGNC+FPLX entity. Do not quote a single number for this; it is dominated by
whether the resource files are in page cache.

[M] Steady state, warm process, at the exact live-path call shape over 300 corpus agent
slots with real `raw_text`: mean 115.7 ms, p50 111.5 ms, p90 138.4 ms, p99 324.0 ms — about
231 ms per record on a cold LRU. [M] With the LRU warm on the same 300: 0.0137 ms per call,
an 8,440× speedup. Caches are process-local [V] `functools.lru_cache` at
`_cached_ground` / `_cached_get_names` in `src/indra_belief/data/entity.py`, so workers are
meaningfully stateful.

**The 116 ms is not grounding and is not gilda's fault.** [M] `gilda.ground` alone is
0.04 ms p50 over the same 300 calls. cProfile attributes 62% of total resolve time to
gilda's `get_names`: 12.376 s of tottime over 88 calls, ~141 ms each. [V] Upstream,
`get_names` is a full linear scan over all 2,045,851 entries, per call, to recover the
aliases for one `(db, id)` — and [V] `GroundedEntity.resolve` hits it for every HGNC
entity, which was 222 of 300 in a representative agent-slot sample.

[M] The fix is a one-time reverse index: building `(db, id) → names` over gilda's entries
takes 3.29 s and yields 724,330 keys, after which lookup is 0.0010 ms — same answer, five
orders of magnitude faster. [R] With that in place, resolve drops to roughly 0.2 ms per
entity and the cold cost becomes a warm-up of order ten seconds and a couple of gigabytes,
which is a process-lifecycle constraint rather than a per-request one.

[M] One further defect inside our own code: `entity_grounding` in
`src/indra_belief/tools/gilda_tools.py` has **no cache at all** — verified by grep, the
file contains no `lru_cache` or `cache` decorator — and is called twice per `Complex` by
the relation step. Measured 133.6 ms immediately after `GroundedEntity.resolve` of the same
name and 155.4 ms on immediate repeat. At [M] 51.7% of pairs being `Complex` (Correction 3),
roughly half of requests would pay ~0.26 s of work that never becomes a cache hit. This is
the cheapest available fix and it is entirely in our code.

### 3.5 The floor

**(a) Remote provider as it stands today** [M-derived, on a mean basis]:

```
today                                             0.720 s
+ connection reuse (removes 2.02 of 4.04 RTT)     0.486 s   (−32.5%)
+ drop "confidence" from the JSON (14 → 8 tokens) 0.453 s
client colocated in us-east-1, 14-token reply     0.256 s   (−64% vs today, NOT stacked
                                                            on the row above)
provider-only floor, all network removed          0.252 s
```

[R] About a quarter of a second is the hard remote floor without changing model or
provider: the provider residual plus decode plus prefill. Nothing client-side touches the
residual. Cross-check: observed p01 is 0.571 s, and subtracting four round trips leaves
roughly a tenth of a second — the same order.

**(b) Self-hosted** — [R], anchored on a single [M] datapoint and stated as a range.
[M] The only real self-hosted measurement available is
`data/results/rasmachine_mono_medpsy_remote_direct.jsonl`, n = 46,837 calls, MedPsy-4B Q8_0
on a gfx1100 host under llama.cpp HIP with 4 workers: decode slope p10 0.01029 s/tok
= 97.2 tok/s, intercept at zero output tokens 0.368 s, which is a full uncached
2,096-token prefill plus LAN plus server overhead. That stack had **no prefix cache**.
[R] Rebuilt at the verdict-only shape with a 95% prefix hit, a 4B-class model lands
somewhere around 0.16–0.20 s per call; a 26B-A4B-class model, roughly 0.2–0.5 s — a range,
not a number.

**Stated plainly: this is unverifiable in this workspace.** There is no GPU here, no vLLM,
SGLang or llama.cpp, and no serving stack. Prefix-cache hit latency, continuous-batching
scheduling, KV eviction and 26B decode rate were **not measured** on any local device. The
97.2 tok/s is a June artifact from a different host, a different model, and a
non-prefix-cached stack.

[R] The consequence for the decision: self-hosting does not beat
remote-with-keep-alive-and-colocation on p50, and it buys its number by giving up the
26B-class model. Self-hosting's case is throughput and cost, not latency.

### 3.6 The single biggest lever: HTTP connection reuse

[M] Size: 0.2343 s per call = 32.5% of the mean call, and 50.07% of all off-model time
(the 50.09% in §3.2 is a different ratio — `connect()` over the fresh cycle).

[V] The one remaining raw transport, `src/indra_belief/bedrock_responses_transport.py`,
constructs a connection via `_connection_factory` inside `call()` and closes it on the way
out. [V] `build_pinned_https_opener` lives in
`src/indra_belief/bedrock_transport_base.py` and is re-exported by
`src/indra_belief/bedrock_responses_transport.py`; it uses `urllib.request.build_opener` —
no pooling, no keep-alive, no HTTP/2. Every call pays DNS, TCP handshake and TLS handshake.
[V] `_DeadlineConnectionMixin.connect` in `src/indra_belief/bedrock_transport_base.py`
additionally spawns a bounded `getaddrinfo` helper thread per connect.

[M] Aggregate over one arm: 32,310 calls × 0.2343 s = 7,573 thread-seconds; at 8 workers
that is about 16 minutes of wall clock per arm, roughly an hour across the four-arm fleet.

**It is digest-neutral, which is what makes it safe under §4.** [V] `provider_request_sha256`,
`provider_wire_request_sha256` and the request-body digest all cover request **bytes**
(verified identical in a live record: `505bddb264779664…`). Socket lifecycle appears in
none of them. Keep-alive reproduces every published substrate digest byte for byte and
needs **no** new `ScoringVariant`.

[R] Scope it as a pooled connection per worker with a reconnect-on-error path, keeping the
absolute-deadline and abort-event semantics — those are the only reason
`_DeadlineHTTPSConnection` exists at all.

Runner-up, bigger but infrastructure rather than code: colocating the client in us-east-1
removes the whole non-provider share, ~65%.

---

## 4. The reproducibility constraint

**Read this before touching grounding.** The entity context is part of the user message,
so a grounding change is a **prompt change**, and a prompt change is a new substrate.

[V] The chain: `ScoringRecord.format_entity_context` emits an `Entities: …` line plus
warnings into the user message; that string becomes `ExecutionBody.entity_context`;
[V] `ExecutionBody.render` joins the five parts; [V] `PreparedCall.prompt_sha256` digests
`{system, messages}`; [V] `assert_replay_digests` in
`src/indra_belief/prepared_execution.py` requires the hydrated request to reproduce the
frozen `main_prompt_base_sha256`. Change what grounding writes and every one of those
digests moves.

[V] `ExecutionBody`'s own docstring records that every frozen digest in
`data/comparison/grounding_replay/manifest.json` depends on that join.

**The mechanism for changing it safely is the variant registry, not a default mutation.**
[V] `src/indra_belief/scorers/monolithic/scorer.py` holds `VARIANTS`, `variant_from_env`
and `DEFAULT_VARIANT` (resolved once at import from `MONO_VARIANT` and threaded as a
value). [M] The registry currently holds four entries — `""`, `disconfirm`,
`disconfirm_relnature`, `disconfirm_relnature_rf` — and `DEFAULT_VARIANT_NAME` is
`disconfirm_relnature_rf`.

**A db_refs-first policy is not a drop-in registration. This is the part to plan for.**

1. [V] `ScoringVariant` is a frozen dataclass with fields `name`, `system_prompt`,
   `render_example`, `structured`, `resolve_relation_nature`. **There is no grounding
   field.**
2. [V] `indra_belief.scorers.monolithic.scorer.score_statement` builds
   `ScoringRecord(statement=…, evidence=…)` — it passes no variant — and
   [V] `ScoringRecord.__post_init__` calls `ScoringRecord.resolve_entities`
   unconditionally. **The variant value never reaches grounding.** This is a new seam, not
   a registry entry, and it is the real work.

The order that keeps published digests intact:

1. Add a grounding-policy field to `ScoringVariant`, defaulting to the current name-first
   callable. All four existing entries keep the default, so their prompts stay
   byte-identical.
2. Thread it to record construction — a new optional field on `ScoringRecord`, defaulting
   to today's behaviour.
3. Register the new variant with the *same* system prompt and renderer as the default;
   `DEFAULT_VARIANT_NAME` does not move. [R] Its `entity_context` differs on the 251
   affected slot observations, so `prompt_sha256` differs on those rows and only those.
4. [V] `src/indra_belief/calibration_constants.py` keys calibration profiles on
   `f"{reader_model}@prompt-sha256:{prompt_sha256}"`. A new variant needs its own profile
   or it falls through to no profile at all.

**A guard warning that would otherwise be discovered the hard way.** [V]
`tests/test_prepared_execution_goldens.py` (since retired with the paid comparison
harness, along with its goldens fixture) monkeypatches `ScoringRecord.resolve_entities`
to a no-op and injects stored rows from `data/comparison/grounding_replay/entities.jsonl`
([M] 6,479 rows, mtime Jul 20 02:21). **The goldens will not catch a regression in the
default grounding path** — they deliberately do not run it. The test that would is
`tests/test_grounding_collision.py`, which calls `GroundedEntity.resolve` live.

---

## 5. What a serving interface would look like

Nothing in this section is built. It records what the code already constrains.

### 5.1 Entry points and the request unit

[V] The public surface is two names in
`src/indra_belief/scorers/monolithic/__init__.py`: `score_evidence(statement, evidence,
client, *, variant=None)` and `score_statement(statement, client)`. The second accepts
neither a variant nor a token limit, so it cannot express the policy the first can.

[V] The real kernel is `indra_belief.scorers.monolithic.scorer.score(client, record, …)`;
the module's own `score_statement` is a two-line adapter over `ScoringRecord` plus `score`.

**Import trap.** [M] `src/indra_belief/scorers/__init__.py` is **0 bytes**, despite the
monolithic package's docstring claiming "the same public shape as
`indra_belief.scorers`". [V] `src/indra_belief/scorers/scorer.py` re-exports the
**decomposed** architecture at module import, and monolithic is reached only through its
resolver. A serving layer must import `indra_belief.scorers.monolithic` explicitly or it
will silently get the non-default architecture.

**The unit is the (Statement, Evidence) pair, not the Statement.** [M] The paper corpus is
1,689 statements over 33,361 evidences — mean 19.75 evidences per statement, max 759, and
only 20.0% single-evidence.

**Correction — the mean is not the median, and the difference decides the API.** [M] The
distribution is heavily right-skewed: median **7**, p75 17, p90 38, p95 73, p99 261, max
759. An earlier [R] sentence here reused the mean 19.75 as "about twenty provider calls at
the median"; that is the MEAN. The median statement carries 7 evidences, so with the
measured second-call rate below it a median statement is order-of-ten calls, not twenty.
Recomputed directly from `data/corpora/indra_paper_unique_pairs_20260717_statements.json`
(n=1,689, 33,361 evidences — reproduces this section's own totals).

[M] Because the mass sits in the tail, a bound on evidence count buys most STATEMENTS
cheaply and most EVIDENCE not at all:

| cap | statements served | evidences served |
|---|---:|---:|
| ≤4  | 41.6% | 4.2% |
| ≤8  | 57.1% | 9.2% |
| ≤16 | 74.1% | 19.3% |
| ≤32 | 88.3% | 35.7% |
| ≤64 | 94.4% | 49.5% |

So the original conclusion holds for the CORPUS and is too strong for the TYPICAL REQUEST:
an unbounded statement-grained synchronous call is indefensible, but a BOUNDED one serves
the majority of statements in-band. A serving layer must therefore admit on a declared cap
rather than refuse the grain outright. And per Correction 3, [M] ~51.7% of pairs issue
**two** provider calls under the default variant, so "one request = one model call" is
false for about half of traffic, and any cap must be enforced against the two-call upper
bound rather than the `estimate_calls()` floor.

### 5.2 Minimum payload and response

[V] Exactly these attributes are read by `ScoringRecord`: statement type name; `agent_list`
entries' `name`, `activity`, `mutations`, `bound_conditions`; `residue`/`position`;
`Complex.members`; translocation endpoints; `evidence.text`;
`evidence.annotations["agents"]["raw_text"]`; `annotations["found_by"]` and `source_api`
(only inside `ScoringRecord.format_provenance`, and only when a flag fires); and the
evidence source hash, for identity only. Everything else on an INDRA Statement — db_refs
(today), matches_hash, epistemics, source_id, pmid, text_refs — is unread.

[M] Dead accepted surface, by repo-wide grep: `ScoringRecord.is_direct`,
`ScoringRecord.raw_grounding` and `ScoringRecord.pmid` are defined and have zero consumers
anywhere. Do not accept them and do not document them.

[M] What goes on the wire, over 12 real records: canonical `{system, messages}` is a median
15,803 bytes (min 15,298 / max 16,057) — system 4,650 B + few-shot prefix 10,707 B +
per-record body 453 B. **97.1% of every request is static**, and per Correction 4 the
prefix space is 9 values on this corpus.

[M] The returned dict has 11 keys: `score, verdict, confidence, raw_text, tokens, tier,
grounding_status, provenance_triggered, selected_example_ids, selected_examples, call_log`.
[V] `score` is a calibrated sentence probability or `None`; an unparseable verdict,
unsupported calibration profile, or failed probe never fabricates a categorical
midpoint. [M] `json.dumps(call_log)` is 19,675 bytes against 2,024
for everything else: **91% of the response is the call log**, which embeds the full system
prompt and all 28 few-shot messages verbatim. [V] The model's own justification is not
top-level — `_stamp_committed_justification` writes `support`/`objection` into the last
call-log entry — so a consumer wanting it must walk a 20 KB blob.

[M] Route distribution over the first 250 statements, first evidence each: plain 84.4%,
tool 12.8%, tier-1 reject 2.4%, no-text 0.4%. So **2.8% of requests are answered
deterministically with zero provider calls and zero tokens** — a real free fast path, and
the thing §2.5 warns would be deleted by dropping gilda.

### 5.3 Which batch guarantees carry over

**Meaningless per request — do not port.**

* **Resume.** [V] The runner's recovery rebuilds an interrupted action from stored rows
  keyed by execution and attempt ordinal; the CLI's `--resume` skips by source hash in an
  output file. A request has no partial-run state: the client either got a response or
  retries.
* **Quarantine.** [V] The diagnostic budget in `src/indra_belief/comparison/runner.py`
  exists to decide whether an *arm* should keep spending across a long workload. A single
  request has no remainder to protect; per request the equivalent is an error status.
* **Ledger replay / substrate-digest reproduction.** [V] `assert_replay_digests` checks a
  hydrated prompt against a frozen digest. A live request has no frozen substrate — it
  *builds* the prompt. `PreparedCall.prompt_sha256` still computes a digest; that is
  attribution, not replay.
* **`eligible_position`.** [V] `execution_identity` in
  `src/indra_belief/comparison/replay.py` includes the pair's index in a workload
  ordering. That has no referent outside a workload and must not appear in a request
  identity.

**Not meaningless — keep.**

* **Pre-flight spend caps.** [V] `SpendGuard.reserve_call` refuses *before* the call when
  the committed spend plus this call's bound would exceed the approved or stage cap, using
  a conservative prompt-token bound plus output overshoot at the model's price. [M] The
  full three-event attempt lifecycle — hash-chained and `fsync`ed — costs 0.28 ms, a
  ceiling of roughly 3,500 executions/s, i.e. 0.04% of a 0.73 s request. The hypothesis
  that fsync would be the bottleneck is **refuted by measurement.**
* **Prompt and model attribution.** [V] `PreparedCall.prompt_sha256`,
  `PreparedExecution.profile_name` (the variant name), the constant
  `prepared_execution.PARSER_ID`, and `provider_request_sha256` /
  `provider_wire_request_sha256` binding the ledger row to the exact wire bytes. The
  caveat this bullet used to carry — that the two name fields were write-only, so they
  were a field that WOULD carry the attribution rather than attribution anything checked
  — is discharged: `research/kernel_unification_findings.md` §7.2 item 5 records that the
  parser field was deleted for being structurally constant, and that the profile name is
  now read by `tests/test_prepared_execution_parity.py`, which is what makes the two
  producers agree on which profile they built. `prompt_sha256` is read by
  `assert_replay_digests` and remains the mechanism attribution actually rests on.
* **No double spend under retry — already exists.** [V] `SpendGuard.attempt` derives an
  execution id by hashing model, workload mode and the caller's identity mapping, and
  `SpendGuard._start_attempt` raises `AttemptLimitReached` on a replayed completed
  identity. [M] Confirmed by running it. That *is* server-side idempotency. [R] The
  consequence to design around: completion is permanent, so a caller who legitimately
  wants a fresh score on the same evidence is refused forever — which is why a request
  identity would need a caller-supplied idempotency key in place of `eligible_position`.
* **Exclusivity is a scaling limit.** [M] A second guard on the same ledger path raises
  `SpendLedgerInUse` — [V] `acquire_spend_lane_lock` takes `flock(LOCK_EX|LOCK_NB)`. **One
  process per ledger.** N worker processes need N ledgers, and then the cap is per worker.
  **A fleet-wide spend cap does not exist.**

### 5.4 The shape the code implies

Three endpoints, and the removals matter as much as the additions.

1. `POST /v1/score` — one (Statement, Evidence) pair, synchronous, parsed by the same
   INDRA JSON loader [V] `CorpusIndex` already uses. Response is the verdict fields with
   `support`/`objection` promoted to top level, plus an attribution block
   (`profile_id`, parser id, `prompt_sha256`, model, execution id, call topology) and a
   trace id.
2. `POST /v1/score:batch` — an array of pairs, per-element status. [M] Its justification is
   measured, not aesthetic: grounding cost is per *new* entity, so a batch sharing entities
   is dramatically cheaper than N separate requests (250.9 ms per record first pass against
   0.04 ms per record on a repeat pass over the same pairs). Bound N by the spend guard's
   admission check, not by a magic number.
3. `GET /v1/variants` — the manifest: variant names, the default, the 9 prefix digests,
   per-variant system-prompt digest, the verdict score grid. This is what makes a served
   score interpretable without shipping 15 KB of prompt in every response.

Removals, each measured: `call_log` out of the body ([M] 19,675 of 21,699 bytes) behind a
trace endpoint; `selected_examples` out of the body, keeping the ids ([M] 14 rows per
response for a set that is a pure function of variant and statement type); the three dead
accessors dropped from the accepted surface; `eligible_position` dropped from request
identity; one architecture only, imported explicitly; and `score_statement(statement,
client)` is not a serving verb.

**Backpressure bounded by the two real limits.** [V] `concurrency_hint` in
`src/indra_belief/model_client.py` already exists and returns a per-lane width — admission
is a semaphore of that width per model. Spend refusal is the 429, naming the cap that would
be breached. [V] The runner ratchets the client timeout down to the remaining budget;
per request, do the same. **Critically** [V]: the billed Bedrock lanes run synchronously on
purpose, because a timeout cannot cancel a live paid request. A server must not model
cancellation as free.

**Worker model.** Warm gilda and the ontology at boot behind a readiness probe (§3.4 —
tens of seconds, page-cache dependent); otherwise request #1 pays it. Caches are
process-local, so workers are stateful and meaningfully warm. One spend guard per process,
so caps are per worker.

**Two images, one kernel.** [V] The batch `Dockerfile` says of itself that the image cannot
build a substrate and cannot score raw text. Serving raw Statements needs indra + gilda
plus their resources — ~724 MB all told once re-weighed in §7.1 (~531 MB of it resources
on a volume), against the Dockerfile header's ~1.44 GB. But [M] importing the monolithic
scorer loads neither, so a
**prepared-payload** serving mode — the consumer sends the already-rendered body parts
instead of a Statement — runs in the 535 MB image with zero grounding dependencies, and
[V] `prepare_from_replay_row` in `src/indra_belief/prepared_execution.py` is already
exactly that entry point. It is a genuinely new capability that arrives as a *removal*.
Its cost is that attribution then covers the prompt we were handed, not the prompt we
built.

---

## 6. Open questions, and what would settle each

1. **Does CoGEx-at-large carry `raw_text`?** [M] The verification path fires on 53.87% of
   slots here, which is the entire reason gilda cannot leave the image. A preliminary look
   at the corpus-representative CoGEx sample found none. If that reflects the graph rather
   than the export, then on the corpus that actually represents 44.9M CoGEx evidences the
   whole raw_text-verification machinery is dead code, grounding reduces to identity plus
   enrichment, and db_refs plus a 6.2 MB table serve it completely with no gilda. **Settled
   by:** one live query against CoGEx for `annotations.agents.raw_text` presence on a
   representative sample. This is the highest-value open item in the document, because it
   flips "db_refs-first is a variant" into "db_refs-first is the only policy CoGEx can
   support."

2. **Should `is_family` come from db_refs even inside the default variant?** [M] 244 of the
   251 affected slot observations are family-flag flips, and this is the one field where
   gilda is not merely redundant but wrong *directionally* — JUN, FOS and PRC1
   over-generalize, CALM, CAV, DNM, MYL, PDC and PLA2 under-generalize. §4 says no default
   mutation, and that governs. But this is the case a reader will push on, so it should be
   answered deliberately rather than by omission. **Settled by:** an explicit decision
   record — either "the published digest is worth 244 wrong family flags" or "we cut a new
   default and re-baseline", not silence.

3. **Where does the id → names table live, and how is it versioned?** [M] 6.2 MB, derived
   from a specific gilda release. If it ships as a data file it must be content-addressed
   like everything else, or the variant is non-reproducible for exactly the reason the
   registry exists. **Settled by:** picking a home and a digest before any code lands.

4. **Is the competing-candidates band still calibrated against a ref *set*?** [V] The
   competing-candidate computation in `src/indra_belief/data/entity.py` is relative to a
   single claim `(db, id)` pair within `_COMPETE_BAND`. Widening the claim side to the whole
   db_refs set changes which candidates count as competing, so the 137 repairs and the 93
   real-collision candidates in §2.4 are **not** independent of that constant. **Settled
   by:** re-running the mismatch classification with the set-comparison in place before
   quoting either number as a result.

5. **Does the gateway's front-door time land inside the provider's own timestamps?** The
   401 probe, which never reaches a model, took 2.14 RTT on a keep-alive connection. If a
   meaningful slice of that is gateway auth and routing rather than wire, the "wire legs"
   line in §3.2 is overstated and "provider residual" understated by the same amount. It
   does **not** change the handshake line or the lever. **Settled by:** a probe that
   separates gateway-terminated from runtime-terminated responses.

6. **Why is the request/response leg 2 RTT and not 1**, on a warm keep-alive connection
   with `TCP_NODELAY` confirmed set? A second hop inside AWS is the obvious guess, and it
   was not confirmed. If it is real, colocating in us-east-1 removes less than §3.5
   projects. **Settled by:** a traceroute-equivalent or a same-region probe.

7. **What drives the 31b arm's cache pressure?** [M] Same corpus, same workers, same day:
   the 31b arm shows a 5.50% fully-uncached rate and a p10 hit rate of 0.405, while e2b
   shows 0.13% and 0.916. Whether that is provider-side cache pressure, per-model cache
   sizing, or an artifact of the 31b arm's longer span is unresolved, and it is the single
   largest driver of the 24.6 s p99. **Settled by:** a controlled re-run with matched span
   and interleaved arms.

8. **Pair belief or statement belief?** `src/indra_belief/statement_belief.py` exists and
   aggregates per-evidence verdicts into a statement scalar. The kernel serves pairs.
   Whether consumers want the aggregate is a product question the code cannot answer.

9. **Is a per-worker spend cap acceptable?** [M] One process per ledger is enforced by
   `flock`. A fleet-wide cap would be genuinely new work — a cap service, not a file.
   **Settled by:** knowing whether a bevy of consumers shares one budget or many.

10. **Is one verdict per (evidence, variant, model) the intended contract?** The ledger
    enforces it permanently today. If re-scoring must be allowed, a caller-supplied
    idempotency key is required; if not, the key should be *omitted* so the content address
    alone is the identity. This is a product decision, not a code fact.

---

## 7. The live image, and the re-weighed cost

§1 carried the Dockerfile's ~1.44 GB saving forward without re-weighing it — re-weighed
in §7.1 below it is ~724 MB, split ~531 MB resources-on-volume and ~193 MB
packages-in-image — and §5.4's
"two images, one kernel" named a grounding-capable image as a thing that did not exist.
Both are now settled by building it. `Dockerfile.live` exists at the repo root; it is a
scorer CLI, not a server, because §1's finding that no serving process exists still holds
and an image implying HTTP would be the same false claim in a new place.

### 7.1 The ~1.44 GB saving re-weighs to ~724 MB, and the reason is instructive

[M] Re-weighed against the versions actually installed here — gilda 1.6.1, indra 1.24.0.
The two quantities are kept apart on purpose, because they live in different places and
only one of them is ever in a layer (`stat -f %z` on this host for the byte figures;
in-container `du` for the package figures):

```
RESOURCES ON THE VOLUME (never in a layer)
  indra ontology cache, bio_ontology.pkl    492,535,810 B    ~470 MiB
  gilda resource dir (the WHOLE of it)       38,060,066 B     ~36 MiB   one file
                                            -------------
  subtotal                                  530,595,876 B    ~531 MB  (506 MiB)
  + gilda_models.json.gz, fetched lazily     10,530,040 B     ~10 MiB
                                            -------------
  subtotal if the models file is present    541,125,916 B    ~541 MB

PACKAGES IN THE IMAGE (never on the volume)
  indra installed, in-container                  190.4 MB    (143.5 MB is indra/resources)
  gilda installed, in-container                    2.7 MB
                                            -------------
  subtotal                                       193.1 MB

RE-WEIGHED TOTAL saving from omitting both      ~724 MB      (<= ~734 MB with the models
                                                              file present)
```

So the saving is **~724 MB, not ~1.44 GB** — of which ~531 MB is resources-on-volume and
~193 MB is packages-in-image. Neither half is the "roughly 700 MB" this section used to
report: that was a rounding of packages+resources carrying a resources-only label.

**Reconciled against the batch `Dockerfile`'s own decomposition** (read only, not edited —
its header is the baseline being corrected). It claims packages 187 + 2.5 = 189.5 MB and
resources 470 + 784 = 1,254 MB, total ~1,443.5 MB. [V] The package half reproduces within
~2%: 190.4 MB and 2.7 MB in-container, 186.5 MB / 2.5 MB in this repo's `.venv`. [M] The
resource half is wrong by ~2.4x (1,254 MB claimed against 531 MB measured), for the reason
below — the "784 MB" is a stale gilda version's directory, not the installed one's.

**Why the header drifted, which is the part worth keeping.** [M] The pystow root holds
THREE gilda version directories — 1.4.1 at 46 MB, 1.5.0 at 702 MB, 1.6.1 at 36 MB — and
the 687,099,904 B `grounding_terms.db` that supplies almost all of the "784 MB of resource
files" lives in the **1.5.0** directory. gilda 1.5.0 is not what is installed. [V]
`gilda.resources` computes `resource_dir = pystow.join('gilda', __version__)`, so the
version is part of the path and stale versions simply accumulate beside the live one; a
`du` over the parent counts all three. The lesson is narrower than "the number was wrong":
a version-scoped cache directory cannot be measured by looking at its parent.

[V] Two further facts about how those resources arrive, both read at the symbol:
`gilda.resources._download_from_s3` builds its client with
`botocore.client.Config(signature_version=botocore.UNSIGNED)` against the public `gilda`
bucket — **no credential is involved**, which is what makes an unauthenticated warm-up
possible — and `gilda.resources.get_gilda_models` fetches `gilda_models.json.gz` lazily,
so it is absent until something disambiguates.

### 7.2 What was built, and what its gate does and does not prove

[M] `docker build -f Dockerfile.live -t indra-belief-live:test .` exits 0 on this host.
Measured on the resulting image:

```
indra-belief-live:test    993,002,546 B   =  993 MB
indra-belief:batch        534,805,452 B   =  535 MB
```

[M] The +458 MB is packages, not data: `find / -name 'grounding_terms*' -o -name
bio_ontology.pkl` inside the image returns nothing. In `docker history` the site-packages
COPY is a single 848 MB layer and no other layer the Dockerfile adds exceeds 2 MB; the
remainder is the `python:3.13-slim` base itself. The corpus stays on `/app/data` and the
grounding resources on `/app/resources`, both declared volumes.

[M] The ownership/content measurement below uses `indra-belief-live:d2`. The
`indra-belief-live:test` tag named by the historical build line above has `/app/data` as
`root root`, whereas `:d2` has both roots owned by `belief`. [V] Because the current
`Dockerfile.live` mkdir+chown covers `/app/data`, that observed `:test` layout identifies
it as a pre-fix build, while `:d2`'s measured layout matches the current instructions; the
file creates `belief` as uid 10001 and runs as that user. [M]
On `:d2`, `/app/data` is empty with link count 2, while `/app/resources` has link count 4
and contains exactly two empty subdirectories, `indra/` and `pystow/`. Their complete
listings contain only `.` and `..`, so no resource DATA is seeded in the image's declared
data/resource roots. The image-wide `find` also printed no matching resource-artifact
path, although its recorded exit was 1 rather than a clean traversal. [V] `Dockerfile.live`
creates this layout with `mkdir -p /app/data
/app/resources/pystow /app/resources/indra`, points `INDRA_RESOURCES` at
`/app/resources/indra` and `PYSTOW_HOME` at `/app/resources/pystow`, and those two children
explain `/app/resources`'s link count 4. [M] The ownership was re-derived because until
that mkdir+chown was extended to cover `/app/data`, only `/app/resources` was owned by
`belief` and `/app/data` was `root root`:

```
docker run --rm --entrypoint sh indra-belief-live:d2 -c 'ls -ld /app/data /app/resources; ls -la /app/resources'
drwxr-xr-x 2 belief belief 4096 /app/data
drwxr-xr-x 4 belief belief 4096 /app/resources
total 16
drwxr-xr-x 4 belief belief 4096 .
drwxr-xr-x 1 root   root   4096 ..
drwxr-xr-x 2 belief belief 4096 indra
drwxr-xr-x 2 belief belief 4096 pystow

docker run --rm --entrypoint sh indra-belief-live:d2 -c 'ls -la /app/resources/indra /app/resources/pystow; ls -la /app/data'
/app/resources/indra:
total 8
drwxr-xr-x 2 belief belief 4096 .
drwxr-xr-x 4 belief belief 4096 ..
/app/resources/pystow:
total 8
drwxr-xr-x 2 belief belief 4096 .
drwxr-xr-x 4 belief belief 4096 ..
/app/data:
total 8
drwxr-xr-x 2 belief belief 4096 .
drwxr-xr-x 1 root   root   4096 ..

docker run --rm --entrypoint sh indra-belief-live:d2 -c "find / -name 'grounding_terms*' -o -name bio_ontology.pkl"
# no matching path printed; host-recorded exit 1

docker run --rm --entrypoint sh indra-belief-live:test -c 'ls -ld /app/data /app/resources'
drwxr-xr-x 2 root   root   4096 /app/data
drwxr-xr-x 4 belief belief 4096 /app/resources
```

**What that fixes, and what it does not.** [M] It fixes the ownership the image SEEDS into
a named or an anonymous volume: `docker run --rm -v /app/data --entrypoint sh <tag> -c
'test -w /app/data'` exited 1 before and exits 0 after, which is the case that was actually
broken — a bare `docker run` with no `-v` gets an anonymous volume. It does **not** reach a
bind-mounted host directory: a bind keeps the host's ownership and the image's chown never
applies to it. [M] On Docker Desktop here, a bind of a host directory appeared in-container
as `belief belief` and was writable, because its virtiofs layer remaps ownership — so on
this host the bind case does not fail either way. [R] On a Linux host, where bind ownership
passes through unremapped, a host directory not owned by uid 10001 is expected to be
unwritable and the operator must chown it; that expectation was reasoned, not run.

[M] The build gate runs `tests/test_prepared_execution_parity.py` inside the image —
**13 passed, 3 skipped**, the three being the `requires_substrate` cases, which gate on
`data/comparison/grounding_replay/manifest.json` and are correctly absent because
`.dockerignore` excludes `data/`. This is the exact complement of the batch gate, which
excludes that file by name and says it belongs "on a grounding-capable environment."

**What the gate does NOT prove, stated because it is the easy misreading.** [V] That
suite's record fixture replaces `ScoringRecord.resolve_entities` with a no-op for the
duration of the test. It therefore proves the live PRODUCER agrees with the batch producer
call-for-call with indra installed; it proves **package presence**, not grounding. [V] The
suite that exercises grounding is `tests/test_grounding_collision.py`, which
`pytest.importorskip`s gilda and calls the real `GroundedEntity.resolve` — it needs the
mounted resources and so belongs to a runtime verify path, not a build gate.

### 7.3 Three findings that only a build surfaces

**(a) The image cannot be built without a compiler, on arm64.** [M] On linux/arm64
`python:3.13-slim`, `pip install gilda indra` fails with `error: command 'gcc' failed: No
such file or directory` while building `adeft.score._score`. adeft is a hard gilda
dependency; pip resolves to an sdist here and compiles a Cython extension. [M] Fixed by
`build-essential` in the BUILDER stage only — the multi-stage copy keeps the toolchain out
of the runtime image, and a runtime-stage import of gilda/adeft/indra then succeeds with
no compiler present.

**(b) nltk refuses to import from a working directory of `/`.** [M] The first build failed
in the runtime stage with `ImportError: Blocked import of pydoc from current working
directory for security reasons`, raised out of `import gilda`. [V] nltk — reached via
adeft, reached via gilda — installs an import finder in `nltk/inisec.py` that resolves each
candidate spec's origin and tests `Path(origin).resolve().relative_to(cwd)` to decide
whether a module is being loaded out of the working directory. When the working directory
is `/`, that test succeeds for every path on the system, so the **stdlib's own** `pydoc` is
misread as a cwd import and refused. [M] cwd `/` fails, cwd `/build` succeeds, and the
`PYTHONSAFEPATH=1` the error text recommends does **not** help, because the finder compares
resolved paths and never consults `sys.path`. [R] Any container process that imports gilda
must therefore have a real working directory; `Dockerfile.live` satisfies this by ordering
its import check after `WORKDIR /app`, which is also what ENTRYPOINT and HEALTHCHECK
inherit.

**(c) Both resource roots relocate cleanly by environment variable.** [V]
`gilda.resources` resolves its directory through `pystow.join`, and pystow honours
`PYSTOW_HOME`; [V] `indra.ontology.bio.ontology.CACHE_DIR` is built from
`get_config('INDRA_RESOURCES')`, and [V] `indra.config.get_config` reads `os.environ`
before the config file. [M] Both verified by setting the variable and printing the
resulting path. That is what allows the resources to live on a volume rather than in a
layer.

### 7.4 The warm-up is an operator step, and it must be

[V] `BioOntology.initialize` has **no prebuilt-pickle download path**: it either loads
`CACHE_FILE` or calls `self._build()` and pickles the result, logging "this may take a few
minutes". The ~470 MB pickle is therefore CONSTRUCTED, not fetched. [R] Two consequences.
Building it during `docker build` would bake it into a layer, which is the design this
image exists to avoid; and building it in-container rather than copying the host's
sidesteps the cross-platform and cross-Python-version unpickle question entirely. So the
volume is populated by running the image once against a named volume with a warm-up that
calls `gilda.ground` and a `bio_ontology` child lookup — the invocation is written out in
`Dockerfile.live`'s operator notes. [R] A plain `docker run` with no `-v` receives an
anonymous volume and silently re-downloads and re-builds on every run.

[M] In `indra-belief-live:d2`, 24 serial runs of `python -c "import time;
t=time.perf_counter(); import gilda, indra.statements; print(time.perf_counter()-t)"`
measured three conditions, with `uptime` recording host load. Fresh-container runs used
`docker run --rm --entrypoint python indra-belief-live:d2 -c "import time;
t=time.perf_counter(); import gilda, indra.statements; print(time.perf_counter()-t)"`.
With no induced loaders (load average 8.49 → 8.46), they gave n=8, min/median/max
1.416/1.485/1.626 s. For the already-running-container condition, the host ran
`docker run -d --entrypoint sleep indra-belief-live:d2 600` and launched the same timer in
fresh Python processes by repeated `docker exec`; at load average 8.89 after, those gave
n=8, 1.422/1.583/1.708 s. [V] This matches the HEALTHCHECK's already-running-container
condition, although the timed command omits its directory test. [M] Fresh-container runs
under induced load from three competing import-loop containers (load average 9.26 → 9.86) gave
n=8, 1.641/1.834/4.911 s. Across all 24 samples the min/median/max were
1.416/1.594/4.911 s. [M] The former 2.2 s median and 1.7–2.6 s range reproduce in none of
those conditions: no-induced-load samples fall below its lower bound and loaded samples
exceed its upper bound. Dividing the batch file's 10 s timeout by each condition's slowest
sample gives 10/1.626 = 6.15× with no induced loaders, 10/1.708 = 5.86× in the
already-running container and 10/4.911 = 2.04× under induced load. [R] The single-attempt
margin is load-dependent and thin under induced load;
the configured `--retries=3` is what carries resilience there. Because the former n=9
measurement recorded no load condition, it cannot be reproduced or bounded by load — the
condition must travel with the latency number. (The earlier single `3.30 s` wall-clock
reading remains superseded; these 24 serial, condition-labelled samples now replace it.)
[V] `Dockerfile.live`'s HEALTHCHECK comment still carries the superseded 2.2 s median,
1.7–2.6 s range and roughly fourfold headroom text; this node may not edit that file, so
correcting the comment is owed to that file's owner.
Calling `gilda.ground()` there would not fit: §3.4 measures first-ground at 10.9-19.7 s and
~2 GB RSS.

### 7.5 What this does not settle

Nothing here is a serving process, and §1's finding stands unchanged. The image is not
wired into `docker-compose.yml` — that file is the batch image's operator surface and its
contract is correct as written, so the compose entry is a separate, operator-owned change.
No provider was called and no credential is in any layer: [M] the image's `Env` is PATH,
the three `python:3.13-slim` build vars, the three `PYTHON*`/`PIP_*` settings, and
`PYSTOW_HOME` / `INDRA_RESOURCES`. And grounding itself is still unproven in-image — see
§7.2 — because proving it needs a warmed volume, which is a runtime check, not a build one.

---

## What was not checked

Stated because absence of a finding is not a pass.

* The 535 MB image was confirmed present on this host, and the ~1.44 GB saving has now
  been **re-weighed** — see §7.1, which corrects it to ~724 MB (~531 MB resources on a
  volume plus ~193 MB packages in the image) and explains the drift. What is still *not*
  checked is whether the batch image rebuilds byte-identically;
  it was not rebuilt, only inspected.
* No self-hosted serving stack was measured (§3.5b) — no GPU, no vLLM/SGLang/llama.cpp in
  this workspace. Every self-hosted latency figure in this document is an extrapolation
  from one June artifact on different hardware with a different model and no prefix cache.
* The 2.1 GB attempts logs were not re-parsed for this document; the latency decomposition
  is carried forward from the extraction described in §3.2, whose method is stated so it
  can be re-run.
* No accuracy read exists for a db_refs-first variant, because no such variant exists. §2.6
  argues it should be validated as a byte-diff rather than a scoring run, which means an
  accuracy read may never be the right instrument for it.
