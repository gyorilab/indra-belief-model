# Kernel unification: what was fixed, what was refuted, what is still owed

A durable record of the refactor arc that landed as `a10df62` ("one semantic kernel
for the live and batch scoring paths", 44 files, +21,673/-901) on `kernel-unification`,
followed by `ae9ab1e` (batch image), `401ded6` (compose) and `a4852ef`
(`research/serving_deployment.md`). Base was `58767d8`.

This is not the design document. What the system *is* lives in
`research/serving_architecture.md` (the measured properties and the refactor plan) and
`research/serving_deployment.md` (what is deployed, latency, the grounding finding).
This file answers the other question: **why is the code like this, what was tried, what
was refuted, and what is still owed.**

Every claim is labelled: **[M]** measured, **[V]** verified against the tree at a named
symbol, **[R]** reasoned. Citations are by symbol, never by line number — the arc moved
almost everything, and line anchors written during it are already stale. Where a
finding described a state that no longer holds, it says so.

Verified 2026-08-03 against the post-landing tree unless noted.

---

## 1. What the arc changed

Before, the live scorer and the batch replayer each built their own request, ran their
own parser, and wrote their own score. Two implementations of one scientific
instrument. The arc collapsed the duplicated semantics into shared modules and deleted
the losers:

- **`src/indra_belief/prepared_execution.py`** is now the one request assembler.
  `prepare_from_record` (live) and `prepare_from_replay_row` (batch) build one
  `PreparedExecution`; `PreparedExecution.calls` is the only place the relation note and
  the entity-lookup block are spliced into the user message; `assert_replay_digests`
  re-checks the stored digests on every path. **[V]** Four assemblers were deleted:
  `grep -rnE "def (_build_messages|format_user_message|main_request|_record)\(" src/ scripts/`
  returns nothing. **[V]**
- **`src/indra_belief/verdict.py`** is the one parser and the one score map, replacing
  three implementations. `grid_score` returns `None` off-grid instead of fabricating
  0.5. **[V]**
- **`src/indra_belief/scorers/monolithic/scorer.py`** carries a `ScoringVariant`
  registry; every entry point takes `variant=`, so the scoring profile is an argument
  rather than an import-time read of `MONO_VARIANT`. **[V]** The *fallback*
  `DEFAULT_VARIANT` is still `variant_from_env()` at import — `research/serving_architecture.md`
  §4 records it as partly landed.
- **`comparison/runner.py` + `comparison/replay.py`** gained derived quarantine, a
  diagnostic budget, and a `settled` action status distinct from `complete` and
  `partial`; **`spend_guard.py`** indexes ledger calls by attempt instead of scanning.

`src/indra_belief/scorers/_shared.py` (sha256 `a95ada5c…`) and
`src/indra_belief/noise_model.py` (sha256 `6a427684…`) were deliberately **not**
touched: their bytes feed the published `implementation_digest` in
`comparison/llm.py`. **[V]**

---

## 2. Defects found and fixed, with the evidence

### 2.1 Ledger replay was quadratic

`SpendGuard._calls_for_attempt` replaced three full scans of `_reservations` with a
bucketed index. **[V]** The evidence that mattered was not the microbenchmark.

An early synthetic measurement of the lookup *in isolation* showed 111x/191x/480x/923x
at N=1k/4k/16k/64k. **[M]** That number was withdrawn as a framing error — it measured
the lookup, not replay. The replacement measured real `_LedgerReplay.apply` on a shipped
ledger: per-event replay went from **rising** (39.6 → 41.5 → 60.0 µs/event at n=1k/4k/16k,
max/min 1.516) to **flat** (39.3 → 37.8 → 39.6, max/min 1.047), with `_attempt_rows`
45x faster at 16k. **[M]** The 16,000-event replay state digest
(sha256 `378b22bf…a92f57`) was identical before and after — the index changed cost, not
outcome. **[M]**

Two structural facts make the index safe rather than merely fast: `_reservations` has
exactly one write site and no removal path anywhere in `src/` (no `del`, `.pop`,
`.clear`), so the index cannot hold a stale row **[V]**; and the duplicate-`call_id`
attack (a dict overwrites, a dict-of-lists appends, which would soften the
`call_ordinal` contiguity check) is closed upstream by rejecting a repeated `call_id`
before the write. **[V]**

Four absolute per-event timings were later **withdrawn** from
`research/serving_architecture.md` rather than corrected: labelled [V] but naming no
script, ledger, size or machine, and unreproducible on two independent runs. The shape
survives because the code structure proves it. *An absolute belongs in prose only with
its run conditions beside it.*

### 2.2 Two sites fabricated 0.5, and 11,645 replies would have taken it

At `58767d8` the retired live parser had two: an unparseable reply returned `0.5`
outright, and the grid lookup carried `_SCORE_GRID.get((verdict, confidence), 0.50)` —
so `{"verdict": "correct", "confidence": "certain"}` landed on 0.50 through a `dict.get`
default. **[V]** at `git show 58767d8:src/indra_belief/scorers/monolithic/_prompts.py`.
0.5 is not on the six-cell grid, so both wrote an invented number where the model gave
no answer.

`verdict.grid_score` now returns `None` on either axis, and absence propagates: the
runner turns `score is None` into `InvalidModelOutput` → retry → an ERROR row once the
per-source budget is spent. Twelve of twelve consumers were checked and **zero coercions
back to a number** were found. **[M]**

The measure of what was being fabricated is the arc's best single number:
`scripts/replay_parser_diff.py` cuts stored responses at seeded random offsets — which
is what truncation does to them — and reports `off_grid_live_score=11,645`. **[M]**
Eleven thousand six hundred forty-five replies on which the retired live parser would
have written a score no grid cell can produce. Part A of the same script re-reads all
228,812 stored LLM responses in `data/comparison*` with all four parsers and finds exact
agreement; Part B exists because Part A is survivorship-biased — an error row writes
`raw_text: ""`, so a parse failure erases its own evidence. **[V]** at
`scripts/replay_parser_diff.py`.

### 2.3 Statement belief: the route and the scalar disagreed

`statement_belief.py` gated the routing verdict on confidence
(`_CREDIBLE_LLM_CONF = {"high","medium"}`) but built the belief numerator without
consulting it. A `low`-confidence `incorrect` therefore drove `belief` to 0.0 while
`verdict_statement` still said `"correct"`. The constant is gone from `src/` **[V]**;
the invariant is stated in the module docstring and holds:
`verdict_statement == "correct"` implies `n_incorrect == 0`. **[V]**

Two things about the fix outlast the fix.

**The freeze was converted from bytes to behaviour, against an external reference.**
`statement_belief.py` had been frozen by sha256 into `implementation_digest` — "a strong
claim about the file and a weak one about the numbers: it forbids a comment and permits
nothing, so the file could not be corrected even where its output was provably
unaffected." `tests/test_published_statement_belief_reproduction.py` re-runs the **live**
aggregator over the frozen observations each bundle's own manifest names, verifies every
input against the sha256 and byte count that manifest declares, and requires exact float
equality with the **shipped prediction files**. Because the reference is immutable
published data, the order in which the test was written relative to the code change is
irrelevant — a characterization test captured after the change would have been
worthless. **[V]**

**Nothing published moved.** All 13,460 published scores across four comparison arms and
two panels re-derive with exact float equality. **[M]** `implementation_digest` itself
moved (published `eeb683fc…` in four `data/comparison/models/*/manifest.json` vs current
`3247f7dd…`) with the `noise_model` component unchanged, and
`tests/test_historical_e2b_bundle.py` still passes because it compares against the
digest stored *in the bundle* rather than a recomputation. **[M]** That is exactly the
freedom the behavioural freeze exists to grant.

### 2.4 Head-of-line blocking, and the budget that replaced it

The 2026-07-31 incident: one off-grid source halted a whole arm mid-corpus after 163 MB
of progress and needed a human to intervene.

The fix is **derived, not stored**. There is no new persisted field and no format
change: the settled set is recomputed inside `load_resume`'s existing single fold from
bytes already on disk plus `action.max_attempts` from the frozen plan, via
`comparison.replay._settled_reason` and `_settled_reason_from_class`. **[V]** A restart
recomputes the identical set from the identical bytes, so it can neither silently
un-quarantine and re-pay nor silently over-quarantine. `_run_source` re-checks the
settled set immediately before a provider reservation, so no-double-spend does not
depend on one caller building its pending list correctly. **[V]**

Disposition is an **allowlist with default halt** (`runner._QUARANTINE_KINDS`,
consumed at the single scheduler site). **[V]** Cap, auth, config, parser, WAL, deadline
and worker-exception all still halt; a pinned 25-case table includes an invented future
error type, a kind-less failure and `{}`, and all halt. **[M]** (F13 recorded 24; the
quarantine-budget case was added later. Re-measured:
`test_failure_disposition_is_an_allowlist` -> 25 passed.)

**The first version of this was wrong, and the way it was wrong is the lesson.** The
allowlist's default direction was verified and accepted as the whole answer. It is not:
`attempt_failed`/`InvalidModelOutput` was itself in the allowlist, and that class *is*
systematic. Measured with the arc's own harness — 8 sources, a client off-grid for all
of them: `quarantined=8, scored=0, PROVIDER_CALLS=40`. **[M]** At corpus scale that is
33,361 × 5 = 166,805 paid calls bounded only by the action cap
($39.96 `gemma_26b_primary` / $309.54 `glm_5_primary`), producing zero usable rows.
Pre-fix, the same run stopped after roughly `workers` sources. **Removing head-of-line
blocking removed the thing that had been bounding a systematic failure.**
*Default direction and membership correctness are different properties.*

The diagnostic budget (§6) is the bound that replaced it. Two further layers the first
fix never reached were also closed: the completion predicate had no notion of `settled`,
so a holed action pinned at `"partial"` forever, `contracts.RunPlan.ready_actions` kept
naming it, dependents deadlocked, and each futile re-run demanded a **live bearer token**
before computing pending. Now `contracts.TERMINAL_STATUSES = frozenset({"complete",
"settled"})` **[V]**, `PreparedRun` computes settled/pending before the readiness write
and only asks for a token `if pending`, and `comparison.replay.resolved_status` returns
`settled` on any hole. **[V]** Deliberately, `settled` does **not** satisfy
`depends_on` — only `complete` does — so a settled sensitivity arm blocks the three
primaries on purpose rather than releasing $39.96 + $39.96 + $309.54 on the strength of
an arm that failed. That is a stop, not a deadlock.

The supervisor was taught the new vocabulary: `scripts/supervise_comparison_arm.sh`
reads `failure["disposition"]` and `summary.quarantined` and distinguishes a `.SETTLED`
marker from a genuine `.ALERT`. **[V]** Its tests **extract the decision heredoc from
the shipped script and execute it**, and
`test_decision_source_is_the_shipped_heredoc` guards the extraction itself from going
vacuous. **[V]** That is §4.1's observe-don't-mirror discipline applied to a shell script.

### 2.5 The batch note path was digest-unchecked

`ReplayIndex` took the insertion-coordinate branch whenever a relation note was set and
never compared a digest; the branch that checks `main_prompt_base_sha256` was only
reached on the no-note path. Reproduced by accepting an arbitrary note string on a real
row. **[M]** **17,235 of the 33,361 primary-workload rows carried a note and therefore
had no prompt-digest constraint at all.** (The manifest's 17,257 / 33,413 counts both
workloads — the same rows plus 22 of 52 `alternate_prompt_sensitivity` rows.) The goldens now record the note-case digest itself under a fixed
literal, assert the insertion coordinates, and prove the coordinates *are* enforced by
tampering with the byte offset and requiring a `ReplayError`. **[V]** at
`tests/test_prepared_execution_goldens.py`.

### 2.6 Guards that could not fail

This is the arc's most-repeated defect. It appeared as a decorative `xfail` gate (no
`xfail_strict` in `pyproject.toml`, so an XPASS cannot fail); as
`git status --porcelain data/` used as a data-integrity check on paths that
`.gitignore` ignores, so it can never report a mutation — **used three separate times**
before it was banned outright; as a doc guard whose `DOCS` list did not contain the
document being written; and as a parser diff that could not fail because error rows
forbid `raw_text`, so the parsers agree on every stored response by construction.

Three fail-open modes in `scripts/check_new_section_anchors.py` were closed in sequence,
and the sequence is the point — each fix exposed the next. (1) **Coverage failed open**:
`SECTION_TITLES` guarded 380 of 762 lines, so an unregistered section carrying a dead
path exited 0 "OK". Closed with exit code 3 as a **dual** of exit 2 — opposite repairs,
and a rename raises both, so collapsing them would destroy the diagnosis. (2) **That
fix's own test was fragile**, hardcoding `SECTION_TITLES[0]`, so the first sibling to use
the documented extension path made exit 2 mask the exit-3 contract under test.
(3) **The guard validated paths, not symbols**: after the assemblers were deleted the
architecture document cited `main_request` (6 mentions), `_record` (5) and
`format_user_message` (3) — one calling the deleted `main_request` method on
`ReplayIndex` "the shipped renderer" — and the guard **exited 0**, because `replay.py`
still existed. Closed with exit code 4 and `find_dead_symbols`, precedence `2→3→1→4` so
coverage failures still outrank content failures. **[V]** The guard now states its own
coverage in its success output and **classifies** what it declined to read. **[V]**

Three shipped tests were separately found unfalsifiable: the goldens mirrored the
assembly instead of driving it (§4.1); the `variant=` seam's only test checked the
*signature*, which holds under both the correct and the broken implementation (**fixed
after the arc closed** — see §7.2 item 10); and the
parser's ordering contract passed 4/4 under **both** orders because the fixture built
`raw_text` so its last verdict object equalled `content`'s.

Mutation testing is what exposed these. On the unified parser an initial sweep killed
10 of 16 mutants and **6 survived; every survivor was a real gap**, each now a named test
carrying its consequence in the docstring
(`tests/test_verdict_parser.py::test_a_malformed_span_is_skipped_not_abandoned`,
`::test_the_last_verdict_phrase_wins_not_the_first`, and the strict-pair trio). **[V]**
For the ledger index six mutants were seeded and all six killed, including the
double-spend mutant that softens the `call_ordinal` contiguity check. **[M]** That
mutant, not the passing suite, is the evidence the index was safe to land.

---

## 3. What was refuted

These are worth more than the confirmations. Several are corrections the orchestrator
made to its own earlier statements; they are part of the record.

### 3.1 "Sharding has no cross-action disjointness check" — REFUTED, and both proposed fixes were wrong

The premise was the ground truth for a whole node. The check already exists, stated over
lanes rather than over keys:

- `comparison.replay.expected_execution_id` digests
  `{"model", "workload_mode", **execution_identity(source)}`, and
  `comparison.replay.execution_identity` returns
  `{eligible_position, paper_statement_hash, source_hash, evidence_json_sha256}` — no
  `action_id`, no `run_id`. Spend identity is a function of `(model, workload, row)`
  alone. **[V]**
- `contracts.load_run_plan` fails `"each spend-guard model must belong to exactly one
  stage"` and `"run plan repeats a model/workload execution lane"`. **[V]**

Together: `(model, workload)` is unique per action, and `execution_id` is a function of
exactly that plus the row, so **two actions in one plan cannot mint the same
`execution_id`**. The lane check already *is* the cross-action disjointness check.

Both proposed fixes were then falsified against shipped data. The **global hash-set**
form rejects two shipped plans — `data/comparison_verdict_only/run_plan.json` has four
actions all with `execution_keys: null` (the whole workload), so all four intersect
maximally, and in `data/comparison/run_plan.json` `e2b_smoke` pins one key the three
null-key actions all cover. **Overlap is the experiment**: those actions deliberately
score one corpus with different models. The **narrow within-lane** form is vacuous —
two actions can never share a lane. **[M]**

The real hazard, which the original review never named: relax the lane key to admit a
shard ordinal *without* adding the ordinal to `execution_identity`, and two shards
claiming one row mint the identical `execution_id`; on a shared ledger
`runner._relevant_attempts` then adopts the other action's attempts as its own, since
its only filter is execution-id membership. Six of eight actions in
`data/comparison/run_plan.json` already share one `spend.ndjson`. **[M]** The recorded
verdict is an **argued kill of the implementation**: implement nothing until
`execution_identity` carries the ordinal, the partition predicate is total
(`int(execution_key_sha256[:8], 16) % shard_count == shard_ordinal`), and each shard has
its own ledger **and** its own output file — the last because derived quarantine is a
per-action fold over that action's own durable output. The order is fixed: settle
`execution_identity` **before** relaxing the lane key, never after. Written up as
`research/serving_architecture.md` §10.

### 3.2 "A bundle excluding quarantined pairs cannot be produced" — REFUTED by an agent that built one

A consumer map claimed three gates made a holed bundle impossible. All three are over
the **input** files, which quarantine never touches, and the raw-validation gate
**passes**, because a quarantined pair does have raw rows: they are error rows. The
refutation was to build the artifact. Nothing shrank — statement predictions 3/3, reader
predictions 2/2, attempt rows 4/4, reader attempts 2/2, identical `ExpectedCounts` and
statement IDs — because predictions are emitted per **statement** and no statement
disappears. The viewer denominator hazard that had been called the worst part is not on
this path at all. **[M]**

The real reason to refuse is sharper than "impossible": **every structural check in the
system stays green while a published belief moves 0.97935 → 0.65.** A third of the
scale, invisibly, at the third decimal of AP/AUROC — not a slightly worse panel but a
different measurement wearing the same shape. **[M]** The ~15 coverage sites and 5 schema
gates are real and correctly reported; they simply are not what would have stopped this.

The measurement is preserved where the next person will meet it:
`tests/test_comparison_llm.py::test_a_quarantined_pair_can_never_be_bundled` carries the
0.979 → 0.65 figure in its docstring, and the gate had no test before. **[V]**

Corollary, and it reframes quarantine entirely: **this corpus is all-or-nothing.** A
published panel is 1,689/33,361 or it does not exist, so quarantine can never buy "finish
the arm anyway" — that outcome is unreachable. What it can buy is knowing **which regime
you are in** before you stop.

*Lesson: a map of consumers is a claim about the code like any other. The agent that
BUILT the artifact beat the agent that READ the gates — and a steer withdrawn for a good
outcome on a bad premise nearly recorded the bad premise as settled fact.*

### 3.3 The `verdict_only` "scoring incident" — REFUTED by its own author

A scorer docstring asserted that `MONO_VARIANT=verdict_only` had scored a whole run
under the wrong variant. Pressed on whether that was an incident or an inference, the
author checked and refuted itself. `data/comparison_verdict_only/grounding_replay`'s
manifest does carry `generation_contract.mono_variant = "verdict_only"`, which is not a
`VARIANTS` key — but that run's prompts never came from the scorer:
`scripts/build_verdict_only_replay.py` builds them directly from
`VERDICT_ONLY_SYSTEM_PROMPT`, and the substrate's stored main system is
sha `5781a5842d` / 3,671 chars, matching that prompt rather than the baseline
`c6845ab46c` / 3,411 chars. `grep -rn MONO_VARIANT src/indra_belief/comparison/` is
empty — **the batch path never reads the env var at all.** **[M]**

No production incident and no scoring consequence. What exists is a **latent trap**:
anyone reproducing that run by setting `MONO_VARIANT=verdict_only` against the *live*
scorer would silently get the baseline prompt. `scorers.monolithic.scorer.variant_from_env`
now warns on an unknown value instead of silently falling through. **[V]**

### 3.4 Four premises corrected by measurement

- **"The preflight SET can be narrowed."** False — a missing status defaults to
  `"pending"`, which would silently skip the advance-before-dependencies check and leave
  7 dependent actions permanently un-ready. The accompanying 2.6x causal claim was also
  wrong: three extra arms add ~11–13% of peak, not 3x. Rescoped to narrowing
  **retention**, measured 6,692 → 5,140 MB (−23%), 34.3 → 21.9 s (−36%). **[M]**
- **"11 prefix strings" vs "18" was not an error at all.** 11 counts components
  (2 systems + 9 prefixes), 18 counts co-occurring `(system, prefix)` **pairs**, which
  are the cacheable units; over the whole file it is 25 from 3×16, so 2×9=18 is
  coincidence, not a cross product. **[M]** Relatedly, the corpus is already
  prefix-grouped and vLLM defaults `enable_prefix_caching` on, so "sort shards by
  statement_type" is largely moot and the control arm is the OFF flag.
- **"A self-hosted lane has no per-token price."** Refuted at
  `src/indra_belief/corpus/cost.py`, which already prices local MLX gemma-4-26b/31b and
  the ollama gemma remote against their Bedrock twins. **[V]**
- **"152:1 prefill:decode is a property of the prompt."** It is a property of the prompt
  **and the decode regime** — the same corpus gives 5.74:1 with reasoning on versus
  154.5:1 verdict-only, 27x apart. **[M]**

### 3.5 The modularity verdict, which does not flatter the arc

Reported without netting:

| | before (rf) | after (rf) |
|---|---|---|
| live hops | 7 | 8 (rose) |
| batch hops | 2 | 4 (rose) |
| shared modules | 0 | 2 (`prepared_execution`, `verdict`) |
| union | 9 | 10 (rose, at every profile) |
| `shared_fraction` | 0.0 | 0.2 |
| `duplicate_site_count` | 10 | 5 (halved) |
| code lines (touched files) | 6,249 | 6,487 (**net +238**) |

By the audit's own proposed gate (max union rose → FAIL) the arc **fails**, and its
"shared up while union up" signature *looks* like ceremony. The counter-argument is not
that the metric is wrong but that it counts **modules**: extracting a shared module
necessarily adds one to both traces and cannot see that anything got smaller. 306 lines
of shared module replaced 104 lines of duplicated code and **the callers did not
shrink** — `scorer.py` grew 18 code lines after *losing* its renderer.

The number that reframes +238 without softening it is the 11,645 of §2.2: **the growth
bought a deleted failure mode, not a wrapper.** Both halves were recorded: "it moved real
structure AND it moved metrics the wrong way; both are true and I am not netting them
out."

---

## 4. Verification techniques that paid off

### 4.1 Drive production; never mirror it

The first golden harness re-derived the assembly rule in a helper whose own docstring
said "Mirror `scorer.py` … exactly". Changing production
(`system=ACTIVE_SYSTEM_PROMPT + _LOOKUP_GUIDANCE` → `system=ACTIVE_SYSTEM_PROMPT`) left
it at **20 passed, twice**. **[M]** A golden that re-implements the code it guards cannot
catch a change to that code. Rewritten to call the real entry point with a recording
client and to **partition the recorded call log on the `kind` production stamps** — the
route and tier are observed, not selected — the same seeded change gives
`1 failed, 24 passed`; reverted, `25 passed`. **[M]** That mutation is the acceptance
criterion, not the passing suite.

The same erosion recurred on one field: a golden's `resolved_variant` moved from
observing module state to reading the loader's own input. Byte-identical value, but that
field now reports what the harness was *told* rather than what the module *did*.

**Seeded mutation was the acceptance test throughout**, and every time it changed a
verdict: 6/6 ledger mutants killed; 6 of 16 parser mutants surviving; a doc guard proven
fallible by six seeded mutations in four failure modes with the file byte-restored each
time (verified by `shasum -c`, not by eye); the tripwire re-proven to bite *after* the
refactor. Mutants were injected via `sys.modules`, never written to `src/`.

### 4.2 Population differentials, not smoke tests

- **Whole-substrate re-derivation.** Every callable row of the shipped verdict-only
  substrate re-derived through the new producer and compared to the stored digest:
  32,357 callable rows, 1,056 deterministic, **0 `main_prompt_base_sha256`
  mismatches** — and `--verify-only` through the real loader over all 33,361
  executions. **[M]** If deleting the hand-copied assembler tail had moved one byte, all
  32,357 would have moved.
- **300 randomised `load_resume` trials**, full observable state (status, rows, done,
  attempts, latest keys, verdicts, per-row execution id / attempt ordinal / row status),
  run once in the working tree and once in a detached-HEAD worktree:
  `{'ok': 232, 'ReplayError': 68}` both times, dumps **byte-identical at 190,918 bytes,
  message for message**. **[M]** That is what proves *which rows count as done* did not
  change — no narrowing (no re-pay), no widening (no missing scores).
- **228,812 stored responses** re-read by all four parsers with exact agreement, plus a
  seeded-truncation arm to manufacture the population the corpus cannot contain (§2.2).

### 4.3 Re-insert the pre-fix rule to prove a change is routing-only

The statement-belief fix was not merely shown to leave published numbers unmoved — the
**pre-fix rule was re-inserted** and the reproduction still passed at delta 0.0. That
distinguishes "nothing moved" from "nothing *could* move", and the structural reason
backs it: `belief` never reads `confidence`, and the published prediction files carry
only `probability_correct` + `statement_id`, no route at all. **[M]**

### 4.4 Content-address gitignored trees; `git status` there is vacuous

`data/comparison/` and `data/comparison_verdict_only/*` are gitignored, so
`git status --porcelain data/` can **never** report a mutation to either. It was used as
an integrity gate three times before being banned. The replacement is
`hashing.sha256_file` over every file a module opens, taken at import and re-asserted in
the test and in fixture teardown; the whole-substrate check hashed all 54
substrate/manifest files before the first source edit and after the full verify chain and
found them byte-identical, corroborated independently by mtime. **[M]**

### 4.5 Prove hermeticity positively, and gate dependencies at build time

The batch image is gated on importing and running **without gilda or indra** present
(`ae9ab1e`), so the dependency claim is a build failure rather than a README sentence.

In the goldens, the obvious hermeticity proof — a Gilda stub that raises on an unknown
key — is *unusable*: the abbreviation path wraps in `except Exception: return []`, so a
raising stub turned a real abbreviation line into `[]` with nothing surfaced. The golden
would have been silently corrupted by the very technique meant to prove it clean. **[M]**
The positive form works: a recording stub collects unknown keys into a misses list
asserted empty afterwards, and the rendered lines are asserted equal to the row's stored
value. Three seams need stubbing, not two — the tool route reaches gilda through a
module-level import in `tools/gilda_tools.py` that bypasses both cached helpers — and
regeneration runs the capture twice, once from live gilda and once against the frozen
table, so a shipped fixture can never be gilda-dependent. **[V]** Against a
substrate-free root, **17 of 25 golden tests still run and pass** — the other 8 carry a
`requires_substrate` / `requires_vo_substrate` marker. **[M]** (A mid-arc note said 12 of
20; the observe-don't-mirror rewrite of §4.1 added five tests, which is why §4.1 quotes
`25 passed`.)

---

## 5. Process lessons

**A snapshot taken during a mutation run is a mutant, not a baseline — and a single read
of a live tree is not a baseline either.** A reviewer watched `verdict.py` change under
it mid-review: two identifiers swapped with no change in byte count, flipping a real
scoring outcome (content-first `('incorrect','high')` = 0.05 vs raw_text-first
`('correct','low')` = 0.65). It was a **seeded mutant** — the orchestrator had commissioned
exactly that swap from a sibling. An "insurance" copy was then staged from the live tree,
and 12 seconds later the tree disagreed with it on a *different* line, from which the
orchestrator concluded its copy was contaminated. **That conclusion was backwards.**
Resolved against a fixed reference: `git show HEAD:` of both retired parsers, which both
take the **last** match. The staged copy was right; the watcher had fired during a
different mutant in the same cycle and read a transient state as the resting state.
*Do not resolve "which of these is the original" by trusting the tree, and do not resolve
it by trusting a copy. Resolve it against a fixed reference — that settles it in one
command and is what should have been run first.*

**Do not run seeded-mutation agents concurrently with reviewers of the same file.** Two
careful reviewers reported opposite readings of the same line at the same stated mtime,
because the file was moving under both. That conflict was never resolved and is recorded
as unresolved; what is settled is that the resting code matches both retired
implementations and the docstring. Serialize mutation and review, or give the mutating
agent a private copy.

**Executing a line is not depending on it.** A proposed discriminator for a parser mutant
was validated with `sys.settrace` showing the target lines executing — and the whole step
could then be **deleted** with the test still passing 51/51, because the input fell
through to a later step that returned the identical answer. The real discriminator is
what the step is *for*: matching both fields inside **one** brace span, where the later
step runs two independent sweeps and **cross-pairs** across objects, turning a
high-confidence rejection into a low-confidence one — a silent 0.30 error with a
confidence the model never attached to that verdict.

**Absence of a finding is not a verdict.** Green verify is necessary, not sufficient;
three statuses set by transport failures rather than by refutation were re-measured green
and still re-reviewed rather than promoted. Equally: a "pass" from a gate that cannot fail
is not evidence, and this arc found four such gates. And one file handed to two concurrent
workstreams collided undetected — diffing the file sets before parallel work costs one
command.

**Withdraw an unreproducible claim rather than repair it.** Four absolute timings and a
review-only figure (`n=19,641`) were both dropped once nothing in the repository
reproduced them; that figure had only ever existed inside a review message, and
`grep -rn "19641"` across `src/ tests/ scripts/ research/ .github/` returned nothing when
checked. **[M]** It no longer does: this sentence is now the only hit. That is the same
self-falsifying-provenance trap §5 records, reproduced by the document that records it —
which is the argument for citing a *command and its date*, not a standing claim about a
repository that the citation itself changes.

---

## 6. The quarantine diagnostic budget, and why the constants are what they are

`runner.QUARANTINE_DIAGNOSTIC_LIMIT = 8` and
`runner.QUARANTINE_DIAGNOSTIC_SOURCES = 200`. **[V]** One mechanism, two terms, because
they are the same question asked of two regimes — and the reasoning is priced in dollars,
so it must not be lost:

- **`QUARANTINE_DIAGNOSTIC_LIMIT = 8`** — eight retirements is systematic breakage, not
  bad luck at the 0.057% sporadic off-grid rate four production arms actually carry. It
  trips almost immediately when every source fails, capping that case at 8 × 5 = **40
  provider calls whatever the corpus size**. Without it, measured: 40 calls at 8 sources,
  1,000 at 200, 5,000 at 1,000, extrapolating to 33,361 × 5 = 166,805 — bounded only by
  the action cap ($39.96 `gemma_26b_primary` / $309.54 `glm_5_primary`), producing zero
  usable rows. **[M]**
- **`QUARANTINE_DIAGNOSTIC_SOURCES = 200`** — the sporadic ceiling. At the measured rate
  the next bad row is ~1,756 sources away, so enumerating them all means paying the whole
  cap for a bundle that cannot exist. 200 further sources costs ~$0.24 (gemma_26b) /
  ~$1.86 (glm_5) and establishes that failures are **not** dense. **[M]**

The bound is **constant, not proportional**: 55 calls (11 dispatched) at n=200, n=1,000
and n=33,361 alike, against 1,000 / 5,000 / 166,805 before — the first two measured, the
166,805 an **extrapolation** from them, never run. **[M]/[R]** Sporadic regime,
n=1,000 with one bad source: 1,004 → 221 calls. A clean n=200 run: 200 calls,
`status=complete`, `failure=None` — the breaker **cannot** fire on a clean arm and is
inert until the first quarantine of the run. **[M]** The halt kind `quarantine_budget`
carries `regime: "systematic" | "sporadic"`; **that field is the product of the whole
mechanism.** Because any hole is terminal, restart is free and self-limiting — no
supervisor can deliver the burn in instalments of one budget per restart. And it survives
plan amendments: `settled` is computed against the *current* plan's `max_attempts`, so
raising it un-settles an `attempts_exhausted` source.

`attempt_failed`/`InvalidModelOutput` stays in the allowlist — correctly, now that the
budget is the bound.

---

## 7. What is still owed

### 7.1 Three operator decisions, open, with their numbers

**(a) medpsy regeneration — reported, deliberately not applied.** If the head-to-head
artifacts were regenerated, the **only** fields that move anywhere are
`statement_error_detection.confusion` and `verdict_statement_counts`, on the two medpsy
runs only. Every AUROC/ECE and every coverage count is identical — independent
confirmation that the scalar did not move. Both medpsy F1s move **up**:

| run | before | after | flips |
|---|---|---|---|
| medpsy `eval_curation_v1` | .7759 | .7785 | 5 (538 → 533 correct→review) |
| medpsy `external_curator_v1` | .7658 | .7869 | 11 (250 → 239) |

The three gemma runs are byte-identical in `statement_error_detection` — 0 flips,
confirmed by row census (gemma 0/0/0; medpsy 11 and 18 rows). **[M]** Nothing was
applied; the frozen artifacts keep their 2026-07-13 mtimes. This is a human call:
regenerate (the improvement is a correctness fix, not tuning) or leave frozen.

**Data-availability caveat:** `holdout_cc_gemma` could **not** be regenerated at all,
because its gold `data/benchmark/holdout_cc.jsonl` no longer exists. Its zero therefore
rests on a **row census**, not on a re-run. Worth knowing before anyone plans work on
that holdout.

**(b) `by_driver` stratification — exported, would move on regeneration.** `driver` is
exported: `results.py` → `tiers.stmt.stratified.by_driver` → `data/exports/*/metrics.json`
→ `viewer/src/lib/components/DriverMosaic.svelte`. **[V]** Replicating the rule over each
export's `per_evidence.jsonl`: **12 of 53 exports would move if regenerated**, every flip
`none → llm`, **4,997** low/absent-confidence `incorrect` rows across them, largest
**976 of 8,716**. Method validated against stored blocks — the `llm` and `deterministic`
counts reproduce exactly. **[M]** **No shipped number has moved** and no scalar can move.
The new label is *more* accurate: `statement_belief` builds its gated set without
consulting confidence, so a low-confidence rejection was already driving belief down
while the old label `"none"` named no path when a path existed. Same class as (a) — a
derived stratification that shifts on regeneration. An earlier "moves no exported number"
assessment was **withdrawn**.

**(c) One fossil row in `glm_5_sensitivity`.** A single stored row carries
`verdict=None, confidence=None, score=0.5` — the very fabrication this arc deleted,
written by the old code. Re-verified 2026-08-03 by census of every
`data/comparison/runs/*/attempts.jsonl`:

```
gemma_26b_primary  rows=33904 scored=33361 fossil=0   e2b_smoke        rows=    1 fossil=0
gemma_31b_primary  rows=33880 scored=33361 fossil=0   *_sensitivity    rows=52/ea fossil=0
glm_5_primary      rows=33838 scored=33361 fossil=0   glm_5_sensitivity rows=  53 fossil=1
```

**[M]** The row is `(stmt_i=36, evidence_i=0)`, tier `llm_tool_use`, `row_status=scored`,
`raw_text` empty. **All three primary arms are clean at 33,361 scored rows each with zero
mismatches.** **[M]**

It **blocks `status`**: `python -m indra_belief.comparison.cli status --plan
data/comparison/run_plan.json` exits non-zero with
`scored row score differs from verdict/confidence`. **[M]** It is **pre-existing**: the
same `ReplayError` string is present at the pre-arc base
(`git show 58767d8:src/indra_belief/comparison/replay.py`), and the attempts file's mtime
is 2026-07-20, well before the arc. **[V]** Untouched, because editing a ledger-backed
attempts log is an operator call.

### 7.2 Named technical debt

1. **The modularity instrument is self-reporting and its anchors have rotted.**
   `scripts/modularity_baseline.py` compares hard-coded `DUPLICATE_SITES` and `CONCEPTS`
   literals against themselves, and its `sys.settrace(None)` clobbers any outer tracer.
   **[V]** **8 of the 10 line anchors in the surviving 5 duplicate-site rows are now
   wrong** — two point into `replay.py` at code that moved to `prepared_execution.py`
   entirely, and four more moved within their files. **[V]** (That breakdown accounts
   for six of the eight; the headline 8-of-10 is the exact count, re-derived cell by
   cell, and the two unclassified cells were not re-traced.) The instrument-hardening
   half of the audit (rewrite the instrument, freeze a baseline JSON, add
   `tests/test_modularity_baseline.py`) was rescoped out and **does not exist**; all
   three gaming attacks the audit reproduced are still live. Any future modularity claim
   rests on this file, so it needs its own piece of work.
2. ~~**Byte-level twins remain, and one is prompt-bearing.**~~
   **DISCHARGED — X1-twins collapsed the duplicated relation text and default-accept core.**
   The label table and mismatch sentence now have one owner in
   `prepared_execution.relation_mismatch_note`; both
   `scorers.monolithic._prompts_relation.resolve_relation_nature` and
   `comparison.replay._relation_note` retain their path-specific parsing and normalization
   before delegating to it. The relation sub-call user message now has one owner in
   `prepared_execution.relation_user_message`; both the live wrapper and
   `ReplayIndex.relation_request` delegate to it, discharging the prompt-bearing twin too.
   The eight immutable no-text values now live in `verdict.NO_TEXT_RESULT`, while each caller
   still creates its own `call_log` and the live-only example-selection keys remain unchanged.
   **[V]** The earlier census mislabelled the first — the two `_relation_note`
   *functions* are not twins (live dispatches a call, batch formats a stored reply); the
   twin is one level down, in the label table.
3. ~~**No test runs `scripts/replay_parser_diff.py`.**~~
   **DISCHARGED — `tests/test_replay_parser_diff.py` freezes the population.**
   `tests/goldens/parser_diff_population.json` holds the Part A / Part B totals and the
   row census (238,039 rows / 228,812 considered LLM responses, 15 logs), the seeded
   constants that define the mutant population, the full untruncated text of all six
   truncation mutants a parser pair disagreed on, and 29 stored responses all four
   parsers agree on. **[M]** The 8.5 GiB of attempt logs are a gitignored published
   artifact, so the corpus-wide scan is skipped only when the corpus is WHOLLY ABSENT —
   present-and-different FAILS, and there is no env var that disables it. The six
   mutants and the 29 agreeing texts are re-read by all four parsers from the fixture
   itself, so those run on a fresh checkout with no `data/` at all. Read-only is proven
   positively rather than by `git status`, which is vacuous over a gitignored tree: a
   per-log `(size, st_mtime_ns)` is recorded at import and re-asserted after the scan.
4. ~~**`research/serving_architecture.md` §3 and §4 have gone partly stale.**~~
   **DISCHARGED — the doc-drift audit applied the repairs.** As recorded: §3's F8 was
   still written as an open `[R] HIGH` finding and still named `_CREDIBLE_LLM_CONF` as
   live code — the constant no longer exists in `src/` **[V]**; F6 was likewise
   unmarked after quarantine landed, while sibling findings F1/F2/F5 *did* get closure
   markers, so a reader reasonably took F8/F6 as still open. Both now carry
   `CLOSED by a10df62`. §4's "Fix (small, surgical)" items 4, 5, 7 and 9 had all landed
   in this tree while still written in the imperative **[V]**; all four are now struck
   through and marked LANDED. **Correction to this item as first written:** item 6
   ("Preflight only the selected action") did NOT land and is not stale prose — it is a
   refuted premise, as §3.4 bullet 1 of this same document records. `runner.inspect_plan`
   and `runner.prepare_run` still fold over the whole `loaded.actions` set on purpose,
   so §4 item 6 now says so explicitly rather than claiming a landing. **[V]**
5. **`PreparedExecution.parser_id` and `.profile_id` are write-only.** Zero readers
   across `src/`, `tests/` and `scripts/` **[V]**, while the module docstring sells
   `profile_id` as the provenance that keeps a score attributable to an exact prompt.
   That is ceremony standing where an invariant is claimed. `profile_id` also
   name-collides with the established, unrelated `profile_id` in
   `calibration_constants._PROFILE_META`. **[V]**
6. **`tests/test_published_statement_belief_reproduction.py` has no skip guard, and CI
   cannot run it.** It hard-requires tens of GB of gitignored attempts logs **[V]**, and
   this arc made two previously-independent freeze tests depend on it. Both concerns are
   real — a skip guard lets a freeze silently stop running; no guard makes CI fail on a
   fresh checkout. The resolution is a guard that **fails loudly** on absent data rather
   than skipping silently or hard-crashing. An earlier one-sided endorsement of "no skip
   guard" was withdrawn.
7. **Crash-safe resume is a stated project invariant this codebase does not satisfy** —
   before or after this arc. A torn trailing JSONL line raises with no
   truncate-and-recover path, so the arm is unresumable until a human truncates the file.
   `AppendLog.append` is one unbuffered write + fsync, so `SIGKILL` cannot tear it, but
   power loss between page-cache landing and fsync can, and so can the short-append
   branch, which raises *after* the bytes are on disk. It fails **closed**, the safe
   direction for a paid lane — but the green suite is not evidence that the invariant
   holds. Pre-existing and byte-unchanged from the base commit.
8. **`nonretryable_failure_on_resume` collapses distinct classes.**
   `comparison.replay._settled_reason_from_class` fires whenever the retry class is
   `None`, lumping auth 401, config and parser errors together. **[V]** First occurrence
   still halts globally, so the invariant is intact; but on restart such a source becomes
   a hole, and any hole is terminal, so it stops the arm — the safer direction, at the
   cost that a transient credential blip near the end of a run terminates the arm instead
   of retrying. The fix is to carry `error.type` into the disposition. **This is the next
   thing to take.**
9. **The digest circularity is not closed by the one-module fix**, and the architecture
   doc says so. `scripts/build_verdict_only_replay.py` computes `main_prompt_base_sha256`
   through the same `prepare_from_replay_row` that `assert_replay_digests` later checks it
   against, so the digest cannot catch a change in the assembly. What catches it is
   external — the 15 prompt components the shipped
   `data/comparison/grounding_replay/manifest.json` commits to, re-derived through the
   live producer by `tests/test_prepared_execution_parity.py`. **[V]**

10. ~~**The `variant=` seam is still untested behaviourally.**~~
    **DISCHARGED — the last of the three unfalsifiable tests now has a behavioural gate.**
    The signature assertion remains and is still not a behavioural test on its own:
    `test_score_entry_points_accept_a_variant` holds under both the correct
    implementation and one where `score` accepts `variant=` and silently ignores it.
    What was missing was a test that passes a non-default variant to the *public*
    entry points rather than proving injection through `_prepare`. **[V]** Four now do,
    in `tests/test_monolithic_variant_profile.py`: `score` puts the requested variant's
    system prompt on the wire, every registered variant reaches the wire through
    `score`, the variant selects the CALL TOPOLOGY and not only the prompt (whether the
    relation-nature sub-call fires), and `score_statement` / `score_evidence` — the seams
    the API layer holds — pass it through. The property gated is the one the frozen
    golden cannot see, because the golden was captured when a process had exactly one
    profile: the variant handed to a CALL is the variant that reaches the wire on that
    call. **[V]**

- **The mutation counts** (10 of 16 parser mutants killed; the ledger's 6/6) and every
  quantity above — 13,460 published scores, 11,645 off-grid, 32,357 re-derived rows, the
  300-trial differential, 228,812 responses, every dollar figure — are the arc's own
  measurements, re-cited here, **not re-measured**. `scripts/replay_parser_diff.py` and
  `scripts/reproduce_published_statement_beliefs.py` reproduce two of them on demand; the
  surviving mutants' tests exist and are named in §2.6. **[V]**
- The **full test suite was not re-run** for this document; the arc's last recorded run
  was 1,473 passed / 1 skipped.
- Whether the parser's field-order line was ever briefly shipped inverted **remains
  unresolved** — two careful reviewers reported opposite readings at the same mtime (§5).
  What is settled is the resting code.
- The prefix-cache numbers in `research/serving_architecture.md` §9 and the latency and
  grounding numbers in `research/serving_deployment.md` were not re-verified here; each
  document carries its own provenance.
