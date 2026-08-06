# Operator data corrections: the two open decisions, with their exact edits

`research/kernel_unification_findings.md` §7.1 records three operator decisions left open
by the kernel-unification arc. Two of them are data corrections that nobody has applied,
because applying them means editing artifacts that are gitignored, ledger-backed, or
both. This file operationalises those two. It does not replace §7.1; it turns §7.1 into
instructions an operator can run, and it corrects the premises that turned out to be
wrong when they were checked against the disk.

**Nothing here has been applied.** No file under `data/**` was written, moved, or deleted
in producing this document, and no provider call was made. Every measurement below was
re-derived on 2026-08-05 against the working tree at `ccf53bf`.

Claims are labelled as in §7.1: **[M]** measured, **[V]** verified against the tree at a
named symbol, **[R]** reasoned. Citations are by symbol, never by line number.

The two corrections are independent. Either can be applied without the other.

---

## 0. Premise corrections

Five things that were treated as settled are wrong or incomplete on disk. They are stated
first because two of them change what the operator is being asked to do.

**(a) The fossil row is not a stuck source.** It is `attempt_ordinal: 1` of
`(stmt_i=36, evidence_i=0)`, and the *last* row of the same file is `attempt_ordinal: 2`
of that same source, with `row_status: "scored"`, `verdict: "incorrect"`,
`confidence: "high"`, `score: 0.05`, `tokens: 2127`. The file holds 53 rows over 52
distinct sources. **[M]** The source was retried and the retry succeeded. Correcting the
fossil therefore costs **zero provider calls** and leaves the arm `complete` at 52/52 —
no re-dispatch is owed, and none should be attempted.

**(b) `data/comparison/exports/` does not exist at all.** The claim that it "is empty" is
wrong in a way that matters: there is no such directory. **[M]** The published bundles
are the five directories under `data/comparison/models/` — `gemma_4_26b`,
`gemma_4_31b`, `gemma_4_e2b`, `glm_5`, `indra_cogex_hybrid`. `glm_5_sensitivity` is not
among them and is not scheduled to become one.

**(c) The blast radius omits the spend WAL.** The fabricated row exists in two durable
places, not one. §1.4 gives the ledger copy, its digest, and why it must not be edited.

**(d) The bundle path does not cross-check the ledger's copy of the row.** The claim that
a bundle build over this arm "would fail at the raw-row/WAL-outcome comparison" is wrong:
`src/indra_belief/comparison/llm.py` reads only `attempt_started`, `attempt_finished`,
`call_reserved` and `call_settled` off the ledger. `attempt_outcome_committed` — the event
that carries `raw_row` — is never read there, and `_validate_raw` never inspects `score`.
**[V]** The only code that compares an attempts row against the ledger's committed copy is
`_reconcile` in `src/indra_belief/comparison/runner.py`. The divergence is real but
dormant everywhere; see §1.4.

**(e) One of the twelve exports does not move where §7.1(b) says it moves.**
`a7faca4104a7466e860e08f4a07d14d1`'s exported `by_driver` block is **unchanged** by
regeneration — its `none` count stays 28, not 28 → 17. All eleven of its flipping
statements fall outside the gold-covered stratum that `by_driver` is computed over. Its
export still moves, but in `per_statement.json`, not in `metrics.json`. Measured by
regenerating it into a scratch directory and diffing. **[M]** §2.2 gives the corrected
table for all twelve.

---

## 1. Correction 2 — the fabricated score in `glm_5_sensitivity`

### 1.1 The file, and what line 37 holds

`data/comparison/runs/glm_5_sensitivity/attempts.jsonl`, 53 lines, 2,494,831 bytes.

```
sha256(data/comparison/runs/glm_5_sensitivity/attempts.jsonl)
  = efa3de5c4e384502e77cd52cb9abe9bee2dd5eaab1f6d03de45becea91fd18ba
```

Line 37 (1-indexed), 241,083 bytes including its trailing newline, digest
`6be57bda02abaa4b72f22099052653a611b35ffcb8ebd2b0bfadd16c1980296d`, holds: **[M]**

| field | value |
|---|---|
| `attempt_id` | `cec9b0b38f1e407bafbe2fde05e4fb7e` |
| `execution_id` | `6e1f2e830305ca0025fe5ea3df9c5fe246a2efac8611e1aa5fd3b017d7318e97` |
| `attempt_ordinal` | `1` |
| `row_status` | `"scored"` |
| `attempt_status` | `"completed"` |
| `verdict` | `null` |
| `confidence` | `null` |
| `score` | `0.5` |
| `tokens` | `32000` |
| `latency_s` | `276.064` |
| `tier` | `"llm_tool_use"` |
| `grounding_status` | `"flagged"` |
| `provenance_triggered` | `true` |
| `raw_text` | `""` |

Two `call_log` entries. The second is `kind: "monolithic_tool_context"`,
`max_tokens: 32000`, `out_tokens: 32000`, `finish_reason: "length"`, `raw_text: ""`. **[M]**
That last field is the proof: the model consumed its entire 32,000-token ceiling and
emitted nothing parseable. `0.5` is not on the six-cell grid — no verdict/confidence pair
can produce it — so it was never a measurement. It is the fabrication the arc deleted,
written by the pre-arc live path and left behind in a durable file.

### 1.2 What the current implementation writes for a new occurrence

Not a guess — this is what the code does today. In
`src/indra_belief/comparison/runner.py`, `_attempt`'s `invalid` branch fires when the
verdict is off the closed set, or the confidence is off it, or `result["score"]` is
`None`. It calls `error_row` in `src/indra_belief/comparison/replay.py` with
`InvalidModelOutput(_INVALID_OUTPUT_MESSAGE)`, producing: **[V]**

```
row_status: "error", attempt_status: "error",
verdict/score/confidence/tier/grounding_status/provenance_triggered/tokens: null,
raw_text: "",
error: {"type": "InvalidModelOutput",
        "message_sha256": "28600b00ab90a869bd70686ba56a5754973b16fc8129a409b419112a995f02f8"}
```

That digest is the SHA-256 of `provider response is not an on-grid (verdict, confidence)
pair`. The recovery path in `_recover` passes the bare string `"InvalidModelOutput"`
instead of an exception, which `error_row` hashes to
`8c044e6d8091275ade4b4e4643f3e5dbf5d62f9099f3cc0bf154d1a954fdd10a`. **[M]** Both are
recorded here so a reader comparing an error row against this section is not surprised by
which one they find.

### 1.3 The patch, and why it is not the error row

**Change one field on one line. `"score": 0.5` becomes `"score": null`. Nothing else.**

The line must stay canonical JSON under `canonical_json_line` in
`src/indra_belief/hashing.py` — sorted keys, `(",", ":")` separators,
`ensure_ascii=False`, trailing newline — or `_scan_resume` in
`src/indra_belief/comparison/replay.py` rejects the whole file. The current line already
satisfies that, and re-emitting the parsed row through `canonical_json_line` preserves it.
**[M]** Every other line stays byte-identical.

Resulting digests, computed in memory without touching the file:

| | bytes | sha256 |
|---|---|---|
| line 37, patched | 241,084 | `98ed45391ffaf99ddcb21bb4c80085766832b1fca217e03548641e886c790043` |
| whole file, patched | 2,494,832 | `b5ad47d52ad54b599b9679e60b13f169848c7e8532013c11166c74c9064e0481` |

The alternative — rewriting the row into the full error shape of §1.2 — was constructed
and measured too, and **rejected**. Its file digest is
`4d74603796114914efff490be4740252505ee40f5c69fc4c11c255c0c93a9879` (2,494,920 bytes),
recorded so the choice is auditable rather than assumed. Three reasons the scored-with-no-
verdict shape wins for this **historical** row:

1. **It validates.** `validate_row` in `src/indra_belief/comparison/replay.py` accepts it:
   `grid_score` in `src/indra_belief/verdict.py` returns `None` for `(None, None)`, so the
   score check passes, and the remaining scored-branch checks already hold — `tier` equals
   the route's `llm_tool_use`, `provenance_triggered` is a bool, and `tokens` is not
   `None` on a callable route. Run against the real source: the row as it stands raises
   `ReplayError: scored row score differs from verdict/confidence`; the patched row
   validates. **[M]**
2. **The ledger already committed `completed`.** `_validate_raw` in
   `src/indra_belief/comparison/llm.py` derives `expected_status` from `row_status` and
   compares it to the WAL's `attempt_finished` status. The WAL committed `"completed"` for
   this attempt (§1.4), so an error row would demand `"error"` and fail with *differs from
   WAL completion*. The same function explicitly permits the scored-abstention shape as a
   retry predecessor — its check is "retry predecessor is not an error/abstention", which a
   non-error row with `verdict: null` passes — and `_cost_rows` in the same module names
   that shape `ParserAbstention`. **[V]**
3. **It is one token of review surface.** One field instead of eight.

It is also the honest state. "Ran, produced no verdict" is exactly what happened: the
provider was called, it was paid for, it returned 32,000 tokens of nothing. No invented
number survives the patch, and none is added by it.

### 1.4 Blast radius, including the ledger copy

The fabricated row lives in **two** durable places.

`data/comparison/runs/spend.ndjson` (5,095,405,915 bytes, SHA-256 hash-chained; shared by
`e2b_smoke`, all four sensitivity arms, and `gemma_31b_primary` — `gemma_26b_primary` and
`glm_5_primary` keep their own ledgers) carries, for attempt
`cec9b0b38f1e407bafbe2fde05e4fb7e` / execution
`6e1f2e830305ca0025fe5ea3df9c5fe246a2efac8611e1aa5fd3b017d7318e97`: **[M]**

- sequence 1330, `attempt_outcome_committed`, whose `raw_row` is the fabricated row
  verbatim, with `raw_row_sha256 = 99acfcb819b271948fa8eb46f78f24f3b0664bfa4511a3c5fde508bb3b54810c`
  and `status: "completed"`;
- sequence 1331, `attempt_finished`, `status: "completed"`.

**Do not touch the ledger.** It is append-only and its chain is tamper-*evident*, not
tamper-proof, as `src/indra_belief/spend_guard.py` says in as many words. Re-chaining a
forged edit is out of scope, and a ledger that has been rewritten to agree with a
corrected row is worth less than one that disagrees visibly. Its digest today, for the
record that it did not change:

```
sha256(data/comparison/runs/spend.ndjson)
  = 83b06853ebbec42e96afa06fe09a3688ea2fae6a528319eb7760531f75af49dc
```

The consequence, bounded honestly: after the patch the attempts row no longer equals its
WAL copy, so `_reconcile` in `src/indra_belief/comparison/runner.py` would raise
`RunnerError("raw output differs from committed WAL evidence")`. **That code is
unreachable for this arm.** `_reconcile` runs only inside `prepare_run` and
`_run_prepared` — the body behind the public `run_prepared` — and nowhere else in the
module; `ready_actions` in `src/indra_belief/comparison/contracts.py` filters out every action
whose status is in `TERMINAL_STATUSES` (`complete`, `settled`); and after the patch the
arm is `complete`. **[V]** The divergence is real, dormant, and confined to one arm that
is never scheduled again.

Nothing else reaches it. Per §0(d), the bundle path does not compare the raw row against
the ledger's committed copy, so a bundle build over `glm_5_sensitivity` would not fail on
the divergence — which is moot anyway, because `glm_5_sensitivity` is not a published
bundle. The five published bundles are named in §0(b) and none of their numbers move.

### 1.5 Verification, three separable legs

Snapshot the file to a scratch path outside `data/` first (`cp` to `/tmp` or similar);
leg 2 needs the pre-image.

**Leg 1 — file digest.**

```
shasum -a 256 data/comparison/runs/glm_5_sensitivity/attempts.jsonl
# expect b5ad47d52ad54b599b9679e60b13f169848c7e8532013c11166c74c9064e0481
```

**Leg 2 — byte identity of the other 52 rows.** A digest match alone does not prove the
edit was surgical; this does.

```python
import json, pathlib
before = pathlib.Path("<scratch>/attempts.jsonl").read_bytes().split(b"\n")
after  = pathlib.Path("data/comparison/runs/glm_5_sensitivity/attempts.jsonl").read_bytes().split(b"\n")
assert len(before) == len(after) == 54          # 53 rows + the empty tail
for i, (a, b) in enumerate(zip(before, after)):
    if i != 36:
        assert a == b, f"line {i+1} changed"
old, new = json.loads(before[36]), json.loads(after[36])
assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {"score"}
assert old["score"] == 0.5 and new["score"] is None
```

**Leg 3 — the status command.**

```
.venv/bin/python -m indra_belief.comparison.cli status --plan data/comparison/run_plan.json
```

Expect exit 0, with `glm_5_sensitivity` reported `complete 52/52`. Measured on a patched
copy: `complete 52/52`, 53 attempts, 0 settled. **[M]**

This leg rescans every attempts file in the plan — 5,475,345,361 bytes across the eight
actions, 5.46 GB of it in the three primary arms. Measured today at 19.9 s + 18.3 s +
12.1 s for the three primaries, so budget about a minute; it is not instant, and it is not
hung.

**The measured pre-state, so the operator can tell a fix from a coincidence.** Today that
command exits **2** with `cli.py: error: scored row score differs from verdict/confidence`,
printing no status document at all. The other seven actions are already `complete`:
`e2b_smoke` 1/1, `e2b_sensitivity` 52/52, `gemma_26b_sensitivity` 52/52,
`gemma_31b_sensitivity` 52/52, and the three primary arms 33,361/33,361 each. **[M]**
Because every dependency is already `complete`, the dependency-consistency check inside
`ready_actions` cannot fire behind the fix.

### 1.6 The validator is right

`src/indra_belief/comparison/replay.py` is correct to reject this row, and no patch, flag,
or lenient path around it is proposed here. Its scored-row branch asserts that a row's
score is the grid score of its own verdict and confidence. That assertion is the single
thing standing between the ledger and a re-admission of fabricated scores — relaxing it to
unblock one historical row would re-open the exact hole the arc closed, across every arm.
The row is what is wrong. Fix the row.

---

## 2. Correction 1 — the reject-driver relabel, and what regeneration moves

### 2.1 Root cause, and the true scope

Commit `a10df62` deleted the confidence gate from `src/indra_belief/statement_belief.py`.
Before it, an LLM rejection was credited toward the review-driver counter only at `high`
or `medium` confidence, through a module constant named `_CREDIBLE_LLM_CONF` — mentioned
here undotted and historically, because the symbol no longer exists. After it, **any**
credited LLM rejection counts, at any confidence. Same rows, same beliefs, one more
rejection credited.

The counter is exported. `src/indra_belief/results.py` derives each statement's `driver`
from it — `deterministic` if a deterministic reject exists, else `llm` if a credited LLM
reject exists, else `none` — and that label rides out through
`tiers.stmt.stratified.by_driver` into `data/exports/<run_id>/metrics.json` and
`viewer/src/lib/components/DriverMosaic.svelte`. **[V]**

§7.1(b) understates the scope. A statement whose driver flips `none` → `llm` had, by
construction, no other credited rejection — so its tiered `verdict_statement` moves
`correct` → `review` at the same time. **Every `by_driver` flip is simultaneously a
`tiers.stmt.verdict_err` movement**, in every affected export, not only the two medpsy
runs §7.1(a) reported. Each stratum block carries its own `verdict_err`, so the sibling
strata — `by_stmt_type`, `by_n_sources`, `by_n_evidence`, `by_dominant_bucket` — move
their `verdict_err` sub-blocks too, even though their membership is unchanged. **[V]**

**No shipped belief scalar moves.** The belief computation never consults the counter: the
gated set is built from `verdict` alone. Verified by regenerating ten exports into a
scratch directory and comparing — every **run-level** belief block is float-identical
(`tiers.stmt.arms.hard`, `tiers.stmt.arms.parametric`, `tiers.stmt.arms.soft` and
`tiers.ev.arms.score`, AUROC and ECE included), and `per_evidence.jsonl` is byte-identical.
**[M]**

*Run-level belief* is load-bearing there, and the exception is worth stating plainly
rather than letting a reader find it in a diff. The `by_driver` strata are a partition
**over the relabelled statements**, so moving a statement from `none` to `llm` changes
which rows each stratum averages. `tiers.stmt.stratified.by_driver.llm.hard` and
`tiers.stmt.stratified.by_driver.none.hard` — `ece`, `auroc`, `auprc`, `brier`,
`reliability`, `resolution`, `uncertainty`, `confusion` and `bins` — therefore **do** move,
in nine of the ten gold-backed exports, 20 to 39 changed leaf keys each (`923e4d3d` moves
31). The tenth is `a7faca41`, whose stratum membership does not change at all. Strata carry
`hard` only: a `by_driver` bucket exposes exactly `n`, `base_rate_correct`, `verdict_err`
and `hard`, never `parametric` or `soft`. **[M]** That movement is arithmetic over a
changed stratum membership, not a changed belief. No belief scalar moves **in these ten
gold-backed schema-8 exports**: their run-level belief blocks and all four per-statement
belief scalars (`belief`, `belief_hard`, `belief_parametric` and `belief_soft`) are
unchanged. The run-level `tiers.stmt.verdict_err` block, a sibling of `arms` and
`stratified`, does move; correction 1's headline instance is the medpsy F1 .7658 → .7868
in §2.2's table, restated in §2.3. **[M]**

The scope qualifier is load-bearing, not decoration. Correction 1's batch is TWELVE
exports: these ten plus `6aeedd3b76c74f06817b44353c8e91a8` and `rasmachine_belief`, both
of which are `"schema_version": 2` in their `export_meta.json`. In those two the four
belief scalars do not exist before regeneration and are CREATED for all 8,716 statements
each, so "unchanged" cannot describe them and the schema 2 → 8 migration in §2.4 governs
instead. Verified on disk: reading `data/exports/6aeedd3b76c74f06817b44353c8e91a8/`
`per_statement.json` (n=8,716) returns none of the four keys. **[M]**

What moves is exactly, **in the ten schema-8 exports**: `tiers.stmt.stratified.by_driver`,
`tiers.stmt.verdict_err` and the `verdict_err` sub-blocks of its sibling strata,
`belief_verdict_statement` and `coherence_summary` in `per_statement.json`, and
`generated_date`. The two schema-2 exports move strictly more than this list — see §2.4.

### 2.2 The twelve exports, with pre-registered numbers

Re-derived today from each export's `per_evidence.jsonl`, and reproducing §7.1(b) exactly:
**12 exports, 4,997 rows**. The `rows` column counts low-or-absent-confidence non-
deterministic `incorrect` rows — raw rows, which is the unit §7.1(b)'s 4,997 is in. The
"976 of 8,716" headline is **statements**, not rows.

Of `data/exports/`'s 53 export directories, these 12 are the ones that carry such rows.
The other 41 are untouched by regeneration for this reason.

**Ten gold-backed exports.** Nine of them move `by_driver`; the tenth
(`a7faca4104a7466e860e08f4a07d14d1`) does not, per §0(e). Every figure below was measured
by regenerating the export into scratch and diffing against the shipped file. **[M]**

| export | model / gold | rows | `none` | `llm` | stmt verdicts moved | `verdict_err` F1 |
|---|---|---|---|---|---|---|
| `923e4d3d65174b30ae3ded7de393a8ae` | medpsy-4b · external_curator_v1 | 18 | 250 → 239 | 195 → 206 | 11 | .7658 → .7868 |
| `e3aec51aec2f4eb7ae254be7944b9679` | gpt-oss-20b · representative_403 | 13 | 164 → 151 | 238 → 251 | 13 | .7800 → .8018 |
| `a7faca4104a7466e860e08f4a07d14d1` | medpsy-4b · rasmachine_v1 | 12 | 28 → 28 | 31 → 31 | 11 | .8750 → .8750 |
| `ae4e2f3b55ad4d2ebc2c2bb534e881ba` | medpsy-4b · eval_curation_v1 | 11 | 538 → 533 | 360 → 365 | 5 | .7759 → .7785 |
| `4bab7d96407a4aa28e367e760510b93f` | gpt-oss-20b · external_curator_v1 | 5 | 212 → 208 | 227 → 231 | 4 | .8059 → .8075 |
| `99b0e2f6f7794bd28b8b74f1c9da79fa` | gemma-4-e2b · representative_403 | 4 | 174 → 170 | 229 → 233 | 4 | .7021 → .7094 |
| `e63a7511665d4ad6b28e16b1aff83b4b` | qwen3-235b · representative_403 | 3 | 175 → 172 | 228 → 231 | 3 | .7685 → .7770 |
| `b32124ee69064efd97327e34bc58d9cf` | qwen3-235b · external_curator_v1 | 2 | 223 → 222 | 222 → 223 | 1 | .7856 → .7881 |
| `e055e3f532f34ce989dc223346f93f0a` | gemma-4-e2b · external_curator_v1 | 2 | 225 → 223 | 220 → 222 | 2 | .7164 → .7219 |
| `7ad60b916b274e3680cf4cf08354f327` | nemotron-super-120b · representative_403 | 1 | 145 → 144 | 258 → 259 | 1 | .7922 → .7948 |

The `deterministic` count is unchanged in every row — `923e4d3d` 19, `ae4e2f3b` 15,
`4bab7d96` 19, `b32124ee` 19, `e055e3f5` 19, the five of the ten gold-backed exports that
carry the bucket at all — which is the point: the rule change touches only the LLM-reject
path. **[M]** The two medpsy F1s reproduce §7.1(a) — .7658 → .7868 and .7759 → .7785. The
first agrees to three places, not four: the regenerated value is 0.7868131868, against
§7.1(a)'s .7869 rounding in the head-to-head artifact. **[M]** And the `none` moves
250 → 239 and 538 → 533 are the same statements §7.1(a) counted. That is what ties (a) and
(b) together.

Two notes on the table. `a7faca41`'s eleven moved verdicts are all outside its 59-statement
gold-covered stratum, which is why its `by_driver` and `verdict_err` are flat while its
`per_statement.json` moves; its twelfth newly-credited row lands on a statement that
already had a credited rejection, so only the counter increments. And where "stmt verdicts
moved" is smaller than `rows`, the extra rows sit on statements that already routed to
`review`.

**Two schema-2 exports, which do not have a `by_driver` block to move.**

| export | model | rows | statements affected |
|---|---|---|---|
| `6aeedd3b76c74f06817b44353c8e91a8` | remote-medpsy-4b | 4,807 | 976 of 8,716 |
| `rasmachine_belief` | remote-gemma-4-26b | 119 | 23 of 8,641 with a defined belief (8,716 total) |

These two are `schema_version: 2`, carry **no `metrics.json` at all**, and their corpus
`latest_statements_rasmachine.json` has no gold mapping in `scripts/reexport_runs.py`.
Regenerating them is a schema 2 → 8 upgrade that **adds** `metrics.json` rather than
relabelling a stratum — and the added file reports `tiers.ev` and `tiers.stmt` both
`unavailable`, reason *no gold baked for this run*. **[M]** So the 976 and the 23 are
statement-grain censuses of the rule change, not movements of any exported block, present
or future. They are the largest counts on the list and the least observable.

Measured, so the operator knows what the upgrade actually does to those two:
`per_evidence.jsonl` keeps all 47,434 rows and every pre-existing key value-identical
(including `our_score` and `verdict`), gaining six: `cost_status`, `cost_usd`,
`input_tokens`, `n_calls`, `output_tokens`, `reasoning_trace`. `per_statement.json` gains
`belief`, `belief_hard`, `belief_parametric`, `belief_soft`, `belief_verdict_statement`,
`coherence_summary` and `gold_statement` for all 8,716 statements — fields that did not
exist in schema 2, so there is nothing to compare them against. `export_meta.json` gains
cost, model metadata, provenance digests and a calibration block; `rasmachine_belief`
picks up an *available* soft profile (gemma-4-26b is fitted), `6aeedd3b` stays
*unavailable* (medpsy is not). One unrelated drift to expect: `our_mean_score` changes by
0.001 on 48 of `6aeedd3b`'s 8,716 statements, a schema-2-to-8 aggregation difference, not
a consequence of the credit rule. **[M]**

### 2.3 The commands

**The `by_driver` relabel and the medpsy F1 move are one regeneration, not two.** The first
line of the batch invocation below, `data/results/external_curator_v1_medpsy-remote.jsonl`,
**is** export `923e4d3d`: the same pass that moves its `by_driver` 250 → 239 / 195 → 206
moves its `tiers.stmt.verdict_err.f1` from .7658 to .7868. The same holds for `ae4e2f3b`
(eval_curation_v1 medpsy, .7759 → .7785). One root cause sits under both —
`git show a10df62 -- src/indra_belief/statement_belief.py` shows the deletion of the module
constant `_CREDIBLE_LLM_CONF`, named undotted and historically because the symbol no longer
exists — and one code path, so one invocation does both. The operator consequence, stated
so nobody is surprised by it after the fact: §7.1(a) is currently *reported, deliberately
not applied*, and applying correction 1 applies it. There is no invocation that relabels
`by_driver` without moving those two medpsy numbers.

`scripts/reexport_runs.py` is the canonical regenerator and takes explicit run paths. It
resolves each corpus from the run's sibling `.meta.json` `input` field and maps that
corpus to its gold through its own `CORPUS_GOLD` table. Verified today: all 12 run files
and their `.meta.json` exist, every `.meta.json` `input` matches the export's own
`export_meta.json` `generated_from.corpus`, every declared gold exists, and the run_id each
run derives to equals the directory it already occupies — for eleven of the twelve. **[M]**

**Snapshot first.** Re-export overwrites in place. Copy the 12 directories to a scratch
path outside `data/` (a plain `cp -R`) before running anything; §2.4 needs the pre-image.

Eleven of the twelve are one invocation:

```
.venv/bin/python scripts/reexport_runs.py \
  data/results/external_curator_v1_medpsy-remote.jsonl \
  data/results/external_curator_v1_bedrock-gpt-oss-20b.jsonl \
  data/results/external_curator_v1_bedrock-qwen3-235b.jsonl \
  data/results/external_curator_v1_bedrock-gemma-4-e2b.jsonl \
  data/results/representative_indra_expanded_403_20260717_bedrock-gpt-oss-20b.jsonl \
  data/results/representative_indra_expanded_403_20260717_bedrock-gemma-4-e2b.jsonl \
  data/results/representative_indra_expanded_403_20260717_bedrock-qwen3-235b.jsonl \
  data/results/representative_indra_expanded_403_20260717_bedrock-nemotron-super-120b.jsonl \
  data/results/rasmachine_v1_medpsy.jsonl \
  data/results/eval_curation_v1_medpsy.jsonl \
  data/results/rasmachine_mono_medpsy_remote_direct.jsonl
```

The twelfth needs an explicit destination:

```
.venv/bin/python -m indra_belief.results \
  data/results/rasmachine_mono_gemma_remote_direct.jsonl \
  --corpus data/corpora/latest_statements_rasmachine.json \
  --out data/exports/rasmachine_belief
```

**Trap.** `rasmachine_belief` is a *named* directory whose derived run_id is
`d004e3970856484689531396132ca055`. **[M]** Putting it in the batch invocation would write
a brand-new directory under that hash and orphan the old one — and because the viewer
discovers runs by globbing for `export_meta.json` (`viewer/src/lib/data/runs.ts`), both
would then appear in the run picker as separate runs of the same model. The `--out` form
above overwrites in place. Its `export_meta.json` will still carry the hash as `run_id`,
which is already true of the shipped file.

### 2.4 Verification: prove only the intended keys moved

**Do not use a byte diff, and do not accept one as a pass.** `src/indra_belief/results.py`
stamps `generated_date` from `datetime.date.today()` into both `metrics.json` and
`export_meta.json`, so a regenerated file is never byte-equal to its predecessor and a
"clean byte diff" acceptance criterion is unfalsifiable. **[V]** The check is key-scoped.

For each of the ten gold-backed exports, compare snapshot against regenerated and assert:

1. `export_meta.json` changed keys are exactly `{generated_date}`.
2. `metrics.json` changed leaf keys are confined to `tiers.stmt.stratified`,
   `tiers.stmt.verdict_err`, and `generated_date`. Measured counts of changed leaf keys,
   for reference: 121, 98, 1, 117, 111, 92, 75, 66, 71, 63 in the table's row order.
3. `tiers.stmt.stratified.by_driver`: the `none` count fell by exactly the table's amount
   and `llm` rose by the same amount, with `deterministic` unchanged. For
   `a7faca4104a7466e860e08f4a07d14d1`, the assertion is that all three are unchanged.
4. Every belief scalar is float-equal: `tiers.stmt.arms.hard`, `.parametric`, `.soft`,
   `tiers.ev.arms.score`, and `tiers.stmt.n` / `base_rate_correct`. (This belief-scalar
   assertion is **run-level**. The `by_driver` strata's own `hard` blocks move by
   construction — see §2.1 — which item 2's three-prefix bound already admits; that is
   what makes the two agree.)
5. `per_evidence.jsonl` is byte-identical.
6. `per_statement.json` changed keys are exactly `{belief_verdict_statement,
   coherence_summary}`, and the count of rows whose `belief_verdict_statement` changed
   equals the table's "stmt verdicts moved".

For the two schema-2 exports the assertion is different and must be written differently.
They gain `metrics.json` and move to `schema_version: 8`, so there is no before-state for
most of what appears. The available checks are: `per_evidence.jsonl` still has 47,434 rows;
every key that existed before is value-identical, in particular `our_score` and `verdict`;
`per_statement.json` still has 8,716 entries; and the added `metrics.json` reports both
tiers `unavailable`. Nothing stronger is available, and claiming otherwise would be
claiming to have compared fields that did not exist.

Finally, assert the **other 41 export directories are untouched** — mtime and digest
unchanged on every file. That is what makes "12 of 53" an observable claim rather than a
description of intent.

---

## 3. What is permanently frozen

`data/benchmark/holdout_cc.jsonl` **no longer exists.** `data/results/holdout_cc_gemma.jsonl`
and its `.meta.json` do, and there is no `data/exports/` directory for that run — so
`holdout_cc_gemma` is not one of the 53 and cannot be regenerated into one. **[M]** Its
§7.1(a) zero rests on a **row census**, not on a re-run, and it can never be upgraded to a
re-run: the gold the census was taken against is gone. Anyone planning work on that holdout
should know the number's provenance before leaning on it.

One nuance keeps that statement true rather than alarming.
`scripts/calibration_ship_gate.py` defaults `--test-gold` to
`data/results/cc_holdout_cc/holdout_cc.jsonl`, which **is** present. **[M]** The ship gate
still runs. What is permanently frozen is the benchmark-gold-backed export path for
`holdout_cc_gemma`, and any re-derivation that needs `data/benchmark/holdout_cc.jsonl`.

And once more, because it governs how much either correction is allowed to matter:
`glm_5_sensitivity` is **not a published bundle**. The five published bundles are the
directories under `data/comparison/models/` named in §0(b). Neither correction moves a
single number in any of them, nor in the paper reproduction figures, nor in any belief
scalar anywhere.
