# Verdict-only re-run of the INDRA paper comparison — runbook

> **Status: run COMPLETE 2026-07-31.** All four arms carry `.COMPLETE` in
> `data/comparison_verdict_only/supervisor/` and every `.state.json` reads
> `phase: complete` (latest 2026-07-31T15:47:15Z). The procedures below are the
> record of how it was run; the self-heal crontab from §Reboot self-heal is
> still installed — remove it with `crontab -r`.

Re-scores the SAME corpus as the 2026-07-20..23 Bedrock run with the scorer's
reasoning removed **at both levels**: the provider's chain-of-thought, and the
prompt scaffolding that made the model deliberate in its answer.

- **Live plan: `data/comparison_verdict_only/run_plan.json`** (tracked), four
  arms, substrate `data/comparison_verdict_only/grounding_replay/`.
- Corpus: `unique_exact_pairs_primary`, 33,361 evidence executions over 1,689
  statements. Routes are carried across unchanged (29,640 plain / 2,670 tool /
  1,051 deterministic), so rows pair one-to-one with the thinking run on
  `(stmt_i, evidence_i)`.

### Why "reasoning_effort: none" was not enough

The first attempt (`data/comparison_noreason/`, **superseded**, ~$0.02 spent)
turned off only the provider's CoT. Its very first scored row came back:

```json
{"relation_check": "direct statement of the relation licenses acceptance",
 "support": "analyzed the effects of WT1-ZNF224 interaction on...",
 "objection": null, "verdict": "correct", "confidence": "high"}
```

That is chain-of-thought — an acceptance check, a warrant, a defeater — emitted
before the verdict, because the frozen substrate carried
`"mono_variant": "disconfirm_relnature_rf"`. On top of it, 442 of the first
1,455 calls were a *second* deliberation call (`relation_nature`) for `[Complex]`
claims. The comparison runner never reads `MONO_VARIANT`
(`grep -rn MONO_VARIANT src/indra_belief/comparison/` is empty) — it hydrates
prompts from a content-addressed store and digest-checks them. The scaffolding
was frozen into the substrate, and no generator for one existed.

`scripts/build_verdict_only_replay.py` is that generator. It DERIVES rather than
regenerates: the user message is rebuilt by `prepared_execution.prepare_from_replay_row` from structured
row fields that do not change, so exactly four things do — the main system
prompt, the few-shot prefixes, `main_prompt_base_sha256`, and the removal of the
relation-nature call. Rules 1-7 of the prompt are reused byte-identically from
`DISCONFIRM_SYSTEM_PROMPT` via the same `.split("HOW TO DECIDE")[0]` seam the
reasoning-first variant uses, so the scaffolding is the only thing that varies.

The model now emits, verified live on all four arms at 14 output tokens:

```json
{"verdict": "correct", "confidence": "high"}
```

## Arms

| action | served model | provider model | stage cap |
|---|---|---|---|
| `gemma_31b_vo_primary` | `bedrock-gemma-4-31b-noreason` | `google.gemma-4-31b` | $40 |
| `gemma_26b_vo_primary` | `bedrock-gemma-4-26b-noreason` | `google.gemma-4-26b-a4b` | $40 |
| `e2b_vo_primary` | `bedrock-gemma-4-e2b-noreason` | `google.gemma-4-e2b` | $3.63 |
| `glm_5_vo_primary` | `bedrock-glm-5-noreason` | `zai.glm-5` | $310 |

Four arms, matching the shipped comparison's four LLM arms — the first attempt
had only three and left `bedrock-gemma-4-e2b` at `reasoning_effort: "high"`.

Each `-noreason` registry entry differs from its thinking sibling in exactly one
field: `reasoning_effort` `"high"` → `"none"`. Backend, endpoint pins, TLS
bundle, byte bounds, `model_id`, `max_tokens` are copied verbatim. All four keep
the `*_raw` paid-lane transports, because only those make the spend guard record
a canonical wire body — the one artifact that can prove reasoning was off.

## Rebuilding the substrate

```
PYTHONPATH=src python scripts/build_verdict_only_replay.py               # build + verify
PYTHONPATH=src python scripts/build_verdict_only_replay.py --verify-only
```

Verification loads it through the real `ReplayIndex`, which re-hydrates and
digest-checks every one of the 32,310 callable prompts and re-derives every
deterministic rejection. The 57 MB substrate is gitignored on purpose: the
generator is deterministic, so the script is the durable artifact, not its output.

## Cost

The thinking run, from its own ledgers:

| arm | provider-measured (lower) | cap-accounted (upper) |
|---|---|---|
| gemma-26B | $26.37 | $35.39 |
| gemma-31B | $22.93 | $31.69 |
| GLM-5 | $203.35 | $262.25 |
| **total** | **$252.64** | **$329.33** |

The gap is 2,473 failed calls settled at their reservation with usage unknown
(`accounting_basis: conservative_reserved_maximum`). Whether AWS billed them is
not recoverable from the artifacts, so quote both bounds — never "the run cost
$252".

Reasoning-off forecast, holding input tokens fixed and collapsing output to the
measured visible fraction: ~$18 / ~$19 / ~$138, **~$176 total**. GLM-5 stays
expensive because $127 of its bill is *input* tokens, which disabling thinking
cannot touch. Stage caps are kept at the thinking run's proven 40/40/310 so a
cap breach — which is terminal under the frozen plan and needs a reviewed
amendment — stays unlikely.

## Preflight

```
PYTHONPATH=src python scripts/verify_reasoning_disabled.py --live
```

Static mode is free and asserts the wire shape (Gemma omits the `reasoning` key
entirely; GLM-5 sends `"reasoning_effort":"none"`), plus that each arm still
matches its sibling on everything else. `--live` spends ~$0.01 to confirm the
provider does not deliberate.

**This check is not optional and cannot be replaced by the ledger.** Across the
thinking run, Gemma reported `reasoning_tokens = 0` on all 99,180 traces while
returning real CoT, and GLM-5 omitted the field entirely. A run can be billed
for hidden deliberation and the artifacts still look clean.

Two limits to know about the `--live` half:

- It exercises a toy prompt, not the frozen scorer prompt, so it proves the
  provider does not deliberate — not that a non-thinking arm still emits a
  parseable verdict on real evidence. Check the first rows of each arm's
  `attempts.jsonl` for `verdict` in `{correct, incorrect}` shortly after launch;
  a run that parses badly wedges sources rather than failing loudly.
- The "extra reasoning chars" guard is live only for the Gemma arms. Our chat
  transport reads `message.reasoning_content` while mantle returns GLM-5's CoT
  under `message.reasoning`, so `raw_text` never carries it and that check is
  structurally dead for GLM-5. The chars-per-output-token density check is the
  only live CoT guard on that arm.

Also confirm the plan loads and the corpus size matches:

```
PYTHONPATH=src python -m indra_belief.comparison status \
    --plan data/comparison_verdict_only/run_plan.json
```

Expect four `pending` actions at `total: 33361`. `status` never opens a
ledger, so it is safe and cheap to re-run at any time.

## Launch

**`export`, not a command prefix.** The monitor heals a dead fleet by
re-launching supervisors, and they inherit the monitor's environment. A
command-scoped prefix (`VAR=x cmd`) is not exported, so a healed supervisor
would silently revert to caffeinate-on and a 4-tick kill window — doubling the
outage window on exactly the unattended path this is for. Both lines must come
from the same shell, after the export:

```
export SUPERVISOR_CAFFEINATE=0 NETFAIL_KILL_TICKS=2 STAGGER_SECONDS=60

bash scripts/supervise_comparison_all.sh start data/comparison_verdict_only/run_plan.json

nohup bash scripts/monitor_comparison_fleet.sh data/comparison_verdict_only/run_plan.json \
  >> data/comparison_verdict_only/supervisor/fleet_monitor.log 2>&1 & disown
```

### Parallelism

All four arms run concurrently at `workers: 8` each — **32 concurrent
executions**, the most the contract permits (`contracts.MAX_WORKERS = 8`, and
`workers > MAX_WORKERS` fails plan load). Raising it further means editing a
hardened contract, not configuration.

`STAGGER_SECONDS` (default 180) only delays the START of each arm by its
position in the list; it is not a throughput limit, and costs
`(n-1) x STAGGER_SECONDS` once. It buys separation of the four preflights —
each streams every action's attempts file — and of the first provider burst.
`STAGGER_SECONDS=0` starts everything at once; expect more `transient prepare
failure` restarts if you do. One was already observed at 180s and self-healed on
the next invocation.

Watch for provider throttling at 32 concurrent: the thinking run's own amendment
records eighteen concurrent as the measured no-throttle peak. A 429 classes as
`transport_or_server` and burns attempt ordinals against the contract's hard cap
of ten, so if `<arm>.err.log` fills with 429s, drop `workers` rather than letting
an arm reach `attempts_exhausted` (which is terminal and needs a plan amendment).

The monitor logs the values it will propagate on its first line — check that it
reads `SUPERVISOR_CAFFEINATE=0 NETFAIL_KILL_TICKS=2` and not `<unset, arm
default ...>`. Both knobs are validated; a typo exits 64 rather than running
with a silently wrong kill window.

Arms are staggered 0/180/360s by position in the derived list.

## Stopping

Order matters once the self-heal is installed:

```
crontab -r                                     # FIRST, or cron restarts within 10 min
pkill -f monitor_comparison_fleet.sh           # else the monitor heals it back
bash scripts/supervise_comparison_all.sh stop data/comparison_verdict_only/run_plan.json
```

`supervise_comparison_all.sh stop` alone does **not** stop the run: it kills
supervisors, and both cron and the monitor exist to bring them back. Stopping is
safe at any point — ledgers are append-only and resume re-derives.

## Sleeping the machine

Sleep is safe and free to resume. What it costs is *attempt ordinals*, and the
run-plan contract caps `max_attempts` at ten, so the budget is genuinely small:

- The deadline is **not** a risk. `PreparedRun.deadline` is computed from
  `time.monotonic()`, which is `mach_absolute_time()` on this machine and does
  not advance while asleep. Sleep time is free against `deadline_seconds`, and
  the budget is re-taken per supervisor invocation anyway.
- The risk is the retry loop. Every in-flight socket dies on sleep; on wake
  those failures classify as `transport_or_server` and retry in-process. This
  run sets `retry_backoff_seconds: 15.0` (the thinking run used 2.0), so
  reaching ordinal 4 takes 15+30+60 = 105s instead of 14s, and
  `NETFAIL_KILL_TICKS=2` stops the runner after ~60s of failed probes. Roughly
  2-3 ordinals burn per wake instead of ~6.
- Ordinals are cumulative across restarts, and only the `workers` sources in
  flight at sleep time are affected (8 per arm), so a given source is unlikely to be
  hit twice across a run of 33,361.

If an arm does exhaust ten attempts on one source it writes
`<arm>.ALERT` with reason `stuck_under_plan_bounds` and stops. That needs a
reviewed plan amendment, not a restart.

## Reboot self-heal

`nohup`'d supervisors survive terminal exit and sleep, but **not** reboot. The
LaunchAgent route is TCC-blocked on `~/Documents` (the failures are recorded in
`data/comparison/supervisor/*.launchd.log`). cron was probed on 2026-07-31 and
**is not blocked** — it resolved the repo working directory and read repo files
as user `noot`. Install:

```
( crontab -l 2>/dev/null; echo '*/10 * * * * SUPERVISOR_CAFFEINATE=0 NETFAIL_KILL_TICKS=2 STAGGER_SECONDS=60 /bin/bash /Users/noot/Documents/indra-belief-model/scripts/supervise_comparison_all.sh start data/comparison_verdict_only/run_plan.json >> /Users/noot/Documents/indra-belief-model/data/comparison_verdict_only/supervisor/cron.log 2>&1' ) | crontab -
```

`start` is idempotent: it skips arms whose supervisor is already running, and
each supervisor additionally self-guards via `shlock` plus COMPLETE/ALERT
markers and an adopt-wait for orphaned runners.

**Remove it when the run finishes** (`crontab -r`), or it will keep trying to
start a fleet forever.

## Resume semantics — why sleeping costs nothing

The spend WAL is the source of truth, not the raw output. On start, `_recover`
replays the action's ledger *before* readiness is emitted — i.e. before the
bearer token is even read, so with zero provider calls — and re-appends each
committed attempt's stored `raw_row` into `attempts.jsonl`, digest-checked. So
`attempts.jsonl` is DERIVED: deleting it regenerates it for free. If
regeneration cannot reproduce it, `_reconcile` fails closed with "raw output and
spend WAL attempts are not bijective" rather than silently re-spending.

The irreplaceable files are `runs/<arm>/spend.ndjson`. Protect those.

Note the standing cost: readiness calls `load_resume` for **every** action in
the plan on every invocation, reading and re-validating each output file whole.
That is why this run has its own plan — pointing it at the thinking run's plan
would re-read ~5.5 GB on every supervisor restart.

## After the run — USE THESE PATHS, never the defaults

Every assembly command's default output path is the **shipped thinking run's**
artifact, and one of the three writers has no overwrite guard: `materialize` and
`metrics` refuse to clobber without `--force`, but `report` goes straight to an
unconditional atomic replace. All five defaults exist from the thinking run, all
five are gitignored and untracked, and
`data/results/indra_belief_comparison_metrics.json` is the live input to the
`/paper` viewer section and to `tests/test_viewer_belief_comparison_publication.py`.
Running `report` with default flags would silently destroy the shipped report;
`--force` on the others would silently repoint `/paper` at the reasoning-off run
while every label still says otherwise.

Back up the five thinking-run artifacts first, then pass explicit paths:

```
python -m indra_belief.comparison model-bundle \
    --inputs data/comparison_verdict_only/inputs.json \
    --plan data/comparison_verdict_only/run_plan.json --action <arm>  # x4

python -m indra_belief.comparison materialize \
    --inputs data/comparison_verdict_only/inputs.json \
    --output data/results/indra_belief_comparison_noreason_spec.json

python -m indra_belief.comparison metrics \
    --spec data/results/indra_belief_comparison_noreason_spec.json \
    --output data/results/indra_belief_comparison_noreason_metrics.json

python -m indra_belief.comparison report \
    --metrics data/results/indra_belief_comparison_noreason_metrics.json \
    --markdown reports/indra_belief_comparison_noreason.md \
    --html reports/indra_belief_comparison_noreason.html \
    --manifest reports/indra_belief_comparison_noreason_manifest.json
```

Never `--force`. `model-bundle` resolves `aggregation.json` and `pricing.json`
from the **inputs file's own parent directory**, which is why both were copied
byte-for-byte into `data/comparison_verdict_only/` — copied rather than re-derived so
the cost tariffs stay identical and the cost-Pareto comparison remains valid.

## Aggregation is hard-gate on BOTH sides — keep it that way

The comparison pipeline does not consult the calibration registry at all:
`grep -rn calibration_for src/indra_belief/comparison/` is empty. The reader
profile is read exclusively from `aggregation.json`, which declares
`"aggregation": "indra_default_hard_gate"` and `"reader_profile": null`, and the
shipped thinking bundle confirms that is what actually ran — the gemma-26B
manifest records `implementation.notes.reader_profile: null`.

So every arm on both runs rolls evidence up with the **hard gate**, and all
three pairs are clean at statement grain as well as evidence grain. This holds
because `aggregation.json` was **copied byte-for-byte** from the thinking run.

The one thing that would break it: setting a non-null `reader_profile` in
`data/comparison_verdict_only/aggregation.json`. That would put a calibrated reader
on the reasoning-off side against a hard-gate thinking side and manufacture an
aggregation confound out of nothing. Don't.

(Separately, and not relevant here: `calibration_for` in the *production*
scorer path does resolve a profile for `bedrock-gemma-4-26b` under the
reasoning-first prompt and resolves none for the `-noreason` names. That
asymmetry is real but lives outside this pipeline.)

## Known limitation — the head-to-head cannot be one artifact

`inputs.json` requires `provider_model_id` to be unique across `llm_models`, and
each reasoning-off arm serves the same provider model as its thinking sibling.
So the two runs cannot be declared in one inputs file, and `metrics`/`report`
cannot currently assemble a single thinking-vs-non-thinking comparison. The
per-arm bundles are directly comparable (identical corpus, one-to-one row
pairing); joining them into one shipped artifact would require relaxing that
uniqueness rule, which is a frozen-contract change and needs its own review.

## Baselines to compare against

Thinking-run verdict counts, all out of 33,361:

| arm | correct | incorrect |
|---|---|---|
| gemma-31B | 25,437 | 7,924 |
| gemma-26B | 24,611 | 8,750 |
| GLM-5 | 23,935 | 9,426 |

## Open defect found while preparing this

GLM-5's chain-of-thought was being **discarded, not skipped**. Mantle returns it
under `message.reasoning`; `parse_bedrock_chat_payload` reads only
`message.reasoning_content`. So every GLM-5 monolithic call in the thinking run
recorded `reasoning_trace.status="none"` with empty `free_cot` while the
response body actually carried ~2.2 kB of CoT — verified by base64-decoding a
retained `response_body_preimage_b64`. Consequences: the thinking run paid for
GLM-5 reasoning it never stored, and that arm's recorded traces cannot serve as
the "before" evidence in a reasoning on/off comparison. This does not affect the
reasoning-off run (there is no CoT to capture) and was deliberately left
unchanged so the transport is byte-stable across both runs.
