#!/bin/bash
# Supervise one primary comparison arm to completion.
#
# Restarts the runner across deadline exits, transient crashes, and network
# outages; kills a live runner after ~2 minutes of continuous network failure
# so in-flight sources lose at most one attempt ordinal instead of retrying
# to exhaustion; refuses to hot-loop when an arm is terminally stuck under
# the frozen plan's retry bounds (that requires a reviewed plan amendment,
# not a restart). Designed for macOS bash 3.2.
#
# Usage: supervise_comparison_arm.sh <action_id> [initial_delay_seconds] [plan_path]
#
# The plan path defaults to the original comparison plan, so the historical
# invocation is unchanged. Passing a third argument supervises a different
# plan; the arm's attempts file is then DERIVED from that plan's own
# actions[].output path rather than assumed, because the supervisor's only
# progress signal is that file growing (a wrong path makes every invocation
# look like zero progress and drives a spurious crash-loop ALERT).
#
# Two knobs are environment-overridable for sleep-tolerant operation:
#   SUPERVISOR_CAFFEINATE   1 (default) holds a no-sleep assertion around the
#                           runner; 0 lets the machine sleep mid-run. Sleeping
#                           is safe — the spend WAL is append-only and resume
#                           re-derives — but every in-flight socket dies on
#                           sleep and each one burns attempt ordinals on wake.
#   NETFAIL_KILL_TICKS      consecutive 30s network-probe failures before the
#                           runner is killed to stop it burning those ordinals.
#                           Default 4 (~120s); lower it when sleeping is
#                           expected, since max_attempts is capped at ten by
#                           the run-plan contract and cannot absorb much.
#
# State (data/comparison/supervisor/):
#   <arm>.supervisor.log   supervisor decisions
#   <arm>.out.log          accumulated runner stdout (readiness + summaries)
#   <arm>.err.log          accumulated runner stderr
#   <arm>.state.json       heartbeat: phase, invocation, crash counters
#   <arm>.COMPLETE         terminal marker: action complete (status-verified)
#   <arm>.SETTLED          terminal marker: nothing left to schedule, but N
#                          sources carry no verdict. Distinct from COMPLETE on
#                          purpose — the arm is finished and must not be
#                          restarted, and the holes must be seen before the
#                          bundle is materialized against them.
#   <arm>.ALERT            terminal marker: operator intervention required
set -u

REPO="/Users/noot/Documents/indra-belief-model"
ARM="${1:?usage: supervise_comparison_arm.sh <action_id> [initial_delay] [plan]}"
STAGGER="${2:-0}"
PLAN="${3:-data/comparison/run_plan.json}"
# Accept an absolute in-repo plan path; every downstream use is repo-relative.
case "$PLAN" in /*) PLAN="${PLAN#"$REPO"/}" ;; esac
SUP="$REPO/$(dirname "$PLAN")/supervisor"
PROBE_HOST="bedrock-mantle.us-east-1.api.aws"
PY="$REPO/.venv/bin/python"
CRASH_ALERT_LIMIT=8
TRANSIENT_ALERT_LIMIT=40
NETFAIL_KILL_TICKS="${NETFAIL_KILL_TICKS:-4}"
CAFFEINATE="${SUPERVISOR_CAFFEINATE:-1}"

# Validate the env knobs rather than trusting them: this value gates a kill, and
# a typo is silent. 0 or a non-number would make the kill fire on a HEALTHY
# network every tick, restarting the runner forever and burning attempt ordinals
# on a run that is otherwise fine.
case "$NETFAIL_KILL_TICKS" in
    ''|*[!0-9]*) echo "NETFAIL_KILL_TICKS must be a positive integer, got '$NETFAIL_KILL_TICKS'" >&2; exit 64 ;;
    0) echo "NETFAIL_KILL_TICKS must be >= 1 (0 kills on a healthy network)" >&2; exit 64 ;;
esac
case "$CAFFEINATE" in
    0|1) ;;
    *) echo "SUPERVISOR_CAFFEINATE must be 0 or 1, got '$CAFFEINATE'" >&2; exit 64 ;;
esac

mkdir -p "$SUP"

# Derive this arm's attempts file from the plan itself. Failing loudly beats
# defaulting: a silently wrong path reports zero progress forever.
ATTEMPTS=$( (cd "$REPO" && "$PY" - "$PLAN" "$ARM" <<'PYEOF'
import json, sys
plan_path, arm = sys.argv[1], sys.argv[2]
with open(plan_path) as handle:
    plan = json.load(handle)
for action in plan["actions"]:
    if action["id"] == arm:
        print(action["output"]["path"])
        break
else:
    raise SystemExit(f"action {arm!r} is not in {plan_path}")
PYEOF
) ) || { echo "cannot resolve attempts path for $ARM in $PLAN" >&2; exit 64; }
ATTEMPTS="$REPO/$ATTEMPTS"
mkdir -p "$(dirname "$ATTEMPTS")"
LOG="$SUP/$ARM.supervisor.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [$ARM] $*" >> "$LOG"; }

# Interruptible sleep: bash defers traps until the current foreground command
# returns, so a bare long sleep would delay TERM handling by minutes.
snooze() { sleep "$1" & wait $!; }

attempts_size() { stat -f %z "$ATTEMPTS" 2>/dev/null || echo 0; }

state() {
    printf '{"ts":"%s","arm":"%s","phase":"%s","invocation":%d,"crashes":%d,"transients":%d,"attempts_bytes":%s}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ARM" "$1" "$INV" "$CRASH" "$TRANSIENT" "$(attempts_size)" \
        > "$SUP/$ARM.state.json"
}

alert() {
    printf '{"ts":"%s","arm":"%s","reason":"%s","detail":%s}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ARM" "$1" "$2" > "$SUP/$ARM.ALERT"
    log "ALERT ($1): $2"
}

probe() { /usr/bin/nc -z -G 5 "$PROBE_HOST" 443 >/dev/null 2>&1; }

wait_net() {
    while ! probe; do
        state "waiting_for_network"
        log "network probe to $PROBE_HOST:443 failed; retrying in 60s"
        snooze 60
    done
}

runner_alive() {
    pgrep -f "indra_belief.comparison (run|_run-child).*--action $ARM" >/dev/null 2>&1
}

kill_runner() {
    [ -n "$RPID" ] && kill "$RPID" 2>/dev/null
    pkill -f "indra_belief.comparison run .*--action $ARM" 2>/dev/null
    pkill -f "indra_belief.comparison _run-child.*--action $ARM" 2>/dev/null
}

# Confirm the arm is really terminal before writing a terminal marker, and say
# WHICH terminal it reached. `settled` is reported separately from `complete`
# because the two must not write the same marker: one means every source carries
# a verdict, the other means some never will.
# Returns 0 complete, 3 settled, 1 definitively not terminal, 2 unverifiable now.
# SETTLED_COUNT carries the quarantined source count for the log line.
SETTLED_COUNT=0
verify_complete() {
    TRIES=0
    while [ "$TRIES" -lt 5 ]; do
        TRIES=$((TRIES + 1))
        RESULT=$( (cd "$REPO" && env PYTHONPATH=src "$PY" -m indra_belief.comparison status --plan "$PLAN") 2>/dev/null | \
            "$PY" -c "
import json, sys
try:
    value = json.load(sys.stdin)
except ValueError:
    print('error'); raise SystemExit(0)
for action in value.get('actions', []):
    if action.get('action_id') == '$ARM':
        status = action.get('status')
        if status in ('complete', 'settled'):
            print('%s %d' % (status, action.get('settled', 0)))
        else:
            print('partial 0')
        break
else:
    print('error')
" )
        SETTLED_COUNT=${RESULT#* }
        case "${RESULT%% *}" in
            complete) return 0 ;;
            settled) return 3 ;;
            partial) return 1 ;;
            *) log "completion verification attempt $TRIES failed (concurrent ledger read?); retrying in 60s"; snooze 60 ;;
        esac
    done
    return 2
}

INV=0
CRASH=0
TRANSIENT=0
RPID=""

[ -f "$SUP/$ARM.COMPLETE" ] && { log "COMPLETE marker present; nothing to do"; exit 0; }
[ -f "$SUP/$ARM.SETTLED" ] && { log "SETTLED marker present; arm is terminal with quarantined sources; nothing to do"; exit 0; }
[ -f "$SUP/$ARM.ALERT" ] && { log "ALERT marker present; refusing to start until it is cleared"; exit 0; }

# Retry the lock through a predecessor's teardown window (shlock steals
# dead-PID locks, so only a live holder blocks us).
LOCK="$SUP/$ARM.pid.lock"
LOCK_TRIES=0
until /usr/bin/shlock -f "$LOCK" -p $$; do
    LOCK_TRIES=$((LOCK_TRIES + 1))
    if [ "$LOCK_TRIES" -ge 6 ]; then
        log "another live supervisor holds $LOCK after ${LOCK_TRIES} tries; exiting"
        exit 0
    fi
    sleep 10
done

cleanup() {
    kill_runner
    rm -f "$LOCK"
}
trap cleanup EXIT
trap 'log "signal received; stopping"; exit 143' TERM INT HUP

# Adopt-wait: never leave an arm unsupervised because an orphaned runner
# from a killed supervisor is still alive. Wait it out, then supervise.
while runner_alive; do
    state "adopted_wait"
    log "pre-existing runner is alive; waiting for it to exit before supervising"
    snooze 60
done

[ "$STAGGER" -gt 0 ] 2>/dev/null && { state "stagger_delay"; log "initial stagger ${STAGGER}s"; snooze "$STAGGER"; }

log "supervisor started (pid $$)"

while :; do
    wait_net
    INV=$((INV + 1))
    OUT="$SUP/$ARM.lastinv.out"
    ERRF="$SUP/$ARM.lastinv.err"
    : > "$OUT"
    : > "$ERRF"
    SIZE_BEFORE=$(attempts_size)
    state "running"
    log "invocation $INV starting (attempts.jsonl ${SIZE_BEFORE} bytes)"

    if [ "$CAFFEINATE" = "1" ]; then
        ( cd "$REPO" && exec env PYTHONPATH=src /usr/bin/caffeinate -i -s \
            "$PY" -m indra_belief.comparison run --plan "$PLAN" --action "$ARM" ) \
            >> "$OUT" 2>> "$ERRF" &
    else
        ( cd "$REPO" && exec env PYTHONPATH=src \
            "$PY" -m indra_belief.comparison run --plan "$PLAN" --action "$ARM" ) \
            >> "$OUT" 2>> "$ERRF" &
    fi
    RPID=$!
    TICK=0
    NETFAIL=0
    NETKILL=0
    while kill -0 "$RPID" 2>/dev/null; do
        snooze 30
        TICK=$((TICK + 1))
        if probe; then
            NETFAIL=0
        else
            NETFAIL=$((NETFAIL + 1))
        fi
        if [ "$NETFAIL" -ge "$NETFAIL_KILL_TICKS" ] && [ "$NETKILL" -eq 0 ]; then
            NETKILL=1
            log "network down for ~$((NETFAIL_KILL_TICKS * 30))s; stopping runner to preserve the attempt budget"
            kill_runner
        fi
        [ $((TICK % 10)) -eq 0 ] && state "running"
    done
    wait "$RPID"
    CODE=$?
    RPID=""
    cat "$OUT" >> "$SUP/$ARM.out.log"
    cat "$ERRF" >> "$SUP/$ARM.err.log"
    SIZE_AFTER=$(attempts_size)
    [ "$SIZE_AFTER" -gt "$SIZE_BEFORE" ] && { CRASH=0; TRANSIENT=0; }

    if [ "$NETKILL" -eq 1 ]; then
        log "invocation $INV stopped by network-outage kill (+$((SIZE_AFTER - SIZE_BEFORE)) bytes); waiting for network"
        continue
    fi

    DECISION=$("$PY" - "$OUT" "$ERRF" <<'PYEOF'
import json, sys
out_path, err_path = sys.argv[1], sys.argv[2]
summary = None
try:
    lines = [l for l in open(out_path, "rb").read().decode("utf-8", "replace").splitlines() if l.strip()]
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "status" in value and value.get("status") != "ready_for_bearer_token":
            summary = value
            break
except OSError:
    pass
if summary is not None:
    status = summary.get("status")
    failure = summary.get("failure") or {}
    kind = failure.get("kind", "")
    # DISPOSITION, not kind. Keying on kind alone mapped all four quarantine
    # kinds to one terminal "stuck" ALERT carrying a single failure, and said a
    # reviewed plan amendment was required. What the operator actually needs is
    # the LIST — how many sources are bad and which — because that is the whole
    # of what quarantine buys against an all-or-nothing corpus: the regime, not
    # a finished arm.
    disposition = failure.get("disposition", "")
    quarantined = summary.get("quarantined", 0)
    detail = json.dumps(json.dumps({
        "failure": failure,
        "quarantined": quarantined,
        "quarantined_sources": summary.get("quarantined_sources", []),
        "quarantined_sources_truncated": summary.get(
            "quarantined_sources_truncated", False),
        "completed_total": summary.get("completed_total"),
        "total": summary.get("total"),
    }))
    if status == "complete":
        print("complete")
    elif status == "settled":
        # Nothing schedulable remains, but some sources have no verdict.
        # Terminal, and NOT the same as complete.
        print("settled " + detail)
    elif status == "spend_cap" or kind == "spend_cap":
        print("spend_cap " + detail)
    elif status == "deadline":
        print("deadline")
    elif disposition == "halt" or disposition == "quarantine" or kind:
        # Everything that stopped an arm short of complete lands here, and the
        # distinction the operator needs is not restartable-vs-not — an arm
        # holding a hole is never restartable, because the bundle requires every
        # pair scored. It is WHY it stopped, and HOW MANY holes it found. The
        # old branch keyed on `kind` alone and called all of this
        # "stuck_under_plan_bounds", which for `invalid_model_output_limit` also
        # claimed a plan amendment would fix it; raising max_attempts does not
        # move the per-source invalid-output cap at all.
        print("stopped " + detail)
    else:
        print("crash")
    raise SystemExit(0)
try:
    tail = open(err_path, "rb").read()[-4000:].decode("utf-8", "replace")
except OSError:
    tail = ""
if ("already complete" in tail
        or "completed during WAL recovery" in tail
        or "is not ready" in tail):
    # The action (or whole plan) finished without a summary line reaching
    # stdout; the shell verifies against `status` before writing COMPLETE.
    print("maybe_complete")
elif ("partial JSONL row" in tail
        or "changed while reading" in tail
        or "lacks a trailing newline" in tail
        or "spend ledger is already in use" in tail):
    print("transient")
else:
    print("crash")
PYEOF
)
    WORD=${DECISION%% *}
    DETAIL=${DECISION#* }
    log "invocation $INV exited code=$CODE decision=$WORD (+$((SIZE_AFTER - SIZE_BEFORE)) bytes)"

    case "$WORD" in
        complete|maybe_complete|settled)
            verify_complete
            VERDICT=$?
            if [ "$VERDICT" -eq 0 ]; then
                date -u +%Y-%m-%dT%H:%M:%SZ > "$SUP/$ARM.COMPLETE"
                state "complete"
                log "action complete (status-verified); supervisor exiting"
                exit 0
            elif [ "$VERDICT" -eq 3 ]; then
                # Terminal, but with holes. Its own marker, so no later reader
                # can mistake it for a clean arm, and so the count is on disk
                # next to the arm it belongs to.
                printf '{"ts":"%s","arm":"%s","quarantined":%s}\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ARM" "$SETTLED_COUNT" \
                    > "$SUP/$ARM.SETTLED"
                state "settled"
                log "action SETTLED (status-verified): nothing left to schedule, $SETTLED_COUNT source(s) carry no verdict; the bundle must record them as exclusions"
                exit 0
            elif [ "$VERDICT" -eq 1 ]; then
                log "runner claimed completion but status says partial; treating as crash"
                CRASH=$((CRASH + 1))
                if [ "$CRASH" -ge "$CRASH_ALERT_LIMIT" ]; then
                    alert "crash_loop" "\"$CRASH consecutive completion-claim/status disagreements\""
                    state "alert_crash_loop"
                    exit 0
                fi
                snooze 60
            else
                log "completion unverifiable under concurrent writes; will re-derive next invocation"
                snooze 120
            fi
            ;;
        spend_cap)
            alert "spend_cap" "$DETAIL"
            state "alert_spend_cap"
            exit 0
            ;;
        stopped)
            # The arm stopped short of complete. Restarting cannot help: it
            # either holds a hole — in which case the runner will not dispatch
            # it again, because a bundle needs every pair scored — or it halted
            # on a failure that is a statement about the run. Either way a human
            # fixes the cause. The ALERT carries the quarantine COUNT and the
            # identities, which is the diagnostic the budget was spent to buy.
            alert "stopped_before_complete" "$DETAIL"
            state "alert_stopped"
            log "the arm stopped before complete; the ALERT carries the quarantined-source list and the regime. Note: raising max_attempts does NOT move the per-source invalid-output cap"
            exit 0
            ;;
        deadline)
            log "action deadline exit; restarting"
            snooze 15
            ;;
        transient)
            TRANSIENT=$((TRANSIENT + 1))
            if [ "$TRANSIENT" -ge "$TRANSIENT_ALERT_LIMIT" ]; then
                alert "transient_loop" "\"$TRANSIENT consecutive transient prepare failures\""
                state "alert_transient_loop"
                exit 0
            fi
            log "transient prepare failure #$TRANSIENT (concurrent ledger read or lock contention); retrying in 90s"
            state "transient_backoff"
            snooze 90
            ;;
        crash|*)
            CRASH=$((CRASH + 1))
            if [ "$CRASH" -ge "$CRASH_ALERT_LIMIT" ]; then
                alert "crash_loop" "\"$CRASH consecutive crashes without progress\""
                state "alert_crash_loop"
                exit 0
            fi
            DELAY=$((60 * CRASH))
            [ "$DELAY" -gt 900 ] && DELAY=900
            log "crash #$CRASH without summary; backing off ${DELAY}s (stderr tail: $(tail -c 200 "$ERRF" 2>/dev/null | tr '\n' ' '))"
            state "crash_backoff"
            snooze "$DELAY"
            ;;
    esac
done
