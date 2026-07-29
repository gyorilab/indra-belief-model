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
# Usage: supervise_comparison_arm.sh <action_id> [initial_delay_seconds]
#
# State (data/comparison/supervisor/):
#   <arm>.supervisor.log   supervisor decisions
#   <arm>.out.log          accumulated runner stdout (readiness + summaries)
#   <arm>.err.log          accumulated runner stderr
#   <arm>.state.json       heartbeat: phase, invocation, crash counters
#   <arm>.COMPLETE         terminal marker: action complete (status-verified)
#   <arm>.ALERT            terminal marker: operator intervention required
set -u

REPO="/Users/noot/Documents/indra-belief-model"
ARM="${1:?usage: supervise_comparison_arm.sh <action_id> [initial_delay]}"
STAGGER="${2:-0}"
SUP="$REPO/data/comparison/supervisor"
PLAN="data/comparison/run_plan.json"
ATTEMPTS="$REPO/data/comparison/runs/$ARM/attempts.jsonl"
PROBE_HOST="bedrock-mantle.us-east-1.api.aws"
PY="$REPO/.venv/bin/python"
CRASH_ALERT_LIMIT=8
TRANSIENT_ALERT_LIMIT=40
NETFAIL_KILL_TICKS=4

mkdir -p "$SUP"
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

# Confirm the arm is really complete before writing the terminal marker.
# Returns 0 complete, 1 definitively not complete, 2 unverifiable right now.
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
        print('complete' if action.get('status') == 'complete' else 'partial')
        break
else:
    print('error')
" )
        case "$RESULT" in
            complete) return 0 ;;
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

    ( cd "$REPO" && exec env PYTHONPATH=src /usr/bin/caffeinate -i -s \
        "$PY" -m indra_belief.comparison run --plan "$PLAN" --action "$ARM" ) \
        >> "$OUT" 2>> "$ERRF" &
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
    if status == "complete":
        print("complete")
    elif status == "spend_cap" or kind == "spend_cap":
        print("spend_cap " + json.dumps(json.dumps(failure)))
    elif status == "deadline":
        print("deadline")
    elif kind in ("attempt_failed", "attempts_exhausted",
                  "nonretryable_failure_on_resume", "invalid_model_output_limit"):
        print("stuck " + json.dumps(json.dumps(failure)))
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
        complete|maybe_complete)
            verify_complete
            VERDICT=$?
            if [ "$VERDICT" -eq 0 ]; then
                date -u +%Y-%m-%dT%H:%M:%SZ > "$SUP/$ARM.COMPLETE"
                state "complete"
                log "action complete (status-verified); supervisor exiting"
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
        stuck)
            alert "stuck_under_plan_bounds" "$DETAIL"
            state "alert_stuck"
            log "a source is terminal under the frozen plan's bounds; a reviewed plan amendment is required"
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
