#!/bin/bash
# Start one detached supervisor per primary comparison arm.
#
# Runs supervisors via nohup from the calling terminal session: macOS TCC
# denies launchd background items access to ~/Documents (exit 126,
# "Operation not permitted"), so a LaunchAgent cannot host this repo's
# supervisors without a manual Full Disk Access grant. nohup'd supervisors
# survive terminal and session exit (orphans are adopted by init and keep
# the launching terminal's TCC file-access attribution) but NOT reboot:
# after a reboot or power loss, re-run this script from a terminal.
#
# Default start is non-disruptive: arms whose supervisor is already running
# are left untouched, and each supervisor refuses to double-start via its
# own lock, COMPLETE/ALERT markers, and adopt-wait of live runners.
# `restart` force-stops supervisors AND their runner invocations (safe —
# runs resume from append-only ledgers — but mid-flight attempts are cut).
# `stop` halts supervision without touching markers.
#
# Usage: supervise_comparison_all.sh [restart|stop] [plan_path]
#
# The arm list is DERIVED from the plan rather than hardcoded: every action on
# the primary workload that is not a pinned-key smoke. On the original plan
# that reproduces the historical list and order exactly
# (gemma_31b_primary gemma_26b_primary glm_5_primary), so the default
# invocation is unchanged; on a second plan it picks up that plan's arms with
# no edit here.
set -eu

REPO="/Users/noot/Documents/indra-belief-model"
MODE="${1:-start}"
PLAN="${2:-data/comparison/run_plan.json}"
# Accept an absolute in-repo plan path; every downstream use is repo-relative.
case "$PLAN" in /*) PLAN="${PLAN#"$REPO"/}" ;; esac
SUP="$REPO/$(dirname "$PLAN")/supervisor"

mkdir -p "$SUP"

ARMS=$( (cd "$REPO" && "$REPO/.venv/bin/python" - "$PLAN" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as handle:
    plan = json.load(handle)
print(" ".join(
    action["id"]
    for action in plan["actions"]
    if action["workload"] == "unique_exact_pairs_primary"
    and action["execution_keys"] is None
))
PYEOF
) ) || { echo "cannot read arms from $PLAN" >&2; exit 64; }
[ -n "$ARMS" ] || { echo "no primary arms in $PLAN" >&2; exit 64; }

# Position in the derived list x STAGGER_SECONDS. Replaces the per-arm case
# table, whose `*) echo 0` fallback silently gave any new arm zero stagger —
# putting every arm's preflight and first provider burst in lockstep.
#
# The stagger is NOT throughput: arms run fully concurrently once started, so it
# costs (n-1) x STAGGER_SECONDS once, at launch. It buys separation of the
# preflights, which each stream every action's attempts file, and of the first
# provider burst. Set STAGGER_SECONDS=0 for a simultaneous start; expect more
# "transient prepare failure" restarts if you do (one was already observed at
# 180s, self-healed on the next invocation).
STAGGER_SECONDS="${STAGGER_SECONDS:-180}"
case "$STAGGER_SECONDS" in
    ''|*[!0-9]*) echo "STAGGER_SECONDS must be a non-negative integer" >&2; exit 64 ;;
esac

stagger_for() {
    index=0
    for candidate in $ARMS; do
        if [ "$candidate" = "$1" ]; then echo $((index * STAGGER_SECONDS)); return; fi
        index=$((index + 1))
    done
    echo 0
}

supervisor_running() {
    pgrep -f "supervise_comparison_arm.sh $1" >/dev/null 2>&1
}

stop_arm() {
    arm="$1"
    pkill -f "supervise_comparison_arm.sh $arm" 2>/dev/null || true
    waited=0
    while supervisor_running "$arm" && [ "$waited" -lt 60 ]; do
        sleep 2; waited=$((waited + 2))
    done
    if supervisor_running "$arm"; then
        echo "$arm: supervisor did not stop within ${waited}s" >&2
        return 1
    fi
    return 0
}

start_arm() {
    arm="$1"
    nohup /bin/bash "$REPO/scripts/supervise_comparison_arm.sh" "$arm" "$(stagger_for "$arm")" "$PLAN" \
        >> "$SUP/$arm.nohup.log" 2>&1 &
    disown $! 2>/dev/null || true
    echo "$arm: supervisor started (pid $!, stagger $(stagger_for "$arm")s, plan $PLAN)"
}

case "$MODE" in
    stop)
        for arm in $ARMS; do
            stop_arm "$arm" && echo "$arm: stopped" || true
        done
        ;;
    restart)
        for arm in $ARMS; do
            stop_arm "$arm" || { echo "$arm: NOT restarted" >&2; continue; }
            start_arm "$arm"
        done
        ;;
    start)
        for arm in $ARMS; do
            if supervisor_running "$arm"; then
                echo "$arm: supervisor already running; leaving untouched"
                continue
            fi
            start_arm "$arm"
        done
        ;;
    *)
        echo "usage: supervise_comparison_all.sh [restart|stop]" >&2
        exit 64
        ;;
esac
