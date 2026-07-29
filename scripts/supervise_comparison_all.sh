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
# Usage: supervise_comparison_all.sh [restart|stop]
set -eu

REPO="/Users/noot/Documents/indra-belief-model"
SUP="$REPO/data/comparison/supervisor"
ARMS="gemma_31b_primary gemma_26b_primary glm_5_primary"
MODE="${1:-start}"

mkdir -p "$SUP"

stagger_for() {
    case "$1" in
        gemma_31b_primary) echo 0 ;;
        gemma_26b_primary) echo 180 ;;
        glm_5_primary) echo 360 ;;
        *) echo 0 ;;
    esac
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
    nohup /bin/bash "$REPO/scripts/supervise_comparison_arm.sh" "$arm" "$(stagger_for "$arm")" \
        >> "$SUP/$arm.nohup.log" 2>&1 &
    disown $! 2>/dev/null || true
    echo "$arm: supervisor started (pid $!, stagger $(stagger_for "$arm")s)"
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
