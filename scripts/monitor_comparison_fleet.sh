#!/bin/bash
# Watch one comparison fleet and heal it. Successor to the hand-written
# data/comparison/supervisor/fleet_monitor.sh, which lived only inside the
# gitignored data tree, hardcoded a three-arm list, and — the reason this
# exists — only REPORTED a dead supervisor. On 2026-07-22 glm_5_primary's
# supervisor died without hitting its trap and nothing restarted it; the arm
# sat idle for ~31 hours until a human noticed.
#
# Differences that matter:
#   * arms and attempts paths are derived from the plan, not hardcoded;
#   * a missing supervisor is RESTARTED (supervise_comparison_all.sh start is
#     idempotent and each supervisor self-guards via shlock + markers) rather
#     than reported once and abandoned;
#   * one arm's ALERT no longer blinds the other arms — it is announced and
#     watching continues, because an ALERT is per-arm terminal, not fleet
#     terminal;
#   * the stall window is 60 minutes, not 40. Under sleep-tolerant operation a
#     wake can be followed by several minutes of network probing before the
#     first row lands, and a spurious STALLED exit is worse than a late one.
#
# Usage: monitor_comparison_fleet.sh [plan_path]
# Intended to be run detached:
#   nohup /bin/bash scripts/monitor_comparison_fleet.sh <plan> \
#       >> <plan_dir>/supervisor/fleet_monitor.log 2>&1 & disown
set -u

REPO="/Users/noot/Documents/indra-belief-model"
PLAN="${1:-data/comparison/run_plan.json}"
# Accept an absolute in-repo plan path; every downstream use is repo-relative.
case "$PLAN" in /*) PLAN="${PLAN#"$REPO"/}" ;; esac
SUP="$REPO/$(dirname "$PLAN")/supervisor"
PY="$REPO/.venv/bin/python"
STALL_POLLS_LIMIT=6
POLL_SECONDS=600

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

# "<arm> <attempts_path>" per line, straight from the plan.
FLEET=$( (cd "$REPO" && "$PY" - "$PLAN" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as handle:
    plan = json.load(handle)
for action in plan["actions"]:
    if action["workload"] == "unique_exact_pairs_primary" and action["execution_keys"] is None:
        print(action["id"], action["output"]["path"])
PYEOF
) ) || { log "cannot read fleet from $PLAN"; exit 64; }
[ -n "$FLEET" ] || { log "no primary arms in $PLAN"; exit 64; }

ARM_COUNT=$(echo "$FLEET" | wc -l | tr -d ' ')
log "watching $ARM_COUNT arm(s) from $PLAN"
# The heal path re-launches supervisors, which inherit THIS process's
# environment. If the operator set the sleep-tolerance knobs as a command-scoped
# prefix on the launch line instead of exporting them, a healed supervisor
# silently reverts to caffeinate-on / 4-tick defaults — doubling the
# network-outage window on exactly the unattended path. Log what will actually
# propagate so a silent revert is visible in the log rather than inferred later.
log "heal will propagate SUPERVISOR_CAFFEINATE=${SUPERVISOR_CAFFEINATE:-<unset, arm default 1>} NETFAIL_KILL_TICKS=${NETFAIL_KILL_TICKS:-<unset, arm default 4>}"

PREV_TOTAL=0
STALL_POLLS=0
ANNOUNCED=""

while :; do
    TERMINAL=0
    TOTAL=0
    ORPHANED=""
    while read -r arm attempts; do
        [ -n "$arm" ] || continue
        arm_terminal=0
        # SETTLED is terminal too. Omitting it would leave a finished arm
        # looking non-terminal with a correctly-exited supervisor, so the healer
        # below would relaunch it forever — and each relaunch would see its own
        # SETTLED marker and exit immediately.
        for marker in ALERT COMPLETE SETTLED; do
            if [ -f "$SUP/$arm.$marker" ]; then
                TERMINAL=$((TERMINAL + 1))
                arm_terminal=1
                case "$ANNOUNCED" in
                    *"[$arm.$marker]"*) ;;
                    *)
                        log "$marker [$arm]: $(cat "$SUP/$arm.$marker")"
                        ANNOUNCED="$ANNOUNCED [$arm.$marker]"
                        ;;
                esac
                break
            fi
        done
        # Per-arm liveness, not fleet-wide. A fleet-wide `pgrep` only heals when
        # EVERY supervisor is dead, so the historical failure — one arm's
        # supervisor dying silently while the others ran on — would still go
        # unhealed for as long as any sibling was alive.
        if [ "$arm_terminal" -eq 0 ] \
           && ! pgrep -f "supervise_comparison_arm.sh $arm " >/dev/null 2>&1; then
            ORPHANED="$ORPHANED $arm"
        fi
        TOTAL=$((TOTAL + $(stat -f %z "$REPO/$attempts" 2>/dev/null || echo 0)))
    done <<EOF
$FLEET
EOF

    if [ "$TERMINAL" -ge "$ARM_COUNT" ]; then
        log "every arm is terminal (complete, settled or alerted) — monitor exiting"
        exit 0
    fi

    # Heal, don't just report. Re-confirm after 30s so a supervisor caught
    # between invocations (it re-execs the runner in a loop) is not called dead.
    if [ -n "$ORPHANED" ]; then
        sleep 30
        still=""
        for arm in $ORPHANED; do
            pgrep -f "supervise_comparison_arm.sh $arm " >/dev/null 2>&1 || still="$still $arm"
        done
        if [ -n "$still" ]; then
            log "unsupervised while work remains:$still (progress bytes: $TOTAL) — healing"
            /bin/bash "$REPO/scripts/supervise_comparison_all.sh" start "$PLAN" 2>&1 | while read -r line; do
                log "  $line"
            done
        fi
    fi

    if [ "$TOTAL" -le "$PREV_TOTAL" ]; then
        STALL_POLLS=$((STALL_POLLS + 1))
        if [ "$STALL_POLLS" -ge "$STALL_POLLS_LIMIT" ]; then
            log "STALLED: no attempts growth in $((STALL_POLLS_LIMIT * POLL_SECONDS / 60)) minutes and no terminal marker (bytes: $TOTAL)"
            STALL_POLLS=0
        fi
    else
        STALL_POLLS=0
    fi
    PREV_TOTAL=$TOTAL
    sleep "$POLL_SECONDS"
done
