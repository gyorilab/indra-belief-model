"""The supervisor's decision function, executed as the shipped script ships it.

The runner classifies every failure as `quarantine` (one source) or `halt` (the
whole action), but that classification is only worth anything if the automation
reads it.  It did not: `supervise_comparison_arm.sh` keyed on `failure["kind"]`
alone and mapped all four quarantine kinds to a terminal "stuck" ALERT that
refuses to restart until a human clears the marker.  An arm that had absorbed a
single off-grid row exactly as designed was stopped dead anyway, and for
`invalid_model_output_limit` the alert text asserted a plan amendment was
required when none was.

These tests extract the decision heredoc from the script and run it, so they
assert on the bytes that actually execute rather than on a transcription of
them.  `test_decision_source_is_the_shipped_heredoc` fails if that extraction
ever stops finding the real thing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "supervise_comparison_arm.sh"

# The runner's own summary shape, trimmed to the keys the supervisor reads.
_BASE = {
    "action_id": "gemma_26b_primary",
    "completed_this_run": 4,
    "completed_total": 33_000,
    "total": 33_361,
    "verdicts": {"correct": 20_000, "incorrect": 13_000},
    "spend_guard": {},
    "quarantined": 0,
    "quarantined_sources": [],
    "quarantined_sources_truncated": False,
}


def _decision_source() -> str:
    """The decision program, lifted verbatim out of the shipped script."""
    text = SCRIPT.read_text()
    match = re.search(
        r"DECISION=\$\(\"\$PY\" - \"\$OUT\" \"\$ERRF\" <<'PYEOF'\n(.*?)\nPYEOF\n",
        text,
        re.DOTALL,
    )
    assert match, "the supervisor's decision heredoc moved or changed shape"
    return match.group(1)


def _decide(tmp_path: Path, summary: dict | None, *, stderr_tail: str = "") -> str:
    out = tmp_path / "out.log"
    err = tmp_path / "err.log"
    out.write_text(
        json.dumps({"status": "ready_for_bearer_token", "plan_sha256": "x"}) + "\n"
        + ("" if summary is None else json.dumps(summary) + "\n")
    )
    err.write_text(stderr_tail)
    program = tmp_path / "decision.py"
    program.write_text(_decision_source())
    result = subprocess.run(
        [sys.executable, str(program), str(out), str(err)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _summary(**overrides) -> dict:
    return {**_BASE, **overrides}


def test_decision_source_is_the_shipped_heredoc(tmp_path: Path) -> None:
    """Guard the extraction itself, so the tests below cannot go vacuous."""
    source = _decision_source()
    assert "disposition" in source
    assert "quarantined" in source
    # A regression to kind-only dispatch would put these three kinds back in a
    # membership test; they must not be classified by name any more.
    assert 'kind in ("attempt_failed"' not in source


@pytest.mark.parametrize(
    "kind",
    [
        "attempt_failed",
        "attempts_exhausted",
        "nonretryable_failure_on_resume",
        "invalid_model_output_limit",
    ],
)
def test_every_quarantine_kind_carries_the_list_to_the_operator(
    tmp_path: Path, kind: str
) -> None:
    """The defect, one kind at a time.

    Every one of these four reached the old branch by NAME, became a single
    terminal "stuck" ALERT carrying ONE failure, and asserted that a reviewed
    plan amendment was required.  Against an all-or-nothing corpus the arm
    really is finished — that part was right — but the operator was told a row
    was bad, not HOW MANY rows are bad or which, and for
    `invalid_model_output_limit` the remedy named does not exist: raising
    max_attempts leaves the per-source invalid-output cap untouched.

    The list is the entire product of the diagnostic budget, so it has to
    survive into the ALERT.
    """
    decision = _decide(tmp_path, _summary(
        status="settled",
        failure={"kind": kind, "type": "InvalidModelOutput",
                 "stmt_i": 7, "evidence_i": 0, "disposition": "quarantine"},
        quarantined=4,
        quarantined_sources=[
            {"kind": kind, "stmt_i": position, "evidence_i": 0}
            for position in (7, 51, 96, 143)
        ],
    ))
    word, _, detail = decision.partition(" ")
    assert word == "settled"
    payload = json.loads(json.loads(detail))
    assert payload["quarantined"] == 4
    assert [row["stmt_i"] for row in payload["quarantined_sources"]] == [
        7, 51, 96, 143
    ]
    assert payload["completed_total"] == 33_000


def test_a_settled_arm_is_terminal_but_is_not_complete(tmp_path: Path) -> None:
    """Finished, with holes: its own decision, and its own marker.

    It must never share the COMPLETE marker. A settled arm cannot be bundled,
    so anything treating it as a finished arm would look for a bundle that the
    publisher will refuse to produce.
    """
    decision = _decide(tmp_path, _summary(
        status="settled",
        failure={"kind": "invalid_model_output_limit", "stmt_i": 7,
                 "evidence_i": 0, "disposition": "quarantine"},
        quarantined=3,
    ))
    word, _, detail = decision.partition(" ")
    assert word == "settled"
    assert word != "complete"
    assert json.loads(json.loads(detail))["quarantined"] == 3


def test_the_diagnostic_budget_alerts_with_the_regime(tmp_path: Path) -> None:
    """A systematic breakage must reach a human WITH its diagnosis.

    "systematic" versus "sporadic" is the one thing the pre-S2 first-row halt
    could never say, and it is what the budget was spent to find out. It has to
    reach the ALERT detail, or the spend bought nothing.
    """
    decision = _decide(tmp_path, _summary(
        status="settled",
        failure={"kind": "quarantine_budget", "quarantined": 8,
                 "dispatched_since_first_quarantine": 8,
                 "quarantine_limit": 8, "source_limit": 200,
                 "regime": "systematic", "disposition": "halt"},
        quarantined=10,
        completed_this_run=0,
    ))
    word, _, detail = decision.partition(" ")
    assert word == "settled"
    payload = json.loads(json.loads(detail))
    assert payload["failure"]["regime"] == "systematic"
    assert payload["failure"]["quarantined"] == 8
    assert payload["quarantined"] == 10


@pytest.mark.parametrize(
    "failure",
    [
        # Genuine global-halt classes must still stop the arm on sight.
        {"kind": "attempt_failed", "type": "HTTPError",
         "provider_http_status": 401, "disposition": "halt"},
        {"kind": "attempt_failed", "type": "ContractError",
         "disposition": "halt"},
        {"kind": "attempt_failed", "type": "SomeFutureProviderError",
         "disposition": "halt"},
    ],
)
def test_genuine_halts_still_alert(tmp_path: Path, failure: dict) -> None:
    decision = _decide(tmp_path, _summary(status="partial", failure=failure))
    assert decision.split()[0] == "stopped"


def test_cap_and_deadline_keep_their_own_decisions(tmp_path: Path) -> None:
    """Quarantine must not have swallowed the two the supervisor already had."""
    assert _decide(tmp_path, _summary(
        status="spend_cap",
        failure={"kind": "spend_cap", "type": "ActionCapReached",
                 "disposition": "halt"},
    )).split()[0] == "spend_cap"
    assert _decide(tmp_path, _summary(
        status="deadline",
        failure={"kind": "deadline", "disposition": "halt"},
    )) == "deadline"
    assert _decide(tmp_path, _summary(status="complete", failure=None)) == "complete"


def test_the_stderr_fallbacks_are_untouched(tmp_path: Path) -> None:
    """No summary line: the pre-existing tail classification still decides."""
    assert _decide(tmp_path, None,
                   stderr_tail="RunnerError: run plan is already complete\n") == "maybe_complete"
    assert _decide(tmp_path, None,
                   stderr_tail="ReplayError: raw output ends in a partial JSONL row\n") == "transient"
    assert _decide(tmp_path, None, stderr_tail="Traceback: something else\n") == "crash"


def test_terminal_markers_are_all_honored_by_the_fleet_monitor() -> None:
    """A settled arm is terminal to the healer, or it is relaunched forever.

    `monitor_comparison_fleet.sh` restarts any arm that is neither terminal nor
    has a live supervisor.  A settled arm's supervisor exits on purpose, so if
    SETTLED were not in the terminal marker set the monitor would relaunch it on
    every poll, and each relaunch would read its own SETTLED marker and exit.
    """
    monitor = (SCRIPT.parent / "monitor_comparison_fleet.sh").read_text()
    assert "for marker in ALERT COMPLETE SETTLED; do" in monitor
    # And the supervisor itself must not start work on an arm already settled.
    assert '[ -f "$SUP/$ARM.SETTLED" ]' in SCRIPT.read_text()
