"""Score-corpus orchestration — Phase 3.1 of the rasmachine task graph.

`score_corpus(con, stmts, client, ...)` iterates statements through
`indra_belief.score_evidence` and writes per-evidence aggregate rows to
`scorer_step` plus a single `score_run` summary row.

This is the minimum viable orchestration: one `scorer_step(step_kind='aggregate')`
per evidence, capturing the full per-evidence dict (score / verdict /
confidence / reasons / call_log) as `output_json`. Phase 3.4 decomposes
this into per-step rows (parse_claim / build_context / substrate_route /
4 probes / grounding / adjudicate) by emitting structured events from
within `score_via_probes` rather than parsing the aggregate dict.

Append-only contract (Phase 2.5): a re-run with a different `scorer_version`
lands new rows alongside the old; same `(stmt_hash, evidence_hash,
scorer_version, step_kind)` is upserted by `step_hash`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Protocol

if TYPE_CHECKING:
    import duckdb
    from indra.statements import Statement

log = logging.getLogger(__name__)


class ScoreEvidenceFn(Protocol):
    """Callable that scores one (Statement, Evidence) pair and returns a dict."""

    def __call__(self, statement, evidence, client) -> dict: ...


Architecture = Literal["decomposed", "monolithic"]
# Pre-started-cancelled denotes a paired-architecture cancel that fires before
# the per-architecture worker spawns and emits `started.run_id`. The viewer's
# score-paired handler synthesizes a score_run row with this status so the
# cancellation is queryable in the same table as canceled/succeeded runs
# instead of only being visible in the sidecar workflow-state JSON.
CANCEL_TERMINAL_STATUSES = {"canceled", "cancelled", "aborted", "pre_started_cancelled"}
PROBE_STEP_KIND_TO_TRACE_KEY = {
    "subject_role_probe": "subject_role",
    "object_role_probe": "object_role",
    "relation_axis_probe": "relation_axis",
    "scope_probe": "scope",
}
PROBE_TRACE_KEY_TO_STEP_KIND = {
    trace_key: step_kind
    for step_kind, trace_key in PROBE_STEP_KIND_TO_TRACE_KEY.items()
}
PROBE_STEP_KINDS = frozenset(PROBE_STEP_KIND_TO_TRACE_KEY)


def _hex(n: int, width: int = 16) -> str:
    return f"{n & ((1 << 64) - 1):0{width}x}"


def _step_hash(
    run_id: str,
    stmt_hash: str,
    evidence_hash: str,
    scorer_version: str,
    model_id: str,
    architecture: str,
    step_kind: str,
    input_payload_hash: str = "",
) -> str:
    raw = (
        f"{run_id}|{stmt_hash}|{evidence_hash}|{scorer_version}|{model_id}|"
        f"{architecture}|{step_kind}|{input_payload_hash}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _sum_reported_tokens(call_log: list[dict], field: str) -> int | None:
    """Sum reported token fields, ignoring missing and negative sentinels."""
    total = 0
    saw_reported = False
    for call in call_log:
        value = call.get(field)
        if isinstance(value, (int, float)) and value >= 0:
            total += int(value)
            saw_reported = True
    return total if saw_reported else None


def _nonnegative_int(value) -> int | None:
    return int(value) if isinstance(value, (int, float)) and value >= 0 else None


def _normalize_probe_step_filter(
    probe_step_filter: Iterable[str] | None,
) -> frozenset[str] | None:
    if probe_step_filter is None:
        return None
    out = set()
    for raw in probe_step_filter:
        step_kind = str(raw).strip()
        if not step_kind:
            continue
        if step_kind not in PROBE_STEP_KINDS:
            raise ValueError(
                "probe_step_filter only accepts decomposed probe step kinds; "
                f"got {step_kind!r}"
            )
        out.add(step_kind)
    return frozenset(out)


def _probe_call_usage(call_log: list[dict], trace_key: str) -> dict:
    kind = f"probe_{trace_key}"
    prompt_tokens = 0
    out_tokens = 0
    saw_prompt = False
    saw_out = False
    duration_s = 0.0
    saw_duration = False
    finish_reason = None
    for call in call_log:
        if call.get("kind") != kind:
            continue
        prompt = _nonnegative_int(call.get("prompt_tokens"))
        out = _nonnegative_int(call.get("out_tokens"))
        if prompt is not None:
            prompt_tokens += prompt
            saw_prompt = True
        if out is not None:
            out_tokens += out
            saw_out = True
        duration = call.get("duration_s")
        if isinstance(duration, (int, float)) and duration >= 0:
            duration_s += float(duration)
            saw_duration = True
        if finish_reason is None and call.get("finish_reason") is not None:
            finish_reason = str(call.get("finish_reason"))
    return {
        "prompt_tokens": prompt_tokens if saw_prompt else None,
        "out_tokens": out_tokens if saw_out else None,
        "latency_ms": int(duration_s * 1000) if saw_duration else None,
        "finish_reason": finish_reason,
    }


def _probe_trace_steps(
    result: dict,
    probe_filter: frozenset[str] | None,
) -> list[tuple[str, dict, bool | None, dict]]:
    trace = result.get("probe_trace")
    if not isinstance(trace, dict):
        return []
    call_log = result.get("call_log") or []
    out: list[tuple[str, dict, bool | None, dict]] = []
    for trace_key, step_kind in PROBE_TRACE_KEY_TO_STEP_KIND.items():
        if probe_filter is not None and step_kind not in probe_filter:
            continue
        payload = trace.get(trace_key)
        if not isinstance(payload, dict):
            continue
        source = payload.get("source")
        out.append((
            step_kind,
            payload,
            source == "substrate" if source in {"substrate", "llm", "abstain"} else None,
            _probe_call_usage(call_log, trace_key),
        ))
    return out


def _probe_filter_trace_keys(probe_filter: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        trace_key
        for step_kind, trace_key in PROBE_STEP_KIND_TO_TRACE_KEY.items()
        if step_kind in probe_filter
    )


def _json_dict(value) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _probe_payload_from_parent(
    con: "duckdb.DuckDBPyConnection",
    *,
    parent_run_id: str,
    stmt_hash: str,
    evidence_hash: str,
    step_kind: str,
    trace_key: str,
) -> dict | None:
    row = con.execute(
        """SELECT output_json
             FROM scorer_step
            WHERE run_id = ?
              AND stmt_hash = ?
              AND evidence_hash = ?
              AND step_kind = ?
            ORDER BY started_at DESC
            LIMIT 1""",
        [parent_run_id, stmt_hash, evidence_hash, step_kind],
    ).fetchone()
    payload = _json_dict(row[0]) if row else None
    if payload is not None:
        return payload

    row = con.execute(
        """SELECT output_json
             FROM scorer_step
            WHERE run_id = ?
              AND stmt_hash = ?
              AND evidence_hash = ?
              AND step_kind = 'aggregate'
            ORDER BY started_at DESC
            LIMIT 1""",
        [parent_run_id, stmt_hash, evidence_hash],
    ).fetchone()
    aggregate = _json_dict(row[0]) if row else None
    trace = aggregate.get("probe_trace") if isinstance(aggregate, dict) else None
    payload = trace.get(trace_key) if isinstance(trace, dict) else None
    return payload if isinstance(payload, dict) else None


def _probe_response_from_payload(payload: dict, expected_kind: str):
    from indra_belief.scorers.probes.types import ProbeResponse

    data = {
        "kind": payload.get("kind") or expected_kind,
        "answer": payload.get("answer"),
        "source": payload.get("source"),
        "confidence": payload.get("confidence") or "medium",
        "perturbation": payload.get("perturbation"),
        "span": payload.get("span"),
        "rationale": payload.get("rationale") or "",
    }
    return ProbeResponse(**data)


def _write_probe_only_merge_aggregate(
    con: "duckdb.DuckDBPyConnection",
    *,
    parent_run_id: str | None,
    child_run_id: str,
    stmt,
    evidence,
    stmt_hash: str,
    evidence_hash: str,
    scorer_version: str,
    model_id: str,
    architecture: str,
    result: dict,
    probe_filter: frozenset[str],
) -> bool:
    """Write a deterministic aggregate row from repaired native probe rows.

    This is not another aggregate LLM call. It combines selected probe answers
    from the child run with the parent run's persisted probe trace, then runs
    the pure adjudicator so before/after repair lanes have a real append-only
    aggregate row to compare.
    """
    if not parent_run_id:
        return False
    trace = result.get("probe_trace")
    if not isinstance(trace, dict):
        return False

    payloads: dict[str, dict] = {}
    source_by_probe: dict[str, str] = {}
    for step_kind, trace_key in PROBE_STEP_KIND_TO_TRACE_KEY.items():
        child_payload = trace.get(trace_key) if step_kind in probe_filter else None
        if isinstance(child_payload, dict):
            payloads[trace_key] = child_payload
            source_by_probe[trace_key] = "child_probe_only"
            continue
        parent_payload = _probe_payload_from_parent(
            con,
            parent_run_id=parent_run_id,
            stmt_hash=stmt_hash,
            evidence_hash=evidence_hash,
            step_kind=step_kind,
            trace_key=trace_key,
        )
        if parent_payload is None:
            log.warning(
                "probe-only merge skipped for %s/%s: missing parent %s",
                stmt_hash,
                evidence_hash,
                step_kind,
            )
            return False
        payloads[trace_key] = parent_payload
        source_by_probe[trace_key] = "parent_trace"

    try:
        from dataclasses import asdict
        from indra_belief.scorers.commitments import adjudication_to_score
        from indra_belief.scorers.context import EvidenceContext
        from indra_belief.scorers.parse_claim import parse_claim
        from indra_belief.scorers.probes.adjudicator import adjudicate
        from indra_belief.scorers.probes.types import ProbeBundle

        claim = parse_claim(stmt)
        ctx = EvidenceContext.from_statement_and_evidence(stmt, evidence)
        bundle = ProbeBundle(
            subject_role=_probe_response_from_payload(
                payloads["subject_role"], "subject_role"
            ),
            object_role=_probe_response_from_payload(
                payloads["object_role"], "object_role"
            ),
            relation_axis=_probe_response_from_payload(
                payloads["relation_axis"], "relation_axis"
            ),
            scope=_probe_response_from_payload(payloads["scope"], "scope"),
        )
        adj = adjudicate(claim, bundle, (), ctx=ctx)
    except Exception as e:
        log.warning(
            "probe-only merge adjudication failed for %s/%s: %s",
            stmt_hash,
            evidence_hash,
            e,
        )
        return False

    output = {
        "score": adjudication_to_score(adj),
        "verdict": adj.verdict,
        "confidence": adj.confidence,
        "raw_text": (
            "[S-PHASE probe repair merge] deterministic re-adjudication "
            "from selected child probe rows and parent probe trace"
        ),
        "tokens": 0,
        "tier": "decomposed_probe_repair_merge",
        "grounding_status": "not_run",
        "provenance_triggered": False,
        "reasons": list(adj.reasons),
        "rationale": (
            "merged selected child probe rows with parent persisted probe "
            "trace; no aggregate LLM call was made"
        ),
        "probe_trace": {
            "subject_role": asdict(bundle.subject_role),
            "object_role": asdict(bundle.object_role),
            "relation_axis": asdict(bundle.relation_axis),
            "scope": asdict(bundle.scope),
        },
        "repair_merge": {
            "parent_run_id": parent_run_id,
            "child_run_id": child_run_id,
            "probe_step_filter": sorted(probe_filter),
            "source_by_probe": source_by_probe,
            "aggregate_llm_call": False,
        },
        "call_log": [],
    }
    step_hash = _step_hash(
        child_run_id,
        stmt_hash,
        evidence_hash,
        scorer_version,
        model_id,
        architecture,
        "aggregate",
        f"probe_repair_merge:{parent_run_id}:{','.join(sorted(probe_filter))}",
    )
    con.execute(
        """INSERT OR REPLACE INTO scorer_step
           (step_hash, stmt_hash, evidence_hash, run_id,
            scorer_version, architecture, model_id, step_kind, is_substrate_answered,
            input_payload_json, output_json, latency_ms,
            prompt_tokens, out_tokens, finish_reason, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'aggregate', ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            step_hash,
            stmt_hash,
            evidence_hash,
            child_run_id,
            scorer_version,
            architecture,
            model_id,
            None,
            json.dumps({
                "kind": "probe_repair_merge",
                "parent_run_id": parent_run_id,
                "probe_step_filter": sorted(probe_filter),
            }),
            json.dumps(output, default=str),
            None,
            None,
            None,
            None,
            None,
        ],
    )
    return True


def _raise_if_run_canceled(con: "duckdb.DuckDBPyConnection", run_id: str) -> None:
    row = con.execute(
        "SELECT status FROM score_run WHERE run_id = ?",
        [run_id],
    ).fetchone()
    status = str(row[0]) if row and row[0] is not None else "running"
    if status in CANCEL_TERMINAL_STATUSES:
        raise RuntimeError(f"score_corpus canceled: score_run {run_id} is {status}")
    # C1 heartbeat: free piggyback on the cancel-check hot path. Every
    # evidence loop already reads score_run.status here, so updating the
    # heartbeat is one additional cheap UPDATE that gives the startup
    # janitor precise liveness signal.
    try:
        con.execute(
            "UPDATE score_run SET heartbeat_at = CURRENT_TIMESTAMP "
            "WHERE run_id = ? AND status = 'running'",
            [run_id],
        )
    except Exception:
        # Heartbeat write failure is non-fatal: the wall-clock fallback in
        # the janitor still catches stale rows. Don't crash the scoring
        # loop just because a column is missing on a very-old DB.
        pass


def _detect_indra_version() -> str:
    try:
        import indra
        return getattr(indra, "__version__", "unknown")
    except Exception:
        return "unknown"


def _decompose_steps(stmt, ev) -> list[tuple[str, dict, bool | None]]:
    """Phase 3.4 partial: capture deterministic substeps as separate rows.

    Returns a list of `(step_kind, output_json_dict, is_substrate_answered)`
    tuples for steps 1-3 (parse_claim, build_context, substrate_route) plus
    one row per substrate-answered probe (steps 4-7 when substrate hits).

    Steps 4-7 (LLM-escalated probes), 8 (grounding), 9 (adjudicate) are
    entangled with the LLM call and remain captured in the aggregate row
    until the orchestrator emits them as structured events.
    """
    out: list[tuple[str, dict, bool | None]] = []

    try:
        from dataclasses import asdict, is_dataclass
        from indra_belief.scorers.parse_claim import parse_claim
        from indra_belief.scorers.context_builder import build_context
        from indra_belief.scorers.probes.router import substrate_route
    except Exception as e:
        log.warning("decompose imports failed: %s", e)
        return out

    def _to_dict(obj):
        if obj is None:
            return None
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "_asdict"):
            return obj._asdict()
        if isinstance(obj, (str, int, float, bool, list, dict, tuple)):
            return obj
        # Fallback: stringify
        return str(obj)

    try:
        claim = parse_claim(stmt)
        out.append(("parse_claim", _to_dict(claim) or {}, None))
    except Exception as e:
        out.append(("parse_claim", {"error": str(e)}, None))
        return out

    try:
        ctx = build_context(stmt, ev)
        ctx_summary = {
            "stmt_type": getattr(ctx, "stmt_type", None),
            "n_aliases": len(getattr(ctx, "aliases", {}) or {}),
            "n_detected_relations": len(getattr(ctx, "detected_relations", []) or []),
            "n_modifications_sites": len(getattr(ctx, "detected_sites", set()) or set()),
            "has_chain_signal": getattr(ctx, "has_chain_signal", False),
            "is_complex": getattr(ctx, "is_complex", False),
            "is_modification": getattr(ctx, "is_modification", False),
            "subject_class": getattr(ctx, "subject_class", "unknown"),
            "object_class": getattr(ctx, "object_class", "unknown"),
        }
        out.append(("build_context", ctx_summary, None))
    except Exception as e:
        out.append(("build_context", {"error": str(e)}, None))
        return out

    try:
        evidence_text = (getattr(ev, "text", "") or "").strip()
        routes = substrate_route(claim, ctx, evidence_text)
        # Top-level row summarizing how each probe was routed
        route_summary = {}
        substrate_answered = {}
        for kind, route in routes.items():
            source = getattr(route, "source", None)
            answer = getattr(route, "answer", None)
            confidence = getattr(route, "confidence", None)
            route_summary[kind] = {
                "source": source,
                "answer": answer,
                "confidence": confidence,
            }
            substrate_answered[kind] = (source == "substrate")
        out.append(("substrate_route", route_summary, None))

        # Per-probe rows for substrate-answered probes (steps 4-7 lit when hit)
        kind_to_step = {
            "subject_role": "subject_role_probe",
            "object_role": "object_role_probe",
            "relation_axis": "relation_axis_probe",
            "scope": "scope_probe",
        }
        for kind, route in routes.items():
            if substrate_answered.get(kind):
                step_kind = kind_to_step[kind]
                out.append((
                    step_kind,
                    {
                        "answer": getattr(route, "answer", None),
                        "confidence": getattr(route, "confidence", None),
                        "source": "substrate",
                        "span": getattr(route, "span", None),
                        "rationale": getattr(route, "rationale", None),
                    },
                    True,
                ))
    except Exception as e:
        out.append(("substrate_route", {"error": str(e)}, None))

    return out


def score_corpus(
    con: "duckdb.DuckDBPyConnection",
    stmts: Iterable["Statement"],
    *,
    run_id: str | None = None,
    client=None,
    scorer_version: str = "dev",
    model_id_default: str = "unknown",
    architecture: Architecture = "decomposed",
    paired_run_group_id: str | None = None,
    parent_run_id: str | None = None,
    score_evidence: ScoreEvidenceFn | None = None,
    on_evidence: Callable[[str, str, dict], None] | None = None,
    decompose: bool = False,
    with_validity: bool = True,
    cost_threshold_usd: float | None = None,
    probe_step_filter: Iterable[str] | None = None,
    probe_only: bool = False,
) -> str:
    """Score a stream of INDRA Statements and write rows to the corpus DB.

    Args:
        con: a DuckDB connection with the corpus schema applied.
        stmts: iterable of INDRA Statement objects (already ingested via
            `ingest_statements`; this function does NOT re-ingest).
        client: a `ModelClient` (or compatible) — passed straight to
            `score_evidence`. Required for real scoring.
        scorer_version: identifier for this run's scorer code (typically
            a git commit hash). Multiple runs at different versions land
            alongside each other in `scorer_step`; never overwrite.
        model_id_default: LLM identifier recorded on `score_run` and
            propagated to per-step rows when not overridden.
        architecture: scoring architecture that produced this run. This is
            persisted and included in `step_hash` so monolithic and decomposed
            runs over the same evidence cannot overwrite each other.
        paired_run_group_id: optional shared id tying two architecture runs
            into one paired experiment.
        parent_run_id: optional baseline run id for repair/rerun loops.
        score_evidence: override the default `indra_belief.score_evidence`.
            Useful for tests with a mock that returns deterministic dicts.
        on_evidence: optional callback `(stmt_hash, evidence_hash, dict)`
            fired after each evidence is scored. Phase 3.7 SSE live-tail
            hooks here.
        probe_step_filter: selected decomposed probe step kinds to
            materialize as native rows.
        probe_only: for reviewed probe-slot repair reruns, resolve and
            persist only the selected probe rows, then deterministically merge
            with parent probe trace into an aggregate row when parent_run_id
            supplies enough trace. This intentionally skips aggregate LLM calls.

    Returns:
        The `run_id` (UUID hex) so the caller can `JOIN metric ON metric.run_id`.
    """
    if architecture not in ("decomposed", "monolithic"):
        raise ValueError(
            "architecture must be 'decomposed' or 'monolithic', "
            f"got {architecture!r}"
        )
    if architecture == "monolithic" and decompose:
        raise ValueError(
            "decompose=True is only valid for architecture='decomposed'; "
            "monolithic traces use their own native aggregate grammar."
        )
    probe_filter = _normalize_probe_step_filter(probe_step_filter)
    if probe_filter is not None and architecture != "decomposed":
        raise ValueError(
            "probe_step_filter is only valid for architecture='decomposed'; "
            "monolithic runs do not have decomposed probe slots"
        )
    if probe_only:
        if architecture != "decomposed":
            raise ValueError(
                "probe_only is only valid for architecture='decomposed'; "
                "monolithic runs do not have decomposed probe slots"
            )
        if not decompose:
            raise ValueError("probe_only requires decompose=True")
        if not probe_filter:
            raise ValueError("probe_only requires a non-empty probe_step_filter")

    if score_evidence is None:
        # Default: use the project scorer. Requires a real ModelClient — we
        # fail fast here rather than letting every evidence collapse into
        # an 'abstain' row from the per-evidence exception handler when
        # `client.call(...)` hits None. Tests pass mock score_evidence and
        # never hit this branch.
        if client is None:
            raise ValueError(
                "score_corpus requires either client= (a ModelClient) or "
                "score_evidence= (a callable). Got both None."
            )
        if probe_only:
            from indra_belief.scorers.probes.orchestrator import (
                score_selected_probes,
            )
            probe_kinds = _probe_filter_trace_keys(probe_filter)

            def default_score_evidence(statement, evidence, client):
                return score_selected_probes(
                    statement, evidence, client, probe_kinds
                )
        elif architecture == "monolithic":
            from indra_belief.scorers.monolithic import (
                score_evidence as default_score_evidence,
            )
        else:
            from indra_belief.scorers.scorer import (
                score_evidence as default_score_evidence,
            )
        score_evidence = default_score_evidence  # type: ignore[assignment]

    # Always estimate cost upfront — needed for the threshold gate, AND
    # persisted to score_run.cost_estimate_usd as part of the audit trail
    # (the model_card surfaces it; the column was schema-defined but
    # forever-NULL until this iter).
    stmts = list(stmts)  # type: ignore[assignment]
    from indra_belief.corpus.cost import (
        estimate_cost,
        model_has_known_cost,
        token_cost_usd,
    )
    estimate = estimate_cost(
        stmts,
        model_id=model_id_default,
        architecture=architecture,
        probe_step_filter=probe_filter,
        probe_only=probe_only,
    )  # type: ignore[arg-type]
    cost_estimate = estimate["cost_usd"]

    if cost_threshold_usd is not None and not model_has_known_cost(model_id_default):
        raise ValueError(
            f"score_corpus aborted: model_id {model_id_default!r} is missing "
            "from MODEL_PRICES_PER_M_TOKENS, so observed spend cannot be "
            "enforced. Add pricing or use a listed model before setting "
            "cost_threshold_usd."
        )

    if cost_threshold_usd is not None and cost_estimate > cost_threshold_usd:
        raise ValueError(
            f"score_corpus aborted: estimated cost ${cost_estimate:.2f} "
            f"exceeds threshold ${cost_threshold_usd:.2f} "
            f"({estimate['n_stmts']} stmts × {estimate['n_evidences_est']} ev "
            f"→ ~{estimate['n_llm_calls_est']:,} LLM calls on model {model_id_default}). "
            f"Raise cost_threshold_usd= or sample smaller stmts."
        )

    run_id = run_id or uuid.uuid4().hex
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id.lower()):
        raise ValueError("run_id must be 32 hex chars when provided")
    started_at = datetime.now(timezone.utc)
    indra_version = _detect_indra_version()

    import os as _os
    con.execute(
        """INSERT INTO score_run
           (run_id, scorer_version, indra_version, model_id_default,
            architecture, paired_run_group_id, parent_run_id,
            started_at, status, n_stmts, cost_estimate_usd,
            heartbeat_at, worker_pid)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', 0, ?, ?, ?)""",
        [run_id, scorer_version, indra_version, model_id_default,
         architecture, paired_run_group_id, parent_run_id, started_at,
         cost_estimate, started_at, _os.getpid()],
    )

    n_stmts = 0
    n_evidences = 0
    status = "running"
    cost_actual_so_far = 0.0
    termination_by: str | None = None
    termination_reason: str | None = None

    try:
        for stmt in stmts:
            stmt_hash = _hex(stmt.get_hash(shallow=True))
            evidences = list(getattr(stmt, "evidence", None) or [])
            if not evidences:
                n_stmts += 1
                continue

            for ev in evidences:
                _raise_if_run_canceled(con, run_id)
                try:
                    evhash = _hex(ev.get_source_hash())
                except Exception:
                    evhash = hashlib.sha256(
                        f"{ev.source_api}|{ev.source_id}|{ev.pmid}|{ev.text}".encode("utf-8")
                    ).hexdigest()[:16]

                t0 = time.perf_counter()

                # Phase 3.4 partial: capture deterministic substeps into
                # their own scorer_step rows BEFORE running the aggregate.
                # Lights rail positions 1-3 (and 4-7 for substrate-answered
                # probes) deterministically; LLM-escalated steps + adjudicate
                # remain in the aggregate row until orchestrator emits them.
                if decompose:
                    for det_kind, det_payload, det_substrate in _decompose_steps(stmt, ev):
                        if (
                            det_kind in PROBE_STEP_KINDS
                            and probe_filter is not None
                            and det_kind not in probe_filter
                        ):
                            continue
                        det_step_hash = _step_hash(
                            run_id,
                            stmt_hash, evhash, scorer_version,
                            model_id_default, architecture, det_kind,
                        )
                        try:
                            con.execute(
                                """INSERT OR REPLACE INTO scorer_step
                                   (step_hash, stmt_hash, evidence_hash, run_id,
                                    scorer_version, architecture, model_id, step_kind, is_substrate_answered,
                                    input_payload_json, output_json, latency_ms,
                                    prompt_tokens, out_tokens, finish_reason, error)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                [det_step_hash, stmt_hash, evhash, run_id, scorer_version,
                                 architecture, model_id_default, det_kind, det_substrate,
                                 None, json.dumps(det_payload, default=str),
                                 None, None, None, None, None],
                            )
                        except Exception as e:
                            log.warning("decompose write failed for %s: %s", det_kind, e)

                try:
                    result = score_evidence(stmt, ev, client)
                except Exception as e:
                    log.warning("score_evidence failed for %s/%s: %s", stmt_hash, evhash, e)
                    result = {
                        "score": None, "verdict": "abstain", "confidence": "low",
                        "error": str(e), "call_log": [],
                    }
                _raise_if_run_canceled(con, run_id)
                latency_ms = int((time.perf_counter() - t0) * 1000)

                model_id = result.get("model_id") or model_id_default
                if decompose:
                    for probe_kind, probe_payload, probe_substrate, usage in _probe_trace_steps(result, probe_filter):
                        probe_step_hash = _step_hash(
                            run_id,
                            stmt_hash, evhash, scorer_version,
                            model_id_default, architecture, probe_kind,
                        )
                        try:
                            con.execute(
                                """INSERT OR REPLACE INTO scorer_step
                                   (step_hash, stmt_hash, evidence_hash, run_id,
                                    scorer_version, architecture, model_id, step_kind, is_substrate_answered,
                                    input_payload_json, output_json, latency_ms,
                                    prompt_tokens, out_tokens, finish_reason, error)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                [probe_step_hash, stmt_hash, evhash, run_id, scorer_version,
                                 architecture, model_id_default, probe_kind, probe_substrate,
                                 None, json.dumps(probe_payload, default=str),
                                 usage["latency_ms"], usage["prompt_tokens"], usage["out_tokens"],
                                 usage["finish_reason"], None],
                            )
                        except Exception as e:
                            log.warning("probe trace write failed for %s: %s", probe_kind, e)
                if probe_only:
                    _write_probe_only_merge_aggregate(
                        con,
                        parent_run_id=parent_run_id,
                        child_run_id=run_id,
                        stmt=stmt,
                        evidence=ev,
                        stmt_hash=stmt_hash,
                        evidence_hash=evhash,
                        scorer_version=scorer_version,
                        model_id=model_id_default,
                        architecture=architecture,
                        result=result,
                        probe_filter=probe_filter,
                    )

                # Sum probe call latency / tokens from call_log if present
                call_log = result.get("call_log") or []
                prompt_tokens = _sum_reported_tokens(call_log, "prompt_tokens")
                out_tokens = _sum_reported_tokens(call_log, "out_tokens")
                if out_tokens is None:
                    out_tokens = _nonnegative_int(result.get("tokens"))

                if not probe_only:
                    step_kind = "aggregate"
                    step_hash = _step_hash(
                        run_id,
                        stmt_hash, evhash, scorer_version, model_id,
                        architecture, step_kind,
                    )
                    con.execute(
                        """INSERT OR REPLACE INTO scorer_step
                           (step_hash, stmt_hash, evidence_hash, run_id,
                            scorer_version, architecture, model_id, step_kind, is_substrate_answered,
                            input_payload_json, output_json, latency_ms,
                            prompt_tokens, out_tokens, finish_reason, error)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [step_hash, stmt_hash, evhash, run_id, scorer_version,
                         architecture, model_id, step_kind,
                         None,  # is_substrate_answered N/A on aggregate
                         None,  # input_payload_json N/A on aggregate
                         json.dumps(result, default=str),
                         latency_ms,
                         prompt_tokens,
                         out_tokens,
                         None,
                         result.get("error")],
                    )
                n_evidences += 1
                try:
                    cost_increment = token_cost_usd(
                        model_id,
                        prompt_tokens,
                        out_tokens,
                        on_unknown="raise" if cost_threshold_usd is not None else "zero",
                    )
                except ValueError as e:
                    termination_by = "system"
                    termination_reason = str(e)
                    raise ValueError(
                        "score_corpus aborted: "
                        f"{termination_reason}. Partial scorer_step rows remain append-only."
                    ) from e
                cost_actual_so_far += cost_increment

                if on_evidence is not None:
                    try:
                        result["_cost_actual_increment_usd"] = cost_increment
                        result["_cost_so_far_usd"] = cost_actual_so_far
                        result["_cost_cap_usd"] = cost_threshold_usd
                        on_evidence(stmt_hash, evhash, result)
                    except Exception:
                        log.exception("on_evidence callback raised")
                    finally:
                        result.pop("_cost_actual_increment_usd", None)
                        result.pop("_cost_so_far_usd", None)
                        result.pop("_cost_cap_usd", None)
                if (
                    cost_threshold_usd is not None
                    and cost_actual_so_far > cost_threshold_usd
                ):
                    termination_by = "system"
                    termination_reason = (
                        f"actual cost ${cost_actual_so_far:.4f} exceeded "
                        f"cap ${cost_threshold_usd:.4f} after {n_evidences} evidences"
                    )
                    raise ValueError(
                        "score_corpus aborted: "
                        f"{termination_reason}. "
                        "Partial scorer_step rows remain append-only."
                    )
            n_stmts += 1
        status = "succeeded"
    except Exception:
        status = "failed"
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        # Compute actual cost from observed tokens × each row's model rate.
        # This must use scorer_step.model_id, not score_run.model_id_default:
        # the brake and the persisted audit row need the same pricing basis.
        if probe_only:
            actual_rows = con.execute(
                """SELECT
                     model_id,
                     COALESCE(SUM(prompt_tokens), 0) AS total_in,
                     COALESCE(SUM(out_tokens), 0) AS total_out
                   FROM scorer_step
                   WHERE run_id = ?
                     AND step_kind IN (
                       'subject_role_probe', 'object_role_probe',
                       'relation_axis_probe', 'scope_probe'
                     )
                   GROUP BY model_id""",
                [run_id],
            ).fetchall()
        else:
            actual_rows = con.execute(
                """SELECT
                     model_id,
                     COALESCE(SUM(prompt_tokens), 0) AS total_in,
                     COALESCE(SUM(out_tokens), 0) AS total_out
                   FROM scorer_step
                   WHERE run_id = ? AND step_kind = 'aggregate'
                   GROUP BY model_id""",
                [run_id],
            ).fetchall()
        cost_actual = sum(
            token_cost_usd(model_id or model_id_default, total_in, total_out)
            for model_id, total_in, total_out in actual_rows
        )
        # Two-step finalize so a row already flipped to 'canceled' by an
        # external cancel still gets finished_at + cost_actual filled in
        # for queryability. The cancel-status itself is preserved (the
        # NOT IN guard on the status-update branch); audit columns are
        # written here.
        #
        # cost_actual_usd is overwritten (not COALESCEd): the canceler
        # computes it from a separate DuckDB connection that may not yet
        # see all the worker's just-written scorer_step rows. The worker's
        # own connection has authoritative row visibility — overwriting
        # ensures the audited cost reflects the actual writes, not the
        # canceler's stale snapshot.
        #
        # finished_at, terminated_by, termination_reason are COALESCEd so
        # the canceler's "who and why" survives if it set them first.
        con.execute(
            """UPDATE score_run
               SET finished_at = COALESCE(finished_at, ?),
                   n_stmts = ?,
                   cost_actual_usd = ?,
                   terminated_by = COALESCE(terminated_by, ?),
                   termination_reason = COALESCE(termination_reason, ?)
               WHERE run_id = ?""",
            [
                finished_at,
                n_stmts,
                cost_actual,
                termination_by,
                termination_reason,
                run_id,
            ],
        )
        con.execute(
            """UPDATE score_run
               SET status = ?
               WHERE run_id = ?
                 AND COALESCE(status, 'running')
                     NOT IN ('canceled', 'cancelled', 'aborted', 'pre_started_cancelled')""",
            [status, run_id],
        )
        log.info(
            "score_corpus run_id=%s status=%s n_stmts=%d n_evidences=%d",
            run_id, status, n_stmts, n_evidences,
        )

    # Auto-compute validity at end of successful run unless caller opts out.
    # Tightens the workflow: ingest → score (validity computed) → export.
    if with_validity and status == "succeeded":
        try:
            from indra_belief.corpus.validity import compute_validity
            compute_validity(con, run_id)
        except Exception as e:
            log.warning("auto compute_validity failed for %s: %s", run_id, e)

    return run_id
