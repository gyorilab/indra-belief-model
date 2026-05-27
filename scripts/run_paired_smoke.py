"""Create a 10-statement paired architecture smoke run in DuckDB.

This is the evidence gate before architecture comparison UI work. It runs
the native decomposed and monolithic scorer dispatch paths serially against
the same synthetic INDRA corpus, using a deterministic local model client so
no provider API is called.

Default output:
  $TMPDIR/indra-agent-trace-hypergraph/paired_smoke.duckdb

Usage:
  PYTHONPATH=src .venv/bin/python scripts/run_paired_smoke.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from indra.statements import (
    Acetylation,
    Activation,
    Agent,
    Complex,
    DecreaseAmount,
    Dephosphorylation,
    Evidence,
    IncreaseAmount,
    Inhibition,
    Phosphorylation,
    Translocation,
    Ubiquitination,
)

from indra_belief.corpus import apply_schema, ingest_statements, score_corpus
from indra_belief.model_client import ModelResponse


DEFAULT_MODEL_ID = "smoke-local"


class SmokeModelClient:
    """Deterministic stand-in for ModelClient.

    The scorer paths still build their normal prompts and call `client.call`.
    This client returns small valid JSON payloads for each native call kind and
    records call telemetry for `pop_call_log()`, so scorer_step cost/token
    fields are populated without network/API spend.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_name = model_id
        self.config = {"model_id": model_id}
        self._call_log: list[dict] = []

    def call(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.1,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        kind: str = "unknown",
    ) -> ModelResponse:
        del response_format, reasoning_effort
        t0 = time.time()
        content = self._content_for(kind, messages)
        prompt_tokens = max(
            1,
            (len(system or "") + sum(len(m.get("content", "") or "") for m in messages))
            // 4,
        )
        out_tokens = max(1, len(content) // 4)
        self._call_log.append({
            "kind": kind,
            "duration_s": round(time.time() - t0, 4),
            "prompt_tokens": prompt_tokens,
            "out_tokens": out_tokens,
            "finish_reason": "stop",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model_id": self.model_name,
        })
        return ModelResponse(
            content=content,
            reasoning="",
            tokens=out_tokens,
            raw_text=content,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
        )

    def pop_call_log(self) -> list[dict]:
        out = list(self._call_log)
        self._call_log.clear()
        return out

    @staticmethod
    def _content_for(kind: str, messages: list[dict]) -> str:
        last = (messages[-1].get("content", "") if messages else "").lower()
        if kind == "verify_grounding":
            return json.dumps({
                "status": "mentioned",
                "rationale": "smoke client treats synthetic named entities as mentioned",
            })
        if kind == "probe_subject_role":
            return json.dumps({
                "answer": "present_as_subject",
                "rationale": "synthetic evidence states the claim subject as actor",
            })
        if kind == "probe_object_role":
            return json.dumps({
                "answer": "present_as_object",
                "rationale": "synthetic evidence states the claim object as target",
            })
        if kind == "probe_relation_axis":
            return json.dumps({
                "answer": "direct_sign_match",
                "rationale": "synthetic evidence directly states the claimed relation",
            })
        if kind == "probe_scope":
            scope = "negated" if " not " in last or "does not" in last else "asserted"
            return json.dumps({
                "answer": scope,
                "rationale": "synthetic evidence has explicit scope",
            })
        return json.dumps({
            "verdict": "correct",
            "confidence": "high",
        })


def _ev(text: str, pmid: str) -> Evidence:
    return Evidence(
        source_api="paired_smoke",
        pmid=pmid,
        text=text,
        epistemics={"direct": True},
    )


def build_smoke_statements() -> list:
    """Ten one-evidence INDRA statements with stable overlap keys."""
    stmts = [
        Phosphorylation(
            Agent("MAP2K1", db_refs={"HGNC": "6840"}),
            Agent("MAPK1", db_refs={"HGNC": "6871"}),
            residue="T",
            position="185",
            evidence=[_ev("MAP2K1 phosphorylates MAPK1 at T185.", "900001")],
        ),
        Activation(
            Agent("RAF1", db_refs={"HGNC": "9829"}),
            Agent("MAP2K1", db_refs={"HGNC": "6840"}),
            evidence=[_ev("RAF1 activates MAP2K1.", "900002")],
        ),
        Inhibition(
            Agent("DUSP6", db_refs={"HGNC": "3072"}),
            Agent("MAPK1", db_refs={"HGNC": "6871"}),
            evidence=[_ev("DUSP6 inhibits MAPK1 activity.", "900003")],
        ),
        IncreaseAmount(
            Agent("TP53", db_refs={"HGNC": "11998"}),
            Agent("CDKN1A", db_refs={"HGNC": "1784"}),
            evidence=[_ev("TP53 increases CDKN1A expression.", "900004")],
        ),
        DecreaseAmount(
            Agent("MDM2", db_refs={"HGNC": "6973"}),
            Agent("TP53", db_refs={"HGNC": "11998"}),
            evidence=[_ev("MDM2 decreases TP53 abundance.", "900005")],
        ),
        Complex(
            [Agent("GRB2", db_refs={"HGNC": "4566"}), Agent("SOS1", db_refs={"HGNC": "11187"})],
            evidence=[_ev("GRB2 binds SOS1 in a complex.", "900006")],
        ),
        Dephosphorylation(
            Agent("DUSP6", db_refs={"HGNC": "3072"}),
            Agent("MAPK1", db_refs={"HGNC": "6871"}),
            residue="T",
            position="185",
            evidence=[_ev("DUSP6 dephosphorylates MAPK1 at T185.", "900007")],
        ),
        Ubiquitination(
            Agent("MDM2", db_refs={"HGNC": "6973"}),
            Agent("TP53", db_refs={"HGNC": "11998"}),
            evidence=[_ev("MDM2 ubiquitinates TP53.", "900008")],
        ),
        Acetylation(
            Agent("EP300", db_refs={"HGNC": "3373"}),
            Agent("TP53", db_refs={"HGNC": "11998"}),
            evidence=[_ev("EP300 acetylates TP53.", "900009")],
        ),
        Translocation(
            Agent("AKT1", db_refs={"HGNC": "391"}),
            from_location="cytoplasm",
            to_location="nucleus",
            evidence=[_ev("AKT1 translocates from the cytoplasm to the nucleus.", "900010")],
        ),
    ]
    for stmt in stmts:
        stmt.belief = 0.85
    return stmts


def default_output_dir() -> Path:
    root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    return root / "indra-agent-trace-hypergraph"


def run_paired_smoke(
    *,
    db_path: Path,
    pair_id: str,
    model_id: str = DEFAULT_MODEL_ID,
    scorer_version: str = "paired-smoke-v1",
) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        apply_schema(con)
        stmts = build_smoke_statements()
        ingest_statements(con, stmts, source_dump_id="paired_smoke_synthetic")

        runs: dict[str, str] = {}
        for architecture in ("decomposed", "monolithic"):
            client = SmokeModelClient(model_id)
            runs[architecture] = score_corpus(
                con,
                stmts,
                client=client,
                scorer_version=scorer_version,
                model_id_default=model_id,
                architecture=architecture,  # type: ignore[arg-type]
                paired_run_group_id=pair_id,
                decompose=(architecture == "decomposed"),
                cost_threshold_usd=10.0,
            )

        overlap = con.execute(
            """
            SELECT COUNT(*)
              FROM scorer_step d
              JOIN scorer_step m
                ON d.stmt_hash = m.stmt_hash
               AND d.evidence_hash = m.evidence_hash
             WHERE d.run_id = ?
               AND m.run_id = ?
               AND d.step_kind = 'aggregate'
               AND m.step_kind = 'aggregate'
            """,
            [runs["decomposed"], runs["monolithic"]],
        ).fetchone()[0]
        run_rows = con.execute(
            """
            SELECT architecture, run_id, paired_run_group_id, status, n_stmts,
                   cost_estimate_usd, cost_actual_usd
              FROM score_run
             WHERE run_id IN (?, ?)
             ORDER BY architecture
            """,
            [runs["decomposed"], runs["monolithic"]],
        ).fetchall()
        step_counts = con.execute(
            """
            SELECT architecture, step_kind, COUNT(*) AS n
              FROM scorer_step
             WHERE run_id IN (?, ?)
             GROUP BY architecture, step_kind
             ORDER BY architecture, step_kind
            """,
            [runs["decomposed"], runs["monolithic"]],
        ).fetchall()
        return {
            "db_path": str(db_path),
            "pair_id": pair_id,
            "source_dump_id": "paired_smoke_synthetic",
            "model_id": model_id,
            "no_provider_api": True,
            "scorer_version": scorer_version,
            "runs": runs,
            "n_statements": len(stmts),
            "overlap_evidences": overlap,
            "run_rows": [
                {
                    "architecture": r[0],
                    "run_id": r[1],
                    "paired_run_group_id": r[2],
                    "status": r[3],
                    "n_stmts": r[4],
                    "cost_estimate_usd": r[5],
                    "cost_actual_usd": r[6],
                }
                for r in run_rows
            ],
            "step_counts": [
                {"architecture": r[0], "step_kind": r[1], "n": r[2]}
                for r in step_counts
            ],
        }
    finally:
        con.close()


def _fixture_step_hash(
    *,
    run_id: str,
    stmt_hash: str,
    evidence_hash: str,
    scorer_version: str,
    model_id: str,
    architecture: str,
    fixture_name: str,
) -> str:
    raw = (
        f"{run_id}|{stmt_hash}|{evidence_hash}|{scorer_version}|{model_id}|"
        f"{architecture}|aggregate|{fixture_name}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _insert_fixture_aggregate(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    architecture: str,
    stmt_hash: str,
    evidence_hash: str,
    scorer_version: str,
    model_id: str,
    verdict: str,
    score: float,
    confidence: str,
    fixture_name: str,
) -> None:
    output = {
        "score": score,
        "verdict": verdict,
        "confidence": confidence,
        "fixture": fixture_name,
    }
    con.execute(
        """
        INSERT INTO scorer_step
           (step_hash, stmt_hash, evidence_hash, run_id,
            scorer_version, architecture, model_id, step_kind, is_substrate_answered,
            input_payload_json, output_json, latency_ms,
            prompt_tokens, out_tokens, finish_reason, error, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'aggregate', NULL, ?, ?, 0, 0, 0, 'fixture', NULL, ?)
        """,
        [
            _fixture_step_hash(
                run_id=run_id,
                stmt_hash=stmt_hash,
                evidence_hash=evidence_hash,
                scorer_version=scorer_version,
                model_id=model_id,
                architecture=architecture,
                fixture_name=fixture_name,
            ),
            stmt_hash,
            evidence_hash,
            run_id,
            scorer_version,
            architecture,
            model_id,
            json.dumps({"fixture": fixture_name}, sort_keys=True),
            json.dumps(output, sort_keys=True),
            datetime.now(timezone.utc),
        ],
    )


def run_exact_support_divergence_fixture(
    *,
    db_path: Path,
    pair_id: str = "paired_exact_support_divergence",
    model_id: str = DEFAULT_MODEL_ID,
    scorer_version: str = "paired-smoke-divergence-v1",
) -> dict:
    """Create a pair where exact label agreement differs from support-state agreement.

    The base paired smoke run is preserved. The fixture appends later aggregate
    rows for two different evidence rows. Those rows make exact labels disagree
    in both non-supported directions (`abstain` vs `incorrect` and `incorrect`
    vs `abstain`) while both sides remain in the support-state matrix's
    `not supported` bucket.
    """
    summary = run_paired_smoke(
        db_path=db_path,
        pair_id=pair_id,
        model_id=model_id,
        scorer_version=scorer_version,
    )
    con = duckdb.connect(str(db_path))
    fixture_name = "exact_support_divergence"
    try:
        runs = summary["runs"]
        divergence_targets = con.execute(
            """
            WITH
              m AS (
                SELECT stmt_hash, evidence_hash,
                       json_extract_string(output_json, '$.verdict') AS verdict,
                       ROW_NUMBER() OVER (
                         PARTITION BY stmt_hash, evidence_hash
                         ORDER BY started_at DESC, step_hash DESC
                       ) AS rn
                  FROM scorer_step
                 WHERE run_id = ?
                   AND step_kind = 'aggregate'
                   AND evidence_hash IS NOT NULL
              ),
              d AS (
                SELECT stmt_hash, evidence_hash,
                       json_extract_string(output_json, '$.verdict') AS verdict,
                       ROW_NUMBER() OVER (
                         PARTITION BY stmt_hash, evidence_hash
                         ORDER BY started_at DESC, step_hash DESC
                       ) AS rn
                  FROM scorer_step
                 WHERE run_id = ?
                   AND step_kind = 'aggregate'
                   AND evidence_hash IS NOT NULL
              )
            SELECT m.stmt_hash, m.evidence_hash
              FROM m
              JOIN d USING (stmt_hash, evidence_hash)
             WHERE m.rn = 1
               AND d.rn = 1
               AND m.verdict = 'correct'
               AND d.verdict = 'correct'
             ORDER BY m.stmt_hash, m.evidence_hash
             LIMIT 2
            """,
            [runs["monolithic"], runs["decomposed"]],
        ).fetchall()
        if len(divergence_targets) < 2:
            raise RuntimeError("could not find two shared correct/correct aggregate rows for divergence fixture")

        divergence_specs = [
            {
                "monolithic": {"verdict": "abstain", "score": 0.45, "confidence": "low"},
                "decomposed": {"verdict": "incorrect", "score": 0.2, "confidence": "high"},
            },
            {
                "monolithic": {"verdict": "incorrect", "score": 0.1, "confidence": "high"},
                "decomposed": {"verdict": "abstain", "score": 0.4, "confidence": "low"},
            },
        ]
        divergence_rows = []
        for (stmt_hash, evidence_hash), spec in zip(divergence_targets, divergence_specs, strict=True):
            for architecture in ("monolithic", "decomposed"):
                row_spec = spec[architecture]
                _insert_fixture_aggregate(
                    con,
                    run_id=runs[architecture],
                    architecture=architecture,
                    stmt_hash=stmt_hash,
                    evidence_hash=evidence_hash,
                    scorer_version=scorer_version,
                    model_id=model_id,
                    verdict=row_spec["verdict"],
                    score=row_spec["score"],
                    confidence=row_spec["confidence"],
                    fixture_name=fixture_name,
                )
            divergence_rows.append({
                "stmt_hash": stmt_hash,
                "evidence_hash": evidence_hash,
                "monolithic_verdict": spec["monolithic"]["verdict"],
                "decomposed_verdict": spec["decomposed"]["verdict"],
            })

        metrics = con.execute(
            """
            WITH
              m AS (
                SELECT stmt_hash, evidence_hash,
                       json_extract_string(output_json, '$.verdict') AS verdict,
                       ROW_NUMBER() OVER (
                         PARTITION BY stmt_hash, evidence_hash
                         ORDER BY started_at DESC, step_hash DESC
                       ) AS rn
                  FROM scorer_step
                 WHERE run_id = ?
                   AND step_kind = 'aggregate'
                   AND evidence_hash IS NOT NULL
              ),
              d AS (
                SELECT stmt_hash, evidence_hash,
                       json_extract_string(output_json, '$.verdict') AS verdict,
                       ROW_NUMBER() OVER (
                         PARTITION BY stmt_hash, evidence_hash
                         ORDER BY started_at DESC, step_hash DESC
                       ) AS rn
                  FROM scorer_step
                 WHERE run_id = ?
                   AND step_kind = 'aggregate'
                   AND evidence_hash IS NOT NULL
              ),
              j AS (
                SELECT m.stmt_hash, m.evidence_hash,
                       m.verdict AS monolithic_verdict,
                       d.verdict AS decomposed_verdict
                  FROM m
                  JOIN d USING (stmt_hash, evidence_hash)
                 WHERE m.rn = 1 AND d.rn = 1
              )
            SELECT
              COUNT(*) AS n_overlap,
              SUM(CASE WHEN monolithic_verdict = decomposed_verdict THEN 1 ELSE 0 END) AS exact_label_agreement_n,
              SUM(CASE
                    WHEN (monolithic_verdict = 'correct') = (decomposed_verdict = 'correct')
                    THEN 1 ELSE 0 END) AS support_state_agreement_n,
              SUM(CASE WHEN monolithic_verdict='correct' AND decomposed_verdict='correct' THEN 1 ELSE 0 END) AS both_supported_n,
              SUM(CASE WHEN monolithic_verdict='correct' AND decomposed_verdict <> 'correct' THEN 1 ELSE 0 END) AS monolithic_only_supported_n,
              SUM(CASE WHEN monolithic_verdict <> 'correct' AND decomposed_verdict='correct' THEN 1 ELSE 0 END) AS decomposed_only_supported_n,
              SUM(CASE WHEN monolithic_verdict <> 'correct' AND decomposed_verdict <> 'correct' THEN 1 ELSE 0 END) AS neither_supported_n
            FROM j
            """,
            [runs["monolithic"], runs["decomposed"]],
        ).fetchone()
        step_counts = con.execute(
            """
            SELECT architecture, step_kind, COUNT(*) AS n
              FROM scorer_step
             WHERE run_id IN (?, ?)
             GROUP BY architecture, step_kind
             ORDER BY architecture, step_kind
            """,
            [runs["monolithic"], runs["decomposed"]],
        ).fetchall()
        summary["step_counts"] = [
            {"architecture": r[0], "step_kind": r[1], "n": r[2]}
            for r in step_counts
        ]
        summary["fixture"] = {
            "name": fixture_name,
            "pair_id": pair_id,
            "divergence_rows": divergence_rows,
            "n_divergence_rows": len(divergence_rows),
            "exact_label_agreement_n": int(metrics[1]),
            "support_state_agreement_n": int(metrics[2]),
            "n_overlap": int(metrics[0]),
            "both_supported_n": int(metrics[3]),
            "monolithic_only_supported_n": int(metrics[4]),
            "decomposed_only_supported_n": int(metrics[5]),
            "neither_supported_n": int(metrics[6]),
        }
        return summary
    finally:
        con.close()


def main() -> int:
    out_dir = default_output_dir()
    default_pair_id = "paired_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=out_dir / "paired_smoke.duckdb")
    parser.add_argument("--pair-id", default=default_pair_id)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--scorer-version", default="paired-smoke-v1")
    parser.add_argument("--fixture", choices=["smoke", "exact-support-divergence"], default="smoke")
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    if args.fixture == "exact-support-divergence":
        summary = run_exact_support_divergence_fixture(
            db_path=args.db,
            pair_id=args.pair_id,
            model_id=args.model_id,
            scorer_version=args.scorer_version,
        )
    else:
        summary = run_paired_smoke(
            db_path=args.db,
            pair_id=args.pair_id,
            model_id=args.model_id,
            scorer_version=args.scorer_version,
        )
    summary_path = args.summary
    if summary_path is None:
        summary_path = args.db.parent / f"{args.pair_id}.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "ok": True,
        "db_path": summary["db_path"],
        "summary_path": str(summary_path),
        "pair_id": summary["pair_id"],
        "runs": summary["runs"],
        "overlap_evidences": summary["overlap_evidences"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
