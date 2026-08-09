from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief import ModelClient
from indra_belief.scorers.monolithic.scorer import (
    score_statement as score_evidence,
)


def stmt_label(stmt) -> tuple[str, str, str]:
    agents = [a for a in (stmt.agent_list() or []) if a is not None]
    subject = agents[0].name if len(agents) >= 1 else "?"
    obj = agents[1].name if len(agents) >= 2 else "?"
    return type(stmt).__name__, subject, obj


def build_jobs(statements) -> list[tuple[int, Any, int, Any]]:
    jobs = []
    for stmt_i, stmt in enumerate(statements):
        for ev_i, ev in enumerate(stmt.evidence or []):
            jobs.append((stmt_i, stmt, ev_i, ev))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="indra_benchmark_corpus.pkl")
    parser.add_argument("--output", default="benchmark_results.jsonl")
    parser.add_argument("--summary", default="benchmark_result.txt")
    parser.add_argument("--model", default="vllm-local")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    with open(args.input, "rb") as fh:
        statements = pickle.load(fh)
    if args.limit:
        statements = statements[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(statements)
    total_evidence = len(jobs)

    tls = threading.local()

    def client() -> ModelClient:
        if not hasattr(tls, "client"):
            tls.client = ModelClient(args.model)
        return tls.client

    def score_job(job: tuple[int, Any, int, Any]) -> dict:
        stmt_i, stmt, ev_i, ev = job
        stmt_type, subject, obj = stmt_label(stmt)

        start = time.perf_counter()
        try:
            result = score_evidence(
                stmt,
                ev,
                client(),
                max_tokens=args.max_tokens,
            )
            return {
                "stmt_i": stmt_i,
                "evidence_i": ev_i,
                "stmt_type": stmt_type,
                "subject": subject,
                "object": obj,
                "verdict": result.get("verdict"),
                "confidence": result.get("confidence"),
                "score": result.get("score"),
                "tier": result.get("tier"),
                "tokens": result.get("tokens"),
                "latency_s": round(time.perf_counter() - start, 3),
            }
        except Exception as exc:
            return {
                "stmt_i": stmt_i,
                "evidence_i": ev_i,
                "stmt_type": stmt_type,
                "subject": subject,
                "object": obj,
                "verdict": None,
                "confidence": None,
                "score": None,
                "tier": "error",
                "tokens": None,
                "latency_s": round(time.perf_counter() - start, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }

    start = time.perf_counter()
    completed = 0
    errors = 0

    with output_path.open("w", buffering=1) as out_fh:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(score_job, job) for job in jobs[: args.workers]}
            next_job = args.workers

            with tqdm(total=total_evidence, unit="ev") as bar:
                while pending:
                    finished, pending = cf.wait(
                        pending, return_when=cf.FIRST_COMPLETED
                    )
                    for future in finished:
                        row = future.result()
                        out_fh.write(json.dumps(row, default=str) + "\n")
                        completed += 1
                        errors += int(row.get("tier") == "error")
                        bar.update(1)

                        if next_job < len(jobs):
                            pending.add(pool.submit(score_job, jobs[next_job]))
                            next_job += 1

    elapsed = time.perf_counter() - start
    processed = len(jobs)
    rate = processed / elapsed if elapsed else 0.0

    with open(args.summary, "w") as fh:
        fh.write(f"model: {args.model}\n")
        fh.write(f"workers: {args.workers}\n")
        fh.write(f"statements: {len(statements)}\n")
        fh.write(f"evidence_total: {total_evidence}\n")
        fh.write(f"processed: {processed}\n")
        fh.write(f"completed: {completed}\n")
        fh.write(f"errors_this_run: {errors}\n")
        fh.write(f"seconds: {elapsed:.2f}\n")
        fh.write(f"evidence_per_second: {rate:.4f}\n")

    print(f"wrote {output_path}")
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
