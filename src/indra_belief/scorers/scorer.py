"""Evidence quality scorer entry point.

One architecture: a single LLM call per (Statement, Evidence) with
type-adaptive contrastive few-shot retrieval, implemented in
`indra_belief.scorers.monolithic.*` and exposing
`score_evidence(stmt, ev, client) -> dict` and
`score_statement(stmt, client) -> list[dict]`.

There used to be a second, `decomposed`: parse_claim → substrate_route → four
probes → ProbeBundle → adjudicate, under `scorers.probes.*`, selected by an
`--arch` flag. It lost to this one on holdout_cc (F1 0.751 vs 0.657, McNemar
p<10^-4) and was kept for a while as a comparison baseline. It has been removed
along with its dependency tail (context_builder, relation_patterns, commitments,
parse_claim, grounding, kg_signal, the panel variant); git history holds it. The
flag is gone with it, so this module no longer dispatches — it runs the scorer.

Run:
    PYTHONPATH=src python -m indra_belief.scorers.scorer
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from indra_belief.model_client import ModelClient
# Re-exported for backward compatibility: older scripts import these names from
# this module. They are the monolithic scorer's, which is now the only one.
from indra_belief.scorers.monolithic import score_evidence, score_statement  # noqa: F401


log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent


def main():
    import argparse
    from indra_belief.data.corpus import CorpusIndex

    parser = argparse.ArgumentParser(
        description="Evidence quality scorer (INDRA native)"
    )
    parser.add_argument("--model", default="gemma-remote")
    parser.add_argument("--holdout", required=True,
                        help="eval gold JSONL to score, e.g. "
                             "data/benchmark/holdout_large.jsonl")
    parser.add_argument("--output",
                        default=str(ROOT / "data" / "results" / "scorer_output.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing output file (skip scored records)")
    args = parser.parse_args()

    from indra_belief.scorers.monolithic import score_evidence as score_fn

    index = CorpusIndex()
    records = index.build_records(args.holdout)
    if args.limit:
        records = records[: args.limit]

    scored_hashes: set = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                for lineno, line in enumerate(f, start=1):
                    try:
                        r = json.loads(line)
                        scored_hashes.add(r.get("source_hash"))
                    except json.JSONDecodeError:
                        # Skip corrupt NDJSON line; surface it for data-integrity visibility.
                        log.warning(
                            "resume: skipping corrupt JSON line %d in %s: %r",
                            lineno, resume_path, line.rstrip("\n")[:200],
                        )
            print(f"Resuming: {len(scored_hashes)} records already scored")

    print(f"\nScorer: {len(records)} records, model={args.model}")

    client = ModelClient(args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    out_fh = open(output_path, mode)

    correct = 0
    total_parsed = 0
    t_start = time.time()

    for i, record in enumerate(records):
        if record.source_hash in scored_hashes:
            continue

        result = score_fn(record.statement, record.evidence, client)

        gt_correct = record.tag == "correct"
        verdict = result.get("verdict")
        llm_correct = (verdict == "correct") if verdict else None
        if llm_correct is not None:
            total_parsed += 1
            if llm_correct == gt_correct:
                correct += 1

        result.update({
            "source_hash": record.source_hash,
            "tag": record.tag or "",
            "subject": record.subject,
            "stmt_type": record.stmt_type,
            "object": record.object,
            "arch": "monolithic",
        })

        r_save = {k: v for k, v in result.items() if k != "raw_text"}
        r_save["raw_text_preview"] = result.get("raw_text") or ""  # full output — no cap
        out_fh.write(json.dumps(r_save) + "\n")
        out_fh.flush()

        acc = correct / total_parsed * 100 if total_parsed > 0 else 0
        mark = ("✓" if (llm_correct == gt_correct)
                else ("✗" if llm_correct is not None else "?"))
        print(f"  [{i + 1:3d}/{len(records)}] {mark} "
              f"{record.subject:>10s} [{record.stmt_type:>15s}] "
              f"{record.object:10s} → "
              f"{verdict or 'PARSE':>9s} acc={acc:.1f}%")

    out_fh.close()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {correct}/{total_parsed} = "
          f"{correct / max(total_parsed, 1) * 100:.1f}% in {elapsed/60:.1f}min")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
