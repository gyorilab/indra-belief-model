#!/usr/bin/env python3
"""Emit a labelled statement corpus as the shard builder's input TSV.

WHY THIS EXISTS
---------------
A calibration is only valid for the population it was fitted on, and this is
the step that lets a calibration be fitted on PRODUCTION statements travelling
the PRODUCTION path instead of on a curated gold read by a different script.

Two distinct skews motivate it, and the second is the one a gold-eval run
cannot touch no matter how faithful its prompt is:

  PATH        which code builds the request. Now zero -- the batch builder and
              the live scorer render byte-identical user messages (pinned in
              tests/test_build_processed_grounding_shards.py).

  POPULATION  which statements the curve is fitted on. `fit_prevalence` is
              BAKED INTO the artifact and is the anchor of every weight:
              weight_of_evidence = logit(p_hat) - logit(fit_prevalence). Fit on
              a balanced curated gold (0.513) and apply to a corpus at 0.70 and
              every one of 60M weights is displaced by +0.88 log-odds. The
              isotonic's SHAPE has the same problem: its knots are placed where
              the fit population's margins fell, and curator-selected margins
              are not corpus margins.

So: emit the labelled statements as `statement_hash<TAB>statement_json`, run
them through `build_processed_grounding_shards.py` and then the ordinary shard
runner, and fit from THAT. Same builder, same runner, same prompt, same
transport as the 60M run.

USAGE
    python scripts/emit_statement_tsv.py \
        --corpus data/corpora/eval_curation_v1_statements.json \
        --out data/corpora/eval_curation_v1.tsv
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def statement_rows(corpus: list) -> tuple[list[tuple[str, str]], dict]:
    """(hash, json) pairs, plus what was skipped and why.

    The hash is the statement's own ``matches_hash`` — the same identity the
    corpus TSV's first column carries in production, so shards prepared here key
    exactly as shards prepared from the real dump do.
    """
    rows, skipped = [], {"no_matches_hash": 0, "no_evidence": 0}
    for statement in corpus:
        if not isinstance(statement, dict):
            continue
        matches_hash = statement.get("matches_hash")
        if matches_hash in (None, ""):
            skipped["no_matches_hash"] += 1
            continue
        if not statement.get("evidence"):
            # The shard builder filters these itself, but counting them here
            # keeps "how many statements went in" answerable without rerunning.
            skipped["no_evidence"] += 1
        rows.append((str(matches_hash),
                     json.dumps(statement, sort_keys=True, separators=(",", ":"))))
    return rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True,
                    help="JSON list of INDRA statements (data/corpora/*_statements.json)")
    ap.add_argument("--out", required=True, help="TSV path; .gz is written gzipped")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    if not isinstance(corpus, list):
        raise SystemExit(f"{args.corpus}: expected a JSON list of statements")
    rows, skipped = statement_rows(corpus)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no statements carried a matches_hash; nothing to emit")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out.suffix == ".gz" else open
    with opener(out, "wt", encoding="utf-8", newline="") as fh:
        for matches_hash, payload in rows:
            # TAB-separated, and the JSON is separator-compact so it cannot
            # contain a raw tab or newline that would split the row.
            fh.write(f"{matches_hash}\t{payload}\n")

    print(f"  wrote {len(rows):,} statements -> {out}")
    if any(skipped.values()):
        print(f"  skipped: {skipped}")
    print("\n  next:")
    print(f"    python scripts/build_processed_grounding_shards.py --input {out} \\")
    print("        --output-dir <shards>")
    print("    python scripts/run_vllm_processed_shards.py --input-dir <shards> \\")
    print("        --output-dir <results> --workers 64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
