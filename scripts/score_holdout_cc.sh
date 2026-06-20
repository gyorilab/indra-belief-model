#!/usr/bin/env bash
# Score holdout_cc with both readers for the C1.2 authoritative calibration
# confirmation. Mirrors EXACTLY how eval_curation_v1 was scored (same builder,
# same runner, same remote models) so the C1 weights fit on eval_curation_v1
# transfer to an INDEPENDENT test set.
#
# Run on a host with: `indra` installed in the venv, the benchmark corpus
# (data/benchmark/indra_benchmark_corpus.json.gz), and Tailscale reach to the
# gateway (100.97.101.59:11434).
#
# IMPORTANT: the gateway serves ONE model at a time (gemma OR medpsy). The two
# score steps are therefore SEQUENTIAL with a manual model swap between them —
# hence separate sub-commands rather than one run. medpsy is ~16x faster than
# gemma (eval_curation_v1: ~47min vs ~12.8h for 1606 ev). holdout_cc is 500 ev
# / 346 stmts → rough budget: medpsy ~15min, gemma ~4h single-worker (~1h with
# --workers 4 if the gateway is launched with -np 4).
#
# Usage:
#   scripts/score_holdout_cc.sh build                 # 1. corpus -> statements json (needs indra)
#   scripts/score_holdout_cc.sh score remote-gemma-4-26b    # 2. (gateway serving gemma)
#   scripts/score_holdout_cc.sh score remote-medpsy-4b   # 3. (after swapping gateway to medpsy)
#   scripts/score_holdout_cc.sh confirm               # 4. local: fit on eval_curation_v1, test on holdout_cc
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

GOLD=data/benchmark/holdout_cc.jsonl
STMTS=data/corpora/holdout_cc_statements.json

case "${1:-}" in
  build)
    python scripts/eval_to_statements_json.py --input "$GOLD" --output "$STMTS"
    ;;
  score)
    MODEL="${2:?usage: score <remote-gemma-4-26b|remote-medpsy-4b>}"
    case "$MODEL" in
      remote-gemma-4-26b)  OUT=data/results/holdout_cc_gemma.jsonl ;;
      remote-medpsy-4b) OUT=data/results/holdout_cc_medpsy.jsonl ;;
      *) echo "unknown model $MODEL" >&2; exit 2 ;;
    esac
    python scripts/run_rasmachine_monolithic.py \
        --model "$MODEL" --input "$STMTS" --output "$OUT" \
        --workers 4 --row-error-policy record
    ;;
  confirm)
    python scripts/calibration_confirm.py
    ;;
  *)
    echo "usage: $0 {build|score <model>|confirm}" >&2; exit 2 ;;
esac
