#!/bin/bash
# Score the open-weight bedrock tier on the external curator gold (587 ev).
# 4 models concurrent, --workers 6 each (~24 bedrock conns); runner retries throttles.
set -u
cd /Users/noot/Documents/indra-belief-model

export AWS_BEARER_TOKEN_BEDROCK=$(.venv/bin/python -c "[print(l.split('=',1)[1].strip().strip(chr(34)).strip(chr(39))) for l in open('.env') if l.strip().startswith('AWS_BEARER_TOKEN_BEDROCK=')]")

INPUT=data/corpora/external_curator_gold_v1_statements.json
MODELS=(
  bedrock-gemma-4-e2b bedrock-gemma-4-31b bedrock-gpt-oss-20b
  bedrock-nemotron-super-120b bedrock-qwen3-235b bedrock-gpt-oss-120b
  bedrock-qwen3-coder-480b bedrock-minimax-m2.5 bedrock-deepseek-v3.2
  bedrock-kimi-k2.5 bedrock-glm-5
)

printf '%s\n' "${MODELS[@]}" | xargs -P 11 -I {} bash -c '
  m="$1"
  echo "START $m"
  PYTHONPATH=src .venv/bin/python scripts/run_rasmachine_monolithic.py \
    --input '"$INPUT"' --model "$m" \
    --output data/results/external_curator_v1_"$m".jsonl \
    --workers 6 --no-export > /tmp/fleet_"$m".log 2>&1 \
    && echo "DONE  $m" || echo "FAIL  $m"
' _ {}

echo "==== OPEN FLEET COMPLETE ===="
