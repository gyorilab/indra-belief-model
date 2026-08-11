#!/usr/bin/env bash
# Serve a reader locally on Apple Silicon via MLX, with token logprobs.
#
# WHY THIS EXISTS. Every hosted arm we have is logprob-blind for gemma: the
# Bedrock responses route accepts `top_logprobs` and returns an EMPTY ARRAY
# (research/serving_architecture.md:129-130), and the noot-1 gateway is down.
# mlx_lm.server does return them, in the OpenAI `choices[0].logprobs.content[]`
# shape, so this is currently the only substrate on which the renormalised
# label probability of research/scoring_methods.md §2.2 can be measured at all.
#
# The MLX stack lives in its OWN virtualenv on purpose. It pulls torch, which
# requires sympy>=1.13.3, while the project's own pysb/INDRA dependency pins
# sympy<1.12 — the two cannot coexist. Nothing in the project imports mlx; the
# scorer reaches this server over HTTP as a plain openai_compat backend.
#
# Setup (once):
#   uv venv ~/.venvs/mlx-serve --python 3.12
#   VIRTUAL_ENV=~/.venvs/mlx-serve uv pip install mlx-lm
#
# Usage:
#   scripts/serve_mlx.sh                    # gemma-4-26b-a4b 8-bit on :8085
#   MODEL=mlx-community/gemma-4-31b-it-8bit PORT=8084 scripts/serve_mlx.sh
#
# The defaults match the `local-gemma-4-26b` entry in model_client.LOCAL_MODELS;
# changing one without the other will produce a 404 on every call.
set -euo pipefail

MLX_VENV="${MLX_VENV:-$HOME/.venvs/mlx-serve}"
MODEL="${MODEL:-mlx-community/gemma-4-26b-a4b-it-8bit}"
PORT="${PORT:-8085}"
HOST="${HOST:-127.0.0.1}"

if [[ ! -x "$MLX_VENV/bin/python" ]]; then
  echo "error: no MLX venv at $MLX_VENV — see the setup lines in this script's header" >&2
  exit 1
fi

# temp 0.0 is the server default and is what we want for scoring. It does NOT
# affect the returned distribution: mlx_lm computes
#   logprobs = logits - logsumexp(logits)
# BEFORE the sampler runs (generate.py:420-421), so temperature, top-p and
# top-k change which token is SAMPLED but never the logprobs we read. p_raw is
# therefore a raw model posterior and is reproducible even where the sampled
# verdict is not.
echo "serving $MODEL on http://$HOST:$PORT/v1  (first run downloads weights)"
exec "$MLX_VENV/bin/python" -m mlx_lm server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --temp 0.0 \
  --log-level INFO
