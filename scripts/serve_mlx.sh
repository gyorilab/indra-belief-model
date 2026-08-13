#!/usr/bin/env bash
# Serve a reader locally on Apple Silicon via MLX, with token logprobs.
#
# WHY THIS EXISTS. Every hosted arm we have is logprob-blind for gemma-4: the
# Bedrock responses route accepts `top_logprobs` and returns an EMPTY ARRAY
# (research/serving_architecture.md, "Provider" — "SETTLED 2026-08-11 [M]:
# gemma-4 cannot return logprobs on Bedrock by ANY route"; gemma-3-27b-it, by
# contrast, DOES return real top_logprobs there, which is why this is scoped to
# gemma-4 and not to gemma), and the noot-1 gateway is down.
# mlx_lm.server does return them, in the OpenAI `choices[0].logprobs.content[]`
# shape, so this is currently the only substrate on which the renormalised
# label probability of research/scoring_methods.md §2.2 can be measured at all.
#
# The MLX stack lives in its OWN virtualenv BY CHOICE, not by necessity. There
# is no dependency conflict, and an earlier revision of this comment claiming one
# was wrong: `VIRTUAL_ENV=$PWD/.venv uv pip install --dry-run mlx-lm` resolves
# clean and would install only mlx 0.32.0, mlx-lm 0.31.3 and mlx-metal 0.32.0 —
# nothing upgraded, nothing removed — and mlx-lm declares neither torch nor sympy.
# torch, sympy and pysb do coexist in .venv today (`.venv/bin/python -c "import
# torch,sympy,pysb"` exits 0); `pip check` flags torch's sympy>=1.13.3 floor,
# which is a violated declaration, not a failed import.
#
# What the split buys is a judgement, not a necessity. The cost it avoids is
# ~192 MB across those three distributions, ~188 MB of it the Apple-Silicon-only
# mlx-metal binary. Reproduce with:
#   ~/.venvs/mlx-serve/bin/python -c 'from importlib.metadata import distribution as D; print([(n, D(n).version, round(sum(D(n).locate_file(f).stat().st_size for f in D(n).files)/1e6, 1)) for n in ("mlx", "mlx-lm", "mlx-metal")])'
#   -> [('mlx', '0.32.0', 1.8), ('mlx-lm', '0.31.3', 1.6), ('mlx-metal', '0.32.0', 188.4)]
# Do NOT quote `du -sm ~/.venvs/mlx-serve` (316 MiB) as that cost: that is the
# whole serving venv, most of whose non-MLX bulk (numpy, transformers) .venv
# already carries. mlx-metal also unpacks INTO the mlx/ import dir, which is why
# `du -sm` on site-packages/mlx reports 189 MiB of blocks and there is no
# mlx_metal/ beside it.
#
# uv.lock has no mlx entry (`grep -c '^name = "mlx' uv.lock` -> 0) and CI runs on
# ubuntu-latest (.github/workflows/ci.yml:9). The marker at work there is mlx-lm's
# OWN — it requires `mlx>=0.31.2; platform_system == "Darwin"`, so off Darwin a
# resolver never REQUESTS mlx; it is not that mlx would refuse to install (mlx
# 0.32.0 carries no Darwin marker; its Darwin-only piece is mlx-metal==0.32.0).
# The scorer never imports mlx — it reaches this server over HTTP as a plain
# openai_compat backend. The one in-process MLX path,
# scripts/run_probe_battery.py, imports mlx_lm lazily inside its read functions
# (zero module-scope mlx imports) and is run under ~/.venvs/mlx-serve/bin/python.
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

# ---------------------------------------------------------------------------
# LOCAL PATCH REQUIRED: the top_logprobs cap
#
# Stock mlx_lm.server hard-codes, in `server.py`:
#     self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])
#
# 11 is not enough to read a verdict probability. At a forced verdict position
# the distribution is dominated by JSON formatting tokens ({", ", ```), and the
# LOSING label was measured at rank 42 / 83 / 168 across four cases on
# 2026-08-13. At k=11, three of those four returned no usable p_raw; at k=2000
# all four did. `_format_top_logprobs` already handles any n via argpartition,
# so only the validator constant blocks it.
#
# The patch, applied in this venv (original kept at server.py.orig-cap11):
#     max_val=11  ->  max_val=262144
#
# WHY THIS MATTERS IF THE VENV IS REBUILT: `pip install --upgrade mlx-lm` or a
# fresh venv silently restores 11. The failure is QUIET — p_raw comes back nan
# on most rows and any probe signal degrades toward noise instead of erroring.
# `src/indra_belief/model_client.py` declares max_top_logprobs: 1024 on the
# assumption this patch is in place. Re-apply it, or drop that back to 11 and
# accept that the probe path cannot run over HTTP.
#
#   S=$MLX_VENV/lib/python3.12/site-packages/mlx_lm/server.py
#   cp "$S" "$S.orig-cap11"
#   sed -i "" 's/max_val=11, whitelist=\[-1\]/max_val=262144, whitelist=[-1]/' "$S"
# ---------------------------------------------------------------------------

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
