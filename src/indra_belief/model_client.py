"""Unified model-transport client.

Supports:
- OpenAI-compatible APIs (LiteLLM → Ollama serving Gemma, Qwen, etc.)
- Anthropic API (Claude)
- Local models that emit reasoning in content (Qwen CRACK variants)
- Local models with separate reasoning_content (Gemma-4, Qwen3-thinking)

Design principles:
1. Single ModelClient interface; backend detail hidden.
2. Plain chat only. Tool-use is implemented by pre-computing the tool
   result and injecting it into the prompt (see
   `scorer._format_entity_lookups`), not by native tool-calling — the
   model ignored tool results after committing a verdict in pass one.
3. This module is pure transport — verdict parsing and score mapping
   live in `scorers/_prompts.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Formal Bedrock raw-HTTP adapters load exactly this PEM bundle.  They never
# consult SSL_CERT_FILE/SSL_CERT_DIR or urllib's platform-default trust store.
# The publication runtime binds these bytes separately from this source file.
_FIXED_BEDROCK_TLS_CA_BUNDLE = "/private/etc/ssl/cert.pem"
_FIXED_BEDROCK_TLS_CA_BUNDLE_SHA256 = (
    "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994"
)
_FIXED_BEDROCK_RESPONSES_ENDPOINT = (
    "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
)


# Model registry — name → (base_url, model_id, notes)
LOCAL_MODELS: dict[str, dict] = {
    "vllm-gemma-4-26b": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model_id": "google/gemma-4-26B-A4B-it",
        "reasoning_in_content": False,
        # 1000/120 raised to 8192/900 on 2026-08-12. MEASURED on 60 monolithic
        # calls with the production reasoning-first prompt: output tokens p50
        # 574, p90 1507, max 4353 — so a 1000 cap truncates 16.7% of calls
        # (Wilson [0.093, 0.280]) while 8192 truncates 0/60 ([0.000, 0.060]).
        # A truncated read costs the full wall clock and yields no verdict, and
        # on this path it is NOT withheld, so it can contribute a mid-thought
        # verdict. The served id must match `--served-model-name` byte-for-byte
        # or reader_configuration_for_run nulls the prompt and calibration can
        # never resolve.
        "max_tokens": 8192,
        "timeout": 900,
        # A CLAIM ABOUT HOW THE SERVER IS LAUNCHED, not a property of vLLM.
        # vLLM's `--max-logprobs` defaults to 20 and the direct probe's losing
        # label was measured at rank 42/83/168, so the server must be started
        # with `--max-logprobs 1024` for the probe to read. If it is not, vLLM
        # rejects the oversized `top_logprobs` and the failure surfaces per row
        # in `score_error` — loudly, rather than as quietly wrong numbers.
        # Declaring it makes this client probe-READABLE; it does not make it
        # calibrated. That needs a fitted artifact registered in
        # indra_belief.probes.calibration._SENTENCE_CALIBRATIONS.
        "max_top_logprobs": 1024,
    },
    "ollama-gemma-3-27b": {
        "base_url": "http://localhost:11434/v1",
        "model_id": "gemma3:27b",
        "reasoning_in_content": False,
        # 1000/120 raised to 8192/900 on 2026-08-12, same measurement as
        # vllm-local. gemma3:27b is not a reasoning model, so its own output is
        # shorter — but the cap is a CEILING, not a reservation, and the old
        # value silently truncated any reasoning-first prompt sent here.
        "max_tokens": 8192,
        "timeout": 900,
    },
    "local-qwen3.5-vl-122b-a10b": {
        "base_url": "http://localhost:8082/v1",
        "model_id": "dealignai/Qwen3.5-VL-122B-A10B-4bit-MLX-CRACK",
        "reasoning_in_content": True,  # CoT is emitted in content
        "typical_tokens": 2500,
        # 8000 -> 8192 on 2026-08-12. 8000 sat just under the measured floor —
        # the near-miss a round number invites. This reader emits its CoT in
        # `content` and has the largest `typical_tokens` in the registry, so it
        # is the entry least able to afford a tight ceiling.
        "max_tokens": 8192,
        "timeout": 180,
    },
    # Minimax-M2.7 (JANGTQ-CRACK quant) served via vmlx-engine on a local
    # MLX backend. Same response shape as gemma — separate reasoning_content
    # field, JSON content. Faster than gemma-remote (~43 tok/sec vs 22).
    "local-minimax-m2.7": {
        "base_url": "http://localhost:8086/v1",
        "model_id": "minimax-m2.7-jangtq-crack",
        "reasoning_in_content": False,
        "typical_tokens": 2500,
        # vmlx server is started with --max-tokens 32768 — match the cap.
        # Reasoning is always-on (--default-enable-thinking true) so 32K
        # gives the model plenty of room for CoT + structured output.
        "max_tokens": 32000,
        "timeout": 600,
    },
    # Served by mlx_lm.server on Apple Silicon; see scripts/serve_mlx.sh.
    # This is the ONLY reader we can currently obtain token logprobs from:
    # Bedrock's responses route accepts `top_logprobs` for gemma and returns an
    # empty array, and the noot-1 gateway is down. See `supports_logprobs`.
    "local-gemma-4-26b": {
        "base_url": "http://localhost:8085/v1",
        "model_id": "mlx-community/gemma-4-26b-a4b-it-8bit",
        "reasoning_in_content": False,  # separate reasoning field (mlx spelling)
        "typical_tokens": 400,
        # Measured on an M5 Max at 20.9-29.3 tok/s, mean 25.2, over the 16
        # --workers 1 monolithic calls for the 8-bit MoE: the reasoning-
        # first prompt spends 700-1500 tokens deliberating before the JSON, so
        # the old 1000/60s pair truncated mid-thought and then timed out. A
        # truncated reply is not a cheap failure here — it costs the full wall
        # clock and yields no verdict.
        #
        # Raised 4096 -> 8192 on 2026-08-12 by operator decision, to cut the
        # measured cap-hit rate before the calibration fit run. At 4096 the rate
        # was 2/88 sampled rows (pooled, Wilson [0.0063, 0.0791]) while the
        # largest UNTRUNCATED call observed was 3695 tokens — 90.2% of that cap,
        # i.e. the tail was pressed right against it. 8192 leaves 2.2x headroom
        # over the observed maximum. The rate at this cap is re-measured rather
        # than assumed: a cap hit still costs full wall clock, so the surcharge
        # per hit roughly doubles even as hits become rarer.
        "max_tokens": 8192,
        "timeout": 900,
        "supports_logprobs": True,
        # RAISED 11 -> 1024 on 2026-08-13, and this DEPENDS ON A LOCAL PATCH to
        # the serving venv. Stock mlx_lm.server hard-codes
        # `_validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])`
        # at server.py:1245, and 11 is nowhere near enough at a forced verdict
        # position: measured, the LOSING label sits at rank 42 / 83 / 168 across
        # four cases, because JSON formatting tokens ({", ", ```) crowd it out.
        # At k=11 three of four cases yielded no usable p_raw; at k=1024, 0 of
        # 1,075 records dropped. `scripts/serve_mlx.sh` documents the patch.
        # If the venv is rebuilt or mlx-lm upgraded, the cap silently returns to
        # 11 and p_raw degrades to nan on most rows rather than erroring.
        "max_top_logprobs": 1024,
    },
    "local-gemma-4-31b": {
        "base_url": "http://localhost:8084/v1",
        "model_id": "mlx-community/gemma-4-31b-it-8bit",
        "reasoning_in_content": False,
        "typical_tokens": 400,
        # This entry carried the literal "1000/60s pair" that the 26b comment
        # above records as catastrophic — truncating mid-thought and then timing
        # out. Raised to 8192/900 on 2026-08-12 to match its 26b sibling.
        "max_tokens": 8192,
        "timeout": 900,
    },
    "remote-gemma-4-26b": {
        "base_url": "http://100.97.101.59:11434/v1",
        # Gateway serves gemma via ollama under this exact id; other id
        # spellings return 400 ("Invalid model name"). Match it exactly.
        "model_id": "gemma-4-26b-ollama",
        "reasoning_in_content": False,
        "reasoning_effort": "medium",
        "typical_tokens": 400,
        # Match the remote server's generation ceiling. Long monolithic
        # reasoning can exceed 2500/12000 tokens before emitting verdict JSON;
        # lower caller-side caps create artificial verdict=None rows.
        "max_tokens": 32000,
        "num_ctx": 32768,
        # Wall-clock guard for monolithic runs at the backend's 32k
        # generation ceiling: keep it high enough that long-but-valid
        # generations are not converted into artificial row errors.
        # Short-form sub-calls (parse_evidence / grounding) finish well
        # under this; the guard exists to fail fast only under genuine
        # endpoint degradation, with retries disabled so a slow record
        # cannot multiply into a runaway per-record cost.
        "timeout": 600,
    },
    # Google AI Studio (Gemma 4) — hosted Gemma via the Gemini API's
    # OpenAI-compatibility endpoint. Same weights as the local gemma-moe /
    # gemma-31b but routed through Google's infrastructure: no tailscale
    # latency, no LiteLLM proxy in the path (eliminates the channel-token
    # 500 class of failures), and significantly higher per-request
    # throughput. Auth: GEMINI_API_KEY env var.
    "google-gemma-4-26b": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_id": "gemma-4-26b-a4b-it",
        "api_key_env": "GEMINI_API_KEY",
        "reasoning_in_content": False,
        "typical_tokens": 400,
        "max_tokens": 8192,
        "timeout": 120,
        # Google's OpenAI-compat endpoint strictly rejects unknown
        # extra_body keys (400 INVALID_ARGUMENT on chat_template_kwargs,
        # format, num_ctx — see model_client.call() for the field list).
        # Without this flag every LLM call 400-fails silently: the
        # deterministic substrate-only fallback masks it from the progress
        # stream, so a whole run can complete substrate-only and look healthy.
        "strict_openai_compat": True,
        # PaidTier3 quota: 16k input tokens/min for Gemma. With ~3k input
        # tokens per parse_evidence call, true sustainable throughput is
        # ~5 req/min — so concurrency above 2 just burns the budget on
        # bursts then waits 15s for the OpenAI client retry. Keep this
        # conservative; --workers can override per-run if quota changes.
        "concurrency_hint": 2,
    },
    "google-gemma-4-31b": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_id": "gemma-4-31b-it",
        "api_key_env": "GEMINI_API_KEY",
        "reasoning_in_content": False,
        "typical_tokens": 400,
        "max_tokens": 8192,
        "timeout": 120,
        "strict_openai_compat": True,
        "concurrency_hint": 2,
    },
    # MedPsy-4B (Tether/QVAC) — Qwen3-based 4B fine-tuned for medical Q&A.
    # In-process transformers + bf16, no HTTP server. The base chat template
    # hard-codes a "You are MedPsy, a medical and healthcare AI assistant"
    # persona; _setup_transformers_client swaps it for a task-specific
    # adjudicator persona at load time so the scorer's SYSTEM_PROMPT carries
    # the actual instructions instead of fighting the persona.
    "local-medpsy-4b": {
        "backend": "transformers_local",
        "model_id": "qvac/MedPsy-4B",
        "reasoning_in_content": False,
        "enable_thinking": True,
        "torch_dtype": "bfloat16",
        "device": "mps",
        "typical_tokens": 800,
        # 4096 -> 8192 on 2026-08-12. `enable_thinking` is True here, so this
        # reader deliberates before answering and its output distribution is
        # unmeasured; a ceiling costs nothing when it is not reached, while a
        # truncated read costs the full wall clock and yields no verdict.
        "max_tokens": 8192,
        "timeout": 300,
        "persona": (
            "You are a biomedical text-mining adjudicator. You judge whether "
            "structured (subject)-(relation)-(object) statements extracted by an "
            "NLP reader are supported by the source evidence sentence."
        ),
    },
    # MedPsy-4B (Tether/QVAC) served on the ROCm box (noot-1) via the llm-gateway
    # stack: LiteLLM gateway (:11434) -> llama.cpp HIP container (:8081) running
    # medpsy-4b-q8_0.gguf on a Radeon RX 7900 XTX (gfx1100). Remote counterpart to
    # medpsy-4b-local; replaces gemma-remote as the active remote model (the
    # gateway serves one model at a time). llama-server runs with
    # --reasoning-format deepseek, so Qwen3-thinking <think> traces arrive as a
    # separate reasoning_content field (reasoning_in_content=False) — same response
    # shape as gemma-remote. Thinking mode emits long CoT, so keep the generation
    # ceiling high.
    "remote-medpsy-4b": {
        "base_url": "http://100.97.101.59:11434/v1",
        "model_id": "medpsy-4b",
        "reasoning_in_content": False,
        "max_tokens": 32000,
        "num_ctx": 65536,
        "timeout": 600,
    },
    # ── AWS Bedrock (OpenAI-compatible "mantle" endpoint) ───────────────
    # Bedrock exposes an OpenAI-compatible Chat Completions API on the
    # bedrock-mantle host. Auth is the Bedrock API key (a bearer token)
    # passed as the OpenAI `api_key` — no SigV4, no boto3. We reuse the
    # openai_compat backend verbatim: the response shape is standard OpenAI
    # (choices[0].message.content + usage). strict_openai_compat=True
    # suppresses the Ollama-only extra_body keys (num_ctx / native `format`
    # / chat_template_kwargs) this endpoint would 400 on — exactly like the
    # Google AI Studio entries above.
    #
    # base_url is the mantle `/openai/v1` root FOR GEMMA 4 (the OpenAI SDK
    # appends /chat/completions, /responses, /models). Mantle routes are
    # PER-MODEL and the wrong-route error is MISLEADING — verified 2026-06-18
    # with this exact token:
    #   gemma-4-26b-a4b : 200 on /openai/v1 ; on /v1 → 401 {"code":"access_denied",
    #                     "message":"Berm is not enabled for this account"}. That
    #                     401 is NOT an account/IAM/provisioning problem — it's
    #                     just the wrong route (the earlier "Gemma 4 not
    #                     provisioned, needs AWS" read was this confound).
    #   gemma-3-27b-it / gpt-oss : 200 on /v1 ; on /openai/v1 → 400 "isn't
    #                     supported on this route".
    # ⇒ a Gemma-3 / gpt-oss entry must use base_url `.../api.aws/v1`, NOT
    # /openai/v1. Listing models at /v1/models works regardless (which is why the
    # token looked validated while gemma-4 inference 401'd). Model IDs are the
    # BARE mantle ids from `GET {base_url}/models`, NOT the control-plane
    # "...-v1:0" forms. Region us-east-1 (also us-east-2 / us-west-2 / eu-central-1).
    # Token env var: AWS_BEARER_TOKEN_BEDROCK (in .env).
    #
    # COST: every chat.completions call here is billed per token by AWS
    # (listing models is free; inference is not). The API key is tagged
    # name="indra-belief-model" for cost allocation.
    #
    # CAVEAT (unverified — needs a paid smoke test): optional OpenAI extras
    # the sub-call scorers pass (response_format=json_object on probes /
    # grounding, reasoning_effort="none") may not be honored identically by
    # every Bedrock-served model. The DEFAULT monolithic path
    # (MONO_VARIANT=disconfirm_relnature, no response_format, no
    # reasoning_effort) sends a minimal request and is the safe first run.
    "bedrock-gemma-4-26b": {
        # Same weights as gemma-remote / gemma-google (gemma-4-26b-a4b), served
        # by Bedrock with no local GPU / tailscale hop. Gemma DOES reason on
        # Bedrock — but the CHAT COMPLETIONS API DROPS the CoT (probed
        # 2026-06-19: zero reasoning at every reasoning_effort on
        # /chat/completions). Only the RESPONSES API surfaces it, so this entry
        # uses backend="bedrock_responses_raw" (POST /openai/v1/responses) with
        # reasoning_effort="high" (only "high" engages a reasoning item;
        # medium/none → none, verified). reasoning_in_content=False ⇒ raw_text =
        # reasoning + answer, so the verdict parse sees both. Gemma is
        # mantle-only (no Converse route) and uses the /openai/v1 path.
        "backend": "bedrock_responses_raw",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "expected_responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "model_id": "google.gemma-4-26b-a4b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "tls_ca_bundle": _FIXED_BEDROCK_TLS_CA_BUNDLE,
        "tls_ca_bundle_sha256": _FIXED_BEDROCK_TLS_CA_BUNDLE_SHA256,
        "max_request_bytes": 16 * 1024 * 1024,
        "max_response_bytes": 16 * 1024 * 1024,
        "reasoning_in_content": False,
        "reasoning_effort": "high",  # only "high" surfaces a reasoning item
        "typical_tokens": 400,
        "max_tokens": 32000,         # CoT + answer share the budget
        "timeout": 600,
    },
    # ── Reasoning-isolation twins ────────────────────────────────────────────
    # One per paid comparison arm, differing from their thinking sibling in
    # exactly one field: reasoning_effort "high" -> "none". Everything else
    # (backend, endpoint pins, TLS bundle, byte bounds, model_id, max_tokens)
    # is copied verbatim, so a paired run isolates reasoning mode and nothing
    # else. The three GEMMA twins keep the surviving `bedrock_responses_raw`
    # paid lane on purpose: its recorded canonical wire body is the ONLY artifact
    # that can prove reasoning was actually off — the
    # provider's own token accounting cannot (gemma reports
    # reasoning_tokens=0 while returning real CoT; glm-5 omits the field).
    #
    # What "none" does on each wire, verified in bedrock_*_transport.py:
    #   Responses lane (gemma): `if reasoning_effort and != "none"` — the
    #     `reasoning` key is OMITTED entirely (bedrock_responses_transport.py
    #     :625-627), and a literal reasoning.effort=="none" is rejected by the
    #     request validator (:658).
    #   Chat lane (glm-5): the twin now rides `openai_compat`; because it declares
    #     `strict_openai_compat: True`, `reasoning_wire_keys(effort, strict=True)`
    #     returns `{"reasoning_effort": effort}` alone, so `"reasoning_effort":
    #     "none"` still goes on the wire — the same wire intent as before.
    "bedrock-gemma-4-26b-noreason": {
        "backend": "bedrock_responses_raw",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "expected_responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "model_id": "google.gemma-4-26b-a4b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "tls_ca_bundle": _FIXED_BEDROCK_TLS_CA_BUNDLE,
        "tls_ca_bundle_sha256": _FIXED_BEDROCK_TLS_CA_BUNDLE_SHA256,
        "max_request_bytes": 16 * 1024 * 1024,
        "max_response_bytes": 16 * 1024 * 1024,
        "reasoning_in_content": False,
        "reasoning_effort": "none",
        "typical_tokens": 400,
        "max_tokens": 32000,
        "timeout": 600,
    },
    "bedrock-gemma-4-31b-noreason": {
        "backend": "bedrock_responses_raw",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "expected_responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "model_id": "google.gemma-4-31b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "tls_ca_bundle": _FIXED_BEDROCK_TLS_CA_BUNDLE,
        "tls_ca_bundle_sha256": _FIXED_BEDROCK_TLS_CA_BUNDLE_SHA256,
        "max_request_bytes": 16 * 1024 * 1024,
        "max_response_bytes": 16 * 1024 * 1024,
        "reasoning_in_content": False,
        "reasoning_effort": "none",
        "typical_tokens": 500,
        "max_tokens": 32000,
        "timeout": 600,
    },
    "bedrock-gemma-4-e2b-noreason": {
        "backend": "bedrock_responses_raw",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "expected_responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "model_id": "google.gemma-4-e2b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "tls_ca_bundle": _FIXED_BEDROCK_TLS_CA_BUNDLE,
        "tls_ca_bundle_sha256": _FIXED_BEDROCK_TLS_CA_BUNDLE_SHA256,
        "max_request_bytes": 16 * 1024 * 1024,
        "max_response_bytes": 16 * 1024 * 1024,
        "reasoning_in_content": False,
        "reasoning_effort": "none",
        "typical_tokens": 400,
        "max_tokens": 16000,
        "timeout": 300,
    },
    "bedrock-glm-5-noreason": {
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "zai.glm-5",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "reasoning_effort": "none",
        "typical_tokens": 800,
        "max_tokens": 32000,
        "timeout": 600,
    },
    # ── AWS Bedrock mantle — additional open-weight models on the /v1 route ──
    # Verified 2026-06-19 with AWS_BEARER_TOKEN_BEDROCK: each returns 200 on
    # `.../api.aws/v1/chat/completions` and 400 "isn't supported on this route"
    # on /openai/v1 (the INVERSE of gemma-4, which is /openai/v1-only). Bare
    # mantle ids from `GET .../v1/models`. openai_compat backend, strict (mantle
    # 400s on the Ollama-only extras). reasoning_in_content=False ⇒ raw_text =
    # reasoning + content: deepseek-v3.2 / kimi-k2.5 are thinking models that may
    # emit a separate reasoning_content (or inline <think>); the scorer's
    # tolerant verdict parse reads both. Per-token billed by AWS.
    #
    # REASONING (probed 2026-06-19, to approximate gemma-remote's medium CoT):
    # mantle's reasoning_effort scale is COARSE — thinking is OFF at unset / none
    # / medium and only engages at "high" (the param is ACCEPTED, no 400, at
    # every level). And only deepseek-v3.2 + kimi-k2.5 deliberate at all: at
    # "high" they spend real reasoning (probe ~700-850 chars CoT; scorer latency
    # ~4-5x, 3s→12-14s ≈ gemma-remote). The CoT lands in reasoning_content OR
    # inline in content depending on the prompt; reasoning_in_content=False
    # captures both into raw_text so the verdict still parses. The other three
    # emit NO extra CoT at any effort — qwen3-235b-a22b-2507 is the non-thinking
    # instruct
    # variant (high TIMED OUT >120s, pathological), qwen3-coder-480b is
    # non-thinking, and bedrock-gemma's Bedrock serving exposes no thinking. So
    # the two thinking models carry reasoning_effort="high" + max_tokens 32000
    # (CoT headroom, like gemma-remote's 32000); the other three cannot match
    # gemma-remote's thinking and are left non-thinking (a literal "medium" would
    # be a silent no-op on mantle).
    "bedrock-deepseek-v3.2": {
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "deepseek.v3.2",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "reasoning_effort": "high",  # only "high" engages CoT on mantle (see above)
        "typical_tokens": 600,
        "max_tokens": 32000,         # CoT + verdict room, like gemma-remote
        "timeout": 600,
    },
    "bedrock-kimi-k2.5": {
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "moonshotai.kimi-k2.5",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "reasoning_effort": "high",  # only "high" engages CoT on mantle (see above)
        "typical_tokens": 600,
        "max_tokens": 32000,         # CoT + verdict room, like gemma-remote
        "timeout": 600,
    },
    "bedrock-qwen3-235b-a22b": {
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "qwen.qwen3-235b-a22b-2507",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "typical_tokens": 500,
        "max_tokens": 8192,
        "timeout": 600,
    },
    "bedrock-qwen3-coder-480b-a35b": {
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "qwen.qwen3-coder-480b-a35b-instruct",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "typical_tokens": 500,
        "max_tokens": 8192,
        "timeout": 600,
    },
    # Z.ai GLM-5 — frontier reasoning model, the largest reasoner reachable on
    # the mantle endpoint. Probed 2026-06-19: surfaces CoT via Chat Completions
    # reasoning_effort="high" -> reasoning_content (rc=1098; medium/none = none),
    # the SAME plain path as deepseek/kimi — no Responses API or Converse needed.
    # max_tokens is high on purpose: at "high" glm-5 spends the budget on
    # reasoning_content and can emit the verdict JSON late (probe showed empty
    # content at max_tokens=3000), so 32000 keeps room for CoT + verdict and
    # avoids truncation to verdict=None. reasoning_in_content=False ⇒ raw_text =
    # reasoning + answer for the verdict parse.
    "bedrock-glm-5": {
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "zai.glm-5",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 800,
        "max_tokens": 32000,
        "timeout": 600,
    },
    # ── Small/mid open-weight reasoners (mantle /v1 chat, reasoning_content) ──
    # Probed 2026-06-19: all surface CoT via chat-completions reasoning_effort="high"
    # (same path as deepseek/kimi/glm), so plain openai_compat. Cheaper/faster
    # alternatives to the big reasoners; favor-latest picks across two size bands.
    # gpt-oss reasons HERE (chat), not via Responses — verified, contra some docs.
    "bedrock-nemotron-nano-3-30b": {  # 31.6B/3.2B active MoE, Dec 2025 — best small cap/cost
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "nvidia.nemotron-nano-3-30b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 500,
        "max_tokens": 32000,
        "timeout": 600,
    },
    "bedrock-gpt-oss-20b": {  # 20.9B/3.6B active MoE, Aug 2025
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "openai.gpt-oss-20b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 500,
        "max_tokens": 32000,
        "timeout": 600,
    },
    "bedrock-minimax-m2.5": {  # 230B/10B active MoE, Feb 2026 — newest frontier-tier mid
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "minimax.minimax-m2.5",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 600,
        "max_tokens": 32000,
        "timeout": 600,
    },
    "bedrock-nemotron-super-3-120b": {  # 120B/12B active MoE, Mar 2026, 1M ctx
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "nvidia.nemotron-super-3-120b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 600,
        "max_tokens": 32000,
        "timeout": 600,
    },
    "bedrock-gpt-oss-120b": {  # 117B/5.1B active MoE, Aug 2025 — cheapest-with-reasoning
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "model_id": "openai.gpt-oss-120b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "strict_openai_compat": True,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 600,
        "max_tokens": 32000,
        "timeout": 600,
    },
    # OpenAI GPT-5.5 — Responses-API-ONLY on Bedrock (probed 2026-06-19: Chat
    # Completions AND /v1/responses both 400; only /openai/v1/responses works).
    # Reuses backend="bedrock_responses". reasoning_effort="high" engages thinking
    # (usage.output_tokens_details.reasoning_tokens > 0), BUT GPT reasoning is
    # ENCRYPTED — the reasoning output item carries only an (empty) summary, no
    # readable CoT, so the `reasoning` telemetry field stays empty (thinking is
    # real + server-side; the verdict comes back in the message item, which the
    # parser extracts). max_tokens high so reasoning_tokens + verdict JSON both fit.
    "bedrock-gpt-5.5": {
        "backend": "bedrock_responses",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "model_id": "openai.gpt-5.5",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 800,
        "max_tokens": 32000,
        "timeout": 600,
    },
    # Gemma 4 31B (dense, newest Gemma) — like bedrock-gemma it reasons ONLY via
    # the Responses API (chat-completions drops the CoT, probed 2026-06-19), so
    # backend=bedrock_responses_raw on the /openai/v1 route. Heavier/slower dense
    # counterpart to the gemma-4-26b-a4b MoE already wired as bedrock-gemma.
    "bedrock-gemma-4-31b": {
        "backend": "bedrock_responses_raw",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "expected_responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "model_id": "google.gemma-4-31b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "tls_ca_bundle": _FIXED_BEDROCK_TLS_CA_BUNDLE,
        "tls_ca_bundle_sha256": _FIXED_BEDROCK_TLS_CA_BUNDLE_SHA256,
        "max_request_bytes": 16 * 1024 * 1024,
        "max_response_bytes": 16 * 1024 * 1024,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 500,
        "max_tokens": 32000,
        "timeout": 600,
    },
    # Gemma 4 E2B (~2.3B effective, tiny on-device tier) — the ONLY clean sub-10B
    # reasoner on Bedrock: reasons only via the Responses API (probed rc=411,
    # chat drops it), so backend=bedrock_responses_raw. Extreme-cheap floor;
    # capacity-gated for subtle relation logic — validate (n=1606) before trusting.
    # (Other sub-10B mantle models — ministral-3-3b, gemma-3-4b — are non-thinking;
    # our MedPsy-4B reasoner is local-only on noot-1, not on AWS.)
    "bedrock-gemma-4-e2b": {
        "backend": "bedrock_responses_raw",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "expected_responses_endpoint": _FIXED_BEDROCK_RESPONSES_ENDPOINT,
        "model_id": "google.gemma-4-e2b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "tls_ca_bundle": _FIXED_BEDROCK_TLS_CA_BUNDLE,
        "tls_ca_bundle_sha256": _FIXED_BEDROCK_TLS_CA_BUNDLE_SHA256,
        "max_request_bytes": 16 * 1024 * 1024,
        "max_response_bytes": 16 * 1024 * 1024,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 400,
        "max_tokens": 16000,
        "timeout": 300,
    },
    # AWS Bedrock CLAUDE (Anthropic) — native Converse API, NOT mantle/OpenAI.
    # Verified 2026-06-18 (this same bearer token): Claude models reject BOTH
    # mantle OpenAI routes (/chat/completions AND /responses → HTTP 400 "does not
    # support the '...' API"). They are served by the native Bedrock-runtime
    # Converse API, which the bearer token authenticates directly (no SigV4, no
    # boto3). Two more gotchas, both verified:
    #   • model_id must be an INFERENCE-PROFILE id (us.* / global.*), NOT the bare
    #     catalog id — the models are inferenceTypesSupported=['INFERENCE_PROFILE'],
    #     so bare ids 400 with "the provided model identifier is invalid".
    #   • there is NO claude-sonnet in this account's catalog — only
    #     haiku-4-5, opus-4-7, opus-4-8 (the old bedrock-claude-sonnet 404'd).
    # backend="bedrock_converse" (see _call_bedrock_converse) maps
    # (system, messages) → Converse and parses output.message.content[].text +
    # usage.{input,output}Tokens. The DEFAULT monolithic path (no response_format,
    # no reasoning_effort) works; OpenAI response_format=json_object used by the
    # decomposed sub-calls is NOT honored on Converse — keep Claude on the
    # monolithic path until a Converse JSON/tool mechanism is wired.
    "bedrock-claude-haiku-4-5": {
        "backend": "bedrock_converse",
        "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "reasoning_in_content": False,
        "typical_tokens": 500,
        "max_tokens": 8192,
        "timeout": 300,
    },
    "bedrock-claude-opus-4-8": {
        "backend": "bedrock_converse",
        "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "model_id": "us.anthropic.claude-opus-4-8",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "reasoning_in_content": False,
        "typical_tokens": 600,
        "max_tokens": 8192,
        "timeout": 300,
    },
}


# Historical / abbreviated registry name -> canonical (host-prefix + full tag).
# Lets old `--model` invocations and already-recorded run `model` fields keep
# working. NOTE: cost + call-log key on each entry's `model_id`, NOT this name,
# so aliasing never affects pricing.
_MODEL_ALIASES: dict[str, str] = {
    # WHY THESE TWO WERE RENAMED. A registry key names the SERVING ARCHITECTURE
    # AND THE MODEL -- bedrock-gemma-4-26b, remote-medpsy-4b, local-gemma-4-31b
    # -- for 29 of 31 entries. `vllm-local` and `ollama-local` named only the
    # server, and that is not cosmetic: the belief profile registry
    # (`calibration_constants._FITTED_CONFIGS`) is keyed on
    # (registry name, prompt sha) with NO served-model id, so the NAME is the
    # only thing tying a fitted profile to the weights it was fitted on. Serve a
    # different model on the same vLLM and a profile registered under a
    # server-shaped name follows it silently.
    #
    # The isotonic registry does carry the served id, so it is guarded; the
    # profile registry is not, which is exactly where a weights-agnostic name
    # must not appear.
    "vllm-local": "vllm-gemma-4-26b",
    "ollama-local": "ollama-gemma-3-27b",
    "qwen-thinker": "local-qwen3.5-vl-122b-a10b",
    "minimax-local": "local-minimax-m2.7",
    "gemma-moe": "local-gemma-4-26b",
    "gemma-31b": "local-gemma-4-31b",
    "gemma-remote": "remote-gemma-4-26b",
    "gemma-google-moe": "google-gemma-4-26b",
    "gemma-google-31b": "google-gemma-4-31b",
    "medpsy-4b-local": "local-medpsy-4b",
    "medpsy-remote": "remote-medpsy-4b",
    "bedrock-gemma": "bedrock-gemma-4-26b",
    "bedrock-qwen3-235b": "bedrock-qwen3-235b-a22b",
    "bedrock-qwen3-coder-480b": "bedrock-qwen3-coder-480b-a35b",
    "bedrock-nemotron-nano-30b": "bedrock-nemotron-nano-3-30b",
    "bedrock-nemotron-super-120b": "bedrock-nemotron-super-3-120b",
    "bedrock-claude-haiku": "bedrock-claude-haiku-4-5",
    "bedrock-claude-opus": "bedrock-claude-opus-4-8",
    # recorded-only legacy literal (old export/viewer `model` field)
    "gemma-4-26b (remote)": "remote-gemma-4-26b",
}


def canonical_model_name(model: str) -> str:
    """Map any historical/abbreviated model name to its canonical host-prefixed,
    full-tag form. Idempotent: canonical names pass through unchanged. Applied at
    run-export time so recorded `model` fields and the viewer read consistently."""
    if not model:
        return model
    return _MODEL_ALIASES.get(model, model)


_RETRY_DELAY_RE = __import__("re").compile(
    r"retry in (\d+(?:\.\d+)?)s|retryDelay['\"]:\s*['\"](\d+)s",
    flags=__import__("re").IGNORECASE,
)


def _parse_retry_delay(error_text: str, default: float = 30.0) -> float:
    """Pull the API's requested retry delay (seconds) out of a 429 error
    payload. Google formats it both as 'Please retry in 50.05s.' and as a
    structured RetryInfo block with retryDelay='50s'. Falls back to a
    reasonable default if neither is present."""
    m = _RETRY_DELAY_RE.search(error_text)
    if m:
        for g in m.groups():
            if g:
                try:
                    # Clamp to [0, 3600s] so a hostile/garbled 429 payload
                    # ("retry in 999999s") can't block a scoring worker for days;
                    # mirrors bedrock_transport_base._retry_after_seconds' cap.
                    return min(max(0.0, float(g)), 3600.0)
                except ValueError:
                    pass
    return default


# ── Rate-limit classification ─────────────────────────────────────────────
# A retry must fire on a genuine 429 and on nothing else. An unrelated failure
# whose body merely CONTAINS the digits "429" (a request id, a token count, a
# byte offset) is not a rate limit, and retrying it just spends the same budget
# again on the same failure. Two precise patterns replace a bare substring test.
#
# The phrase set is EXACTLY the two literals the previous substring test used
# ("rate limit", "resource_exhausted"), case-insensitively and nothing more.
# Widening it is a live hazard, not a hypothetical one: an earlier revision added
# "too many requests" / "quota exceeded" / a `[ _-]?` separator class, and that
# turned a permanent HTTP 400 "Quota exceeded for this account" rejection into
# five pointless retries of the same failure (155s of sleeps, five extra billed
# POSTs). Those tokens are a trustworthy rate-limit signal only when the provider
# ALSO reports 429 — via a structured status (rule 1) or in status position in the
# text (rule 3) — both of which are already covered. As free body text they are
# not evidence of throttling.
_RATE_LIMIT_PHRASE_RE = __import__("re").compile(
    r"rate limit|resource_exhausted",
    flags=__import__("re").IGNORECASE,
)
# 429 only where a provider puts a STATUS, never loose inside a payload. Covers
# "Bedrock Responses HTTP 429; retry in 30s", "Bedrock Converse HTTP 429: ...",
# the OpenAI SDK's "Error code: 429 - {...}", urllib's "HTTP Error 429: Too Many
# Requests", "Throttled (429)", "Server returned 429", and the gRPC/Google
# leading-code form "429 Resource has been exhausted (e.g. check quota)." — a
# bare 429 at the very START of the message IS the status. (No re.MULTILINE, so
# `^` means start-of-string, not start-of-line: a payload that happens to wrap
# onto a line beginning with 429 does not qualify.)
#
# This pattern may GROW toward the old substring test (every string it adds
# already contains "429", so the old test was already True for it) but never
# past it. The phrase set above may not grow past the old test at all.
_RATE_LIMIT_STATUS_RE = __import__("re").compile(
    r"(?:http(?:\s+error)?\s+|error code:\s*|status(?:[ _]code)?[\s:=]+"
    r"|code[\s:=]+|returned\s+)429\b"
    r"|\(429\)"
    r"|^\s*429\b",
    flags=__import__("re").IGNORECASE,
)


def _provider_http_status(error: BaseException) -> int | None:
    """Structured HTTP status carried by a provider exception, or None.

    This mirrored the retired spend guard's provider-exception status parser,
    which read the identical candidate set under the same 100..599 bound. The
    duplication was DELIBERATE: importing the append-only ledger module here
    would have dragged it onto this client's error path, which the transport
    layer must stay free of. If this provider surface grows a new status field,
    update this list.
    """
    candidates = [
        # Stamped by this module's legacy Bedrock adapters; the name sits
        # deliberately outside the spend-guard candidate set (see there).
        getattr(error, "_bedrock_http_status", None),
        getattr(error, "status_code", None),
        getattr(error, "status", None),
    ]
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        candidates += [response.get("status_code"), response.get("status")]
    elif response is not None:
        candidates += [
            getattr(response, "status_code", None),
            getattr(response, "status", None),
        ]
    trace = getattr(error, "transport_trace", None)
    if isinstance(trace, dict):
        candidates.append(trace.get("response_http_status"))
    import urllib.error
    if isinstance(error, urllib.error.HTTPError):
        candidates.append(getattr(error, "code", None))
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _is_rate_limit_error(error: BaseException) -> bool:
    """True only for genuine provider rate limiting. ONE contract, status-first:

    1. an authoritative structured status of 429                  -> True
    2. any OTHER authoritative structured status                  -> False
       (the provider told us what this is; body text does not overrule it)
    3. no structured status at all, and the text carries either an explicit
       rate-limit PHRASE or a 429 in STATUS POSITION                -> True

    Relation to the predicate this replaced (`msg = str(e).lower()`;
    `"429" in msg or "rate limit" in msg or "resource_exhausted" in msg`):
    the new rule is a SUBSET of it for every error our transports raise —
    nothing the old test refused to retry is now retried. Two deliberate
    NARROWINGS, both pinned by the differential battery in
    tests/test_model_client_paid_lane_hardening.py:

      (a) an authoritative NON-429 status is never a rate limit regardless of
          body text — a hard 400 "quota exceeded" rejection raises on the first
          occurrence instead of retrying the same permanent failure five times;
      (b) a statusless "429" that is NOT in a recognized status position (a
          request id "9f-429-77", a token count `{"input_tokens": 429}`) is not
          a rate limit.

    Rule 1 is the only text-independent rule. It cannot widen against the old
    predicate in practice: every error these lanes raise stamps the code into
    its own message (`Bedrock Converse HTTP 429: ...`), and provider SDK errors
    carrying `status_code` render it too (`Error code: 429 - {...}`), so a
    structured 429 always comes with a "429" the old substring test also saw.
    """
    text = str(error)
    status = _provider_http_status(error)
    if status == 429:
        return True
    # Text speaks only when the provider said nothing structured.
    return status is None and bool(
        _RATE_LIMIT_PHRASE_RE.search(text) or _RATE_LIMIT_STATUS_RE.search(text)
    )


# ── Legacy-lane absolute deadline ─────────────────────────────────────────
class _LaneDeadlineExpired(Exception):
    """Internal signal: a legacy Bedrock lane blew its absolute wall deadline.

    Private to this module and never escapes it: `_call_bedrock_converse` /
    `_call_bedrock_responses` catch it OUTSIDE their `with` block — so the
    response context manager has provably already closed the billed request —
    and convert it to the lane's existing TimeoutError message.
    """


def _response_socket(resp):
    """The live socket under a urllib response, or None once it is gone."""
    return getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)


def _read_body_within_deadline(
    resp, *, absolute_deadline: float, lane_label: str
) -> bytes:
    """Read a response body under a monotonic ABSOLUTE deadline.

    `urlopen(req, timeout=N)` bounds each individual socket operation, not total
    wall time: a provider that dribbles one byte every N/2 seconds keeps every
    operation inside the timeout while the call runs unboundedly long. This is
    the missing outer bound for the two legacy urllib lanes, which run
    synchronously in the scoring worker with no thread-pool wall cap.

    Local twin, deliberately NOT an import of
    `bedrock_responses_transport.RawBedrockResponsesTransport._read_response`,
    whose loop shape this mirrors (reach the socket via `resp.fp.raw._sock`,
    re-arm `settimeout(remaining)` before each 64KiB `read1`, re-check the
    deadline after each chunk). That method is welded to `_ResponseReadFailure`
    and the sealed transport-trace contract — partial-body capture, framing
    validation, replay evidence — and importing it would drag the trace
    machinery onto lanes that deliberately carry none of it.

    Fallback: when `resp.fp.raw._sock` resolves to None there is no socket to
    bound, so the body is read in a single `resp.read()` call. That covers every
    non-HTTP opener, including the in-repo fakes whose `read()` takes no size
    argument (tests/test_reasoning_trace.py, tests/test_bedrock_responses_transport.py).
    """
    import time as _time

    if _response_socket(resp) is None:
        return resp.read()
    read_one = getattr(resp, "read1", None) or resp.read
    chunks: list[bytes] = []
    while True:
        remaining = absolute_deadline - _time.monotonic()
        if remaining <= 0:
            raise _LaneDeadlineExpired(lane_label)
        # Re-resolved every pass, exactly as the transport does it: http.client
        # drops `resp.fp` — really closing the fd — as soon as Content-Length is
        # exhausted, so a socket reference hoisted out of this loop would be a
        # closed descriptor on the final (empty) read.
        sock = _response_socket(resp)
        if sock is not None:
            sock.settimeout(remaining)
        chunk = read_one(64 * 1024)
        if _time.monotonic() >= absolute_deadline:
            raise _LaneDeadlineExpired(lane_label)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def concurrency_hint(model_name: str) -> int:
    """Reasonable default max-concurrency for a model. 1 means serial.

    Local Ollama endpoints default to 1 (single-GPU serving); hosted
    backends like Google AI Studio can fan out. Callers may override."""
    cfg = LOCAL_MODELS.get(model_name)
    if cfg is None:
        return 1
    return int(cfg.get("concurrency_hint", 1))


# ── Reasoning-trace normalization ──────────────────────────────────────────
# A model-agnostic capture of chain-of-thought, so a downstream interface can
# present reasoning uniformly for ANY backend. The hard part is that CoT lives
# in different places (separate reasoning field / Responses reasoning item /
# inline <think> / encrypted / not returned at all), and "the model reasoned"
# (reasoning_tokens) is ORTHOGONAL to "we can read it" (text). We normalize both
# at the adapter boundary (the only place provider-specific knowledge lives) into
# one dict with an explicit status, so adding a model touches only its _call_*.
class ReasoningStatus:
    """Single source of truth for reasoning-trace status values (plain strings
    so the trace is trivially JSON-serializable in the call log)."""
    PLAINTEXT = "plaintext"          # readable CoT in a separate field
    INLINE = "inline"                # CoT was inside content, split out by adapter
    ENCRYPTED = "encrypted"          # reasoned (tokens>0) but no readable text (gpt-5.5)
    NOT_RETURNED = "not_returned"    # thinking requested + tokens>0 but text empty
    NONE = "none"                    # no reasoning emitted / not requested


def _reasoning_tokens(usage) -> int:
    """Pull the reasoning-token count from a usage object/dict, or -1 if the
    backend doesn't report it. The two field paths the providers use:
      OpenAI chat:       usage.completion_tokens_details.reasoning_tokens
      Bedrock Responses: usage["output_tokens_details"]["reasoning_tokens"]"""
    if usage is None:
        return -1
    if isinstance(usage, dict):  # bedrock_responses raw payload
        d = usage.get("output_tokens_details") or {}
        try:
            return int(d.get("reasoning_tokens", -1))
        except (TypeError, ValueError):
            return -1
    d = getattr(usage, "completion_tokens_details", None)  # openai SDK object
    if d is None:
        return -1
    try:
        return int(getattr(d, "reasoning_tokens", -1))
    except (TypeError, ValueError):
        return -1


def _classify_reasoning(reasoning: str, reasoning_tokens: int, *, inline: bool) -> str:
    """Status from (text, token-count): present text → plaintext|inline;
    no text but tokens>0 → not_returned (suppressed); else none. The encrypted
    case is set explicitly by the adapter that can see the empty reasoning item."""
    if reasoning:
        return ReasoningStatus.INLINE if inline else ReasoningStatus.PLAINTEXT
    if reasoning_tokens > 0:
        return ReasoningStatus.NOT_RETURNED
    return ReasoningStatus.NONE


def _build_trace(*, reasoning: str, reasoning_tokens: int, status: str,
                 provider_source: str, backend: str, model_id: str | None,
                 finish_reason: str,
                 observed_message_keys: tuple[str, ...] = ()) -> dict:
    """The uniform reasoning-trace dict persisted per call. committed_justification
    is stamped later by the structured scorer (it owns the answer format).

    `provider_source` names the field the reasoning ACTUALLY came from, and is
    "" when none supplied any — it used to name the single field the code tried,
    whether or not that field existed in the reply. `observed_message_keys` is
    what the reply actually carried in that case, so `status: "none"` beside a
    large `out_tokens` can be diagnosed from the durable record instead of being
    read as "this model did not reason".
    """
    return {
        "free_cot": reasoning,
        "status": status,
        "reasoning_tokens": reasoning_tokens,
        "provider_source": provider_source,
        "observed_message_keys": list(observed_message_keys),
        "backend": backend,
        "model_id": model_id,
        "finish_reason": finish_reason,
        "committed_justification": {"support": None, "objection": None, "source": None},
    }


# Cap on how many token positions of the output distribution are inlined into
# the thread-local call log. Chosen to comfortably hold a verdict-only reply
# (measured at 14 completion tokens) while refusing to inline a reasoning-first
# one (median ~437).
_LOGPROBS_CALL_LOG_MAX_POSITIONS = 64


def _normalize_openai_logprobs(choice) -> list[dict] | None:
    """Normalize an OpenAI-shaped `choice.logprobs` into our flat form.

    Returns None when the response carried no logprobs object at all, and a
    (possibly empty) list otherwise — the caller turns that into the
    three-valued status.

    Two provider behaviours are handled here rather than at every call site:

    1. The entry scalar reports the ARGMAX, not the SAMPLED token. mlx_lm.server
       builds each content entry as `dict(top[0], top_logprobs=top)`
       (server.py:1318-1321), so the entry is by construction the highest-scoring
       alternative — which is NOT what was emitted once sampling is stochastic.
       Measured at temperature 2.0: 12/12 positions where the returned text said
       one token and `logprobs.content[].token` said another. At temperature 0
       argmax == sample and the divergence is invisible, which is exactly why
       `call()` refuses the combination rather than trusting the caller.
       We therefore recompute `logprob` as the max over `top` and require
       consumers that want a specific label to sum over `top` themselves — that
       path is correct under either reading.
    2. Entries may be bare `{}` (mlx emits that for a position with no
       alternatives) or carry ids without token strings. Missing fields become
       "" / -inf rather than raising, so one odd position cannot fail a run.
    """
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None
    content = getattr(lp, "content", None)
    if content is None:
        return []
    out: list[dict] = []
    for entry in content:
        if isinstance(entry, dict):
            tok = entry.get("token", "") or ""
            raw_lp = entry.get("logprob")
            alts_src = entry.get("top_logprobs") or []
        else:
            tok = getattr(entry, "token", "") or ""
            raw_lp = getattr(entry, "logprob", None)
            alts_src = getattr(entry, "top_logprobs", None) or []
        alts: list[dict] = []
        for a in alts_src:
            if isinstance(a, dict):
                a_tok, a_lp = a.get("token", "") or "", a.get("logprob")
            else:
                a_tok = getattr(a, "token", "") or ""
                a_lp = getattr(a, "logprob", None)
            if a_lp is not None:
                alts.append({"token": a_tok, "logprob": float(a_lp)})
        # Quirk 1: prefer the max over alternatives to the entry scalar.
        if alts:
            best = max(alts, key=lambda d: d["logprob"])
            tok, lp_val = best["token"], best["logprob"]
        else:
            lp_val = float(raw_lp) if raw_lp is not None else float("-inf")
        out.append({"token": tok, "logprob": lp_val, "top": alts})
    return out


@dataclass
class ModelResponse:
    """Response from a model call with unified fields."""
    content: str            # Final assistant message (may be empty if all reasoning)
    reasoning: str          # Chain-of-thought text (may be empty) — kept for compat
    tokens: int             # Total completion tokens
    raw_text: str           # Content + reasoning joined (for parsing)
    finish_reason: str      # "stop", "length", etc.
    prompt_tokens: int = -1  # Input tokens (-1 if backend doesn't report)
    # Uniform CoT capture (free_cot == reasoning by construction). Defaults to a
    # status="none" trace so every existing ModelResponse(...) call-site stays valid.
    reasoning_trace: dict = field(default_factory=lambda: _build_trace(
        reasoning="", reasoning_tokens=-1, status=ReasoningStatus.NONE,
        provider_source="", backend="", model_id=None, finish_reason="stop"))
    # Bounded wire provenance for dependency-free raw transports. Complete
    # response bytes may be retained as base64 replay evidence, but bearer
    # material is always redacted and never appears here.
    transport_trace: dict = field(default_factory=dict)
    # Per-token output distribution, normalized across backends to
    #   [{"token": str, "logprob": float,
    #     "top": [{"token": str, "logprob": float}, ...]}, ...]
    # in generated-token order. `top` is the provider's top-k alternatives AT
    # that position (may be empty if only the sampled token was returned).
    #
    # THREE-VALUED, and the distinction is load-bearing. A provider that
    # accepts `top_logprobs` and returns nothing must never be mistaken for a
    # provider that was never asked — we have already measured exactly that
    # failure on the Bedrock responses route for gemma, where the parameter is
    # accepted and an EMPTY array comes back. Read `logprobs_status`, never
    # `if response.logprobs:`:
    #   "not_requested" — logprobs is None; nobody asked.
    #   "unsupported"   — the registry says this route cannot supply them, so
    #                     the request was not sent. logprobs is None.
    #   "empty"         — asked, provider answered, returned no positions.
    #                     logprobs is []. THIS IS A SILENT PROVIDER REFUSAL.
    #   "ok"            — logprobs is a non-empty list.
    logprobs: list[dict] | None = None
    logprobs_status: str = "not_requested"


def reasoning_wire_keys(effort: str | None, *, strict: bool = False) -> dict:
    """The wire keys that express a reasoning-effort intent, for ANY transport.

    Split out of :meth:`ModelClient.call` for the same reason
    ``build_probe_request`` was split out of ``read_probe``: the corpus-scale
    shard runner drives a bare ``httpx.Client`` and cannot adopt our transport,
    but it must express "no CoT" the SAME way — and this is a rule no single
    key carries.

    Two mechanisms, and which one works is a property of the backend:

      * ``reasoning_effort`` — the standard OpenAI extension, honored by Google
        AI Studio and some Ollama builds;
      * ``chat_template_kwargs.enable_thinking`` — what Ollama-served Gemma
        actually honors. It SILENTLY DROPS ``reasoning_effort="none"``, leaving
        thinking on at the model's default.

    So "none" sends BOTH and lets whichever the backend understands win. A
    caller that sends only one gets a silent failure on half the substrates: the
    model deliberates anyway, and the only symptom is a bill.

    ``strict`` backends (Google's OpenAI-compat) 400 on unknown extra_body
    fields, so they get the standard key alone.
    """
    if not effort:
        return {}
    keys: dict = {"reasoning_effort": effort}
    if effort == "none" and not strict:
        keys["chat_template_kwargs"] = {"enable_thinking": False}
    return keys


class ModelClient:
    """Unified client for calling LLMs across backends.

    Telemetry: every successful or failed call() appends one entry to a
    thread-local call log (`_tls.call_log`). Callers can snapshot and
    clear the log via `pop_call_log()`. The thread-local design works
    cleanly with ThreadPoolExecutor — each worker accumulates its own
    record's calls without cross-contamination.

    Wall-time guard: SDK/local backends (openai_compat, anthropic,
    transformers_local) are dispatched on a class-level ThreadPoolExecutor and
    `result(timeout=N)` enforces a hard wall-time cap. All three dependency-free
    paid Bedrock transports (bedrock_converse, bedrock_responses, and
    bedrock_responses_raw) are deliberately synchronous instead: an executor
    timeout cannot cancel an already-running HTTP side effect, so dispatching them
    would let a timeout abandon a still-billing request and create an
    unobserved billed duplicate. What bounds each of them differs, and the
    difference matters:
      • the formal `*_raw` lane holds a monotonic ABSOLUTE deadline spanning
        DNS/connect/TLS/send/read, and the sealed parent additionally owns the
        outer absolute child-process deadline;
      • the two legacy urllib lanes (bedrock_converse, bedrock_responses) bound
        each individual socket operation at the configured timeout (300s
        converse, 600s responses) AND hold their own monotonic ABSOLUTE deadline
        across the body read (`_read_body_within_deadline`), so a slow-drip
        response cannot outlive the configured timeout. They are NOT part of the
        former sealed formal runner: its provider wire request built a canonical
        wire body only for the `bedrock_responses_raw` backend and returned
        `(None, None)` for every other backend — openai_compat included — so no
        sealed-parent deadline applies to these two lanes.
    In every case the request settles before control returns.
    """
    # Shared across instances; each call only consumes one slot for its
    # duration. 8 max workers covers single-threaded scoring + a few
    # concurrent ModelClient instances without bloat.
    import concurrent.futures as _cf
    _WALL_POOL = _cf.ThreadPoolExecutor(max_workers=8,
                                        thread_name_prefix="mc-wall")

    def __init__(
        self,
        model_name: str,
        *,
        bedrock_bearer_token: str | None = None,
        bedrock_ca_bundle: str | None = None,
        bedrock_ca_bundle_sha256: str | None = None,
    ):
        """Construct a model client.

        Formal Bedrock children pass the three keyword-only values directly
        after reading the bearer token from their authenticated pipe.  This
        avoids inherited credential/trust environment state.  Existing callers
        remain source-compatible: absent overrides retain the registry + env
        behavior used by legacy and interactive runs.
        """
        self._bedrock_bearer_token_override = bedrock_bearer_token
        self._bedrock_ca_bundle_override = bedrock_ca_bundle
        self._bedrock_ca_bundle_sha256_override = bedrock_ca_bundle_sha256
        # Resolve historical/abbreviated names to canonical so old --model calls
        # and recorded runs keep working; set name first so setup helpers can use
        # it in error messages.
        model_name = _MODEL_ALIASES.get(model_name, model_name)
        self.model_name = model_name
        # Thread-local call log; see `pop_call_log()`.
        import threading as _threading
        self._tls = _threading.local()
        if model_name in LOCAL_MODELS:
            # Own copy: callers mutate `client.config` per call (the runner
            # ratchets `config["timeout"]` down to the action deadline), so the
            # registry stays the process-wide source of truth. Values are all
            # scalars, so a shallow copy is a full copy.
            self.config = dict(LOCAL_MODELS[model_name])
            self.backend = self.config.get("backend", "openai_compat")
            if (
                any(
                    value is not None
                    for value in (
                        bedrock_bearer_token,
                        bedrock_ca_bundle,
                        bedrock_ca_bundle_sha256,
                    )
                )
                and not self.backend.startswith("bedrock_")
            ):
                raise ValueError(
                    "explicit Bedrock credentials/trust require a Bedrock backend"
                )
            if self.backend == "openai_compat":
                self._setup_openai_client()
            elif self.backend == "transformers_local":
                self._setup_transformers_client()
            elif self.backend == "bedrock_converse":
                self._setup_bedrock_token()
            elif self.backend == "bedrock_responses":
                self._setup_bedrock_token()
            elif self.backend == "bedrock_responses_raw":
                self._setup_bedrock_responses_transport()
            else:
                raise ValueError(
                    f"Unknown backend {self.backend!r} for model {model_name!r}"
                )
        elif model_name.startswith("claude-"):
            self.config = {"model_id": model_name, "reasoning_in_content": False,
                           "max_tokens": 2000, "timeout": 120}
            self.backend = "anthropic"
            self._setup_anthropic_client()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _get_call_log(self) -> list[dict]:
        if not hasattr(self._tls, "call_log"):
            self._tls.call_log = []
        return self._tls.call_log

    def pop_call_log(self) -> list[dict]:
        """Return the current thread's call log and clear it."""
        log = self._get_call_log()
        snapshot = list(log)
        log.clear()
        return snapshot

    def _invoke_with_wall_timeout(self, fn, timeout: int, *args, **kwargs):
        """Run `fn(*args, **kwargs)` with a hard wall-time cap.

        Only SDK and local backends are dispatched here — never a billed
        dependency-free Bedrock transport, which runs synchronously in the
        caller so a timeout cannot abandon a live paid request.

        On timeout: raise TimeoutError immediately. The in-flight thread
        is abandoned (cannot cleanly cancel a running urllib3 request);
        urllib3's transport timeout will eventually reap it. The leak is
        bounded — at most one zombie thread per timeout incident, and
        the pool's max_workers=8 caps total concurrent leaks.

        The OpenAI SDK's `timeout` field is per-connection / per-chunk and
        does NOT bound total wall time on streaming generations: a call can
        run far past the configured `timeout` while individual chunks keep
        arriving. This wrapper is the actual circuit breaker.
        """
        future = self._WALL_POOL.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except self._cf.TimeoutError as e:
            raise TimeoutError(
                f"ModelClient.call exceeded {timeout}s wall-clock time"
            ) from e

    def _setup_openai_client(self):
        import os
        from openai import OpenAI
        # Hosted endpoints (Google, etc.) need a real key; local Ollama
        # endpoints don't care. `api_key_env` in the model config names the
        # env var to read; absence falls back to "not-needed" for local.
        api_key = "not-needed"
        env_var = self.config.get("api_key_env")
        if env_var:
            api_key = os.environ.get(env_var)
            if not api_key:
                raise RuntimeError(
                    f"model {self.model_name!r} requires {env_var} in the "
                    f"environment (not set). Source it from your .env or "
                    f"export it before instantiating ModelClient."
                )
        self._client = OpenAI(
            base_url=self.config["base_url"],
            api_key=api_key,
        )

    def _setup_anthropic_client(self):
        import anthropic
        self._client = anthropic.Anthropic()

    def _load_bedrock_token(self) -> str:
        """Read one Bedrock bearer token without ever including it in errors."""
        import os
        explicit = getattr(self, "_bedrock_bearer_token_override", None)
        if explicit is not None:
            return explicit
        env_var = self.config.get("api_key_env")
        token = os.environ.get(env_var) if env_var else None
        if not token:
            raise RuntimeError(
                f"model {self.model_name!r} requires {env_var} in the "
                f"environment (not set). Source it from your .env or export it "
                f"before instantiating ModelClient."
            )
        return token

    def _setup_bedrock_token(self):
        """Set up the explicit, proxy-free TLS transport for Responses/Converse.

        Responses and Converse authenticate the same way.  Both load only the
        configured CA PEM bytes; urllib's ambient proxy settings, redirect
        behavior, and platform trust store are absent.
        """
        self._bedrock_token = self._load_bedrock_token()
        from indra_belief.bedrock_transport_base import build_pinned_https_opener

        explicit_ca_bundle = getattr(self, "_bedrock_ca_bundle_override", None)
        ca_bundle = (
            explicit_ca_bundle
            if explicit_ca_bundle is not None
            else self.config.get("tls_ca_bundle", _FIXED_BEDROCK_TLS_CA_BUNDLE)
        )
        (
            self._bedrock_url_opener,
            self._bedrock_tls_ca_bundle_sha256,
            _,
        ) = build_pinned_https_opener(ca_bundle)
        expected_ca_sha256 = getattr(
            self, "_bedrock_ca_bundle_sha256_override", None
        )
        if (
            expected_ca_sha256 is not None
            and self._bedrock_tls_ca_bundle_sha256 != expected_ca_sha256
        ):
            raise RuntimeError("explicit Bedrock CA bundle differs from its SHA-256")

    def _setup_bedrock_responses_transport(self):
        """Set up the frozen stdlib transport for the formal Gemma-4 lanes."""
        self._bedrock_token = self._load_bedrock_token()
        from indra_belief.bedrock_responses_transport import (
            RawBedrockResponsesTransport,
        )

        endpoint = self.config.get("responses_endpoint")
        expected = self.config.get("expected_responses_endpoint")
        explicit_ca_sha256 = getattr(
            self, "_bedrock_ca_bundle_sha256_override", None
        )
        ca_sha256 = (
            explicit_ca_sha256
            if explicit_ca_sha256 is not None
            else self.config.get("tls_ca_bundle_sha256")
        )
        if (
            not isinstance(endpoint, str)
            or not isinstance(expected, str)
            or not isinstance(ca_sha256, str)
        ):
            raise RuntimeError(
                f"model {self.model_name!r} lacks its frozen Responses transport"
            )
        explicit_ca_bundle = getattr(self, "_bedrock_ca_bundle_override", None)
        self._bedrock_responses_transport = RawBedrockResponsesTransport(
            endpoint=endpoint,
            expected_endpoint=expected,
            expected_model_id=self.config["model_id"],
            bearer_token=self._bedrock_token,
            ca_bundle=(
                explicit_ca_bundle
                if explicit_ca_bundle is not None
                else self.config.get("tls_ca_bundle")
            ),
            expected_ca_bundle_sha256=ca_sha256,
            max_request_bytes=int(
                self.config.get("max_request_bytes", 16 * 1024 * 1024)
            ),
            max_response_bytes=int(
                self.config.get("max_response_bytes", 16 * 1024 * 1024)
            ),
        )

    def _call_bedrock_converse(
        self, system: str, messages: list[dict], mt: int, temp: float, timeout: int,
    ) -> ModelResponse:
        """Native AWS Bedrock Converse API (Anthropic/Claude models).

        Bearer-token auth (the Bedrock API key), no SigV4 / boto3. Maps the
        unified (system, messages) shape onto Converse and parses
        output.message.content[].text + usage.{input,output}Tokens. See the
        bedrock-claude-* registry comment for the route/id/availability gotchas.
        """
        import json as _json
        import socket as _socket
        import time as _time
        import urllib.error
        import urllib.request

        # OpenAI-style messages -> Converse content blocks; the system prompt is
        # a top-level Converse field, not a message role.
        conv_messages = [
            {"role": m.get("role", "user"),
             "content": [{"text": m.get("content", "") or ""}]}
            for m in messages
        ]
        # NB: `temperature` is intentionally omitted. Claude 4.x on Bedrock
        # (e.g. opus-4-8) rejects it with HTTP 400 "temperature is deprecated for
        # this model"; the unified `temp` arg is accepted-and-ignored here.
        body: dict = {
            "messages": conv_messages,
            "inferenceConfig": {"maxTokens": mt},
        }
        if system:
            body["system"] = [{"text": system}]
        url = f"{self.config['base_url']}/model/{self.config['model_id']}/converse"
        req = urllib.request.Request(
            url,
            data=_json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._bedrock_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        # Absolute wall bound, armed BEFORE the first byte leaves. `open(...,
        # timeout=timeout)` stays the per-operation connect/header bound; the
        # deadline is what a slow-drip body cannot outlive now that this lane
        # runs synchronously with no thread-pool wall cap.
        deadline = _time.monotonic() + timeout
        try:
            with self._bedrock_url_opener.open(req, timeout=timeout) as resp:
                payload = _json.loads(
                    _read_body_within_deadline(
                        resp, absolute_deadline=deadline, lane_label="Converse"
                    ).decode("utf-8")
                )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            err = RuntimeError(f"Bedrock Converse HTTP {e.code}: {detail}")
            # Authoritative status for _is_rate_limit_error, stamped under a name
            # deliberately outside the retired spend guard's candidate set
            # (status / status_code / http_status / response.status* /
            # transport_trace.response_http_status), so that guard classified
            # this lane from the unchanged message text alone.
            err._bedrock_http_status = e.code
            raise err from e
        except _LaneDeadlineExpired as e:
            # Signalled from INSIDE the `with`, handled here: the context manager
            # already ran `resp.__exit__`, so the billed request is settled before
            # this TimeoutError can reach the caller. Message is the lane's
            # existing one, byte for byte.
            raise TimeoutError(
                f"Bedrock Converse request timed out after {timeout}s"
            ) from e
        except TimeoutError as e:
            # This lane now runs in the caller (see call()), so its socket
            # deadline IS the caller-visible deadline. Normalize to the same
            # TimeoutError the raw transports raise, so classify_provider_failure
            # and grounding's `type(e).__name__` abstain reason do not drift.
            raise TimeoutError(
                f"Bedrock Converse request timed out after {timeout}s"
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (_socket.timeout, TimeoutError)):
                raise TimeoutError(
                    f"Bedrock Converse request timed out after {timeout}s"
                ) from e
            raise

        blocks = payload.get("output", {}).get("message", {}).get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        usage = payload.get("usage", {}) or {}
        finish = payload.get("stopReason", "stop") or "stop"
        return ModelResponse(
            content=text,
            reasoning="",
            tokens=usage.get("outputTokens", -1),
            raw_text=text,
            finish_reason=finish,
            prompt_tokens=usage.get("inputTokens", -1),
            reasoning_trace=_build_trace(
                reasoning="", reasoning_tokens=-1, status=ReasoningStatus.NONE,
                provider_source="bedrock_converse (no reasoning channel)",
                backend=self.backend, model_id=self.config.get("model_id"),
                finish_reason=finish),
        )

    def _call_bedrock_responses(
        self, system: str, messages: list[dict], mt: int, temp: float, timeout: int,
        reasoning_effort: str | None = None,
    ) -> ModelResponse:
        """Bedrock mantle OpenAI-compatible RESPONSES API (POST {base_url}/responses).

        Some Bedrock-served models (Gemma 4, and other reasoning-suppressing
        families) run chain-of-thought server-side but DROP it on Chat
        Completions — only the Responses API surfaces it. Maps the unified
        (system, messages) shape onto a Responses request and reads reasoning +
        answer out of the output[] item list. Bearer-token auth, raw HTTP (no
        SDK), mirroring _call_bedrock_converse.

        Verified 2026-06-19 against gemma-4: reasoning is gated to effort="high"
        (medium/none → no reasoning item). CoT comes back as an output item of
        type "reasoning" (content[].text); the answer is the "message" item's
        output_text block — the top-level `output_text` field is null here and
        must not be relied on.
        """
        import json as _json
        import socket as _socket
        import time as _time
        import urllib.error
        import urllib.request

        # system -> top-level `instructions`; user/assistant turns -> typed input
        # items (user=input_text, assistant=output_text). A stray system-role
        # message folds into instructions.
        instructions = system or ""
        input_items: list[dict] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "") or ""
            if role == "system":
                instructions = f"{instructions}\n{text}" if instructions else text
                continue
            ctype = "output_text" if role == "assistant" else "input_text"
            input_items.append(
                {"role": role, "content": [{"type": ctype, "text": text}]}
            )

        body: dict = {
            "model": self.config["model_id"],
            "input": input_items,
            "max_output_tokens": mt,
        }
        if instructions:
            body["instructions"] = instructions
        # Reasoning is gated to "high" on this endpoint; omit for none/unset so
        # extraction sub-calls (reasoning_effort="none") stay fast and cheap.
        effort = reasoning_effort if reasoning_effort is not None \
            else self.config.get("reasoning_effort")
        if effort and effort != "none":
            body["reasoning"] = {"effort": effort}
        # NB: temperature intentionally omitted — reasoning models on this
        # endpoint can reject non-default temperature; `temp` is accepted-and-
        # ignored (same posture as _call_bedrock_converse).

        url = f"{self.config['base_url']}/responses"
        req = urllib.request.Request(
            url,
            data=_json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._bedrock_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        # Same absolute wall bound as _call_bedrock_converse, armed before the
        # first byte leaves; see _read_body_within_deadline.
        deadline = _time.monotonic() + timeout
        try:
            with self._bedrock_url_opener.open(req, timeout=timeout) as resp:
                payload = _json.loads(
                    _read_body_within_deadline(
                        resp, absolute_deadline=deadline, lane_label="Responses"
                    ).decode("utf-8")
                )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            err = RuntimeError(f"Bedrock Responses HTTP {e.code}: {detail}")
            # Same private-attribute rationale as _call_bedrock_converse: an
            # authoritative status for the rate-limit test that the spend-guard
            # status extractor deliberately cannot see.
            err._bedrock_http_status = e.code
            raise err from e
        except _LaneDeadlineExpired as e:
            # Handled outside the `with`: the response is already closed.
            raise TimeoutError(
                f"Bedrock Responses request timed out after {timeout}s"
            ) from e
        except TimeoutError as e:
            raise TimeoutError(
                f"Bedrock Responses request timed out after {timeout}s"
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (_socket.timeout, TimeoutError)):
                raise TimeoutError(
                    f"Bedrock Responses request timed out after {timeout}s"
                ) from e
            raise

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        had_reasoning_item = False
        for item in payload.get("output", []) or []:
            itype = item.get("type")
            blocks = item.get("content", []) or []
            if itype == "reasoning":
                had_reasoning_item = True  # present even when text-empty (encrypted)
                reasoning_parts += [
                    b["text"] for b in blocks
                    if isinstance(b, dict) and b.get("text")
                ]
            elif itype == "message":
                # The answer is normally an "output_text" block, but accept ANY
                # text-bearing non-reasoning block — guards against Responses-API
                # schema drift silently dropping the verdict (reasoning blocks are
                # handled above, so excluding them keeps CoT out of the answer).
                content_parts += [
                    b["text"] for b in blocks
                    if isinstance(b, dict) and b.get("text")
                    and b.get("type") != "reasoning"
                ]
        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)
        if self.config.get("reasoning_in_content"):
            raw_text = content
        else:
            raw_text = (reasoning + "\n" + content) if reasoning else content

        usage = payload.get("usage", {}) or {}
        # status="incomplete" signals a max_output_tokens cutoff (truncation),
        # mapped to the "length" finish_reason the verdict parser already knows.
        finish = "length" if payload.get("status") == "incomplete" else "stop"
        rtok = _reasoning_tokens(usage)
        if reasoning:
            status = ReasoningStatus.PLAINTEXT          # readable CoT (gemma)
        elif had_reasoning_item and rtok > 0:
            status = ReasoningStatus.ENCRYPTED          # reasoned, summary-only (gpt-5.5)
        else:
            status = _classify_reasoning(reasoning, rtok, inline=False)
        return ModelResponse(
            content=content,
            reasoning=reasoning,
            tokens=usage.get("output_tokens", -1),
            raw_text=raw_text,
            finish_reason=finish,
            prompt_tokens=usage.get("input_tokens", -1),
            reasoning_trace=_build_trace(
                reasoning=reasoning, reasoning_tokens=rtok, status=status,
                provider_source="bedrock_responses.output[].reasoning",
                backend=self.backend, model_id=self.config.get("model_id"),
                finish_reason=finish),
        )

    def _call_bedrock_responses_raw(
        self,
        system: str,
        messages: list[dict],
        mt: int,
        temp: float,
        timeout: int,
        reasoning_effort: str | None = None,
    ) -> ModelResponse:
        """Bounded raw Responses adapter used by the formal Gemma-4 lanes.

        ``temp`` remains accepted-and-ignored, matching the legacy adapter and
        avoiding a field that reasoning models on this route can reject.
        """
        from indra_belief.bedrock_responses_transport import (
            build_bedrock_responses_body,
        )

        effort = (
            reasoning_effort
            if reasoning_effort is not None
            else self.config.get("reasoning_effort")
        )
        body = build_bedrock_responses_body(
            model_id=self.config["model_id"],
            system=system,
            messages=messages,
            max_output_tokens=mt,
            reasoning_effort=effort,
        )
        result = self._bedrock_responses_transport.call(body, timeout=timeout)
        content = result.content
        reasoning = result.reasoning
        if self.config.get("reasoning_in_content"):
            raw_text = content
        else:
            raw_text = (reasoning + "\n" + content) if reasoning else content

        if reasoning:
            status = ReasoningStatus.PLAINTEXT
        elif result.reasoning_item_present and result.reasoning_tokens > 0:
            status = ReasoningStatus.ENCRYPTED
        else:
            status = _classify_reasoning(
                reasoning,
                result.reasoning_tokens,
                inline=bool(self.config.get("reasoning_in_content")),
            )
        return ModelResponse(
            content=content,
            reasoning=reasoning,
            tokens=result.output_tokens,
            raw_text=raw_text,
            finish_reason=result.finish_reason,
            prompt_tokens=result.prompt_tokens,
            reasoning_trace=_build_trace(
                reasoning=reasoning,
                reasoning_tokens=result.reasoning_tokens,
                status=status,
                provider_source="bedrock_responses.output[].reasoning",
                backend=self.backend,
                model_id=self.config.get("model_id"),
                finish_reason=result.finish_reason,
            ),
            transport_trace=result.transport_trace,
        )

    def call(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.1,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        kind: str = "unknown",
        top_logprobs: int | None = None,
    ) -> ModelResponse:
        """Call the model with a system prompt and messages.

        Returns ModelResponse with unified fields regardless of backend.

        Retry doctrine: the only retry-on-error is for 429 rate
        limits, which respect the server-requested delay. Timeouts and
        connection errors raise on the first occurrence — callers
        (parse_evidence, verify_grounding) abstain via their existing
        TimeoutError handlers. Retrying other errors only amplifies
        endpoint degradation: it multiplies an already-slow record's
        cost by the retry count before failing anyway.

        `response_format` constrains the output. Pass
        `{"type": "json_object"}` to force JSON-only output on backends that
        support it. Sub-calls that consume JSON (parse_evidence, grounding)
        opt in; backends that don't honor the constraint fall through to
        the previous behavior (caller still parses tolerantly).

        `reasoning_effort` (when set) overrides the per-model config's
        reasoning_effort. Pass "none" on sub-calls that are pure extraction
        (parse_evidence, verify_grounding) — at the config default of
        "medium", a thinking model can burn its whole token budget on
        reasoning_content before emitting JSON, causing silent truncation
        on the per-call budget. "none" keeps the reasoning brief and lets
        content populate.

        `kind` is a free-form telemetry label. Sub-callers pass
        "parse_evidence" / "verify_grounding" / "monolithic" so post-hoc
        analysis can stratify latency and truncation rates per call type.

        `top_logprobs` (when set) asks the backend for the top-k output
        distribution at every generated token position, so callers can compute
        a renormalised label probability at the verdict token instead of
        relying on the model's verbalized `confidence` field. Read the outcome
        from `ModelResponse.logprobs_status`, NOT from truthiness of
        `.logprobs` — "asked and silently refused" is a real and measured
        provider behavior here and it must not read as "never asked".

        The request is only sent on routes whose registry entry declares
        `supports_logprobs`; anywhere else the call proceeds normally and the
        response is stamped status="unsupported". This is deliberate: Google's
        strict OpenAI-compat endpoint 400s on unknown fields, and a 400 on
        every scoring call would be masked by the substrate-only fallback.
        `max_top_logprobs` in the registry clamps k to what the route accepts
        (mlx_lm.server rejects k > 11).
        """
        import time as _time

        mt = max_tokens or self.config.get("max_tokens", 2000)
        timeout = self.config.get("timeout", 120)
        # Approximate prompt size up-front for telemetry (cheap when the
        # backend doesn't report prompt_tokens in the response).
        prompt_chars = len(system or "") + sum(
            len(m.get("content", "") or "") for m in messages
        )

        # 429 quota retries are bounded by this counter; everything else
        # raises on first occurrence (see retry doctrine above).
        # Paid corpus runs wrap this client with the append-only spend guard.
        # Their retry budget is owned by the runner's two-attempt execution
        # identity, so an opaque in-client retry would evade the attempt ledger.
        # Unguarded behavior remains unchanged.
        rate_limit_retries = (
            0 if getattr(self, "_spend_guard_disable_internal_retries", False) else 5
        )
        t_start = _time.time()

        # Resolve the logprobs ask against what this route can actually do.
        # `eff_top_logprobs is None` means "do not put the field on the wire".
        eff_top_logprobs = None
        logprobs_unsupported = False
        if top_logprobs is not None:
            if top_logprobs < 1:
                raise ValueError("top_logprobs must be >= 1 when requested")
            if temperature > 0:
                # Not a style preference — a correctness precondition. The
                # returned per-position entry is the ARGMAX (mlx builds it as
                # `dict(top[0], ...)`), while `content` holds what was SAMPLED.
                # Above temperature 0 those diverge (measured 12/12 at temp 2.0),
                # so the token stream we scan to locate the verdict is not the
                # token stream that produced the text we parse the verdict from.
                # The two would disagree silently and only on the hard cases.
                raise ValueError(
                    "top_logprobs requires temperature=0: above it the reported "
                    "argmax token stream diverges from the sampled text, so the "
                    f"verdict position cannot be trusted (got temperature={temperature})"
                )
            if self.config.get("supports_logprobs"):
                cap = self.config.get("max_top_logprobs")
                eff_top_logprobs = min(top_logprobs, cap) if cap else top_logprobs
            else:
                # Declared incapable: proceed without the field rather than
                # risk a 400 that the substrate-only fallback would hide.
                logprobs_unsupported = True

        try:
            while True:
                try:
                    if self.backend == "openai_compat":
                        response = self._invoke_with_wall_timeout(
                            self._call_openai_compat, timeout,
                            system, messages, mt, temperature, timeout,
                            response_format=response_format,
                            reasoning_effort=reasoning_effort,
                            top_logprobs=eff_top_logprobs,
                        )
                    elif self.backend == "anthropic":
                        response = self._invoke_with_wall_timeout(
                            self._call_anthropic, timeout,
                            system, messages, mt, temperature, timeout,
                        )
                    elif self.backend == "transformers_local":
                        response = self._invoke_with_wall_timeout(
                            self._call_transformers, timeout,
                            system, messages, mt, temperature,
                        )
                    # ── Dependency-free paid Bedrock lanes: synchronous ──────
                    # A future that times out cannot cancel an already running
                    # HTTP request.  Returning while that paid side effect is
                    # still live would let the runner start an unobserved
                    # duplicate attempt.  All three billed raw-HTTP transports
                    # therefore execute in the scoring worker itself, so the
                    # request always settles before an error can reach the retry
                    # ledger.  The formal *_raw lane is bounded by its
                    # monotonic absolute transport deadline, with the sealed
                    # parent owning the outer child-process deadline; the two
                    # legacy urllib lanes are bounded by their per-socket-
                    # operation timeout PLUS their own monotonic absolute
                    # deadline over the body read, and normalize either expiry
                    # to TimeoutError before it escapes.
                    elif self.backend == "bedrock_converse":
                        response = self._call_bedrock_converse(
                            system, messages, mt, temperature, timeout,
                        )
                    elif self.backend == "bedrock_responses":
                        response = self._call_bedrock_responses(
                            system, messages, mt, temperature, timeout,
                            reasoning_effort=reasoning_effort,
                        )
                    elif self.backend == "bedrock_responses_raw":
                        response = self._call_bedrock_responses_raw(
                            system, messages, mt, temperature, timeout,
                            reasoning_effort=reasoning_effort,
                        )
                    else:
                        raise ValueError(f"Unknown backend: {self.backend}")
                    if logprobs_unsupported:
                        # The caller asked; this route cannot answer. Say so
                        # explicitly so downstream cannot read the absence as
                        # a measurement.
                        response.logprobs_status = "unsupported"
                    call_row = {
                        "kind": kind,
                        "duration_s": round(_time.time() - t_start, 3),
                        "prompt_chars": prompt_chars,
                        "prompt_tokens": response.prompt_tokens,
                        "out_tokens": response.tokens,
                        "finish_reason": response.finish_reason,
                        "max_tokens": mt,
                        # Layer B capture — persist raw LLM I/O for tracing.
                        # Lets the viewer reconstruct what the model saw and
                        # what it said. Cost: ~1-10KB per call (typical) up
                        # to 30+KB when reasoning_content runs hot.
                        "system": system,
                        "messages": messages,
                        "model_id": self.config.get("model_id"),
                        "content": response.content,
                        "reasoning": response.reasoning,
                        "raw_text": response.raw_text,
                        # Uniform CoT capture (status + tokens + provenance, and
                        # committed support/objection stamped later by the
                        # structured scorer). Same dict object the scorer mutates.
                        "reasoning_trace": response.reasoning_trace,
                    }
                    if response.transport_trace:
                        call_row["transport_trace"] = response.transport_trace
                    if response.logprobs_status != "not_requested":
                        # Always record the STATUS — it is one short string and
                        # it is what distinguishes a real measurement from a
                        # silent provider refusal after the fact.
                        call_row["logprobs_status"] = response.logprobs_status
                        call_row["n_logprob_positions"] = (
                            len(response.logprobs)
                            if response.logprobs is not None else 0
                        )
                        # The array itself is only persisted when it is small.
                        # A reasoning-first reply runs a median ~437 positions;
                        # at k=11 that is ~5k entries per call, which would add
                        # hundreds of MB to a 1.6k-row gold run. Callers that
                        # need the distribution at one position must extract it
                        # from the live ModelResponse (see logprobs.py) and
                        # persist that, rather than relying on this field.
                        if (response.logprobs is not None
                                and len(response.logprobs) <= _LOGPROBS_CALL_LOG_MAX_POSITIONS):
                            call_row["logprobs"] = response.logprobs
                    self._get_call_log().append(call_row)
                    return response
                except Exception as e:
                    # 429 / rate-limit: respect the server's requested delay.
                    # This is the ONLY in-client retry. Classification is
                    # status-AUTHORITATIVE (see _is_rate_limit_error): a
                    # structured 429 retries, ANY other structured status is
                    # final however its body reads, and message text is consulted
                    # only when the provider reported no status at all.
                    if _is_rate_limit_error(e) and rate_limit_retries > 0:
                        delay = _parse_retry_delay(str(e))
                        # tiny safety pad so the next request lands clean
                        _time.sleep(delay + 1)
                        rate_limit_retries -= 1
                        continue
                    raise
        except BaseException as e:
            # The spend WAL consumes exactly one log row immediately after
            # every return *or raise*.  Record process-control exceptions too,
            # then re-raise them unchanged; the wrapper remains fail-closed and
            # can persist honest evidence instead of finding an empty TLS log.
            call_row = {
                "kind": kind,
                "duration_s": round(_time.time() - t_start, 3),
                "prompt_chars": prompt_chars,
                "prompt_tokens": -1,
                "out_tokens": 0,
                "finish_reason": None,
                "max_tokens": mt,
                "error": type(e).__name__,
                "error_detail": str(e),
                "system": system,
                "messages": messages,
                "model_id": self.config.get("model_id"),
            }
            transport_trace = getattr(e, "transport_trace", None)
            if (
                not isinstance(transport_trace, dict)
                and self.backend == "bedrock_responses_raw"
            ):
                # Failures before a raw transport can attach its own trace still
                # retain the exact pre-side-effect request commitment.
                try:
                    effort = (
                        reasoning_effort
                        if reasoning_effort is not None
                        else self.config.get("reasoning_effort")
                    )
                    from indra_belief.bedrock_responses_transport import (
                        build_bedrock_responses_body,
                    )

                    body = build_bedrock_responses_body(
                        model_id=self.config["model_id"],
                        system=system,
                        messages=messages,
                        max_output_tokens=mt,
                        reasoning_effort=effort,
                    )
                    transport_trace = (
                        self._bedrock_responses_transport.request_trace(body)
                    )
                except BaseException:
                    transport_trace = None
            if isinstance(transport_trace, dict):
                call_row["transport_trace"] = transport_trace
            self._get_call_log().append(call_row)
            raise

    def _call_openai_compat(
        self, system: str, messages: list[dict], mt: int, temp: float, timeout: int,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        top_logprobs: int | None = None,
    ) -> ModelResponse:
        full_messages = [{"role": "system", "content": system}] + messages
        kwargs = dict(
            model=self.config["model_id"],
            messages=full_messages,
            max_tokens=mt,
            temperature=temp,
            timeout=timeout,
        )
        if top_logprobs is not None:
            # Both fields go on the wire together: OpenAI, vLLM and llama.cpp
            # all reject `top_logprobs` unless `logprobs` is true, and
            # mlx_lm.server only populates token_logprobs when `logprobs` is
            # set. Callers reach this only for routes declaring
            # `supports_logprobs`, and k is already clamped to the route cap.
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs
        if response_format is not None:
            # OpenAI / Google AI Studio honor `response_format` directly.
            # Ollama-backed endpoints expose JSON mode under their native
            # `format` field — set it via extra_body as a fallback for
            # backends where response_format alone isn't enough.
            kwargs["response_format"] = response_format
        # Pass backend-specific options via extra_body. Two reasoning-
        # control mechanisms are honored across backends:
        #   - `reasoning_effort` (low/medium/high): standard OpenAI extension
        #     used by Google AI Studio and some Ollama versions.
        #   - `chat_template_kwargs.enable_thinking` (bool): the actual
        #     mechanism Ollama-served Gemma honors. `reasoning_effort="none"`
        #     is silently dropped by Ollama, leaving thinking ON at the
        #     model's default. We send BOTH; whichever the backend understands
        #     wins. Sending only reasoning_effort is insufficient: Ollama
        #     ignores it, so extraction calls still emit large
        #     reasoning_content blocks despite the "reasoning_effort=none"
        #     intent.
        extra_body = {}
        # Backend strictness: Google's OpenAI-compat endpoint
        # (generativelanguage.googleapis.com/v1beta/openai/) rejects unknown
        # extra_body fields with 400 INVALID_ARGUMENT. Ollama / LiteLLM proxy
        # backends ignore unknown keys. Skip the Ollama-isms when the model
        # is flagged strict_openai_compat.
        strict = bool(self.config.get("strict_openai_compat"))
        effort = reasoning_effort if reasoning_effort is not None \
                 else self.config.get("reasoning_effort")
        extra_body.update(reasoning_wire_keys(effort, strict=strict))
        if self.config.get("num_ctx") and not strict:
            extra_body["num_ctx"] = self.config["num_ctx"]
        if response_format is not None and not strict:
            # Belt-and-suspenders: set Ollama's native `format` field too.
            # Backends that don't recognize it ignore unknown extra_body keys
            # — except Google's strict OpenAI-compat, which 400s. The
            # standard `response_format` (set above) is sufficient for
            # Google AI Studio's JSON mode.
            extra_body["format"] = "json"
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = msg.content or ""
        # Field name for the CoT is not standardised across OpenAI-compatible
        # servers. LiteLLM/Ollama/vLLM use `reasoning_content`; mlx_lm.server
        # emits `reasoning`. Reading only the former against mlx yields an
        # EMPTY content AND an empty raw_text on any reply that spends its whole
        # budget deliberating — the reply looks blank rather than truncated, and
        # the verdict parser then reports absence instead of length-truncation.
        reasoning = (getattr(msg, "reasoning_content", None)
                     or getattr(msg, "reasoning", None) or "")

        # For models where reasoning is IN content, raw_text = content
        # For models with separate reasoning, raw_text = reasoning + content
        if self.config.get("reasoning_in_content"):
            raw_text = content
        else:
            raw_text = (reasoning + "\n" + content) if reasoning else content

        finish = response.choices[0].finish_reason or "stop"
        rtok = _reasoning_tokens(response.usage)
        if top_logprobs is None:
            norm_lp, lp_status = None, "not_requested"
        else:
            norm_lp = _normalize_openai_logprobs(response.choices[0])
            if norm_lp is None:
                # Asked; the response carried no logprobs object whatsoever.
                norm_lp, lp_status = [], "empty"
            else:
                lp_status = "ok" if norm_lp else "empty"
        return ModelResponse(
            content=content,
            reasoning=reasoning,
            tokens=response.usage.completion_tokens,
            raw_text=raw_text,
            finish_reason=finish,
            prompt_tokens=getattr(response.usage, "prompt_tokens", -1),
            logprobs=norm_lp,
            logprobs_status=lp_status,
            reasoning_trace=_build_trace(
                reasoning=reasoning, reasoning_tokens=rtok,
                status=_classify_reasoning(
                    reasoning, rtok,
                    inline=bool(self.config.get("reasoning_in_content"))),
                provider_source="openai_compat.message.reasoning_content",
                backend=self.backend, model_id=self.config.get("model_id"),
                finish_reason=finish),
        )

    def _call_anthropic(
        self, system: str, messages: list[dict], mt: int, temp: float, timeout: int,
    ) -> ModelResponse:
        response = self._client.messages.create(
            model=self.config["model_id"],
            max_tokens=mt,
            system=system,
            messages=messages,
            temperature=temp,
        )
        # Anthropic returns a list of content blocks. Extract:
        #   - `text` block → goes into `content`
        #   - `thinking` block (extended-thinking) → goes into `reasoning`
        # Tool-use blocks intentionally ignored — we don't use native tools
        # (see module docstring). Iterating preserves capture under future
        # API additions where order matters.
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_parts.append(getattr(block, "text", "") or "")
            elif block_type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
        content = "".join(content_parts)
        reasoning = "\n".join(thinking_parts) if thinking_parts else ""
        raw_text = (reasoning + "\n" + content) if reasoning else content
        finish = response.stop_reason or "stop"
        return ModelResponse(
            content=content,
            reasoning=reasoning,
            tokens=response.usage.output_tokens,
            raw_text=raw_text,
            finish_reason=finish,
            prompt_tokens=getattr(response.usage, "input_tokens", -1),
            reasoning_trace=_build_trace(
                reasoning=reasoning, reasoning_tokens=-1,
                status=(ReasoningStatus.PLAINTEXT if reasoning else ReasoningStatus.NONE),
                provider_source="anthropic.thinking blocks",
                backend=self.backend, model_id=self.config.get("model_id"),
                finish_reason=finish),
        )

    _DEFAULT_MEDPSY_PERSONA = (
        "You are MedPsy, a medical and healthcare AI assistant developed by QVAC."
    )

    def _setup_transformers_client(self):
        """In-process transformers backend for local models served without HTTP.

        Loads the tokenizer and model once at init. Replaces the chat
        template's hard-coded persona with the task-specific persona from
        the model config when one is provided — the alternative is for the
        scorer's SYSTEM_PROMPT to share airtime with a clinical-assistant
        framing the model was post-trained to obey.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.config["model_id"]
        dtype_str = self.config.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_str)
        device = self.config.get("device", "cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        persona_override = self.config.get("persona")
        if persona_override and tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.replace(
                self._DEFAULT_MEDPSY_PERSONA, persona_override
            )
        self._tokenizer = tokenizer
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        self._torch = torch

    def _call_transformers(
        self, system: str, messages: list[dict], mt: int, temp: float,
    ) -> ModelResponse:
        torch = self._torch
        full_messages = [{"role": "system", "content": system}] + messages
        prompt_ids = self._tokenizer.apply_chat_template(
            full_messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=self.config.get("enable_thinking", True),
        )
        prompt_ids = prompt_ids.to(self._model.device)
        prompt_len = prompt_ids.shape[-1]

        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id or eos_id
        gen_kwargs = dict(
            max_new_tokens=mt,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        if temp and temp > 0:
            gen_kwargs.update(do_sample=True, temperature=temp)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output = self._model.generate(prompt_ids, **gen_kwargs)
        gen_ids = output[0][prompt_len:]
        completion_tokens = int(gen_ids.shape[-1])
        finish_reason = "length" if completion_tokens >= mt else "stop"

        # skip_special_tokens=False so we can see <think>/</think>; then
        # strip the chat-template control tokens manually.
        raw = self._tokenizer.decode(gen_ids, skip_special_tokens=False)
        for tok in ("<|im_end|>", "<|endoftext|>"):
            raw = raw.replace(tok, "")
        raw = raw.strip()

        if "</think>" in raw:
            reasoning_part, _, content_part = raw.partition("</think>")
            reasoning = reasoning_part.replace("<think>", "").strip()
            content = content_part.strip()
        else:
            reasoning = ""
            content = raw

        raw_text = (reasoning + "\n" + content) if reasoning else content
        return ModelResponse(
            content=content,
            reasoning=reasoning,
            tokens=completion_tokens,
            raw_text=raw_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_len,
            reasoning_trace=_build_trace(
                reasoning=reasoning, reasoning_tokens=-1,
                status=(ReasoningStatus.INLINE if reasoning else ReasoningStatus.NONE),
                provider_source="transformers <think> partition",
                backend=self.backend, model_id=self.config.get("model_id"),
                finish_reason=finish_reason),
        )


# Verdict parsing and score mapping live in `indra_belief.verdict` — this module
# is the model client, not an output parser.
