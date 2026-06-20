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


# Model registry — name → (base_url, model_id, notes)
LOCAL_MODELS: dict[str, dict] = {
    "local-qwen3.5-vl-122b-a10b": {
        "base_url": "http://localhost:8082/v1",
        "model_id": "dealignai/Qwen3.5-VL-122B-A10B-4bit-MLX-CRACK",
        "reasoning_in_content": True,  # CoT is emitted in content
        "typical_tokens": 2500,
        "max_tokens": 8000,
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
    "local-gemma-4-26b": {
        "base_url": "http://localhost:8085/v1",
        "model_id": "mlx-community/gemma-4-26b-a4b-it-8bit",
        "reasoning_in_content": False,  # separate reasoning_content field
        "typical_tokens": 400,
        "max_tokens": 1000,
        "timeout": 60,
    },
    "local-gemma-4-31b": {
        "base_url": "http://localhost:8084/v1",
        "model_id": "mlx-community/gemma-4-31b-it-8bit",
        "reasoning_in_content": False,
        "typical_tokens": 400,
        "max_tokens": 1000,
        "timeout": 60,
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
        "max_tokens": 4096,
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
        # uses backend="bedrock_responses" (POST /openai/v1/responses) with
        # reasoning_effort="high" (only "high" engages a reasoning item;
        # medium/none → none, verified). reasoning_in_content=False ⇒ raw_text =
        # reasoning + answer, so the verdict parse sees both. Gemma is
        # mantle-only (no Converse route) and uses the /openai/v1 path.
        "backend": "bedrock_responses",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "model_id": "google.gemma-4-26b-a4b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "reasoning_in_content": False,
        "reasoning_effort": "high",  # only "high" surfaces a reasoning item
        "typical_tokens": 400,
        "max_tokens": 32000,         # CoT + answer share the budget
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
    # backend=bedrock_responses on the /openai/v1 route. Heavier/slower dense
    # counterpart to the gemma-4-26b-a4b MoE already wired as bedrock-gemma.
    "bedrock-gemma-4-31b": {
        "backend": "bedrock_responses",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "model_id": "google.gemma-4-31b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "typical_tokens": 500,
        "max_tokens": 32000,
        "timeout": 600,
    },
    # Gemma 4 E2B (~2.3B effective, tiny on-device tier) — the ONLY clean sub-10B
    # reasoner on Bedrock: reasons only via the Responses API (probed rc=411,
    # chat drops it), so backend=bedrock_responses. Extreme-cheap floor;
    # capacity-gated for subtle relation logic — validate (n=1606) before trusting.
    # (Other sub-10B mantle models — ministral-3-3b, gemma-3-4b — are non-thinking;
    # our MedPsy-4B reasoner is local-only on noot-1, not on AWS.)
    "bedrock-gemma-4-e2b": {
        "backend": "bedrock_responses",
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "model_id": "google.gemma-4-e2b",
        "api_key_env": "AWS_BEARER_TOKEN_BEDROCK",
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
                    return float(g)
                except ValueError:
                    pass
    return default


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
                 finish_reason: str) -> dict:
    """The uniform reasoning-trace dict persisted per call. committed_justification
    is stamped later by the structured scorer (it owns the answer format)."""
    return {
        "free_cot": reasoning,
        "status": status,
        "reasoning_tokens": reasoning_tokens,
        "provider_source": provider_source,
        "backend": backend,
        "model_id": model_id,
        "finish_reason": finish_reason,
        "committed_justification": {"support": None, "objection": None, "source": None},
    }


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


class ModelClient:
    """Unified client for calling LLMs across backends.

    Telemetry: every successful or failed call() appends one entry to a
    thread-local call log (`_tls.call_log`). Callers can snapshot and
    clear the log via `pop_call_log()`. The thread-local design works
    cleanly with ThreadPoolExecutor — each worker accumulates its own
    record's calls without cross-contamination.

    Wall-time guard: each call is dispatched on a class-level
    ThreadPoolExecutor and `result(timeout=N)` enforces a hard wall-time
    cap. The OpenAI SDK's `timeout` field is a per-chunk / connection
    timeout — for streaming generations it does not bound total wall
    time. This wrapper raises TimeoutError after `timeout` seconds
    regardless of underlying transport behavior. The in-flight urllib3
    request continues until the SDK's transport timeout reaps it (a
    transient thread leak; documented and accepted).
    """
    # Shared across instances; each call only consumes one slot for its
    # duration. 8 max workers covers single-threaded scoring + a few
    # concurrent ModelClient instances without bloat.
    import concurrent.futures as _cf
    _WALL_POOL = _cf.ThreadPoolExecutor(max_workers=8,
                                        thread_name_prefix="mc-wall")

    def __init__(self, model_name: str):
        # Resolve historical/abbreviated names to canonical so old --model calls
        # and recorded runs keep working; set name first so setup helpers can use
        # it in error messages.
        model_name = _MODEL_ALIASES.get(model_name, model_name)
        self.model_name = model_name
        # Thread-local call log; see `pop_call_log()`.
        import threading as _threading
        self._tls = _threading.local()
        if model_name in LOCAL_MODELS:
            self.config = LOCAL_MODELS[model_name]
            self.backend = self.config.get("backend", "openai_compat")
            if self.backend == "openai_compat":
                self._setup_openai_client()
            elif self.backend == "transformers_local":
                self._setup_transformers_client()
            elif self.backend == "bedrock_converse":
                self._setup_bedrock_token()
            elif self.backend == "bedrock_responses":
                self._setup_bedrock_token()
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

    def _setup_bedrock_token(self):
        """Read the Bedrock bearer token for the raw-HTTP backends
        (bedrock_converse + bedrock_responses); both authenticate the same way."""
        import os
        env_var = self.config.get("api_key_env")
        token = os.environ.get(env_var) if env_var else None
        if not token:
            raise RuntimeError(
                f"model {self.model_name!r} requires {env_var} in the "
                f"environment (not set). Source it from your .env or export it "
                f"before instantiating ModelClient."
            )
        self._bedrock_token = token

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
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Bedrock Converse HTTP {e.code}: {detail}") from e

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
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Bedrock Responses HTTP {e.code}: {detail}") from e

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

    def call(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.1,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        kind: str = "unknown",
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
        rate_limit_retries = 5
        t_start = _time.time()

        try:
            while True:
                try:
                    if self.backend == "openai_compat":
                        response = self._invoke_with_wall_timeout(
                            self._call_openai_compat, timeout,
                            system, messages, mt, temperature, timeout,
                            response_format=response_format,
                            reasoning_effort=reasoning_effort,
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
                    elif self.backend == "bedrock_converse":
                        response = self._invoke_with_wall_timeout(
                            self._call_bedrock_converse, timeout,
                            system, messages, mt, temperature, timeout,
                        )
                    elif self.backend == "bedrock_responses":
                        response = self._invoke_with_wall_timeout(
                            self._call_bedrock_responses, timeout,
                            system, messages, mt, temperature, timeout,
                            reasoning_effort=reasoning_effort,
                        )
                    else:
                        raise ValueError(f"Unknown backend: {self.backend}")
                    self._get_call_log().append({
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
                        # Uniform CoT capture (status + tokens + provenance, and
                        # committed support/objection stamped later by the
                        # structured scorer). Same dict object the scorer mutates.
                        "reasoning_trace": response.reasoning_trace,
                    })
                    return response
                except Exception as e:
                    msg = str(e).lower()
                    # 429 / rate-limit: respect the server's requested delay.
                    # This is the ONLY in-client retry.
                    if ("429" in msg or "rate limit" in msg
                            or "resource_exhausted" in msg) and rate_limit_retries > 0:
                        delay = _parse_retry_delay(str(e))
                        # tiny safety pad so the next request lands clean
                        _time.sleep(delay + 1)
                        rate_limit_retries -= 1
                        continue
                    raise
        except Exception as e:
            self._get_call_log().append({
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
            })
            raise

    def _call_openai_compat(
        self, system: str, messages: list[dict], mt: int, temp: float, timeout: int,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> ModelResponse:
        full_messages = [{"role": "system", "content": system}] + messages
        kwargs = dict(
            model=self.config["model_id"],
            messages=full_messages,
            max_tokens=mt,
            temperature=temp,
            timeout=timeout,
        )
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
        if effort:
            extra_body["reasoning_effort"] = effort
            # When the caller asks for "none", that's a request to disable
            # thinking entirely. Translate to the chat_template_kwargs
            # mechanism Ollama honors — but only on permissive backends.
            if effort == "none" and not strict:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
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
        reasoning = getattr(msg, "reasoning_content", None) or ""

        # For models where reasoning is IN content, raw_text = content
        # For models with separate reasoning, raw_text = reasoning + content
        if self.config.get("reasoning_in_content"):
            raw_text = content
        else:
            raw_text = (reasoning + "\n" + content) if reasoning else content

        finish = response.choices[0].finish_reason or "stop"
        rtok = _reasoning_tokens(response.usage)
        return ModelResponse(
            content=content,
            reasoning=reasoning,
            tokens=response.usage.completion_tokens,
            raw_text=raw_text,
            finish_reason=finish,
            prompt_tokens=getattr(response.usage, "prompt_tokens", -1),
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


# Verdict parsing and score mapping live in scorers._prompts — this module
# is the model client, not an output parser. See _prompts.extract_verdict.
