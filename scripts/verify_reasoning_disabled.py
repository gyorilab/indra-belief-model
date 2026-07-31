"""Prove, before spending, that a comparison arm's reasoning is actually off.

Why this exists: the provider's own accounting cannot answer the question.
Across the 2026-07 thinking run, gemma reported
``output_tokens_details.reasoning_tokens = 0`` on all 99,180 traces while
returning real chain-of-thought, and glm-5 omitted the field entirely. So a
run can be billed for hidden deliberation and the ledger still looks clean.

What CAN answer it is the wire body, which the two ``*_raw`` paid-lane
transports build deterministically and the spend guard records per call:

  * Responses lane (gemma): reasoning_effort "none" omits the ``reasoning``
    key entirely (bedrock_responses_transport.py:625-627).
  * Chat lane (glm-5): "none" is truthy, so ``"reasoning_effort":"none"`` IS
    sent (bedrock_chat_transport.py:571-572); mantle engages thinking only at
    "high".

Static mode (default) is free and offline: it builds each model's wire body
through the same function the spend guard uses and asserts the reasoning
shape. ``--live`` additionally spends ~$0.01 total to confirm the provider
does not deliberate, comparing visible-content density against the thinking
run's measured baselines (~3.3 chars per output token when plaintext, ~0.46
when most output tokens went to discarded CoT).

    PYTHONPATH=src python scripts/verify_reasoning_disabled.py
    PYTHONPATH=src python scripts/verify_reasoning_disabled.py --live
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.model_client import LOCAL_MODELS, ModelClient  # noqa: E402

# (model, thinking sibling) for every arm the reasoning-off plan will run.
ARMS = [
    ("bedrock-gemma-4-e2b-noreason", "bedrock-gemma-4-e2b"),
    ("bedrock-gemma-4-26b-noreason", "bedrock-gemma-4-26b"),
    ("bedrock-gemma-4-31b-noreason", "bedrock-gemma-4-31b"),
    ("bedrock-glm-5-noreason", "bedrock-glm-5"),
]

# Chars of visible content per output token. The thinking run measured ~0.46
# for arms whose output tokens were mostly CoT; plaintext answers run ~3.3.
# Anything at or below this is evidence the provider still deliberated.
CHARS_PER_OUTPUT_TOKEN_FLOOR = 1.5

_SYSTEM = "You answer with a single JSON object and nothing else."
_MESSAGES = [{"role": "user", "content": 'Reply exactly {"ok": true}'}]


def _wire_body(model: str) -> dict:
    """Build the exact provider body the spend guard would record."""
    config = LOCAL_MODELS[model]
    backend = config["backend"]
    effort = config.get("reasoning_effort")
    if backend == "bedrock_responses_raw":
        from indra_belief.bedrock_responses_transport import build_bedrock_responses_body

        return build_bedrock_responses_body(
            model_id=config["model_id"], system=_SYSTEM, messages=_MESSAGES,
            max_output_tokens=256, reasoning_effort=effort,
        )
    if backend == "bedrock_chat_completions_raw":
        from indra_belief.bedrock_chat_transport import build_bedrock_chat_body

        return build_bedrock_chat_body(
            model_id=config["model_id"], system=_SYSTEM, messages=_MESSAGES,
            max_tokens=256, temperature=0.1, response_format=None,
            reasoning_effort=effort,
        )
    raise SystemExit(
        f"{model}: backend {backend!r} is not a paid-lane *_raw transport, so no "
        "canonical wire body is recorded and reasoning mode cannot be proven"
    )


def check_static(model: str, sibling: str) -> list[str]:
    failures = []
    config = LOCAL_MODELS[model]
    if config.get("reasoning_effort") != "none":
        failures.append(
            f'registry reasoning_effort is {config.get("reasoning_effort")!r}, expected "none"'
        )
    sib = LOCAL_MODELS[sibling]
    if config["model_id"] != sib["model_id"]:
        failures.append(
            f'model_id {config["model_id"]!r} differs from thinking sibling {sib["model_id"]!r} '
            "— the pair would not isolate reasoning mode"
        )
    if config["backend"] != sib["backend"]:
        failures.append(
            f'backend {config["backend"]!r} differs from sibling {sib["backend"]!r}'
        )
    for field in ("max_tokens", "base_url", "reasoning_in_content"):
        if config.get(field) != sib.get(field):
            failures.append(
                f"{field} {config.get(field)!r} differs from sibling {sib.get(field)!r}"
            )

    body = _wire_body(model)
    sibling_body = _wire_body(sibling)
    if config["backend"] == "bedrock_responses_raw":
        if "reasoning" in body:
            failures.append(f'wire body still carries reasoning={body["reasoning"]!r}')
        if sibling_body.get("reasoning") != {"effort": "high"}:
            failures.append(
                f"sibling wire body should carry reasoning={{'effort': 'high'}}, "
                f"got {sibling_body.get('reasoning')!r} — the comparison is not a contrast"
            )
    else:
        if body.get("reasoning_effort") != "none":
            failures.append(
                f'wire body reasoning_effort={body.get("reasoning_effort")!r}, expected "none"'
            )
        if sibling_body.get("reasoning_effort") != "high":
            failures.append(
                f'sibling wire body reasoning_effort={sibling_body.get("reasoning_effort")!r}, '
                'expected "high" — the comparison is not a contrast'
            )
    print(f"  wire body keys: {sorted(body)}")
    print(f"  sibling  keys:  {sorted(sibling_body)}")
    return failures


def check_live(model: str) -> list[str]:
    failures = []
    client = ModelClient(model)
    response = client.call(
        system=_SYSTEM, messages=_MESSAGES, max_tokens=256,
        temperature=0.1, kind="reasoning_probe",
    )
    content = response.content or ""
    raw = response.raw_text or ""
    out_tokens = int(response.tokens or 0)
    trace = getattr(response, "reasoning_trace", None)
    free_cot = len(raw) - len(content)
    print(f"  content={len(content)}ch raw={len(raw)}ch out_tokens={out_tokens} "
          f"extra_reasoning_chars={free_cot}")
    if isinstance(trace, dict):
        print(f"  trace status={trace.get('status')!r} "
              f"reasoning_tokens={trace.get('reasoning_tokens')!r}")
    # NOTE this guard is live only on the Responses lane (gemma). For glm-5 our
    # chat transport reads `message.reasoning_content` and mantle returns CoT
    # under `message.reasoning`, so raw_text never contains it and free_cot is
    # structurally always 0 — see the runbook's closing defect note. The density
    # check below is the ONLY live chain-of-thought guard for that arm: hidden
    # CoT still shows up as output tokens with nothing visible to account for.
    if free_cot > 0:
        failures.append(
            f"response carried {free_cot} chars beyond the answer — the model still deliberated"
        )
    if out_tokens > 0:
        density = len(content) / out_tokens
        print(f"  chars per output token: {density:.2f}")
        if density < CHARS_PER_OUTPUT_TOKEN_FLOOR:
            failures.append(
                f"visible density {density:.2f} < {CHARS_PER_OUTPUT_TOKEN_FLOOR} — output "
                "tokens were spent on something not returned (hidden CoT)"
            )
    return failures


def audit_run(plan_path: str) -> int:
    """Audit every call a RUN actually made, per arm, from its persisted rows.

    The static and live modes check configuration and one probe call. This mode
    checks the artifact of record: for each arm, every entry in every row's
    call_log, asserting the recorded provider wire body carries no reasoning and
    that no deliberation sub-call (relation_nature) fired at all. It is the only
    check that covers ALL models across the WHOLE run rather than a sample.
    """
    plan = json.loads(Path(plan_path).read_text())
    stages = {s["id"]: s for s in plan["stages"]}
    failures = 0
    print(f"auditing persisted calls in {plan_path}\n")
    for action in plan["actions"]:
        path = Path(action["output"]["path"])
        model = stages[action["stage"]]["model"]
        if not path.exists() or path.stat().st_size == 0:
            print(f"  {action['id']:24s} {model:30s} NO ROWS YET")
            continue
        kinds: dict[str, int] = {}
        with_reasoning = 0
        rows = calls = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rows += 1
                for call in row.get("call_log") or []:
                    calls += 1
                    kind = str(call.get("kind"))
                    kinds[kind] = kinds.get(kind, 0) + 1
                    body = call.get("provider_request_body") or {}
                    if "reasoning" in body or body.get("reasoning_effort") not in (None, "none"):
                        with_reasoning += 1
        deliberation = sorted(k for k in kinds if k == "relation_nature")
        ok = with_reasoning == 0 and not deliberation
        failures += 0 if ok else 1
        print(f"  {action['id']:24s} {model:30s} {'PASS' if ok else 'FAIL'}")
        print(f"      rows={rows} calls={calls} kinds={kinds}")
        print(f"      calls carrying reasoning: {with_reasoning}"
              f"{'' if not deliberation else '  deliberation sub-calls: ' + ','.join(deliberation)}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="also make one real billed call per arm (~$0.01 total)")
    parser.add_argument("--run", metavar="PLAN",
                        help="audit every persisted call of a run instead of the registry")
    args = parser.parse_args()

    if args.run:
        return audit_run(args.run)

    verdicts = {}
    for model, sibling in ARMS:
        print(f"\n{model}  (vs {sibling})")
        failures = check_static(model, sibling)
        if args.live and not failures:
            try:
                failures += check_live(model)
            except Exception as exc:  # noqa: BLE001 — a probe failure is a result
                failures.append(f"live call raised {type(exc).__name__}: {exc}")
        verdicts[model] = failures
        for failure in failures:
            print(f"  FAIL: {failure}")
        if not failures:
            print("  PASS")

    print("\n" + "=" * 70)
    failed = [model for model, failures in verdicts.items() if failures]
    print(json.dumps({
        "mode": "live" if args.live else "static",
        "arms": len(ARMS),
        "passed": sorted(set(verdicts) - set(failed)),
        "failed": sorted(failed),
    }, indent=2))
    if failed:
        print("\nREASONING-OFF NOT PROVEN — do not launch the paid run.")
        return 1
    print("\nreasoning-off verified" + ("" if args.live else " (static; add --live before launching)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
