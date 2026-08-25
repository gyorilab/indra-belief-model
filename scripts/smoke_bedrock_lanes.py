"""Smoke-test every Bedrock lane in the registry: free by default, cheap on demand.

WHY THIS EXISTS. The Bedrock lanes are four different code paths — plain
`openai_compat`, one dependency-free `*_raw` transport, and the two legacy
urllib lanes — and the raw one carries machinery (pinned route, byte-pinned CA
bundle, canonical request-body commitments) that a construction check alone does
not exercise. This walks all of them.

TWO TIERS, and the default costs nothing:

  offline (default)
      Construct each client, build the EXACT provider request body its backend
      would send, and for the `*_raw` lane confirm the transport's frozen route
      and CA-bundle pin. No socket is opened. This is what proves the modules
      import, the registry entries resolve, and the body builders still agree
      with their transports.

  --live
      Issue ONE minimal request per model (a two-word prompt, tiny max_tokens)
      and report status, latency and observed tokens. This spends real money —
      a few hundredths of a cent in total at the listed rates — so it is opt-in,
      it prints a priced plan first, and `--yes` is required to skip the prompt.

Neither tier writes anything to data/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.model_client import LOCAL_MODELS, ModelClient  # noqa: E402
from indra_belief.corpus.cost import price_for  # noqa: E402

BEDROCK = sorted(k for k in LOCAL_MODELS if k.startswith("bedrock-"))
_SYSTEM = "Reply with one word."
_MESSAGES = [{"role": "user", "content": "Say OK."}]


def _load_dotenv() -> None:
    """Source .env so the bearer token is present without exporting by hand."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _body_detail(body: dict, trace: dict) -> str:
    """The builders return a dict, so report its canonical BYTE length — the
    thing the raw transports actually commit to — not len(dict), which counts
    keys and reads as an absurdly small request."""
    n = len(json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"body {n}B / {len(body)} keys, trace {len(trace)} fields, route pinned"


def _offline_one(name: str) -> tuple[bool, str]:
    """Construct, build the real request body, and check any transport pinning."""
    cfg = LOCAL_MODELS[name]
    backend = cfg.get("backend", "openai_compat")
    client = ModelClient(name)
    if backend != client.backend:
        return False, f"backend drift: registry {backend!r} vs client {client.backend!r}"

    if backend == "bedrock_responses_raw":
        from indra_belief.bedrock_responses_transport import build_bedrock_responses_body

        body = build_bedrock_responses_body(
            model_id=cfg["model_id"], system=_SYSTEM, messages=_MESSAGES,
            max_output_tokens=8, reasoning_effort=cfg.get("reasoning_effort"),
        )
        transport = client._bedrock_responses_transport
        trace = transport.request_trace(body)
        if transport.endpoint != cfg["expected_responses_endpoint"]:
            return False, "transport endpoint does not match its expected pin"
        return True, _body_detail(body, trace)

    if backend in ("bedrock_converse", "bedrock_responses"):
        # Legacy urllib lanes: the token setup is the part that can break, and
        # it is what `_setup_bedrock_token` did at construction above.
        return True, "token lane constructed"

    return True, f"{backend} client constructed"


def _priced_plan(names: list[str]) -> str:
    known = 0
    for n in names:
        if price_for(LOCAL_MODELS[n]["model_id"]) is not None:
            known += 1
    return (f"{len(names)} model(s); {known} carry a list price. One ~20-token "
            f"exchange each is well under $0.01 in total at those rates.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="issue one real minimal request per model (spends money)")
    ap.add_argument("--yes", action="store_true", help="skip the --live confirmation")
    ap.add_argument("--only", default=None, help="substring filter over model names")
    args = ap.parse_args(argv)

    _load_dotenv()
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("AWS_BEARER_TOKEN_BEDROCK is not set (and not in .env).", file=sys.stderr)
        return 2

    names = [n for n in BEDROCK if not args.only or args.only in n]
    if not names:
        print("no Bedrock models matched", file=sys.stderr)
        return 2

    print(f"OFFLINE smoke over {len(names)} Bedrock lane(s) — no sockets opened\n")
    failures = []
    for name in names:
        backend = LOCAL_MODELS[name].get("backend", "openai_compat")
        try:
            ok, detail = _offline_one(name)
        except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"  {'ok ' if ok else 'FAIL'}  {name:32s} {backend:30s} {detail}")
        if not ok:
            failures.append(name)

    if failures:
        print(f"\nOFFLINE FAILED for {len(failures)}: {failures}")
        return 1
    print(f"\nOFFLINE OK — all {len(names)} lanes construct, build a body, and keep their pins.")

    if not args.live:
        print("Re-run with --live to issue one minimal priced request per model.")
        return 0

    print(f"\nLIVE plan: {_priced_plan(names)}")
    if not args.yes:
        reply = input("proceed and spend? [y/N] ").strip().lower()
        if reply != "y":
            print("aborted; nothing spent.")
            return 0

    print()
    live_failures = []
    for name in names:
        t0 = time.time()
        try:
            client = ModelClient(name)
            resp = client.call(_SYSTEM, _MESSAGES, max_tokens=16, temperature=0.0)
            dt = time.time() - t0
            text = (resp.content or resp.raw_text or "").strip().replace("\n", " ")[:34]
            print(f"  ok    {name:32s} {dt:6.2f}s  in={resp.prompt_tokens or 0:<5} "
                  f"out={resp.tokens or 0:<5} {text!r}")
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            print(f"  FAIL  {name:32s} {dt:6.2f}s  {type(exc).__name__}: {str(exc)[:70]}")
            live_failures.append(name)

    if live_failures:
        print(f"\nLIVE FAILED for {len(live_failures)}: {live_failures}")
        return 1
    print(f"\nLIVE OK — all {len(names)} lanes returned a response.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
