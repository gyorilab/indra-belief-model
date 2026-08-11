#!/usr/bin/env python3
"""Reproducible capability probe: does this reader return usable token logprobs?

Our only prior evidence that a route does or does not supply logprobs is a table
committed into research/serving_architecture.md — a point-in-time observation
with no script behind it. This is that script. It answers, for any model in
`LOCAL_MODELS`, the question the belief work actually depends on:

    can we compute p_raw for THIS reader, today, and is it non-degenerate?

It is deliberately not a unit test. It performs live inference and its answer is
about a provider, not about our code.

Usage:
    python scripts/probe_logprobs.py --model local-gemma-4-26b
    python scripts/probe_logprobs.py --model local-gemma-4-26b -k 11 --repeat 3

Exit code is 0 only when status == "ok" AND both labels were observed, so this
can gate a run. A route that ACCEPTS top_logprobs and returns nothing exits
non-zero with status "empty" rather than looking like a clean pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indra_belief.logprobs import from_response  # noqa: E402
from indra_belief.model_client import LOCAL_MODELS, ModelClient  # noqa: E402

# Shaped like a real scoring call: a claim, an evidence sentence, and the
# two-key JSON contract of the verdict-only variant. Synthetic entities, so
# this can never contaminate a gold set.
SYSTEM = (
    "You judge whether an evidence sentence supports a claim about molecular "
    "biology.\n"
    'Output JSON ONLY, exactly these two keys:\n'
    '{"verdict": "correct" | "incorrect", "confidence": "high" | "medium" | "low"}\n'
    "Do NOT explain, quote, justify, or emit any other field."
)
CASES = [
    ("PROTX phosphorylates PROTY.",
     "We show that PROTX directly phosphorylates PROTY at Ser42 in vitro."),
    ("PROTX phosphorylates PROTY.",
     "PROTX and PROTY were both elevated in treated cells."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="local-gemma-4-26b",
                    help="key in model_client.LOCAL_MODELS")
    ap.add_argument("-k", "--top-logprobs", type=int, default=11)
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat each case, to expose non-reproducibility")
    # Must clear the model's chain of thought. gemma-4-26b spends ~70 tokens on
    # a `<|channel>thought` block before emitting the JSON even on a trivial
    # pair; at 64 the reply truncates to empty content and the probe reports
    # "no_position", which looks like a capability failure but is a budget one.
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    cfg = LOCAL_MODELS.get(args.model)
    if cfg is None:
        print(f"unknown model {args.model!r}; known: {sorted(LOCAL_MODELS)}")
        return 2
    print(f"model    : {args.model} -> {cfg.get('model_id')}")
    print(f"base_url : {cfg.get('base_url')}")
    print(f"declares : supports_logprobs={cfg.get('supports_logprobs', False)} "
          f"max_top_logprobs={cfg.get('max_top_logprobs')}")
    print()

    client = ModelClient(args.model)
    ok = True
    for i, (claim, evidence) in enumerate(CASES):
        for rep in range(args.repeat):
            user = f'CLAIM: {claim}\nEVIDENCE: "{evidence}"'
            resp = client.call(
                system=SYSTEM,
                messages=[{"role": "user", "content": user}],
                max_tokens=args.max_tokens,
                temperature=0.0,
                kind="probe_logprobs",
                top_logprobs=args.top_logprobs,
            )
            info = from_response(resp)
            tag = f"case{i}" + (f".r{rep}" if args.repeat > 1 else "")
            print(f"[{tag}] content={resp.content.strip()!r}")
            print(f"[{tag}] logprobs_status={resp.logprobs_status} "
                  f"positions={len(resp.logprobs) if resp.logprobs is not None else 0}")
            if info["status"] == "ok":
                print(f"[{tag}] p_raw={info['p_raw']:.6f} "
                      f"label_mass={info['label_mass']:.4f} "
                      f"both_observed={info['both_observed']} "
                      f"pos={info['position']}")
                if not info["both_observed"]:
                    print(f"[{tag}] NOTE: losing label outside top-{args.top_logprobs}; "
                          f"p_raw is a lower bound, not a measurement")
                    ok = False
            else:
                print(f"[{tag}] NO p_raw: {info['status']}")
                ok = False
            print()

    print(json.dumps({"model": args.model, "usable": ok}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
