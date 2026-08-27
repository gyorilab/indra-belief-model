"""Smoke-test the scoring path end to end: free by default, live on demand.

WHY THIS EXISTS. `scripts/smoke_bedrock_lanes.py` walks the paid provider lanes.
Nothing walked the path those lanes exist to serve: an INDRA Statement and one
of its Evidence objects going in, a calibrated belief coming out. The suite
covers the pieces; this covers the seam between them, which is where a crash on
a documented default invocation once survived a green suite.

TWO TIERS, and the default costs nothing and opens no socket:

  hermetic (default)
      Every registry entry either constructs or refuses for a named missing
      credential; every backend is reachable; the in-process vLLM engine batches
      through ModelClient against a stub; the reply parser handles the shapes
      models actually emit; verdicts aggregate to a belief in [0, 1]; and the
      lifted per-evidence aggregation still reduces EXACTLY to the published
      model when fed verdict-derived weights.

  --live BASE_URL MODEL_ID
      Score a real Statement/Evidence pair through an OpenAI-compatible server
      you already have running, then aggregate two evidences into a belief.
      Free against a local server; this issues no paid request and never
      selects a provider lane.

Neither tier writes anything under data/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _row(name: str, ok: bool, detail: str) -> str:
    return f"  {'ok  ' if ok else 'FAIL'} {name:46} {detail}"


def hermetic() -> int:
    from indra_belief.model_client import LOCAL_MODELS, ModelClient

    results: list[tuple[str, bool, str]] = []

    built, refused, broken = [], [], []
    for name in LOCAL_MODELS:
        try:
            ModelClient(name)
            built.append(name)
        except (RuntimeError, ModuleNotFoundError, ImportError) as exc:
            # A refusal naming a missing credential or optional dependency is
            # the contract working: the alternative is a client that constructs
            # and then fails mid-run against a provider.
            if "API_KEY" in str(exc) or "requires" in str(exc) or isinstance(
                exc, (ModuleNotFoundError, ImportError)
            ):
                refused.append(name)
            else:
                broken.append(f"{name}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
    results.append((
        "every registry entry constructs or refuses namedly",
        not broken,
        "; ".join(broken[:2]) if broken
        else f"{len(built)} built, {len(refused)} refused, {len(LOCAL_MODELS)} total",
    ))

    declared = {(c.get("backend") or "openai_compat") for c in LOCAL_MODELS.values()}
    missing = {
        "openai_compat", "transformers_local", "bedrock_converse",
        "bedrock_responses", "bedrock_responses_raw", "vllm_offline",
    } - declared
    # `anthropic` is selected by NAME PREFIX rather than by a registry entry, so
    # a registry scan cannot see it. An "Unknown model" here means that arm died.
    try:
        ModelClient("claude-sonnet-5")
        prefix = "constructed"
    except Exception as exc:  # noqa: BLE001
        prefix = "dead" if "Unknown model" in str(exc) else f"refused({type(exc).__name__})"
    results.append((
        "every backend reachable (registry + claude-* prefix)",
        not missing and prefix != "dead",
        f"missing={sorted(missing)}" if missing else f"{len(declared)} declared, anthropic={prefix}",
    ))

    results.append(_vllm_offline_batches())
    results.append(_parser_shapes())
    results.append(_belief_math())

    print("HERMETIC — no socket opened, no credential required\n")
    for name, ok, detail in results:
        print(_row(name, ok, detail))
    bad = [n for n, ok, _ in results if not ok]
    print()
    if bad:
        print(f"HERMETIC FAILED — {len(bad)} of {len(results)}: {', '.join(bad)}")
        return 1
    print(f"HERMETIC OK — {len(results)} checks. Re-run with --live to score for real.")
    return 0


def _vllm_offline_batches() -> tuple[str, bool, str]:
    """The engine exists to batch; ModelClient.call is one-at-a-time.

    Routing it through the class-level wall-timeout pool would cap batch fill at
    that pool's width regardless of the caller's worker count, which is the
    failure this asserts against: more concurrent callers than pool slots must
    still land in ONE engine call.
    """
    seen: list[int] = []

    class _Params:
        def __init__(self, **kw):
            self.kw = kw

    class _Out:
        def __init__(self, text: str):
            self.outputs = [types.SimpleNamespace(
                text=text, logprobs=None, token_ids=[1],
                finish_reason="stop", cumulative_logprob=0.0)]
            self.prompt_token_ids = [1]

    class _LLM:
        def __init__(self, **kw):
            pass

        def chat(self, conversations, sampling_params, use_tqdm=False, **kw):
            seen.append(len(conversations))
            return [_Out('{"verdict": "correct"}') for _ in conversations]

    stub = types.ModuleType("vllm")
    stub.LLM, stub.SamplingParams = _LLM, _Params
    sys.modules["vllm"] = stub
    params = types.ModuleType("vllm.sampling_params")
    params.SamplingParams = _Params
    sys.modules["vllm.sampling_params"] = params

    from indra_belief.model_client import ModelClient

    client = ModelClient("vllm-offline-gemma-4-26b")
    if getattr(client, "_vllm_offline_client", None) is not None:
        return ("offline vLLM lazy + batches past the pool width", False,
                "engine built eagerly at construction")
    n = 12
    with cf.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(
            lambda i=i: client.call(system="s", max_tokens=8,
                                    messages=[{"role": "user", "content": f"m{i}"}])
        ) for i in range(n)]
        got = [f.result() for f in futures]
    ok = len(got) == n and max(seen) > 8
    return ("offline vLLM lazy + batches past the pool width", ok,
            f"{n} concurrent -> engine batches {seen}")


def _parser_shapes() -> tuple[str, bool, str]:
    from indra_belief.verdict import DEFAULT_CONFIDENCE, parse_response

    # The expected values are LITERAL. Comparing against the imported
    # constant would move both sides of the assertion together, so a
    # changed default would pass — the same tautology as asserting F1
    # only where precision equals recall.

    class _Resp:
        def __init__(self, text: str):
            self.content = self.raw_text = self.text = text

    cases = [
        ('{"verdict": "correct", "confidence": "high"}', ("correct", "high")),
        ('prose\n{"verdict":"incorrect","confidence":"low"}\nmore', ("incorrect", "low")),
        ('```json\n{"verdict": "correct"}\n```', ("correct", "medium")),
        ('<think>weighing</think>\n{"verdict": "incorrect"}', ("incorrect", "medium")),
        ("verdict correct", ("correct", "medium")),
        ("no json here", None),
    ]
    if DEFAULT_CONFIDENCE != "medium":
        return ("parser: json / fenced / CoT-prefixed / bare / none", False,
                f"DEFAULT_CONFIDENCE moved to {DEFAULT_CONFIDENCE!r}; the literals below assume 'medium'")
    bad = []
    for text, want in cases:
        read = parse_response(_Resp(text))
        got = None if read is None else (read.label, read.confidence)
        if got != want:
            bad.append(f"{text[:24]!r} -> {got}, wanted {want}")
    return ("parser: json / fenced / CoT-prefixed / bare / none", not bad,
            "; ".join(bad[:2]) if bad else f"{len(cases)} reply shapes")


def _belief_math() -> tuple[str, bool, str]:
    from indra_belief.evidence_weights import (
        belief_from_weights, source_logodds_for, verdict_weight,
    )
    from indra_belief.noise_model import compute_gated_belief
    from indra_belief.statement_belief import statement_belief

    rows = [
        {"verdict": "correct", "source_api": "reach", "matches_hash": "1", "source_hash": "a"},
        {"verdict": "incorrect", "source_api": "sparser", "matches_hash": "1", "source_hash": "b"},
    ]
    record = statement_belief(rows)
    if record is None or record.belief is None:
        return ("verdicts -> belief; lifted reduces to the published model",
                False, "statement_belief produced no belief for scored rows")
    if not 0.0 <= float(record.belief) <= 1.0:
        return ("verdicts -> belief; lifted reduces to the published model",
                False, f"belief outside [0,1]: {record.belief}")

    # A soft belief without a fitted profile would be an invented number.
    try:
        compute_gated_belief(rows, soft_weights=True)
        return ("verdicts -> belief; lifted reduces to the published model",
                False, "soft belief computed with no calibration — the guard is gone")
    except ValueError:
        pass

    lrs = {"log_lr_confirm": 1.2, "log_lr_reject": -1.4}
    frozen = compute_gated_belief(rows, soft_weights=True, **lrs)
    lifted = belief_from_weights(
        [{"source_api": r["source_api"],
          "weight_of_evidence": verdict_weight(
              r["verdict"], source_logodds_for(r["source_api"], {}), **lrs)}
         for r in rows],
        None, prior_logodds=0.0,
    )
    a = float(getattr(frozen, "belief", frozen))
    b = float(getattr(lifted, "belief", lifted))
    ok = abs(a - b) <= 1e-12
    return ("verdicts -> belief; lifted reduces to the published model", ok,
            f"belief={float(record.belief):.4f}; reduction {'exact' if ok else f'BROKEN {a} vs {b}'} at {a:.6f}")


def live(base_url: str, model_id: str) -> int:
    """Score a real pair through a server the operator already has running."""
    import time

    from indra.statements import Agent, Evidence, Phosphorylation

    from indra_belief.model_client import LOCAL_MODELS, ModelClient

    entry = "local-gemma-4-26b"
    LOCAL_MODELS[entry] = dict(LOCAL_MODELS[entry], base_url=base_url,
                               model_id=model_id, timeout=300)
    from indra_belief.scorers.monolithic import score_statement
    from indra_belief.statement_belief import statement_belief

    statement = Phosphorylation(
        Agent("MAP2K1", db_refs={"HGNC": "6840"}),
        Agent("MAPK1", db_refs={"HGNC": "6871"}),
    )
    statement.evidence = [
        Evidence(source_api="reach", text="MEK1 directly phosphorylates ERK2 in vitro."),
        Evidence(source_api="sparser", text="The weather in Boston was mild that week."),
    ]
    print(f"LIVE — {model_id} at {base_url}\n")
    started = time.time()
    rows = score_statement(statement, ModelClient(entry))
    elapsed = time.time() - started
    for i, (row, ev) in enumerate(zip(rows, statement.evidence)):
        row["source_api"] = ev.source_api
        print(_row(f"evidence {i} ({ev.source_api})", row.get("verdict") is not None,
                   f"verdict={row.get('verdict')!r} confidence={row.get('confidence')!r} "
                   f"tier={row.get('tier')}"))
    record = statement_belief(rows)
    in_range = record.belief is None or 0.0 <= float(record.belief) <= 1.0
    print(_row("statement belief", in_range,
               f"belief={record.belief!r} verdict={record.verdict_statement!r} "
               f"n={record.n_evidence}"))
    print(f"\n{len(rows)} evidence scored in {elapsed:.1f}s")
    return 0 if rows and in_range else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--live", nargs=2, metavar=("BASE_URL", "MODEL_ID"),
        help="score a real pair through an OpenAI-compatible server you are "
             "already running, e.g. --live http://localhost:8085/v1 my-model")
    args = parser.parse_args(argv)
    if args.live:
        return live(*args.live)
    return hermetic()


if __name__ == "__main__":
    sys.exit(main())
