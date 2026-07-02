"""Re-fetch evidence text for the FULL multi-curator pool by matches_hash.

The plan: mock7ee (self, anchor) + every other curator EXCEPT the ben/bachman
benchmark pair. Their curations carry no embedded evidence text (only ~8% have
ev_json) and don't join any local corpus — so we recover the text from live INDRA
via POST /statements/from_hashes (public, batched, evidence inline).

Efficiency: one POST carries many matches_hashes; we issue batches concurrently
with bounded async (httpx), high ev_limit so the curated evidence isn't truncated
out of high-evidence statements, retries with backoff. The curated source_hash is
matched against the live evidence source_hash (recent curations join cleanly; older
ones may be version-skewed -> reported as misses, never silently dropped).

Output: data/corpora/recovered_curation_statements.json — INDRA statement JSON for
every fetched statement (drop-in for the runner + the gold builder), plus a
recovery report. A second pass assembles the balanced multi-curator gold.

    PYTHONPATH=src .venv/bin/python scripts/recover_curation_evidence.py \
        [--batch 20] [--concurrency 8] [--ev-limit 10000]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import aggregate_gold  # noqa: E402

UNIVERSE = ROOT / "data" / "results" / "curation_universe_all.jsonl"
OUT_CORPUS = ROOT / "data" / "corpora" / "recovered_curation_statements.json"
PAIRS = {"ben.gyori@gmail.com", "bachmanjohn@gmail.com", "ben.gyori", "bachmanjohn"}


def _load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return os.environ["INDRA_DB_REST_URL"].rstrip("/")


def target_pool() -> tuple[dict, list[int]]:
    """Return (curations_by_pair, unique_matches_hashes) for the full multi-curator
    pool (everyone except the ben/bachman benchmark pair)."""
    by_pair: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for line in UNIVERSE.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        cur = (d.get("curator") or "").strip()
        if cur in PAIRS or not cur:
            continue
        try:
            k = (int(d["pa_hash"]), int(d["source_hash"]))
        except (TypeError, ValueError, KeyError):
            continue
        by_pair[k].append(d)
    mhs = sorted({mh for mh, _ in by_pair})
    return dict(by_pair), mhs


async def fetch_batches(base: str, mhs: list[int], batch: int, concurrency: int,
                        ev_limit: int, retries: int = 4) -> dict[int, dict]:
    import httpx

    chunks = [mhs[i:i + batch] for i in range(0, len(mhs), batch)]
    sem = asyncio.Semaphore(concurrency)
    out: dict[int, dict] = {}
    t0 = time.time()
    done = 0

    async def one(client: httpx.AsyncClient, chunk: list[int]) -> None:
        nonlocal done
        async with sem:
            for attempt in range(retries + 1):
                try:
                    r = await client.post("/statements/from_hashes",
                                          json={"hashes": chunk},
                                          params={"ev_limit": ev_limit, "format": "json-js"})
                    if r.status_code == 200:
                        res = r.json().get("results", {})
                        for s in (res.values() if isinstance(res, dict) else res):
                            mh = s.get("matches_hash")
                            if mh is not None:
                                out[int(mh)] = s
                        break
                except (httpx.TransportError, httpx.HTTPError):
                    pass
                if attempt < retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
            done += len(chunk)
            if done % 200 < batch:
                print(f"  {done}/{len(mhs)} hashes · {done/max(1e-9,time.time()-t0):.0f}/s", flush=True)

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(120.0, connect=30.0),
                                 limits=limits, headers={"User-Agent": "indra-belief-recover/1"}) as client:
        await asyncio.gather(*(one(client, c) for c in chunks))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--ev-limit", type=int, default=10000)
    args = ap.parse_args()

    base = _load_env()
    by_pair, mhs = target_pool()
    n_pairs = len(by_pair)
    print(f"full multi-curator pool: {n_pairs} (stmt,evidence) pairs over {len(mhs)} statements")

    fetched = asyncio.run(fetch_batches(base, mhs, args.batch, args.concurrency, args.ev_limit))
    print(f"\nfetched {len(fetched)}/{len(mhs)} statements")

    # index live evidence by (mh, sh) and join to curated targets
    live_text: dict[tuple[int, int], dict] = {}
    for mh, s in fetched.items():
        for ev in s.get("evidence", []) or []:
            sh = ev.get("source_hash")
            if sh is None:
                continue
            try:
                live_text[(mh, int(sh))] = ev
            except (TypeError, ValueError):
                pass

    recovered = {k: by_pair[k] for k in by_pair if k in live_text}
    # save the recovered statements (only those carrying >=1 recovered target pair)
    keep_mhs = {mh for mh, _ in recovered}
    corpus = [fetched[mh] for mh in keep_mhs if mh in fetched]
    OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(corpus, open(OUT_CORPUS, "w"))

    # recovery report
    gold = {k: aggregate_gold([c.get("tag", "") for c in curs]) for k, curs in recovered.items()}
    nc = sum(1 for g in gold.values() if g == "correct")
    ni = sum(1 for g in gold.values() if g == "incorrect")
    spread = Counter()
    for k, curs in recovered.items():
        for c in curs:
            if c.get("curator"):
                spread[c["curator"]] += 1
    print(f"\n=== recovery ===")
    print(f"  target pairs:        {n_pairs}")
    print(f"  recovered (mh,sh):   {len(recovered)}  ({100*len(recovered)/max(n_pairs,1):.0f}%)")
    print(f"  version-skew misses: {n_pairs - len(recovered)}  (statement fetched but source_hash not in live evidence)")
    print(f"  balance:             {nc} correct / {ni} incorrect  (balanced ceiling {2*min(nc,ni)})")
    print(f"  curator spread:      {len(spread)} curators; top {dict(spread.most_common(12))}")
    print(f"\n  wrote {len(corpus)} statements -> {OUT_CORPUS}")
    print("  next: build the balanced multi-curator gold from this corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
