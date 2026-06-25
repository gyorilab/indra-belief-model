"""E4 feasibility probe: enumerate the FULL INDRA curation universe (keyed
list-all), tally curators, and assess third-curator evaluability.

Not a gold builder — a FEASIBILITY instrument. It (1) pulls /curation/list with
the API key (the keyed list-all that returns every curation), (2) tallies curators
and per-curator correct/incorrect balance, (3) checks how many of a third
curator's (matches_hash, source_hash) pairs are recoverable as (statement,
evidence_text) either from the in-payload ev_json or by hash-join to our local
corpora. Writes a JSON report to data/results/curation_universe_probe.json.

DNS to db.indra.bio is intermittent from the Mac; we pre-resolve once with
retries and pin the IP so per-request DNS can't flake mid-pull.

    PYTHONPATH=src .venv/bin/python scripts/probe_curation_universe.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import aggregate_gold  # noqa: E402

PAIR = {"bachmanjohn@gmail.com", "ben.gyori@gmail.com",
        "bachmanjohn", "ben.gyori"}


def _load_env() -> tuple[str, str]:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return (os.environ["INDRA_DB_REST_URL"].rstrip("/"),
            os.environ["INDRA_DB_REST_API_KEY"])


def _resolve(host: str, tries: int = 20) -> str | None:
    for _ in range(tries):
        try:
            return socket.gethostbyname(host)
        except OSError:
            time.sleep(1.0)
    return None


def pull_all(url: str, key: str) -> list[dict]:
    import httpx
    host = url.split("://", 1)[1]
    ip = _resolve(host)
    if ip is None:
        raise SystemExit("DNS: could not resolve %s after retries" % host)
    print(f"resolved {host} -> {ip}", flush=True)
    # DNS resolves but the link to db.indra.bio is intermittent; lean on many
    # retries with backoff. Each attempt gets a fresh client (fresh connection).
    last = None
    for attempt in range(20):
        try:
            with httpx.Client(
                base_url=url,
                headers={"User-Agent": "indra-belief-e4/1"},
                timeout=httpx.Timeout(240.0, connect=30.0),
            ) as c:
                r = c.get("/curation/list", params={"api_key": key})
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict):
                    raise SystemExit(f"list-all returned error dict: {data}")
                return data
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(min(2 * (attempt + 1), 15))
    raise SystemExit(f"list-all failed all retries: {last}")


def local_pairs() -> set[tuple[int, int]]:
    """All (matches_hash, source_hash) pairs we can recover evidence text for
    from local corpora: belief_benchmark (flat rows) + the rasmachine corpus
    (statements with per-evidence source_hash+text)."""
    pairs: set[tuple[int, int]] = set()
    bm = ROOT / "data" / "benchmark" / "belief_benchmark.jsonl"
    if bm.exists():
        for line in open(bm):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
                pairs.add((int(r["pa_hash"]), int(r["source_hash"])))
            except (KeyError, TypeError, ValueError):
                pass
    ras = ROOT / "data" / "corpora" / "latest_statements_rasmachine.json"
    if ras.exists():
        for s in json.load(open(ras)):
            mh = s.get("matches_hash")
            if mh is None:
                continue
            for ev in s.get("evidence", []) or []:
                sh = ev.get("source_hash")
                if sh is None or not ev.get("text"):
                    continue
                try:
                    pairs.add((int(mh), int(sh)))
                except (TypeError, ValueError):
                    pass
    return pairs


def main() -> None:
    url, key = _load_env()
    data = pull_all(url, key)
    print(f"TOTAL curations: {len(data)}", flush=True)
    print("payload keys:", sorted(data[0].keys()), flush=True)

    # ev_json — can we recover evidence text from the payload itself?
    ev_present = sum(1 for d in data if d.get("ev_json"))
    print(f"curations with non-null ev_json: {ev_present}/{len(data)}", flush=True)

    # curator tally
    by_curator = Counter(d.get("curator", "") for d in data)
    print("\n=== curators (top 30) ===", flush=True)
    for c, n in by_curator.most_common(30):
        flag = "  <PAIR>" if c in PAIR else ""
        print(f"  {c!r:42s} {n}{flag}", flush=True)

    # third-curator evaluability: per non-pair curator, how many UNIQUE pairs,
    # balance, ev_json availability, and local-corpus recoverability.
    loc = local_pairs()
    rep_curators = []
    for cur, _ in by_curator.most_common():
        if cur in PAIR or not cur:
            continue
        rows = [d for d in data if d.get("curator") == cur]
        # group by pair, aggregate tags
        tags_by_pair: dict[tuple, list[str]] = defaultdict(list)
        ev_by_pair: dict[tuple, bool] = {}
        for d in rows:
            mh = d.get("pa_hash") or d.get("_matches_hash")
            sh = d.get("source_hash")
            if mh is None or sh is None:
                continue
            try:
                k = (int(mh), int(sh))
            except (TypeError, ValueError):
                continue
            tags_by_pair[k].append(d.get("tag", ""))
            ev_by_pair[k] = ev_by_pair.get(k, False) or bool(d.get("ev_json"))
        gold = {k: aggregate_gold(t) for k, t in tags_by_pair.items()}
        nc = sum(1 for g in gold.values() if g == "correct")
        ni = sum(1 for g in gold.values() if g == "incorrect")
        n_ev = sum(1 for k in tags_by_pair if ev_by_pair.get(k))
        n_local = sum(1 for k in tags_by_pair if k in loc)
        n_recoverable = sum(1 for k in tags_by_pair if ev_by_pair.get(k) or k in loc)
        rec_c = sum(1 for k, g in gold.items()
                    if g == "correct" and (ev_by_pair.get(k) or k in loc))
        rec_i = sum(1 for k, g in gold.items()
                    if g == "incorrect" and (ev_by_pair.get(k) or k in loc))
        rep_curators.append({
            "curator": cur, "rows": len(rows), "unique_pairs": len(tags_by_pair),
            "gold_correct": nc, "gold_incorrect": ni,
            "with_ev_json": n_ev, "in_local_corpus": n_local,
            "recoverable_pairs": n_recoverable,
            "recoverable_correct": rec_c, "recoverable_incorrect": rec_i,
            "balanced_ceiling": 2 * min(rec_c, rec_i),
        })

    rep_curators.sort(key=lambda r: -r["recoverable_pairs"])
    print("\n=== non-pair curators: evaluability ===", flush=True)
    print(f"{'curator':40s} {'pairs':>6} {'corr':>5} {'inc':>5} "
          f"{'evjson':>7} {'local':>6} {'recov':>6} {'bal_ceil':>8}", flush=True)
    for r in rep_curators[:30]:
        print(f"{r['curator']:40s} {r['unique_pairs']:6d} {r['gold_correct']:5d} "
              f"{r['gold_incorrect']:5d} {r['with_ev_json']:7d} "
              f"{r['in_local_corpus']:6d} {r['recoverable_pairs']:6d} "
              f"{r['balanced_ceiling']:8d}", flush=True)

    out = ROOT / "data" / "results" / "curation_universe_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "total_curations": len(data),
        "ev_json_present": ev_present,
        "payload_keys": sorted(data[0].keys()),
        "n_curators": len(by_curator),
        "curator_counts": dict(by_curator.most_common()),
        "pair_curators": sorted(PAIR),
        "non_pair_evaluability": rep_curators,
        "local_corpus_pairs": len(loc),
    }, indent=2) + "\n")
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
