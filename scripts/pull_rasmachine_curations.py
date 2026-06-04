"""Pull INDRA curations for the rasmachine statements — thin CLI over the library.

The domain + transport (async bounded-concurrency GETs to
db.indra.bio/curation/list/<matches_hash>, retries, pooling) live in
indra_belief.curation.fetch_curations. This script is the CLI shell: enumerate
the corpus's statement hashes, resume from a sidecar, stream progress, write the
JSONL. Output rows are joinable to per_evidence.jsonl on (matches_hash, source_hash).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/pull_rasmachine_curations.py            # full
  ... --limit 200 --concurrency 16                                                # test
  ... --smoke 60                                                                  # sample only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import INDRA_DB_REST_URL, fetch_curations  # noqa: E402

DEFAULT_CORPUS = ROOT / "data" / "corpora" / "latest_statements_rasmachine.json"
DEFAULT_OUT = ROOT / "data" / "benchmark" / "rasmachine_curations.jsonl"


def _unique_hashes(corpus_path: Path) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for s in json.load(open(corpus_path)):
        mh = s.get("matches_hash")
        if mh is None:
            continue
        mh = int(mh)
        if mh not in seen:
            seen.add(mh)
            out.append(mh)
    return out


def _done_hashes(sidecar: Path) -> set[int]:
    done: set[int] = set()
    if sidecar.exists():
        for line in open(sidecar):
            line = line.strip()
            if line:
                try:
                    done.add(int(line))
                except ValueError:
                    pass
    return done


async def _run(args) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out_path.with_suffix(".attempted")

    hashes = _unique_hashes(Path(args.corpus))
    print(f"unique matches_hash in corpus: {len(hashes)}")
    if args.smoke:
        step = max(1, len(hashes) // args.smoke)
        hashes = hashes[::step][: args.smoke]
        print(f"smoke mode: sampling {len(hashes)} spread across corpus")
    elif args.limit:
        hashes = hashes[: args.limit]

    if not args.no_resume:
        done = _done_hashes(sidecar)
        if done:
            before = len(hashes)
            hashes = [h for h in hashes if h not in done]
            print(f"resume: {len(done)} already attempted, {len(hashes)} remaining (of {before})")
    if not hashes:
        print("nothing to do.")
        return 0

    t0 = time.time()
    n_done = 0

    def progress(_mh: int, _ok: bool) -> None:
        nonlocal n_done
        n_done += 1
        if n_done % 200 == 0 or n_done == len(hashes):
            rate = n_done / max(1e-9, time.time() - t0)
            print(f"  {n_done}/{len(hashes)} · {rate:.0f}/s", flush=True)

    curations, failed = await fetch_curations(
        hashes,
        base_url=args.base_url,
        concurrency=args.concurrency,
        retries=args.retries,
        backoff=args.backoff,
        timeout=args.timeout,
        on_progress=progress,
    )

    # Write curations + mark every non-failed hash attempted (so resume is exact
    # even for statements that legitimately returned zero curations).
    failed_set = set(failed)
    by_tag: dict[str, int] = {}
    with_cur = {c["_matches_hash"] for c in curations}
    with open(out_path, "a", buffering=1) as out_f, open(sidecar, "a", buffering=1) as att_f:
        for c in curations:
            out_f.write(json.dumps(c) + "\n")
            by_tag[c.get("tag")] = by_tag.get(c.get("tag"), 0) + 1
        for mh in hashes:
            if mh not in failed_set:
                att_f.write(f"{mh}\n")

    print(f"\nDONE: {len(curations)} curations across {len(with_cur)} statements "
          f"({len(failed)} hard failures) -> {out_path}")
    print(f"tag distribution: {by_tag}")
    if failed:
        print(f"note: {len(failed)} hashes failed all retries — rerun to retry just those.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--base-url", default=INDRA_DB_REST_URL)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--backoff", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=None, help="cap statements (first N)")
    ap.add_argument("--smoke", type=int, default=None, help="sample N spread across corpus, then stop")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
