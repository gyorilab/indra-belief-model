"""Reliability test for INDUCED relation-nature reasoning (run on noot-1).

Question: does a FOCUSED relation-nature characterizer fire RELIABLY (consistent
across reps at temp 0.1) where the holistic Arm A fires the same check
intermittently? If yes, induced reasoning beats the deterministic crutch.
Measures consistency (same nature 3/3) + discrimination (physical_binding for
gold=correct Complex vs other-nature for gold=incorrect), sidestepping the
aggregate-F1 noise floor by measuring the characterization's stability directly.
"""
import collections
import json
import sys

sys.path.insert(0, "src")
from indra_belief.model_client import ModelClient
from indra_belief.scorers.probes._llm import _extract_json

MASK = (1 << 64) - 1
def um(x):
    try: return int(x) & MASK
    except Exception: return None

G = {int(k): v for k, v in json.load(open("/tmp/ibr_gold.json")).items()}
meta = {um(json.loads(l).get("source_hash")): json.loads(l) for l in open("/tmp/ibr_medpsy_arma.jsonl")}
ev = {}
d = json.load(open("/tmp/ibr_eval_statements.json"))
for s in (d if isinstance(d, list) else d.get("statements", d)):
    for e in (s.get("evidence") or []):
        if e.get("source_hash") is not None:
            ev[um(e["source_hash"])] = e.get("text", "")

NATURE_SYS = (
    "You characterize the RELATIONSHIP a biomedical EVIDENCE sentence actually ASSERTS "
    "between two named entities — nothing else. Pick the single BEST-fitting nature:\n"
    "- physical_binding: the sentence states, as a finding, that the two entities directly "
    "bind / form a complex / physically associate WITH EACH OTHER.\n"
    "- fusion_construct: the two are named as a gene FUSION or chimeric protein "
    '("X-Y fusion", "X-Y") — ONE molecule, not two binding partners.\n'
    "- signaling_cascade: a functional/regulatory relationship (pathway, axis, activates, "
    "induces, downstream of), NOT a stated physical bind.\n"
    "- co_binding_third: each entity binds or acts on a shared THIRD entity, not each other.\n"
    "- topic_or_aim: the pairing appears only in a title/topic phrase, or inside an aim/"
    "methods clause ('to detect binding of...'), not an asserted result.\n"
    "- other: none of the above.\n"
    "Judge ONLY what THIS sentence asserts (textual) — NEVER background knowledge of what the "
    'proteins do. Output JSON ONLY: {"nature": <one>, "span": <exact words that decide it>}.'
)
def nature_user(subj, obj, text):
    return (f'Entities: {subj}, {obj}\nSentence: "{text}"\n'
            f"What relationship does the sentence assert between {subj} and {obj}?")

client = ModelClient("medpsy-remote")
SUPPORTS = {"physical_binding"}
cohort = [(sh, g) for sh, g in G.items() if meta.get(sh, {}).get("stmt_type") == "Complex"]
print("Complex cohort n=%d" % len(cohort), flush=True)

results = []
for sh, g in cohort:
    r = meta[sh]; subj, obj, text = r["subject"], r["object"], ev.get(sh, "")
    natures = []
    for _ in range(3):
        try:
            resp = client.call(system=NATURE_SYS,
                               messages=[{"role": "user", "content": nature_user(subj, obj, text)}],
                               max_tokens=3000, temperature=0.1,
                               response_format={"type": "json_object"},
                               reasoning_effort="none", kind="relation_nature")
            o = _extract_json(((resp.content or "") or (resp.raw_text or "")).strip())
            natures.append(o.get("nature") if isinstance(o, dict) else None)
        except Exception:
            natures.append(None)
    cnt = collections.Counter(n for n in natures if n)
    top, topn = (cnt.most_common(1)[0] if cnt else (None, 0))
    consistent = topn == 3
    pred = "correct" if top in SUPPORTS else "incorrect"
    ok = pred == g["gold"]
    results.append({"sh": sh, "tag": g["tag"], "gold": g["gold"], "subj": subj, "obj": obj,
                    "natures": natures, "consistent": consistent, "pred": pred, "correct": ok})
    print("  %-9s %s[C]%s natures=%s -> %s %s" % (g["gold"], subj, obj, natures, pred, "OK" if ok else "X"), flush=True)

cons = sum(x["consistent"] for x in results)
acc = sum(x["correct"] for x in results)
inc = [x for x in results if x["gold"] == "incorrect"]
cor = [x for x in results if x["gold"] == "correct"]
print("\nCONSISTENCY (same nature 3/3): %d/%d" % (cons, len(results)))
print("ACCURACY vs gold (majority nature): %d/%d" % (acc, len(results)))
print("  gold=incorrect correctly flagged: %d/%d" % (sum(x["correct"] for x in inc), len(inc)))
print("  gold=correct kept (physical_binding): %d/%d" % (sum(x["correct"] for x in cor), len(cor)))
json.dump(results, open("/tmp/relation_nature_test.json", "w"))
print("wrote /tmp/relation_nature_test.json")
