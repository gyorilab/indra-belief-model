"""Does the 4B hallucinate entity-presence to justify a mapping? (run on noot-1)

Asks, per claim entity, whether a phrase in the sentence REFERS TO that gene/protein
(possibly under a synonym/alias/descriptive name). Two cohorts:
 - PRESENT: synonym-bail FPs (entity genuinely present under a synonym) -> correct = present.
 - CONTROL: grounding / entity_boundary gold=incorrect (surface form mis-denotes the
   entity, i.e. entity not truly present) -> correct = absent; a "present" + invented
   span/alias here is a HALLUCINATION (the risk of letting the LLM justify mappings).
"""
import json, re, sys
sys.path.insert(0, "src")
from indra_belief.model_client import ModelClient
from indra_belief.scorers.probes._llm import _extract_json

MASK = (1 << 64) - 1
def um(x):
    try: return int(x) & MASK
    except Exception: return None

G = {int(k): v for k, v in json.load(open("/tmp/eval_curation_v1_gold.json")).items()}
meta = {um(json.loads(l).get("source_hash")): json.loads(l) for l in open("/tmp/ec1_arma_r1.jsonl")}
txt = {}
d = json.load(open("/tmp/eval_curation_v1_statements.json"))
for s in (d if isinstance(d, list) else d.get("statements", d)):
    for e in (s.get("evidence") or []):
        if e.get("source_hash") is not None:
            txt[um(e["source_hash"])] = e.get("text", "")

# PRESENT cohort: synonym-bail FPs (real complexes step-1 bailed on)
present = json.load(open("/tmp/relnat_newFP.json"))[:12]
present = [{"subj": c["subj"], "obj": c["obj"], "text": c["text"], "cohort": "PRESENT"} for c in present]
# CONTROL cohort: grounding/entity_boundary incorrect from eval_curation_v1
control = []
for sh, g in G.items():
    if g["gold"] == "incorrect" and g.get("tag") in ("grounding", "entity_boundaries") and sh in meta and txt.get(sh):
        control.append({"subj": meta[sh]["subject"], "obj": meta[sh]["object"],
                        "text": txt[sh], "cohort": "CONTROL"})
    if len(control) >= 12:
        break

SYS = (
    "You decide whether a specific GENE/PROTEIN is referred to in a sentence. The entity may "
    "appear under a synonym, alias, or descriptive name. Answer PRESENT only if a phrase in the "
    "SENTENCE actually denotes that gene/protein itself. Answer ABSENT if no phrase refers to it. "
    "Do NOT answer present for a phrase that merely shares letters but denotes something else (a "
    "protein DOMAIN, a different gene, a fragment). Do NOT use background knowledge to assert a "
    'name appears if it is not in the sentence. Output JSON: {"answer": "present"|"absent", '
    '"span": "<exact quote from the sentence, or empty>", "alias_reason": "<why the span denotes '
    'the entity, or empty>"}.'
)
def user(ent, text):
    return f'Gene/protein: {ent}\nSentence: "{text}"\nIs {ent} referred to in this sentence?'

client = ModelClient("remote-medpsy-4b")
rows = present + control
res = []
for r in rows:
    for ent in (r["subj"], r["obj"]):
        if not ent or ent == "?":
            continue
        try:
            resp = client.call(system=SYS, messages=[{"role": "user", "content": user(ent, r["text"])}],
                               max_tokens=2500, temperature=0.1,
                               response_format={"type": "json_object"}, reasoning_effort="none",
                               kind="alias_probe")
            o = _extract_json((((resp.content or "") or (resp.raw_text or "")).strip())) or {}
        except Exception:
            o = {}
        res.append({"cohort": r["cohort"], "ent": ent, "ans": o.get("answer"),
                    "span": str(o.get("span", ""))[:80], "reason": str(o.get("alias_reason", ""))[:90]})
        print("  [%s] %-12s -> %-7s span=%r" % (r["cohort"], ent, o.get("answer"), str(o.get("span", ""))[:50]), flush=True)

json.dump(res, open("/tmp/alias_probe_result.json", "w"))
def rate(cohort, ans):
    sub = [x for x in res if x["cohort"] == cohort]
    return sum(1 for x in sub if x["ans"] == ans), len(sub)
pp, pn = rate("PRESENT", "present")
ca, cn = rate("CONTROL", "absent")
cp, _ = rate("CONTROL", "present")
print("\nPRESENT cohort: said present %d/%d (recall of real synonyms)" % (pp, pn))
print("CONTROL cohort: said absent %d/%d (correct); said PRESENT %d/%d (HALLUCINATION)" % (ca, cn, cp, cn))
