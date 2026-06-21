Confirmed. All mechanism claims are reproducible against the live substrate. Here is the report.

---

# Upstream Bug Report: Gilda/HGNC Alias Resolution Manufactures False Grounding "MATCH" Verdicts

**Components:** `gilda` (alias scoring) · INDRA bio-ontology HGNC alias lists · downstream verifier (`indra_belief.data.entity.GroundedEntity`)
**Severity:** High — a relation-extraction verifier silently accepts grounding errors with no signal to surface them.
**Reproduced:** 2026-06-22 against installed `gilda>=1.1.0` + INDRA bio_ontology 1.34.

## 1. Summary & Impact

When a reading system extracts a surface string that differs from a claim's canonical entity, Gilda frequently scores the *wrong* gene as the top hit because HGNC alias lists are (a) dirty — they contain stale/erroneous synonyms — and (b) overloaded — short ambiguous tokens appear as aliases of multiple genes. Gilda assigns a **uniform 0.5556 floor** to any exact-but-non-canonical alias hit and resolves ties toward the gene the claim already owns. The result is a "grounding match" that exists **by construction**, not because the reader text denotes the claim entity.

Downstream, `GroundedEntity.resolve(claim, raw_text)` declares `verification_status="MATCH"` whenever the raw_text's top Gilda hit shares `(db, id)` with the claim. Because `0.556 > LOW_CONFIDENCE_THRESHOLD (0.53)`, `has_grounding_signal` returns `False` (entity.py:240–248) and `format_warning()` emits the empty string — **so no warning reaches the LLM verifier and the grounding error is accepted silently.**

In our verifier evaluation, **29 of 31** distinct `claim<-reader` pairs that triggered a false acceptance were manufactured matches of this kind; **12 of those 29 are genuine substrate errors** (6 gene collisions + 6 ambiguous abbreviations).

All outputs below are from live `GroundedEntity.resolve(claim, reader)` calls; every case returns `status=MATCH`, `has_grounding_signal=False`, `format_warning()=''`.

## 2. Genuine bugs — Class A: collision with a *different* gene

The reader text is a real, distinct gene, but Gilda's top hit (or a tie) lands on the claim's gene via a dirty HGNC alias.

| claim | reader | gilda | what reader text actually denotes | manufacturing alias on the claim |
|---|---|---|---|---|
| SH3BGRL3 | SH3BP1 | 0.778 | SH3BP1 = HGNC:10824 (ARHGAP43, a RhoGAP); evidence "SH3BP1 down-regulates Rac1" | SH3BGRL3 lists "SH3 domain-binding protein 1" / "SH3BP-1" |
| SRC | SRC1 | 1.0 | SRC1 = NCOA1 (HGNC:7668, steroid-receptor coactivator-1); "SRC1 interacted weakly with ERRα" | "SRC1" is also an SRC alias |
| SLU7 | 9G8 | 0.556 | 9G8 = SRSF7 (HGNC:10789), splicing factor | "9G8" listed as a SLU7 alias |
| IL17B | NIRF | 0.556 | NIRF = UHRF2 (HGNC:12557); "NIRF binds pRb" | IL17B lists "NIRF"/"Neuronal interleukin-17-related factor" |
| TM7SF2 | Ang-1 | 0.555 | Ang-1 = ANGPT1 (HGNC:484, angiopoietin-1) | TM7SF2 lists "ANG1"/"Another new gene 1" (string collision) |
| TRAF3 | CRAF1 / c-Raf-1 | 0.556 / 0.549 | RAF1 (HGNC:9829); "TC21 binds physically to c-Raf-1" | TRAF3 lists stale "CD40 Receptor-Associated Factor 1" (CRAF1) |

These are **wrong groundings** — the verifier should reject, but receives no signal.

## 3. Genuine bugs — Class B: ambiguous abbreviation

A short, polysemous token resolves to one specific HGNC gene though it has non-gene or multi-gene meanings that the evidence context actually intends.

| claim | reader | gilda | the intended (non-claim) meaning |
|---|---|---|---|
| MBTPS1 | S1P | 0.556 (3 records) | sphingosine-1-phosphate (CHEBI:37550, same 0.556) / S1P receptors; "S1P1/S1P3 abolished S1P-induced YAP activation" is the lipid axis, not Site-1 protease |
| ARR3 | CAR | 0.556 (3 records) | NR1I3 (Constitutive Androstane Receptor); "CAR and RXR heterodimer binds DNA". CAR ties across SPG7, CXADR, NR1I3, ARR3 |
| CSH1 | PL | 0.556 | phospholipid (a lipid, not a protein); "apoA-I–PL interaction… PL-stabilized emulsion" |
| SH2D1A | SAP | 0.556 | serum amyloid P = APCS (HGNC:584); "C1q and C4BP bind SAP". (Standalone "SAP" tops at EFO:sap 0.706; SH2D1A and APCS tie below at 0.556 — the SH2D1A match strength is 0.556, not 0.706.) |
| TKT | TK | 0.556 | generic 2-letter token (tyrosine/thymidine kinase); not transketolase |
| HJV | JH | 0.556 | JAK homology-1 (JH1) kinase domain; "CIS3-SH2 bound Y1007 of JH1" — not hemojuvelin |

For these, the **right fix is context disambiguation, not picking a winner** — Gilda's tie-break toward the claim's gene is what manufactures the false match.

## 4. NON-cases — do **not** "fix" these (over-tightening trap)

The same code path produces **correct** matches for legitimate aliases. Any naive substrate tightening (raising the Gilda floor, requiring ≥2 shared tokens, dropping 0.556 alias hits) **also destroys these** — which is the whole reason a substrate-side threshold fix is wrong.

| claim | reader | gilda | why it's correct |
|---|---|---|---|
| YAP1 | YAP | 1.0 | canonical short form |
| APOA1 | apoA-I | 0.743 | standard protein nomenclature |
| PTK2 | FAK | 1.0 | FAK is *the* common name |
| MMP9 | MMP-9 | 1.0 | hyphenation only |
| INSR | insulin receptor | 1.0 | descriptive name = gene |
| ESRRA | ERRα | **0.549** | Greek alias of estrogen-related receptor-α |
| C4BPA | C4BP | **0.556** | same entity; gold is strict on alpha-chain vs complex |
| KRT23 | K23 | 0.556 | K23/CK23 is the standard keratin-23 short form |
| RRAS2 | TC21 | 0.556 | TC21 is the canonical alias of R-Ras2 |
| SOCS3 | CIS3 | 0.556 | CIS-3 is a true historical SOCS3 alias |

Note the collision (ESRRA 0.549, C4BPA 0.556) directly against the genuine bugs (TRAF3 0.556, MBTPS1 0.556): **legit and buggy cases share identical scores.** No scalar threshold separates them.

## 5. Mechanism (reproducible)

`GroundedEntity._verify_raw_text` (entity.py:89) sets `MATCH` whenever the raw_text's top Gilda hit shares `(db, id)` with the claim's grounding. Three paths manufacture it:

1. **Same-(db,id) top hit** — both #1 results are the same HGNC id (MMP-9, FAK, S1P→MBTPS1).
2. **Single shared specific alias token** between raw_text and the claim's HGNC name list (SH3BP1, SRC1, 9G8, NIRF, CAR, SAP).
3. **Family/descendant rules** (ERK←MAPK, JUN←c-Jun) — intended behavior.

The acceptance gate is the threshold at entity.py:19/105/244:

```
LOW_CONFIDENCE_THRESHOLD = 0.53
is_low_confidence = gilda_score <= 0.53
has_grounding_signal: MATCH & not is_low_confidence -> False  # entity.py:244
```

Because the alias floor (0.5556) sits just above 0.53, every alias hit is "high confidence" → no signal → empty `format_warning()`. **Verified live (2026-06-22):**

```
SH3BGRL3 <- SH3BP1   status=MATCH gilda=0.778 sig=False warn=''
SRC      <- SRC1     status=MATCH gilda=1.0   sig=False warn=''
TRAF3    <- CRAF1    status=MATCH gilda=0.556 sig=False warn='' (is_known_alias=True)
MBTPS1   <- S1P      status=MATCH gilda=0.556 sig=False warn='' (is_known_alias=True)
ARR3     <- CAR      status=MATCH gilda=0.556 sig=False warn=''
ESRRA    <- ERRα     status=MATCH gilda=0.549 sig=False warn=''   # legit, would die if floor raised
C4BPA    <- C4BP     status=MATCH gilda=0.556 sig=False warn=''   # legit
YAP1     <- YAP      status=MATCH gilda=1.0   sig=False warn=''   # legit
```

Two related non-manufactured controls confirm the boundary: `DAPK3<-zip` resolves at gilda 0.484 (≤0.53) → `has_grounding_signal=True` (correctly signalled); `PRKAA2<-"AMP-dependent protein kinase (AMPK)"` is UNRESOLVABLE (no manufactured match).

### Upstream data defects worth fixing at the source (HGNC alias hygiene)
These are clearly wrong/stale synonyms and *can* be corrected without collateral:
- SH3BGRL3 alias "SH3 domain-binding protein 1" (belongs to SH3BP1)
- TRAF3 alias "CRAF1 / CD40 Receptor-Associated Factor 1" colliding with c-Raf-1/RAF1
- TM7SF2 alias "ANG1 / Another new gene 1" colliding with angiopoietin-1
- IL17B alias "NIRF" colliding with UHRF2
- SLU7 alias "9G8" colliding with SRSF7

Scrubbing these specific erroneous aliases removes ~half the Class A collisions at the data layer without touching scoring.

## 6. Recommended fix domain

**Do NOT tighten the substrate threshold or alias-match arity.** Section 4 proves any scalar/arity tightening that kills the buggy 0.556/0.549 hits also kills legit aliases at the *same* scores (ESRRA 0.549, C4BPA/KRT23/RRAS2/SOCS3 0.556). The substrate cannot make the semantic call.

Per this project's principle — **determinism's role is INPUT, never OUTPUT** — the substrate should *surface ambiguity as context* and let the LLM verifier make the call using the evidence sentence:

1. **Stop suppressing the signal on overloaded/aliased hits.** When a `MATCH` is reached only via a non-canonical alias (path 2) *or* the token has competing high-scoring candidates within a small delta of the winner, set `has_grounding_signal=True` and emit the competition as context rather than silently accepting. This widens signal without rejecting anything.
2. **Expose the candidate set as LLM input-grounding.** Supply the runner-up groundings *and* non-gene meanings as context, e.g.:
   - `S1P → {MBTPS1 0.556, sphingosine-1-phosphate CHEBI:37550 0.556, S1PR receptors}`
   - `CAR → {NR1I3, CXADR, SPG7, ARR3 — all 0.556}`
   - `CRAF1/c-Raf-1 → {RAF1, TRAF3}`
   - `SRC1 → {NCOA1 1.0, SRC alias}`
   The LLM then disambiguates against the evidence sentence (the only place the intended sense lives), which correctly rejects the collisions/ambiguities while keeping YAP/apoA-I/FAK/MMP-9.
3. **(Optional, data-layer)** Scrub the specific erroneous HGNC aliases listed in §5 — these are unambiguous data errors and reduce Class A at the source.

**Scope of the real bug:** 12/29 unique manufactured pairs (6 collision + 6 ambiguous) are genuine substrate errors; the other 17 are correct aliases where the gold tag is over-strict. Gilda exposing the competing candidates — not raising 0.53 — is the separating fix.

---

Reproduction script: `GroundedEntity.resolve(claim, reader)` from `src/indra_belief/data/entity.py` (threshold at line 19, signal logic at lines 240–248); source data at `/tmp/grounding_misses.json` (38 records).