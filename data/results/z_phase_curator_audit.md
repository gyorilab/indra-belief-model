# Z6 — curator-note audit (eval_set_v4, n=100)

Heuristic bucketing of curator notes; output of the
`scripts/z_phase_curator_audit.py` keyword classifier.

## Bucket distribution

| bucket | n | % |
|---|---|---|
| clear_correct | 48 | 48% |
| clear_incorrect | 48 | 48% |
| hedged_curator | 2 | 2% |
| reader_error_flagged | 2 | 2% |

## Practical accuracy ceiling

- Hedged curator notes: **2** (scorer cannot beat human ambiguity)
- Reader-error-flagged: **2** (upstream INDRA grounding errors outside scorer's reach)
- Strict ceiling (hedged only): **98%**
- Generous ceiling (hedged + reader errors): **96%**

## Examples — clear_correct

- `-59602209119…` (Complex, tag=correct): 
- `125118464432…` (Phosphorylation, tag=correct): 
- `273453866828…` (Phosphorylation, tag=correct): 
- `615316902254…` (Complex, tag=correct): 
- `-16019870858…` (Complex, tag=correct): 

## Examples — clear_incorrect

- `673118733655…` (Complex, tag=no_relation): 
- `650109647137…` (DecreaseAmount, tag=act_vs_amt): 
- `769399020684…` (Activation, tag=polarity): 
- `-85909311798…` (Complex, tag=grounding): 
- `-13001732833…` (Activation, tag=entity_boundaries): 

## Examples — hedged_curator

- `-59800851291…` (Complex, tag=negative_result): Repeat curation: Technically, says that no interaction between VHL and a fragment of RalBP1 (RalBP1-RBD)
- `-69329024309…` (Activation, tag=correct): Should be increase amount, probably, but allowing it

## Examples — reader_error_flagged

- `-39820809696…` (Activation, tag=grounding): [Xa] -> HGNC:3528
- `-70611955159…` (Phosphorylation, tag=correct): [Mps1] -> HGNC:12401

