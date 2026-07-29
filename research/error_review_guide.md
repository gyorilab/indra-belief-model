# INDRA belief error review

This protocol is a human-only, offline review of the complete threshold-error census for one exact LLM arm and panel. It is designed to answer a narrow scientific question: when the model and released reference disagree at the fixed primary threshold of `0.5`, what conclusion does an independent reader reach from the exact displayed claim and evidence?

Reviewers never see the model, panel, arm, run, gold label, system prediction, error direction, probability, cost, statement identity, execution identity, or clear provenance. They classify only the displayed material as `supports_claim`, `rejects_claim`, or `indeterminate`. The administrator later derives `defensible` versus `non_defensible` from that outcome-blind classification and the hidden false-positive/false-negative mapping.

The workflow has two phases:

1. Two humans independently review a small authenticated pilot. Their comments may refine the dimension taxonomy, after which a human explicitly freezes the codebook.
2. Two humans independently review every threshold error using two separately bound and independently randomized assignment workbooks. A third human must resolve every substantive disagreement before a scientific report can be emitted.

No model adjudication, external lookup, statement override, full-review sampling, or use of the viewer `/review` route is permitted.

## Reviewer task

Use only the assembled claim and exact panel evidence displayed in the offline workbook:

- `supports_claim`: the displayed material establishes the assembled claim under the frozen rubric.
- `rejects_claim`: the displayed material warrants treating the assembled claim as not established, including an explicit contradiction or a clear lack of the evidence the rubric requires.
- `indeterminate`: material insufficiency, conflict, or ambiguity prevents either binary classification.

Every classification requires one or more frozen dimensions. A comment is optional except for `taxonomy_gap`, which requires a substantive explanation. Ordinary comment differences are retained for taxonomy refinement but do not, by themselves, trigger resolution. A difference in classification, dimensions, or a `taxonomy_gap` comment is a disagreement.

Each reviewer must use only the assignment bearing their slot, choose a non-identifying pseudonym, and affirm the human-only attestation in the workbook before export. Reviewer A and reviewer B have the same complete material but different keyed task orders, assignment bindings, browser-storage keys, and workbook bytes. They must not share a browser state or ledger.

## Administrator-only derivation

The administrator manifest retains the hidden mapping. It is never given to reviewers or the resolver.

- `supports_claim` corresponds to a positive human label.
- `rejects_claim` corresponds to a negative human label.
- If the human label matches the hidden system label, the error is `defensible`.
- If the human label matches the hidden released-reference label, the error is `non_defensible`.
- `indeterminate` is `defensible`: the displayed material does not warrant the reference's binary exclusion of the system side.

Thus, on a hidden false positive, `supports_claim` is defensible and `rejects_claim` is non-defensible. On a hidden false negative, the mapping is reversed. Reviewers do not receive these rules in their workbook because knowing the system side would defeat outcome blinding.

The final report contains exact descriptive counts and proportions for the complete fixed census, overall and by hidden false-positive/false-negative stratum. It discloses all three human-classification totals and splits defensible cases into human-matches-system versus indeterminate-ambiguity components, so ambiguity cannot be mistaken for affirmative support of the system. It also reports pre-resolution reviewer agreement, resolved disagreements, dimension counts, and taxonomy-refinement comments. It does not attach Wilson or other IID/binomial confidence intervals to this census.

## What is cryptographically bound

Preparation validates the canonical LLM bundle and exact comparison spec panel/arm, including the fixed threshold and its protocol digest. It binds the model result, execution/cost ledger, gold labels, statement corpus, execution map, ordered statement projection, and ordered panel execution identities. Reader evidence comes only from the exact five-reader ledger projection; evidence merely associated with the same assembled statement is not admitted.

The public packet carries opaque HMAC commitments, recursively identity-scrubbed scientific material, and protocol/codebook digests. It does not carry an error direction or any field that identifies the gold or system side. The private authenticated administrator manifest contains the clear mapping and source-file provenance.

Reviewer ledgers bind their packet, slot, assignment, exact workbook bytes, protocol, codebook, pseudonym, affirmative attestation, timestamps, and complete classifications. Pilot freeze additionally authenticates the administrator manifest, exact frozen codebook content, and the A and B pilot workbooks and ledgers under the blinding key. Resolver artifacts bind both reviewer ledgers and contain every disagreement and no agreement. Any HMAC, digest, schema, identity projection, coverage, assignment, workbook-byte, or ordering mismatch invalidates the chain.

## Administrator setup

Run commands from the repository root with `PYTHONPATH=src`. The existing blinding key is passed only by filename; none of these commands prints its contents. Keep it mode `0600` and never send it to a reviewer.

Administrator artifacts must live outside the repository. Reviewer packets and workbooks are generated artifacts and may live in the ignored reviewer directory.

```bash
export KEY="$PWD/data/comparison/error_review.key"
export REVIEW_DIR="$PWD/data/comparison/error_reviews"
export ADMIN_DIR="$HOME/.local/state/indra-belief/error-review"

mkdir -p "$REVIEW_DIR" "$ADMIN_DIR"
chmod 700 "$ADMIN_DIR"
chmod 600 "$KEY"
```

If the comparison protocol or any bound input changed, rematerialize the canonical comparison spec first:

```bash
PYTHONPATH=src .venv/bin/python -m indra_belief.comparison materialize \
  --inputs data/comparison/inputs.json \
  --output data/results/indra_belief_comparison_spec.json \
  --force
```

## Authenticated human pilot

The checked-in `data/comparison/error_review_codebook.json` is a pilot codebook. It defines the initial dimensions but does not claim that a human pilot or freeze has occurred.

Prepare the blinded 24-case all-source pilot. Direction balancing happens privately; no case exposes its direction.

```bash
PILOT_RESULT=$(PYTHONPATH=src .venv/bin/python -m indra_belief.comparison error-review-prepare \
  --spec data/results/indra_belief_comparison_spec.json \
  --bundle data/comparison/models/gemma_4_e2b/manifest.json \
  --panel paper_all_source \
  --arm llm_gemma_4_e2b \
  --protocol data/comparison/error_review.json \
  --codebook data/comparison/error_review_codebook.json \
  --blinding-key-file "$KEY" \
  --reviewer-output-dir "$REVIEW_DIR" \
  --admin-output-dir "$ADMIN_DIR" \
  --pilot-case-count 24)

PILOT_PACKET=$(printf '%s' "$PILOT_RESULT" | jq -r .packet)
PILOT_ADMIN=$(printf '%s' "$PILOT_RESULT" | jq -r .admin_manifest)
```

Generate the two independently randomized pilot assignments:

```bash
PILOT_WORKBOOK_RESULT=$(PYTHONPATH=src .venv/bin/python -m indra_belief.comparison error-review-workbook \
  --packets "$PILOT_PACKET" \
  --protocol data/comparison/error_review.json \
  --codebook data/comparison/error_review_codebook.json \
  --blinding-key-file "$KEY" \
  --output-dir "$REVIEW_DIR")

PILOT_WORKBOOK_A=$(printf '%s' "$PILOT_WORKBOOK_RESULT" | jq -r '.workbooks[] | select(.reviewer_slot == "A") | .workbook')
PILOT_WORKBOOK_B=$(printf '%s' "$PILOT_WORKBOOK_RESULT" | jq -r '.workbooks[] | select(.reviewer_slot == "B") | .workbook')
```

Give only `PILOT_WORKBOOK_A` to pilot reviewer A and only `PILOT_WORKBOOK_B` to pilot reviewer B. Each reviewer must complete every task, affirm the attestation, and export the ledger for their assigned slot. Do not give either reviewer the packet, administrator manifest, key, other assignment, or any model/reference-side mapping.

Inspect the human taxonomy comments. Create `data/comparison/error_review_codebook_candidate.json` from the pilot codebook and edit only the dimension taxonomy as warranted. Do not manually change its status, classification semantics, pilot contract, or provenance.

Freeze only after the two real pilot ledgers exist. The flag `--attest-human-freeze` is an affirmative human action: the administrator is attesting that the candidate taxonomy reflects review of the completed human pilot. Use the actual freeze time.

```bash
FROZEN_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

PYTHONPATH=src .venv/bin/python -m indra_belief.comparison error-review-codebook-freeze \
  --protocol data/comparison/error_review.json \
  --pilot-codebook data/comparison/error_review_codebook.json \
  --candidate-codebook data/comparison/error_review_codebook_candidate.json \
  --pilot-packet "$PILOT_PACKET" \
  --pilot-admin-manifest "$PILOT_ADMIN" \
  --blinding-key-file "$KEY" \
  --pilot-workbooks "$PILOT_WORKBOOK_A" "$PILOT_WORKBOOK_B" \
  --reviews REVIEWER_A_PILOT.json REVIEWER_B_PILOT.json \
  --attest-human-freeze \
  --frozen-at "$FROZEN_AT" \
  --output data/comparison/error_review_codebook_frozen.json
```

## Complete E2B census

After the codebook is frozen, prepare the complete all-source and five-reader packets. Omitting `--pilot-case-count` is deliberate and enforced: these are censuses, not samples.

```bash
ALL_RESULT=$(PYTHONPATH=src .venv/bin/python -m indra_belief.comparison error-review-prepare \
  --spec data/results/indra_belief_comparison_spec.json \
  --bundle data/comparison/models/gemma_4_e2b/manifest.json \
  --panel paper_all_source \
  --arm llm_gemma_4_e2b \
  --protocol data/comparison/error_review.json \
  --codebook data/comparison/error_review_codebook_frozen.json \
  --blinding-key-file "$KEY" \
  --reviewer-output-dir "$REVIEW_DIR" \
  --admin-output-dir "$ADMIN_DIR")
ALL_PACKET=$(printf '%s' "$ALL_RESULT" | jq -r .packet)
ALL_ADMIN=$(printf '%s' "$ALL_RESULT" | jq -r .admin_manifest)

READER_RESULT=$(PYTHONPATH=src .venv/bin/python -m indra_belief.comparison error-review-prepare \
  --spec data/results/indra_belief_comparison_spec.json \
  --bundle data/comparison/models/gemma_4_e2b/manifest.json \
  --panel paper_readers \
  --arm llm_gemma_4_e2b \
  --protocol data/comparison/error_review.json \
  --codebook data/comparison/error_review_codebook_frozen.json \
  --blinding-key-file "$KEY" \
  --reviewer-output-dir "$REVIEW_DIR" \
  --admin-output-dir "$ADMIN_DIR")
READER_PACKET=$(printf '%s' "$READER_RESULT" | jq -r .packet)
READER_ADMIN=$(printf '%s' "$READER_RESULT" | jq -r .admin_manifest)
```

Generate one pair of assignments over both packets. Byte-identical scrubbed material is shown once within an assignment and expanded into the complete, separately bound packet ledgers at export.

```bash
FULL_WORKBOOK_RESULT=$(PYTHONPATH=src .venv/bin/python -m indra_belief.comparison error-review-workbook \
  --packets "$ALL_PACKET" "$READER_PACKET" \
  --protocol data/comparison/error_review.json \
  --codebook data/comparison/error_review_codebook_frozen.json \
  --blinding-key-file "$KEY" \
  --output-dir "$REVIEW_DIR")

FULL_WORKBOOK_A=$(printf '%s' "$FULL_WORKBOOK_RESULT" | jq -r '.workbooks[] | select(.reviewer_slot == "A") | .workbook')
FULL_WORKBOOK_B=$(printf '%s' "$FULL_WORKBOOK_RESULT" | jq -r '.workbooks[] | select(.reviewer_slot == "B") | .workbook')
```

Give only `FULL_WORKBOOK_A` to reviewer A and only `FULL_WORKBOOK_B` to reviewer B. The ledger exports for the all-source and reader packets are distinct even where one displayed task supplies both classifications.

## Mandatory disagreement resolution

Generate a resolver workload separately for each packet. The ordered `--reviews` and `--reviewer-workbooks` arguments are slot-sensitive: A first, B second. `--workbook-packets` names the exact ordered packet set from which both assignments were generated.

```bash
PYTHONPATH=src .venv/bin/python -m indra_belief.comparison error-review-resolver \
  --packet "$ALL_PACKET" \
  --protocol data/comparison/error_review.json \
  --codebook data/comparison/error_review_codebook_frozen.json \
  --blinding-key-file "$KEY" \
  --reviews REVIEWER_A_ALL.json REVIEWER_B_ALL.json \
  --workbook-packets "$ALL_PACKET" "$READER_PACKET" \
  --reviewer-workbooks "$FULL_WORKBOOK_A" "$FULL_WORKBOOK_B" \
  --output-dir "$REVIEW_DIR"
```

A disagreement is any classification difference, any dimension-set difference, or any difference between the required comments when both decisions select `taxonomy_gap`. If there is at least one disagreement, a third human must complete the generated outcome-blind resolver workbook. The resolver receives neither the administrator manifest nor the hidden system/reference mapping. Adjudication fails until its exact resolver workload and complete resolver ledger are supplied.

If the command reports `not_required`, the two reviewers agree on every resolution-bearing field and no resolver artifact is supplied. The implementation never invents an `unresolved` decision or substitutes an automatic tie-break.

## Final adjudication and report

For a packet that required resolution:

```bash
PYTHONPATH=src .venv/bin/python -m indra_belief.comparison error-review-adjudicate \
  --packet "$ALL_PACKET" \
  --admin-manifest "$ALL_ADMIN" \
  --protocol data/comparison/error_review.json \
  --codebook data/comparison/error_review_codebook_frozen.json \
  --blinding-key-file "$KEY" \
  --reviews REVIEWER_A_ALL.json REVIEWER_B_ALL.json \
  --workbook-packets "$ALL_PACKET" "$READER_PACKET" \
  --reviewer-workbooks "$FULL_WORKBOOK_A" "$FULL_WORKBOOK_B" \
  --resolver-workload OPAQUE_ALL_RESOLVER_WORKLOAD.json \
  --resolver-workbook OPAQUE_ALL_RESOLVER_WORKBOOK.html \
  --resolver-ledger RESOLVER_ALL.json \
  --output data/comparison/error_review_all_source_report.json
```

If the resolver command reported `not_required`, omit all three resolver arguments. Repeat resolver generation and adjudication for the reader packet using `READER_PACKET`, `READER_ADMIN`, and the two reader ledger exports. Never substitute a ledger across packets or slots.

The report is emitted only when the complete chain validates and every disagreement has a real human resolution. It reports exact fixed-census descriptions, not sampling inference.

Render the final comparison with both completed panel reviews. The report command fails closed unless each review's comparison-spec, bundle, panel-gold, prediction, and execution-ledger commitments match the supplied metrics artifact.

```bash
PYTHONPATH=src .venv/bin/python -m indra_belief.comparison report \
  --metrics data/results/indra_belief_comparison_metrics.json \
  --literature data/benchmark/indra_paper_2023_published_method_metrics.json \
  --error-review data/comparison/error_review_all_source_report.json \
  --error-review data/comparison/error_review_reader_report.json \
  --markdown reports/indra_belief_comparison.md \
  --html reports/indra_belief_comparison.html \
  --manifest reports/indra_belief_comparison_manifest.json
```

Then run the publication gate against the same exact three artifacts. This additionally enforces the final 10,000-resample, complete-model, structured-cost, strict-sensitivity, and blinded-review requirements.

```bash
node --experimental-strip-types viewer/scripts/validate-belief-comparison-publication.mjs \
  data/results/indra_belief_comparison_metrics.json \
  data/comparison/error_review_all_source_report.json \
  data/comparison/error_review_reader_report.json
```

## Completion checklist

- The comparison threshold is exactly `0.5` and bound to this protocol.
- Public packets and workbooks reveal no gold label, system prediction, error direction, model identity, or clear source identity.
- Administrator manifests and the blinding key remain private and outside version control; administrator manifests live outside the repository.
- Assignment A and assignment B have separate keyed orders, bindings, workbook files, and browser state.
- Two independent reviewer pseudonyms are distinct; any resolver pseudonym is distinct from both.
- Every human explicitly affirmed the human-only attestation; the pilot freeze has a separate affirmative administrator attestation.
- Start and completion timestamps are present and ordered.
- Every assigned case appears exactly once, uses a valid classification, and has at least one frozen dimension.
- Every `taxonomy_gap` has a substantive human comment.
- Every classification, dimension, or `taxonomy_gap`-comment disagreement has exactly one third-human resolution.
- Final provenance includes the protocol, frozen codebook, packet, private administrator manifest, both exact reviewer workbooks and ledgers, and any resolver artifacts.
- No HMAC, digest, schema, identity projection, coverage, assignment, workbook-byte, or ordering check differs.
- Reported proportions use the full error census as their denominator and carry no IID/binomial confidence interval.
