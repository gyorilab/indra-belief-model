/** Publication gate layered on the viewer's exact artifact contract. */
import { readFileSync } from 'node:fs';

import { validateBeliefComparisonArtifact } from '../src/lib/data/belief-comparison.ts';

const EXPECTED_LLM_ARMS = [
	'llm_gemma_4_e2b',
	'llm_gemma_4_26b',
	'llm_gemma_4_31b',
	'llm_glm_5'
];
const EXPECTED_PANELS = ['paper_all_source', 'paper_readers'];
const EXPECTED_CANONICAL_ARMS = {
	paper_all_source: {
		paper: [
			'rf_2k_d13_type_pmids_promoter_all_sources_specific',
			'rf_2k_d13_type_pmids_promoter_avglen_all_sources_specific'
		],
		current: [
			'indra_1.24.0_simple_default_direct',
			'indra_1.24.0_bayesian_source_oof_all_sources',
			'indra_1.24.0_bayesian_source_subtype_oof_all_sources',
			'indra_1.24.0_simple_hierarchy_all_sources',
			'indra_1.24.0_counts_rf_2kd13_source_only_oof_all_sources',
			'indra_1.24.0_counts_rf_2kd13_full_features_oof_all_sources',
			'indra_cogex_hybrid_all_source'
		]
	},
	paper_readers: {
		paper: ['orig_belief_readers'],
		current: [
			'indra_1.24.0_simple_direct_readers_only',
			'indra_1.24.0_bayesian_source_oof_readers_only',
			'indra_1.24.0_bayesian_source_subtype_oof_readers_only',
			'indra_1.24.0_simple_hierarchy_readers_only',
			'indra_1.24.0_counts_rf_2kd13_source_only_oof_readers_only',
			'indra_1.24.0_counts_rf_2kd13_full_features_oof_readers_only',
			'indra_cogex_hybrid_readers'
		]
	}
};
const EXPECTED_ERROR_REVIEW_PROTOCOL_SHA256 =
	'910b660d626202668f72c941f277ebee95fb8794e83208082995c58a4fe1987a';
const HUMAN_ATTESTATION =
	'I attest that I personally reviewed every assigned case without model-generated adjudication and that this ledger accurately records my decisions.';
const REPORT_FIELDS = [
	'artifact_kind',
	'status',
	'panel_id',
	'arm_id',
	'model_id',
	'packet_id',
	'evaluated_statements',
	'threshold_errors',
	'error_types',
	'human_classifications',
	'review',
	'defensibility',
	'dimensions',
	'taxonomy_refinements',
	'adjudications',
	'provenance'
];
const REVIEW_FIELDS = [
	'reviewer_pseudonyms',
	'resolver_pseudonym',
	'exact_agreement',
	'disagreement_count',
	'resolved_by_resolver_count',
	'classification_reliability',
	'human_attestation'
];
const PROVENANCE_FIELDS = [
	'protocol',
	'codebook',
	'packet',
	'admin_manifest',
	'reviewer_ledgers',
	'reviewer_workbooks',
	'reviewer_assignments',
	'reviewer_workbook_packets',
	'comparison_inputs',
	'resolver_workload',
	'resolver_workbook',
	'resolver_ledger'
];
const ADJUDICATION_FIELDS = [
	'case_id',
	'error_type',
	'human_classification',
	'judgment',
	'defensibility_basis',
	'dimensions',
	'comment',
	'decision_source'
];
const COMPARISON_INPUT_FILES = [
	'aggregation_config',
	'spec',
	'bundle_manifest',
	'protocol',
	'gold',
	'predictions',
	'execution_ledger',
	'statements',
	'execution_map',
	'raw_attempts',
	'pricing_config',
	'spend_ledger'
];
const CLASSIFICATIONS = new Set(['supports_claim', 'rejects_claim', 'indeterminate']);
const ERROR_TYPES = ['false_positive', 'false_negative'];
const JUDGMENTS = new Set(['defensible', 'non_defensible']);
const DEFENSIBILITY_FIELDS = [
	'denominator',
	'defensible',
	'non_defensible',
	'system_supported_defensible',
	'indeterminate_ambiguity_defensible',
	'unresolved'
];
const PSEUDONYM = /^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$/;
const OPAQUE_PACKET = /^packet_[a-f0-9]{64}$/;
const OPAQUE_CASE = /^case_[a-f0-9]{64}$/;
const SHA256 = /^[a-f0-9]{64}$/i;

function readJson(path) {
	try {
		return JSON.parse(readFileSync(path, 'utf8'));
	} catch (error) {
		throw new Error(`could not parse ${path}: ${String(error)}`);
	}
}

function object(value) {
	return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function integer(value, minimum = 0) {
	return Number.isInteger(value) && value >= minimum;
}

function sameStrings(actual, expected) {
	return JSON.stringify([...actual].sort()) === JSON.stringify([...expected].sort());
}

function isNonlocalUrl(value) {
	return /^(?!file:)[A-Za-z][A-Za-z0-9+.-]*:\/\/\S+$/i.test(value);
}

function absoluteLocalPathLocations(value, context = 'artifact') {
	const locations = [];
	const visit = (item, itemContext) => {
		if (typeof item === 'string') {
			if (isNonlocalUrl(item)) return;
			const posix = /(?:^|[^A-Za-z0-9._~+\\/-])(\/(?!\/)[^\s'"<>]*)/;
			const windowsDrive = /(?:^|[^A-Za-z0-9._~+\\/-])([A-Za-z]:[\\/][^\s'"<>]*)/;
			const windowsUnc = /(?:^|[^A-Za-z0-9._~+\\/-])(\\\\[^\\/\s]+[\\/][^\s'"<>]*)/;
			if (/^file:/i.test(item) || posix.test(item) || windowsDrive.test(item) || windowsUnc.test(item)) {
				locations.push(itemContext);
			}
			return;
		}
		if (Array.isArray(item)) {
			item.forEach((child, index) => visit(child, `${itemContext}[${index}]`));
			return;
		}
		if (object(item)) {
			for (const [key, child] of Object.entries(item)) visit(child, `${itemContext}.${key}`);
		}
	};
	visit(value, context);
	return locations;
}

function exactFields(value, expected, prefix, failures) {
	if (!object(value)) {
		failures.push(`${prefix}: expected an object`);
		return false;
	}
	const actual = Object.keys(value);
	const missing = expected.filter((key) => !Object.hasOwn(value, key));
	const extra = actual.filter((key) => !expected.includes(key));
	if (missing.length || extra.length) {
		failures.push(
			`${prefix}: schema differs` +
				(missing.length ? `; missing ${missing.join(', ')}` : '') +
				(extra.length ? `; unexpected ${extra.join(', ')}` : '')
		);
		return false;
	}
	return true;
}

function descriptor(value) {
	if (!object(value)) return false;
	const fields = Object.keys(value);
	return Boolean(
		SHA256.test(value.sha256 ?? '') &&
		integer(value.bytes) &&
		fields.every((key) => ['sha256', 'bytes', 'rows'].includes(key)) &&
		(!Object.hasOwn(value, 'rows') || integer(value.rows))
	);
}

function summary(value, expectedDenominator, prefix, failures, extraFields = []) {
	if (!exactFields(value, ['count', 'denominator', 'proportion', ...extraFields], prefix, failures)) return false;
	if (!integer(value.count) || value.count > expectedDenominator || value.denominator !== expectedDenominator) {
		failures.push(`${prefix}: count/denominator is inconsistent`);
		return false;
	}
	const expected = expectedDenominator === 0 ? null : value.count / expectedDenominator;
	if (
		(expected === null && value.proportion !== null) ||
		(expected !== null &&
			(typeof value.proportion !== 'number' || Math.abs(value.proportion - expected) > 1e-12))
	) {
		failures.push(`${prefix}: proportion is inconsistent`);
		return false;
	}
	return true;
}

function derivedJudgment(errorType, classification) {
	if (classification === 'indeterminate') return 'defensible';
	if (errorType === 'false_positive') {
		return classification === 'supports_claim' ? 'defensible' : 'non_defensible';
	}
	return classification === 'rejects_claim' ? 'defensible' : 'non_defensible';
}

function derivedDefensibilityBasis(classification, judgment) {
	if (classification === 'indeterminate') return 'indeterminate_ambiguity';
	return judgment === 'defensible' ? 'human_matches_system' : 'human_matches_reference';
}

function reviewFailures(value, expectedPanel, path) {
	const failures = [];
	const prefix = `review ${path}`;
	if (!exactFields(value, REPORT_FIELDS, prefix, failures)) return failures;
	if (value.artifact_kind !== 'indra_belief_error_review_report') failures.push(`${prefix}: wrong artifact_kind`);
	if (value.status !== 'complete') failures.push(`${prefix}: status must be complete`);
	if (value.panel_id !== expectedPanel) failures.push(`${prefix}: panel_id must be ${expectedPanel}`);
	if (value.arm_id !== 'llm_gemma_4_e2b' || value.model_id !== 'llm_gemma_4_e2b') {
		failures.push(`${prefix}: must embed the Gemma 4 E2B arm/model identity`);
	}
	if (!OPAQUE_PACKET.test(value.packet_id ?? '')) failures.push(`${prefix}: packet_id is invalid`);
	if (!integer(value.evaluated_statements, 1)) failures.push(`${prefix}: evaluated_statements must be positive`);

	const adjudications = Array.isArray(value.adjudications) ? value.adjudications : [];
	if (!Array.isArray(value.adjudications)) failures.push(`${prefix}: adjudications must be an array`);
	const total = integer(value.threshold_errors?.count) ? value.threshold_errors.count : -1;
	if (integer(value.evaluated_statements, 1)) {
		summary(value.threshold_errors, value.evaluated_statements, `${prefix}: threshold_errors`, failures);
	}
	if (total !== adjudications.length) failures.push(`${prefix}: threshold-error count must equal adjudication count`);

	const caseIds = new Set();
	const observed = {
		false_positive: { total: 0, defensible: 0, non_defensible: 0 },
		false_negative: { total: 0, defensible: 0, non_defensible: 0 }
	};
	const observedClassifications = {
		supports_claim: 0,
		rejects_claim: 0,
		indeterminate: 0
	};
	const observedBases = {
		human_matches_system: 0,
		human_matches_reference: 0,
		indeterminate_ambiguity: 0
	};
	let resolverDecisions = 0;
	for (const [index, row] of adjudications.entries()) {
		const rowPrefix = `${prefix}: adjudications[${index}]`;
		if (!exactFields(row, ADJUDICATION_FIELDS, rowPrefix, failures)) continue;
		if (!OPAQUE_CASE.test(row.case_id ?? '') || caseIds.has(row.case_id)) {
			failures.push(`${rowPrefix}: case_id is invalid or duplicate`);
		} else {
			caseIds.add(row.case_id);
		}
		if (!ERROR_TYPES.includes(row.error_type)) failures.push(`${rowPrefix}: error_type is invalid`);
		if (!CLASSIFICATIONS.has(row.human_classification)) failures.push(`${rowPrefix}: human_classification is invalid`);
		if (!JUDGMENTS.has(row.judgment)) failures.push(`${rowPrefix}: judgment is invalid`);
		if (
			ERROR_TYPES.includes(row.error_type) &&
			CLASSIFICATIONS.has(row.human_classification) &&
			row.judgment !== derivedJudgment(row.error_type, row.human_classification)
		) {
			failures.push(`${rowPrefix}: judgment is inconsistent with the blinded classification`);
		}
		if (
			CLASSIFICATIONS.has(row.human_classification) &&
			JUDGMENTS.has(row.judgment) &&
			row.defensibility_basis !== derivedDefensibilityBasis(row.human_classification, row.judgment)
		) {
			failures.push(`${rowPrefix}: defensibility_basis is inconsistent with the exact derivation`);
		}
		if (
			!Array.isArray(row.dimensions) ||
			row.dimensions.length === 0 ||
			row.dimensions.some((item) => typeof item !== 'string' || !/^[a-z][a-z0-9_]{1,63}$/.test(item)) ||
			new Set(row.dimensions).size !== row.dimensions.length
		) {
			failures.push(`${rowPrefix}: dimensions must contain unique canonical identifiers`);
		}
		if (row.comment !== null && (typeof row.comment !== 'string' || row.comment.length === 0)) {
			failures.push(`${rowPrefix}: comment must be null or non-empty text`);
		}
		if (!['reviewer_agreement', 'resolver'].includes(row.decision_source)) {
			failures.push(`${rowPrefix}: decision_source is invalid`);
		}
		if (row.decision_source === 'resolver') resolverDecisions += 1;
		if (CLASSIFICATIONS.has(row.human_classification)) {
			observedClassifications[row.human_classification] += 1;
		}
		if (Object.hasOwn(observedBases, row.defensibility_basis)) {
			observedBases[row.defensibility_basis] += 1;
		}
		if (ERROR_TYPES.includes(row.error_type) && JUDGMENTS.has(row.judgment)) {
			observed[row.error_type].total += 1;
			observed[row.error_type][row.judgment] += 1;
		}
	}

	if (!exactFields(value.human_classifications, [...CLASSIFICATIONS], `${prefix}: human_classifications`, failures)) {
		// The exact-fields failure describes the schema mismatch.
	} else if (total >= 0) {
		let classified = 0;
		for (const classification of CLASSIFICATIONS) {
			const row = value.human_classifications[classification];
			summary(row, total, `${prefix}: human_classifications.${classification}`, failures);
			if (row?.count !== observedClassifications[classification]) {
				failures.push(`${prefix}: human_classifications.${classification} count differs from adjudications`);
			}
			classified += row?.count ?? 0;
		}
		if (classified !== total) failures.push(`${prefix}: human classification counts do not cover all errors`);
	}

	if (!exactFields(value.error_types, ERROR_TYPES, `${prefix}: error_types`, failures)) {
		failures.push(`${prefix}: error_types must contain false_positive and false_negative`);
	} else if (total >= 0) {
		for (const errorType of ERROR_TYPES) {
			const row = value.error_types[errorType];
			const rowPrefix = `${prefix}: error_types.${errorType}`;
			if (!object(row)) {
				failures.push(`${rowPrefix}: expected an object`);
				continue;
			}
			summary(row, total, rowPrefix, failures, ['defensible', 'non_defensible']);
			if (row.count !== observed[errorType].total) failures.push(`${rowPrefix}: count differs from adjudications`);
			for (const judgment of JUDGMENTS) {
				summary(row[judgment], row.count, `${rowPrefix}.${judgment}`, failures);
				if (row[judgment]?.count !== observed[errorType][judgment]) {
					failures.push(`${rowPrefix}.${judgment}: count differs from adjudications`);
				}
			}
		}
	}

	if (!exactFields(value.defensibility, DEFENSIBILITY_FIELDS, `${prefix}: defensibility`, failures)) {
		// The exact-fields failure describes stale or missing summary fields.
	} else if (value.defensibility.denominator !== 'all_threshold_errors') {
		failures.push(`${prefix}: defensibility denominator is invalid`);
	} else if (total >= 0) {
		for (const field of DEFENSIBILITY_FIELDS.slice(1)) {
			summary(value.defensibility[field], total, `${prefix}: defensibility.${field}`, failures);
		}
		const defensible = value.defensibility.defensible?.count;
		const nonDefensible = value.defensibility.non_defensible?.count;
		if (defensible + nonDefensible !== total) failures.push(`${prefix}: defensibility counts do not cover all errors`);
		if (value.defensibility.unresolved?.count !== 0) failures.push(`${prefix}: completed review must have zero unresolved errors`);
		if (defensible !== observedBases.human_matches_system + observedBases.indeterminate_ambiguity) {
			failures.push(`${prefix}: defensible count differs from adjudicated bases`);
		}
		if (nonDefensible !== observedBases.human_matches_reference) {
			failures.push(`${prefix}: non-defensible count differs from adjudicated bases`);
		}
		if (value.defensibility.system_supported_defensible?.count !== observedBases.human_matches_system) {
			failures.push(`${prefix}: system-supported defensible count differs from adjudications`);
		}
		if (
			value.defensibility.indeterminate_ambiguity_defensible?.count !== observedBases.indeterminate_ambiguity ||
			value.defensibility.indeterminate_ambiguity_defensible?.count !== observedClassifications.indeterminate
		) {
			failures.push(`${prefix}: indeterminate ambiguity split differs from human classifications`);
		}
		if (
			value.defensibility.system_supported_defensible?.count +
				value.defensibility.indeterminate_ambiguity_defensible?.count !==
			defensible
		) {
			failures.push(`${prefix}: defensible subtypes do not reconcile`);
		}
	}
	if (!object(value.dimensions) || !Array.isArray(value.dimensions.rows)) failures.push(`${prefix}: dimensions summary is invalid`);
	if (!Array.isArray(value.taxonomy_refinements)) failures.push(`${prefix}: taxonomy_refinements must be an array`);

	if (!exactFields(value.review, REVIEW_FIELDS, `${prefix}: review`, failures)) return failures;
	const reviewers = value.review.reviewer_pseudonyms;
	if (
		!Array.isArray(reviewers) ||
		reviewers.length !== 2 ||
		reviewers.some((name) => typeof name !== 'string' || !PSEUDONYM.test(name)) ||
		new Set(reviewers.map((name) => name.toLocaleLowerCase('en-US'))).size !== 2
	) {
		failures.push(`${prefix}: exactly two distinct reviewer pseudonyms are required`);
	}
	if (value.review.human_attestation !== HUMAN_ATTESTATION) failures.push(`${prefix}: exact human-only attestation is required`);
	if (!integer(value.review.disagreement_count) || value.review.disagreement_count > total) {
		failures.push(`${prefix}: disagreement_count is invalid`);
	}
	if (value.review.disagreement_count !== value.review.resolved_by_resolver_count) {
		failures.push(`${prefix}: every disagreement must be resolved`);
	}
	if (resolverDecisions !== value.review.disagreement_count) failures.push(`${prefix}: resolver decisions do not equal disagreements`);
	if (total >= 0) {
		summary(value.review.exact_agreement, total, `${prefix}: review.exact_agreement`, failures);
		if (value.review.exact_agreement?.count !== total - value.review.disagreement_count) {
			failures.push(`${prefix}: exact agreement does not reconcile with disagreements`);
		}
	}
	if (!object(value.review.classification_reliability)) failures.push(`${prefix}: classification_reliability is required`);

	const provenance = value.provenance;
	if (!exactFields(provenance, PROVENANCE_FIELDS, `${prefix}: provenance`, failures)) return failures;
	for (const key of ['protocol', 'codebook', 'packet', 'admin_manifest']) {
		if (!descriptor(provenance[key])) failures.push(`${prefix}: provenance.${key} descriptor is required`);
	}
	if (provenance.protocol?.sha256 !== EXPECTED_ERROR_REVIEW_PROTOCOL_SHA256) {
		failures.push(`${prefix}: error-review protocol digest differs from the frozen protocol`);
	}
	if (
		provenance.packet?.sha256 &&
		(!Array.isArray(provenance.reviewer_workbook_packets) ||
			!provenance.reviewer_workbook_packets.some((row) => row?.sha256 === provenance.packet.sha256))
	) {
		failures.push(`${prefix}: reviewer workbook packets do not include the reviewed packet`);
	}
	for (const [key, label] of [
		['reviewer_ledgers', 'reviewer-ledger'],
		['reviewer_workbooks', 'reviewer-workbook']
	]) {
		const rows = provenance[key];
		if (!Array.isArray(rows) || rows.length !== 2 || rows.some((row) => !descriptor(row)) || new Set(rows.map((row) => row.sha256)).size !== 2) {
			failures.push(`${prefix}: two distinct ${label} descriptors are required`);
		}
	}
	const assignments = provenance.reviewer_assignments;
	if (
		!Array.isArray(assignments) ||
		assignments.length !== 2 ||
		assignments.map((row) => row?.reviewer_slot).sort().join(',') !== 'A,B' ||
		assignments.some(
			(row) =>
				!object(row) ||
				!/^assignment_[a-f0-9]{64}$/.test(row.assignment_id ?? '') ||
				!SHA256.test(row.workbook_content_sha256 ?? '')
		) ||
		new Set(assignments.map((row) => row.assignment_id)).size !== 2 ||
		new Set(assignments.map((row) => row.workbook_content_sha256)).size !== 2
	) {
		failures.push(`${prefix}: two distinct A/B reviewer assignments are required`);
	}
	if (
		!Array.isArray(provenance.reviewer_workbook_packets) ||
		provenance.reviewer_workbook_packets.length === 0 ||
		provenance.reviewer_workbook_packets.some((row) => !descriptor(row))
	) {
		failures.push(`${prefix}: reviewer-workbook packet descriptors are required`);
	}

	const comparisonInputs = provenance.comparison_inputs;
	if (!object(comparisonInputs)) {
		failures.push(`${prefix}: comparison_inputs provenance is required`);
	} else {
		if (
			comparisonInputs.panel_id !== value.panel_id ||
			comparisonInputs.arm_id !== value.arm_id ||
			comparisonInputs.model_id !== value.model_id
		) {
			failures.push(`${prefix}: comparison_inputs identity differs from the report`);
		}
		if (
			!object(comparisonInputs.files) ||
			COMPARISON_INPUT_FILES.some((key) => !descriptor(comparisonInputs.files[key]))
		) {
			failures.push(`${prefix}: comparison_inputs file provenance is incomplete`);
		}
		if (comparisonInputs.files?.protocol?.sha256 !== EXPECTED_ERROR_REVIEW_PROTOCOL_SHA256) {
			failures.push(`${prefix}: comparison_inputs protocol digest differs`);
		}
	}

	const resolver = value.review.resolver_pseudonym;
	const resolverUsed = value.review.disagreement_count > 0;
	if (resolverUsed) {
		if (
			typeof resolver !== 'string' ||
			!PSEUDONYM.test(resolver) ||
			(Array.isArray(reviewers) &&
				reviewers.some(
					(name) =>
						typeof name === 'string' &&
						name.toLocaleLowerCase('en-US') === resolver.toLocaleLowerCase('en-US')
				))
		) {
			failures.push(`${prefix}: resolver must be present and distinct from reviewers`);
		}
		for (const key of ['resolver_workload', 'resolver_workbook', 'resolver_ledger']) {
			if (!descriptor(provenance[key])) failures.push(`${prefix}: provenance.${key} descriptor is required when resolving disagreements`);
		}
	} else {
		if (resolver !== null) failures.push(`${prefix}: resolver pseudonym must be null without disagreements`);
		for (const key of ['resolver_workload', 'resolver_workbook', 'resolver_ledger']) {
			if (provenance[key] !== null) failures.push(`${prefix}: unused provenance.${key} must be null`);
		}
	}
	return failures;
}

const [artifactPath, allSourceReviewPath, readerReviewPath] = process.argv.slice(2);
if (artifactPath === '--reviews-only') {
	if (!allSourceReviewPath || !readerReviewPath) {
		console.error('usage: node --experimental-strip-types validate-belief-comparison-publication.mjs --reviews-only ALL_SOURCE_REVIEW.json READER_REVIEW.json');
		process.exit(2);
	}
	const failures = [];
	for (const [path, panelId] of [[allSourceReviewPath, EXPECTED_PANELS[0]], [readerReviewPath, EXPECTED_PANELS[1]]]) {
		try {
			failures.push(...reviewFailures(readJson(path), panelId, path));
		} catch (error) {
			failures.push(String(error));
		}
	}
	if (failures.length) {
		console.error(`publication review gate failed (${failures.length}):\n- ${failures.join('\n- ')}`);
		process.exit(1);
	}
	console.log('publication-ready blinded error-review reports');
	process.exit(0);
}
if (!artifactPath) {
	console.error('usage: node --experimental-strip-types validate-belief-comparison-publication.mjs ARTIFACT.json ALL_SOURCE_REVIEW.json READER_REVIEW.json');
	process.exit(2);
}

let artifact;
try {
	artifact = readJson(artifactPath);
} catch (error) {
	console.error(String(error));
	process.exit(2);
}
const validation = validateBeliefComparisonArtifact(artifact);
if (validation.status !== 'available') {
	console.error(validation.reasons.join('\n'));
	process.exit(1);
}

const failures = [];
const absolutePathLocations = absoluteLocalPathLocations(artifact);
if (absolutePathLocations.length) {
	failures.push(
		`public metrics artifact contains absolute local filesystem paths at ${absolutePathLocations.join(', ')}`
	);
}
if (validation.provenance.bootstrap_resamples !== 10_000) {
	failures.push(`bootstrap resamples must be exactly 10,000 (got ${validation.provenance.bootstrap_resamples})`);
}
for (const panel of validation.panels) {
	const expected = EXPECTED_CANONICAL_ARMS[panel.substrate_id];
	for (const family of ['paper', 'current']) {
		const actualIds = panel.arms
			.filter((arm) => arm.family === family)
			.map((arm) => arm.arm_id);
		if (!sameStrings(actualIds, expected[family])) {
			failures.push(
				`${panel.substrate_id}: ${family === 'paper' ? 'paper' : 'current INDRA'} arm identities differ from the canonical set`
			);
		}
	}
	const llmIds = panel.arms.filter((arm) => arm.family === 'llm').map((arm) => arm.arm_id).sort();
	if (!sameStrings(llmIds, EXPECTED_LLM_ARMS)) {
		failures.push(`${panel.substrate_id}: requires exactly the four planned LLM arms`);
	}
	if (panel.excluded_arms.some((arm) => arm.family === 'llm')) {
		failures.push(`${panel.substrate_id}: missing-model LLM exclusions are forbidden`);
	}
	const strictIds = panel.strict_e0_resolved_sensitivity.arms.filter((arm) => arm.family === 'llm').map((arm) => arm.arm_id).sort();
	if (!sameStrings(strictIds, EXPECTED_LLM_ARMS)) {
		failures.push(`${panel.substrate_id}: strict sensitivity requires all four LLM arms`);
	}
	for (const arm of panel.arms.filter((candidate) => candidate.family === 'llm')) {
		if (arm.cost.status !== 'available') failures.push(`${panel.substrate_id}/${arm.arm_id}: structured cost must be available`);
	}
}

if (!allSourceReviewPath || !readerReviewPath) {
	failures.push('complete all-source and reader blinded-review reports are required');
} else {
	for (const [path, panelId] of [[allSourceReviewPath, EXPECTED_PANELS[0]], [readerReviewPath, EXPECTED_PANELS[1]]]) {
		try {
			const review = readJson(path);
			failures.push(...reviewFailures(review, panelId, path));
			const panel = validation.panels.find((candidate) => candidate.substrate_id === panelId);
			const arm = panel?.arms.find((candidate) => candidate.arm_id === review.arm_id);
			const files = review.provenance?.comparison_inputs?.files;
			if (files?.spec?.sha256 !== validation.provenance.source_manifest_sha256) {
				failures.push(`review ${path}: comparison spec digest differs from the metrics artifact`);
			}
			if (!panel || !arm) {
				failures.push(`review ${path}: reviewed arm is absent from the exact metrics panel`);
			} else {
				if (files?.bundle_manifest?.sha256 !== arm.provenance.implementation_digest) {
					failures.push(`review ${path}: bundle digest differs from the evaluated arm`);
				}
				if (files?.gold?.sha256 !== panel.contract.gold_sha256) {
					failures.push(`review ${path}: gold digest differs from the evaluated panel`);
				}
				if (files?.predictions?.sha256 !== arm.provenance.predictions_sha256) {
					failures.push(`review ${path}: prediction digest differs from the evaluated arm`);
				}
				if (arm.cost.status !== 'available' || files?.execution_ledger?.sha256 !== arm.cost.ledger_sha256) {
					failures.push(`review ${path}: execution-ledger digest differs from the evaluated arm`);
				}
			}
		} catch (error) {
			failures.push(String(error));
		}
	}
}

if (failures.length) {
	console.error(`publication gate failed (${failures.length}):\n- ${failures.join('\n- ')}`);
	process.exit(1);
}
console.log('publication-ready statement-belief comparison: 10,000 resamples, four LLM arms in both panels, structured costs, strict sensitivities, and complete blinded reviews');
