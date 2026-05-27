import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createServer } from 'vite';

const pairPage = readFileSync(new URL('../src/routes/pairs/[pair_id]/+page.svelte', import.meta.url), 'utf-8');

assert.match(pairPage, /buildPairedComparisonCard/);
assert.match(pairPage, /pairedComparisonCardJson/);
assert.match(pairPage, /pairedComparisonCardMarkdown/);
assert.match(pairPage, /pairedComparisonCardSchemaJson/);
assert.match(pairPage, /let comparisonCardGeneratedAt = \$state\('client-pending'\)/);
assert.match(pairPage, /comparisonCardGeneratedAt = new Date\(\)\.toISOString\(\)/);
assert.match(pairPage, /function comparisonCardSnapshot\(\)/);
assert.match(pairPage, /function comparisonCardDataHref\(kind: 'json' \| 'md' \| 'schema'\): string/);
assert.match(pairPage, /download=\{comparisonCardFilename\('json'\)\}>json card/);
assert.match(pairPage, /download=\{comparisonCardFilename\('md'\)\}>markdown card/);
assert.match(pairPage, /download=\{comparisonCardFilename\('schema'\)\}>json schema/);

function schemaAt(root, ref) {
	assert.ok(ref.startsWith('#/'), `local schema ref expected: ${ref}`);
	return ref
		.slice(2)
		.split('/')
		.reduce((node, rawPart) => {
			const part = rawPart.replace(/~1/g, '/').replace(/~0/g, '~');
			assert.ok(node && Object.prototype.hasOwnProperty.call(node, part), `missing schema ref part ${part} in ${ref}`);
			return node[part];
		}, root);
}

function assertJsonType(type, value, path) {
	if (type === 'null') assert.equal(value, null, `${path} must be null`);
	else if (type === 'array') assert.ok(Array.isArray(value), `${path} must be array`);
	else if (type === 'object') assert.ok(value !== null && typeof value === 'object' && !Array.isArray(value), `${path} must be object`);
	else if (type === 'integer') assert.ok(Number.isInteger(value), `${path} must be integer`);
	else assert.equal(typeof value, type, `${path} must be ${type}`);
}

function validateSchema(schema, value, root = schema, path = '$') {
	if (schema.$ref) {
		validateSchema(schemaAt(root, schema.$ref), value, root, path);
		return;
	}
	if (schema.anyOf) {
		const failures = [];
		for (const option of schema.anyOf) {
			try {
				validateSchema(option, value, root, path);
				return;
			} catch (err) {
				failures.push(err.message);
			}
		}
		assert.fail(`${path} matched no anyOf schema: ${failures.join('; ')}`);
	}
	if (Object.prototype.hasOwnProperty.call(schema, 'const')) {
		assert.deepEqual(value, schema.const, `${path} must equal schema const`);
	}
	if (schema.enum) {
		assert.ok(schema.enum.includes(value), `${path} must be one of ${schema.enum.join(', ')}`);
	}
	if (schema.type) assertJsonType(schema.type, value, path);
	if (schema.minLength != null && typeof value === 'string') {
		assert.ok(value.length >= schema.minLength, `${path} must satisfy minLength ${schema.minLength}`);
	}
	if (schema.pattern && typeof value === 'string') {
		assert.match(value, new RegExp(schema.pattern), `${path} must match ${schema.pattern}`);
	}
	if (schema.minItems != null && Array.isArray(value)) {
		assert.ok(value.length >= schema.minItems, `${path} must satisfy minItems ${schema.minItems}`);
	}
	if (schema.type === 'array' && schema.items) {
		value.forEach((item, index) => validateSchema(schema.items, item, root, `${path}[${index}]`));
	}
	if (schema.type === 'object') {
		const required = schema.required ?? [];
		for (const key of required) assert.ok(Object.prototype.hasOwnProperty.call(value, key), `${path}.${key} is required`);
		if (schema.additionalProperties === false) {
			for (const key of Object.keys(value)) {
				assert.ok(schema.properties?.[key], `${path}.${key} is not in schema properties`);
			}
		}
		for (const [key, propSchema] of Object.entries(schema.properties ?? {})) {
			if (Object.prototype.hasOwnProperty.call(value, key)) validateSchema(propSchema, value[key], root, `${path}.${key}`);
		}
	}
}

const server = await createServer({
	configFile: './vite.config.ts',
	optimizeDeps: {
		disabled: true
	},
	environments: {
		client: {
			optimizeDeps: {
				disabled: true
			}
		}
	},
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	const {
		PAIRED_COMPARISON_CARD_SCHEMA_VERSION,
		buildPairedComparisonCard,
		pairedComparisonCardJson,
		pairedComparisonCardMarkdown,
		pairedComparisonCardSchema,
		pairedComparisonCardSchemaJson
	} = await server.ssrLoadModule('/src/lib/pairedComparisonCard.ts');
	const {
		PAIRED_LEDGER_ROLES,
		PAIRED_METRIC_KINDS,
		PANEL_APPLICABILITY_KINDS
	} = await server.ssrLoadModule('/src/lib/pairedMetricKinds.ts');

	const workbench = {
		pair_id: 'pair_export_card',
		runs: [],
		monolithic: {
			run_id: 'mono_run',
			architecture: 'monolithic',
			scorer_version: 'v1',
			model_id_default: 'demo-model',
			started_at: '2026-05-26 00:00:00',
			finished_at: '2026-05-26 00:01:00',
			status: 'succeeded',
			n_stmts: 2,
			n_evidences: 2,
			duration_s: 60,
			cost_estimate_usd: 0.08,
			cost_actual_usd: 0.06
		},
		decomposed: {
			run_id: 'decomp_run',
			architecture: 'decomposed',
			scorer_version: 'v1',
			model_id_default: 'demo-model',
			started_at: '2026-05-26 00:02:00',
			finished_at: '2026-05-26 00:02:45',
			status: 'succeeded',
			n_stmts: 2,
			n_evidences: 2,
			duration_s: 45,
			cost_estimate_usd: 0.05,
			cost_actual_usd: 0.04
		},
		overlap: {
			monolithic_evidences: 2,
			decomposed_evidences: 2,
			monolithic_comparable_evidences: 2,
			decomposed_comparable_evidences: 2,
			overlap_evidences: 2,
			overlap_statements: 2,
			monolithic_only_evidences: 0,
			decomposed_only_evidences: 0,
			monolithic_true_nonoverlap_evidences: 0,
			decomposed_true_nonoverlap_evidences: 0,
			monolithic_overlap_pct: 1,
			decomposed_overlap_pct: 1,
			monolithic_step_error_evidences: 0,
			decomposed_step_error_evidences: 0,
			monolithic_nonaggregate_step_error_evidences: 0,
			decomposed_nonaggregate_step_error_evidences: 0,
			monolithic_missing_aggregate_evidences: 0,
			decomposed_missing_aggregate_evidences: 0,
			monolithic_nonverdict_aggregate_evidences: 0,
			decomposed_nonverdict_aggregate_evidences: 0
		},
		comparable: {
			n_overlap: 2,
			n_truth_overlap: 2,
			verdict_agreement_n: 1,
			verdict_agreement_rate: 0.5,
			both_correct_n: 1,
			monolithic_only_correct_n: 0,
			decomposed_only_correct_n: 1,
			both_incorrect_n: 0,
			verdict_label_pairs: [],
			monolithic_score_mean: 0.55,
			decomposed_score_mean: 0.65,
			monolithic_mae: 0.12,
			decomposed_mae: 0.18,
			monolithic_bias: -0.02,
			decomposed_bias: 0.08,
			mean_score_delta: 0.1,
			monolithic_latency_mean_ms: 150,
			decomposed_latency_mean_ms: 250,
			monolithic_latency_observed_n: 2,
			decomposed_latency_observed_n: 2,
			monolithic_tokens_total: 60,
			decomposed_tokens_total: 100,
			monolithic_tokens_observed_n: 2,
			decomposed_tokens_observed_n: 2
		},
		resource_frontier: {
			monolithic: {
				architecture: 'monolithic',
				run_id: 'mono_run',
				run_cost_usd: 0.06,
				run_cost_basis: 'actual',
				n_evidences: 2,
				cost_per_evidence_usd: 0.03,
				duration_s: 60,
				wall_seconds_per_evidence: 30,
				clean_overlap_n: 2,
				clean_overlap_latency_mean_ms: 150,
				clean_overlap_latency_observed_n: 2,
				clean_overlap_tokens_total: 60,
				clean_overlap_tokens_observed_n: 2,
				clean_overlap_tokens_per_observed_evidence: 30,
				truth_overlap_n: 2,
				mae: 0.12
			},
			decomposed: {
				architecture: 'decomposed',
				run_id: 'decomp_run',
				run_cost_usd: 0.04,
				run_cost_basis: 'actual',
				n_evidences: 2,
				cost_per_evidence_usd: 0.02,
				duration_s: 45,
				wall_seconds_per_evidence: 22.5,
				clean_overlap_n: 2,
				clean_overlap_latency_mean_ms: 250,
				clean_overlap_latency_observed_n: 2,
				clean_overlap_tokens_total: 100,
				clean_overlap_tokens_observed_n: 2,
				clean_overlap_tokens_per_observed_evidence: 50,
				truth_overlap_n: 2,
				mae: 0.18
			},
			spend_scope: 'whole-run spend; denominator is each side aggregate evidence count, not clean overlap',
			latency_scope: 'clean shared aggregate verdict evidence with per-row telemetry counts shown',
			quality_scope: 'MAE over truth-anchored clean overlap',
			not_defined_reason: null
		},
		denominator_ledger: [
			{
				key: 'resource_counter_metric',
				panel: 'latency and token counters',
				applicability: 'paired_only',
				metric_kind: 'arch_arch_resource',
				ledger_role: 'metric',
				parent_key: 'clean_shared_aggregate_verdict_evidence',
				unit: 'clean shared aggregate verdict evidence',
				denominator_n: 2,
				monolithic_n: 2,
				decomposed_n: 2,
				overlap_n: 2,
				excluded_n: 0,
				reason: 'resource counters over the clean shared denominator'
			}
		],
		exemplars: {
			monolithic_wins: [],
			decomposed_wins: [],
			verdict_disagreements: [],
			mutual_failures: [],
			monolithic_only: [],
			decomposed_only: [],
			excluded_by_integrity: []
		},
		arch_conditioned: {
			monolithic_tiers: [{ tier: 'direct', n: 2, mean_score: 0.55 }],
			decomposed_probes: [{ name: 'subject_role_probe', n: 2, substrate_n: 1, error_n: 0 }]
		},
		not_defined_reason: null
	};

	const card = buildPairedComparisonCard(workbench, {
		generated_at: '2026-05-26T00:00:00.000Z',
		pair_href: '/pairs/pair_export_card',
		workflow_status: 'succeeded'
	});
	assert.equal(PAIRED_COMPARISON_CARD_SCHEMA_VERSION, 'indra_pair_comparison_card_v1');
	assert.equal(card.schema_version, PAIRED_COMPARISON_CARD_SCHEMA_VERSION);
	assert.equal(card.status, 'defined');
	assert.equal(card.runs.monolithic.run_id, 'mono_run');
	assert.equal(card.overlap.overlap_evidences, 2);
	assert.equal(card.resource_frontier.spend_scope, 'whole-run spend; denominator is each side aggregate evidence count, not clean overlap');
	assert.ok(card.guardrails.includes('Lower resource use is not a quality signal.'));
	assert.ok(card.guardrails.includes('Architecture-native diagnostics are not converted into the other architecture grammar.'));

	const json = pairedComparisonCardJson(card);
	assert.equal(JSON.parse(json).pair_id, 'pair_export_card');
	assert.ok(json.includes('"denominator_ledger"'));

	const schema = JSON.parse(pairedComparisonCardSchemaJson());
	assert.deepEqual(schema, pairedComparisonCardSchema);
	assert.equal(schema.$schema, 'https://json-schema.org/draft/2020-12/schema');
	assert.equal(schema.properties.schema_version.const, PAIRED_COMPARISON_CARD_SCHEMA_VERSION);
	assert.ok(schema.$id.endsWith('/indra_pair_comparison_card_v1.schema.json'));
	assert.deepEqual([...schema.required].sort(), Object.keys(card).sort());
	assert.deepEqual(schema.$defs.denominatorRow.properties.metric_kind.enum, PAIRED_METRIC_KINDS);
	assert.deepEqual(schema.$defs.denominatorRow.properties.applicability.enum, PANEL_APPLICABILITY_KINDS);
	assert.deepEqual(schema.$defs.denominatorRow.properties.ledger_role.enum, PAIRED_LEDGER_ROLES);
	validateSchema(schema, card);

	const markdown = pairedComparisonCardMarkdown(card);
	assert.match(markdown, /^# Paired Architecture Comparison: pair_export_card/m);
	assert.match(markdown, /Paired metrics default to clean shared aggregate verdict evidence/);
	assert.match(markdown, /Lower resource use is not a quality signal/);
	assert.match(markdown, /Spend scope: whole-run spend; denominator is each side aggregate evidence count, not clean overlap/);
	assert.match(markdown, /\| exact verdict agreement \| - \| - \| 1\/2 exact labels \| clean shared aggregate verdict evidence \|/);
	assert.match(markdown, /\| MAE vs INDRA \| 0\.12 \| 0\.18 \| - \| truth overlap n=2 \|/);
	assert.match(markdown, /resource_counter_metric/);
	assert.match(markdown, /Monolithic tier rows: 2/);
	assert.match(markdown, /Decomposed probe rows: 2/);

	console.log('paired comparison card tests passed');
} finally {
	await server.close();
}
