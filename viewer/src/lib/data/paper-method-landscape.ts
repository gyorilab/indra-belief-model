import { fail, record } from './paper-validate.ts';
/**
 * Strict display adapter for the checksum-pinned method summaries published with
 * the 2023 INDRA assembly paper.
 *
 * These rows are contextual literature anchors. They are never promoted into
 * the shared-statement arm set, paired deltas, or a cost frontier: the paper
 * published rounded fold summaries, not statement-level predictions or costs.
 */

export const PAPER_METHOD_ARTIFACT_KIND = 'indra_assembly_paper_published_method_metrics';
export const PAPER_METHOD_SCHEMA_VERSION = 1;
export const PAPER_METHOD_EXPECTED_SHA256 =
	'1c3af970a146dd8a1fdd5d35f43f366422cfdd7567a66bccfc932b2efde2f7d4';

export const PAPER_METHOD_FAMILIES = [
	'original belief',
	'random forest',
	'logistic regression',
	'SVC',
	'KNN'
] as const;

export type PaperMethodFamily = (typeof PAPER_METHOD_FAMILIES)[number];

/**
 * The four INPUT CONFIGURATIONS the paper reports every method under. They are
 * the trailing ` - <configuration>` segment of the published method name, and
 * they are the single easiest way to be wrong about these rows: `, specific` is
 * the `include_more_specific` featurizer flag, and `readers` vs `all sources`
 * is which source set was admitted. Two rows from different configurations are
 * NOT two measurements of the same thing, so a delta across them is a category
 * error. Ordered comparable-first; `PAPER_COMPARABLE_SLICE` is the only one our
 * 1,689-statement panel matches.
 */
export const PAPER_METHOD_SLICES = [
	'all sources, specific',
	'all sources',
	'readers, specific',
	'readers'
] as const;

export type PaperMethodSlice = (typeof PAPER_METHOD_SLICES)[number];

/** The one published configuration our all-sources-specific panel reproduces. */
export const PAPER_COMPARABLE_SLICE: PaperMethodSlice = 'all sources, specific';

/** Frozen census of the 59 published rows by configuration; drift gates the load. */
export const PAPER_METHOD_SLICE_COUNTS: Record<PaperMethodSlice, number> = {
	'all sources, specific': 15,
	'all sources': 14,
	'readers, specific': 15,
	readers: 15
};

export interface PaperMethodRow {
	method_id: string;
	method: string;
	family: PaperMethodFamily;
	/** Derived from the method name's trailing configuration segment. */
	slice: PaperMethodSlice;
	/** The method name with its ` - <configuration>` segment removed. */
	base_method: string;
	table_id: 'paper_table_6' | 'paper_not_table_6';
	notebook_cell_index: 47 | 48;
	row: number;
	fold_count: 10;
	fold_mean_trapezoidal_pr_auc: number;
	fold_population_sd: number;
}

export interface PaperMethodSource {
	repository: string;
	commit: string;
	notebook_path: string;
	notebook_sha256: string;
	executed_output_cells: [47, 48];
}

export interface PaperMethodLandscape {
	method_count: 59;
	methods: PaperMethodRow[];
	family_counts: Record<PaperMethodFamily, number>;
	slice_counts: Record<PaperMethodSlice, number>;
	baseline: PaperMethodRow;
	/** Best row anywhere in the artifact — spans configurations, so never a target. */
	best: PaperMethodRow;
	/**
	 * Best row WITHIN `PAPER_COMPARABLE_SLICE`. This is the only published number
	 * anything measured on our panel may be read against; `best` is not.
	 */
	comparable_best: PaperMethodRow;
	source: PaperMethodSource;
	metric_contract: {
		unit: 'assembled_statement';
		positive_class: 'correct assembled statement';
		per_fold_metric: 'sklearn precision_recall_curve followed by auc(recall, precision)';
		summary: 'arithmetic mean over 10 cross-validation folds';
		uncertainty_field: 'population standard deviation over the 10 folds';
		uncertainty_is_confidence_interval: false;
		metric_is_pooled_average_precision: false;
		directly_comparable_to_pair_error_f1: false;
	};
}

export interface PaperMethodLandscapeAvailable {
	status: 'available';
	landscape: PaperMethodLandscape;
	reason: null;
	artifact_path: string;
	artifact_sha256: string;
}

export interface PaperMethodLandscapeUnavailable {
	status: 'unavailable';
	landscape: null;
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
}

export type PaperMethodLandscapeLoad =
	| PaperMethodLandscapeAvailable
	| PaperMethodLandscapeUnavailable;

type UnknownRecord = Record<string, unknown>;



function exactKeys(value: UnknownRecord, keys: readonly string[], context: string): void {
	const got = Object.keys(value).sort();
	const want = [...keys].sort();
	if (got.length !== want.length || got.some((key, index) => key !== want[index])) {
		fail(context, `unexpected fields; got ${got.join(', ')}, want ${want.join(', ')}`);
	}
}

function string(value: unknown, context: string): string {
	if (typeof value !== 'string' || value.length === 0) fail(context, 'expected a non-empty string');
	return value;
}

function integer(value: unknown, context: string, minimum = 0): number {
	if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum) {
		fail(context, `expected an integer >= ${minimum}`);
	}
	return value;
}

function finite(value: unknown, context: string, minimum: number, maximum: number): number {
	if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
		fail(context, `expected a finite number in [${minimum}, ${maximum}]`);
	}
	return value;
}

function familyOf(method: string): PaperMethodFamily {
	if (method.startsWith('Belief Orig')) return 'original belief';
	if (method.startsWith('RF ')) return 'random forest';
	if (method.startsWith('Log LR')) return 'logistic regression';
	if (method.startsWith('SVC')) return 'SVC';
	if (method.startsWith('KNN')) return 'KNN';
	return fail('method.family', `unknown method family for ${method}`);
}

/**
 * Split a published method name into its base method and its input
 * configuration. Fails closed: an unrecognised trailing segment means the
 * artifact grew a configuration we have not reasoned about, and silently
 * bucketing it would be exactly the slice error this figure exists to prevent.
 */
function sliceOf(method: string, context: string): { slice: PaperMethodSlice; base: string } {
	const cut = method.lastIndexOf(' - ');
	if (cut < 1) fail(context, `method name carries no " - <configuration>" segment: ${method}`);
	const suffix = method.slice(cut + 3);
	const slice = PAPER_METHOD_SLICES.find((candidate) => candidate === suffix);
	if (!slice) fail(context, `unknown input configuration "${suffix}" in ${method}`);
	return { slice, base: method.slice(0, cut) };
}

function parseMethod(value: unknown, context: string): PaperMethodRow {
	const obj = record(value, context);
	exactKeys(
		obj,
		[
			'method_id',
			'method',
			'table_id',
			'notebook_cell_index',
			'row',
			'fold_count',
			'fold_mean_trapezoidal_pr_auc',
			'fold_population_sd'
		],
		context
	);
	const tableId = string(obj.table_id, `${context}.table_id`);
	if (tableId !== 'paper_table_6' && tableId !== 'paper_not_table_6') {
		fail(`${context}.table_id`, 'expected paper_table_6 or paper_not_table_6');
	}
	const cell = integer(obj.notebook_cell_index, `${context}.notebook_cell_index`);
	const expectedCell = tableId === 'paper_table_6' ? 47 : 48;
	if (cell !== expectedCell) fail(`${context}.notebook_cell_index`, `must be ${expectedCell}`);
	const row = integer(obj.row, `${context}.row`, 1);
	const expectedId = `${tableId}:${String(row).padStart(2, '0')}`;
	const methodId = string(obj.method_id, `${context}.method_id`);
	if (methodId !== expectedId) fail(`${context}.method_id`, `must be ${expectedId}`);
	const folds = integer(obj.fold_count, `${context}.fold_count`, 1);
	if (folds !== 10) fail(`${context}.fold_count`, 'must be 10');
	const method = string(obj.method, `${context}.method`);
	const configuration = sliceOf(method, `${context}.method`);
	return {
		method_id: methodId,
		method,
		family: familyOf(method),
		slice: configuration.slice,
		base_method: configuration.base,
		table_id: tableId,
		notebook_cell_index: cell as 47 | 48,
		row,
		fold_count: 10,
		fold_mean_trapezoidal_pr_auc: finite(
			obj.fold_mean_trapezoidal_pr_auc,
			`${context}.fold_mean_trapezoidal_pr_auc`,
			0,
			1
		),
		fold_population_sd: finite(obj.fold_population_sd, `${context}.fold_population_sd`, 0, 1)
	};
}

function source(value: unknown): PaperMethodSource {
	const obj = record(value, 'artifact.source');
	exactKeys(
		obj,
		['repository', 'commit', 'notebook_path', 'notebook_sha256', 'executed_output_cells'],
		'artifact.source'
	);
	if (obj.repository !== 'https://github.com/sorgerlab/indra_assembly_paper') {
		fail('artifact.source.repository', 'unexpected repository');
	}
	if (obj.commit !== '63abdf1274d2f5534ed822585775031712916c83') {
		fail('artifact.source.commit', 'unexpected paper commit');
	}
	if (obj.notebook_path !== 'notebooks/Training Belief ML Models.ipynb') {
		fail('artifact.source.notebook_path', 'unexpected notebook path');
	}
	if (obj.notebook_sha256 !== '3bd1a684fdc33c0b4963dd3e0c834c5420d90703112a91773f43415e1125ad26') {
		fail('artifact.source.notebook_sha256', 'unexpected notebook digest');
	}
	if (
		!Array.isArray(obj.executed_output_cells) ||
		obj.executed_output_cells.length !== 2 ||
		obj.executed_output_cells[0] !== 47 ||
		obj.executed_output_cells[1] !== 48
	) {
		fail('artifact.source.executed_output_cells', 'must be exactly [47, 48]');
	}
	return {
		repository: obj.repository,
		commit: obj.commit,
		notebook_path: obj.notebook_path,
		notebook_sha256: obj.notebook_sha256,
		executed_output_cells: [47, 48]
	};
}

function metricContract(value: unknown): PaperMethodLandscape['metric_contract'] {
	const obj = record(value, 'artifact.metric_contract');
	exactKeys(
		obj,
		[
			'unit',
			'positive_class',
			'per_fold_metric',
			'summary',
			'uncertainty_field',
			'uncertainty_is_confidence_interval',
			'metric_is_pooled_average_precision',
			'directly_comparable_to_pair_error_f1'
		],
		'artifact.metric_contract'
	);
	const expected = {
		unit: 'assembled_statement',
		positive_class: 'correct assembled statement',
		per_fold_metric: 'sklearn precision_recall_curve followed by auc(recall, precision)',
		summary: 'arithmetic mean over 10 cross-validation folds',
		uncertainty_field: 'population standard deviation over the 10 folds',
		uncertainty_is_confidence_interval: false,
		metric_is_pooled_average_precision: false,
		directly_comparable_to_pair_error_f1: false
	} as const;
	for (const [key, wanted] of Object.entries(expected)) {
		if (obj[key] !== wanted) fail(`artifact.metric_contract.${key}`, `must be ${String(wanted)}`);
	}
	return expected;
}

export function validatePaperMethodLandscape(
	raw: unknown,
	artifactSha256: string
): PaperMethodLandscape {
	if (artifactSha256 !== PAPER_METHOD_EXPECTED_SHA256) {
		fail('artifact.sha256', `expected ${PAPER_METHOD_EXPECTED_SHA256}`);
	}
	const obj = record(raw, 'artifact');
	exactKeys(
		obj,
		['schema_version', 'artifact_kind', 'source', 'metric_contract', 'method_count', 'methods', 'tables'],
		'artifact'
	);
	if (obj.schema_version !== PAPER_METHOD_SCHEMA_VERSION) fail('artifact.schema_version', 'must be 1');
	if (obj.artifact_kind !== PAPER_METHOD_ARTIFACT_KIND) {
		fail('artifact.artifact_kind', `must be ${PAPER_METHOD_ARTIFACT_KIND}`);
	}
	if (obj.method_count !== 59) fail('artifact.method_count', 'must be 59');
	if (!Array.isArray(obj.methods) || obj.methods.length !== 59) {
		fail('artifact.methods', 'must contain exactly 59 rows');
	}
	const methods = obj.methods.map((value, index) => parseMethod(value, `artifact.methods[${index}]`));
	const ids = methods.map((method) => method.method_id);
	if (new Set(ids).size !== ids.length) fail('artifact.methods', 'method IDs must be unique');

	if (!Array.isArray(obj.tables) || obj.tables.length !== 2) {
		fail('artifact.tables', 'must contain exactly two tables');
	}
	const tableRows: unknown[] = [];
	for (const [index, rawTable] of obj.tables.entries()) {
		const context = `artifact.tables[${index}]`;
		const table = record(rawTable, context);
		exactKeys(table, ['table_id', 'notebook_cell_index', 'rows'], context);
		const expectedTableId = index === 0 ? 'paper_table_6' : 'paper_not_table_6';
		const expectedCell = index === 0 ? 47 : 48;
		const expectedRows = index === 0 ? 41 : 18;
		if (table.table_id !== expectedTableId) fail(`${context}.table_id`, `must be ${expectedTableId}`);
		if (table.notebook_cell_index !== expectedCell) {
			fail(`${context}.notebook_cell_index`, `must be ${expectedCell}`);
		}
		if (!Array.isArray(table.rows) || table.rows.length !== expectedRows) {
			fail(`${context}.rows`, `must contain ${expectedRows} rows`);
		}
		tableRows.push(...table.rows);
	}
	if (JSON.stringify(tableRows) !== JSON.stringify(obj.methods)) {
		fail('artifact.tables', 'table rows must be the exact ordered projection of methods');
	}

	const familyCounts = Object.fromEntries(
		PAPER_METHOD_FAMILIES.map((family) => [
			family,
			methods.filter((method) => method.family === family).length
		])
	) as Record<PaperMethodFamily, number>;
	const expectedFamilyCounts: Record<PaperMethodFamily, number> = {
		'original belief': 3,
		'random forest': 20,
		'logistic regression': 20,
		SVC: 8,
		KNN: 8
	};
	for (const family of PAPER_METHOD_FAMILIES) {
		if (familyCounts[family] !== expectedFamilyCounts[family]) {
			fail(`artifact.methods.${family}`, `expected ${expectedFamilyCounts[family]} methods`);
		}
	}
	const sliceCounts = Object.fromEntries(
		PAPER_METHOD_SLICES.map((slice) => [
			slice,
			methods.filter((method) => method.slice === slice).length
		])
	) as Record<PaperMethodSlice, number>;
	for (const slice of PAPER_METHOD_SLICES) {
		if (sliceCounts[slice] !== PAPER_METHOD_SLICE_COUNTS[slice]) {
			fail(
				`artifact.methods.${slice}`,
				`expected ${PAPER_METHOD_SLICE_COUNTS[slice]} rows in this input configuration`
			);
		}
	}

	const baseline = methods.find((method) => method.method_id === 'paper_table_6:01');
	if (!baseline || baseline.method !== 'Belief Orig - readers') {
		fail('artifact.methods', 'published original-belief reader anchor is missing');
	}
	const best = [...methods].sort(rankPublished)[0];
	const comparable = methods.filter((method) => method.slice === PAPER_COMPARABLE_SLICE);
	const comparableBest = [...comparable].sort(rankPublished)[0];
	return {
		method_count: 59,
		methods,
		family_counts: familyCounts,
		slice_counts: sliceCounts,
		baseline,
		best,
		comparable_best: comparableBest,
		source: source(obj.source),
		metric_contract: metricContract(obj.metric_contract)
	};
}

/**
 * Deterministic published-row ranking: fold mean descending, then the tighter
 * fold SD, then method_id. Total on this artifact, so `best` and
 * `comparable_best` never depend on the artifact's row order.
 */
export function rankPublished(a: PaperMethodRow, b: PaperMethodRow): number {
	return (
		b.fold_mean_trapezoidal_pr_auc - a.fold_mean_trapezoidal_pr_auc ||
		a.fold_population_sd - b.fold_population_sd ||
		a.method_id.localeCompare(b.method_id)
	);
}
