/**
 * Calibration-contract semantics shared by the run and compare surfaces.
 *
 * Metrics schema v2's `soft` arm is the historical survival-weight score.
 * Schema v3 promoted a different, configuration-specific hybrid log-odds score
 * under the same legacy JSON key. Presence of an arm is therefore not enough to
 * establish semantic compatibility: selection must be schema- and
 * contract-aware, and cross-run deltas must stay on one substrate and gold set.
 */
import type {
	ArmSlot,
	CalibrationProvenance,
	MetricArm,
	ReaderCalibrationProfile,
	RunMeta,
	RunMetrics
} from './types';

export const HYBRID_METRICS_SCHEMA = 3;
const EXPECTED_EXPORT_SCHEMA = new Map([
	[1, 6],
	[2, 7],
	[3, 8]
]);

export type CalibrationTier = 'ev' | 'stmt';
export type CalibrationContractKind = 'production-hybrid' | 'legacy-soft-survival';

export interface CalibrationTierCompatibility {
	compatible: boolean;
	arm: string | null;
	reason: string | null;
}

export interface CalibrationCompatibility {
	/** Contract compatibility before asking whether a particular tier has an arm. */
	compatible: boolean;
	reasons: string[];
	substrate: string | null;
	gold_source: string | null;
	schema_version: number | null;
	contract_kind: CalibrationContractKind | null;
	provenance: CalibrationProvenance | null;
	a_consistency: CalibrationArtifactConsistency;
	b_consistency: CalibrationArtifactConsistency;
	tiers: Record<CalibrationTier, CalibrationTierCompatibility>;
}

export interface CalibrationArtifactConsistency {
	valid: boolean;
	reasons: string[];
}

export type CalibrationEvaluationKind =
	| 'in-sample-fit'
	| 'independent-validation-pass'
	| 'independent-validation-fail'
	| 'out-of-sample'
	| 'unprofiled';

/** How the displayed metrics relate to the fitted profile. This is presentation
 * provenance, not a new metric and not an inference from performance. */
export interface CalibrationEvaluation {
	kind: CalibrationEvaluationKind;
	label: string;
	fit_run: string | null;
	fit_gold: string | null;
	fit_gold_sha256: string | null;
	validation_run: string | null;
	validation_gold: string | null;
	validation_gold_sha256: string | null;
	validation_gate: string | null;
	validation_result: 'pass' | 'fail' | null;
}

export interface CalibrationPredecessor {
	run: RunMeta;
	metrics: RunMetrics;
	arm: string;
}

type CalibrationRun = Pick<
	RunMeta,
	| 'run_id'
	| 'model'
	| 'substrate'
	| 'export_schema_version'
	| 'source_run'
	| 'provenance'
	| 'soft_calibration'
	| 'finished_at'
	| 'started_at'
	| 'generated_date'
>;

function asRecord(value: unknown): Record<string, unknown> | null {
	return value != null && typeof value === 'object' && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: null;
}

function normalizedPath(value: string | null | undefined): string | null {
	if (!value) return null;
	return value.replaceAll('\\', '/').replace(/^\.\//, '');
}

function readerProfile(run: Pick<RunMeta, 'soft_calibration'>): ReaderCalibrationProfile | null {
	const weights = run.soft_calibration?.soft_weights;
	return weights && 'profile_id' in weights ? weights : null;
}

function runProfileId(run: Pick<RunMeta, 'soft_calibration'>): string | null {
	return readerProfile(run)?.profile_id ?? null;
}

/** Exact scorer identity persisted in export metadata. Unknown/mixed identities
 * deliberately have no fallback to model family. */
export function readerConfigurationIdentity(
	run: Pick<RunMeta, 'model' | 'soft_calibration'>
): string | null {
	const config = run.soft_calibration?.reader_configuration;
	if (!config || config.status !== 'identified') return null;
	if (config.id) return config.id;
	if (config.model && config.prompt_sha256) {
		return `${config.model}@prompt-sha256:${config.prompt_sha256.toLowerCase()}`;
	}
	return null;
}

function metricsSoftCalibration(metrics: RunMetrics): Record<string, unknown> | null {
	return asRecord(asRecord(metrics.metrics_basis)?.soft_calibration);
}

function metricsConfigurationIdentity(metrics: RunMetrics): string | null {
	const raw = metricsSoftCalibration(metrics)?.reader_configuration;
	if (typeof raw === 'string' && raw) return raw;
	const config = asRecord(raw);
	if (!config) return null;
	if (typeof config.id === 'string' && config.id) return config.id;
	if (typeof config.model === 'string' && typeof config.prompt_sha256 === 'string') {
		return `${config.model}@prompt-sha256:${config.prompt_sha256.toLowerCase()}`;
	}
	return null;
}

function metricsProfileId(metrics: RunMetrics): string | null {
	const value = metricsSoftCalibration(metrics)?.profile_id;
	return typeof value === 'string' && value ? value : null;
}

function metricsProfileProvenance(metrics: RunMetrics): Record<string, unknown> {
	const soft = metricsSoftCalibration(metrics) ?? {};
	const validation = asRecord(soft.validation) ?? {};
	return {
		fit_run: normalizedPath(typeof soft.fit_run === 'string' ? soft.fit_run : null),
		fit_gold: normalizedPath(typeof soft.fit_gold === 'string' ? soft.fit_gold : null),
		fit_gold_sha256: digest(soft.fit_gold_sha256),
		fit_unique_pairs: typeof soft.fit_unique_pairs === 'number' ? soft.fit_unique_pairs : null,
		gold_rule: typeof soft.gold_rule === 'string' ? soft.gold_rule : null,
		deployment_status:
			typeof soft.deployment_status === 'string' ? soft.deployment_status : null,
		validation: {
			result: typeof validation.result === 'string' ? validation.result : null,
			gold: normalizedPath(typeof validation.gold === 'string' ? validation.gold : null),
			gold_sha256: digest(validation.gold_sha256),
			run: normalizedPath(typeof validation.run === 'string' ? validation.run : null),
			gate: typeof validation.gate === 'string' ? validation.gate : null
		}
	};
}

function exportProfileProvenance(profile: ReaderCalibrationProfile): Record<string, unknown> {
	return {
		fit_run: normalizedPath(profile.fit_run),
		fit_gold: normalizedPath(profile.fit_gold),
		fit_gold_sha256: digest(profile.fit_gold_sha256),
		fit_unique_pairs: profile.fit_unique_pairs,
		gold_rule: profile.gold_rule,
		deployment_status: profile.deployment_status ?? null,
		validation: {
			result: profile.validation?.result ?? null,
			gold: normalizedPath(profile.validation?.gold),
			gold_sha256: digest(profile.validation?.gold_sha256),
			run: normalizedPath(profile.validation?.run),
			gate: profile.validation?.gate ?? null
		}
	};
}

function digest(value: unknown): string | null {
	return typeof value === 'string' && /^[a-f0-9]{64}$/i.test(value)
		? value.toLowerCase()
		: null;
}

function normalizedProvenance(value: CalibrationProvenance | null | undefined): CalibrationProvenance | null {
	const corpus = digest(value?.corpus_sha256);
	const gold = digest(value?.gold_sha256);
	const evaluation = digest(value?.evaluation_set_sha256);
	return corpus && gold && evaluation
		? { corpus_sha256: corpus, gold_sha256: gold, evaluation_set_sha256: evaluation }
		: null;
}

export function classifyCalibrationEvaluation(
	run: Pick<RunMeta, 'source_run' | 'soft_calibration'>,
	metrics: RunMetrics | null
): CalibrationEvaluation {
	const profile = readerProfile(run);
	const evalGold = normalizedPath(metrics?.gold?.source);
	const fitGold = normalizedPath(profile?.fit_gold);
	const fitRun = normalizedPath(profile?.fit_run);
	const validationGold = normalizedPath(profile?.validation?.gold);
	const validationRun = normalizedPath(profile?.validation?.run);
	const sourceRun = normalizedPath(run.source_run);
	const evaluationGoldDigest = digest(metrics?.provenance?.gold_sha256);
	const fitGoldDigest = digest(profile?.fit_gold_sha256);
	const validationGoldDigest = digest(profile?.validation?.gold_sha256);
	const modern = metrics?.schema_version === HYBRID_METRICS_SCHEMA;
	const isFitGold = modern
		? evaluationGoldDigest != null && fitGoldDigest != null && evaluationGoldDigest === fitGoldDigest
		: evalGold != null && fitGold != null && evalGold === fitGold;
	const isValidationGold = modern
		? evaluationGoldDigest != null &&
			validationGoldDigest != null &&
			evaluationGoldDigest === validationGoldDigest
		: evalGold != null && validationGold === evalGold;
	const base = {
		fit_run: fitRun,
		fit_gold: fitGold,
		fit_gold_sha256: fitGoldDigest,
		validation_run: validationRun,
		validation_gold: validationGold,
		validation_gold_sha256: validationGoldDigest,
		validation_gate: profile?.validation?.gate ?? null,
		validation_result: profile?.validation?.result ?? null
	};

	// Fit-set identity takes precedence even if malformed provenance also names it
	// as a validation set: it is never independent of its own fit.
	if (profile && isFitGold) {
		return { kind: 'in-sample-fit', label: 'in-sample fit diagnostic', ...base };
	}
	if (
		profile &&
		isValidationGold &&
		validationRun != null &&
		validationRun === sourceRun
	) {
		const pass =
			profile.validation?.result === 'pass' && profile.deployment_status !== 'disabled';
		return {
			kind: pass ? 'independent-validation-pass' : 'independent-validation-fail',
			label: pass ? 'independent validation · PASS' : 'independent validation · FAIL',
			...base
		};
	}
	if (
		profile &&
		(modern
			? evaluationGoldDigest != null && fitGoldDigest != null && evaluationGoldDigest !== fitGoldDigest
			: evalGold != null && fitGold != null && evalGold !== fitGold)
	) {
		return {
			kind: 'out-of-sample',
			label: 'out-of-sample diagnostic · not the recorded validation gate',
			...base
		};
	}
	return { kind: 'unprofiled', label: 'no fitted-profile evaluation provenance', ...base };
}

export function metricArm(
	metrics: RunMetrics | null,
	tier: CalibrationTier,
	arm: string
): MetricArm | null {
	const block = metrics?.tiers?.[tier];
	if (!block || block.status !== 'available') return null;
	const slot: ArmSlot | undefined = block.arms[arm];
	return slot && !('status' in slot) ? slot : null;
}

export function calibrationContractKind(metrics: RunMetrics): CalibrationContractKind | null {
	if (metrics.schema_version === HYBRID_METRICS_SCHEMA) return 'production-hybrid';
	if (metrics.schema_version === 1 || metrics.schema_version === 2) {
		return 'legacy-soft-survival';
	}
	return null;
}

export function calibrationContractLabel(metrics: RunMetrics | null): string {
	if (!metrics) return 'no metrics contract';
	const kind = calibrationContractKind(metrics);
	if (kind === 'production-hybrid') {
		return `metrics schema v${metrics.schema_version} · production hybrid log-odds contract`;
	}
	if (kind === 'legacy-soft-survival') {
		return `metrics schema v${metrics.schema_version} · legacy soft-survival contract`;
	}
	return `unsupported metrics schema v${metrics.schema_version}`;
}

export function calibrationArmLabel(metrics: RunMetrics | null, arm: string): string {
	if (arm === 'score') return 'per-evidence score';
	if (arm === 'hard') {
		return metrics && metrics.schema_version === HYBRID_METRICS_SCHEMA
			? 'hard gate (fallback)'
			: 'hard gate';
	}
	if (arm === 'parametric') return 'parametric (ablation)';
	if (arm === 'soft') {
		return metrics && metrics.schema_version === HYBRID_METRICS_SCHEMA
			? 'calibrated hybrid log-odds'
			: `legacy soft survival${metrics ? ` (schema v${metrics.schema_version})` : ''}`;
	}
	return arm;
}

/** The arm safe to promote as canonical for one metrics document.
 *
 * Crucially, a v2 `soft` slot is never selected as the current hybrid arm. The
 * historical value can still be displayed with an explicit legacy label, while
 * the canonical v2 statement headline remains the common hard gate.
 */
export function canonicalCalibrationArm(
	metrics: RunMetrics | null,
	tier: CalibrationTier
): string | null {
	if (!metrics) return null;
	if (!calibrationContractKind(metrics)) return null;
	if (tier === 'ev') return metricArm(metrics, tier, 'score') ? 'score' : null;
	if (
		metrics.schema_version === HYBRID_METRICS_SCHEMA &&
		metricArm(metrics, tier, 'soft')
	) {
		return 'soft';
	}
	return metricArm(metrics, tier, 'hard') ? 'hard' : null;
}

/** Pick one semantically common canonical arm. Parametric is an ablation, not a
 * fallback headline, and legacy-v2 soft is never mixed with the v3 hybrid. */
export function commonCalibrationArm(
	a: RunMetrics | null,
	b: RunMetrics | null,
	tier: CalibrationTier
): string | null {
	if (!a || !b || a.schema_version !== b.schema_version) return null;
	if (!calibrationContractKind(a) || !calibrationContractKind(b)) return null;
	if (tier === 'ev') {
		return metricArm(a, tier, 'score') && metricArm(b, tier, 'score') ? 'score' : null;
	}
	if (
		a.schema_version === HYBRID_METRICS_SCHEMA &&
		metricArm(a, tier, 'soft') &&
		metricArm(b, tier, 'soft')
	) {
		return 'soft';
	}
	return metricArm(a, tier, 'hard') && metricArm(b, tier, 'hard') ? 'hard' : null;
}

function stableValue(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(stableValue);
	if (value && typeof value === 'object') {
		return Object.fromEntries(
			Object.entries(value as Record<string, unknown>)
				.sort(([a], [b]) => a.localeCompare(b))
				.map(([key, child]) => [key, stableValue(child)])
		);
	}
	return value;
}

/** Fingerprint the metric definitions, excluding the reader-specific fitted
 * profile. Different readers legitimately carry different confusion matrices;
 * every other basis field is part of the comparison contract. */
export function metricsContractFingerprint(metrics: RunMetrics): string {
	const basis = { ...(metrics.metrics_basis ?? {}) } as Record<string, unknown>;
	delete basis.soft_calibration;
	return JSON.stringify({
		schema_version: metrics.schema_version,
		metrics_basis: stableValue(basis)
	});
}

/** Validate the export_meta.json ↔ metrics.json ↔ fitted-profile seam.
 * Compatibility callers expose every reason and withhold deltas on failure. */
export function calibrationArtifactConsistency(
	run: CalibrationRun,
	metrics: RunMetrics | null
): CalibrationArtifactConsistency {
	const reasons: string[] = [];
	if (!metrics) return { valid: false, reasons: ['metrics.json is absent'] };

	const expectedExport = EXPECTED_EXPORT_SCHEMA.get(metrics.schema_version);
	if (expectedExport == null) {
		reasons.push(`unsupported metrics schema v${metrics.schema_version}`);
	} else if (run.export_schema_version !== expectedExport) {
		reasons.push(
			`export schema v${run.export_schema_version ?? 'unknown'} does not match metrics schema v${metrics.schema_version} (expected export v${expectedExport})`
		);
	}
	if (!metrics.run_id || metrics.run_id !== run.run_id) {
		reasons.push(`metrics run_id ${metrics.run_id ?? 'missing'} does not match export ${run.run_id}`);
	}
	if (!metrics.model || metrics.model !== run.model) {
		reasons.push(`metrics model ${metrics.model ?? 'missing'} does not match export ${run.model}`);
	}

	if (metrics.schema_version === HYBRID_METRICS_SCHEMA) {
		const metricsCorpus = digest(metrics.provenance?.corpus_sha256);
		const exportCorpus = digest(run.provenance?.corpus_sha256);
		if (!metricsCorpus) reasons.push('metrics corpus provenance digest is missing or invalid');
		if (!exportCorpus) reasons.push('export corpus provenance digest is missing or invalid');
		if (metricsCorpus && exportCorpus && metricsCorpus !== exportCorpus) {
			reasons.push('export and metrics corpus provenance digests disagree');
		}
		for (const key of ['gold_sha256', 'evaluation_set_sha256'] as const) {
			const metricsRaw = metrics.provenance?.[key];
			const exportRaw = run.provenance?.[key];
			const metricsValue = digest(metricsRaw);
			const exportValue = digest(exportRaw);
			const required = metrics.gold != null;
			if (required && !metricsValue) reasons.push(`metrics ${key} is missing or invalid`);
			if (required && !exportValue) reasons.push(`export ${key} is missing or invalid`);
			if (!required && metricsRaw != null && !metricsValue) reasons.push(`metrics ${key} is invalid`);
			if (!required && exportRaw != null && !exportValue) reasons.push(`export ${key} is invalid`);
			if ((metricsValue ?? null) !== (exportValue ?? null)) {
				reasons.push(`export and metrics ${key} disagree`);
			}
		}

		const runSoft = run.soft_calibration;
		const metricsSoft = metricsSoftCalibration(metrics);
		const runStatus = runSoft?.status ?? null;
		const metricsStatus =
			metricsSoft?.status === 'available' || metricsSoft?.status === 'unavailable'
				? metricsSoft.status
				: null;
		if (!runStatus) reasons.push('export soft-calibration status is missing');
		if (!metricsStatus) reasons.push('metrics soft-calibration status is missing');
		if (runStatus && metricsStatus && runStatus !== metricsStatus) {
			reasons.push(
				`export soft-calibration status ${runStatus} disagrees with metrics status ${metricsStatus}`
			);
		}

		const runConfiguration = readerConfigurationIdentity(run);
		const metricsConfiguration = metricsConfigurationIdentity(metrics);
		if (runConfiguration && metricsConfiguration && runConfiguration !== metricsConfiguration) {
			reasons.push('export and metrics reader configurations disagree');
		}

		const stmt = metrics.tiers?.stmt;
		const softAvailable = metricArm(metrics, 'stmt', 'soft') != null;
		if (stmt?.status === 'available') {
			if (softAvailable && (runStatus !== 'available' || metricsStatus !== 'available')) {
				reasons.push('a realized hybrid arm has unavailable profile provenance');
			}
			if (!softAvailable && (runStatus === 'available' || metricsStatus === 'available')) {
				reasons.push('an available profile has no realized hybrid arm');
			}
		}

		if (runStatus === 'available') {
			const profile = readerProfile(run);
			if (!profile) {
				reasons.push('available hybrid provenance does not carry a reader profile');
			} else {
				if (profile.deployment_status !== 'enabled') {
					reasons.push('available hybrid profile is not deployment-enabled');
				}
				if (profile.validation?.result !== 'pass') {
					reasons.push('available hybrid profile has no passing validation result');
				}
				if (!profile.fit_run || !profile.fit_gold || !profile.validation?.run || !profile.validation?.gold) {
					reasons.push('available hybrid profile lacks fit/validation provenance');
				}
				if (!digest(profile.fit_gold_sha256) || !digest(profile.validation?.gold_sha256)) {
					reasons.push('available hybrid profile lacks content-addressed fit/validation gold provenance');
				}
				if (profile.reader_model && profile.reader_model !== run.model) {
					reasons.push(`profile reader ${profile.reader_model} does not match export ${run.model}`);
				}
				if (!runConfiguration || profile.reader_configuration !== runConfiguration) {
					reasons.push('profile reader configuration does not match the persisted run configuration');
				}
				const basisProfile = metricsProfileId(metrics);
				if (!basisProfile || basisProfile !== profile.profile_id) {
					reasons.push('export and metrics profile ids disagree');
				}
				if (
					JSON.stringify(exportProfileProvenance(profile)) !==
					JSON.stringify(metricsProfileProvenance(metrics))
				) {
					reasons.push('export and metrics fit/validation profile provenance disagree');
				}
			}
		} else if (runStatus === 'unavailable' && runSoft?.soft_weights != null) {
			reasons.push('unavailable export profile unexpectedly carries soft weights');
		}
	}

	return { valid: reasons.length === 0, reasons };
}

function tierCompatibility(
	baseCompatible: boolean,
	a: RunMetrics | null,
	b: RunMetrics | null,
	tier: CalibrationTier
): CalibrationTierCompatibility {
	if (!baseCompatible) {
		return {
			compatible: false,
			arm: null,
			reason: 'run contracts must match before an arm can be compared'
		};
	}
	const arm = commonCalibrationArm(a, b, tier);
	return arm
		? { compatible: true, arm, reason: null }
		: {
				compatible: false,
				arm: null,
				reason: `no common canonical ${tier === 'ev' ? 'per-evidence' : 'statement'} arm`
			};
}

/** Establish whether two calibration products describe the same measurable
 * substrate. This is deliberately conservative: unknown provenance is not
 * treated as matching provenance. */
export function calibrationCompatibility(
	a: CalibrationRun,
	b: CalibrationRun,
	am: RunMetrics | null,
	bm: RunMetrics | null
): CalibrationCompatibility {
	const reasons: string[] = [];
	const aConsistency = calibrationArtifactConsistency(a, am);
	const bConsistency = calibrationArtifactConsistency(b, bm);
	reasons.push(...aConsistency.reasons.map((reason) => `run A: ${reason}`));
	reasons.push(...bConsistency.reasons.map((reason) => `run B: ${reason}`));

	const substrate =
		a.substrate && a.substrate === b.substrate ? a.substrate : 'content-addressed match';

	const aGold = am?.gold?.source ?? null;
	const bGold = bm?.gold?.source ?? null;
	const goldSource = aGold && aGold === bGold ? aGold : 'content-addressed match';

	const ap = normalizedProvenance(am?.provenance);
	const bp = normalizedProvenance(bm?.provenance);
	let provenance: CalibrationProvenance | null = null;
	if (!ap || !bp) {
		reasons.push('one or both metrics products lack valid content provenance');
	} else {
		for (const key of [
			'corpus_sha256',
			'gold_sha256',
			'evaluation_set_sha256'
		] as const) {
			if (ap[key] !== bp[key]) reasons.push(`${key} differs between the runs`);
		}
		if (
			ap.corpus_sha256 === bp.corpus_sha256 &&
			ap.gold_sha256 === bp.gold_sha256 &&
			ap.evaluation_set_sha256 === bp.evaluation_set_sha256
		) {
			provenance = ap;
		}
	}

	let schemaVersion: number | null = null;
	let contractKind: CalibrationContractKind | null = null;
	if (am && bm) {
		if (am.schema_version !== bm.schema_version) {
			reasons.push(
				`different metrics schemas: v${am.schema_version} vs v${bm.schema_version}`
			);
		} else {
			schemaVersion = am.schema_version;
			contractKind = calibrationContractKind(am);
			if (!contractKind) reasons.push(`metrics schema v${am.schema_version} is unsupported`);
			if (metricsContractFingerprint(am) !== metricsContractFingerprint(bm)) {
				reasons.push('metrics basis/contract differs between the runs');
			}
		}
	}

	const compatible = reasons.length === 0;
	return {
		compatible,
		reasons,
		substrate,
		gold_source: goldSource,
		schema_version: schemaVersion,
		contract_kind: contractKind,
		provenance,
		a_consistency: aConsistency,
		b_consistency: bConsistency,
		tiers: {
			ev: tierCompatibility(compatible, am, bm, 'ev'),
			stmt: tierCompatibility(compatible, am, bm, 'stmt')
		}
	};
}

interface RunMoment {
	basis: 'run-event' | 'generated-date';
	value: number;
}

function runMoment(run: Pick<RunMeta, 'finished_at' | 'started_at' | 'generated_date'>): RunMoment | null {
	for (const event of [run.finished_at, run.started_at]) {
		if (!event) continue;
		const value = Date.parse(event);
		if (Number.isFinite(value)) return { basis: 'run-event', value };
	}
	if (run.generated_date) {
		const value = Date.parse(`${run.generated_date}T00:00:00Z`);
		if (Number.isFinite(value)) return { basis: 'generated-date', value };
	}
	return null;
}

/** Select the unique most-recent earlier run that is genuinely comparable.
 *
 * Selection is per tier and requires the same exact reader configuration, model,
 * temporal basis, content-addressed substrate/evaluation, schema, metrics contract,
 * and a common canonical arm. A hybrid trend also stays on one fitted profile.
 * Same-date/timestamp ties intentionally return null rather than using directory
 * enumeration order as invisible state.
 */
export function selectCalibrationPredecessor(
	current: RunMeta,
	candidates: RunMeta[],
	currentMetrics: RunMetrics | null,
	tier: CalibrationTier,
	metricsFor: (run: RunMeta) => RunMetrics | null
): CalibrationPredecessor | null {
	if (!currentMetrics) return null;
	const currentMoment = runMoment(current);
	if (!currentMoment) return null;
	const currentConfiguration = readerConfigurationIdentity(current);
	if (!currentConfiguration) return null;

	const eligible: Array<CalibrationPredecessor & { moment: number }> = [];
	for (const candidate of candidates) {
		if (candidate.run_id === current.run_id || candidate.model !== current.model) continue;
		if (readerConfigurationIdentity(candidate) !== currentConfiguration) continue;
		const moment = runMoment(candidate);
		if (
			!moment ||
			moment.basis !== currentMoment.basis ||
			moment.value >= currentMoment.value
		) {
			continue;
		}
		const metrics = metricsFor(candidate);
		const compatibility = calibrationCompatibility(current, candidate, currentMetrics, metrics);
		const arm = compatibility.tiers[tier].arm;
		if (!compatibility.compatible || !arm || !metrics) continue;
		if (
			arm === 'soft' &&
			(!runProfileId(current) || runProfileId(candidate) !== runProfileId(current))
		) {
			continue;
		}
		eligible.push({ run: candidate, metrics, arm, moment: moment.value });
	}

	eligible.sort((a, b) => b.moment - a.moment);
	if (!eligible.length) return null;
	if (eligible[1]?.moment === eligible[0].moment) return null;
	const { run, metrics, arm } = eligible[0];
	return { run, metrics, arm };
}
