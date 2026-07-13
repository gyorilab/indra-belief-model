/** Pure-function gate for calibration schema/provenance/compatibility selection. */
import { readFileSync } from 'node:fs';
import {
	calibrationArtifactConsistency,
	calibrationArmLabel,
	calibrationCompatibility,
	canonicalCalibrationArm,
	classifyCalibrationEvaluation,
	commonCalibrationArm,
	metricsContractFingerprint,
	readerConfigurationIdentity,
	selectCalibrationPredecessor
} from '../src/lib/data/calibration.ts';
import { compareRunRecency } from '../src/lib/data/runs.ts';

let failures = 0;

function eq(got, want, label) {
	if (got !== want) {
		failures++;
		console.error(`FAIL ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
	}
}

function ok(got, label) {
	if (!got) {
		failures++;
		console.error(`FAIL ${label}`);
	}
}

const SHA = {
	corpusA: 'a'.repeat(64),
	corpusB: 'b'.repeat(64),
	goldFit: 'c'.repeat(64),
	goldValidation: 'd'.repeat(64),
	evalA: 'e'.repeat(64),
	evalB: 'f'.repeat(64)
};

function provenance(overrides = {}) {
	return {
		corpus_sha256: SHA.corpusA,
		gold_sha256: SHA.goldFit,
		evaluation_set_sha256: SHA.evalA,
		...overrides
	};
}

function arm(ece = 0.1, brier = 0.2) {
	return { ece, brier };
}

function configId(model = 'reader', prompt = '1'.repeat(64)) {
	return `${model}@prompt-sha256:${prompt}`;
}

function profile({
	model = 'reader',
	configuration = configId(model),
	profileId = `${model}@fit`,
	fitGold = 'fit-gold.jsonl',
	fitGoldSha = SHA.goldFit,
	validationGold = 'validation-gold.jsonl',
	validationGoldSha = SHA.goldValidation,
	validationRun = 'validation-run.jsonl',
	validationResult = 'pass',
	deploymentStatus = 'enabled'
} = {}) {
	return {
		profile_id: profileId,
		reader_configuration: configuration,
		reader_model: model,
		deployment_status: deploymentStatus,
		fit_run: 'fit-run.jsonl',
		fit_gold: fitGold,
		fit_gold_sha256: fitGoldSha,
		fit_unique_pairs: 10,
		gold_rule: 'exact pair',
		validation: {
			result: validationResult,
			gold: validationGold,
			gold_sha256: validationGoldSha,
			run: validationRun,
			gate: '4/4'
		},
		confusion: { cc: 4, ci: 1, ic: 1, ii: 4 },
		sensitivity: 0.8,
		false_positive_rate: 0.2,
		specificity: 0.8,
		miss_rate: 0.2,
		log_lr_confirm: 1,
		log_lr_reject: -1,
		prior_correct: 0.5,
		prior_logodds: 0
	};
}

function run({
	id = 'run-a',
	model = 'reader',
	substrate = 'corpus-a.json',
	sourceRun = 'fit-run.jsonl',
	generated = '2026-07-05',
	finished = '2026-07-05T00:00:00Z',
	exportSchema = 8,
	prompt = '1'.repeat(64),
	configuration = configId(model, prompt),
	configurationStatus = 'identified',
	profileId = `${model}@fit`,
	profileStatus = 'available',
	profileOverrides = {},
	prov = provenance(),
	exportDir = `/exports/${id}`
} = {}) {
	const readerConfiguration = {
		status: configurationStatus,
		id: configurationStatus === 'identified' ? configuration : null,
		model,
		prompt_sha256: configurationStatus === 'identified' ? prompt : null
	};
	const fitted = profile({ model, configuration, profileId, ...profileOverrides });
	return {
		run_id: id,
		export_dir: exportDir,
		model,
		generated_date: generated,
		export_schema_version: exportSchema,
		counts: {},
		bucket_counts: {},
		source_run: sourceRun,
		substrate,
		gold_coverage: null,
		status: null,
		started_at: null,
		finished_at: finished,
		provenance: prov,
		soft_calibration: {
			status: profileStatus,
			model,
			reader_configuration: readerConfiguration,
			soft_weights: profileStatus === 'available' ? fitted : null,
			...(profileStatus === 'unavailable' ? { reason: 'no approved profile' } : {})
		}
	};
}

function metrics({
	schema = 3,
	runId = 'run-a',
	model = 'reader',
	gold = 'fit-gold.jsonl',
	basis = {},
	stmtArms = { hard: arm(0.2), parametric: arm(0.15), soft: arm(0.05) },
	evArms = { score: arm(0.1) },
	configuration = configId(model),
	profileId = `${model}@fit`,
	profileStatus = 'available',
	profileOverrides = {},
	prov = provenance()
} = {}) {
	const fitted = profile({ model, configuration, profileId, ...profileOverrides });
	return {
		schema_version: schema,
		run_id: runId,
		model,
		generated_date: '2026-07-05',
		provenance: prov,
		metrics_basis: {
			bins: 'BINS_8',
			tau: 0.5,
			join: 'exact pair',
			soft_calibration:
				profileStatus === 'available'
					? {
							status: 'available',
							profile_id: profileId,
							reader_configuration: configuration,
							fit_run: fitted.fit_run,
							fit_gold: fitted.fit_gold,
							fit_gold_sha256: fitted.fit_gold_sha256,
							fit_unique_pairs: fitted.fit_unique_pairs,
							gold_rule: fitted.gold_rule,
							deployment_status: fitted.deployment_status,
							validation: fitted.validation
						}
					: {
							status: 'unavailable',
							reader_configuration: { status: 'identified', id: configuration }
						},
			...basis
		},
		gold: gold ? { source: gold, covered: 10, total: 10 } : null,
		tiers: {
			ev: { status: 'available', arms: evArms },
			stmt: { status: 'available', arms: stmtArms }
		}
	};
}

function pair({
	id = 'run-a',
	model = 'reader',
	prompt = '1'.repeat(64),
	profileId = `${model}@fit`,
	prov = provenance(),
	gold = 'fit-gold.jsonl',
	sourceRun = 'fit-run.jsonl',
	runOverrides = {},
	metricsOverrides = {}
} = {}) {
	const configuration = configId(model, prompt);
	return {
		r: run({
			id,
			model,
			prompt,
			configuration,
			profileId,
			prov,
			sourceRun,
			...runOverrides
		}),
		m: metrics({
			runId: id,
			model,
			configuration,
			profileId,
			prov,
			gold,
			profileOverrides: runOverrides.profileOverrides ?? {},
			...metricsOverrides
		})
	};
}

// Schema selection: v2 soft is historical; unknown future schemas fail closed.
const v2 = metrics({ schema: 2 });
const v3 = metrics({ schema: 3 });
const v4 = metrics({ schema: 4 });
eq(canonicalCalibrationArm(v2, 'stmt'), 'hard', 'v2 canonical statement arm is hard');
eq(canonicalCalibrationArm(v3, 'stmt'), 'soft', 'v3 canonical statement arm is hybrid');
eq(canonicalCalibrationArm(v4, 'stmt'), null, 'unknown future schema is not promoted');
eq(commonCalibrationArm(v2, v3, 'stmt'), null, 'v2 soft never mixes with v3 hybrid');
eq(commonCalibrationArm(v2, v2, 'stmt'), 'hard', 'two v2 contracts share hard arm');
ok(calibrationArmLabel(v2, 'soft').includes('legacy soft survival'), 'v2 soft is visibly legacy');
ok(calibrationArmLabel(v3, 'soft').includes('hybrid log-odds'), 'v3 soft is labelled hybrid');

// End-to-end schema/run/model/profile/provenance consistency.
const base = pair();
eq(calibrationArtifactConsistency(base.r, base.m).valid, true, 'v8/v3 artifact seam valid');
const legacyRun = run({ exportSchema: 7, profileStatus: 'unavailable' });
const legacyMetrics = metrics({ schema: 2 });
eq(calibrationArtifactConsistency(legacyRun, legacyMetrics).valid, true, 'v7/v2 seam valid');
eq(
	calibrationArtifactConsistency({ ...base.r, export_schema_version: 7 }, base.m).valid,
	false,
	'v7/v3 schema mismatch rejected'
);
eq(
	calibrationArtifactConsistency(base.r, { ...base.m, model: 'other' }).valid,
	false,
	'metrics/export model mismatch rejected'
);
eq(
	calibrationArtifactConsistency(base.r, { ...base.m, run_id: null }).valid,
	false,
	'modern metrics missing run id rejected'
);
eq(
	calibrationArtifactConsistency(base.r, {
		...base.m,
		provenance: provenance({ gold_sha256: SHA.goldValidation })
	}).valid,
	false,
	'export/metrics digest mismatch rejected'
);
const tamperedProfileMetrics = structuredClone(base.m);
tamperedProfileMetrics.metrics_basis.soft_calibration.validation.gold_sha256 = SHA.goldFit;
const tamperedProfileResult = calibrationArtifactConsistency(base.r, tamperedProfileMetrics);
eq(tamperedProfileResult.valid, false, 'tampered metrics profile provenance rejected');
ok(
	tamperedProfileResult.reasons.some((reason) => reason.includes('fit/validation profile provenance')),
	'tampered profile provenance reason exposed'
);
const noProfile = pair({
	runOverrides: { profileStatus: 'unavailable' },
	metricsOverrides: { profileStatus: 'unavailable' }
});
noProfile.m.tiers.stmt.arms.soft = { status: 'unavailable', reason: 'no profile' };
eq(calibrationArtifactConsistency(noProfile.r, noProfile.m).valid, true, 'unfitted v3 hard fallback valid');
noProfile.m.tiers.stmt.arms.soft = arm(0.05);
eq(
	calibrationArtifactConsistency(noProfile.r, noProfile.m).valid,
	false,
	'unavailable profile cannot back a realized hybrid arm'
);
const noGoldProvenance = provenance({ gold_sha256: null, evaluation_set_sha256: null });
const noGold = pair({
	id: 'no-gold',
	prov: noGoldProvenance,
	gold: null,
	runOverrides: { profileStatus: 'unavailable' },
	metricsOverrides: {
		profileStatus: 'unavailable',
		stmtArms: {
			hard: { status: 'unavailable', reason: 'gold absent' },
			parametric: { status: 'unavailable', reason: 'gold absent' },
			soft: { status: 'unavailable', reason: 'gold absent' }
		},
		evArms: { score: { status: 'unavailable', reason: 'gold absent' } }
	}
});
eq(
	calibrationArtifactConsistency(noGold.r, noGold.m).valid,
	true,
	'no-gold v3 artifact permits explicitly null gold and evaluation provenance'
);
eq(
	calibrationCompatibility(noGold.r, noGold.r, noGold.m, noGold.m).compatible,
	false,
	'no-gold artifact remains ineligible for a cross-run calibration delta'
);

// Evaluation classification: fit-set precedence, exact recorded validation, and fail-closed digests.
eq(classifyCalibrationEvaluation(base.r, base.m).kind, 'in-sample-fit', 'fit gold is in-sample');
const validation = pair({
	id: 'validation-run',
	prov: provenance({ gold_sha256: SHA.goldValidation }),
	gold: 'validation-gold.jsonl',
	sourceRun: 'validation-run.jsonl'
});
eq(
	classifyCalibrationEvaluation(validation.r, validation.m).kind,
	'independent-validation-pass',
	'exact validation run and gold classified as independent pass'
);
eq(
	classifyCalibrationEvaluation(
		{ ...validation.r, source_run: 'different-run.jsonl' },
		validation.m
	).kind,
	'out-of-sample',
	'validation gold reused by another run is not the recorded gate'
);
const failedValidation = pair({
	id: 'validation-run',
	prov: provenance({ gold_sha256: SHA.goldValidation }),
	gold: 'validation-gold.jsonl',
	sourceRun: 'validation-run.jsonl',
	runOverrides: { profileOverrides: { validationResult: 'fail', deploymentStatus: 'disabled' } }
});
eq(
	classifyCalibrationEvaluation(failedValidation.r, failedValidation.m).kind,
	'independent-validation-fail',
	'failed profile gate never labels validation pass'
);
const missingFitDigest = pair({ runOverrides: { profileOverrides: { fitGoldSha: null } } });
eq(
	classifyCalibrationEvaluation(missingFitDigest.r, missingFitDigest.m).kind,
	'unprofiled',
	'modern fit classification fails closed without fit digest'
);

// Arbitrary A/B comparison may use different readers, but content + contracts must match.
const a = pair({ id: 'run-a', model: 'reader-a', profileId: 'profile-a' });
const b = pair({ id: 'run-b', model: 'reader-b', profileId: 'profile-b' });
eq(metricsContractFingerprint(a.m), metricsContractFingerprint(b.m), 'profiles excluded from metric-definition fingerprint');
const compatible = calibrationCompatibility(a.r, b.r, a.m, b.m);
eq(compatible.compatible, true, 'different exact readers can compare on identical content');
eq(compatible.tiers.ev.arm, 'score', 'compatible Tier-1 common arm');
eq(compatible.tiers.stmt.arm, 'soft', 'compatible Tier-2 common hybrid arm');

for (const [field, value] of [
	['corpus_sha256', SHA.corpusB],
	['gold_sha256', SHA.goldValidation],
	['evaluation_set_sha256', SHA.evalB]
]) {
	const changed = pair({
		id: 'run-b',
		model: 'reader-b',
		profileId: 'profile-b',
		prov: provenance({ [field]: value })
	});
	const result = calibrationCompatibility(a.r, changed.r, a.m, changed.m);
	eq(result.compatible, false, `${field} mismatch rejected`);
	ok(result.reasons.some((reason) => reason.includes(field)), `${field} reason exposed`);
}
const relocated = pair({
	id: 'run-b',
	model: 'reader-b',
	profileId: 'profile-b',
	gold: '/relocated/fit-gold.jsonl',
	runOverrides: { substrate: 'renamed-corpus.json' }
});
eq(
	calibrationCompatibility(a.r, relocated.r, a.m, relocated.m).compatible,
	true,
	'different paths with identical baked digests remain content-compatible'
);
const invalidDigest = pair({
	id: 'run-b',
	model: 'reader-b',
	profileId: 'profile-b',
	prov: provenance({ corpus_sha256: 'not-a-digest' })
});
eq(
	calibrationCompatibility(a.r, invalidDigest.r, a.m, invalidDigest.m).compatible,
	false,
	'invalid digest fails closed'
);
const wrongBasis = pair({
	id: 'run-b',
	model: 'reader-b',
	profileId: 'profile-b',
	metricsOverrides: { basis: { join: 'source hash only' } }
});
eq(
	calibrationCompatibility(a.r, wrongBasis.r, a.m, wrongBasis.m).compatible,
	false,
	'different metrics basis rejected'
);

// Predecessors are stricter than arbitrary A/B: exact config and, for soft, exact profile.
const current = pair({ id: 'current', model: 'reader', profileId: 'profile' });
current.r.finished_at = '2026-07-05T00:00:00Z';
const good = pair({ id: 'good', model: 'reader', profileId: 'profile' });
good.r.finished_at = '2026-07-04T00:00:00Z';
const changedPrompt = pair({
	id: 'changed-prompt',
	model: 'reader',
	prompt: '2'.repeat(64),
	profileId: 'profile'
});
changedPrompt.r.finished_at = '2026-07-04T12:00:00Z';
const changedProfile = pair({ id: 'changed-profile', model: 'reader', profileId: 'other-profile' });
changedProfile.r.finished_at = '2026-07-04T18:00:00Z';
const later = pair({ id: 'later', model: 'reader', profileId: 'profile' });
later.r.finished_at = '2026-07-06T00:00:00Z';
const metricsById = new Map(
	[good, changedPrompt, changedProfile, later].map(({ r, m }) => [r.run_id, m])
);
const predecessor = selectCalibrationPredecessor(
	current.r,
	[later.r, changedPrompt.r, changedProfile.r, good.r],
	current.m,
	'stmt',
	(r) => metricsById.get(r.run_id) ?? null
);
eq(predecessor?.run.run_id, 'good', 'hybrid predecessor keeps exact config and profile');
eq(readerConfigurationIdentity(changedPrompt.r) === readerConfigurationIdentity(current.r), false, 'prompt changes exact identity');
const missingConfiguration = { ...good.r, soft_calibration: { ...good.r.soft_calibration, reader_configuration: { status: 'mixed', id: null } } };
eq(
	selectCalibrationPredecessor(
		{ ...current.r, soft_calibration: missingConfiguration.soft_calibration },
		[good.r],
		current.m,
		'stmt',
		() => good.m
	),
	null,
	'mixed/missing current configuration withholds predecessor'
);
const tie = pair({ id: 'tie', model: 'reader', profileId: 'profile' });
tie.r.finished_at = good.r.finished_at;
metricsById.set('tie', tie.m);
eq(
	selectCalibrationPredecessor(
		current.r,
		[good.r, tie.r],
		current.m,
		'stmt',
		(r) => metricsById.get(r.run_id) ?? null
	),
	null,
	'equal-time predecessor tie is withheld'
);

// Run discovery ordering is permutation-invariant and never falls back to readdir order.
const datedA = run({ id: 'aaa', finished: '2026-07-05T01:00:00Z' });
const datedB = run({ id: 'bbb', finished: '2026-07-05T02:00:00Z' });
const tiedZ = run({ id: 'zzz', finished: null, generated: '2026-07-04' });
const tiedY = run({ id: 'yyy', finished: null, generated: '2026-07-04' });
const malformed = run({ id: 'bad', finished: null, generated: 'not-a-date' });
const order = (rows) => [...rows].sort(compareRunRecency).map((row) => row.run_id).join(',');
const expectedOrder = 'bbb,aaa,zzz,yyy,bad';
eq(order([tiedY, malformed, datedA, tiedZ, datedB]), expectedOrder, 'recency order uses timestamp then stable id');
eq(order([datedB, tiedZ, datedA, malformed, tiedY]), expectedOrder, 'recency order is permutation-invariant');

// Copy gate: fit diagnostics and neutral deltas must stay explicit.
const compareSource = readFileSync(new URL('../src/routes/compare/+page.svelte', import.meta.url), 'utf8');
const runSource = readFileSync(new URL('../src/routes/runs/[run_id]/+page.svelte', import.meta.url), 'utf8');
ok(!compareSource.includes('more honest'), 'compare page has no unqualified honesty winner claim');
ok(!compareSource.includes('bought honesty'), 'compare page has no causal honesty claim');
ok(compareSource.includes('in-sample diagnostic'), 'compare page labels in-sample diagnostics');
ok(runSource.includes('descriptive fit diagnostics'), 'run page surfaces fit-set provenance');
ok(!runSource.includes('· independent gate'), 'run page does not call every profile gate independent of displayed data');

if (failures > 0) {
	console.error(`\n${failures} calibration-contract assertion(s) failed`);
	process.exit(1);
}
console.log('calibration-contract: all assertions passed');
