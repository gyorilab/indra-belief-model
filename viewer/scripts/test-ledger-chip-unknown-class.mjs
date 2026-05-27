import assert from 'node:assert/strict';
import { createServer } from 'vite';

// D3 of deferred hypergraph: JS unit test for the ledger chip
// unknown-class contract. The chips (LedgerKindChip,
// LedgerRoleMark, LedgerScopeChip) all render a CSS class derived
// from pairedMetricKindFamily / panelApplicabilityClass / pairedLedgerRole.
// Each helper must return a stable 'unknown' sentinel for inputs
// outside the closed enumeration so the chips render an unknown-state
// chip instead of a typo'd class.

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	const mod = await server.ssrLoadModule('/src/lib/pairedMetricKinds.ts');

	// pairedMetricKindFamily(known) -> known family
	assert.equal(mod.pairedMetricKindFamily('denominator_base'), 'base');
	assert.equal(mod.pairedMetricKindFamily('arch_arch_exact_label'), 'arch-arch');
	assert.equal(mod.pairedMetricKindFamily('arch_indra_residual'), 'arch-indra');

	// pairedMetricKindFamily(unknown) -> 'unknown' sentinel
	assert.equal(mod.pairedMetricKindFamily('does_not_exist'), 'unknown');
	assert.equal(mod.pairedMetricKindFamily(''), 'unknown');
	assert.equal(mod.pairedMetricKindFamily('arch_arch_typo'), 'unknown');

	// pairedMetricKindLabel(known) -> human label
	assert.equal(mod.pairedMetricKindLabel('denominator_base'), 'denominator base');
	assert.equal(mod.pairedMetricKindLabel('integrity_gate'), 'integrity gate');

	// pairedMetricKindLabel(unknown) -> diagnostic label naming the value
	assert.match(mod.pairedMetricKindLabel('mystery_kind'), /unknown metric kind: mystery_kind/);
	assert.match(mod.pairedMetricKindLabel(''), /unknown metric kind: blank/);

	// isPairedMetricKind narrows correctly
	assert.equal(mod.isPairedMetricKind('arch_arch_resource'), true);
	assert.equal(mod.isPairedMetricKind('not_a_kind'), false);

	// panelApplicabilityClass(known) -> known class
	assert.equal(mod.panelApplicabilityClass('arch_blind'), 'arch_blind');
	assert.equal(mod.panelApplicabilityClass('paired_only'), 'paired_only');
	assert.equal(mod.panelApplicabilityClass('not_defined'), 'not_defined');

	// panelApplicabilityClass(unknown) -> 'unknown' sentinel
	assert.equal(mod.panelApplicabilityClass('mystery'), 'unknown');
	assert.equal(mod.panelApplicabilityClass(''), 'unknown');

	console.log('ledger chip unknown-class tests passed');
} finally {
	await server.close();
}
