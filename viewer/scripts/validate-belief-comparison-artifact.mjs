/** Validate one on-disk statement-belief artifact with the viewer's exact gate. */
import { readFileSync } from 'node:fs';

import { validateBeliefComparisonArtifact } from '../src/lib/data/belief-comparison.ts';

const path = process.argv[2];
if (!path) {
	console.error('usage: node --experimental-strip-types validate-belief-comparison-artifact.mjs ARTIFACT.json');
	process.exit(2);
}

let raw;
try {
	raw = JSON.parse(readFileSync(path, 'utf8'));
} catch (error) {
	console.error(`could not parse ${path}: ${String(error)}`);
	process.exit(2);
}

const validation = validateBeliefComparisonArtifact(raw);
if (validation.status !== 'available') {
	console.error(validation.reasons.join('\n'));
	process.exit(1);
}
console.log(`valid statement-belief artifact: ${validation.panels.length} panel(s)`);
