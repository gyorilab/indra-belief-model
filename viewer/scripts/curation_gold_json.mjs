/**
 * Parity harness: reduce a curations JSONL through the viewer's canonical
 * curation domain (src/lib/data/curation.ts) and emit the gold index as JSON on
 * stdout. tests/test_curation_parity.py runs this and the Python module on the
 * SAME fixture and asserts identical output — the cross-language drift guard.
 *
 * Run: node --experimental-strip-types viewer/scripts/curation_gold_json.mjs <curations.jsonl>
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const { buildCurationIndex } = await import(
	resolve(here, '../src/lib/data/curation.ts')
);

const path = process.argv[2];
if (!path) {
	console.error('usage: curation_gold_json.mjs <curations.jsonl>');
	process.exit(2);
}

const rows = [];
for (const line of readFileSync(path, 'utf8').split('\n')) {
	if (!line.trim()) continue;
	try {
		rows.push(JSON.parse(line));
	} catch {
		/* skip */
	}
}

const index = buildCurationIndex(rows);

// Emit a stable, sorted, language-neutral view of the gold index.
const gold = {};
for (const [key, gv] of [...index.goldByKey.entries()].sort()) {
	gold[key] = {
		verdict: gv.verdict,
		n: gv.n,
		tags: gv.tags,
		curators: gv.curators,
		notes: gv.notes
	};
}
console.log(
	JSON.stringify({
		n_statements: index.nStatements,
		n_evidences: index.nEvidences,
		gold
	})
);
