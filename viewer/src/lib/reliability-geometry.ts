// Reliability-plot geometry shared by ReliabilityDiagram and PaperReliabilityStrip.
//
// Calibration space [0,1]×[0,1], y inverted (1 at top), with a small pad for
// stroke breathing room. px/py map calibration values into the 0..100 viewBox;
// rOf sizes a mark by its bin share (√(n/maxN)) so a sparse bin can't read as a
// confident mark; pathOf connects a series' occupied bins left→right into a curve.
// Formulas copied verbatim from the two components — a changed pixel is a fail.

export const REL_PAD = 6; // % padding inside the 0..100 box for stroke breathing room

export function px(v: number): number {
	return REL_PAD + v * (100 - 2 * REL_PAD);
}

export function py(v: number): number {
	return REL_PAD + (1 - v) * (100 - 2 * REL_PAD);
}

// radius 1.6%..5% of the box, by bin share — sparse bins read as small.
// maxBinN is a per-component runtime $derived (the shared bin-n scale), so it is
// a parameter rather than a module constant.
export function rOf(n: number, maxBinN: number): number {
	return 1.6 + 3.4 * Math.sqrt(n / maxBinN);
}

// connect a series' occupied bins left→right into a curve, so a model's
// trajectory off the diagonal reads as one line.
export function pathOf(ps: { x: number; y: number }[]): string {
	if (ps.length < 2) return '';
	return ps
		.slice()
		.sort((a, b) => a.x - b.x)
		.map((p, i) => `${i === 0 ? 'M' : 'L'}${px(p.x).toFixed(2)} ${py(p.y).toFixed(2)}`)
		.join(' ');
}
