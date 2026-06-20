<script lang="ts">
	// The frontier plate, read through a switchable x-axis (the SAME runs, re-
	// measured): COST ($/1k evidence, with a pinned "$0 free" gutter) or SIZE
	// (total params, log) — both against error-detection F1 (y). Position is the
	// encoding. On axis switch the dots MORPH horizontally (object constancy: the
	// model you were looking at slides to its new x), so "is it worth the money?"
	// and "does scale buy accuracy?" read as two views of one population.
	//
	// The Pareto front is a brick staircase (the best F1 a budget/size buys);
	// dominated runs recede to bare faint dots; the bootstrap 95% CI is a soft
	// BAND behind the staircase (uncertainty as an area, so the trace reads as a
	// crisp line through it), shown for frontier + hovered runs. Runs with no
	// x-datum on the active axis
	// (unpriced, or closed-weight with no published size) drop to an off-rail —
	// never plotted at a fabricated 0.
	import type { FrontierRun } from '$lib/data/queries';
	import { fmtCostFull } from '$lib/format';

	let {
		runs,
		axis = 'cost',
		selected = [],
		onpick
	}: {
		runs: FrontierRun[];
		axis?: 'cost' | 'size';
		selected?: string[];
		onpick?: (runId: string) => void;
	} = $props();

	const W = 760, H = 360;
	const padL = 46, padR = 132, padT = 18, padB = 46;
	const plotL = padL, plotR = W - padR, plotT = padT, plotB = H - padB;
	const plotH = plotB - plotT;
	const gutterW = 60;
	const gutterX = plotL + gutterW / 2;
	const dividerX = plotL + gutterW + 9;

	let hovered = $state<string | null>(null);

	const stripHost = (m: string) => m.replace(/^(bedrock|remote|local|google)-/, '');
	const fmtB = (v: number) => (v >= 1000 ? `${v / 1000}T` : `${v}B`);

	// ── Axis as a first-class abstraction ───────────────────────────────────────
	// Each plotted axis defines, ONCE, everything that varies between axes: the x
	// datum (null ⇒ off-rail), whether that x is an ESTIMATE (→ hollow dot — the one
	// general rule, not a cost-specific special case), the Pareto status, the per-run
	// x description, tick formatting, the $0-free gutter, and labels. Adding a third
	// axis = adding a third AxisSpec; the renderer below is axis-agnostic.
	interface AxisSpec {
		value: (r: FrontierRun) => number | null;
		estimated: (r: FrontierRun) => boolean;
		onFrontier: (r: FrontierRun) => boolean;
		dominatedBy: (r: FrontierRun) => string | null;
		describe: (r: FrontierRun) => string;
		tick: (v: number) => string;
		gutter: boolean;
		title: string;
		rail: string;
		xName: string;
	}
	const COST_AXIS: AxisSpec = {
		value: (r) => (r.cost_known && r.usd_per_1k != null ? r.usd_per_1k : null),
		estimated: (r) => r.cost_estimated,
		onFrontier: (r) => r.on_frontier_cost,
		dominatedBy: (r) => r.dominated_by_cost,
		describe: (r) =>
			(r.cost_estimated ? '~' : '') + fmtCostFull(r.usd_per_1k) + ' per 1k' + (r.cost_estimated ? ' (estimated)' : ''),
		tick: (v) => (v >= 1 ? `$${v}` : `$${v.toFixed(v >= 0.1 ? 1 : 2)}`),
		gutter: true,
		title: 'cost · USD / 1k evidence (log) · priced →',
		rail: 'unpriced (not plotted, never $0)',
		xName: 'cost'
	};
	const SIZE_AXIS: AxisSpec = {
		value: (r) => (r.size_known ? r.params_total_b : null),
		estimated: (r) => r.size_estimated,
		onFrontier: (r) => r.on_frontier_size,
		dominatedBy: (r) => r.dominated_by_size,
		describe: (r) => r.params_total_b + 'B params' + (r.size_estimated ? ' (estimated)' : ''),
		tick: (v) => fmtB(v),
		gutter: false,
		title: 'model size · total params (log) →',
		rail: 'size undisclosed (closed-weight; not plotted)',
		xName: 'model size'
	};
	const ax = $derived(axis === 'cost' ? COST_AXIS : SIZE_AXIS);

	// The renderer reads everything through the active axis spec.
	const xRaw = (r: FrontierRun) => ax.value(r);
	const onFront = (r: FrontierRun) => ax.onFrontier(r);
	const domBy = (r: FrontierRun) => ax.dominatedBy(r);
	const xEstimated = (r: FrontierRun) => ax.estimated(r);

	const plotRuns = $derived(runs.filter((r) => xRaw(r) != null));
	const offRail = $derived(runs.filter((r) => xRaw(r) == null));

	// Plot labels drop the host prefix for brevity — EXCEPT when two runs share a
	// bare model name across hosts (e.g. bedrock- vs remote- gemma-4-26b). There we
	// keep the full canonical name on BOTH so the plot never shows two
	// indistinguishable dots. (The ledger always shows full canonical names.)
	const bareNameCounts = $derived.by(() => {
		const c = new Map<string, number>();
		for (const r of plotRuns) {
			const b = stripHost(r.model);
			c.set(b, (c.get(b) ?? 0) + 1);
		}
		return c;
	});
	const isCrossHostDup = (r: FrontierRun) => (bareNameCounts.get(stripHost(r.model)) ?? 0) > 1;
	const displayName = (model: string) => {
		const bare = stripHost(model);
		return (bareNameCounts.get(bare) ?? 0) > 1 ? model : bare;
	};
	// A run carries a persistent (always-on) label when it's on the frontier OR
	// when it shares a bare name across hosts — so a same-model/different-host pair
	// (e.g. bedrock- vs remote- gemma-4-26b) is BOTH labelled, not just the winner.
	const hasPersistentLabel = (r: FrontierRun) => onFront(r) || isCrossHostDup(r);

	const posVals = $derived(plotRuns.map((r) => xRaw(r)!).filter((v) => v > 0));
	// Round the log domain OUT to the enclosing decades so the decade ticks
	// ($0.1/$1/$10/$100, or 1B/10B/…/1T) land INSIDE the plot — data fills the centre
	// with a little breathing room. A tight data-fit domain pushed the outer decade
	// ticks off-axis (behind the y-axis / off the right edge).
	const lmin = $derived(posVals.length ? Math.floor(Math.log10(Math.min(...posVals))) : 0);
	const lmax = $derived(posVals.length ? Math.ceil(Math.log10(Math.max(...posVals))) : 1);
	// free gutter only exists on the cost axis (a $0 column; size has no zero).
	const hasGutter = $derived(ax.gutter && plotRuns.some((r) => (xRaw(r) ?? -1) === 0));
	const pricedL = $derived(hasGutter ? plotL + gutterW + 20 : plotL + 12);
	const pricedR = plotR;

	// y-range from ALL plottable dots (either axis) + frontier CIs of EITHER axis,
	// so y is stable across an axis switch → the morph is pure horizontal motion.
	const yvals = $derived(
		runs.flatMap((r) => {
			const plottableEither = (r.cost_known && r.usd_per_1k != null) || r.size_known;
			if (!plottableEither) return [] as number[];
			return r.on_frontier_cost || r.on_frontier_size ? [r.err_f1_lo, r.err_f1_hi, r.err_f1] : [r.err_f1];
		})
	);
	const ylo = $derived(yvals.length ? Math.max(0, Math.min(...yvals) - 0.04) : 0);
	const yhi = $derived(yvals.length ? Math.min(1, Math.max(...yvals) + 0.04) : 1);

	function xOfVal(v: number | null): number {
		if (v == null || v <= 0) return gutterX;
		if (lmax === lmin) return (pricedL + pricedR) / 2;
		return pricedL + ((Math.log10(v) - lmin) / (lmax - lmin)) * (pricedR - pricedL);
	}
	function yOf(f1: number): number {
		const span = yhi - ylo || 1;
		return plotB - ((f1 - ylo) / span) * plotH;
	}

	// Dodge same-x runs into separate columns so collinear whiskers don't merge.
	const xByRun = $derived.by(() => {
		const m = new Map<string, number>();
		const groups = new Map<number, FrontierRun[]>();
		for (const r of plotRuns) {
			const bx = Math.round(xOfVal(xRaw(r)));
			const g = groups.get(bx) ?? groups.set(bx, []).get(bx)!;
			g.push(r);
		}
		const DODGE = 8;
		for (const [bx, rs] of groups) {
			if (rs.length === 1) {
				m.set(rs[0].run_id, bx);
				continue;
			}
			rs.sort((a, b) => b.err_f1 - a.err_f1 || a.run_id.localeCompare(b.run_id));
			const mid = (rs.length - 1) / 2;
			rs.forEach((r, i) => m.set(r.run_id, bx + (i - mid) * DODGE));
		}
		return m;
	});
	const rx = (r: FrontierRun) => xByRun.get(r.run_id) ?? xOfVal(xRaw(r));

	const xTicks = $derived.by(() => {
		if (!posVals.length) return [] as Array<{ x: number; label: string }>;
		if (lmax === lmin) return [{ x: (pricedL + pricedR) / 2, label: ax.tick(posVals[0]) }];
		const lo = Math.floor(lmin), hi = Math.ceil(lmax);
		const out: Array<{ x: number; label: string }> = [];
		for (let p = lo; p <= hi; p++) {
			const v = Math.pow(10, p);
			out.push({ x: xOfVal(v), label: ax.tick(v) });
		}
		return out;
	});
	const yTicks = $derived.by(() => {
		const out: Array<{ y: number; label: string }> = [];
		for (let t = Math.ceil(ylo * 10) / 10; t <= yhi + 1e-9; t += 0.1) {
			out.push({ y: yOf(t), label: t.toFixed(1) });
		}
		return out;
	});

	// Pareto staircase over the active axis's frontier runs (cheapest/smallest first).
	const stair = $derived.by(() => {
		const f = plotRuns
			.filter((r) => onFront(r))
			.slice()
			.sort((a, b) => (xRaw(a) ?? 0) - (xRaw(b) ?? 0));
		if (!f.length) return '';
		const pt = (r: FrontierRun) => ({ x: rx(r), y: yOf(r.err_f1) });
		let p = pt(f[0]);
		let d = `M ${plotL} ${p.y} L ${p.x} ${p.y}`;
		for (let i = 1; i < f.length; i++) {
			const n = pt(f[i]);
			d += ` L ${n.x} ${p.y} L ${n.x} ${n.y}`;
			p = n;
		}
		d += ` L ${plotR} ${p.y}`;
		return d;
	});

	// Every plotted dot's centre — labels must avoid these, not just each other.
	const dotCenters = $derived(plotRuns.map((r) => ({ x: rx(r), y: yOf(r.err_f1) })));

	// Place persistent labels (frontier + cross-host duplicates) avoiding BOTH other
	// dots AND already-placed labels: try right/left × mid/up/down anchors and pick
	// the one overlapping the fewest dots; connect with a leader. Frontier labels are
	// placed first so they get first pick of the clean 'right' slot.
	type Placed = { id: string; model: string; cx: number; cy: number; lx: number; ly: number; anchor: 'start' | 'end'; x0: number; x1: number };
	const placedLabels = $derived.by(() => {
		const labeled = plotRuns
			.filter((r) => hasPersistentLabel(r))
			.sort((a, b) => (onFront(b) ? 1 : 0) - (onFront(a) ? 1 : 0) || yOf(a.err_f1) - yOf(b.err_f1));
		const placed: Placed[] = [];
		for (const r of labeled) {
			const cx = rx(r), cy = yOf(r.err_f1);
			const model = displayName(r.model);
			const w = model.length * 5.4 + 3;
			const cands: Array<{ lx: number; ly: number; anchor: 'start' | 'end' }> = [
				{ lx: cx + 8, ly: cy, anchor: 'start' },
				{ lx: cx - 8, ly: cy, anchor: 'end' },
				{ lx: cx + 8, ly: cy + 13, anchor: 'start' },
				{ lx: cx + 8, ly: cy - 13, anchor: 'start' },
				{ lx: cx - 8, ly: cy + 13, anchor: 'end' },
				{ lx: cx - 8, ly: cy - 13, anchor: 'end' }
			];
			let best = cands[0], bestScore = Infinity, bestBox = { x0: cx + 8, x1: cx + 8 + w };
			cands.forEach((c, i) => {
				const x0 = c.anchor === 'start' ? c.lx : c.lx - w;
				const x1 = c.anchor === 'start' ? c.lx + w : c.lx;
				let score = i * 0.4; // tie-break: prefer right, then left, then offset
				for (const d of dotCenters) {
					if (Math.abs(d.x - cx) < 0.5 && Math.abs(d.y - cy) < 0.5) continue; // own dot
					if (d.x >= x0 - 4 && d.x <= x1 + 4 && Math.abs(d.y - c.ly) < 7) score += 10;
				}
				for (const p of placed) {
					if (x0 < p.x1 && x1 > p.x0 && Math.abs(c.ly - p.ly) < 11) score += 10;
				}
				if (x0 < 2 || x1 > W - 2) score += 6; // keep roughly in frame
				if (score < bestScore) { bestScore = score; best = c; bestBox = { x0, x1 }; }
			});
			placed.push({ id: r.run_id, model, cx, cy, lx: best.lx, ly: best.ly, anchor: best.anchor, x0: bestBox.x0, x1: bestBox.x1 });
		}
		return placed;
	});

	function activate(id: string) {
		onpick?.(id);
	}

	// Transient labels for hovered / selected DOMINATED dots (frontier dots already
	// carry a persistent label). Rendered LAST, on a solid paper chip, so they sit
	// on top of and cleanly occlude any persistent label or staircase behind them.
	const overlayLabels = $derived.by(() => {
		const ids = new Set([...(hovered ? [hovered] : []), ...selected]);
		const out: Array<{ id: string; x: number; y: number; model: string; w: number }> = [];
		for (const id of ids) {
			const r = plotRuns.find((x) => x.run_id === id);
			if (!r || hasPersistentLabel(r)) continue; // already labelled (frontier / cross-host dup)
			const model = displayName(r.model);
			out.push({ id, x: rx(r), y: yOf(r.err_f1), model, w: model.length * 5.4 + 6 });
		}
		return out;
	});

	const nFrontier = $derived(plotRuns.filter((r) => onFront(r)).length);
	const maxGold = $derived(plotRuns.length ? Math.max(...plotRuns.map((r) => r.n_gold)) : 0);
	const ariaLabel = $derived(
		`${ax.xName} versus error-detection F1 for ${plotRuns.length} runs; ${nFrontier} on the frontier. ` +
			`Hollow dots = estimated ${ax.xName}; 95% confidence intervals shown as soft bands; gold n=${maxGold}, so bands overlap and rank is indicative.`
	);
</script>

<figure class="frontier">
	<svg viewBox="0 0 {W} {H}" role="img" aria-label={ariaLabel}>
		{#each yTicks as t}
			<line x1={plotL} y1={t.y} x2={plotR} y2={t.y} stroke="var(--rule)" stroke-width="0.6" />
			<text x={plotL - 6} y={t.y + 3} text-anchor="end" class="tick">{t.label}</text>
		{/each}
		<text x={12} y={plotT + plotH / 2} class="axis-title" transform="rotate(-90 12 {plotT + plotH / 2})">error-detection F1</text>

		{#each xTicks as t}
			<line x1={t.x} y1={plotT} x2={t.x} y2={plotB} stroke="var(--rule)" stroke-width="0.5" opacity="0.7" />
			<text x={t.x} y={plotB + 14} text-anchor="middle" class="tick">{t.label}</text>
		{/each}

		{#if hasGutter}
			<line x1={dividerX} y1={plotT} x2={dividerX} y2={plotB} stroke="var(--ink-faint)" stroke-width="0.8" stroke-dasharray="2 3" />
			<text x={gutterX} y={plotB + 14} text-anchor="middle" class="tick gutter-lbl">$0 free</text>
		{/if}
		<text x={(pricedL + pricedR) / 2} y={plotB + 30} text-anchor="middle" class="axis-title">
			{ax.title}
		</text>

		<!-- CI as a soft BAND (uncertainty = an area), drawn BEFORE the staircase so
		     the frontier trace reads as a crisp line THROUGH the band — never a
		     second parallel line competing with it. Shown for frontier + hovered +
		     selected runs only. -->
		{#each plotRuns as r (r.run_id)}
			{@const shown = onFront(r) || hovered === r.run_id || selected.includes(r.run_id)}
			{#if shown}
				<g class="run-grp" style="transform: translate({rx(r)}px, 0)">
					<rect
						x={-3}
						y={yOf(r.err_f1_hi)}
						width={6}
						height={Math.max(1, yOf(r.err_f1_lo) - yOf(r.err_f1_hi))}
						fill="var(--ink-faint)"
						opacity={onFront(r) ? 0.16 : 0.24}
					/>
				</g>
			{/if}
		{/each}

		{#if stair}
			<path d={stair} fill="none" stroke="var(--accent)" stroke-width="1.4" stroke-linejoin="miter" opacity="0.9" />
		{/if}

		<!-- dots, on TOP of the staircase -->
		{#each plotRuns as r (r.run_id)}
			{@const cx = rx(r)}
			{@const cy = yOf(r.err_f1)}
			{@const sel = selected.includes(r.run_id)}
			{@const front = onFront(r)}
			{@const estd = xEstimated(r)}
			<g class="run-grp" style="transform: translate({cx}px, 0)">
				{#if sel}
					<circle cx={0} {cy} r={front ? 8 : 6.5} fill="none" stroke="var(--accent)" stroke-width="1.4" />
				{/if}
				<g
					role="button"
					tabindex="0"
					aria-label="{r.model}: error-F1 {r.err_f1.toFixed(2)}, {ax.describe(r)}{front
						? ', on frontier'
						: domBy(r)
							? ', dominated by ' + domBy(r)
							: ''}"
					onmouseenter={() => (hovered = r.run_id)}
					onmouseleave={() => (hovered = null)}
					onfocus={() => (hovered = r.run_id)}
					onblur={() => (hovered = null)}
					onclick={() => activate(r.run_id)}
					onkeydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') {
							e.preventDefault();
							activate(r.run_id);
						}
					}}
				>
					<circle
						cx={0}
						{cy}
						r={front ? 5 : 3}
						fill={estd ? 'var(--paper)' : front ? 'var(--ink)' : 'var(--ink-faint)'}
						stroke={estd ? (front ? 'var(--ink)' : 'var(--ink-faint)') : 'none'}
						stroke-width={estd ? 1.6 : 0}
						opacity={front ? 1 : 0.5}
						class="dot"
					/>
				</g>
			</g>
		{/each}

		{#each placedLabels as p (p.id)}
			<line x1={p.cx} y1={p.cy} x2={p.anchor === 'start' ? p.lx - 2 : p.lx + 2} y2={p.ly} stroke="var(--ink-faint)" stroke-width="0.6" opacity="0.55" />
			<text x={p.lx} y={p.ly + 3} text-anchor={p.anchor} class="run-lbl frontier">{p.model}</text>
		{/each}

		<!-- top-most hover/selection labels for dominated dots: a paper chip so they
		     occlude the persistent frontier labels they may overlap -->
		{#each overlayLabels as o (o.id)}
			<g transform="translate({o.x}, 0)">
				<rect x={6} y={o.y - 13} width={o.w} height={14} fill="var(--paper)" />
				<text x={9} y={o.y - 3} class="run-lbl hov">{o.model}</text>
			</g>
		{/each}
	</svg>

	{#if offRail.length}
		<p class="offrail">{ax.rail}: {offRail.map((r) => stripHost(r.model)).join(', ')}</p>
	{/if}
</figure>

<style>
	.frontier {
		margin: 0;
	}
	.frontier svg {
		display: block;
		width: 100%;
		height: auto;
		overflow: visible;
	}
	.tick {
		font-family: var(--mono);
		font-size: 8.5px;
		fill: var(--ink-faint);
	}
	.gutter-lbl {
		fill: var(--ink-muted);
	}
	.axis-title {
		font-family: var(--mono);
		font-size: 8.5px;
		fill: var(--ink-muted);
		letter-spacing: 0.03em;
	}
	.run-lbl {
		font-family: var(--mono);
		font-size: 9px;
		fill: var(--ink-muted);
		paint-order: stroke;
		stroke: var(--paper);
		stroke-width: 2.5px;
		stroke-linejoin: round;
	}
	.run-lbl.frontier {
		fill: var(--ink);
	}
	.run-lbl.hov {
		fill: var(--ink);
	}
	.dot {
		cursor: pointer;
		transition: r 120ms ease-out;
	}
	/* object constancy: the dot/whisker/label slide to their new x on axis switch */
	.run-grp {
		transition: transform 200ms ease-out;
	}
	g[role='button'] {
		outline: none;
	}
	g[role='button']:focus-visible .dot {
		stroke: var(--accent);
		stroke-width: 1.5;
	}
	.offrail {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
		margin: 0.4rem 0 0;
	}
	@media (prefers-reduced-motion: reduce) {
		.run-grp {
			transition: none;
		}
	}
</style>
