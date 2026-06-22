<!--
  ConfusionMosaic (E5) — 2×2 vs gold, cell area ∝ count. Reuses the /compare
  mosaic geometry (rows weighted by row total, cells by count) but in the
  run-on-its-own SINGLE palette: this is not an A/B contrast, it is one model's
  hits and misses. The imbalance is geometry — a tiny "errors caught" cell next
  to a vast "agree" cell tells the precision/recall story without a number.

  The confusion AXIS differs by tier, so labels are passed in:
    · Tier-1 (positive=correct):  rows gold✓/gold✗,  cols pred✓/pred✗
    · Tier-2 (positive=error):    same grid, read on the error axis
  We always render the gold-correct row on top. Counts are served byte-exact
  from metrics.json (G4).
-->
<script lang="ts">
	let {
		tp,
		fp,
		fn,
		tn,
		axis = 'correct'
	}: {
		tp: number;
		fp: number;
		fn: number;
		tn: number;
		/** Which class is "positive". 'correct' → Tier-1; 'error' → Tier-2. */
		axis?: 'correct' | 'error';
	} = $props();

	// Map the served {tp,fp,fn,tn} (positive class = `axis`) onto a fixed gold×pred
	// grid so the geometry is identical regardless of tier:
	//   gold-correct row = [pred-correct, pred-incorrect]
	//   gold-error   row = [pred-correct, pred-incorrect]
	// Tier-1 positive=correct ⇒ tp=gc&pc, fp=gi&pc, fn=gc&pi? No: confusion_metrics
	// is over (gold_positive, pred_positive). For axis='correct': positive=correct,
	// so tp=g✓p✓, fp=g✗p✓, fn=g✓p✗, tn=g✗p✗.
	// For axis='error': positive=error, so tp=g✗p✗(caught), fp=g✓p✗, fn=g✗p✓, tn=g✓p✓.
	const grid = $derived(
		axis === 'correct'
			? { gc_pc: tp, gc_pi: fn, gi_pc: fp, gi_pi: tn }
			: { gc_pc: tn, gc_pi: fp, gi_pc: fn, gi_pi: tp }
	);
	const rc = $derived(grid.gc_pc + grid.gc_pi); // gold-correct row total
	const ri = $derived(grid.gi_pc + grid.gi_pi); // gold-error row total
</script>

<figure class="cm">
	<div class="mosaic" style="--rc:{rc}; --ri:{ri}">
		<div class="mrow" style="flex:{rc || 1}">
			<div
				class="mcell agree"
				style="flex:{grid.gc_pc}"
				title={`gold correct, model said correct — ${grid.gc_pc}`}
			>
				<span>{grid.gc_pc}</span>
			</div>
			<div
				class="mcell miss-fn"
				style="flex:{grid.gc_pi}"
				title={`gold correct, model said error (over-rejected) — ${grid.gc_pi}`}
			>
				<span>{grid.gc_pi}</span>
			</div>
		</div>
		<div class="mrow" style="flex:{ri || 1}">
			<div
				class="mcell miss-fp"
				style="flex:{grid.gi_pc}"
				title={`gold ERROR, model said correct (over-accepted, the dangerous miss) — ${grid.gi_pc}`}
			>
				<span>{grid.gi_pc}</span>
			</div>
			<div
				class="mcell catch"
				style="flex:{grid.gi_pi}"
				title={`gold error, model caught it — ${grid.gi_pi}`}
			>
				<span>{grid.gi_pi}</span>
			</div>
		</div>
	</div>
	<div class="legend">
		<span><i class="sw agree"></i> agree ✓</span>
		<span><i class="sw catch"></i> error caught</span>
		<span><i class="sw miss-fp"></i> error missed (said ✓)</span>
		<span><i class="sw miss-fn"></i> supported rejected</span>
	</div>
</figure>

<style>
	.cm {
		margin: 0;
	}
	.mosaic {
		display: flex;
		flex-direction: column;
		gap: 2px;
		height: 9rem;
		max-width: 22rem;
	}
	.mrow {
		display: flex;
		gap: 2px;
		min-height: 0;
	}
	.mcell {
		display: flex;
		align-items: center;
		justify-content: center;
		min-width: 0;
		overflow: hidden;
	}
	.mcell span {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--paper);
		font-variant-numeric: tabular-nums;
		opacity: 0.95;
	}
	/* Single-palette run-on-its-own register: forest-green = right, accent = the
	   dangerous over-accept, muted-amber = the over-reject. No A/B hues. */
	.mcell.agree {
		background: color-mix(in srgb, var(--ok-green) 78%, black);
	}
	.mcell.catch {
		background: var(--ok-green);
	}
	.mcell.miss-fp {
		background: var(--accent);
	}
	.mcell.miss-fn {
		background: #6f5a16;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 1rem;
		margin-top: 0.6rem;
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--ink-faint);
	}
	.sw {
		display: inline-block;
		width: 0.66rem;
		height: 0.66rem;
		vertical-align: -1px;
		margin-right: 0.2rem;
	}
	.sw.agree {
		background: color-mix(in srgb, var(--ok-green) 78%, black);
	}
	.sw.catch {
		background: var(--ok-green);
	}
	.sw.miss-fp {
		background: var(--accent);
	}
	.sw.miss-fn {
		background: #6f5a16;
	}
</style>
