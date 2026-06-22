<!--
  BrierBar (E5) — the resolution-vs-reliability trade made visible.

  Murphy: brier = uncertainty + reliability − resolution.
    · uncertainty   = irreducible floor (base-rate variance) — neutral.
    · reliability   = miscalibration penalty — ADDS error (lower is better).
    · resolution    = discriminating power — SUBTRACTS error (higher is better).

  So we stack the two ADDITIVE terms (uncertainty floor + reliability penalty)
  upward, then draw resolution as a CREDIT bar pulling the total back down to the
  realized Brier. The eye reads: floor you can't beat, penalty you're paying,
  credit your discrimination earns back. Numbers served byte-exact from
  metrics.json (G4) — nothing recomputed.
-->
<script lang="ts">
	let {
		reliability,
		resolution,
		uncertainty,
		brier
	}: { reliability: number; resolution: number; uncertainty: number; brier: number } = $props();

	// Scale: the tallest thing is uncertainty + reliability (pre-credit total).
	const gross = $derived(uncertainty + reliability);
	const scale = $derived(Math.max(gross, brier, 1e-6));
	function h(v: number): number {
		return (v / scale) * 100;
	}
	function f3(v: number): string {
		return v.toFixed(3);
	}
</script>

<figure class="brier">
	<div class="cols">
		<!-- gross column: uncertainty floor + reliability penalty -->
		<div class="col" title={`uncertainty ${f3(uncertainty)} (floor) + reliability ${f3(reliability)} (penalty)`}>
			<div class="stack">
				<div class="seg penalty" style="height:{h(reliability)}%">
					<span class="seg-n">{f3(reliability)}</span>
				</div>
				<div class="seg floor" style="height:{h(uncertainty)}%">
					<span class="seg-n">{f3(uncertainty)}</span>
				</div>
			</div>
			<div class="col-lab">uncertainty + reliability</div>
		</div>

		<div class="op">−</div>

		<!-- resolution credit -->
		<div class="col" title={`resolution ${f3(resolution)} — discrimination credit`}>
			<div class="stack">
				<div class="seg credit" style="height:{h(resolution)}%">
					<span class="seg-n">{f3(resolution)}</span>
				</div>
			</div>
			<div class="col-lab">resolution</div>
		</div>

		<div class="op">=</div>

		<!-- realized brier -->
		<div class="col" title={`Brier ${f3(brier)}`}>
			<div class="stack">
				<div class="seg brier-seg" style="height:{h(brier)}%">
					<span class="seg-n">{f3(brier)}</span>
				</div>
			</div>
			<div class="col-lab">Brier</div>
		</div>
	</div>
	<p class="legend">
		<span><i class="sw floor"></i> uncertainty — irreducible floor</span>
		<span><i class="sw penalty"></i> reliability — miscalibration (lower better)</span>
		<span><i class="sw credit"></i> resolution — discrimination (higher better)</span>
	</p>
</figure>

<style>
	.brier {
		margin: 0;
	}
	.cols {
		display: flex;
		align-items: flex-end;
		gap: 0.8rem;
		height: 9rem;
	}
	.col {
		display: flex;
		flex-direction: column;
		justify-content: flex-end;
		align-items: stretch;
		min-width: 4.5rem;
		height: 100%;
	}
	.stack {
		display: flex;
		flex-direction: column;
		justify-content: flex-end;
		flex: 1;
		min-height: 0;
	}
	.seg {
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: 1px;
		border-top: 1px solid var(--paper);
	}
	.seg-n {
		font-family: var(--mono);
		font-size: 0.66rem;
		font-variant-numeric: tabular-nums;
		color: var(--paper);
	}
	.floor {
		background: var(--ink-faint);
	}
	.penalty {
		background: var(--accent);
	}
	.credit {
		background: var(--ok-green);
	}
	.brier-seg {
		background: var(--ink);
	}
	.op {
		font-family: var(--mono);
		font-size: 1.1rem;
		color: var(--ink-muted);
		align-self: center;
		padding-bottom: 1.2rem;
	}
	.col-lab {
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-muted);
		text-align: center;
		margin-top: 0.3rem;
		line-height: 1.2;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 1rem;
		margin: 0.7rem 0 0;
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
	.sw.floor {
		background: var(--ink-faint);
	}
	.sw.penalty {
		background: var(--accent);
	}
	.sw.credit {
		background: var(--ok-green);
	}
</style>
