<script lang="ts">
	import { pairedMetricKindFamily, pairedMetricKindLabel } from '$lib/pairedMetricKinds';

	type Variant = 'cell' | 'inline';

	let {
		kind,
		variant = 'cell'
	}: {
		kind: string;
		variant?: Variant;
	} = $props();

	const family = $derived(pairedMetricKindFamily(kind));
	const label = $derived(pairedMetricKindLabel(kind));
</script>

<span class={`ledger-kind ledger-kind-${family} ledger-kind-${variant}`} data-metric-kind={kind}>
	{label}
</span>

<style>
	.ledger-kind {
		border: 1px solid var(--rule);
		color: var(--ink);
		display: inline-block;
		font-family: var(--mono);
		font-size: 0.68rem;
		line-height: 1.35;
		padding: 0 0.24rem;
		white-space: nowrap;
	}
	.ledger-kind-inline {
		font-size: 0.66rem;
		vertical-align: baseline;
	}
	.ledger-kind-arch-arch {
		border-color: var(--ink);
	}
	.ledger-kind-arch-indra {
		border-color: var(--accent);
		color: var(--accent);
		font-weight: 600;
	}
	.ledger-kind-base {
		border-style: dotted;
		color: var(--ink-muted);
	}
	.ledger-kind-gate {
		border-style: dashed;
		color: var(--ink-muted);
	}
	.ledger-kind-native {
		background: var(--ledger-chip-native-bg, #f7f4ec);
		color: var(--ink-muted);
	}
	.ledger-kind-unknown {
		border-color: var(--ledger-chip-danger-ink, #9f2d20);
		color: var(--ledger-chip-danger-ink, #9f2d20);
		font-weight: 700;
	}
</style>
