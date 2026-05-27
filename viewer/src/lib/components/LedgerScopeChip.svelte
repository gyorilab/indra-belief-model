<script lang="ts">
	import { panelApplicabilityClass, panelApplicabilityLabel } from '$lib/pairedMetricKinds';
	import type { PanelApplicabilityKind } from '$lib/pairedMetricKinds';

	let { applicability }: { applicability: PanelApplicabilityKind | string } = $props();

	const scopeClass = $derived(panelApplicabilityClass(applicability));
	const label = $derived(panelApplicabilityLabel(applicability));
</script>

<span
	class={`ledger-scope ledger-scope-${scopeClass}`}
	data-applicability={applicability}
	data-applicability-class={scopeClass}
>
	{label}
</span>

<style>
	.ledger-scope {
		color: var(--ink-muted);
		white-space: nowrap;
	}
	.ledger-scope-arch_blind {
		color: var(--ink-muted);
	}
	.ledger-scope-arch_conditioned {
		color: var(--ink);
	}
	.ledger-scope-paired_only {
		color: var(--ink);
	}
	.ledger-scope-not_defined {
		color: var(--ink-muted);
		font-style: italic;
	}
	.ledger-scope-unknown {
		color: var(--ledger-chip-danger-ink, #9f2d20);
		font-weight: 700;
	}
</style>
