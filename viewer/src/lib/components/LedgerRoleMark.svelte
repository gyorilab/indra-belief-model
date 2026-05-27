<script lang="ts">
	import { pairedLedgerRoleClass, pairedLedgerRoleLabel } from '$lib/pairedMetricKinds';
	import type { PairedLedgerRole } from '$lib/pairedMetricKinds';

	let { role }: { role: PairedLedgerRole | string } = $props();

	const roleClass = $derived(pairedLedgerRoleClass(role));
	const label = $derived(pairedLedgerRoleLabel(role));
</script>

<span
	class={`ledger-role-mark ledger-role-mark-${roleClass}`}
	data-ledger-role={role}
	data-ledger-role-class={roleClass}
>
	{label}
</span>

<style>
	.ledger-role-mark {
		background: var(--ink);
		color: var(--paper);
		display: inline-block;
		font-family: var(--mono);
		font-size: 0.65rem;
		font-weight: 700;
		line-height: 1.35;
		margin-right: 0.35rem;
		min-width: 2.55rem;
		padding: 0 0.24rem;
		text-align: center;
		text-transform: lowercase;
		vertical-align: baseline;
	}
	.ledger-role-mark-base {
		background: transparent;
		box-shadow: inset 0 0 0 1px var(--ink);
		color: var(--ink);
	}
	.ledger-role-mark-metric {
		background: transparent;
		color: var(--ink);
		font-size: 0.9rem;
		min-width: 0;
		padding: 0;
	}
	.ledger-role-mark-gate,
	.ledger-role-mark-native {
		background: transparent;
		box-shadow: inset 0 0 0 1px var(--rule);
		color: var(--ink-muted);
	}
	.ledger-role-mark-unknown {
		background: var(--ledger-chip-danger-ink, #9f2d20);
		color: var(--ledger-chip-danger-paper, #fff);
	}
</style>
