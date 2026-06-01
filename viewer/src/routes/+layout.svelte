<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';
	import SiteNav from '$lib/components/SiteNav.svelte';

	import type { Snippet } from 'svelte';
	let { children }: { children: Snippet } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<a href="#main" class="skip-link">Skip to main content</a>

<SiteNav />

{@render children()}

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
		--ink-faint: #727272;
		--paper: #fdfcf8;
		--rule: #e6e2d6;
		--accent: #7d2a1a;
		--accent-wash: rgba(125, 42, 26, 0.04);
		--blocked: #6f5a16;
		--ok-green: #2a6f2a;
		--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
		--sans: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
	}

	.skip-link {
		position: absolute;
		left: -9999px;
		top: 0;
		font-family: var(--mono);
		font-size: 0.82rem;
		background: var(--ink);
		color: var(--paper);
		padding: 0.5rem 0.8rem;
		text-decoration: none;
		z-index: 100;
	}

	.skip-link:focus {
		left: 1rem;
		top: 1rem;
	}

	:global(.visually-hidden) {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}
</style>
