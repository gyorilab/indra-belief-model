<script lang="ts">
	import BeliefVsIndra from '$lib/components/BeliefVsIndra.svelte';
	import SiteNav from '$lib/components/SiteNav.svelte';
	let { data } = $props();
</script>

<svelte:head><title>Belief · INDRA and reading, side by side</title></svelte:head>

<SiteNav />

<main>
	<h1>Belief</h1>
	<p class="intro">
		INDRA assigns every assembled statement a belief: the probability that the statement is
		correct. Today that number is computed from how many sources reported the statement and how
		reliable each source is on average. This page puts that model next to one that reads the
		evidence sentence, on the same statements and the same curations.
	</p>

	{#if data.beliefVsIndra.available}
		<BeliefVsIndra data={data.beliefVsIndra} />
	{:else}
		<p class="unavailable">
			This comparison is unavailable: {data.beliefVsIndra.reason}.
		</p>
	{/if}
</main>

<style>
	main { max-width: 62rem; margin: 0 auto; padding: 1.5rem 1.25rem 5rem; }
	h1 { font-size: 1.7rem; margin: 0 0 .5rem; font-weight: 600; }
	.intro { color: var(--ink-muted); max-width: 44rem; margin: 0 0 2rem; }
	.unavailable { font-family: var(--mono, monospace); font-size: .8rem; color: var(--blocked, #6f5a16); }
</style>
