<script lang="ts">
	import { page } from '$app/state';
	const status = $derived(page.status);
	const msg = $derived(page.error?.message ?? '');
</script>

<svelte:head>
	<title>{status} · INDRA Belief</title>
</svelte:head>

<main id="main">
	<section class="generic-err">
		<h1>{status}</h1>
		<p>{msg || 'Something went wrong.'}</p>
		<p><a href="/">← back to dashboard</a></p>
	</section>
</main>

<style>
	:global(html, body) {
		background: #fdfcf8;
		color: #1a1a1a;
		font-family: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
		font-size: 16px;
		line-height: 1.5;
		margin: 0;
	}
	main {
		max-width: 640px;
		margin: 4rem auto;
		padding: 0 1.5rem;
	}
	h1 {
		font-weight: 400;
		font-size: 1.6rem;
		margin: 0 0 1rem;
	}
	a {
		color: #7d2a1a;
	}
</style>
