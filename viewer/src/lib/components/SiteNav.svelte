<script lang="ts">
	import { page } from '$app/state';

	const path = $derived(page.url.pathname);

	function isActive(prefix: string): boolean {
		if (prefix === '/') return path === '/';
		return path === prefix || path.startsWith(prefix + '/');
	}
</script>

<nav class="site-nav" aria-label="site sections">
	<ul class="sections">
		<li><a href="/" class:active={isActive('/')}>overview</a></li>
		<li><a href="/statements" class:active={isActive('/statements')}>statements</a></li>
		<li><a href="/runs" class:active={isActive('/runs')}>runs</a></li>
		<li><a href="/compare" class:active={isActive('/compare')}>compare</a></li>
		<li><a href="/adjudicate" class:active={isActive('/adjudicate')}>adjudicate</a></li>
		<li><a href="/review" class:active={isActive('/review')}>review</a></li>
	</ul>
</nav>

<style>
	.site-nav {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 1.2rem;
		padding: 0.55rem 1.5rem;
		border-bottom: 1px solid var(--rule, #e6e2d6);
		font-family: var(--mono, ui-monospace, 'SF Mono', Menlo, monospace);
		font-size: 0.78rem;
		background: var(--paper, #fdfcf8);
		color: var(--ink-muted, #6a6a6a);
	}
	.sections {
		display: flex;
		gap: 1.5rem;
		list-style: none;
		margin: 0;
		padding: 0;
		min-width: 0;
	}
	.sections a {
		color: var(--ink-muted, #6a6a6a);
		text-decoration: none;
	}
	.sections a:hover {
		color: var(--ink, #1a1a1a);
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.sections a.active {
		color: var(--ink, #1a1a1a);
		font-weight: 500;
	}
	@media (max-width: 720px) {
		.site-nav {
			flex-wrap: wrap;
			padding: 0.5rem 1rem;
			gap: 0.4rem 1rem;
		}
		.sections {
			gap: 1rem;
		}
	}
</style>
