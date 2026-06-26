<script lang="ts">
	import { page } from '$app/state';

	const path = $derived(page.url.pathname);
	const user = $derived(page.data.user as { email: string } | null);

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
		<li><a href="/frontier" class:active={isActive('/frontier')}>frontier</a></li>
		<li><a href="/adjudicate" class:active={isActive('/adjudicate')}>adjudicate</a></li>
		<li><a href="/review" class:active={isActive('/review')}>review</a></li>
		<li><a href="/curate" class:active={isActive('/curate')}>curate</a></li>
	</ul>
	{#if user}
		<form method="POST" action="/logout" class="who">
			<span class="who-email" title={user.email}>{user.email}</span>
			<button type="submit" class="who-out">sign out</button>
		</form>
	{/if}
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
	.who {
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
		margin: 0;
	}
	.who-email {
		color: var(--ink-faint, #727272);
		max-width: 14rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.who-out {
		font-family: inherit;
		font-size: inherit;
		color: var(--ink-muted, #6a6a6a);
		background: none;
		border: none;
		padding: 0;
		cursor: pointer;
	}
	.who-out:hover {
		color: var(--accent, #7d2a1a);
		text-decoration: underline;
		text-underline-offset: 3px;
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
