<script lang="ts">
	import { enhance } from '$app/forms';
	import type { PageData, ActionData } from './$types';

	let { data, form }: { data: PageData; form: ActionData } = $props();
	let busy = $state(false);

	const submit = () => {
		busy = true;
		return async ({ update }: { update: () => Promise<void> }) => {
			await update();
			busy = false;
		};
	};
</script>

<svelte:head><title>sign in · indra-belief viewer</title></svelte:head>

<main class="lg">
	<header class="lg-head">
		<h1>sign in</h1>
		<p class="lg-sub">
			Authenticate with your <a href="https://db.indra.bio" target="_blank" rel="noreferrer">INDRA</a>
			account. Curations you submit are attributed to you and authenticated by INDRA — no shared key.
		</p>
	</header>

	{#if form?.error}<p class="lg-err">{form.error}</p>{/if}

	<form method="POST" use:enhance={submit} class="lg-form">
		<input type="hidden" name="redirectTo" value={data.redirectTo} />
		<label>
			email
			<input
				type="email"
				name="email"
				value={form?.email ?? ''}
				required
				autocomplete="username"
				placeholder="you@institution.org"
			/>
		</label>
		<label>
			password
			<input type="password" name="password" required autocomplete="current-password" />
		</label>
		<button type="submit" disabled={busy}>{busy ? 'signing in…' : 'sign in →'}</button>
	</form>

	<p class="lg-note">
		No INDRA account? Registering needs only an email — create one at
		<a href="https://db.indra.bio" target="_blank" rel="noreferrer">db.indra.bio</a>.
	</p>
</main>

<style>
	:root {
		--ink: #1a1a1a; --ink-muted: #6a6a6a; --ink-faint: #727272;
		--paper: #fdfcf8; --rule: #e6e2d6; --accent: #7d2a1a;
		--accent-wash: rgba(125, 42, 26, 0.04);
		--mono: ui-monospace, 'SF Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
	}
	.lg { max-width: 24rem; margin: 4rem auto; padding: 1.5rem; font-family: var(--serif); color: var(--ink); }
	.lg-head { border-bottom: 1px solid var(--rule); padding-bottom: 0.7rem; margin-bottom: 1.2rem; }
	.lg-head h1 { font-size: 1.1rem; font-weight: 400; margin: 0 0 0.4rem; }
	.lg-sub { font-size: 0.86rem; line-height: 1.5; color: var(--ink-muted); margin: 0; }
	.lg-sub a, .lg-note a { color: var(--accent); text-decoration: none; }
	.lg-sub a:hover, .lg-note a:hover { text-decoration: underline; }
	.lg-err {
		font-family: var(--mono); font-size: 0.78rem; color: var(--accent);
		background: var(--accent-wash); border: 1px solid var(--accent); border-radius: 3px;
		padding: 0.5rem 0.7rem; margin-bottom: 1rem;
	}
	.lg-form label {
		display: block; font-family: var(--mono); font-size: 0.72rem; color: var(--ink-muted); margin-bottom: 0.9rem;
	}
	.lg-form input {
		display: block; width: 100%; margin-top: 0.25rem; font-family: var(--mono); font-size: 0.9rem;
		border: 1px solid var(--rule); border-radius: 2px; padding: 0.5rem; background: var(--paper); color: var(--ink);
	}
	.lg-form button {
		font-family: var(--mono); font-size: 0.82rem; color: var(--paper); background: var(--ink);
		border: none; padding: 0.55rem 1.1rem; cursor: pointer; border-radius: 2px; margin-top: 0.2rem;
	}
	.lg-form button:hover:not(:disabled) { background: var(--accent); }
	.lg-form button:disabled { opacity: 0.5; cursor: default; }
	.lg-note { font-size: 0.78rem; color: var(--ink-faint); margin-top: 1.4rem; }
</style>
