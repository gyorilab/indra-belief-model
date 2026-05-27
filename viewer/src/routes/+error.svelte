<script lang="ts">
	import { page } from '$app/state';
	const status = $derived(page.status);
	const msg = $derived(page.error?.message ?? '');
	const code = $derived(page.error?.code ?? '');
	const isMalformedWriterLock = $derived(code === 'writer_lock_malformed' || msg.startsWith('writer_lock_malformed:'));
	const isWriterLock = $derived(code === 'writer_in_progress' || msg.startsWith('writer_in_progress:'));
	const isCorruptCorpus = $derived(
		code === 'corrupt_corpus_schema' || msg.startsWith('corrupt_corpus_schema:')
	);
	const malformedWriterMessage = $derived(msg.replace(/^writer_lock_malformed:\s*/, ''));
	const writerMessage = $derived(msg.replace(/^writer_in_progress:\s*/, ''));
	const schemaMessage = $derived(msg.replace(/^corrupt_corpus_schema:\s*/, ''));
</script>

<svelte:head>
	<title>{isMalformedWriterLock ? 'writer lock needs repair' : isCorruptCorpus ? 'schema mismatch' : status} · INDRA Belief</title>
</svelte:head>

<main id="main">
	{#if isMalformedWriterLock}
		<section class="lock-state malformed-lock-state">
			<h1>writer lock needs repair</h1>
			<p class="lock-line">
				The writer-lock sidecar exists but cannot be trusted. Reads and writer
				actions pause so the viewer does not write around an ambiguous DuckDB
				writer.
			</p>
			<p class="lock-hint">
				Confirm no ingest, score, truth-set, or repair worker is still active,
				then repair or remove <code>viewer_state/writer_lock.json</code> and
				reload.
			</p>
			<code class="error-code writer-code">{malformedWriterMessage}</code>
			<button type="button" onclick={() => location.reload()}>reload after repair</button>
		</section>
	{:else if isWriterLock}
		<section class="lock-state">
			<h1>writer in progress</h1>
			<p class="lock-line">
				A writer workflow is active for this DuckDB file. Dashboard reads pause
				until it finishes so the UI does not mix stale counts with a live write.
			</p>
			<p class="lock-hint">
				If another tab started the writer, watch that tab. This page will keep
				showing the writer state until you reload after the worker reports
				<code>done</code>.
			</p>
			<code class="error-code writer-code">{writerMessage}</code>
			<button type="button" onclick={() => location.reload()}>reload now</button>
		</section>
	{:else if isCorruptCorpus}
		<section class="corrupt-state">
			<h1>corpus schema mismatch</h1>
			<p class="corrupt-line">
				The DuckDB file exists, but this viewer build expects overview columns
				the file does not provide. Rebuild or migrate the corpus database before
				reading dashboard counts.
			</p>
			<p class="corrupt-hint">
				Counts are intentionally hidden here so a schema break is not mistaken
				for an empty corpus.
			</p>
			<code class="error-code">{schemaMessage}</code>
			<button type="button" onclick={() => location.reload()}>reload after repair</button>
		</section>
	{:else}
		<section class="generic-err">
			<h1>{status}</h1>
			<p>{msg || 'Something went wrong.'}</p>
			<p><a href="/">← back to dashboard</a></p>
		</section>
	{/if}
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
	.lock-state h1 {
		color: #7d2a1a;
	}
	.malformed-lock-state h1 {
		color: #4f3a7a;
	}
	.corrupt-state h1 {
		color: #4f3a7a;
	}
	.lock-line,
	.corrupt-line {
		font-size: 1.05rem;
		margin: 0 0 1rem;
	}
	.lock-hint,
	.corrupt-hint {
		font-size: 0.95rem;
		color: #6a6a6a;
		margin: 0 0 1.4rem;
	}
	code {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.86rem;
		background: rgba(125, 42, 26, 0.04);
		padding: 0 0.3rem;
	}
	.error-code {
		display: block;
		margin: 0 0 1.4rem;
		padding: 0.7rem 0.8rem;
		border-left: 3px solid #4f3a7a;
		background: rgba(79, 58, 122, 0.05);
		color: #2f2546;
		overflow-wrap: anywhere;
	}
	.writer-code {
		border-left-color: #7d2a1a;
		background: rgba(125, 42, 26, 0.04);
		color: #4a2016;
	}
	button {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.86rem;
		border: 1px solid #7d2a1a;
		background: transparent;
		color: #7d2a1a;
		padding: 0.3rem 0.8rem;
		cursor: pointer;
	}
	button:hover {
		background: rgba(125, 42, 26, 0.04);
	}
	.corrupt-state button {
		border-color: #4f3a7a;
		color: #4f3a7a;
	}
	.corrupt-state button:hover {
		background: rgba(79, 58, 122, 0.05);
	}
	a {
		color: #7d2a1a;
	}
</style>
