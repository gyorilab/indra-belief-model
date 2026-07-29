<script lang="ts">
	/**
	 * THE PAGE'S ONE VERIFICATION BOUNDARY — last block, after every figure.
	 *
	 * Six blocks above used to put the result files' own wording on screen:
	 *
	 *   1. ReviewQueue — its method note, three of whose paragraphs opened with
	 *      "in the artifact's own words", plus the whole robustness block;
	 *   2-3. StatementErrorF1 — two summaries offering "the cutoff rule and its
	 *      disclosure, in the artifact's own words";
	 *   4. StatementErrorF1 — "Caveats, as shipped", and the value-table note
	 *      above it;
	 *   5. BeliefModelLadder — its method note;
	 *   6. DeployedBaseline — its method note, and the SVG description, which was
	 *      behind no boundary at all and reached a screen-reader user unmediated.
	 *
	 * Every one of them handed a curator a sentence written for a referee. They
	 * render their plain twin now; what they used to show is here, once, in a
	 * section introduced as what it is, doing a job nothing else on the page does:
	 * letting someone check what they just read against the files it came from.
	 *
	 * IT IS NOT "THE REAL VERSION". Every sentence above says what its twin here
	 * says — that is enforced upstream, by a guard that fails when a restatement
	 * carries fewer numbers, fewer denials or fewer named models than the sentence
	 * it restates. So the framing here is never "the fuller wording" or "the
	 * original": those would tell a serious reader that the page they just read
	 * was the simplified one, and it was not.
	 *
	 * ONE `<details>` IN THIS FILE, and that is the boundary. Grouping inside it
	 * is headings and lists, never a second toggle — four toggles are what this
	 * section exists to replace, and rebuilding them one level down would be the
	 * same defect with an extra click in front of it.
	 *
	 * Nothing is decided here. Which sentences exist, which file each came from,
	 * which digest each file carries and which groups gate are all settled in
	 * `$lib/data/paper-audit-trail.ts`; this file prints them.
	 *
	 * PROP CONTRACT. `data` is the page's own server load, passed straight through:
	 * `<PaperAuditTrail {data} />`. `PageData` satisfies `PaperAuditPageLoads`
	 * structurally, so the mount takes no adapter and no second serialization of
	 * anything. This component reads no file and holds no state.
	 */
	import { paperAuditTrail, type PaperAuditPageLoads } from '$lib/data/paper-audit-trail';

	let { data }: { data: PaperAuditPageLoads } = $props();

	const trail = $derived(paperAuditTrail(data));

	/** Enough digest to compare by eye, with the whole thing on hover. */
	const shortSha = (sha: string) => sha.slice(0, 12);
</script>

<section class="audit" aria-labelledby="audit-title">
	<header>
		<p class="eyebrow">checking this page</p>
		<!-- THE HEADING IS THE FRAMING, and it was the one line still selling this
		     section as the authentic version. It read "The result files, and what they
		     say in their own words" — a near-synonym of the exact phrasing this
		     section exists to retire, sitting in the most prominent line a reader
		     meets. "Their own words" tells a curator the plain text upstream is a
		     paraphrase to bypass; it is not, it is the same claim. The heading now
		     names the JOB: checking the page against the files. -->
		<h2 id="audit-title">Check this page against the result files</h2>
		<p class="intro">
			Every claim above is written in ordinary words. The files those claims are read from are
			written in the shorthand of the people who generated them. Below is each file, the sha256 of
			the bytes this page drew, and every sentence it took: the file's own wording first, then the
			wording used above. The second is not a simplification of the first — it is the same claim,
			with every number, every limit and every model named in both. The point of having both is
			that you can check one against the other.
		</p>
	</header>

	<details class="boundary">
		<summary>
			for verification against the shipped files &#183; {trail.nEntries} sentences from {trail.nFiles}
			files
		</summary>

		{#each trail.files as file, index (`${file.file}#${index}`)}
			<article class="file">
				<h3>{file.file}</h3>
				<p class="provenance">
					{#if file.path}<code>{file.path}</code>{:else}<span class="dark"
							>no path is carried for this file</span
						>{/if}
					{#if file.sha256}<span class="sha" title={file.sha256}>sha256 {shortSha(file.sha256)}</span
						>{:else}<span class="dark"
							>no digest is carried for this file — hash it yourself to check it</span
						>{/if}
				</p>
				{#if file.unavailable}
					<p class="dark">{file.unavailable}</p>
				{:else}
					<dl class="pairs">
						{#each file.entries as entry (entry.field)}
							<div>
								<dt>
									<span class="field" title="where this sits on the loaded figure"
										>{entry.field}</span
									>
									<span class="shipped">{entry.shipped}</span>
								</dt>
								<dd>{entry.plain}</dd>
							</div>
						{/each}
					</dl>
				{/if}
			</article>
		{/each}

		{#if trail.nFilesUnavailable > 0}
			<p class="dark tally">
				{trail.nFilesUnavailable} of {trail.files.length} files contribute nothing above: each says why
				under its own name.
			</p>
		{/if}
		{#if trail.nConflicts > 0}
			<p class="dark tally">
				{trail.nConflicts}
				{trail.nConflicts === 1 ? 'shipped sentence is' : 'shipped sentences are'} restated more than
				one way on this page. Every restatement is listed above, under the file that sentence ships
				in.
			</p>
		{/if}
	</details>
</section>

<style>
	.audit {
		margin-top: 2.4rem;
		padding-top: 1.1rem;
		border-top: 2px solid var(--ink);
	}
	.eyebrow {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	.audit h2 {
		margin: 0.18rem 0 0;
		font-family: var(--serif);
		font-weight: 400;
		font-size: 1.35rem;
	}
	.intro {
		margin: 0.35rem 0 0;
		max-width: 66ch;
		font-family: var(--serif);
		font-size: 0.86rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.boundary {
		margin-top: 1rem;
	}
	.boundary > summary {
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		cursor: pointer;
	}
	.boundary > summary:hover {
		color: var(--ink-muted);
	}
	.file {
		margin-top: 1.5rem;
		padding-top: 0.7rem;
		border-top: 1px dotted var(--rule);
	}
	.file h3 {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.82rem;
		font-weight: 600;
		color: var(--ink);
	}
	.provenance {
		display: flex;
		flex-wrap: wrap;
		gap: 0.2rem 0.9rem;
		margin: 0.22rem 0 0;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
	}
	.provenance code {
		font-size: 0.66rem;
		overflow-wrap: anywhere;
	}
	.sha {
		font-variant-numeric: tabular-nums;
	}
	.pairs {
		margin: 0.7rem 0 0;
	}
	.pairs > div {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		gap: 0.4rem 1.6rem;
		padding: 0.55rem 0;
		border-top: 1px dotted var(--rule);
	}
	.pairs dt,
	.pairs dd {
		margin: 0;
		min-width: 0;
		overflow-wrap: anywhere;
	}
	.field {
		display: block;
		margin-bottom: 0.22rem;
		font-family: var(--mono);
		font-size: 0.6rem;
		color: var(--ink-faint);
	}
	.shipped {
		font-family: var(--mono);
		font-size: 0.72rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.pairs dd {
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.5;
		color: var(--ink);
	}
	.dark {
		margin: 0.3rem 0 0;
		font-family: var(--mono);
		font-size: 0.72rem;
		line-height: 1.45;
		color: var(--ink-faint);
	}
	.tally {
		margin-top: 1.2rem;
	}
	@media (max-width: 720px) {
		.pairs > div {
			grid-template-columns: minmax(0, 1fr);
			gap: 0.45rem;
		}
	}
</style>
