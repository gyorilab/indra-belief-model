<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { LayoutData } from './$types';

	// The RunLensNav (cohort / repairs / truth-sets tabs) was removed with the
	// DuckDB layer; the monolithic JSONL exports have no probe/trace/truth-set
	// lenses. The run-detail page carries its own crumb back to /runs.
	let { children }: { data: LayoutData; children: Snippet } = $props();
</script>

{@render children()}
