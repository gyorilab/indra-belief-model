<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { LayoutData } from './$types';

	// The JSONL exports have no probe/trace/truth-set lenses, so there is no
	// RunLensNav; the run-detail page carries its own crumb back to /runs.
	let { children }: { data: LayoutData; children: Snippet } = $props();
</script>

{@render children()}
