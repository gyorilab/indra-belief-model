import { loadBeliefHeuristic } from '$lib/server/belief-heuristic';
import { loadBeliefLadder } from '$lib/server/paper-belief-ladder';
import { loadFramingCorrection } from '$lib/server/paper-framing-correction';
import { loadPaperLiteral } from '$lib/server/paper-literal';
import { loadDeployedBaseline } from '$lib/server/paper-deployed-baseline';
import { loadPaperOwnMetric } from '$lib/server/paper-own-metric';
import { loadPaperPerEvidence } from '$lib/server/paper-per-evidence';
 import { loadPaperRobustness } from '$lib/server/paper-robustness';
import { loadPaperTable6Extended } from '$lib/server/paper-table6-extended';
import { loadPaperTieInflation } from '$lib/server/paper-tie-inflation';
import { loadReviewQueue } from '$lib/server/paper-review-queue';
import { loadStatementErrorF1 } from '$lib/server/paper-error-f1';
import type { PageServerLoad } from './$types';

// The fs/crypto loaders are server-only; the page consumes the serialized payload.
// Synchronous loaders (mirrors the frontier/compare PageServerLoad idiom), so no await.
// One loader per ARTIFACT, never one per beat: the review queue, the belief-model
// ladder and the framing correction each carry their own sha in the run manifest,
// so each gates independently — one artifact going dark leaves the rest of the
// page standing. The framing correction reads two artifacts and gates on both.
export const load: PageServerLoad = () => {
	// One read of the 6 MB prediction bundles: the own-metric figure derives from
	// the same payload rather than loading it a second time.
	const paperLiteral = loadPaperLiteral();
	return {
		paperLiteral,
		paperOwnMetric: loadPaperOwnMetric(paperLiteral),
		deployedBaseline: loadDeployedBaseline(),
		paperPerEvidence: loadPaperPerEvidence(),
		tieInflation: loadPaperTieInflation(),
		paperRobustness: loadPaperRobustness(),
		framingCorrection: loadFramingCorrection(),
		reviewQueue: loadReviewQueue(),
		// Two artifacts, two loaders, two independent gates — the error-F1 surface
		// and the extended Table 6 each carry their own sha in the same manifest, so
		// one going dark leaves the other, and the rest of the page, standing.
		statementErrorF1: loadStatementErrorF1(),
		paperTable6Extended: loadPaperTable6Extended(),
		beliefLadder: loadBeliefLadder(),
		beliefHeuristic: loadBeliefHeuristic()
	};
};
