import pricesJson from './modelPrices.json';

const pricesParsed = pricesJson as unknown as {
	prices_per_million_tokens: Record<string, [number, number]>;
	zero_cost_model_ids: string[];
};

export const MODEL_PRICES_PER_M_TOKENS: Record<string, [number, number]> =
	pricesParsed.prices_per_million_tokens;
export const ZERO_COST_MODEL_IDS: Set<string> = new Set(pricesParsed.zero_cost_model_ids);

/**
 * Build a DuckDB CASE expression that maps `model_id` to observed USD given
 * `prompt_tokens` and `out_tokens` columns. Single source of truth: pricing
 * comes from `modelPrices.json`, which Python's `cost.py` also reads at
 * module-load time. The CASE here, the Python `token_cost_usd`, and the
 * UI's cost display are now derived from the same constants.
 */
export function buildObservedCostSql(
	promptCol = 'prompt_tokens',
	outCol = 'out_tokens'
): string {
	const branches: string[] = [];
	for (const [modelId, [inPrice, outPrice]] of Object.entries(MODEL_PRICES_PER_M_TOKENS)) {
		const safeId = modelId.replace(/'/g, "''");
		branches.push(
			`WHEN '${safeId}' THEN COALESCE(${promptCol}, 0) * ${inPrice} / 1000000.0 ` +
				`+ COALESCE(${outCol}, 0) * ${outPrice} / 1000000.0`
		);
	}
	for (const modelId of ZERO_COST_MODEL_IDS) {
		const safeId = modelId.replace(/'/g, "''");
		branches.push(`WHEN '${safeId}' THEN 0.0`);
	}
	return `CASE model_id\n\t\t\t\t${branches.join('\n\t\t\t\t')}\n\t\t\t\tELSE NULL\n\t\t\tEND`;
}
