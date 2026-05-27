// Canonical SQL escape helper.
//
// CONTRACT: `sqlQuote(s)` doubles single quotes inside `s`. Use it ONLY
// when interpolating an untrusted string into a single-quoted string
// literal in DuckDB SQL (e.g. `WHERE name='${sqlQuote(input)}'`).
//
// NOT a general-purpose escape. It does not:
//  - handle identifiers (use proper parameter binding or strict validation)
//  - escape LIKE-pattern metachars (% and _)
//  - escape backslashes or byte sequences (DuckDB-native usage assumes
//    the host string is well-formed text)
//
// Six independent copies existed before this consolidation; do not
// re-introduce a local copy. If a use case requires more escaping (LIKE,
// identifiers), add a new helper here next to this one — never patch
// `sqlQuote` in isolation.
export function sqlQuote(s: string): string {
	return s.replace(/'/g, "''");
}
