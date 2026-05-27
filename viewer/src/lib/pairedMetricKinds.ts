export const PAIRED_METRIC_KIND_DEFS = {
	denominator_base: { label: 'denominator base', family: 'base' },
	arch_arch_exact_label: { label: 'arch↔arch exact label', family: 'arch-arch' },
	arch_arch_support_recode: { label: 'arch↔arch support recode', family: 'arch-arch' },
	arch_arch_score_posture: { label: 'arch↔arch score posture', family: 'arch-arch' },
	arch_arch_resource: { label: 'arch↔arch resource', family: 'arch-arch' },
	arch_indra_residual: { label: 'arch↔INDRA residual', family: 'arch-indra' },
	integrity_gate: { label: 'integrity gate', family: 'gate' },
	paired_nonoverlap: { label: 'paired non-overlap', family: 'gate' },
	architecture_native: { label: 'architecture-native', family: 'native' },
	architecture_native_trace_health: { label: 'architecture-native trace health', family: 'native' }
} as const;

export type PairedMetricKind = keyof typeof PAIRED_METRIC_KIND_DEFS;
export type PairedMetricKindFamily = (typeof PAIRED_METRIC_KIND_DEFS)[PairedMetricKind]['family'];
export type PairedMetricKindClassFamily = PairedMetricKindFamily | 'unknown';

export const PAIRED_METRIC_KINDS = Object.keys(PAIRED_METRIC_KIND_DEFS) as PairedMetricKind[];

export function isPairedMetricKind(value: string): value is PairedMetricKind {
	return Object.prototype.hasOwnProperty.call(PAIRED_METRIC_KIND_DEFS, value);
}

function pairedMetricKindDef(kind: string): (typeof PAIRED_METRIC_KIND_DEFS)[PairedMetricKind] | null {
	return isPairedMetricKind(kind) ? PAIRED_METRIC_KIND_DEFS[kind] : null;
}

export function pairedMetricKindLabel(kind: string): string {
	return pairedMetricKindDef(kind)?.label ?? `unknown metric kind: ${kind || 'blank'}`;
}

export function pairedMetricKindFamily(kind: string): PairedMetricKindClassFamily {
	return pairedMetricKindDef(kind)?.family ?? 'unknown';
}

export const PANEL_APPLICABILITY_KINDS = [
	'arch_blind',
	'arch_conditioned',
	'paired_only',
	'not_defined'
] as const;

export type PanelApplicabilityKind = (typeof PANEL_APPLICABILITY_KINDS)[number];
export type PanelApplicabilityClass = PanelApplicabilityKind | 'unknown';

export function isPanelApplicabilityKind(value: string): value is PanelApplicabilityKind {
	return (PANEL_APPLICABILITY_KINDS as readonly string[]).includes(value);
}

export function panelApplicabilityClass(value: string): PanelApplicabilityClass {
	return isPanelApplicabilityKind(value) ? value : 'unknown';
}

export function panelApplicabilityLabel(value: string): string {
	if (value === 'paired_only') return '|| paired';
	if (value === 'arch_conditioned') return '. arch';
	if (value === 'not_defined') return '-- n/d';
	if (value === 'arch_blind') return '.. all';
	return `unknown scope: ${value || 'blank'}`;
}

export const PAIRED_LEDGER_ROLES = ['root', 'base', 'metric', 'gate', 'native'] as const;

export type PairedLedgerRole = (typeof PAIRED_LEDGER_ROLES)[number];
export type PairedLedgerRoleClass = PairedLedgerRole | 'unknown';

export function isPairedLedgerRole(value: string): value is PairedLedgerRole {
	return (PAIRED_LEDGER_ROLES as readonly string[]).includes(value);
}

export function pairedLedgerRoleClass(value: string): PairedLedgerRoleClass {
	return isPairedLedgerRole(value) ? value : 'unknown';
}

export function pairedLedgerRoleLabel(value: string): string {
	if (value === 'root') return 'source';
	if (value === 'base') return 'base';
	if (value === 'metric') return '↳';
	if (value === 'gate') return 'gate';
	if (value === 'native') return 'native';
	return `unknown role: ${value || 'blank'}`;
}
