/**
 * A broad, deliberately diverse panel of biological agents (gene/protein symbols)
 * used to sample evidence widely from the live INDRA DB. There is no true
 * "random statement" endpoint, so broad-random sampling is approximated as
 *   random agent  ×  random offset  ×  random statement  ×  random evidence.
 * The wider and more cross-pathway this pool, the broader the sample. These are
 * all high-coverage human symbols (kinases, TFs, receptors, GPCRs, cytokines,
 * tumour suppressors, oncogenes, metabolic + signalling nodes, epigenetic
 * machinery) — chosen because each is near-certain to have many INDRA statements,
 * which keeps sampling from dead-ending on sparse agents.
 *
 * Server-only ($lib/server is never shipped to the browser).
 */
export const AGENT_POOL: readonly string[] = [
	// tumour suppressors / oncogenes
	'TP53', 'RB1', 'PTEN', 'APC', 'VHL', 'BRCA1', 'BRCA2', 'CDKN2A', 'NF1', 'NF2',
	'MYC', 'MYCN', 'KRAS', 'HRAS', 'NRAS', 'BRAF', 'RAF1', 'EGFR', 'ERBB2', 'MET',
	'ALK', 'RET', 'KIT', 'PDGFRA', 'FLT3', 'ABL1', 'BCR', 'JAK2', 'MDM2', 'CCND1',
	// PI3K / AKT / mTOR
	'PIK3CA', 'PIK3CB', 'PIK3R1', 'AKT1', 'AKT2', 'AKT3', 'MTOR', 'RICTOR', 'RPTOR',
	'TSC1', 'TSC2', 'PDPK1', 'GSK3B', 'FOXO1', 'FOXO3', 'RHEB',
	// MAPK / ERK
	'MAPK1', 'MAPK3', 'MAPK8', 'MAPK14', 'MAP2K1', 'MAP2K2', 'MAP3K1', 'DUSP1',
	'ELK1', 'RPS6KB1', 'RPS6KA1',
	// cell cycle / apoptosis
	'CDK1', 'CDK2', 'CDK4', 'CDK6', 'CCNE1', 'CCNB1', 'CDKN1A', 'CDKN1B', 'E2F1',
	'BAX', 'BCL2', 'BCL2L1', 'BAD', 'CASP3', 'CASP8', 'CASP9', 'BID', 'BAK1',
	'CYCS', 'APAF1', 'XIAP', 'BIRC5', 'PARP1',
	// transcription factors
	'STAT1', 'STAT3', 'STAT5A', 'NFKB1', 'RELA', 'JUN', 'FOS', 'SP1', 'CREB1',
	'HIF1A', 'EPAS1', 'SMAD2', 'SMAD3', 'SMAD4', 'GATA1', 'GATA3', 'RUNX1',
	'FOXP3', 'SOX2', 'POU5F1', 'NANOG', 'KLF4', 'TFEB', 'YAP1', 'WWTR1', 'TEAD1',
	// receptors / signalling
	'INSR', 'IGF1R', 'FGFR1', 'FGFR2', 'NOTCH1', 'NOTCH2', 'WNT1', 'CTNNB1',
	'GLI1', 'SMO', 'PTCH1', 'TGFBR1', 'TGFBR2', 'BMPR1A', 'ESR1', 'AR', 'PGR',
	'NR3C1', 'PPARG', 'VDR', 'RXRA',
	// immune / cytokines
	'TNF', 'IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL17A', 'IFNG', 'TGFB1',
	'CXCL8', 'CCL2', 'TLR4', 'MYD88', 'IRF3', 'IRF7', 'CD4', 'CD8A', 'CD28',
	'CTLA4', 'PDCD1', 'CD274', 'FOXM1', 'NLRP3', 'CGAS', 'TMEM173',
	// kinases / phosphatases (broad)
	'SRC', 'LCK', 'FYN', 'SYK', 'BTK', 'PRKCA', 'PRKCD', 'PRKACA', 'CAMK2A',
	'CDK5', 'GSK3A', 'CSNK2A1', 'PLK1', 'AURKA', 'AURKB', 'CHEK1', 'CHEK2',
	'ATM', 'ATR', 'WEE1', 'PTPN11', 'PTPN1', 'DUSP6',
	// ubiquitin / proteostasis / autophagy
	'UBE2I', 'CUL1', 'SKP2', 'FBXW7', 'BTRC', 'NEDD4', 'STUB1', 'MAP1LC3B',
	'SQSTM1', 'BECN1', 'ATG5', 'ATG7', 'HSPA5', 'HSP90AA1', 'VCP',
	// epigenetics / chromatin
	'EZH2', 'DNMT1', 'DNMT3A', 'TET2', 'HDAC1', 'HDAC2', 'SIRT1', 'KAT2B',
	'EP300', 'CREBBP', 'BRD4', 'KMT2A', 'SETD2', 'ARID1A', 'SMARCA4',
	// metabolism
	'HK2', 'PKM', 'LDHA', 'GAPDH', 'IDH1', 'IDH2', 'SDHB', 'FH', 'PRKAA1',
	'ACACA', 'FASN', 'SREBF1', 'G6PD', 'GLS', 'SLC2A1', 'ALDOA',
	// neuro / misc high-coverage
	'APP', 'MAPT', 'SNCA', 'PSEN1', 'BACE1', 'GRIN1', 'BDNF', 'HTT', 'LRRK2',
	'PARK2', 'PINK1', 'SOD1', 'TARDBP', 'FUS',
	// RNA-binding / splicing (incl. the user's example)
	'RBFOX1', 'RBFOX2', 'ELAVL1', 'PTBP1', 'SRSF1', 'HNRNPA1', 'MBNL1', 'TARBP2',
	// vasculature / GPCR / other
	'VEGFA', 'KDR', 'FLT1', 'EDN1', 'AGTR1', 'ADRB2', 'DRD2', 'HTR2A', 'CNR1',
	'OPRM1', 'GNAS', 'GNAQ', 'ARRB1', 'CXCR4'
];
