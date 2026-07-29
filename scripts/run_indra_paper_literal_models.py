"""Run the 2023 INDRA paper's LITERAL belief-model code under indra 1.24.0.

The evaluation engine (TrainTestResult / ModelResults / eval_models_relation
and the shuffle helpers) is copied VERBATIM from the released notebook
`notebooks/Training Belief ML Models.ipynb` (cells 15 + 17). The featurizer is
indra.belief.skl.CountsScorer and the classifiers are the paper's own
bioexp.curation.classifiers. Only two things are added, both documented:

  * an explicit RandomForest random_state (the paper published no RF seed), and
  * headless result capture (fold PR-AUC table + per-statement OOF probs).

Outputs JSON to --out:
  { "table6": [ {method, fold_mean_trapezoidal_pr_auc, fold_population_sd,
                 fold_count, folds:[...]} ],
    "oof_predictions": { "<panel/method>": [ {stmt_hash, prob_correct,
                 y_true, fold_ix} ] } }
"""
import argparse
import json
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             matthews_corrcoef)

# --- paper's own code (verbatim imports the notebook uses) ---
from indra.belief.skl import CountsScorer
from indra.belief import get_ev_for_stmts_from_supports
from bioexp.curation.classifiers import (BinaryRandomForest,  # noqa: F401
                                         LogLogisticRegression)
from sklearn.ensemble import RandomForestClassifier

# Paths are arguments (see main()). CORPUS/CURATION are module-level so the
# verbatim notebook helpers can stay unchanged; main() assigns them.
CORPUS = None
CURATION = None
RF_RANDOM_STATE = 1  # paper published no RF seed; fixed for determinism.

# Provenance: this driver runs the paper's OWN code from a clone of
# github.com/sorgerlab/indra_assembly_paper pinned at commit
# 63abdf1274d2f5534ed822585775031712916c83, whose four critical files match
# data/benchmark/indra_paper_2023.manifest.json byte-for-byte. Pass its path
# via --paper-repo. Corpus digest (verified) is recorded in that manifest.


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ======================================================================
# VERBATIM notebook cell 15 (TrainTestResult / ModelResults / shuffle)
# ======================================================================
class TrainTestResult:
    def __init__(self, y_preds, y_probs, y_test, base_fpr, sample_wts):
        self.y_preds = y_preds
        self.y_probs = y_probs
        self.y_test = y_test
        self.sample_wts = sample_wts
        self.fpr, self.tpr, self.thresholds = roc_curve(
            y_test, y_probs[:, 1], sample_weight=self.sample_wts)
        self.roc_auc = auc(self.fpr, self.tpr)
        self.tpr_interp = np.interp(base_fpr, self.fpr, self.tpr)
        self.tpr_interp[0] = 0.0
        self.precision, self.recall, self.thresholds = \
            precision_recall_curve(y_test, y_probs[:, 1],
                                   sample_weight=self.sample_wts)
        self.pr_auc = auc(self.recall, self.precision)
        self.prec_interp = np.interp(base_fpr, self.thresholds,
                                     self.precision[:-1])
        self.rec_interp = np.interp(base_fpr, self.thresholds,
                                    self.recall[:-1])
        self.mcc = matthews_corrcoef(y_test, y_preds,
                                     sample_weight=self.sample_wts)


class ModelResults:
    def __init__(self, clf_name, feat_set_name, feat_kwargs, base_fpr):
        self.base_fpr = base_fpr
        self.clf_name = clf_name
        self.feat_set_name = feat_set_name
        self.feat_kwargs = feat_kwargs
        self.tt_results = []

    def add_result(self, tt_result):
        self.tt_results.append(tt_result)

    def get_summary(self):
        clf_results = [{'y_preds': tt.y_preds, 'y_probs': tt.y_probs,
                        'y_test': tt.y_test} for tt in self.tt_results]
        dim = (len(self.tt_results), len(self.base_fpr))
        tpr_arr = np.zeros(dim); prec_arr = np.zeros(dim); rec_arr = np.zeros(dim)
        for ix, ttr in enumerate(self.tt_results):
            tpr_arr[ix, :] = ttr.tpr_interp
            prec_arr[ix, :] = ttr.prec_interp
            rec_arr[ix, :] = ttr.rec_interp
        return {'clf': clf_results,
                'mcc': np.array([ttr.mcc for ttr in self.tt_results]),
                'roc': tpr_arr.mean(axis=0),
                'roc_auc': np.array([ttr.roc_auc for ttr in self.tt_results]),
                'prec': prec_arr.mean(axis=0), 'rec': rec_arr.mean(axis=0),
                'pr_auc': np.array([ttr.pr_auc for ttr in self.tt_results]),
                'x_interp': self.base_fpr}


def stmts_for_df(df, stmts_by_hash):
    return [stmts_by_hash[row.stmt_hash] for row in df.itertuples()]


def shuffle_train_df(df, stmts_by_hash, seed=1):
    stmts = stmts_for_df(df, stmts_by_hash)
    y_arr = df['correct'].values
    return shuffle_train_stmts(stmts, y_arr, seed)


def shuffle_train_stmts(stmts, y_arr, seed=1):
    random.seed(seed)
    stmt_y_pairs = list(zip(stmts, y_arr))
    random.shuffle(stmt_y_pairs)
    stmts, y_vals = list(zip(*stmt_y_pairs))
    return stmts, np.array(y_vals)


# ======================================================================
# VERBATIM notebook cell 17 (eval_models_relation) + OOF capture
# ======================================================================
def eval_models_relation(model_dict, predictors, df, readers, split_func,
                         stmts_by_hash, num_folds=10, cols_to_include=None,
                         seed=1, include_more_specific=False, use_weights=False):
    cols_to_drop = [c for c in df.columns
                    if c not in readers + ['stmt_hash', 'correct']]
    df = df.drop(cols_to_drop, axis=1)
    df = df[df[readers].any(axis=1)]  # notebook wrote .any(1); modern pandas needs axis=
    print("Readers", str(readers), "Num_rows", len(df),
          "Pct corr", df['correct'].mean())
    base_fpr = np.linspace(0, 1, 101)
    stmts, y_arr = shuffle_train_df(df, stmts_by_hash, seed=seed)
    skf = StratifiedKFold(num_folds, shuffle=False)
    skf.split(stmts, y_arr)
    model_results = {}
    oof = {}  # model_key -> list of per-stmt OOF records
    for fold_ix, (train_ix, test_ix) in enumerate(skf.split(stmts, y_arr)):
        for clf_name, clf in model_dict.items():
            for feat_set_name, feat_kwargs in predictors.items():
                if clf_name == 'Belief Orig' and feat_set_name == '+ Type/#PMIDs':
                    continue
                model = CountsScorer(clf, readers, include_more_specific,
                                     **feat_kwargs)
                model_key = '%s %s' % (clf_name, feat_set_name)
                if model_key not in model_results:
                    model_results[model_key] = ModelResults(
                        clf_name, feat_set_name, feat_kwargs, base_fpr)
                    oof[model_key] = []
                x_train_stmts = [stmts[i] for i in train_ix]
                x_test_stmts = [stmts[i] for i in test_ix]
                y_train = y_arr[train_ix]
                y_test = y_arr[test_ix]
                if include_more_specific:
                    train_evidences = get_ev_for_stmts_from_supports(x_train_stmts)
                    test_evidences = get_ev_for_stmts_from_supports(x_test_stmts)
                else:
                    train_evidences = test_evidences = None
                if use_weights:
                    train_wts = [s.weight for s in x_train_stmts]
                    test_wts = [s.weight for s in x_test_stmts]
                    model.fit(x_train_stmts, y_train,
                              extra_evidence=train_evidences,
                              sample_weight=train_wts)
                else:
                    test_wts = None
                    model.fit(x_train_stmts, y_train,
                              extra_evidence=train_evidences)
                y_preds = model.predict(x_test_stmts, extra_evidence=test_evidences)
                y_probs = model.predict_proba(x_test_stmts,
                                              extra_evidence=test_evidences)
                tt_result = TrainTestResult(y_preds, y_probs, y_test, base_fpr,
                                            test_wts)
                model_results[model_key].add_result(tt_result)
                for s, yp, yt in zip(x_test_stmts, y_probs[:, 1], y_test):
                    oof[model_key].append({
                        'stmt_hash': int(s.get_hash()),
                        'prob_correct': float(yp),
                        'y_true': int(yt), 'fold_ix': int(fold_ix)})
    return model_results, oof


def rf(): return RandomForestClassifier(n_estimators=2000, max_depth=13,
                                        random_state=RF_RANDOM_STATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper-repo", required=True,
                    help="clone of sorgerlab/indra_assembly_paper at commit "
                         "63abdf1274d2f5534ed822585775031712916c83")
    ap.add_argument("--corpus",
                    default="data/benchmark/indra_benchmark_corpus.pkl",
                    help="assembled statements pickle (bioexp_asmb_preassembled)")
    ap.add_argument("--quick", action="store_true",
                    help="RF n_estimators=200 for a fast smoke run")
    args = ap.parse_args()
    global CORPUS, CURATION, rf
    CORPUS = args.corpus
    CURATION = f"{args.paper_repo}/data/curation/extended_curation_dataset.pkl"
    if args.quick:
        rf = lambda: RandomForestClassifier(n_estimators=200, max_depth=13,
                                            random_state=RF_RANDOM_STATE)

    log("loading corpus + curation")
    with open(CORPUS, "rb") as fh:
        all_stmts = pickle.load(fh)
    stmts_by_hash = {s.get_hash(): s for s in all_stmts}
    with open(CURATION, "rb") as fh:
        cur_df = pd.DataFrame.from_records(pickle.load(fh)).fillna(0)
    dtype = {c: 'int64' for c in cur_df.columns
             if c not in ('agA_name', 'agA_ns', 'agA_id', 'stmt_type',
                          'agB_name', 'agB_ns', 'agB_id')}
    cur_df = cur_df.astype(dtype)
    cur_stmts = stmts_for_df(cur_df, stmts_by_hash)
    all_sources = list(set(ev.source_api for stmt in cur_stmts
                           for ev in stmt.evidence))
    reader_list = ['reach', 'sparser', 'medscan', 'rlimsp', 'trips']
    seed = 4
    log(f"corpus={len(all_stmts)} cur={len(cur_df)} all_sources={sorted(all_sources)}")

    table6 = []
    oof_all = {}

    def record(results, oof, suffix):
        for model_condition, obj in results.items():
            key = f"{model_condition.strip()} - {suffix}"
            pr = obj.get_summary()['pr_auc']
            table6.append({
                'method': key,
                'fold_mean_trapezoidal_pr_auc': float(np.mean(pr)),
                'fold_population_sd': float(np.std(pr)),
                'fold_count': int(len(pr)),
                'folds': [float(x) for x in pr]})
            if model_condition.strip() in oof:
                oof_all[key] = oof[model_condition.strip()]

    # ---- all sources, specific: the headline 0.942 block (cell 30 tail) ----
    predictors_head = {
        '+ Type/#PMIDs/avglen': {'use_stmt_type': True, 'use_num_pmids': True,
                                 'use_avg_evidence_len': True},
        '+ Type/#PMIDs/promoter': {'use_stmt_type': True, 'use_num_pmids': True,
                                   'use_promoter': True},
        '+ Type/#PMIDs/prom/avglen': {'use_stmt_type': True, 'use_num_pmids': True,
                                      'use_promoter': True, 'use_avg_evidence_len': True}}
    models_head = {'Log LR': LogLogisticRegression(solver='liblinear'),
                   'RF 2k-d13': rf()}
    log("PANEL: all sources, specific (headline promoter/avglen)")
    r, o = eval_models_relation(models_head, predictors_head, cur_df, all_sources,
                                None, stmts_by_hash, num_folds=10, seed=seed,
                                include_more_specific=True, use_weights=False)
    record(r, o, 'all sources, specific')

    # ---- all sources, specific: base + Type/#PMIDs (cell 30 head) ----
    predictors_base = {'': {'use_stmt_type': False, 'use_num_members': False},
                       '+ Type/#PMIDs': {'use_stmt_type': True, 'use_num_pmids': True}}
    models_base = {'Log LR': LogLogisticRegression(solver='liblinear'),
                   'RF 2k-d13': rf()}
    log("PANEL: all sources, specific (base + Type/#PMIDs)")
    r, o = eval_models_relation(models_base, predictors_base, cur_df, all_sources,
                                None, stmts_by_hash, num_folds=10, seed=seed,
                                include_more_specific=True, use_weights=False)
    record(r, o, 'all sources, specific')

    # ---- readers cross-check (cell 25): RF/LogLR base + Type/#PMIDs ----
    log("PANEL: readers (base + Type/#PMIDs) cross-check")
    r, o = eval_models_relation(
        {'Log LR': LogLogisticRegression(solver='liblinear'), 'RF 2k-d13': rf()},
        predictors_base, cur_df, reader_list, None, stmts_by_hash, num_folds=10,
        seed=seed, include_more_specific=False, use_weights=False)
    record(r, o, 'readers')

    with open(args.out, "w") as fh:
        json.dump({'rf_random_state': RF_RANDOM_STATE, 'seed': seed,
                   'quick': args.quick, 'table6': table6,
                   'oof_predictions': oof_all}, fh)
    log(f"wrote {args.out}: {len(table6)} methods, "
        f"{len(oof_all)} OOF prediction sets")
    for row in sorted(table6, key=lambda r: -r['fold_mean_trapezoidal_pr_auc']):
        log(f"  {row['method']:52s} "
            f"PR-AUC={row['fold_mean_trapezoidal_pr_auc']:.3f} "
            f"+/- {row['fold_population_sd']:.3f}")


if __name__ == "__main__":
    main()
