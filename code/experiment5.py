"""
experiment5.py -- a soft-set parameter-reduction baseline, run on exactly the
cohorts, folds and downstream model of experiment2.py (paired replicate by
replicate).

  SS-reduct   budgeted adaptation of normal parameter reduction for soft sets
              (Chen et al. 2005; Zhan and Alcantud 2019).  Normal parameter
              reduction removes parameters while keeping the choice values
              c_x(A) = sum_{e in A} 1[x in F_t(e)] of all objects in the same
              order.  Under a budget an exact normal reduction rarely exists, so
              per window we start from every available assay and remove, one at
              a time, the assay whose removal keeps the ordering of the training
              objects' choice values closest to the full ordering (Spearman rank
              correlation; ties broken towards the more expensive assay), until
              the panel fits the window budget.  Like the soft-set literature it
              uses the readings only, not the outcome.

Usage: python experiment5.py [n_rep] [n_proc]
Outputs results/v2/ssr_XXX.pkl (one file per replicate; resumable).
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtools import n_proc_arg, note_existing, run_all
import casestudy as CS
import engine as E
from experiment2 import BUDGETS, GAMMA, N_FOLD, N_PATIENTS, OUTCOME_NOISE, OUT
from lattice import to_mask

INF = float("inf")


def _spearman(r_full, c):
    r = rankdata(c)
    if r.std() == 0 or r_full.std() == 0:
        return 0.0
    return float(np.corrcoef(r_full, r)[0, 1])


def ss_reduct_window(Xw, cost_row, B):
    """Xw: (n_obj, n_par) 0/1 readings of one window on the training fold."""
    A = [e for e in range(len(cost_row)) if cost_row[e] < INF]
    if not A:
        return 0
    r_full = rankdata(Xw[:, A].sum(axis=1))
    while sum(cost_row[e] for e in A) > B + 1e-9 and A:
        best, best_key = None, None
        for e in A:
            rest = [f for f in A if f != e]
            rho = _spearman(r_full, Xw[:, rest].sum(axis=1)) if rest else -1.0
            key = (rho, cost_row[e])
            if best_key is None or key > best_key:
                best, best_key = e, key
        A.remove(best)
    return to_mask(A)


def run_rep(rep: int):
    path = os.path.join(OUT, f"ssr_{rep:03d}.pkl")
    if os.path.exists(path):
        return path
    t0 = time.time()
    coh = CS.make_cohort(n=N_PATIENTS, seed=100 + rep, outcome_noise=OUTCOME_NOISE)
    C = CS.cost_matrix()
    folds = CS.kfold(N_PATIENTS, N_FOLD, seed=500 + rep)
    rows = []
    for fi, (tr, te) in enumerate(folds):
        sls = E.build_slices(coh, tr, C, sub_seed=rep)
        sig = [s.sigma for s in sls]
        tsl = E.build_slices(coh, te, C, sigmas=sig, subsample=False)
        qmax_tr = sum(s.qmax() for s in sls)
        qmax_te = sum(s.qmax() for s in tsl)
        lam = GAMMA * E.qbar(sls)
        Xtr = coh.X[tr].astype(np.int8)
        cache = {}
        for bf in BUDGETS:
            B = bf * CS.FULL_PANEL_COST
            pol = [ss_reduct_window(Xtr[:, w, :], C[w], B) for w in range(C.shape[0])]
            key = tuple(pol)
            if key not in cache:
                cache[key] = E.evaluate(coh, tr, te, pol, seed=fi)
            auc, _ = cache[key]
            rows.append(dict(
                rep=rep, fold=fi, budget_frac=bf, method="SS-reduct", auroc=auc,
                cost=E.realised_cost(pol, C),
                q_train=sum(s.L[p] for s, p in zip(sls, pol)) / qmax_tr,
                q_test=sum(s.L[p] for s, p in zip(tsl, pol)) / qmax_te,
                switches=E.n_switches(pol),
                n_tests=int(sum(E.POP[p] for p in pol)),
                J=E.J_value(sls, pol, lam), masks=list(pol)))
    res = dict(rows=rows, seconds=time.time() - t0)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(path + ".tmp", path)
    print(f"ssr rep {rep} done in {time.time() - t0:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_proc = n_proc_arg(2)
    note_existing(OUT, "ssr", n_rep)
    run_all(run_rep, range(n_rep), n_proc)
    print("all done")
