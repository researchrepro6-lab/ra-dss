"""
experiment4.py -- two further policies (Sections 6.4 and 7.1), run on exactly
the cohorts, folds, selection subsamples and downstream model of experiment2.py,
so that every result is paired with the main experiment replicate by replicate.

  Static exact          the best CONSTANT order set: one panel A, the same in
                        every window, maximising sum_t Q_t(A & avail_t) subject
                        to c_t(A & avail_t) <= B_t in every window, found by
                        exhaustive search over all 2^14 panels.  Unlike the
                        static reduct it uses the true per-window prices and is
                        exact, so it isolates what time variation buys.
  RA-DSS DP-exact (w)   the exact multi-period program on the endpoint-weighted
                        objective sum_t w_t Q_t(A_t) - lambda * kappa, with
                        window weights learned on the training fold: w_t is the
                        share of the standardised coefficient mass of window t
                        in a ridge-logistic model on the full available panel
                        (mean weight 1), and lambda = gamma * mean single-
                        parameter weighted quality, as in the main experiment.

Per-window selection is invariant to positive window weights under per-window
budgets (argmax of w_t Q_t equals argmax of Q_t), so only the multi-period
program is rerun with weights.

As a check of the pairing, the static reduct is recomputed and its held-out
AUROC must equal the value stored in rep_XXX.pkl.

Usage: python experiment4.py [n_rep] [n_proc]
Outputs results/v2/ext_XXX.pkl (one file per replicate; resumable).
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtools import n_proc_arg, note_existing, run_all
import casestudy as CS
import engine as E
from experiment2 import BUDGETS, GAMMA, N_FOLD, N_PATIENTS, OUTCOME_NOISE, OUT

INF = float("inf")


class _Weighted:
    """A slice whose objective is scaled by a window weight."""

    def __init__(self, s, w):
        self._s, self.L, self.cost = s, w * s.L, s.cost

    def feasible(self, B):
        return self._s.feasible(B)


def static_exact(sls, B):
    val = np.zeros(E.NM)
    ok = np.ones(E.NM, dtype=bool)
    for s in sls:
        At = E.ALLM & s.avail_mask
        val += s.L[At]
        ok &= s.feasible(B)[At]
    A = int(np.argmax(np.where(ok, val, -INF)))
    return [A & s.avail_mask for s in sls]


def endpoint_weights(coh, tr, C, seed):
    pairs = [(w, e) for w in range(C.shape[0]) for e in range(C.shape[1]) if C[w, e] < INF]
    Xtr = coh.X[tr]
    D = np.column_stack([Xtr[:, w, e].astype(float) for w, e in pairs])
    beta = CS.fit_logreg_cv(D, coh.y[tr], seed=seed)[1:]
    mass = np.zeros(C.shape[0])
    for (w, e), b, sd in zip(pairs, beta, D.std(axis=0)):
        mass[w] += abs(b) * sd
    return mass / mass.mean() if mass.sum() > 0 else np.ones(C.shape[0])


def run_rep(rep: int):
    path = os.path.join(OUT, f"ext_{rep:03d}.pkl")
    if os.path.exists(path):
        return path
    t0 = time.time()
    coh = CS.make_cohort(n=N_PATIENTS, seed=100 + rep, outcome_noise=OUTCOME_NOISE)
    C = CS.cost_matrix()
    folds = CS.kfold(N_PATIENTS, N_FOLD, seed=500 + rep)
    rows, wrows = [], []
    for fi, (tr, te) in enumerate(folds):
        sls = E.build_slices(coh, tr, C, sub_seed=rep)
        sig = [s.sigma for s in sls]
        tsl = E.build_slices(coh, te, C, sigmas=sig, subsample=False)
        qmax_tr = sum(s.qmax() for s in sls)
        qmax_te = sum(s.qmax() for s in tsl)
        wts = endpoint_weights(coh, tr, C, seed=fi)
        wrows.append(dict(rep=rep, fold=fi, **{f"w{t}": float(v) for t, v in enumerate(wts)}))
        wsl = [_Weighted(s, w) for s, w in zip(sls, wts)]
        lam = GAMMA * E.qbar(sls)
        lam_w = GAMMA * E.qbar(wsl)
        cache = {}

        def ev(pol):
            key = tuple(pol)
            if key not in cache:
                cache[key] = E.evaluate(coh, tr, te, pol, seed=fi)
            return cache[key]

        for bf in BUDGETS:
            B = bf * CS.FULL_PANEL_COST
            pols = {
                "Static reduct (check)": E.static_pooled(sls, B),
                "Static exact": static_exact(sls, B),
                "RA-DSS DP-exact (w)": E.dp_exact(wsl, B, lam_w)[0],
            }
            for m, pol in pols.items():
                auc, _ = ev(pol)
                rows.append(dict(
                    rep=rep, fold=fi, budget_frac=bf, method=m, auroc=auc,
                    cost=E.realised_cost(pol, C),
                    q_train=sum(s.L[p] for s, p in zip(sls, pol)) / qmax_tr,
                    q_test=sum(s.L[p] for s, p in zip(tsl, pol)) / qmax_te,
                    switches=E.n_switches(pol),
                    n_tests=int(sum(E.POP[p] for p in pol)),
                    J=E.J_value(sls, pol, lam),
                    masks=list(pol)))
    res = dict(rows=rows, weights=wrows, seconds=time.time() - t0)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(path + ".tmp", path)
    print(f"ext rep {rep} done in {time.time() - t0:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_proc = n_proc_arg(2)
    note_existing(OUT, "ext", n_rep)
    run_all(run_rep, range(n_rep), n_proc)
    print("all done")
