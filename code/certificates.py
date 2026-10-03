"""
certificates.py -- how tight are the three a-posteriori upper bounds on the
case-study slices, where the exact multi-period optimum is known?

  alpha   OPT <= alpha^{-1} sum_t Q_t(A_t^greedy), alpha = 0.427 (Feldman et al.)
  dd      OPT <= sum_t U_t (per-period data-dependent bound, Leskovec et al.)
  lag     OPT <= min_mu Lambda(mu) (switching-aware Lagrangian bound)
For each we report UB/OPT (1 = exact) and the certificate J(myopic)/UB.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtools import n_proc_arg, run_all
import casestudy as CS
import engine as E
from bounds import lagrangian_bound

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "v2")
ALPHA = 0.427
BUDGETS = (0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
GAMMAS = (0.2, 1.0)


def job(rep):
    coh = CS.make_cohort(n=3000, seed=100 + rep, outcome_noise=2.8)
    C = CS.cost_matrix()
    rows = []
    for fi, (tr, te) in enumerate(CS.kfold(3000, 5, seed=500 + rep)):
        sls = E.build_slices(coh, tr, C, sub_seed=rep)
        qb = E.qbar(sls)
        for bf in BUDGETS:
            B = bf * CS.FULL_PANEL_COST
            myo = [E.guarded(s, B) for s in sls]
            sumQ = sum(s.L[m] for s, m in zip(sls, myo))
            U = sum(E.data_dependent_bound(s, B, m) for s, m in zip(sls, myo))
            for g in GAMMAS:
                lam = g * qb
                _, opt = E.dp_exact(sls, B, lam)
                Jm = E.J_value(sls, myo, lam)
                lag = lagrangian_bound([s.L for s in sls], C, [B] * len(sls),
                                       lam, myo)
                ub_a = sumQ / ALPHA
                rows.append(dict(rep=rep, fold=fi, budget_frac=bf, gamma=g,
                                 opt=opt, J_myopic=Jm,
                                 ub_alpha=ub_a, ub_dd=U, ub_lag=lag,
                                 r_alpha=ub_a / opt, r_dd=U / opt, r_lag=lag / opt,
                                 cert_alpha=Jm / ub_a, cert_dd=Jm / U,
                                 cert_lag=Jm / lag, true_ratio=Jm / opt))
    return rows


if __name__ == "__main__":
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    res = run_all(job, range(n_rep), n_proc_arg(2), ordered=True)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(OUT, "certificates.pkl"), "wb") as f:
        pickle.dump(rows, f)
    print("certificates done", len(rows))
