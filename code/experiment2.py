"""
experiment2.py -- the main experiment (50 cohort replicates).

For every replicate r, fold f and per-window budget fraction b:
  * all selection policies are run on the TRAINING fold only;
  * the downstream ridge-logistic model (penalty by inner 3-fold CV) is fitted
    on the training fold and scored on the held-out fold (AUROC);
  * the selection objective is reported both in-sample (training subsample used
    for selection) and HELD-OUT (test fold, training orientation sigma);
  * out-of-fold predictions are pooled per replicate for DeLong tests.
Per replicate and fold, at the 10% budget, the switching sweep and the exact
multi-period optimum are also recorded.

Usage: python experiment2.py [n_rep] [n_proc]
Outputs results/v2/rep_XXX.pkl (one file per replicate; resumable).
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
from lattice import from_mask

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "v2")
os.makedirs(OUT, exist_ok=True)

N_PATIENTS = 3000
N_FOLD = 5
OUTCOME_NOISE = 2.8
BUDGETS = [0.05, 0.0625, 0.075, 0.0875, 0.10, 0.1125, 0.125, 0.15, 0.175,
           0.20, 0.25, 0.30, 0.40, 0.60, 1.00]
WRAPPER_MAX = 0.40
GAMMA = 0.20
GAMMAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
N_RAND = 3
METHODS = ["Full panel", "Random (budget)", "Cost-blind greedy",
           "Static reduct", "TCS-reduct", "L1-cost", "Wrapper",
           "NB-VOI greedy", "GA", "RA-DSS myopic", "RA-DSS myopic+enum",
           "RA-DSS exact/window", "RA-DSS DP", "RA-DSS DP-exact"]
ALPHA = 1.0 - np.exp(-0.5)


def policies_for(coh, tr, sls, C, B, bf, rep, fi):
    Xtr, ytr = coh.X[tr], coh.y[tr]
    lam = GAMMA * E.qbar(sls)
    rng = np.random.default_rng(7000 + 97 * rep + 13 * fi + int(1000 * bf))
    pols = {
        "Full panel": [s.avail_mask for s in sls],
        "Cost-blind greedy": [E.greedy(s, B, density=False) for s in sls],
        "Static reduct": E.static_pooled(sls, B),
        "TCS-reduct": [E.tcs_reduct_window(Xtr, ytr, w, C[w], B)
                       for w in range(len(sls))],
        "L1-cost": E.l1_cost(Xtr, ytr, C, B),
        "NB-VOI greedy": E.nb_voi(Xtr, ytr, C, B),
        "GA": [E.ga_window(s, B, rng) for s in sls],
        "RA-DSS myopic": [E.guarded(s, B) for s in sls],
        "RA-DSS myopic+enum": [E.partial_enum(s, B, 1) for s in sls],
        "RA-DSS exact/window": [E.exact_window(s, B) for s in sls],
        "RA-DSS DP": E.dp_restricted(sls, B, lam)[0],
        "RA-DSS DP-exact": E.dp_exact(sls, B, lam)[0],
    }
    if bf <= WRAPPER_MAX + 1e-12:
        pols["Wrapper"] = E.wrapper(Xtr, ytr, C, B, seed=rep * 10 + fi)
    rand = [[E.random_policy(s, B, rng) for s in sls] for _ in range(N_RAND)]
    return pols, rand, lam


def run_rep(rep: int):
    path = os.path.join(OUT, f"rep_{rep:03d}.pkl")
    if os.path.exists(path):
        return path
    t0 = time.time()
    coh = CS.make_cohort(n=N_PATIENTS, seed=100 + rep, outcome_noise=OUTCOME_NOISE)
    C = CS.cost_matrix()
    folds = CS.kfold(N_PATIENTS, N_FOLD, seed=500 + rep)
    rows, mp_rows, lam_rows, sel_rows, win_rows = [], [], [], [], []
    oof = {}          # (bf, method) -> array of out-of-fold predictions
    for fi, (tr, te) in enumerate(folds):
        sls = E.build_slices(coh, tr, C, sub_seed=rep)
        sig = [s.sigma for s in sls]
        tsl = E.build_slices(coh, te, C, sigmas=sig, subsample=False)
        qmax_tr = sum(s.qmax() for s in sls)
        qmax_te = sum(s.qmax() for s in tsl)
        cache = {}

        def ev(pol):
            key = tuple(pol)
            if key not in cache:
                cache[key] = E.evaluate(coh, tr, te, pol, seed=fi)
            return cache[key]

        for bf in BUDGETS:
            B = bf * CS.FULL_PANEL_COST
            pols, rand, lam = policies_for(coh, tr, sls, C, B, bf, rep, fi)
            for m, pol in pols.items():
                auc, pred = ev(pol)
                oof.setdefault((bf, m), np.zeros(N_PATIENTS))[te] = pred
                rows.append(dict(
                    rep=rep, fold=fi, budget_frac=bf, method=m, auroc=auc,
                    cost=E.realised_cost(pol, C),
                    q_train=sum(s.L[p] for s, p in zip(sls, pol)) / qmax_tr,
                    q_test=sum(s.L[p] for s, p in zip(tsl, pol)) / qmax_te,
                    switches=E.n_switches(pol),
                    n_tests=int(sum(E.POP[p] for p in pol)),
                    J=E.J_value(sls, pol, lam)))
            ra, rc, rqt, rqs = [], [], [], []
            for pol in rand:
                auc, _ = ev(pol)
                ra.append(auc)
                rc.append(E.realised_cost(pol, C))
                rqt.append(sum(s.L[p] for s, p in zip(sls, pol)) / qmax_tr)
                rqs.append(sum(s.L[p] for s, p in zip(tsl, pol)) / qmax_te)
            rows.append(dict(rep=rep, fold=fi, budget_frac=bf,
                             method="Random (budget)", auroc=float(np.mean(ra)),
                             cost=float(np.mean(rc)), q_train=float(np.mean(rqt)),
                             q_test=float(np.mean(rqs)), switches=np.nan,
                             n_tests=np.nan, J=np.nan))
            # --- multi-period optimum, certificates, per-window gaps -------
            myo = pols["RA-DSS myopic"]
            Jex = E.J_value(sls, pols["RA-DSS DP-exact"], lam)
            Jr = E.J_value(sls, pols["RA-DSS DP"], lam)
            Jm = E.J_value(sls, myo, lam)
            sumQ = sum(s.L[p] for s, p in zip(sls, myo))
            U = sum(E.data_dependent_bound(s, B, p) for s, p in zip(sls, myo))
            mp_rows.append(dict(rep=rep, fold=fi, budget_frac=bf, lam=lam,
                                J_exact=Jex, J_restricted=Jr, J_myopic=Jm,
                                sumQ_myopic=sumQ, U_dd=U,
                                cert_alpha=(ALPHA * Jm / sumQ) if (Jm > 0 and sumQ > 0) else np.nan,
                                cert_dd=(Jm / U) if (Jm > 0 and U > 0) else np.nan))
            for w, (s, p) in enumerate(zip(sls, myo)):
                opt = s.L[pols["RA-DSS exact/window"][w]]
                if opt <= 0:
                    continue
                Uw = E.data_dependent_bound(s, B, p)
                win_rows.append(dict(rep=rep, fold=fi, budget_frac=bf, window=w,
                                     r_guarded=s.L[p] / opt,
                                     r_enum1=s.L[pols["RA-DSS myopic+enum"][w]] / opt,
                                     r_blind=s.L[pols["Cost-blind greedy"][w]] / opt,
                                     r_ga=s.L[pols["GA"][w]] / opt,
                                     bound_ratio=s.L[p] / Uw if Uw > 0 else np.nan,
                                     opt_over_U=opt / Uw if Uw > 0 else np.nan))
            if abs(bf - 0.10) < 1e-12:
                for m in ("Static reduct", "RA-DSS myopic", "RA-DSS myopic+enum",
                          "RA-DSS exact/window", "RA-DSS DP", "RA-DSS DP-exact",
                          "TCS-reduct", "NB-VOI greedy", "Cost-blind greedy"):
                    sel_rows.append(dict(rep=rep, fold=fi, method=m,
                                         masks=list(pols[m])))
                # switching sweep with both DPs
                pool = E.candidate_pool(sls, B)
                qb = E.qbar(sls)
                for g in GAMMAS:
                    for lab, fn in (("restricted", lambda l: E.dp_restricted(sls, B, l, pool)[0]),
                                    ("exact", lambda l: E.dp_exact(sls, B, l)[0])):
                        pol = fn(g * qb)
                        auc, _ = ev(pol)
                        lam_rows.append(dict(
                            rep=rep, fold=fi, gamma=g, dp=lab, auroc=auc,
                            q_train=sum(s.L[p] for s, p in zip(sls, pol)) / qmax_tr,
                            q_test=sum(s.L[p] for s, p in zip(tsl, pol)) / qmax_te,
                            switches=E.n_switches(pol),
                            n_distinct=len(set(pol)),
                            J=E.J_value(sls, pol, g * qb),
                            cost=E.realised_cost(pol, C)))
    # --- replicate-level pooled out-of-fold AUROC and DeLong tests ----------
    dl_rows = []
    y = coh.y
    for bf in BUDGETS:
        ref = oof.get((bf, "Static reduct"))
        ref2 = oof.get((bf, "RA-DSS myopic"))
        for m in METHODS:
            if (bf, m) not in oof:
                continue
            s = oof[(bf, m)]
            a1, a0, p = E.delong(y, s, ref)
            _, a2, p2 = E.delong(y, s, ref2)
            dl_rows.append(dict(rep=rep, budget_frac=bf, method=m, auc_oof=a1,
                                auc_ref_static=a0, p_vs_static=p,
                                auc_ref_myopic=a2, p_vs_myopic=p2))
    res = dict(rows=rows, mp=mp_rows, lam=lam_rows, sel=sel_rows,
               win=win_rows, delong=dl_rows, seconds=time.time() - t0)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(path + ".tmp", path)
    print(f"rep {rep} done in {time.time() - t0:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_proc = n_proc_arg(2)
    note_existing(OUT, "rep", n_rep)
    run_all(run_rep, range(n_rep), n_proc)
    print("all done")
