"""
experiment6.py -- a case study with graded (fuzzy) readings, in which the
graded separation of Proposition 4.6 is applied to the case-study cohort
(Section 7.4.3).

Graded readings.  The crisp reading of the main cohort is
X = 1[u < sigma(lin)] with u ~ U(0,1) (casestudy.make_cohort).  Using the same
uniforms we define the graded reading
    mu = sigma(lin - logit(u))  in (0, 1),
the degree to which the assay reads abnormal.  Thresholding mu at 1/2 returns
exactly the crisp reading X of the main cohort, so the graded cohort refines the
crisp one without changing it; this script checks that X is reproduced.

Graded objective (Proposition 4.6, sum pooling): for an oriented positive-
negative pair (p, q), d_e = (mu_pe - mu_qe)^+ if sigma(e) = 1 and
(mu_qe - mu_pe)^+ otherwise, and Q^S(A) = sum_(p,q) min(sum_{e in A} d_e, 3).
It is tabulated on all 2^14 panels by a Gray-code pass over a subsample of 120
positive and 240 negative training objects per window.

Compared, per window budget and on the same folds as the main experiment:
  crisp exact/window   per-window optimum of the crisp objective (thresholded
                       readings), as in the main experiment
  graded exact/window  per-window optimum of the graded objective Q^S
  crisp DP-exact, graded DP-exact   the exact multi-period programs, gamma 0.2
Downstream AUROC is reported with graded features (the readings actually
available when they are graded) and with crisp features.

Usage: python experiment6.py [n_rep] [n_proc]   -> results/v2/grd_XXX.pkl
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtools import n_proc_arg, note_existing, run_all
import casestudy as CS
import engine as E
from experiment2 import GAMMA, N_FOLD, N_PATIENTS, OUTCOME_NOISE, OUT
from lattice import from_mask

BUDGETS_G = [0.075, 0.10, 0.15, 0.20, 0.30]
K = 3
N_POS, N_NEG = 120, 240
NP = CS.N_PAR


def make_graded_cohort(n, seed, outcome_noise, sharp=1.0, obs_noise=1.0, target_pos=0.25):
    """Replays casestudy.make_cohort's random stream and keeps the uniforms."""
    rng = np.random.default_rng(seed)
    beta = CS.loadings(sharp=sharp)
    arche = rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])
    drift = np.where(arche == 0, rng.normal(-0.18, 0.10, n),
             np.where(arche == 1, rng.normal(+0.34, 0.12, n),
                                  rng.normal(+0.04, 0.30, n)))
    W = CS.N_WIN
    z = np.zeros((n, W))
    z[:, 0] = rng.normal(0.0, 1.0, n)
    for w in range(1, W):
        z[:, w] = z[:, w - 1] + drift + rng.normal(0.0, 0.32, n)
    score = z[:, -1] + 0.45 * z[:, -2] + rng.normal(0.0, outcome_noise, n)
    thr = np.quantile(score, 1.0 - target_pos)
    y = (score > thr).astype(int)
    intercept = np.array([-0.25, -0.55, -0.40, -0.45, -0.85, -0.60,
                          -0.15, -0.30, -0.20, -0.50, -1.30, -0.95,
                          -1.05, -0.90])
    X = np.zeros((n, W, NP), dtype=np.uint8)
    G = np.zeros((n, W, NP))
    for w in range(W):
        lin = (beta[w][None, :] / obs_noise) * z[:, w][:, None] + intercept[None, :]
        p = 1.0 / (1.0 + np.exp(-lin))
        u = rng.random((n, NP))
        X[:, w, :] = (u < p).astype(np.uint8)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        G[:, w, :] = 1.0 / (1.0 + np.exp(-(lin - np.log(u / (1 - u)))))
    return X, G, y


def graded_pairs(Gw, yy, sub_seed, w, sigma=None, subsample=True):
    pos, neg = Gw[yy == 1], Gw[yy == 0]
    if subsample:
        rs = np.random.default_rng(3000 + 17 * w + sub_seed)
        pos = pos[rs.choice(pos.shape[0], min(N_POS, pos.shape[0]), replace=False)]
        neg = neg[rs.choice(neg.shape[0], min(N_NEG, neg.shape[0]), replace=False)]
    if sigma is None:
        sigma = pos.mean(axis=0) >= neg.mean(axis=0)
    diff = pos[:, None, :] - neg[None, :, :]            # (P, N, n)
    D = np.where(sigma[None, None, :], diff, -diff)
    D = np.maximum(D, 0.0).reshape(-1, NP)             # (pairs, n)
    return D, sigma


def graded_lattice(D):
    """Q^S(A) = sum_pairs min(sum_{e in A} d_e, K) for all 2^n masks (Gray code)."""
    N = 1 << NP
    L = np.zeros(N)
    S = np.zeros(D.shape[0])
    prev = 0
    for i in range(1, N):
        g = i ^ (i >> 1)
        bit = (g ^ prev).bit_length() - 1
        if (g >> bit) & 1:
            S += D[:, bit]
        else:
            S -= D[:, bit]
        L[g] = np.minimum(S, K).sum()
        prev = g
    return L


def graded_Q(D, m):
    idx = from_mask(m, NP)
    if not idx:
        return 0.0
    return float(np.minimum(D[:, idx].sum(axis=1), K).sum())


class _GSlice:
    def __init__(self, L, crisp_slice):
        self.L, self.cost, self._s = L, crisp_slice.cost, crisp_slice

    def feasible(self, B):
        return self._s.feasible(B)


def run_rep(rep: int):
    path = os.path.join(OUT, f"grd_{rep:03d}.pkl")
    if os.path.exists(path):
        return path
    t0 = time.time()
    coh = CS.make_cohort(n=N_PATIENTS, seed=100 + rep, outcome_noise=OUTCOME_NOISE)
    X, G, y = make_graded_cohort(N_PATIENTS, 100 + rep, OUTCOME_NOISE)
    assert np.array_equal(X, coh.X) and np.array_equal(y, coh.y), "graded cohort must refine the crisp one"
    gcoh = SimpleNamespace(X=G, y=y)
    C = CS.cost_matrix()
    folds = CS.kfold(N_PATIENTS, N_FOLD, seed=500 + rep)
    rows = []
    for fi, (tr, te) in enumerate(folds):
        sls = E.build_slices(coh, tr, C, sub_seed=rep)
        gl, gsig, Dte = [], [], []
        for w in range(C.shape[0]):
            D, sg = graded_pairs(G[tr][:, w, :], y[tr], rep, w)
            gl.append(_GSlice(graded_lattice(D), sls[w]))
            gsig.append(sg)
            Dt, _ = graded_pairs(G[te][:, w, :], y[te], rep, w, sigma=sg, subsample=False)
            Dte.append(Dt)
        qmax_te = sum(graded_Q(Dte[w], s.avail_mask) for w, s in enumerate(sls))
        lam_c = GAMMA * E.qbar(sls)
        lam_g = GAMMA * E.qbar(gl)
        cache_g, cache_c = {}, {}
        for bf in BUDGETS_G:
            B = bf * CS.FULL_PANEL_COST
            pols = {
                "crisp exact/window": [E.exact_window(s, B) for s in sls],
                "graded exact/window": [E.exact_window(s, B) for s in gl],
                "crisp DP-exact": E.dp_exact(sls, B, lam_c)[0],
                "graded DP-exact": E.dp_exact(gl, B, lam_g)[0],
            }
            for m, pol in pols.items():
                key = tuple(pol)
                if key not in cache_g:
                    cache_g[key] = E.evaluate(gcoh, tr, te, pol, seed=fi)[0]
                    cache_c[key] = E.evaluate(coh, tr, te, pol, seed=fi)[0]
                rows.append(dict(rep=rep, fold=fi, budget_frac=bf, method=m,
                                 auroc_graded=cache_g[key], auroc_crisp=cache_c[key],
                                 cost=E.realised_cost(pol, C),
                                 qg_test=sum(graded_Q(Dte[w], p) for w, p in enumerate(pol)) / qmax_te,
                                 n_tests=int(sum(E.POP[p] for p in pol)), masks=list(pol)))
    res = dict(rows=rows, seconds=time.time() - t0)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(path + ".tmp", path)
    print(f"grd rep {rep} done in {time.time() - t0:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_proc = n_proc_arg(2)
    note_existing(OUT, "grd", n_rep)
    run_all(run_rep, range(n_rep), n_proc)
    print("all done")
