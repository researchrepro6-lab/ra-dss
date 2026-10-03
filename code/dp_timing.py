"""Timing of the exact multi-period DP (distance transform vs naive pairwise
maximisation) and of the objective lattice (pattern transform vs direct
evaluation).  Output: results/v2/dp_timing.pkl.  Used for fig_dp_timing."""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

import lattice as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "v2", "dp_timing.pkl")
T = 6
K = 3


def naive_multiperiod(Qlat, feas, n, lam, init_mask=0):
    N = 1 << n
    idx = np.arange(N)
    ham = L.popcount(idx[:, None] ^ idx[None, :]).astype(np.float64)  # N x N
    V = np.where(feas[0], Qlat[0] - lam * ham[init_mask], -np.inf)
    for t in range(1, len(Qlat)):
        G = np.max(V[:, None] - lam * ham, axis=0)
        V = np.where(feas[t], Qlat[t] + G, -np.inf)
    return float(V.max())


def instance(n, rng):
    Q, F = [], []
    for _ in range(T):
        pos = rng.integers(0, 1 << n, 400)
        neg = rng.integers(0, 1 << n, 800)
        c = L.pattern_counts(pos, neg, n)
        q = L.q_lattice_fast(c, n, lambda d: np.minimum(d, K), K)
        cost = rng.uniform(5, 60, n)
        mc = L.mask_costs(cost, n)
        F.append(mc <= 0.3 * cost.sum())
        Q.append(q / q.max())
    return Q, F


def direct_q(c, n):
    N = 1 << n
    s = np.nonzero(c)[0]
    w = c[s]
    out = np.empty(N)
    for A in range(N):
        out[A] = np.dot(w, np.minimum(L.popcount(s & A), K))
    return out


def best_of(f, reps=3):
    b = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t0)
    return b, r


def main():
    rng = np.random.default_rng(7)
    rows = []
    for n in range(4, 15):
        for inst in range(3):
            Q, F = instance(n, rng)
            t_dt, (_, v_dt) = best_of(lambda: L.exact_multiperiod(Q, F, n, 0.05))
            r = dict(n=n, inst=inst, T=T, t_dt=t_dt, t_naive=np.nan, agree=np.nan,
                     t_pt=np.nan, t_direct=np.nan)
            if n <= 11:
                t_nv, v_nv = best_of(lambda: naive_multiperiod(Q, F, n, 0.05), 1)
                r.update(t_naive=t_nv, agree=float(abs(v_nv - v_dt) < 1e-9))
            pos = rng.integers(0, 1 << n, 400)
            neg = rng.integers(0, 1 << n, 800)
            c = L.pattern_counts(pos, neg, n)
            t_pt, q1 = best_of(lambda: L.q_lattice_fast(c, n, lambda d: np.minimum(d, K), K))
            r["t_pt"] = t_pt
            if n <= 12:
                t_dir, q2 = best_of(lambda: direct_q(c, n), 1)
                r["t_direct"] = t_dir
                r["agree_q"] = float(np.allclose(q1, q2))
            rows.append(r)
            print(r, flush=True)
    # linear growth in the horizon at n = 12
    n = 12
    for TT in (2, 4, 6, 8, 12, 16, 24):
        Q, F = instance(n, rng)
        Q = (Q * ((TT + T - 1) // T))[:TT]; F = (F * ((TT + T - 1) // T))[:TT]
        t_dt, _ = best_of(lambda: L.exact_multiperiod(Q, F, n, 0.05))
        rows.append(dict(n=n, inst=-1, T=TT, t_dt=t_dt))
        print(TT, t_dt, flush=True)
    pickle.dump(rows, open(OUT, "wb"))


if __name__ == "__main__":
    main()
