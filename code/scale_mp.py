"""
scale_mp.py -- the multi-period methods beyond the size where the exact
program applies (Section 7.5.2).

Synthetic instances with |T| = 6 windows, n in {20, 50, 100} parameters and
5,000 demands per window; parameter e meets a demand in window t with
probability p_e * m_{e,t}, where the multiplier m_{e,t} drifts linearly from
0.3 to 1.7 (or from 1.7 to 0.3) over the horizon so that the informative
parameters change with time, as in the case study.  phi = min(., 3), unit
weights, lognormal prices, 10% of (t, e) pairs unavailable, per-window budget
10% of the total available price, symmetric switching cost with
gamma = lambda / Q_bar = 0.2.

For every instance we report
  * the myopic policy (Algorithm 1 per window) and the restricted program
    (Algorithm 3 over the pools of Section 5.1.2: empty set, greedy prefixes,
    Algorithm 1, Algorithm 2 with l = 1, singletons), their J values and times;
  * the switching-aware Lagrangian certificate of Theorem 5.4(iii) with the
    myopic sets as reference, and the certified ratios J / UB;
  * at n = 20 only, the exact optimum of Theorem 5.3 (Q_t tabulated on all
    2^20 subsets by the pattern transform), so that the certified ratios can
    be compared with the true ones.

Usage: python scale_mp.py            -> results/v2/scale_mp.pkl
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bounds import lagrangian_bound_delta
from lattice import exact_multiperiod, mask_costs, q_lattice_fast, to_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "v2")
K, T, N_DEM, GAMMA, BFRAC = 3, 6, 5000, 0.2, 0.10
INF = float("inf")


class Window:
    def __init__(self, M, cost, B):
        self.M = M                      # (N_DEM, n) bool: parameter meets demand
        self.cost = cost                # (n,) with inf = unavailable
        self.B = B
        self.n = M.shape[1]

    def Q(self, A):
        if not A:
            return 0.0
        return float(np.minimum(self.M[:, list(A)].sum(axis=1), K).sum())

    def gains(self, cnt):
        return (self.M & (cnt < K)[:, None]).sum(axis=0).astype(float)

    def greedy(self, seed=()):
        A = list(seed)
        spent = sum(self.cost[e] for e in A)
        cnt = self.M[:, A].sum(axis=1) if A else np.zeros(self.M.shape[0], int)
        trace = []
        while True:
            g = self.gains(cnt)
            ok = np.array([(e not in A) and np.isfinite(self.cost[e]) and
                           spent + self.cost[e] <= self.B + 1e-9 for e in range(self.n)])
            if not ok.any():
                break
            dens = np.where(ok, g / np.where(np.isfinite(self.cost), self.cost, 1.0), -1.0)
            e = int(np.argmax(dens))
            if g[e] <= 0:
                break
            A.append(e)
            spent += self.cost[e]
            cnt = cnt + self.M[:, e]
            trace.append(list(A))
        return A, trace

    def singles(self):
        return [e for e in range(self.n) if np.isfinite(self.cost[e]) and self.cost[e] <= self.B + 1e-9]

    def alg1(self):
        A, trace = self.greedy()
        s = self.singles()
        if s:
            g = max(s, key=lambda e: self.Q([e]))
            if self.Q([g]) > self.Q(A):
                return [g], trace
        return A, trace

    def alg2_l1(self):
        best, bv = [], -1.0
        for seed in [()] + [(e,) for e in self.singles()]:
            A, _ = self.greedy(seed)
            v = self.Q(A)
            if v > bv:
                best, bv = A, v
        return best


def make_instance(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.10, n)
    up = rng.random(n) < 0.5
    ramp = np.linspace(0.3, 1.7, T)
    wins = []
    cost0 = np.exp(rng.normal(0.0, 0.75, n))
    for t in range(T):
        m = np.where(up, ramp[t], ramp[T - 1 - t])
        M = rng.random((N_DEM, n)) < (p * m)[None, :]
        cost = cost0.copy()
        cost[rng.random(n) < 0.10] = INF
        wins.append((M, cost))
    B = BFRAC * np.mean([np.sum(c[np.isfinite(c)]) for _, c in wins])
    return [Window(M, c, B) for M, c in wins]


def kappa(a, b):
    return len(set(a) ^ set(b))


def J(wins, pol, lam):
    prev, tot = [], 0.0
    for w, A in zip(wins, pol):
        tot += w.Q(A) - lam * kappa(prev, A)
        prev = A
    return tot


def restricted_dp(wins, pools, lam):
    V = [w0 - lam * kappa([], A) for A, w0 in zip(pools[0], [wins[0].Q(A) for A in pools[0]])]
    backs = []
    for t in range(1, T):
        q = [wins[t].Q(A) for A in pools[t]]
        newV, bk = [], []
        for j, A in enumerate(pools[t]):
            vals = [V[i] - lam * kappa(Ap, A) for i, Ap in enumerate(pools[t - 1])]
            i = int(np.argmax(vals))
            newV.append(vals[i] + q[j])
            bk.append(i)
        V, backs = newV, backs + [bk]
    j = int(np.argmax(V))
    pol = [None] * T
    pol[T - 1] = pools[T - 1][j]
    for t in range(T - 1, 0, -1):
        j = backs[t - 1][j]
        pol[t - 1] = pools[t - 1][j]
    return pol


def run_instance(n, seed):
    wins = make_instance(n, seed)
    qbar = np.mean([w.Q([e]) for w in wins for e in range(n) if np.isfinite(w.cost[e])])
    lam = GAMMA * qbar
    t0 = time.time()
    myo = []
    pools = []
    for w in wins:
        a1, trace = w.alg1()
        myo.append(a1)
        pool = {(): None}
        for A in trace:
            pool[tuple(sorted(A))] = None
        pool[tuple(sorted(a1))] = None
        pool[tuple(sorted(w.alg2_l1()))] = None
        for e in w.singles():
            pool[(e,)] = None
        pools.append([list(A) for A in pool])
    t_pools = time.time() - t0
    t1 = time.time()
    rdp = restricted_dp(wins, pools, lam)
    t_rdp = time.time() - t1
    t2 = time.time()
    base = np.array([w.Q(A) for w, A in zip(wins, myo)])
    delta = np.zeros((T, n))
    for t, (w, A) in enumerate(zip(wins, myo)):
        cnt = w.M[:, A].sum(axis=1) if A else np.zeros(N_DEM, int)
        g = w.gains(cnt)
        g[A] = 0.0
        delta[t] = g
    cost = np.array([w.cost for w in wins])
    UB = lagrangian_bound_delta(base, delta, cost, [w.B] * T, lam, np.ones(n), np.ones(n))
    t_ub = time.time() - t2
    row = dict(n=n, seed=seed, lam=lam, J_myopic=J(wins, myo, lam), J_restricted=J(wins, rdp, lam),
               UB=UB, pool_size=float(np.mean([len(p) for p in pools])),
               t_pools=t_pools, t_restricted=t_rdp, t_bound=t_ub)
    if n <= 20:
        t3 = time.time()
        N = 1 << n
        Qlat, feas = [], []
        for w in wins:
            pat = (w.M.astype(np.int64) * (1 << np.arange(n))[None, :]).sum(axis=1)
            c = np.bincount(pat, minlength=N).astype(np.float64)
            Qlat.append(q_lattice_fast(c, n, lambda d: np.minimum(d, K).astype(np.float64), kmax=K))
            feas.append(mask_costs(w.cost, n) <= w.B + 1e-9)
        pol, val = exact_multiperiod(Qlat, feas, n, lam)
        row.update(J_exact=val, t_exact=time.time() - t3,
                   J_exact_check=J(wins, pol, lam))
    return row


if __name__ == "__main__":
    rows = []
    for n, reps in ((20, 10), (50, 10), (100, 10)):
        for r in range(reps):
            row = run_instance(n, 9000 + 100 * n + r)
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(os.path.join(OUT, "scale_mp.pkl"), "wb") as f:
        pickle.dump(rows, f)
    print("done")
