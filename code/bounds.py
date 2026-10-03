"""
bounds.py -- certificates and the depleting-budget dynamic program.

lagrangian_bound
    Switching-aware upper bound on OPT = max_pi J(pi) (Theorem 5.4).  For
    reference sets R_t (e.g. the myopic greedy output) let
        delta_t(e) = Q_t(R_t + e) - Q_t(R_t)   (0 for e in R_t).
    Since Q_t is monotone submodular, Q_t(A) <= Q_t(R_t) + sum_{e in A} delta_t(e).
    Relaxing the per-period budgets with multipliers mu_t >= 0 gives
        Lambda(mu) = sum_t [Q_t(R_t) + mu_t B_t]
                     + sum_e max over 0/1 chains x_{.,e} of
                       sum_t x_t (delta_t(e) - mu_t c_t(e))
                       - lam * (activation / deactivation costs along the chain),
    which separates over parameters into two-state chains (O(|T||E|)).  Every
    Lambda(mu) >= OPT; we minimise the convex function Lambda over mu >= 0 by
    cyclic golden-section search.  With lam = 0 the bound reduces to the
    per-period fractional-knapsack bound of Leskovec et al. (2007).

depleting_dp
    Exact DP when the budget is a state: b_{t+1} = min(Bmax, b_t - c_t(A_t)
    + r_t), integer costs.  O(|T| (Bmax+1) n 2^n).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from lattice import distance_transform, from_mask, to_mask

INF = float("inf")


def _chain_value(rew: np.ndarray, allowed: np.ndarray, lam: float,
                 kp: float, km: float) -> float:
    """max over x in {0,1}^T (x_t=1 only if allowed[t]) of
    sum_t x_t rew_t - lam*(kp*#(0->1) + km*#(1->0)), starting from x_0 = 0."""
    v0, v1 = 0.0, -INF
    for t in range(len(rew)):
        n0 = max(v0, v1 - lam * km)
        n1 = max(v0 - lam * kp, v1) + rew[t] if allowed[t] else -INF
        v0, v1 = n0, n1
    return max(v0, v1)


def lagrangian_value(mu, base, delta, cost, B, lam, kp, km):
    T, n = delta.shape
    val = float(np.sum(base) + np.sum(mu * B))
    for e in range(n):
        # a parameter dearer than the whole period budget is in no feasible set
        allowed = np.isfinite(cost[:, e]) & (np.where(np.isfinite(cost[:, e]), cost[:, e], 0.0) <= B + 1e-9)
        rew = np.where(allowed, delta[:, e] - mu * np.where(allowed, cost[:, e], 0.0), 0.0)
        val += _chain_value(rew, allowed, lam, kp[e], km[e])
    return val


def lagrangian_bound(Qlat: Sequence[np.ndarray], cost: np.ndarray, B: Sequence[float],
                     lam: float, ref_masks: Sequence[int],
                     kp: Optional[Sequence[float]] = None,
                     km: Optional[Sequence[float]] = None,
                     sweeps: int = 6) -> float:
    T, n = cost.shape
    kp = np.ones(n) if kp is None else np.asarray(kp, float)
    km = np.ones(n) if km is None else np.asarray(km, float)
    B = np.asarray(B, float)
    base = np.array([Qlat[t][ref_masks[t]] for t in range(T)])
    delta = np.zeros((T, n))
    for t in range(T):
        R = ref_masks[t]
        for e in range(n):
            if not (R >> e) & 1:
                delta[t, e] = Qlat[t][R | (1 << e)] - Qlat[t][R]
    return lagrangian_bound_delta(base, delta, cost, B, lam, kp, km, sweeps)


def lagrangian_bound_delta(base: np.ndarray, delta: np.ndarray, cost: np.ndarray,
                           B, lam: float, kp, km, sweeps: int = 6) -> float:
    """The same bound from precomputed Q_t(R_t) (base) and marginal gains
    delta[t, e]; used when Q_t cannot be tabulated on all 2^n subsets."""
    T, n = cost.shape
    B = np.asarray(B, float)
    kp = np.asarray(kp, float)
    km = np.asarray(km, float)
    fin = np.where(np.isfinite(cost), cost, np.nan)
    hi = np.array([np.nanmax(delta[t] / fin[t]) if np.any(np.isfinite(fin[t])) else 0.0
                   for t in range(T)])
    hi = np.maximum(hi, 0.0) + 1e-9
    # start from the per-period LP-optimal multipliers (critical density of
    # the fractional knapsack); at lam = 0 this point is optimal, and for
    # lam > 0 it already gives Lambda <= sum_t U_t.
    mu = np.zeros(T)
    for t in range(T):
        items = sorted([(delta[t, e] / cost[t, e], cost[t, e]) for e in range(n)
                        if np.isfinite(cost[t, e]) and cost[t, e] <= B[t] + 1e-9
                        and delta[t, e] > 0], reverse=True)
        cap = B[t]
        for d, c in items:
            if c <= cap:
                cap -= c
            else:
                mu[t] = d
                break
    f = lambda m: lagrangian_value(m, base, delta, cost, B, lam, kp, km)
    best = f(mu)
    g = (math.sqrt(5) - 1) / 2
    for _ in range(sweeps):
        for t in range(T):
            a, b = 0.0, hi[t]
            for _ in range(40):
                c1, c2 = b - g * (b - a), a + g * (b - a)
                m1, m2 = mu.copy(), mu.copy()
                m1[t], m2[t] = c1, c2
                if f(m1) <= f(m2):
                    b = c2
                else:
                    a = c1
            mu[t] = 0.5 * (a + b)
            best = min(best, f(mu))
    # the endpoint mu_t = 0 is also valid; keep the best value seen
    return float(best)


# ---------------------------------------------------------------------------
# depleting budget
# ---------------------------------------------------------------------------
def depleting_dp(Qlat: Sequence[np.ndarray], icost: np.ndarray, b0: int,
                 recharge: Sequence[int], bmax: int, lam: float,
                 kp: Optional[Sequence[float]] = None,
                 km: Optional[Sequence[float]] = None):
    """Exact optimum of sum_t Q_t(A_t) - lam*kappa subject to
    c_t(A_t) <= b_t, b_{t+1} = min(bmax, b_t - c_t(A_t) + r_t), b_1 = b0.
    icost : (T, n) integer costs, a negative value marks unavailability."""
    T, n = icost.shape
    N = 1 << n
    kp = [1.0] * n if kp is None else list(kp)
    km = [1.0] * n if km is None else list(km)
    idx = np.arange(N)
    mc = []
    for t in range(T):
        c = np.zeros(N, dtype=np.int64)
        bad = np.zeros(N, dtype=bool)
        for e in range(n):
            has = (idx >> e) & 1 == 1
            if icost[t, e] < 0:
                bad |= has
            else:
                c += np.where(has, icost[t, e], 0)
        mc.append(np.where(bad, 10 ** 9, c))
    NEG = -INF
    # V[b][A]: best value with configuration A just chosen and budget b left
    # for the NEXT period (after spending and recharge)
    V = {}
    start = np.full(N, NEG)
    start[0] = 0.0
    G0, _ = distance_transform(start, n, lam, kp, km)
    for A in range(N):
        pass
    cur = {b0: G0}               # G over A for the incoming budget level
    for t in range(T):
        nxt = {}
        for b, G in cur.items():
            ok = mc[t] <= b
            val = np.where(ok, Qlat[t] + G, NEG)
            for A in np.flatnonzero(np.isfinite(val)):
                nb = int(min(bmax, b - mc[t][A] + recharge[t]))
                arr = nxt.setdefault(nb, np.full(N, NEG))
                if val[A] > arr[A]:
                    arr[A] = val[A]
        if t == T - 1:
            return float(max(np.max(a) for a in nxt.values()))
        cur = {}
        for nb, arr in nxt.items():
            Gd, _ = distance_transform(arr, n, lam, kp, km)
            cur[nb] = Gd
    return NEG
