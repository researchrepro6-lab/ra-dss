"""
verify2.py -- brute-force checks of the multi-period, certificate,
surrogate and lattice results.

  V10  exact multi-period DP (distance transform)  == exhaustive search
  V11  Lagrangian certificate >= exact OPT; tightness statistics
  V12  depleting-budget DP == exhaustive search
  V13  modular Q, no budget: per-parameter two-state chains == exhaustive
       search, and -J is submodular on T x E
  V14  surrogate sandwich: Q_k(A) - R(A) <= sum min((s(p)-s(n))^+, k) <= Q_k(A)
  V15  data-dependent bound (Leskovec et al.) >= exact per-period optimum
  V16  lattice transform == direct evaluation of Q (case-study slices)
  V17  graded separation: additive, max and probabilistic-sum aggregations
       are normalised, monotone, submodular; Alg. 1 and Alg. 2 guarantees hold

Writes results/v2/verification2.json.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bounds import depleting_dp, lagrangian_bound, _chain_value
from lattice import exact_multiperiod, to_mask, from_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "v2")
INF = float("inf")


def rand_Q(rng, n, n_dem=12, k=2):
    dem = [rng.random(n) < rng.uniform(0.2, 0.6) for _ in range(n_dem)]
    w = rng.random(n_dem)
    N = 1 << n
    L = np.zeros(N)
    for m in range(N):
        A = np.array([(m >> e) & 1 for e in range(n)], bool)
        L[m] = sum(wi * min(int((A & d).sum()), k) for wi, d in zip(w, dem))
    return L


def kappa(prev, m, n, kp, km):
    add, rem = m & ~prev, prev & ~m
    return sum(kp[i] for i in range(n) if add >> i & 1) + sum(km[i] for i in range(n) if rem >> i & 1)


def brute_J(Q, feas, n, lam, kp, km):
    T, N = len(Q), 1 << n
    best = -INF
    for combo in itertools.product(range(N), repeat=T):
        if not all(feas[t][combo[t]] for t in range(T)):
            continue
        prev, v = 0, 0.0
        for t, m in enumerate(combo):
            v += Q[t][m] - lam * kappa(prev, m, n, kp, km)
            prev = m
        best = max(best, v)
    return best


def v10_v11_v15(trials=300, seed=0):
    rng = np.random.default_rng(seed)
    bad_dp = bad_lag = bad_dd = 0
    tight_lag, tight_sum = [], []
    for _ in range(trials):
        n, T = int(rng.integers(2, 6)), int(rng.integers(2, 4))
        Q = [rand_Q(rng, n) for _ in range(T)]
        cost = rng.uniform(1, 5, (T, n))
        cost[rng.random((T, n)) < 0.15] = INF
        Bv = [float(rng.uniform(2, 9)) for _ in range(T)]
        N = 1 << n
        feas = []
        for t in range(T):
            mc = np.array([sum(cost[t, e] for e in range(n) if m >> e & 1) for m in range(N)])
            feas.append(mc <= Bv[t] + 1e-9)
        lam = float(rng.uniform(0, 1.5))
        kp, km = list(rng.uniform(0, 1.5, n)), list(rng.uniform(0, 1.5, n))
        pol, val = exact_multiperiod(Q, feas, n, lam, kp, km)
        opt = brute_J(Q, feas, n, lam, kp, km)
        bad_dp += int(abs(opt - val) > 1e-8)
        # reference sets: per-period greedy by density
        ref = []
        for t in range(T):
            m, sp = 0, 0.0
            while True:
                be, bd = -1, 0.0
                for e in range(n):
                    if m >> e & 1 or not np.isfinite(cost[t, e]) or sp + cost[t, e] > Bv[t]:
                        continue
                    d = (Q[t][m | 1 << e] - Q[t][m]) / cost[t, e]
                    if d > bd + 1e-12:
                        be, bd = e, d
                if be < 0:
                    break
                m |= 1 << be
                sp += cost[t, be]
            ref.append(m)
        lb = lagrangian_bound(Q, cost, Bv, lam, ref, kp, km)
        bad_lag += int(lb < opt - 1e-7)
        # per-period data-dependent bounds
        U = 0.0
        for t in range(T):
            items = []
            for e in range(n):
                if ref[t] >> e & 1 or not np.isfinite(cost[t, e]) or cost[t, e] > Bv[t]:
                    continue
                g = Q[t][ref[t] | 1 << e] - Q[t][ref[t]]
                if g > 0:
                    items.append((g / cost[t, e], g, cost[t, e]))
            items.sort(reverse=True)
            cap, add = Bv[t], 0.0
            for d, g, c in items:
                if c <= cap:
                    add, cap = add + g, cap - c
                else:
                    add += d * cap
                    break
            Ut = Q[t][ref[t]] + add
            optt = max(Q[t][m] for m in range(N) if feas[t][m])
            bad_dd += int(Ut < optt - 1e-9)
            U += Ut
        if opt > 1e-9:
            tight_lag.append(lb / opt)
            tight_sum.append(U / opt)
    return dict(V10_trials=trials, V10_mismatch=bad_dp,
                V11_trials=trials, V11_violations=bad_lag,
                V11_ratio_mean=float(np.mean(tight_lag)),
                V11_sumU_ratio_mean=float(np.mean(tight_sum)),
                V11_lag_tighter_frac=float(np.mean(np.array(tight_lag) <= np.array(tight_sum) + 1e-9)),
                V15_violations=bad_dd)


def v12(trials=150, seed=1):
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        n, T = int(rng.integers(2, 5)), int(rng.integers(2, 4))
        Q = [rand_Q(rng, n) for _ in range(T)]
        ic = rng.integers(1, 5, (T, n))
        ic[rng.random((T, n)) < 0.15] = -1
        bmax = int(rng.integers(4, 10))
        b0 = int(rng.integers(0, bmax + 1))
        r = [int(x) for x in rng.integers(0, 4, T)]
        lam = float(rng.uniform(0, 1.5))
        val = depleting_dp(Q, ic, b0, r, bmax, lam)
        N = 1 << n
        best = -INF
        for combo in itertools.product(range(N), repeat=T):
            b, prev, v, ok = b0, 0, 0.0, True
            for t, m in enumerate(combo):
                es = from_mask(m, n)
                if any(ic[t, e] < 0 for e in es):
                    ok = False
                    break
                c = int(sum(ic[t, e] for e in es))
                if c > b:
                    ok = False
                    break
                v += Q[t][m] - lam * bin(prev ^ m).count("1")
                b = min(bmax, b - c + r[t])
                prev = m
            if ok:
                best = max(best, v)
        bad += int(abs(best - val) > 1e-8)
    return dict(V12_trials=trials, V12_mismatch=bad)


def v13(trials=300, seed=2):
    rng = np.random.default_rng(seed)
    bad_chain = bad_sub = 0
    for _ in range(trials):
        n, T = int(rng.integers(1, 3)), int(rng.integers(2, 4))
        v = rng.normal(0, 1, (T, n))
        lam = float(rng.uniform(0, 1.5))
        kp, km = rng.uniform(0, 1.5, n), rng.uniform(0, 1.5, n)
        chains = sum(_chain_value(v[:, e], np.ones(T, bool), lam, kp[e], km[e]) for e in range(n))
        N = 1 << n
        Q = [np.array([sum(v[t, e] for e in range(n) if m >> e & 1) for m in range(N)]) for t in range(T)]
        opt = brute_J(Q, [np.ones(N, bool)] * T, n, lam, kp, km)
        bad_chain += int(abs(opt - chains) > 1e-8)
        # -J on Omega = T x E is submodular: check all pairs of subsets
        nn = T * n

        def J(S):
            prev, tot = 0, 0.0
            for t in range(T):
                m = (S >> (t * n)) & ((1 << n) - 1)
                tot += Q[t][m] - lam * kappa(prev, m, n, kp, km)
                prev = m
            return tot
        vals = [-J(S) for S in range(1 << nn)]
        for S1 in range(1 << nn):
            for S2 in range(1 << nn):
                if vals[S1] + vals[S2] < vals[S1 | S2] + vals[S1 & S2] - 1e-9:
                    bad_sub += 1
    return dict(V13_trials=trials, V13_chain_mismatch=bad_chain,
                V13_submod_violations=bad_sub)


def v14(trials=400, seed=3):
    rng = np.random.default_rng(seed)
    bad = 0
    auc_bad = 0
    for _ in range(trials):
        n = int(rng.integers(1, 7))
        P, Nn = int(rng.integers(2, 20)), int(rng.integers(2, 20))
        pos = rng.random((P, n)) < rng.uniform(0.2, 0.8, n)
        neg = rng.random((Nn, n)) < rng.uniform(0.2, 0.8, n)
        sig = rng.random(n) < 0.5
        k = int(rng.integers(1, 4))
        for m in range(1, 1 << n):
            A = [e for e in range(n) if m >> e & 1]
            a = pos[:, A] == sig[A]
            b = neg[:, A] != sig[A]
            ar = pos[:, A] != sig[A]
            br = neg[:, A] == sig[A]
            mp = (a[:, None, :] & b[None, :, :]).sum(-1)
            mm = (ar[:, None, :] & br[None, :, :]).sum(-1)
            Qk = np.minimum(mp, k).sum()
            R = mm.sum()
            sp, sn = a.sum(1), (~b).sum(1)
            diff = sp[:, None] - sn[None, :]
            mid = np.minimum(np.maximum(diff, 0), k).sum()
            if not (Qk - R - 1e-9 <= mid <= Qk + 1e-9):
                bad += 1
            if k == 1:
                auc_gt = (diff > 0).mean()
                if not ((np.minimum(mp, 1).sum() - R) / (P * Nn) - 1e-12 <= auc_gt
                        <= np.minimum(mp, 1).sum() / (P * Nn) + 1e-12):
                    auc_bad += 1
    return dict(V14_trials=trials, V14_violations=bad, V14_auc_violations=auc_bad)


def v16():
    import engine as E
    return {"V16_" + k: v for k, v in E.verify_engine().items()}


# ----------------------------------------------------------------------------
# V17  graded separation (Proposition 4.6): for graded
#      degrees d_e(pi) in [0,1], the additive, max and probabilistic-sum
#      aggregations are normalised, monotone and submodular; with 0/1 degrees
#      the additive form reproduces Q^{w,phi}; Alg. 1 and Alg. 2 (l = 2) keep
#      their guarantees on graded instances.
# ----------------------------------------------------------------------------
_PHIS = {
    "min(x,1)": lambda x: np.minimum(x, 1.0),
    "min(x,2)": lambda x: np.minimum(x, 2.0),
    "sqrt": np.sqrt,
    "log1p": np.log1p,
    "1-exp(-2x)": lambda x: 1.0 - np.exp(-2.0 * x),
    "identity": lambda x: x,
}


def _graded_table(D, w, phi, agg):
    """Q over all 2^n masks. D: (n_dem, n) degrees in [0,1]; w: (n_dem,)."""
    n_dem, n = D.shape
    N = 1 << n
    L = np.zeros(N)
    for m in range(N):
        sel = np.array([(m >> e) & 1 for e in range(n)], bool)
        Ds = D[:, sel]
        if agg == "sum":
            u = Ds.sum(1)
        elif agg == "max":
            u = Ds.max(1) if sel.any() else np.zeros(n_dem)
        else:  # probabilistic sum 1 - prod(1 - d)
            u = 1.0 - np.prod(1.0 - Ds, axis=1)
        L[m] = float((w * phi(u)).sum())
    return L


def _check_props(L, n, tol=1e-10):
    """Violations of normalisation, monotonicity and submodularity of table L."""
    norm = int(abs(L[0]) > tol)
    mono = sub = 0
    N = 1 << n
    for A in range(N):
        for e in range(n):
            if A >> e & 1:
                continue
            gA = L[A | 1 << e] - L[A]
            if gA < -tol:
                mono += 1
            rest = [f for f in range(n) if not (A >> f & 1) and f != e]
            for r in range(1 << len(rest)):
                B = A
                for j, f in enumerate(rest):
                    if r >> j & 1:
                        B |= 1 << f
                if B == A:
                    continue
                if gA < L[B | 1 << e] - L[B] - tol:
                    sub += 1
    return norm, mono, sub


def _greedy_table(L, c, B, n, seeds=((),)):
    """Alg. 1 (seeds = [()]) or Alg. 2 (seeds = all sets of size <= l) on table L."""
    best_val, best = -INF, 0
    for S in seeds:
        m = sum(1 << e for e in S)
        s = sum(c[e] for e in S)
        if s > B + 1e-12:
            continue
        while True:
            cand = [e for e in range(n) if not (m >> e & 1) and s + c[e] <= B + 1e-12]
            if not cand:
                break
            e = max(cand, key=lambda f: ((L[m | 1 << f] - L[m]) / c[f], -f))
            if L[m | 1 << e] - L[m] <= 1e-15:
                break
            m |= 1 << e
            s += c[e]
        if L[m] > best_val:
            best_val, best = L[m], m
    if seeds == ((),):  # singleton guard of Alg. 1
        singles = [e for e in range(n) if c[e] <= B + 1e-12]
        if singles:
            g = max(singles, key=lambda e: L[1 << e])
            if L[1 << g] > best_val:
                best_val = L[1 << g]
    return best_val


def v17(trials=300, seed=4):
    rng = np.random.default_rng(seed)
    viol = dict(norm=0, mono=0, sub=0)
    crisp_mismatch = 0
    g1_min, g2_min, g_bad1, g_bad2, n_opt = 1.0, 1.0, 0, 0, 0
    phi_names = list(_PHIS)
    for it in range(trials):
        n = int(rng.integers(2, 7))
        n_dem = int(rng.integers(3, 12))
        # graded degrees with a share of exact zeros and ones
        D = rng.random((n_dem, n))
        D[rng.random((n_dem, n)) < 0.3] = 0.0
        D[rng.random((n_dem, n)) < 0.1] = 1.0
        w = rng.random(n_dem)
        for agg in ("sum", "max", "psum"):
            name = phi_names[it % len(phi_names)]
            L = _graded_table(D, w, _PHIS[name], agg)
            a, b, s = _check_props(L, n)
            viol["norm"] += a
            viol["mono"] += b
            viol["sub"] += s
            # approximation guarantees on a random knapsack instance
            c = rng.uniform(0.2, 1.0, n)
            Bt = float(rng.uniform(0.3, 0.7) * c.sum())
            feas = [m for m in range(1 << n)
                    if sum(c[e] for e in range(n) if m >> e & 1) <= Bt + 1e-12]
            opt = max(L[m] for m in feas)
            if opt > 1e-12:
                n_opt += 1
                r1 = _greedy_table(L, c, Bt, n) / opt
                seeds2 = tuple(S for k in range(3) for S in itertools.combinations(range(n), k))
                r2 = _greedy_table(L, c, Bt, n, seeds2) / opt
                g1_min, g2_min = min(g1_min, r1), min(g2_min, r2)
                g_bad1 += int(r1 < 0.427 - 1e-12)
                g_bad2 += int(r2 < 1 - 1 / np.e - 1e-12)
        # crisp degrees: additive aggregation equals Q^{w,phi} with phi = min(., k)
        k = int(rng.integers(1, 4))
        Dc = (rng.random((n_dem, n)) < 0.5).astype(float)
        L1 = _graded_table(Dc, w, lambda x: np.minimum(x, float(k)), "sum")
        for m in range(1 << n):
            A = np.array([(m >> e) & 1 for e in range(n)], bool)
            direct = sum(wi * min(int(di[A].sum()), k) for wi, di in zip(w, Dc))
            crisp_mismatch += int(abs(L1[m] - direct) > 1e-9)
    # power control: a convex transform (psi = x^2) must fail the check
    rc = np.random.default_rng(seed + 1)
    ctrl = {"sum": 0, "psum": 0}
    for _ in range(200):
        n = int(rc.integers(2, 7))
        n_dem = int(rc.integers(3, 12))
        D = rc.random((n_dem, n))
        D[rc.random((n_dem, n)) < 0.3] = 0.0
        D[rc.random((n_dem, n)) < 0.1] = 1.0
        w = rc.random(n_dem)
        for agg in ctrl:
            ctrl[agg] += int(_check_props(_graded_table(D, w, lambda x: x ** 2, agg), n)[2] > 0)
    return dict(V17_trials=trials, V17_instances=3 * trials,
                V17_control_convex_trials=400,
                V17_control_convex_failed=ctrl["sum"] + ctrl["psum"],
                V17_norm_violations=viol["norm"], V17_mono_violations=viol["mono"],
                V17_submod_violations=viol["sub"], V17_crisp_mismatch=crisp_mismatch,
                V17_greedy_instances=n_opt,
                V17_alg1_min_ratio=float(g1_min), V17_alg1_violations=g_bad1,
                V17_alg2_min_ratio=float(g2_min), V17_alg2_violations=g_bad2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "v17":
        # run only the graded-separation check and merge into the existing file
        path = os.path.join(OUT, "verification2.json")
        res = json.load(open(path)) if os.path.exists(path) else {}
        r = v17()
        print(r, flush=True)
        res.update(r)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2)
        sys.exit(0)
    res = {}
    for f in (v10_v11_v15, v12, v13, v14, v16, v17):
        r = f()
        print(r, flush=True)
        res.update(r)
    with open(os.path.join(OUT, "verification2.json"), "w") as fh:
        json.dump(res, fh, indent=2)
