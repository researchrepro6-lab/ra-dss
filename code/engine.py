"""
engine.py -- lattice-based RA-DSS engine used by the experiments of the paper.

Every window's objective Q_t is tabulated over all 2^14 parameter subsets with
lattice.q_lattice_fast, so that greedy, partial enumeration, exact per-window
optima, the genetic-algorithm baseline and the exact multi-period dynamic
program all read the SAME numbers.  The construction reproduces
casestudy.FastSlice exactly (same subsample, same orientation estimate); this is
checked in verify_engine() below.

Selection policies implemented here (names as in the paper):
  Full panel, Random (budget), Cost-blind greedy, Static reduct,
  RA-DSS myopic (Alg. 1), RA-DSS myopic+enum (Alg. 2, l=1),
  RA-DSS exact/window, RA-DSS DP (restricted pools, Alg. 3),
  RA-DSS DP-exact (distance-transform DP, Thm 5.3),
  GA (binary genetic algorithm with knapsack repair, per window),
  TCS-reduct (lambda-weighted information-gain heuristic, Min et al. 2011),
  NB-VOI greedy (naive-Bayes value-of-information, population level),
  L1-cost (cost-weighted L1 logistic regression),
  Wrapper (budgeted forward selection by validation AUROC).
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

import casestudy as CS
from lattice import (distance_transform, exact_multiperiod, from_mask,
                     mask_costs, pattern_counts, q_lattice_fast, to_mask)

INF = float("inf")
NP = CS.N_PAR
NM = 1 << NP
ALLM = np.arange(NM)
POP = np.array([bin(m).count("1") for m in range(NM)])


# ---------------------------------------------------------------------------
# lattice slice
# ---------------------------------------------------------------------------
class LSlice:
    """One window: Q over all masks, costs, availability."""

    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, w: int,
                 cost_row: np.ndarray, k_red: int = CS.K_RED,
                 max_pos: Optional[int] = 400, max_neg: Optional[int] = 800,
                 sub_seed: int = 0, sigma: Optional[np.ndarray] = None,
                 subsample: bool = True):
        Xw = X[idx][:, w, :].astype(bool)
        yy = y[idx]
        pos = Xw[yy == 1]
        neg = Xw[yy == 0]
        if subsample:
            rs = np.random.default_rng(1000 + 17 * w + sub_seed)
            if max_pos is not None and pos.shape[0] > max_pos:
                pos = pos[rs.choice(pos.shape[0], max_pos, replace=False)]
            if max_neg is not None and neg.shape[0] > max_neg:
                neg = neg[rs.choice(neg.shape[0], max_neg, replace=False)]
        self.P, self.N = pos.shape[0], neg.shape[0]
        if sigma is None:
            p1 = pos.mean(axis=0) if self.P else np.zeros(NP)
            p0 = neg.mean(axis=0) if self.N else np.zeros(NP)
            sigma = (p1 >= p0)
        self.sigma = np.asarray(sigma, dtype=bool)
        a = (pos == self.sigma[None, :])          # positive reads 'target'
        b = (neg != self.sigma[None, :])          # negative reads 'non-target'
        pw = (1 << np.arange(NP)).astype(np.int64)
        apos = (a.astype(np.int64) * pw).sum(axis=1)
        bneg = (b.astype(np.int64) * pw).sum(axis=1)
        c = pattern_counts(apos, bneg, NP)
        self.k = k_red
        self.L = q_lattice_fast(c, NP, CS.phi_min(k_red), kmax=k_red)
        # reverse-separation counts (for the surrogate bound, Prop. 4.7)
        ar = (pos != self.sigma[None, :]).astype(np.int64)
        br = (neg == self.sigma[None, :]).astype(np.int64)
        self.rev_single = (ar.sum(axis=0) * br.sum(axis=0)).astype(float)
        self.pairs = float(self.P * self.N)
        self.set_cost(cost_row)

    def set_cost(self, cost_row: np.ndarray):
        self.cost = np.asarray(cost_row, dtype=float)
        self.avail_mask = to_mask([e for e in range(NP) if self.cost[e] < INF])
        cc = np.where(self.cost < INF, self.cost, 0.0)
        mc = mask_costs(cc, NP)
        unav = (ALLM & ~self.avail_mask) != 0
        self.mcost = np.where(unav, INF, mc)

    def feasible(self, B: float) -> np.ndarray:
        return self.mcost <= B + 1e-9

    def Q(self, A) -> float:
        return float(self.L[to_mask(A) if not isinstance(A, (int, np.integer)) else A])

    def qmax(self) -> float:
        return float(self.L[self.avail_mask])


def build_slices(coh, idx, cost_mat, sub_seed=0, k_red=CS.K_RED,
                 max_pos=400, max_neg=800, sigmas=None, subsample=True):
    W = cost_mat.shape[0]
    out = []
    for w in range(W):
        out.append(LSlice(coh.X, coh.y, idx, w, cost_mat[w], k_red=k_red,
                          max_pos=max_pos, max_neg=max_neg, sub_seed=sub_seed,
                          sigma=None if sigmas is None else sigmas[w],
                          subsample=subsample))
    return out


# ---------------------------------------------------------------------------
# single-window policies on the lattice
# ---------------------------------------------------------------------------
def greedy(sl: LSlice, B: float, seed: int = 0, density: bool = True,
           trace: bool = False):
    m = seed
    spent = float(sum(sl.cost[e] for e in from_mask(m, NP)))
    tr = []
    while True:
        best, bsc, bg = -1, -INF, 0.0
        for e in range(NP):
            if (m >> e) & 1 or sl.cost[e] == INF:
                continue
            if spent + sl.cost[e] > B + 1e-9:
                continue
            g = sl.L[m | (1 << e)] - sl.L[m]
            sc = g / sl.cost[e] if density else g
            if sc > bsc + 1e-12:
                best, bsc, bg = e, sc, g
        if best < 0 or bg <= 1e-9:
            break
        if trace:
            tr.append((best, bsc))
        m |= 1 << best
        spent += sl.cost[best]
    return (m, tr) if trace else m


def guarded(sl: LSlice, B: float) -> int:
    m = greedy(sl, B)
    best_s, bv = 0, 0.0
    for e in range(NP):
        if sl.cost[e] <= B + 1e-9 and sl.cost[e] < INF:
            v = sl.L[1 << e]
            if v > bv:
                best_s, bv = 1 << e, v
    return m if sl.L[m] >= bv else best_s


def partial_enum(sl: LSlice, B: float, ell: int = 1) -> int:
    av = from_mask(sl.avail_mask, NP)
    best, bv = 0, -1.0
    for r in range(ell + 1):
        for S in itertools.combinations(av, r):
            ms = to_mask(S)
            if sl.mcost[ms] > B + 1e-9:
                continue
            m = greedy(sl, B, seed=ms)
            if sl.L[m] > bv:
                best, bv = m, sl.L[m]
    return best


def exact_window(sl: LSlice, B: float) -> int:
    v = np.where(sl.feasible(B), sl.L, -INF)
    return int(np.argmax(v))


def random_policy(sl: LSlice, B: float, rng) -> int:
    av = from_mask(sl.avail_mask, NP)
    m, spent = 0, 0.0
    for e in rng.permutation(av):
        if spent + sl.cost[e] <= B + 1e-9:
            m |= 1 << int(e)
            spent += sl.cost[e]
    return m


def ga_window(sl: LSlice, B: float, rng, pop: int = 40, gens: int = 60,
              pmut: Optional[float] = None) -> int:
    """Binary GA with greedy knapsack repair (drop lowest marginal density
    until feasible, then add highest density while feasible)."""
    av = np.array(from_mask(sl.avail_mask, NP))
    if len(av) == 0:
        return 0
    pmut = pmut if pmut is not None else 1.0 / len(av)

    def repair(m: int) -> int:
        m &= sl.avail_mask
        while sl.mcost[m] > B + 1e-9:
            worst, wd = -1, INF
            for e in from_mask(m, NP):
                d = (sl.L[m] - sl.L[m & ~(1 << e)]) / sl.cost[e]
                if d < wd:
                    worst, wd = e, d
            m &= ~(1 << worst)
        while True:
            best, bd = -1, 0.0
            for e in av:
                if (m >> e) & 1 or sl.mcost[m] + sl.cost[e] > B + 1e-9:
                    continue
                d = (sl.L[m | (1 << e)] - sl.L[m]) / sl.cost[e]
                if d > bd + 1e-12:
                    best, bd = e, d
            if best < 0:
                break
            m |= 1 << int(best)
        return m

    P = [repair(int(to_mask(av[rng.random(len(av)) < 0.3]))) for _ in range(pop)]
    fit = np.array([sl.L[m] for m in P])
    for _ in range(gens):
        new = [P[int(np.argmax(fit))]]                     # elitism
        while len(new) < pop:
            i, j = rng.integers(0, pop, 2), rng.integers(0, pop, 2)
            p1 = P[i[0]] if fit[i[0]] >= fit[i[1]] else P[i[1]]
            p2 = P[j[0]] if fit[j[0]] >= fit[j[1]] else P[j[1]]
            cm = int(to_mask(av[rng.random(len(av)) < 0.5]))
            child = (p1 & cm) | (p2 & ~cm)
            flips = int(to_mask(av[rng.random(len(av)) < pmut]))
            new.append(repair(child ^ flips))
        P = new
        fit = np.array([sl.L[m] for m in P])
    return P[int(np.argmax(fit))]


def data_dependent_bound(sl: LSlice, B: float, m: int) -> float:
    """Leskovec et al. (2007) online bound: OPT <= Q(A) + fractional knapsack
    over the marginal gains of the affordable parameters (Prop. 4.11)."""
    items = []
    for e in range(NP):
        if (m >> e) & 1 or sl.cost[e] == INF or sl.cost[e] > B + 1e-9:
            continue
        g = sl.L[m | (1 << e)] - sl.L[m]
        if g > 0:
            items.append((g / sl.cost[e], g, sl.cost[e]))
    items.sort(reverse=True)
    cap, add = B, 0.0
    for d, g, c in items:
        if c <= cap:
            add += g
            cap -= c
        else:
            add += d * cap
            break
    return float(sl.L[m] + add)


# ---------------------------------------------------------------------------
# multi-window policies
# ---------------------------------------------------------------------------
def static_pooled(sls: List[LSlice], B: float) -> List[int]:
    """One panel on the time-summed LATENT objective, every assay treated as
    available at its cheapest finite price; unavailable members return nothing."""
    Lp = np.sum([s.L for s in sls], axis=0)
    cmin = np.array([min([s.cost[e] for s in sls if s.cost[e] < INF] or [INF])
                     for e in range(NP)])
    m, spent = 0, 0.0
    while True:
        best, bsc, bg = -1, -INF, 0.0
        for e in range(NP):
            if (m >> e) & 1 or cmin[e] == INF or spent + cmin[e] > B + 1e-9:
                continue
            g = Lp[m | (1 << e)] - Lp[m]
            sc = g / cmin[e]
            if sc > bsc + 1e-12:
                best, bsc, bg = e, sc, g
        if best < 0 or bg <= 1e-9:
            break
        m |= 1 << best
        spent += cmin[best]
    return [m & s.avail_mask for s in sls]


def candidate_pool(sls: List[LSlice], B: float) -> List[int]:
    pool = {0}
    for s in sls:
        m, tr = greedy(s, B, trace=True)
        pref = 0
        for (e, _) in tr:
            pref |= 1 << e
            pool.add(pref)
        pool.add(guarded(s, B))
        pool.add(partial_enum(s, B, 1))
        for e in range(NP):
            if s.cost[e] <= B + 1e-9:
                pool.add(1 << e)
    return sorted(pool)


def dp_restricted(sls: List[LSlice], B: float, lam: float,
                  pool: Optional[List[int]] = None) -> Tuple[List[int], float]:
    if pool is None:
        pool = candidate_pool(sls, B)
    pool = np.array(pool)
    T = len(sls)
    cands = []
    for s in sls:
        ok = s.feasible(B)[pool]
        cs = pool[ok] if ok.any() else np.array([0])
        cands.append(cs)
    ham = lambda a, b: POP[np.bitwise_xor.outer(a, b)]
    V = sls[0].L[cands[0]] - lam * POP[cands[0]]
    backs = []
    for t in range(1, T):
        M = V[:, None] - lam * ham(cands[t - 1], cands[t])
        bi = np.argmax(M, axis=0)
        backs.append(bi)
        V = M[bi, np.arange(len(cands[t]))] + sls[t].L[cands[t]]
    j = int(np.argmax(V))
    val = float(V[j])
    pol = [0] * T
    pol[T - 1] = int(cands[T - 1][j])
    for t in range(T - 1, 0, -1):
        j = int(backs[t - 1][j])
        pol[t - 1] = int(cands[t - 1][j])
    return pol, val


def dp_exact(sls: List[LSlice], B: float, lam: float) -> Tuple[List[int], float]:
    pol, val = exact_multiperiod([s.L for s in sls], [s.feasible(B) for s in sls],
                                 NP, lam)
    return [to_mask(A) for A in pol], val


def J_value(sls, pol_masks, lam) -> float:
    prev, tot = 0, 0.0
    for s, m in zip(sls, pol_masks):
        tot += s.L[m] - lam * POP[prev ^ m]
        prev = m
    return float(tot)


def n_switches(pol_masks) -> int:
    prev, tot = 0, 0
    for m in pol_masks:
        tot += int(POP[prev ^ m])
        prev = m
    return tot


def qbar(sls) -> float:
    v = [s.L[1 << e] for s in sls for e in range(NP) if s.cost[e] < INF]
    return float(np.mean(v)) if v else 1.0


# ---------------------------------------------------------------------------
# TCS-reduct: lambda-weighted information-gain heuristic (Min et al. 2011)
# ---------------------------------------------------------------------------
def _cond_entropy(Xb: np.ndarray, y: np.ndarray, cols: Sequence[int]) -> float:
    if len(cols) == 0:
        p = y.mean()
        return float(-(p * math.log(p) + (1 - p) * math.log(1 - p))) if 0 < p < 1 else 0.0
    pw = (1 << np.arange(len(cols))).astype(np.int64)
    code = (Xb[:, cols].astype(np.int64) * pw).sum(axis=1)
    key = code * 2 + y
    cnt = np.bincount(key, minlength=2 * (1 << len(cols)))
    c0, c1 = cnt[0::2].astype(float), cnt[1::2].astype(float)
    tot = c0 + c1
    n = float(len(y))
    h = 0.0
    for a, b, t in zip(c0, c1, tot):
        if t == 0:
            continue
        for v in (a, b):
            if v > 0:
                h -= (v / n) * math.log(v / t)
    return h


def tcs_reduct_window(X: np.ndarray, y: np.ndarray, w: int, cost_row: np.ndarray,
                      B: float, lams=(0.0, -0.5, -1.0, -1.5, -2.0)) -> int:
    """Budgeted adaptation of the lambda-weighted information-gain heuristic:
    addition by f(B,a) * c(a)^lambda while affordable, then deletion of
    attributes whose removal does not raise H(D|B) (most expensive first);
    the 'competition' over lambda keeps the subset with the lowest H(D|B)."""
    Xb = X[:, w, :].astype(np.uint8)
    av = [e for e in range(NP) if cost_row[e] < INF]
    best, bH, bC = 0, INF, INF
    for lam in lams:
        S, spent = [], 0.0
        H = _cond_entropy(Xb, y, S)
        while True:
            cand, csc, cH = -1, -INF, H
            for e in av:
                if e in S or spent + cost_row[e] > B + 1e-9:
                    continue
                He = _cond_entropy(Xb, y, S + [e])
                gain = H - He
                sc = gain * (cost_row[e] ** lam)
                if gain > 1e-12 and sc > csc:
                    cand, csc, cH = e, sc, He
            if cand < 0:
                break
            S.append(cand)
            spent += cost_row[cand]
            H = cH
        for e in sorted(S, key=lambda a: -cost_row[a]):
            rest = [a for a in S if a != e]
            if _cond_entropy(Xb, y, rest) <= H + 1e-12:
                S = rest
        cst = sum(cost_row[e] for e in S)
        if H < bH - 1e-12 or (abs(H - bH) <= 1e-12 and cst < bC):
            best, bH, bC = to_mask(S), H, cst
    return best


# ---------------------------------------------------------------------------
# joint (t, e) policies: NB-VOI greedy, cost-weighted L1, wrapper
# ---------------------------------------------------------------------------
def _pairs_available(cost_mat):
    W = cost_mat.shape[0]
    return [(w, e) for w in range(W) for e in range(NP) if cost_mat[w, e] < INF]


def _to_policy(sel, W):
    pol = [0] * W
    for (w, e) in sel:
        pol[w] |= 1 << e
    return pol


def nb_voi(X: np.ndarray, y: np.ndarray, cost_mat: np.ndarray, B: float) -> List[int]:
    """Population-level value-of-information greedy (Krause & Guestrin 2005)
    under a naive-Bayes model fitted to the training fold: add the (t,e) pair
    with the largest reduction in expected posterior entropy of the outcome
    per dollar, the expectation taken over the training objects, subject to
    every per-window budget."""
    W = cost_mat.shape[0]
    n = len(y)
    pairs = _pairs_available(cost_mat)
    py = y.mean()
    base = np.where(y == 1, math.log(py), math.log(1 - py))
    # per-pair log-likelihood contributions log p(x | y)
    contrib = {}
    for (w, e) in pairs:
        x = X[:, w, e]
        p1 = (x[y == 1].sum() + 1.0) / ((y == 1).sum() + 2.0)
        p0 = (x[y == 0].sum() + 1.0) / ((y == 0).sum() + 2.0)
        l1 = np.where(x == 1, math.log(p1), math.log(1 - p1))
        l0 = np.where(x == 1, math.log(p0), math.log(1 - p0))
        contrib[(w, e)] = (l1, l0)
    s1 = np.full(n, math.log(py))
    s0 = np.full(n, math.log(1 - py))

    def loglik(a1, a0):
        # negative expected posterior entropy (higher is better)
        p = 1.0 / (1.0 + np.exp(-np.clip(a1 - a0, -35, 35)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(np.sum(p * np.log(p) + (1 - p) * np.log(1 - p)))

    cur = loglik(s1, s0)
    spent = np.zeros(W)
    sel = []
    while True:
        best, bsc, bnew = None, 0.0, None
        for (w, e) in pairs:
            if (w, e) in sel or spent[w] + cost_mat[w, e] > B + 1e-9:
                continue
            l1, l0 = contrib[(w, e)]
            v = loglik(s1 + l1, s0 + l0)
            sc = (v - cur) / cost_mat[w, e]
            if sc > bsc + 1e-12:
                best, bsc, bnew = (w, e), sc, v
        if best is None:
            break
        sel.append(best)
        l1, l0 = contrib[best]
        s1, s0 = s1 + l1, s0 + l0
        spent[best[0]] += cost_mat[best]
        cur = bnew
    return _to_policy(sel, W)


def _design_pairs(X, pairs):
    return np.column_stack([X[:, w, e].astype(float) for (w, e) in pairs])


def l1_cost(X: np.ndarray, y: np.ndarray, cost_mat: np.ndarray, B: float,
            n_lam: int = 30, iters: int = 300) -> List[int]:
    """Cost-weighted L1 logistic regression (penalty lam * sum c_j |beta_j|):
    walk the regularisation path from the empty model and keep the least
    penalised support that satisfies every per-window budget."""
    W = cost_mat.shape[0]
    pairs = _pairs_available(cost_mat)
    Z = _design_pairs(X, pairs)
    n, d = Z.shape
    mu = Z.mean(axis=0)
    Zc = Z - mu
    wts = np.array([cost_mat[p] for p in pairs])
    ybar = y.mean()
    g0 = np.abs(Zc.T @ (y - ybar)) / n
    lam_max = float(np.max(g0 / wts))
    lams = lam_max * np.geomspace(1.0, 1e-3, n_lam)
    beta = np.zeros(d)
    b0 = math.log(ybar / (1 - ybar))
    L = 0.25 * np.linalg.norm(Zc, 2) ** 2 / n
    step = 1.0 / L
    best = [0] * W
    for lam in lams:
        # FISTA with warm start
        z, bz, t = beta.copy(), b0, 1.0
        bprev, b0prev = beta.copy(), b0
        for _ in range(iters):
            eta = bz + Zc @ z
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
            gr = Zc.T @ (p - y) / n
            g0_ = float(np.mean(p - y))
            u = z - step * gr
            bnew = np.sign(u) * np.maximum(np.abs(u) - step * lam * wts, 0.0)
            b0new = bz - step * g0_
            tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
            z = bnew + ((t - 1) / tn) * (bnew - bprev)
            bz = b0new + ((t - 1) / tn) * (b0new - b0prev)
            if np.max(np.abs(bnew - bprev)) < 1e-7:
                bprev, b0prev = bnew, b0new
                break
            bprev, b0prev, t = bnew, b0new, tn
        beta, b0 = bprev, b0prev
        sup = [pairs[j] for j in range(d) if abs(beta[j]) > 1e-8]
        spend = np.zeros(W)
        for (w, e) in sup:
            spend[w] += cost_mat[w, e]
        if np.all(spend <= B + 1e-9):
            best = _to_policy(sup, W)
        else:
            break
    return best


def _fast_logreg(Z, y, l2=10.0, iters=25):
    n, d = Z.shape
    Zb = np.column_stack([np.ones(n), Z])
    beta = np.zeros(d + 1)
    R = l2 * np.eye(d + 1)
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Zb @ beta, -35, 35)))
        g = Zb.T @ (y - p) - R @ beta
        H = (Zb * (p * (1 - p))[:, None]).T @ Zb + R
        stp = np.linalg.solve(H, g)
        beta += stp
        if np.max(np.abs(stp)) < 1e-8:
            break
    return beta


def fast_auroc(y, s) -> float:
    y = np.asarray(y)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def wrapper(X: np.ndarray, y: np.ndarray, cost_mat: np.ndarray, B: float,
            seed: int = 0) -> List[int]:
    """Budgeted forward selection: add the (t,e) pair with the largest gain in
    validation AUROC per dollar (ridge logistic, 2/3 : 1/3 split of the
    training fold), stop when no affordable pair improves validation AUROC."""
    W = cost_mat.shape[0]
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    cut = (2 * n) // 3
    a, b = perm[:cut], perm[cut:]
    pairs = _pairs_available(cost_mat)
    Zall = _design_pairs(X, pairs)
    col = {p: j for j, p in enumerate(pairs)}
    sel, spent = [], np.zeros(W)
    cur = 0.5
    while True:
        best, bsc, bauc = None, 0.0, None
        base_cols = [col[p] for p in sel]
        for p in pairs:
            if p in sel or spent[p[0]] + cost_mat[p] > B + 1e-9:
                continue
            cols = base_cols + [col[p]]
            beta = _fast_logreg(Zall[a][:, cols], y[a])
            s = beta[0] + Zall[b][:, cols] @ beta[1:]
            auc = fast_auroc(y[b], s)
            sc = (auc - cur) / cost_mat[p]
            if sc > bsc + 1e-12:
                best, bsc, bauc = p, sc, auc
        if best is None:
            break
        sel.append(best)
        spent[best[0]] += cost_mat[best]
        cur = bauc
    return _to_policy(sel, W)


# ---------------------------------------------------------------------------
# downstream evaluation
# ---------------------------------------------------------------------------
def design(X, pol_masks):
    cols = []
    for w, m in enumerate(pol_masks):
        for e in from_mask(m, NP):
            cols.append(X[:, w, e].astype(np.float64))
    if not cols:
        return np.zeros((X.shape[0], 1))
    return np.column_stack(cols)


def evaluate(coh, tr, te, pol_masks, seed=0):
    Dtr = design(coh.X[tr], pol_masks)
    Dte = design(coh.X[te], pol_masks)
    beta = CS.fit_logreg_cv(Dtr, coh.y[tr], seed=seed)
    pred = CS.predict(beta, Dte)
    return fast_auroc(coh.y[te], pred), pred


def realised_cost(pol_masks, cost_mat) -> float:
    tot = 0.0
    for w, m in enumerate(pol_masks):
        for e in from_mask(m, NP):
            tot += cost_mat[w, e] if cost_mat[w, e] < INF else 0.0
    return float(tot)


# ---------------------------------------------------------------------------
# DeLong test for two correlated AUROCs (fast version of Sun & Xu, 2014)
# ---------------------------------------------------------------------------
def _midrank(x):
    return rankdata(x)


def delong(y, s1, s2) -> Tuple[float, float, float]:
    """Returns (auc1, auc2, two-sided p-value for auc1 == auc2)."""
    from scipy.stats import norm
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    out = []
    V10s, V01s = [], []
    for s in (s1, s2):
        x, z = s[pos], s[neg]
        tx, tz, tall = _midrank(x), _midrank(z), _midrank(np.concatenate([x, z]))
        auc = (tall[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        V10s.append((tall[:m] - tx) / n)
        V01s.append(1.0 - (tall[m:] - tz) / m)
        out.append(auc)
    S10 = np.cov(np.vstack(V10s))
    S01 = np.cov(np.vstack(V01s))
    S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return out[0], out[1], 1.0
    zst = (out[0] - out[1]) / math.sqrt(var)
    return out[0], out[1], float(2 * norm.sf(abs(zst)))


# ---------------------------------------------------------------------------
# self-check against casestudy.FastSlice
# ---------------------------------------------------------------------------
def verify_engine(n_masks: int = 400, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    coh = CS.make_cohort(n=3000, seed=100, outcome_noise=2.8)
    tr, te = CS.kfold(3000, 5, seed=500)[0]
    C = CS.cost_matrix()
    bad = 0
    agree_g = agree_e = tot = 0
    for w in range(CS.N_WIN):
        fs = CS.FastSlice(coh.X, coh.y, tr, w, C, 0.1 * CS.FULL_PANEL_COST,
                          sub_seed=0)
        ls = LSlice(coh.X, coh.y, tr, w, C[w], sub_seed=0)
        for m in rng.integers(0, NM, n_masks):
            if abs(fs.Q(from_mask(int(m), NP)) - ls.L[int(m)]) > 1e-6:
                bad += 1
        for bf in (0.05, 0.1, 0.2, 0.4):
            fs.budget = bf * CS.FULL_PANEL_COST
            B = fs.budget
            tot += 1
            agree_g += int(to_mask(CS.sel_guarded_greedy(fs)) == guarded(ls, B))
            agree_e += int(to_mask(CS.sel_partial_enum(fs, 1)) ==
                           partial_enum(ls, B, 1))
    return dict(lattice_mismatches=bad, greedy_agree=agree_g,
                enum_agree=agree_e, cases=tot)
