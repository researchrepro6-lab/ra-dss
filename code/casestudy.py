"""
casestudy.py -- ICU laboratory-monitoring case study for RA-DSS.

Cohort is SIMULATED from the documented generative model below; acquisition
costs are the REAL Medicare Clinical Laboratory Fee Schedule amounts effective
2025-04-01 to 2026-03-31.

Objective
---------
Redundancy-aware weighted discernibility. For a concave nondecreasing
phi : Z_{>=0} -> R_{>=0} with phi(0) = 0,

    Q_t^{w,phi}(A) = sum_{pi in Pi} w(pi) * phi(|A n d_t(pi)|)

where Pi is the set of unordered object pairs, d_t(pi) the set of parameters
separating pi at time t, and w >= 0 pair weights. phi = min(.,1) recovers plain
discernibility (a coverage function); phi = min(.,k) with k > 1 demands that
each pair be separated by at least k parameters, i.e. corroborated evidence.

Fast evaluation
---------------
With w(i,j) = 1[y_i != y_j] only positive-negative pairs carry weight. Writing
M_e for the (P,N) boolean matrix M_e[p,n] = 1[X_p,e != X_n,e] and
D_A = sum_{e in A} M_e (the per-pair separation multiplicity),

    Q(A)            = sum_{p,n} phi(D_A[p,n])
    Q(A+e) - Q(A)   = sum_{p,n} M_e[p,n] * Dphi(D_A[p,n]),  Dphi(d)=phi(d+1)-phi(d)

Because phi is concave, Dphi is nonincreasing, so marginal gains shrink as D_A
grows: submodularity is structural in this representation. Cross-checked
against the general Slice implementation in radss.py by selftest().
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radss import INF, Slice

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameter set: 14 ICU laboratory tests.
# cost  = Medicare CLFS payment rate, USD, effective 2025-04-01 -- 2026-03-31
# phase = "early" (resuscitation / perfusion) or "late" (organ dysfunction /
#         inflammation): governs the time-varying loading beta_{e,t}
# sal   = salience multiplier on the loading
# Both phases deliberately contain cheap and expensive tests, so that timing
# and cost are NOT confounded.
# ---------------------------------------------------------------------------
# Phase assignment is balanced ACROSS cost tiers so that timing and price are
# not confounded: each of the cheap / mid / dear tiers contains both phases.
#   cheap ($4-9) : PT, CBC, BMP (early)   |  PHOS, CRP, MG  (late)
#   mid   ($10-13): CMP, LAC   (early)    |  DDIM, CULT, TROP (late)
#   dear  ($26-40): ABG       (early)     |  PCT, BNP        (late)
TESTS: List[dict] = [
    dict(key="PT",   cpt="85610", name="Prothrombin time",                  cost=4.29,  phase="early", sal=0.80),
    dict(key="CBC",  cpt="85025", name="Complete blood count w/ auto diff", cost=7.77,  phase="early", sal=1.00),
    dict(key="BMP",  cpt="80048", name="Basic metabolic panel",             cost=8.46,  phase="early", sal=1.05),
    dict(key="CMP",  cpt="80053", name="Comprehensive metabolic panel",     cost=10.56, phase="early", sal=1.20),
    dict(key="LAC",  cpt="83605", name="Lactate (lactic acid)",             cost=11.57, phase="early", sal=1.35),
    dict(key="ABG",  cpt="82803", name="Blood gases, any combination",      cost=26.07, phase="early", sal=1.45),
    dict(key="PHOS", cpt="84100", name="Phosphorus",                        cost=4.74,  phase="late",  sal=0.75),
    dict(key="CRP",  cpt="86140", name="C-reactive protein",                cost=5.18,  phase="late",  sal=0.95),
    dict(key="MG",   cpt="83735", name="Magnesium",                         cost=6.70,  phase="late",  sal=0.80),
    dict(key="DDIM", cpt="85379", name="D-dimer (fibrin degradation)",      cost=10.18, phase="late",  sal=1.05),
    dict(key="CULT", cpt="87040", name="Blood culture, bacteria",           cost=10.32, phase="late",  sal=1.15),
    dict(key="TROP", cpt="84484", name="Troponin, quantitative",            cost=12.47, phase="late",  sal=1.25),
    dict(key="PCT",  cpt="84145", name="Procalcitonin",                     cost=27.22, phase="late",  sal=1.50),
    dict(key="BNP",  cpt="83880", name="Natriuretic peptide (BNP)",         cost=39.26, phase="late",  sal=1.10),
]
KEYS = [t["key"] for t in TESTS]
COST = {i: TESTS[i]["cost"] for i in range(len(TESTS))}
FULL_PANEL_COST = sum(COST.values())
N_PAR = len(TESTS)
N_WIN = 6                      # six 4-hour windows over the first 24 h

# Availability: PCT and BNP are batched send-outs; cultures need incubation.
AVAIL: Dict[str, set] = {
    "PCT":  {0, 3},
    "BNP":  {0, 2, 4},
    "CULT": {2, 3, 4, 5},
}

# redundancy level of the default objective
K_RED = 3


def availability_matrix() -> np.ndarray:
    M = np.ones((N_WIN, N_PAR), dtype=bool)
    for j, t in enumerate(TESTS):
        if t["key"] in AVAIL:
            for w in range(N_WIN):
                M[w, j] = (w in AVAIL[t["key"]])
    return M


def cost_matrix() -> np.ndarray:
    A = availability_matrix()
    C = np.empty((N_WIN, N_PAR))
    for w in range(N_WIN):
        for j in range(N_PAR):
            C[w, j] = COST[j] if A[w, j] else INF
    return C


# ---------------------------------------------------------------------------
# phi library
# ---------------------------------------------------------------------------
def phi_min(k: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda d: np.minimum(d, k).astype(np.float64)


def phi_sqrt(d):
    return np.sqrt(d).astype(np.float64)


def phi_log(d):
    return np.log1p(d).astype(np.float64)


def dphi_table(phi: Callable, dmax: int) -> np.ndarray:
    """Dphi(d) = phi(d+1) - phi(d) for d = 0..dmax."""
    d = np.arange(dmax + 2)
    v = phi(d)
    return (v[1:] - v[:-1]).astype(np.float64)


# ---------------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------------
@dataclass
class Cohort:
    X: np.ndarray        # (n, N_WIN, N_PAR) uint8 abnormality indicators
    y: np.ndarray        # (n,) int {0,1}
    z: np.ndarray        # (n, N_WIN) latent severity
    beta: np.ndarray     # (N_WIN, N_PAR) loadings actually used


def loadings(sharp: float = 1.0) -> np.ndarray:
    """Time-varying loadings beta_{e,t}: the regime shift.

    Early-phase tests load strongly at t=0 and fade to near-zero by t=5;
    late-phase tests do the reverse. `sharp` in [0,1] interpolates between a
    time-invariant regime (0) and the full shift (1).
    """
    beta = np.zeros((N_WIN, N_PAR))
    hi, lo = 2.30, 0.05
    mid = 0.5 * (hi + lo)
    up = np.linspace(lo, hi, N_WIN)
    dn = np.linspace(hi, lo, N_WIN)
    for j, t in enumerate(TESTS):
        prof = dn if t["phase"] == "early" else up
        prof = mid + sharp * (prof - mid)
        beta[:, j] = prof * t["sal"]
    return beta


def make_cohort(n: int = 600, seed: int = 0, target_pos: float = 0.25,
                sharp: float = 1.0, obs_noise: float = 1.0,
                outcome_noise: float = 0.85) -> Cohort:
    """Generative model.

      arche_i  in {stable, deteriorating, fluctuating}  with p = .55/.25/.20
      drift_i  ~ N(-0.18, .10) / N(+0.34, .12) / N(+0.04, .30)
      z_i(0)   ~ N(0,1);   z_i(t+1) = z_i(t) + drift_i + N(0, .32)
      score_i  = z_i(T-1) + 0.45 z_i(T-2) + N(0, outcome_noise)
      y_i      = 1[score_i > (1-target_pos) quantile]
      X_i,e,t  ~ Bernoulli( sigma( beta_{e,t} z_i(t)/obs_noise + alpha_e ) )

    `outcome_noise` is the part of the outcome that NO panel can explain; it
    is what keeps the downstream task away from ceiling.
    """
    rng = np.random.default_rng(seed)
    beta = loadings(sharp=sharp)

    arche = rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])
    drift = np.where(arche == 0, rng.normal(-0.18, 0.10, n),
             np.where(arche == 1, rng.normal(+0.34, 0.12, n),
                                  rng.normal(+0.04, 0.30, n)))
    z = np.zeros((n, N_WIN))
    z[:, 0] = rng.normal(0.0, 1.0, n)
    for w in range(1, N_WIN):
        z[:, w] = z[:, w - 1] + drift + rng.normal(0.0, 0.32, n)

    score = z[:, -1] + 0.45 * z[:, -2] + rng.normal(0.0, outcome_noise, n)
    thr = np.quantile(score, 1.0 - target_pos)
    y = (score > thr).astype(int)

    #        PT    CBC   BMP   CMP   LAC   ABG  PHOS   CRP    MG  DDIM  CULT  TROP   PCT   BNP
    intercept = np.array([-0.25, -0.55, -0.40, -0.45, -0.85, -0.60,
                          -0.15, -0.30, -0.20, -0.50, -1.30, -0.95,
                          -1.05, -0.90])
    X = np.zeros((n, N_WIN, N_PAR), dtype=np.uint8)
    for w in range(N_WIN):
        lin = (beta[w][None, :] / obs_noise) * z[:, w][:, None] + intercept[None, :]
        p = 1.0 / (1.0 + np.exp(-lin))
        X[:, w, :] = (rng.random((n, N_PAR)) < p).astype(np.uint8)
    return Cohort(X=X, y=y, z=z, beta=beta)


# ---------------------------------------------------------------------------
# Fast redundancy-aware discernibility on a training index set
# ---------------------------------------------------------------------------
class FastSlice:
    """Q_t^{w,phi} with w(i,j) = 1[y_i != y_j], for one window."""

    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray,
                 w: int, cost: np.ndarray, budget: float,
                 phi: Optional[Callable] = None, k_red: int = K_RED,
                 pair_w: Optional[np.ndarray] = None,
                 oriented: bool = True,
                 max_pos: int = 400, max_neg: int = 800,
                 sub_seed: int = 0):
        Xw = X[idx][:, w, :].astype(bool)
        yy = y[idx]
        pos = Xw[yy == 1]
        neg = Xw[yy == 0]
        # Selection-time subsampling of the pair set keeps the (P,N) separation
        # tensors tractable for large cohorts. The downstream classifier always
        # uses the full training fold; only the SELECTION objective is estimated
        # on this subsample.
        rs = np.random.default_rng(1000 + 17 * w + sub_seed)
        if pos.shape[0] > max_pos:
            pos = pos[rs.choice(pos.shape[0], max_pos, replace=False)]
        if neg.shape[0] > max_neg:
            neg = neg[rs.choice(neg.shape[0], max_neg, replace=False)]
        self.P, self.N = pos.shape[0], neg.shape[0]
        self.window = w
        self.k_red = k_red
        self.oriented = oriented
        self.phi = phi if phi is not None else phi_min(k_red)
        # orientation sigma(e), estimated on the TRAINING objects only
        self.sigma = np.ones(N_PAR, dtype=bool)
        if oriented:
            p1 = pos.mean(axis=0) if self.P else np.zeros(N_PAR)
            p0 = neg.mean(axis=0) if self.N else np.zeros(N_PAR)
            self.sigma = (p1 >= p0)
        # separation structure M_e[p,n] = 1[ e separates (p,n) ]
        #   plain    : X_p,e != X_n,e
        #   oriented : X_p,e == sigma(e)  AND  X_n,e == not sigma(e)
        self.M = np.empty((N_PAR, self.P, self.N), dtype=bool)
        for e in range(N_PAR):
            if oriented:
                a = (pos[:, e] == self.sigma[e])
                b = (neg[:, e] != self.sigma[e])
                self.M[e] = a[:, None] & b[None, :]
            else:
                self.M[e] = pos[:, e][:, None] ^ neg[:, e][None, :]
        self.pair_w = (np.ones((self.P, self.N)) if pair_w is None
                       else np.asarray(pair_w, dtype=np.float64))
        self.dphi = dphi_table(self.phi, N_PAR + 1)
        self.cost = {j: float(cost[w, j]) for j in range(N_PAR)}
        self.budget = float(budget)
        self.total = float(self.pair_w.sum() * self.phi(np.array([N_PAR]))[0])
        # flattened separation tensor: one matrix-vector product yields the
        # marginal gains of ALL parameters simultaneously
        self._Mf = self.M.reshape(N_PAR, -1).astype(np.float32)
        self._wf = self.pair_w.reshape(-1).astype(np.float32)

    # -- objective ---------------------------------------------------------
    def _D(self, A: Sequence[int]) -> np.ndarray:
        if len(A) == 0:
            return np.zeros((self.P, self.N), dtype=np.int16)
        return self.M[list(A)].sum(axis=0).astype(np.int16)

    def Q(self, A: Sequence[int]) -> float:
        A = list(A)
        if not A:
            return 0.0
        D = self._D(A)
        return float(np.sum(self.pair_w * self.phi(D)))

    def marginal(self, e: int, A: Sequence[int]) -> float:
        D = self._D(A)
        return float(np.sum(self.pair_w * self.dphi[D] * self.M[e]))

    def gains_all(self, A: Sequence[int],
                  D: Optional[np.ndarray] = None) -> np.ndarray:
        """Marginal gain of every parameter given A, in one matvec.

        gain(e) = sum_{p,n} w * Dphi(D_A) * M_e   =  M_flat @ (w * Dphi(D_A))
        """
        if D is None:
            D = self._D(A)
        wd = (self._wf * self.dphi[D.reshape(-1)].astype(np.float32))
        return self._Mf @ wd

    def add_to_D(self, D: np.ndarray, e: int) -> np.ndarray:
        return D + self.M[e]

    def Q_from_D(self, D: np.ndarray) -> float:
        return float(np.sum(self.pair_w * self.phi(D)))

    def Q_max(self) -> float:
        return self.Q(self.available())

    # -- feasibility -------------------------------------------------------
    def available(self) -> List[int]:
        return [j for j in range(N_PAR) if self.cost[j] < INF]

    def affordable(self) -> List[int]:
        return [j for j in range(N_PAR) if self.cost[j] <= self.budget]

    def cost_of(self, A: Sequence[int]) -> float:
        return float(sum(self.cost[e] for e in A))

    def feasible(self, A: Sequence[int]) -> bool:
        return self.cost_of(A) <= self.budget + 1e-9


# ---------------------------------------------------------------------------
# Selection policies (all respect the per-window budget)
# ---------------------------------------------------------------------------
def sel_full(sl: FastSlice) -> List[int]:
    return sorted(sl.available())


def sel_random(sl: FastSlice, rng: np.random.Generator,
               n_draw: int = 12) -> List[List[int]]:
    out = []
    av = sl.available()
    for _ in range(n_draw):
        order = list(rng.permutation(av))
        A, spent = [], 0.0
        for e in order:
            if spent + sl.cost[e] <= sl.budget + 1e-9:
                A.append(int(e))
                spent += sl.cost[e]
        out.append(sorted(A))
    return out


def _greedy_core(sl: FastSlice, seed: Sequence[int] = (),
                 density: bool = True, trace: bool = False):
    """Shared greedy engine. density=True -> gain/cost; False -> raw gain."""
    A: List[int] = list(seed)
    spent = sl.cost_of(A)
    D = sl._D(A)
    rem = set(sl.available()) - set(A)
    costs = np.array([sl.cost[j] for j in range(N_PAR)], dtype=np.float64)
    tr = []
    step = 0
    while rem:
        gains = sl.gains_all(A, D)
        score = gains / costs if density else gains.copy()
        mask = np.zeros(N_PAR, dtype=bool)
        for e in rem:
            if spent + sl.cost[e] <= sl.budget + 1e-9:
                mask[e] = True
        if not mask.any():
            break
        sc = np.where(mask, score, -np.inf)
        best = int(np.argmax(sc))
        if gains[best] <= 1e-9:
            break
        if trace:
            step += 1
            dens = {int(e): float(score[e]) for e in range(N_PAR) if mask[e]}
            tr.append((step, best, float(score[best]), float(gains[best]), dens))
        A.append(best)
        rem.discard(best)
        spent += sl.cost[best]
        D = D + sl.M[best]
    return (sorted(A), tr, D) if trace else (sorted(A), D)


def sel_density_greedy(sl: FastSlice, trace: bool = False):
    if trace:
        A, tr, _ = _greedy_core(sl, density=True, trace=True)
        return A, tr
    A, _ = _greedy_core(sl, density=True)
    return A


def sel_guarded_greedy(sl: FastSlice) -> List[int]:
    A, D = _greedy_core(sl, density=True)
    qA = sl.Q_from_D(D)
    # best affordable singleton, all at once
    aff = [e for e in sl.available() if sl.cost[e] <= sl.budget + 1e-9]
    bs, bv = [], 0.0
    if aff:
        singles = sl.gains_all([], np.zeros((sl.P, sl.N), dtype=np.int16))
        for e in aff:
            if singles[e] > bv:
                bs, bv = [e], float(singles[e])
    return A if qA >= bv else bs


def sel_cost_blind_greedy(sl: FastSlice) -> List[int]:
    A, _ = _greedy_core(sl, density=False)
    return A


def sel_partial_enum(sl: FastSlice, kappa: int = 2) -> List[int]:
    av = sl.available()
    best, bv = [], -1.0
    for r in range(kappa + 1):
        for seed in itertools.combinations(av, r):
            if sl.cost_of(seed) > sl.budget + 1e-9:
                continue
            A, D = _greedy_core(sl, seed=seed, density=True)
            v = sl.Q_from_D(D)
            if v > bv:
                best, bv = A, v
    return best


def sel_exact(sl: FastSlice) -> Tuple[List[int], float]:
    """Brute force over the budget-feasible family (|E| = 14)."""
    av = sl.available()
    best, bv = (), -1.0
    for r in range(len(av) + 1):
        for S in itertools.combinations(av, r):
            if sl.cost_of(S) > sl.budget + 1e-9:
                continue
            v = sl.Q(S)
            if v > bv:
                best, bv = S, v
    return sorted(best), bv


def sel_static_pooled(slices: List[FastSlice]) -> List[List[int]]:
    """Time-invariant cost-sensitive reduct: one subset chosen on the pooled
    (time-summed) objective with every test treated as always available, then
    reused in every window. Selections that are unavailable in a given window
    simply return no result -- the operational price of ignoring time-varying
    availability."""
    class Pooled:
        def __init__(self, sls):
            self.sls = sls
            # price of an assay, taken from the slices themselves rather than
            # the module-level CLFS table, so that the baseline respects
            # whatever price structure the caller supplied
            self.cost = {}
            for j in range(N_PAR):
                fin = [s.cost[j] for s in sls if s.cost[j] < INF]
                self.cost[j] = min(fin) if fin else INF
            self.budget = sls[0].budget

        def Q(self, A):
            return float(sum(s.Q(A) for s in self.sls))

        def marginal(self, e, A):
            return float(sum(s.marginal(e, A) for s in self.sls))

        def available(self):
            return list(range(N_PAR))

        def affordable(self):
            return [j for j in range(N_PAR) if self.cost[j] <= self.budget]

        def cost_of(self, A):
            return float(sum(self.cost[e] for e in A))

    P = Pooled(slices)
    A: List[int] = []
    rem = set(j for j in range(N_PAR) if P.cost[j] < INF)
    spent = 0.0
    budget = slices[0].budget
    costs = np.array([P.cost[j] if P.cost[j] < INF else np.inf
                      for j in range(N_PAR)], dtype=np.float64)
    Ds = [sl._D(A) for sl in slices]
    while rem:
        gains = np.zeros(N_PAR)
        for sl, D in zip(slices, Ds):
            gains += sl.gains_all(A, D)
        score = gains / costs
        mask = np.zeros(N_PAR, dtype=bool)
        for e in rem:
            if spent + costs[e] <= budget + 1e-9:
                mask[e] = True
        if not mask.any():
            break
        best = int(np.argmax(np.where(mask, score, -np.inf)))
        if gains[best] <= 1e-9:
            break
        A.append(best)
        rem.discard(best)
        spent += costs[best]
        Ds = [D + sl.M[best] for sl, D in zip(slices, Ds)]
    chosen = sorted(A)
    return [[e for e in chosen if slices[w].cost[e] < INF]
            for w in range(len(slices))]


# --- temporally coherent DP ------------------------------------------------
def switch_cost(A_prev: Sequence[int], A_cur: Sequence[int]) -> float:
    return float(len(set(A_prev) ^ set(A_cur)))


def candidate_pool(slices: List[FastSlice]) -> List[Tuple[int, ...]]:
    pool = {tuple()}
    for sl in slices:
        _, tr = sel_density_greedy(sl, trace=True)
        pref: List[int] = []
        for (_, e, _, _, _) in tr:
            pref.append(e)
            pool.add(tuple(sorted(pref)))
        pool.add(tuple(sel_guarded_greedy(sl)))
        pool.add(tuple(sel_partial_enum(sl, kappa=1)))
        for e in sl.affordable():
            pool.add((e,))
    return sorted(pool, key=lambda s: (len(s), s))


def sel_dp(slices: List[FastSlice], lam: float,
           pool: Optional[Sequence[Tuple[int, ...]]] = None
           ) -> Tuple[List[List[int]], float]:
    if pool is None:
        pool = candidate_pool(slices)
    Tn = len(slices)
    cand, qual = [], []
    for sl in slices:
        cs = [S for S in pool if sl.feasible(S) and
              all(sl.cost[e] < INF for e in S)]
        if not cs:
            cs = [tuple()]
        cand.append(cs)
        qual.append({S: sl.Q(S) for S in cs})
    value = [dict() for _ in range(Tn)]
    back = [dict() for _ in range(Tn)]
    for S in cand[0]:
        value[0][S] = qual[0][S] - lam * switch_cost((), S)
        back[0][S] = None
    for t in range(1, Tn):
        for S in cand[t]:
            bv, bp = -math.inf, None
            qs = qual[t][S]
            for Pp in cand[t - 1]:
                v = value[t - 1][Pp] + qs - lam * switch_cost(Pp, S)
                if v > bv:
                    bv, bp = v, Pp
            value[t][S] = bv
            back[t][S] = bp
    Sl, bv = None, -math.inf
    for S, v in value[Tn - 1].items():
        if v > bv:
            Sl, bv = S, v
    policy = [[] for _ in range(Tn)]
    S = Sl
    for t in range(Tn - 1, -1, -1):
        policy[t] = sorted(S)
        S = back[t][S]
    return policy, bv


# ---------------------------------------------------------------------------
# Downstream classifier
# ---------------------------------------------------------------------------
def build_design(X: np.ndarray, policy: Sequence[Sequence[int]]) -> np.ndarray:
    cols = []
    for w, A in enumerate(policy):
        for e in sorted(A):
            cols.append(X[:, w, e].astype(np.float64))
    if not cols:
        return np.zeros((X.shape[0], 1))
    return np.column_stack(cols)


def fit_logreg(Xtr: np.ndarray, ytr: np.ndarray,
               l2: float = 2.0, iters: int = 200) -> np.ndarray:
    n, d = Xtr.shape
    Z = np.column_stack([np.ones(n), Xtr])
    beta = np.zeros(d + 1)
    R = l2 * np.eye(d + 1)
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -35, 35)))
        Wd = np.clip(p * (1 - p), 1e-9, None)
        g = Z.T @ (ytr - p) - R @ beta
        H = (Z * Wd[:, None]).T @ Z + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


L2_GRID = (2.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def fit_logreg_cv(Xtr: np.ndarray, ytr: np.ndarray, n_inner: int = 3,
                  grid: Sequence[float] = L2_GRID,
                  seed: int = 0) -> np.ndarray:
    """Ridge strength chosen by inner cross-validation on the training fold,
    then refit on the whole training fold. Without this, AUROC comparisons
    between panels of different dimension are confounded by overfitting."""
    n = Xtr.shape[0]
    if n < 40 or Xtr.shape[1] == 0:
        return fit_logreg(Xtr, ytr, l2=grid[len(grid) // 2])
    inner = kfold(n, n_inner, seed=seed)
    best_l2, best_a = grid[0], -1.0
    for l2 in grid:
        aucs = []
        for itr, ite in inner:
            b = fit_logreg(Xtr[itr], ytr[itr], l2=l2)
            aucs.append(auroc(ytr[ite], predict(b, Xtr[ite])))
        m = float(np.mean(aucs))
        if m > best_a:
            best_a, best_l2 = m, l2
    return fit_logreg(Xtr, ytr, l2=best_l2)


def predict(beta: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones(Xte.shape[0]), Xte])
    return 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -35, 35)))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def kfold(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((tr, te))
    return out


# ---------------------------------------------------------------------------
# self-test against the general Slice implementation (phi = min(.,1))
# ---------------------------------------------------------------------------
def selftest(n_trials: int = 150, seed: int = 11) -> bool:
    rng = np.random.default_rng(seed)
    bad_cov = bad_red = bad_or = bad_gain = 0
    for _ in range(n_trials):
        n = int(rng.integers(6, 26))
        npar = int(rng.integers(1, 6))
        Xf = np.zeros((n, 1, N_PAR), dtype=np.uint8)
        Xf[:, 0, :npar] = (rng.random((n, npar)) < 0.5).astype(np.uint8)
        y = rng.integers(0, 2, size=n)
        if y.sum() in (0, n):
            continue
        cost = np.full((1, N_PAR), 1.0)
        # (i) plain separation, phi = min(.,1) == general coverage implementation
        fs = FastSlice(Xf, y, np.arange(n), 0, cost, float(npar),
                       phi=phi_min(1), oriented=False)
        F = {j: Xf[:, 0, j].astype(bool) for j in range(npar)}
        W = (y[:, None] != y[None, :]).astype(float)
        np.fill_diagonal(W, 0.0)
        gs = Slice(n_obj=n, F=F, cost={j: 1.0 for j in range(npar)},
                   budget=float(npar), W=W)
        for r in range(npar + 1):
            for A in itertools.combinations(range(npar), r):
                if abs(fs.Q(A) - gs.Q(A)) > 1e-6:
                    bad_cov += 1
        # (ii) plain separation, phi = min(.,k) == direct definition
        kk = int(rng.integers(1, 4))
        fr = FastSlice(Xf, y, np.arange(n), 0, cost, float(npar),
                       phi=phi_min(kk), oriented=False)
        Xb = Xf[:, 0, :].astype(bool)
        for r in range(npar + 1):
            for A in itertools.combinations(range(npar), r):
                direct = 0.0
                for i in range(n):
                    for j in range(i + 1, n):
                        if y[i] == y[j]:
                            continue
                        m = sum(1 for e in A if Xb[i, e] != Xb[j, e])
                        direct += min(m, kk)
                if abs(fr.Q(A) - direct) > 1e-6:
                    bad_red += 1
        # (iii) ORIENTED separation, phi = min(.,k) == direct definition
        fo = FastSlice(Xf, y, np.arange(n), 0, cost, float(npar),
                       phi=phi_min(kk), oriented=True)
        sig = fo.sigma
        for r in range(npar + 1):
            for A in itertools.combinations(range(npar), r):
                direct = 0.0
                for i in range(n):
                    for j in range(n):
                        if y[i] != 1 or y[j] != 0:
                            continue
                        m = sum(1 for e in A
                                if (Xb[i, e] == sig[e]) and (Xb[j, e] != sig[e]))
                        direct += min(m, kk)
                if abs(fo.Q(A) - direct) > 1e-6:
                    bad_or += 1
        # (iv) gains_all must agree with marginal() one parameter at a time
        for r in range(min(npar, 3) + 1):
            for A in itertools.combinations(range(npar), r):
                ga = fo.gains_all(list(A))
                for e in range(npar):
                    if e in A:
                        continue
                    if abs(ga[e] - fo.marginal(e, list(A))) > 1e-3:
                        bad_gain += 1
    print(f"  plain    phi=min(.,1) vs general Slice : {bad_cov} mismatches")
    print(f"  plain    phi=min(.,k) vs definition    : {bad_red} mismatches")
    print(f"  oriented phi=min(.,k) vs definition    : {bad_or} mismatches")
    print(f"  gains_all vs marginal()                : {bad_gain} mismatches")
    return bad_cov == bad_red == bad_or == bad_gain == 0


def saturation_report(k_values=(1, 2, 3, 4), n=600, seed=100):
    coh = make_cohort(n=n, seed=seed)
    C = cost_matrix()
    tr = np.arange(n)
    rng = np.random.default_rng(0)
    out = {}
    for k in k_values:
        sl = FastSlice(coh.X, coh.y, tr, 0, C, 1e9, phi=phi_min(k))
        qmax = sl.Q(sl.available())
        rows = []
        for size in (1, 2, 3, 5, 8, 11, 14):
            vals = []
            for _ in range(15):
                A = list(rng.choice(sl.available(),
                                    size=min(size, len(sl.available())),
                                    replace=False))
                vals.append(sl.Q(A) / qmax)
            rows.append((size, float(np.mean(vals)),
                         float(np.min(vals)), float(np.max(vals))))
        out[k] = rows
    return out


if __name__ == "__main__":
    print("full-panel cost per window : $%.2f" % FULL_PANEL_COST)
    print("full-panel cost per 24 h   : $%.2f" % (FULL_PANEL_COST * N_WIN))
    print("\nself-test of the fast objective:")
    ok = selftest()
    print("\nsaturation of Q as a function of the redundancy level k")
    print("  (Q(A)/Q(E) for random A; spread max-min shows discriminating power)")
    rep = saturation_report()
    for k, rows in rep.items():
        print(f"  k={k}")
        for (size, m, lo, hi) in rows:
            print(f"     |A|={size:2d}  mean={m:.4f}  min={lo:.4f}  max={hi:.4f}"
                  f"  spread={hi-lo:.4f}")
    sys.exit(0 if ok else 1)
