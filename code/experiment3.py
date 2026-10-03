"""
experiment3.py -- secondary experiments of the paper.

  stat      time-varying FINITE prices ("routine-or-STAT" scenario, hypothetical
            STAT multipliers) and the value of modelling them
  price     price-structure sensitivity, measured on downstream AUROC
  corr      organ-system-correlated cohort
  sens      one-factor-at-a-time sensitivity: k, sigma_out, subsample size,
            number of windows, loading sharpness
  headroom  temporal-adaptation headroom (exact, 50 replicates)
  sat       saturation of plain vs redundancy-aware discernibility
  inc       warm-start re-selection (evaluation counts, 20 replicates)

Usage: python experiment3.py [which] [n_proc]
  Without arguments (or with 'all') every experiment runs, in the order
  sat headroom stat price corr sens inc.  'python experiment3.py help' lists them.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtools import n_proc_arg, run_all
import casestudy as CS
import engine as E
from lattice import from_mask, to_mask

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "v2")
os.makedirs(OUT, exist_ok=True)
INF = float("inf")
NP = CS.N_PAR
N_PATIENTS, N_FOLD, SIG_OUT, GAMMA = 3000, 5, 2.8, 0.20


# ---------------------------------------------------------------------------
# generalised cohort (reproduces casestudy.make_cohort exactly for W = 6)
# ---------------------------------------------------------------------------
def loadings_w(W: int, sharp: float = 1.0) -> np.ndarray:
    beta = np.zeros((W, NP))
    hi, lo = 2.30, 0.05
    mid = 0.5 * (hi + lo)
    up, dn = np.linspace(lo, hi, W), np.linspace(hi, lo, W)
    for j, t in enumerate(CS.TESTS):
        prof = dn if t["phase"] == "early" else up
        prof = mid + sharp * (prof - mid)
        beta[:, j] = prof * t["sal"]
    return beta


def make_cohort_w(n=N_PATIENTS, seed=0, W=6, sharp=1.0, outcome_noise=SIG_OUT,
                  correlated=False, rho=1.2):
    """Per-window drift and innovation scale with window length so that the
    24-hour dynamics are the same for every W."""
    sc = 6.0 / W
    rng = np.random.default_rng(seed)
    beta = loadings_w(W, sharp)
    arche = rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])
    drift = np.where(arche == 0, rng.normal(-0.18, 0.10, n),
             np.where(arche == 1, rng.normal(+0.34, 0.12, n),
                                  rng.normal(+0.04, 0.30, n))) * sc
    z = np.zeros((n, W))
    z[:, 0] = rng.normal(0.0, 1.0, n)
    for w in range(1, W):
        z[:, w] = z[:, w - 1] + drift + rng.normal(0.0, 0.32 * math.sqrt(sc), n)
    score = z[:, -1] + 0.45 * z[:, -2] + rng.normal(0.0, outcome_noise, n)
    y = (score > np.quantile(score, 0.75)).astype(int)
    icpt = np.array([-0.25, -0.55, -0.40, -0.45, -0.85, -0.60, -0.15, -0.30,
                     -0.20, -0.50, -1.30, -0.95, -1.05, -0.90])
    X = np.zeros((n, W, NP), dtype=np.uint8)
    if not correlated:
        for w in range(W):
            lin = beta[w][None, :] * z[:, w][:, None] + icpt[None, :]
            p = 1.0 / (1.0 + np.exp(-lin))
            X[:, w, :] = (rng.random((n, NP)) < p).astype(np.uint8)
    else:
        groups = {"chem": ["BMP", "CMP", "MG", "PHOS"],
                  "inflam": ["CRP", "PCT", "CULT"], "cardiac": ["TROP", "BNP"],
                  "coag": ["PT", "DDIM"], "perf": ["LAC", "ABG", "CBC"]}
        gidx = {g: [CS.KEYS.index(k) for k in ks] for g, ks in groups.items()}
        for w in range(W):
            fac = {g: rng.normal(0, 1, n) for g in gidx}
            lin = np.zeros((n, NP))
            for g, js in gidx.items():
                for j in js:
                    lin[:, j] = (beta[w, j] * z[:, w] + rho * fac[g]
                                 + math.sqrt(max(0.0, 1 - rho ** 2)) * rng.normal(0, 1, n)
                                 + icpt[j])
            p = 1.0 / (1.0 + np.exp(-lin))
            Xw = (rng.random((n, NP)) < p).astype(np.uint8)
            jb, jc = CS.KEYS.index("BMP"), CS.KEYS.index("CMP")
            Xw[:, jc] = np.maximum(Xw[:, jc], Xw[:, jb])
            X[:, w, :] = Xw
    return CS.Cohort(X=X, y=y, z=z, beta=beta)


def avail_w(W: int) -> np.ndarray:
    """Clock-time availability: PCT resulted at 0 h and 12 h, BNP at 0, 8,
    16 h, blood culture from 8 h; window w covers [24w/W, 24(w+1)/W)."""
    A = np.ones((W, NP), dtype=bool)
    start = [24.0 * w / W for w in range(W)]
    end = [24.0 * (w + 1) / W for w in range(W)]
    for key, hours in (("PCT", (0, 12)), ("BNP", (0, 8, 16))):
        j = CS.KEYS.index(key)
        for w in range(W):
            A[w, j] = any(start[w] <= h < end[w] for h in hours)
    j = CS.KEYS.index("CULT")
    for w in range(W):
        A[w, j] = start[w] >= 8.0
    return A


def cost_mat_w(W: int, price: np.ndarray) -> np.ndarray:
    A = avail_w(W)
    return np.where(A, price[None, :], INF)


CLFS = np.array([CS.COST[j] for j in range(NP)])


def price_vector(mode: str) -> np.ndarray:
    base = CLFS.copy()
    if mode == "clfs":
        return base
    if mode == "flat":
        return np.full(NP, base.mean())
    if mode == "spread":
        v = base ** 2.2
        return v * (base.sum() / v.sum())
    if mode == "dominant":
        v = base.copy()
        j = int(np.argmax([t["sal"] for t in CS.TESTS]))
        v[j] = 1.5 * (v.sum() - v[j])
        return v
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# generic cell runner
# ---------------------------------------------------------------------------
def run_cells(coh, C, budgets, methods, rep, tag, k_red=CS.K_RED,
              max_pos=400, max_neg=800, panel=None, extra=None):
    panel = panel if panel is not None else float(np.nanmin(np.where(C < INF, C, np.nan), axis=0).sum())
    rows = []
    folds = CS.kfold(coh.X.shape[0], N_FOLD, seed=500 + rep)
    for fi, (tr, te) in enumerate(folds):
        sls = E.build_slices(coh, tr, C, sub_seed=rep, k_red=k_red,
                             max_pos=max_pos, max_neg=max_neg)
        tsl = E.build_slices(coh, te, C, sigmas=[s.sigma for s in sls],
                             subsample=False, k_red=k_red)
        qtr = sum(s.qmax() for s in sls)
        qte = sum(s.qmax() for s in tsl)
        cache = {}
        for bf in budgets:
            B = bf * panel
            lam = GAMMA * E.qbar(sls)
            pols = {}
            for m in methods:
                if m == "Cost-blind greedy":
                    pols[m] = [E.greedy(s, B, density=False) for s in sls]
                elif m == "Static reduct":
                    pols[m] = E.static_pooled(sls, B)
                elif m == "RA-DSS myopic":
                    pols[m] = [E.guarded(s, B) for s in sls]
                elif m == "RA-DSS exact/window":
                    pols[m] = [E.exact_window(s, B) for s in sls]
                elif m == "RA-DSS DP-exact":
                    pols[m] = E.dp_exact(sls, B, lam)[0]
                elif m == "NB-VOI greedy":
                    pols[m] = E.nb_voi(coh.X[tr], coh.y[tr], C, B)
            if extra is not None:
                pols.update(extra(sls, C, B, coh, tr))
            for m, pol in pols.items():
                key = tuple(pol)
                if key not in cache:
                    cache[key] = E.evaluate(coh, tr, te, pol, seed=fi)[0]
                r = dict(tag=tag, rep=rep, fold=fi, budget_frac=bf, method=m,
                         auroc=cache[key], cost=E.realised_cost(pol, C),
                         q_train=sum(s.L[p] for s, p in zip(sls, pol)) / qtr,
                         q_test=sum(s.L[p] for s, p in zip(tsl, pol)) / qte,
                         switches=E.n_switches(pol),
                         viol=int(sum(1 for w, p in enumerate(pol)
                                      if sum(C[w, e] for e in from_mask(p, NP)) > B + 1e-9)),
                         masks=tuple(int(p) for p in pol))
                rows.append(r)
    return rows


BASE_METHODS = ["Cost-blind greedy", "Static reduct", "RA-DSS myopic",
                "RA-DSS DP-exact", "NB-VOI greedy"]


# ---------------------------------------------------------------------------
# (a) routine-or-STAT: genuinely time-varying finite prices
# ---------------------------------------------------------------------------
STAT_MULTS = (1.5, 2.0, 3.0)
STAT_BUDGETS = (0.10, 0.20, 0.30, 0.40)


ROUTINE = (0, 3)                 # routine collections at admission and +12 h (ASM)
POC = ("ABG", "LAC")             # point-of-care assays, no STAT premium (ASM)


def stat_cost_matrix(mult: float) -> np.ndarray:
    """Routine-or-STAT pricing (hypothetical): in the routine collection
    windows every assay costs its fee-schedule price; in the other windows every
    central-laboratory assay costs `mult` x its price, while the point-of-care
    assays keep their price.  Availability is as in the main design."""
    base = CS.cost_matrix()
    C = base.copy()
    for w in range(CS.N_WIN):
        if w in ROUTINE:
            continue
        for j, key in enumerate(CS.KEYS):
            if key not in POC and C[w, j] < INF:
                C[w, j] = mult * CLFS[j]
    return C


def _time_avg_extra(sls, C, B, coh, tr):
    """RA-DSS exact/window computed with each assay's TIME-AVERAGED finite
    price (the price model a static cost-sensitive method would carry), then
    repaired to the true prices by dropping the lowest-density member."""
    avg = np.array([np.mean([C[w, e] for w in range(C.shape[0]) if C[w, e] < INF])
                    for e in range(NP)])
    pol, viol = [], 0
    for w, s in enumerate(sls):
        true = s.cost.copy()
        s.set_cost(np.where(true < INF, avg, INF))
        m = E.exact_window(s, B)
        s.set_cost(true)
        if s.mcost[m] > B + 1e-9:
            viol += 1
        while s.mcost[m] > B + 1e-9:
            worst, wd = -1, INF
            for e in from_mask(m, NP):
                d = (s.L[m] - s.L[m & ~(1 << e)]) / s.cost[e]
                if d < wd:
                    worst, wd = e, d
            m &= ~(1 << worst)
        pol.append(m)
    _time_avg_extra.violations.append(viol)
    return {"Time-avg price + repair": pol}


_time_avg_extra.violations = []


def job_stat(rep):
    coh = CS.make_cohort(n=N_PATIENTS, seed=100 + rep, outcome_noise=SIG_OUT)
    rows = []
    for mult in STAT_MULTS:
        C = stat_cost_matrix(mult)
        _time_avg_extra.violations = []
        r = run_cells(coh, C, STAT_BUDGETS,
                      ["Static reduct", "RA-DSS myopic", "RA-DSS exact/window",
                       "RA-DSS DP-exact"], rep, f"stat{mult}",
                      panel=CS.FULL_PANEL_COST, extra=_time_avg_extra)
        for x in r:
            x["mult"] = mult
        rows += r
        rows.append(dict(tag=f"stat{mult}", rep=rep, mult=mult,
                         method="__violations__",
                         violations=list(_time_avg_extra.violations)))
    return rows


# ---------------------------------------------------------------------------
# (b) price structures, (c) correlated cohort
# ---------------------------------------------------------------------------
SENS_BUDGETS = (0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


def job_price(rep):
    coh = CS.make_cohort(n=N_PATIENTS, seed=100 + rep, outcome_noise=SIG_OUT)
    rows = []
    for mode in ("flat", "clfs", "spread", "dominant"):
        pv = price_vector(mode)
        C = cost_mat_w(6, pv)
        rows += run_cells(coh, C, SENS_BUDGETS, BASE_METHODS, rep, mode,
                          panel=float(pv.sum()))
    return rows


def job_corr(rep):
    rows = []
    for lab, corr in (("independent", False), ("correlated", True)):
        coh = make_cohort_w(seed=100 + rep, correlated=corr)
        rows += run_cells(coh, CS.cost_matrix(), SENS_BUDGETS, BASE_METHODS,
                          rep, lab, panel=CS.FULL_PANEL_COST)
    jb, jc = CS.KEYS.index("BMP"), CS.KEYS.index("CMP")
    for r in rows:
        r["nested_both"] = sum(1 for m in r["masks"] if (m >> jb) & 1 and (m >> jc) & 1)
    return rows


# ---------------------------------------------------------------------------
# (d) one-factor-at-a-time sensitivity
# ---------------------------------------------------------------------------
OFAT_BUDGETS = (0.075, 0.10, 0.15, 0.20)
OFAT = [("base", {}),
        ("k=1", dict(k=1)), ("k=2", dict(k=2)), ("k=4", dict(k=4)), ("k=5", dict(k=5)),
        ("sigma_out=1.5", dict(sig=1.5)), ("sigma_out=4.0", dict(sig=4.0)),
        ("subsample=200/400", dict(sub=(200, 400))),
        ("subsample=all", dict(sub=(None, None))),
        ("windows=4", dict(W=4)), ("windows=8", dict(W=8)),
        ("sharpness=0", dict(sharp=0.0)), ("sharpness=0.5", dict(sharp=0.5))]


def job_sens(rep):
    rows = []
    for tag, o in OFAT:
        W = o.get("W", 6)
        coh = make_cohort_w(seed=100 + rep, W=W, sharp=o.get("sharp", 1.0),
                            outcome_noise=o.get("sig", SIG_OUT))
        C = cost_mat_w(W, CLFS)
        mp, mn = o.get("sub", (400, 800))
        rows += run_cells(coh, C, OFAT_BUDGETS, BASE_METHODS, rep, tag,
                          k_red=o.get("k", CS.K_RED), max_pos=mp, max_neg=mn,
                          panel=CS.FULL_PANEL_COST)
    return rows


# ---------------------------------------------------------------------------
# (e) headroom, saturation
# ---------------------------------------------------------------------------
HEAD_BUDGETS = [0.05, 0.0625, 0.075, 0.0875, 0.10, 0.1125, 0.125, 0.15, 0.175,
                0.20, 0.25, 0.30, 0.40, 0.60, 0.80, 1.00]


def job_headroom(rep):
    coh = CS.make_cohort(n=N_PATIENTS, seed=100 + rep, outcome_noise=SIG_OUT)
    C = CS.cost_matrix()
    sls = E.build_slices(coh, np.arange(N_PATIENTS), C, sub_seed=rep)
    pooled = np.sum([s.L for s in sls], axis=0)
    rows = []
    for bf in HEAD_BUDGETS:
        B = bf * CS.FULL_PANEL_COST
        per = [E.exact_window(s, B) for s in sls]
        qsum = sum(s.L[m] for s, m in zip(sls, per))
        okall = np.ones(E.NM, dtype=bool)
        for s in sls:
            okall &= s.feasible(B)
        bfix = int(np.argmax(np.where(okall, pooled, -INF)))
        bv = float(pooled[bfix])
        rows.append(dict(rep=rep, budget_frac=bf, q_adaptive=qsum, q_fixed=bv,
                         gain_pct=100 * (qsum - bv) / bv if bv > 0 else 0.0,
                         n_distinct=len(set(per))))
    return rows


def job_sat(rep):
    coh = CS.make_cohort(n=N_PATIENTS, seed=100 + rep, outcome_noise=SIG_OUT)
    C = CS.cost_matrix()
    rng = np.random.default_rng(3000 + rep)
    rows = []
    for k in (1, 2, 3, 4, 5):
        s = E.LSlice(coh.X, coh.y, np.arange(N_PATIENTS), 0, C[0], k_red=k,
                     sub_seed=rep)
        qm = s.qmax()
        av = from_mask(s.avail_mask, NP)
        for size in (1, 2, 3, 4, 5, 6, 8, 10, 13):
            vals = [s.L[to_mask(rng.choice(av, size=size, replace=False))] / qm
                    for _ in range(20)]
            rows.append(dict(rep=rep, k=k, size=size, mean=float(np.mean(vals)),
                             lo=float(np.min(vals)), hi=float(np.max(vals)),
                             spread=float(np.max(vals) - np.min(vals))))
    return rows


JOBS = dict(stat=(job_stat, 20), price=(job_price, 20), corr=(job_corr, 20),
            sens=(job_sens, 10), headroom=(job_headroom, 50), sat=(job_sat, 20))


ORDER = ("sat", "headroom", "stat", "price", "corr", "sens", "inc")
HELP = """experiment3.py runs the secondary experiments.  Usage:
  python experiment3.py                 all of them, in this order: sat headroom stat price corr sens inc
  python experiment3.py <name> [n_proc] one of them
    sat       saturation of the quality functional (Fig. 2)
    headroom  headroom of time-varying panels (Fig. 9(a))
    stat      routine-or-STAT prices (Table 7, Fig. 11)
    price     price structures (Table 8, top)
    corr      correlated cohort (Table 8, bottom)
    sens      one-factor-at-a-time sensitivity (Table 10, Fig. 12)
    inc       warm-start re-selection (Section 7.5.1)
Each writes results/v2/<name>.pkl; then run analysis2.py."""


def run_inc():
    import experiment as X1
    X1.N_REP = 20
    t0 = time.time()
    rows = X1.run_incremental()
    with open(os.path.join(OUT, "inc.pkl"), "wb") as f:
        pickle.dump(rows, f)
    print("inc done", f"{time.time() - t0:.0f}s", flush=True)


def run_job(which, n_proc):
    if which == "inc":
        return run_inc()
    fn, n_rep = JOBS[which]
    t0 = time.time()
    print(f"{which}: {n_rep} replicates on {n_proc} process(es) ...", flush=True)
    res = run_all(fn, range(n_rep), n_proc, ordered=True)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(OUT, f"{which}.pkl"), "wb") as f:
        pickle.dump(rows, f)
    print(which, "done", len(rows), "rows", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)
    if which != "all" and which not in ORDER:
        sys.exit(f"Unknown experiment '{which}'.\n\n{HELP}")
    n_proc = n_proc_arg(2)
    jobs = ORDER if which == "all" else (which,)
    if which == "all":
        print(HELP, "\n", flush=True)
        print(f"Running all {len(jobs)} secondary experiments on {n_proc} process(es); "
              "this takes about 40 min with 1 process and about 25 min with 2.\n", flush=True)
    os.makedirs(OUT, exist_ok=True)
    for j in jobs:
        run_job(j, n_proc)
    print("all secondary experiments done; now run: python analysis2.py")
