"""Additional figures for Sections 5-7 (DP timing, cohort, routine-or-STAT).
Imported by analysis2.py; shares its style, paths and macro store."""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import casestudy as CS


def fig_dp_timing(RES, FIG, mac):
    """Section 5: measured cost of the exact DP and of the objective lattice."""
    p = os.path.join(RES, "dp_timing.pkl")
    if not os.path.exists(p):
        return None
    d = pd.DataFrame(pickle.load(open(p, "rb")))
    g = d[d.inst >= 0].groupby("n").mean(numeric_only=True)
    h = d[d.inst < 0].sort_values("T")
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
    n = g.index.values
    ax[0].semilogy(n, 1e3 * g.t_dt, "o-", color="#3f007d", label=r"distance transform, $O(|T|\,n\,2^n)$")
    ax[0].semilogy(n, 1e3 * g.t_naive, "s--", color="#e69f00", label=r"pairwise maximisation, $O(|T|\,4^n)$")
    ax[0].set_xlabel(r"number of parameters $n=|E|$")
    ax[0].set_ylabel("wall-clock time (ms)")
    ax[0].set_title(r"(a) exact DP over $\mathcal{P}(E)$, $|T|=6$")
    ax[0].legend(fontsize=8.5, loc="upper left")
    ax[1].semilogy(n, 1e3 * g.t_pt, "o-", color="#0072b2", label=r"pattern transform, $O(k\,n\,2^n)$")
    ax[1].semilogy(n, 1e3 * g.t_direct, "s--", color="#d55e00", label=r"direct evaluation of every $A$")
    ax[1].set_xlabel(r"number of parameters $n=|E|$")
    ax[1].set_ylabel("wall-clock time (ms)")
    ax[1].set_title(r"(b) tabulating $Q_t$ on $\mathcal{P}(E)$, $k=3$")
    ax[1].legend(fontsize=8.5, loc="upper left")
    for a in ax[:2]:
        a.set_xticks(n)
    ax[2].plot(h["T"], 1e3 * h.t_dt, "o-", color="#3f007d")
    ax[2].set_xlabel(r"horizon length $|T|$")
    ax[2].set_ylabel("wall-clock time (ms)")
    ax[2].set_title(r"(c) exact DP at $n=12$ against $|T|$")
    ax[2].set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dp_timing.png"))
    fig.savefig(os.path.join(FIG, "fig_dp_timing.pdf"))
    plt.close(fig)
    mac("TimeDtFourteen", 1e3 * g.t_dt.loc[14], ".0f")
    mac("TimeNaiveEleven", 1e3 * g.t_naive.loc[11], ".0f")
    mac("TimeDtEleven", 1e3 * g.t_dt.loc[11], ".1f")
    mac("TimeDirectTwelve", 1e3 * g.t_direct.loc[12], ".0f")
    mac("TimePtTwelve", 1e3 * g.t_pt.loc[12], ".1f")
    mac("TimeAgreeN", int(d[d.inst >= 0].agree.notna().sum()))
    return g


def fig_cohort(FIG, mac, n_patients, out_noise):
    """Section 6: severity trajectories and time-varying assay informativeness."""
    coh = CS.make_cohort(n=n_patients, seed=100, outcome_noise=out_noise)
    z, y, X = coh.z, coh.y, coh.X
    w = np.arange(CS.N_WIN)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    rng = np.random.default_rng(0)
    for lab, cls, col in ((r"no deterioration ($y=0$)", 0, "#0072b2"),
                          (r"deterioration ($y=1$)", 1, "#d55e00")):
        zz = z[y == cls]
        for i in rng.choice(len(zz), 25, replace=False):
            ax[0].plot(w, zz[i], color=col, lw=0.5, alpha=0.25)
        q1, q3 = np.percentile(zz, [25, 75], axis=0)
        ax[0].fill_between(w, q1, q3, color=col, alpha=0.18, lw=0)
        ax[0].plot(w, zz.mean(0), color=col, lw=2.2, label=lab)
    ax[0].set_xticks(w)
    ax[0].set_xticklabels([f"$t_{i}$" for i in w])
    ax[0].set_xlabel(r"window $t$")
    ax[0].set_ylabel(r"latent severity $z_i(t)$")
    ax[0].set_title(r"(a) severity by outcome: mean, IQR and $25$ paths")
    ax[0].legend(loc="upper left", fontsize=9)
    order = sorted(range(CS.N_PAR), key=lambda j: (CS.TESTS[j]["phase"] != "early", CS.COST[j]))
    A = np.array([[CS.auroc(y, X[:, t, j].astype(float)) for t in range(CS.N_WIN)] for j in order])
    vmax = max(0.7, A.max())
    im = ax[1].imshow(A, cmap="viridis", vmin=0.5, vmax=vmax, aspect="auto")
    ax[1].set_yticks(range(CS.N_PAR))
    ax[1].set_yticklabels([f"{CS.KEYS[j]} ({CS.TESTS[j]['phase'][0]})" for j in order], fontsize=9)
    ax[1].set_xticks(w)
    ax[1].set_xticklabels([f"$t_{i}$" for i in w])
    ax[1].set_xlabel(r"window $t$")
    ax[1].grid(False)
    nearly = sum(CS.TESTS[j]["phase"] == "early" for j in order)
    ax[1].axhline(nearly - 0.5, color="white", lw=2)
    for (r, c), v in np.ndenumerate(A):
        ax[1].text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                   color=("white" if v < 0.5 + 0.6 * (vmax - 0.5) else "black"))
    cb = fig.colorbar(im, ax=ax[1], fraction=0.045, pad=0.02)
    cb.set_label(r"single-assay AUROC of $X_{i,e,t}$")
    ax[1].set_title(r"(b) informativeness by window (e: early, l: late phase)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_cohort.png"))
    fig.savefig(os.path.join(FIG, "fig_cohort.pdf"))
    plt.close(fig)
    ear = [r for r, j in enumerate(order) if CS.TESTS[j]["phase"] == "early"]
    lat = [r for r, j in enumerate(order) if CS.TESTS[j]["phase"] != "early"]
    mac("CohEarlyFirst", A[ear, 0].mean(), ".2f")
    mac("CohEarlyLast", A[ear, -1].mean(), ".2f")
    mac("CohLateFirst", A[lat, 0].mean(), ".2f")
    mac("CohLateLast", A[lat, -1].mean(), ".2f")


def fig_stat(df, FIG, tci):
    """Section 7: routine-or-STAT pricing."""
    if df is None:
        return
    poc = [CS.KEYS.index(k) for k in ("ABG", "LAC")]
    mults = sorted(df.mult.unique())
    buds = sorted(df.budget_frac.unique())
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.1))
    ex = df[df.method == "RA-DSS exact/window"]
    width = 0.13
    x = np.arange(len(buds))
    for k, m in enumerate(mults):
        sr, ss = [], []
        for bf in buds:
            nr = ns = pr_ = ps = 0
            for masks in ex[(ex.mult == m) & (ex.budget_frac == bf)].masks:
                for wdx, mm in enumerate(masks):
                    kk = sum((mm >> j) & 1 for j in range(CS.N_PAR))
                    kp = sum((mm >> j) & 1 for j in poc)
                    if wdx in (0, 3):
                        nr += kk; pr_ += kp
                    else:
                        ns += kk; ps += kp
            sr.append(100 * pr_ / max(nr, 1))
            ss.append(100 * ps / max(ns, 1))
        off = (k - 1) * 2 * width
        ax[0].bar(x + off - width / 2, sr, width, color=["#c6dbef", "#6baed6", "#08519c"][k],
                  label=f"routine, $m={m:g}$")
        ax[0].bar(x + off + width / 2, ss, width, color=["#fdd0a2", "#fd8d3c", "#a63603"][k],
                  label=f"STAT, $m={m:g}$")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{100 * b:.0f}%" for b in buds])
    ax[0].set_xlabel(r"per-window budget $B_t$ (share of panel price)")
    ax[0].set_ylabel("point-of-care share of selection (%)")
    ax[0].set_title("(a) point-of-care share, routine vs STAT")
    ax[0].legend(fontsize=7.5, ncol=2, loc="upper right")
    ax[0].set_ylim(0, 65)
    styles = {1.5: ("o", "-"), 2.0: ("s", "--"), 3.0: ("^", ":")}
    cols = {1.5: "#56b4e9", 2.0: "#0072b2", 3.0: "#3f007d"}
    for panel, (val, lab) in enumerate((("q_test", r"held-out $Q$"), ("auroc", "AUROC"))):
        a = ax[panel + 1]
        for m in mults:
            mu, hw = [], []
            for bf in buds:
                s = df[(df.mult == m) & (df.budget_frac == bf)]
                w_ = s.groupby(["method", "rep"])[val].mean()
                d = (w_.xs("RA-DSS exact/window", level="method")
                     - w_.xs("Time-avg price + repair", level="method")).values
                mm, hh = tci(d)
                mu.append(mm); hw.append(hh)
            mk, ls = styles[m]
            a.errorbar(100 * np.array(buds), mu, yerr=hw, marker=mk, ls=ls,
                       color=cols[m], capsize=2.5, ms=5.5, elinewidth=1.0, zorder=3)
        a.axhline(0, color="k", lw=0.8)
        a.set_xticks([100 * b for b in buds])
        a.set_xlim(100 * buds[0] - 4, 100 * buds[-1] + 4)
        a.set_xticklabels([f"{100 * b:.0f}%" for b in buds])
        a.set_xlabel(r"per-window budget $B_t$ (share of panel price)")
        a.set_ylabel(r"$\Delta$" + lab + ": true minus time-averaged")
        a.set_title(f"({'bc'[panel]}) gain in {lab} from true prices")
        from matplotlib.lines import Line2D
        hs = [Line2D([0], [0], color=cols[m], marker=styles[m][0], ls=styles[m][1], ms=5.5, lw=1.5,
                     label=f"$m={m:g}$") for m in mults]
        a.legend(handles=hs, fontsize=8.5, title="STAT multiplier", title_fontsize=8.5,
                 loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_stat.png"))
    fig.savefig(os.path.join(FIG, "fig_stat.pdf"))
    plt.close(fig)
