"""
analysis2.py -- turns results/v2/*.pkl into every table, figure and number
quoted in the paper.

Outputs:
  outputs/numbers2.tex      LaTeX macros (auto-generated; do not edit)
  outputs/tables/*.tex      table bodies
  outputs/figures/*.png     figures (a PDF copy of each is also written)
  results/v2/summary2.json  all headline statistics
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import casestudy as CS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results", "v2")
TAB = os.path.join(ROOT, "outputs", "tables")
FIG = os.path.join(ROOT, "outputs", "figures")
PAP = os.path.join(ROOT, "outputs")
for d in (TAB, FIG):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
                     "font.size": 11, "legend.fontsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "savefig.bbox": "tight", "figure.dpi": 150, "savefig.dpi": 300})

REF = "Static reduct"
ORDER = ["Random (budget)", "Cost-blind greedy", "Static reduct", "Static exact",
         "SS-reduct", "TCS-reduct", "L1-cost", "Wrapper", "NB-VOI greedy", "GA", "RA-DSS myopic",
         "RA-DSS myopic+enum", "RA-DSS exact/window", "RA-DSS DP",
         "RA-DSS DP-exact", "RA-DSS DP-exact (w)"]
# the fourteen policies of the main experiment (experiment2.py); the two others
# come from experiment4.py and are paired with them replicate by replicate
ORDER_MAIN = [m for m in ORDER if m not in ("Static exact", "RA-DSS DP-exact (w)", "SS-reduct")]
REF2 = "Static exact"
LABEL = {"Random (budget)": "Random", "Cost-blind greedy": "Cost-blind greedy",
         "Static reduct": "Static reduct", "TCS-reduct": "TCS-reduct",
         "L1-cost": r"$\ell_1$-cost", "Wrapper": "Wrapper",
         "NB-VOI greedy": "NB-VOI greedy", "GA": "GA",
         "RA-DSS myopic": "RA-DSS myopic", "RA-DSS myopic+enum": "RA-DSS myopic+enum",
         "RA-DSS exact/window": "RA-DSS exact/window", "RA-DSS DP": "RA-DSS DP",
         "RA-DSS DP-exact": "RA-DSS DP-exact", "Full panel": "Full panel",
         "Static exact": "Static exact", "RA-DSS DP-exact (w)": "RA-DSS DP-exact ($w$)",
         "SS-reduct": "SS-reduct"}
COL = {"Random (budget)": "#9a9a9a", "Cost-blind greedy": "#cc79a7",
       "Static reduct": "#d55e00", "TCS-reduct": "#8c564b", "L1-cost": "#e69f00",
       "Wrapper": "#7f7f7f", "NB-VOI greedy": "#009e73", "GA": "#bcbd22",
       "RA-DSS myopic": "#0072b2", "RA-DSS myopic+enum": "#56b4e9",
       "RA-DSS exact/window": "#17becf", "RA-DSS DP": "#6a3d9a",
       "RA-DSS DP-exact": "#3f007d", "Full panel": "black",
       "Static exact": "#8c510a", "RA-DSS DP-exact (w)": "#c51b7d", "SS-reduct": "#4d4d4d"}
MAC = {}
MK = {"Cost-blind greedy": "v", "Static reduct": "s", "NB-VOI greedy": "D", "Wrapper": "P",
      "RA-DSS myopic": "o", "RA-DSS DP-exact": "^", "Static exact": "X"}
LS = {"Cost-blind greedy": (0, (4, 1.5)), "Static reduct": "-", "NB-VOI greedy": "-", "Wrapper": (0, (1.5, 1.2)),
      "RA-DSS myopic": "-", "RA-DSS DP-exact": (0, (6, 2)), "Static exact": "-"}
BUD_TICKS = [5, 7.5, 10, 15, 20, 30, 40, 60, 100]


def budget_axis(a):
    """log budget axis labelled in plain percentages (5, 10, 20, ... %)."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    a.set_xscale("log")
    a.xaxis.set_major_locator(FixedLocator(BUD_TICKS))
    a.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in BUD_TICKS]))
    a.xaxis.set_minor_locator(NullLocator())
    a.tick_params(axis="x", labelsize=8.5)


def policy_handles(meths):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=COL[m], marker=MK.get(m, "o"), ms=5, lw=1.4, ls=LS.get(m, "-"),
                   label=LABEL[m]) for m in meths]


def mac(name, val, fmt=None):
    if fmt is not None:
        val = format(val, fmt)
    MAC[name] = str(val)


def tex_group(v):
    """Thousands separators for plain numbers of four or more integer digits,
    written as {,} so that they also typeset correctly in math mode."""
    m = re.fullmatch(r"([+-]?)(\d{4,})(\.\d+)?", str(v))
    if not m:
        return v
    return m.group(1) + f"{int(m.group(2)):,}".replace(",", "{,}") + (m.group(3) or "")


def tci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean()), float(h)


def wilcox(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    x = x[x != 0]
    if len(x) < 5:
        return np.nan
    return float(stats.wilcoxon(x).pvalue)


def fmt_pm(p):
    """p-value as a math-mode relation: '<0.001' or '=0.042'."""
    if p is None or np.isnan(p):
        return "=\\text{n/a}"
    if p < 0.001:
        return "<0.001"
    return f"={p:.3f}"


def fmt_p(p):
    if p is None or np.isnan(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


# ---------------------------------------------------------------------------
def load_main():
    files = sorted(glob.glob(os.path.join(RES, "rep_*.pkl")))
    R = {k: [] for k in ("rows", "mp", "lam", "sel", "win", "delong")}
    for f in files:
        d = pickle.load(open(f, "rb"))
        for k in R:
            R[k] += d[k]
    ext, W = [], []
    for f in sorted(glob.glob(os.path.join(RES, "ext_*.pkl"))):
        d = pickle.load(open(f, "rb"))
        ext += [{k: v for k, v in r.items() if k != "masks"} for r in d["rows"]
                if r["method"] != "Static reduct (check)"]
        W += d["weights"]
    for f in sorted(glob.glob(os.path.join(RES, "ssr_*.pkl"))):     # experiment5.py
        d = pickle.load(open(f, "rb"))
        ext += [{k: v for k, v in r.items() if k != "masks"} for r in d["rows"]]
    out = {k: pd.DataFrame(v) for k, v in R.items()}
    if ext:
        out["rows"] = pd.concat([out["rows"], pd.DataFrame(ext)], ignore_index=True)
    out["weights"] = pd.DataFrame(W)
    out["n_rep"] = len(files)
    return out


def per_rep(df, value):
    return df.groupby(["method", "budget_frac", "rep"])[value].mean().reset_index()


def interp_cost(cost, auc, target):
    """Cost at which the (cost, AUROC) curve, traversed in budget order with
    AUROC made non-decreasing, first reaches target (linear interpolation)."""
    auc = np.maximum.accumulate(np.asarray(auc, float))
    cost = np.asarray(cost, float)
    if auc[0] >= target:
        return cost[0]
    for i in range(1, len(auc)):
        if auc[i] >= target:
            a0, a1 = auc[i - 1], auc[i]
            f = (target - a0) / (a1 - a0) if a1 > a0 else 1.0
            return cost[i - 1] + f * (cost[i] - cost[i - 1])
    return np.nan


def main_results(M):
    df = M["rows"]
    nrep = M["n_rep"]
    mac("NRepV", nrep)
    S = {"n_rep": nrep}
    budgets = sorted(df.budget_frac.unique())
    full = df[df.method == "Full panel"].groupby("rep").auroc.mean()
    fullcost = df[df.method == "Full panel"].cost.mean()
    m, h = tci(full.values)
    mac("FullAUROC", m, ".3f"); mac("FullAUROCci", h, ".3f")
    mac("FullCost", f"\\${fullcost:.0f}")
    S["full_auroc"] = (m, h)
    pr = {v: per_rep(df, v) for v in ("auroc", "cost", "q_train", "q_test", "switches")}
    wide = {v: pr[v].pivot_table(index=["budget_frac", "rep"], columns="method",
                                 values=v) for v in pr}
    # --- paired differences vs static reduct at every budget -----------------
    diffs = []
    for bf in budgets:
        A = wide["auroc"].loc[bf]
        Qt = wide["q_test"].loc[bf]
        for meth in ORDER:
            if meth not in A.columns or meth == REF:
                continue
            d = (A[meth] - A[REF]).values
            dq = (Qt[meth] - Qt[REF]).values
            mm, hh = tci(d)
            mq, hq = tci(dq)
            diffs.append(dict(budget_frac=bf, method=meth, d_auc=mm, d_auc_ci=hh,
                              p_auc=wilcox(d), d_q=mq, d_q_ci=hq, p_q=wilcox(dq),
                              n=int(np.sum(~np.isnan(d)))))
    D = pd.DataFrame(diffs)
    D.to_csv(os.path.join(RES, "paired_diffs.csv"), index=False)
    # --- operating-range mean AUROC (5%..30%) --------------------------------
    op = pr["auroc"][(pr["auroc"].budget_frac >= 0.05 - 1e-9) &
                     (pr["auroc"].budget_frac <= 0.30 + 1e-9)]
    opm = op.groupby(["method", "rep"]).auroc.mean().unstack(0)
    # --- interpolated cost to reach shares of full-panel AUROC ---------------
    cost_at = {}
    for meth in ORDER + ["Full panel"]:
        sub_a = pr["auroc"][pr["auroc"].method == meth]
        sub_c = pr["cost"][pr["cost"].method == meth]
        for tgt in (0.97, 0.98, 0.99):
            vals = []
            for r in range(nrep):
                a = sub_a[sub_a.rep == r].sort_values("budget_frac")
                c = sub_c[sub_c.rep == r].sort_values("budget_frac")
                if len(a) == 0:
                    vals.append(np.nan)
                    continue
                vals.append(interp_cost(c.cost.values, a.auroc.values,
                                        tgt * full.loc[r]))
            cost_at[(meth, tgt)] = np.array(vals)
    # --- main table at 10% --------------------------------------------------
    bf0 = 0.10
    lines = []
    for meth in ORDER:
        A = wide["auroc"].loc[bf0]
        if meth not in A.columns:
            continue
        a, ah = tci(A[meth].values)
        c, ch = tci(wide["cost"].loc[bf0][meth].values)
        qs, qsh = tci(wide["q_train"].loc[bf0][meth].values)
        qt, qth = tci(wide["q_test"].loc[bf0][meth].values)
        swd = wide["switches"].loc[bf0]
        sw = swd[meth].values if meth in swd.columns else np.array([np.nan])
        s, sh = tci(sw)
        if meth == REF:
            dtxt = "--"
        else:
            row = D[(D.budget_frac == bf0) & (D.method == meth)].iloc[0]
            dtxt = f"${row.d_auc:+.4f}\\pm{row.d_auc_ci:.4f}$"
        stxt = "--" if np.isnan(s) else f"${s:.1f}\\pm{sh:.1f}$"
        lines.append(f"{LABEL[meth]} & ${a:.4f}\\pm{ah:.4f}$ & {dtxt} & "
                     f"${c:.0f}\\pm{ch:.0f}$ & ${qs:.3f}\\pm{qsh:.3f}$ & "
                     f"${qt:.3f}\\pm{qth:.3f}$ & {stxt} \\\\")
    body = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
            "policy & AUROC & $\\Delta$AUROC vs static & cost (\\$) & "
            "$Q$ in-sample & $Q$ held-out & changes \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\midrule\n"
            f"Full panel & ${S['full_auroc'][0]:.4f}\\pm{S['full_auroc'][1]:.4f}$ & -- & "
            f"{fullcost:.0f} & 1 & 1 & 22.0 \\\\\n\\bottomrule\n\\end{{tabular}}")
    open(os.path.join(TAB, "tab_main.tex"), "w").write(body)
    # --- cost table ----------------------------------------------------------
    lines = []
    for meth in ORDER:
        if meth not in opm.columns:
            continue
        cells = []
        for tgt in (0.97, 0.98, 0.99):
            v = cost_at[(meth, tgt)]
            reach = np.mean(~np.isnan(v))
            mm, hh = tci(v)
            if np.isnan(mm):
                cells.append("--")
            else:
                star = "" if reach >= 0.999 else f"$^{{{int(round(100 * reach))}\\%}}$"
                cells.append(f"${mm:.0f}\\pm{hh:.0f}${star}")
        o, oh = tci(opm[meth].values)
        if meth == REF:
            dtxt = "--"
            ptxt = "--"
        else:
            dd = (opm[meth] - opm[REF]).values
            md, hd = tci(dd)
            dtxt = f"${md:+.4f}\\pm{hd:.4f}$"
            ptxt = fmt_p(wilcox(dd))
        lines.append(f"{LABEL[meth]} & " + " & ".join(cells) +
                     f" & ${o:.4f}\\pm{oh:.4f}$ & {dtxt} & {ptxt} \\\\")
    body = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
            "& \\multicolumn{3}{c}{cost (\\$) to reach share of full-panel AUROC} & "
            "\\multicolumn{3}{c}{mean AUROC, budgets 5--30\\%} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "policy & 97\\% & 98\\% & 99\\% & mean & $\\Delta$ vs static & $p$ \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    open(os.path.join(TAB, "tab_cost.tex"), "w").write(body)
    # --- macros --------------------------------------------------------------
    def dget(meth, bf, col):
        return float(D[(D.budget_frac == bf) & (D.method == meth)][col].iloc[0])
    names = {"RA-DSS myopic": "Myo", "RA-DSS DP-exact": "Dpx", "RA-DSS DP": "Dpr",
             "RA-DSS exact/window": "Exw", "NB-VOI greedy": "Voi",
             "Cost-blind greedy": "Blind", "Wrapper": "Wrap", "L1-cost": "Lone",
             "TCS-reduct": "Tcs", "GA": "Ga", "RA-DSS myopic+enum": "Enum",
             "Random (budget)": "Rand", "Static reduct": "Static",
             "Static exact": "Sex", "RA-DSS DP-exact (w)": "Dpw", "SS-reduct": "Ssr"}
    for meth, nm in names.items():
        for tgt, tn in ((0.97, "NS"), (0.98, "NE"), (0.99, "NN")):
            v = cost_at[(meth, tgt)]
            mm, hh = tci(v)
            mac(f"Cost{tn}{nm}", "--" if np.isnan(mm) else f"\\${mm:.0f}")
            mac(f"Cost{tn}{nm}ci", "--" if np.isnan(hh) else f"{hh:.0f}")
            mac(f"Cost{tn}{nm}Pct", "--" if np.isnan(mm) else f"{100 * mm / fullcost:.1f}")
            mac(f"Reach{tn}{nm}", f"{100 * np.mean(~np.isnan(v)):.0f}")
        o, oh = tci(opm[meth].values)
        mac(f"Op{nm}", o, ".4f"); mac(f"Op{nm}ci", oh, ".4f")
        if meth != REF:
            dd = (opm[meth] - opm[REF]).values
            md, hd = tci(dd)
            mac(f"OpD{nm}", f"{md:+.4f}"); mac(f"OpD{nm}ci", hd, ".4f")
            mac(f"OpP{nm}", fmt_pm(wilcox(dd)))
            for bf, bn in ((0.10, "Ten"), (0.075, "Seven"), (0.20, "Twenty"), (0.40, "Forty"), (0.60, "Sixty"), (0.15, "Fifteen"), (0.25, "TwentyFive"), (0.1125, "Eleven")):
                mac(f"DAuc{nm}{bn}", f"{dget(meth, bf, 'd_auc'):+.4f}")
                mac(f"DAuc{nm}{bn}ci", dget(meth, bf, "d_auc_ci"), ".4f")
                mac(f"PAuc{nm}{bn}", fmt_pm(dget(meth, bf, "p_auc")))
                mac(f"DQ{nm}{bn}", f"{dget(meth, bf, 'd_q'):+.3f}")
                mac(f"DQ{nm}{bn}ci", dget(meth, bf, "d_q_ci"), ".3f")
                mac(f"PQ{nm}{bn}", fmt_pm(dget(meth, bf, "p_q")))
        for bf, bn in ((0.10, "Ten"),):
            for v in ("auroc", "q_test", "q_train", "cost", "switches"):
                wv = wide[v].loc[bf]
                if meth not in wv.columns:
                    continue
                mm, hh = tci(wv[meth].values)
                if not np.isnan(mm):
                    f = ".4f" if v == "auroc" else (".3f" if v.startswith("q") else ".1f")
                    mac(f"{v.replace('_', '').capitalize()}{nm}{bn}", mm, f)
    # count of budgets (5..30%) at which DP-exact / myopic beat static significantly
    for meth, nm in (("RA-DSS DP-exact", "Dpx"), ("RA-DSS myopic", "Myo"),
                     ("NB-VOI greedy", "Voi"), ("RA-DSS exact/window", "Exw")):
        sub = D[(D.method == meth) & (D.budget_frac <= 0.30 + 1e-9)]
        mac(f"NBud{nm}Pos", int(np.sum((sub.d_auc - sub.d_auc_ci) > 0)))
        mac(f"NBud{nm}Neg", int(np.sum((sub.d_auc + sub.d_auc_ci) < 0)))
        mac(f"NBud{nm}Qpos", int(np.sum((sub.d_q - sub.d_q_ci) > 0)))
    mac("NBudOp", int(np.sum(np.array(budgets) <= 0.30 + 1e-9)))
    # --- DeLong summary ------------------------------------------------------
    DL = M["delong"]
    for meth, nm in (("RA-DSS DP-exact", "Dpx"), ("RA-DSS myopic", "Myo"),
                     ("NB-VOI greedy", "Voi")):
        sub = DL[(DL.method == meth) & (DL.budget_frac == 0.10)]
        sig_pos = np.mean((sub.p_vs_static < 0.05) & (sub.auc_oof > sub.auc_ref_static))
        sig_neg = np.mean((sub.p_vs_static < 0.05) & (sub.auc_oof < sub.auc_ref_static))
        mac(f"DL{nm}Pos", f"{100 * sig_pos:.0f}")
        mac(f"DL{nm}Neg", f"{100 * sig_neg:.0f}")
    sub = DL[(DL.method == "NB-VOI greedy") & (DL.budget_frac == 0.10)]
    mac("DLVoiVsMyoPos", f"{100 * np.mean((sub.p_vs_myopic < 0.05) & (sub.auc_oof > sub.auc_ref_myopic)):.0f}")
    mac("DLVoiVsMyoNeg", f"{100 * np.mean((sub.p_vs_myopic < 0.05) & (sub.auc_oof < sub.auc_ref_myopic)):.0f}")
    # --- association between Q and AUROC -------------------------------------
    agg = df[df.method.isin(ORDER_MAIN)].groupby(["method", "budget_frac"])[["auroc", "q_test"]].mean()
    rho_all = stats.spearmanr(agg.q_test, agg.auroc).correlation
    mac("RhoQAuc", rho_all, ".2f")
    within = []
    for bf in budgets:
        a = agg.xs(bf, level="budget_frac")
        if len(a) > 3:
            within.append(stats.spearmanr(a.q_test, a.auroc).correlation)
    mac("RhoQAucWithin", float(np.nanmedian(within)), ".2f")
    S.update(dict(diffs=D.to_dict("records")))
    return S, D, pr, wide, cost_at, opm, full, fullcost


# ---------------------------------------------------------------------------
def fig_frontier(pr, D, fullauc):
    show = ["Cost-blind greedy", "Static reduct", "Static exact", "NB-VOI greedy", "Wrapper",
            "RA-DSS myopic", "RA-DSS DP-exact"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for meth in show:
        kw = dict(color=COL[meth], marker=MK[meth], ms=3.8, lw=1.2, ls=LS[meth], capsize=2,
                  elinewidth=0.8, markeredgewidth=0.6)
        a = pr["auroc"][pr["auroc"].method == meth].groupby("budget_frac").auroc
        c = pr["cost"][pr["cost"].method == meth].groupby("budget_frac").cost.mean()
        m = a.mean()
        h = a.apply(lambda x: tci(x.values)[1])
        ax[0].errorbar(c.values, m.values, yerr=h.values, **kw)
        q = pr["q_test"][pr["q_test"].method == meth].groupby("budget_frac").q_test
        ax[2].errorbar(c.values, q.mean().values, yerr=q.apply(lambda x: tci(x.values)[1]).values, **kw)
    ax[0].axhline(fullauc, color="k", ls="--", lw=0.8)
    ax[0].text(0.98, 0.04, f"full panel: ${fullauc:.3f}$", transform=ax[0].transAxes,
               ha="right", va="bottom", fontsize=8)
    ax[0].set_xscale("log"); ax[2].set_xscale("log")
    ax[0].set_xlabel(r"cost per patient-day (US\$)"); ax[0].set_ylabel("held-out AUROC")
    ax[0].set_title("(a) discrimination")
    for meth in [m for m in show if m != REF]:
        s_ = D[D.method == meth].sort_values("budget_frac")
        ax[1].errorbar(100 * s_.budget_frac, s_.d_auc, yerr=s_.d_auc_ci, color=COL[meth],
                       marker=MK[meth], ms=3.8, lw=1.2, ls=LS[meth], capsize=2, elinewidth=0.8,
                       markeredgewidth=0.6)
    ax[1].axhline(0, color=COL[REF], lw=1.2)
    budget_axis(ax[1]); ax[1].set_xlabel(r"per-window budget $b$ (% of panel)")
    ax[1].set_ylabel(r"$\Delta$AUROC vs static reduct")
    ax[1].set_title("(b) paired difference")
    ax[2].set_xlabel(r"cost per patient-day (US\$)")
    ax[2].set_ylabel(r"held-out $\Sigma_t\, Q_t(A_t)/\Sigma_t\, Q_t(E^{\mathrm{av}}_t)$")
    ax[2].set_title(r"(c) held-out objective $Q$")
    fig.legend(handles=policy_handles(show), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_frontier.{ext}"))
    plt.close(fig)


def fig_selection(M):
    sel = M["sel"]
    meths = ["Static reduct", "RA-DSS myopic", "RA-DSS exact/window",
             "RA-DSS DP-exact", "NB-VOI greedy"]
    order = sorted(range(CS.N_PAR), key=lambda j: CS.COST[j])
    fig, ax = plt.subplots(1, len(meths), figsize=(11, 4.2), sharey=True)
    freqs = {}
    for k, meth in enumerate(meths):
        s = sel[sel.method == meth]
        F = np.zeros((CS.N_PAR, CS.N_WIN))
        for masks in s.masks:
            for w, m in enumerate(masks):
                for j in range(CS.N_PAR):
                    F[j, w] += (m >> j) & 1
        F /= max(len(s), 1)
        freqs[meth] = F
        im = ax[k].imshow(F[order], aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax[k].set_title(LABEL[meth].replace("RA-DSS ", "RA-DSS\n"), fontsize=10)
        ax[k].set_xticks(range(CS.N_WIN))
        ax[k].set_xticklabels([f"$t_{w}$" for w in range(CS.N_WIN)])
        ax[k].grid(False)
    ax[0].set_yticks(range(CS.N_PAR))
    ax[0].set_yticklabels([f"{CS.KEYS[j]} ({CS.TESTS[j]['phase'][0]})" for j in order])
    fig.colorbar(im, ax=ax, shrink=0.8, label="selection frequency")
    fig.text(0.44, -0.02, r"window $t$", ha="center")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_selection.{ext}"))
    plt.close(fig)
    return freqs


def selection_stats(M, freqs):
    sel = M["sel"]
    out = {}
    for meth, nm in (("RA-DSS myopic", "Myo"), ("RA-DSS exact/window", "Exw"),
                     ("RA-DSS DP-exact", "Dpx"), ("Static reduct", "Static"),
                     ("RA-DSS myopic+enum", "Enum")):
        s = sel[sel.method == meth]
        nd = [len(set(m)) for m in s.masks]
        mac(f"Distinct{nm}", np.mean(nd), ".2f")
    # myopic vs exact per window: fraction of windows where they differ
    a = sel[sel.method == "RA-DSS myopic"].set_index(["rep", "fold"]).masks
    b = sel[sel.method == "RA-DSS exact/window"].set_index(["rep", "fold"]).masks
    diff = [np.mean([x != y for x, y in zip(a[k], b[k])]) for k in a.index]
    mac("MyoExwDiffPct", 100 * np.mean(diff), ".0f")
    # share of selections that are late-phase in t0..t2 vs t3..t5
    late = [j for j in range(CS.N_PAR) if CS.TESTS[j]["phase"] == "late"]
    for meth, nm in (("RA-DSS myopic", "Myo"), ("RA-DSS exact/window", "Exw"),
                     ("RA-DSS DP-exact", "Dpx"), ("Static reduct", "Static")):
        F = freqs[meth]
        e = F[late, :3].sum() / max(F[:, :3].sum(), 1e-9)
        l = F[late, 3:].sum() / max(F[:, 3:].sum(), 1e-9)
        mac(f"LateEarly{nm}", 100 * e, ".0f")
        mac(f"LateLate{nm}", 100 * l, ".0f")
    for key in ("PCT", "BNP", "CULT", "ABG"):
        j = CS.KEYS.index(key)
        mac(f"Sel{key}", 100 * max(freqs[m][j].max() for m in freqs), ".0f")


# ---------------------------------------------------------------------------
def mp_results(M):
    mp = M["mp"].copy()
    mp["gap"] = mp.J_restricted / mp.J_exact
    ok = mp[mp.J_exact > 0]
    mac("DprGapMean", ok.gap.mean(), ".4f")
    mac("DprGapMin", ok.gap.min(), ".4f")
    mac("DprExactPct", 100 * np.mean(ok.gap > 1 - 1e-9), ".1f")
    win = M["win"]
    for c, nm in (("r_guarded", "Guard"), ("r_enum1", "Enum"), ("r_blind", "Blind"),
                  ("r_ga", "Ga")):
        mac(f"Gap{nm}Mean", win[c].mean(), ".4f")
        mac(f"Gap{nm}Min", win[c].min(), ".4f")
        mac(f"Gap{nm}Exact", 100 * np.mean(win[c] > 1 - 1e-9), ".1f")
    mac("GapNSlices", len(win))
    mac("BoundRatioMean", win.opt_over_U.mean(), ".3f")
    mac("BoundRatioMin", win.opt_over_U.min(), ".3f")
    return mp, win


def lam_results(M):
    L = M["lam"]
    g = L.groupby(["dp", "gamma"])[["q_train", "q_test", "switches", "auroc",
                                    "n_distinct", "cost"]].mean().reset_index()
    ex = g[g.dp == "exact"].set_index("gamma")
    mac("LamZeroSwitch", ex.loc[0.0, "switches"], ".1f")
    mac("LamZeroQ", ex.loc[0.0, "q_train"], ".3f")
    mac("LamZeroDistinct", ex.loc[0.0, "n_distinct"], ".1f")
    for gm, nm in ((0.2, "Knee"), (0.5, "Half"), (1.0, "One")):
        mac(f"Lam{nm}Switch", ex.loc[gm, "switches"], ".1f")
        mac(f"Lam{nm}Q", ex.loc[gm, "q_train"], ".3f")
        mac(f"Lam{nm}QDrop", 100 * (1 - ex.loc[gm, "q_train"] / ex.loc[0.0, "q_train"]), ".1f")
        mac(f"Lam{nm}SwitchDrop", 100 * (1 - ex.loc[gm, "switches"] / ex.loc[0.0, "switches"]), ".0f")
        mac(f"Lam{nm}Distinct", ex.loc[gm, "n_distinct"], ".2f")
    mac("LamAurocMin", ex.auroc.min(), ".4f"); mac("LamAurocMax", ex.auroc.max(), ".4f")
    # paired AUROC difference gamma=1 vs gamma=0 (exact DP)
    pv = L[L.dp == "exact"].groupby(["rep", "gamma"]).auroc.mean().unstack()
    d = (pv[1.0] - pv[0.0]).values
    m, h = tci(d)
    mac("LamAucDiff", f"{m:+.4f}"); mac("LamAucDiffci", h, ".4f")
    mac("LamAucDiffP", fmt_pm(wilcox(d)))
    return g


def fig_lambda_cert(g, cert):
    """(a) objective and (b) panel changes against the switching price, one
    colour/marker/line style per dynamic program; (c) certificate tightness."""
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    style = {"exact": dict(color="#3f007d", ls="-", marker="o", label="exact DP (Thm 5.3)"),
             "restricted": dict(color="#e69f00", ls="--", marker="s", label="restricted DP (Alg. 3)")}
    for lab in ("exact", "restricted"):
        s = g[g.dp == lab].sort_values("gamma")
        ax[0].plot(s.gamma, s.q_train, ms=4, lw=1.6, **style[lab])
        ax[1].plot(s.gamma, s.switches, ms=4, lw=1.6, **style[lab])
    ax[0].set_xlabel(r"switching price $\gamma=\lambda/\bar Q$")
    ax[0].set_ylabel(r"$\Sigma_t\, Q_t(A_t)/\Sigma_t\, Q_t(E_t)$")
    ax[0].set_title(r"(a) objective, $b=10\%$")
    ax[1].set_xlabel(r"switching price $\gamma=\lambda/\bar Q$")
    ax[1].set_ylabel(r"panel changes $\Sigma_t\,|A_{t-1}\triangle A_t|$")
    ax[1].set_title(r"(b) panel changes, $b=10\%$")
    h, l = ax[0].get_legend_handles_labels()
    if cert is not None:
        c = cert[cert.gamma == 0.2]
        data = [c.r_alpha.values, c.r_dd.values, c.r_lag.values]
        bp = ax[2].boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                           medianprops=dict(color="black"))
        for patch, col in zip(bp["boxes"], ("#bdbdbd", "#9ecae1", "#3182bd")):
            patch.set_facecolor(col)
        ax[2].set_xticks([1, 2, 3])
        ax[2].set_xticklabels([r"$\alpha^{-1}\Sigma_t\, Q_t$", r"$\Sigma_t\, U_t$",
                               r"$\min_{\mu}\,\Lambda(\mu)$"])
        ax[2].axhline(1, color="k", lw=0.8)
        ax[2].set_ylabel(r"$\mathrm{UB}/\mathrm{OPT}$")
        ax[2].set_title(r"(c) certificates, $\gamma=0.2$")
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.37, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_lambda_cert.{ext}"))
    plt.close(fig)


def cert_results():
    p = os.path.join(RES, "certificates.pkl")
    if not os.path.exists(p):
        return None
    c = pd.DataFrame(pickle.load(open(p, "rb")))
    for gm, gn in ((0.2, "Knee"), (1.0, "One")):
        s = c[c.gamma == gm]
        for col, nm in (("r_alpha", "Alpha"), ("r_dd", "Dd"), ("r_lag", "Lag")):
            mac(f"Cert{nm}{gn}", s[col].mean(), ".3f")
        for col, nm in (("cert_alpha", "Alpha"), ("cert_dd", "Dd"), ("cert_lag", "Lag"),
                        ("true_ratio", "True")):
            mac(f"CertR{nm}{gn}", s[col].mean(), ".3f")
        mac(f"CertLagTighter{gn}", 100 * np.mean(s.r_lag <= s.r_dd + 1e-12), ".0f")
    mac("CertN", len(c[c.gamma == 0.2]))
    mac("CertNRep", c.rep.nunique())
    return c


# ---------------------------------------------------------------------------
def headroom_results():
    p = os.path.join(RES, "headroom.pkl")
    if not os.path.exists(p):
        return None
    h = pd.DataFrame(pickle.load(open(p, "rb")))
    g = h.groupby("budget_frac").gain_pct
    m = g.mean()
    mac("HeadroomMax", m.max(), ".1f")
    mac("HeadroomMaxAt", 100 * m.idxmax(), ".1f")
    mac("HeadroomMean", m[(m.index >= 0.05) & (m.index <= 0.30)].mean(), ".1f")
    mac("HeadroomFull", m.loc[1.0], ".1f")
    mac("HeadroomNRep", h.rep.nunique())
    lo = m[(m.index >= 0.30) & (m.index <= 0.60)]
    mac("HeadroomMidMax", lo.max(), ".1f")
    return h


def fig_headroom_gap(h, win):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    if h is not None:
        g = h.groupby("budget_frac")
        m = g.gain_pct.mean()
        e = g.gain_pct.apply(lambda x: tci(x.values)[1])
        ax[0].errorbar(100 * m.index, m.values, yerr=e.values, color="#0072b2",
                       marker="o", ms=3.5, capsize=2)
        budget_axis(ax[0])
        ax[0].set_xlabel(r"per-window budget $b$ (% of panel)")
        ax[0].set_ylabel(r"gain in $\Sigma_t\, Q_t$ (%)")
        ax[0].set_title(r"(a) headroom, mean and 95% CI")
    cols = [("r_blind", "cost-blind"), ("r_guarded", "Alg. 1"), ("r_ga", "GA"),
            ("r_enum1", r"Alg. 2, $\ell=1$")]
    ax[1].boxplot([win[c].values for c, _ in cols], widths=0.5, whis=(0, 100), showfliers=False,
                  showmeans=True,
                  medianprops=dict(color="#0072b2", lw=1.6),
                  meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="k", ms=4.5))
    ax[1].set_xticks(range(1, len(cols) + 1))
    ax[1].set_xticklabels([l for _, l in cols])
    ax[1].axhline(0.427, color="#d55e00", ls=":", lw=1.2)
    ax[1].text(4.45, 0.427, r"guarantee of Alg. 1: $0.427$", fontsize=7.5, va="bottom", ha="right",
               color="#d55e00")
    from matplotlib.lines import Line2D
    ax[1].legend(handles=[Line2D([0], [0], color="#0072b2", lw=1.6, label="median"),
                          Line2D([0], [0], marker="D", ls="none", markerfacecolor="white",
                                 markeredgecolor="k", ms=4.5, label="mean")],
                 loc="lower center", bbox_to_anchor=(0.62, 0.0), ncol=2, fontsize=8, frameon=False)
    ax[1].set_ylabel(r"$Q_t(\hat A_t)/Q_t(A_t^\star)$")
    ax[1].set_title(r"(b) attained fraction of $Q_t(A_t^\star)$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_headroom_gap.{ext}"))
    plt.close(fig)


def sat_results():
    p = os.path.join(RES, "sat.pkl")
    if not os.path.exists(p):
        return
    s = pd.DataFrame(pickle.load(open(p, "rb")))
    g = s.groupby(["k", "size"])[["mean", "spread"]].mean()
    mac("SatOneFive", g.loc[(1, 5), "mean"], ".3f")
    mac("SatThreeFive", g.loc[(3, 5), "mean"], ".3f")
    mac("SatOneFiveRel", g.loc[(1, 5), "spread"] / g.loc[(1, 5), "mean"], ".3f")
    mac("SatThreeFiveRel", g.loc[(3, 5), "spread"] / g.loc[(3, 5), "mean"], ".3f")
    mac("NSatRep", s.rep.nunique())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    for k in sorted(s.k.unique()):
        gg = g.loc[k]
        ax[0].plot(gg.index, gg["mean"], marker="o", ms=3, label=f"$k={k}$")
        ax[1].plot(gg.index, gg["spread"], marker="o", ms=3, label=f"$k={k}$")
    ax[0].set_xlabel(r"$|A|$"); ax[0].set_ylabel(r"$Q_t(A)/Q_t(E)$")
    ax[1].set_xlabel(r"$|A|$"); ax[1].set_ylabel(r"spread of $Q_t(A)/Q_t(E)$")
    ax[0].set_title(r"(a) attained fraction, $\varphi=\min(\cdot,k)$"); ax[1].set_title("(b) discriminating spread")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_saturation.{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def paired_table(df, tagcol, tags, ref, meths, value="auroc"):
    out = {}
    for tg in tags:
        s = df[df[tagcol] == tg]
        w = s.groupby(["method", "rep"])[value].mean().unstack(0)
        for mth in meths:
            if mth not in w.columns or ref not in w.columns:
                continue
            d = (w[mth] - w[ref]).values
            out[(tg, mth)] = (*tci(d), wilcox(d))
    return out


def price_results():
    p = os.path.join(RES, "price.pkl")
    if not os.path.exists(p):
        return None
    df = pd.DataFrame(pickle.load(open(p, "rb")))
    tags = ["flat", "clfs", "spread", "dominant"]
    names = {"flat": "Flat", "clfs": "Clfs", "spread": "Spread", "dominant": "Dom"}
    # cost-blind vs RA-DSS myopic: AUROC difference, Q ratios
    pa = paired_table(df, "tag", tags, "RA-DSS myopic", ["Cost-blind greedy", "Static reduct",
                                                         "RA-DSS DP-exact", "NB-VOI greedy"])
    pq = paired_table(df, "tag", tags, "RA-DSS myopic", ["Cost-blind greedy"], value="q_test")
    rows = []
    for tg in tags:
        s = df[df.tag == tg]
        w = s.groupby(["method", "rep"]).q_test.mean().unstack(0)
        ratio = (w["Cost-blind greedy"] / w["RA-DSS myopic"]).values
        rm, rh = tci(ratio)
        mac(f"PriceRatio{names[tg]}", rm, ".3f")
        tight = s[s.budget_frac == 0.05].groupby(["method", "rep"]).q_test.mean().unstack(0)
        mac(f"PriceRatioTight{names[tg]}",
            float(np.mean(tight["Cost-blind greedy"] / tight["RA-DSS myopic"])), ".3f")
        m, h, pv = pa[(tg, "Cost-blind greedy")]
        mac(f"PriceDBlind{names[tg]}", f"{m:+.4f}"); mac(f"PriceDBlind{names[tg]}ci", h, ".4f")
        mac(f"PricePBlind{names[tg]}", fmt_pm(pv))
        cells = []
        for mth in ("Cost-blind greedy", "Static reduct", "RA-DSS DP-exact", "NB-VOI greedy"):
            m, h, pv = pa[(tg, mth)]
            cells.append(f"${m:+.4f}\\pm{h:.4f}$")
        rows.append(f"{tg if tg != 'clfs' else 'CLFS'} & " + " & ".join(cells) +
                    f" & ${rm:.3f}\\pm{rh:.3f}$ \\\\")
    return df, rows


def corr_results():
    p = os.path.join(RES, "corr.pkl")
    if not os.path.exists(p):
        return None
    df = pd.DataFrame(pickle.load(open(p, "rb")))
    rows = []
    pa = paired_table(df, "tag", ["independent", "correlated"], "Static reduct",
                      ["Cost-blind greedy", "RA-DSS myopic", "RA-DSS DP-exact", "NB-VOI greedy"])
    for tg, nm in (("independent", "Indep"), ("correlated", "Corr")):
        cells = []
        for mth in ("Cost-blind greedy", "RA-DSS myopic", "RA-DSS DP-exact", "NB-VOI greedy"):
            m, h, pv = pa[(tg, mth)]
            cells.append(f"${m:+.4f}\\pm{h:.4f}$")
        rows.append((tg, cells))
        s = df[(df.tag == tg) & (df.method == "Cost-blind greedy")]
        mac(f"Nested{nm}", s.nested_both.mean(), ".2f")
        s2 = df[(df.tag == tg) & (df.method == "RA-DSS myopic")]
        mac(f"NestedMyo{nm}", s2.nested_both.mean(), ".2f")
        m, h, pv = pa[(tg, "RA-DSS DP-exact")]
        mac(f"CorrDDpx{nm}", f"{m:+.4f}"); mac(f"CorrDDpx{nm}ci", h, ".4f")
        m, h, pv = pa[(tg, "NB-VOI greedy")]
        mac(f"CorrDVoi{nm}", f"{m:+.4f}"); mac(f"CorrDVoi{nm}ci", h, ".4f")
    return df, rows


def write_sens_table(price_rows, corr_rows):
    lines = []
    if price_rows:
        lines.append("\\multicolumn{6}{l}{\\emph{price structure; $\\Delta$AUROC relative to RA-DSS myopic}}\\\\")
        lines += price_rows
    body = ("\\begin{tabular}{lccccc}\n\\toprule\n"
            "& cost-blind & static reduct & DP-exact & NB-VOI & $Q_{\\mathrm{blind}}/Q_{\\mathrm{myopic}}$ \\\\\n"
            "\\midrule\n" + "\n".join(lines))
    if corr_rows:
        body += ("\n\\midrule\n\\multicolumn{6}{l}{\\emph{assay dependence; $\\Delta$AUROC "
                 "relative to static reduct}}\\\\\n"
                 "& cost-blind & myopic & DP-exact & NB-VOI & \\\\\n")
        for tg, cells in corr_rows:
            body += f"{tg} & " + " & ".join(cells) + " & \\\\\n"
    body += "\\bottomrule\n\\end{tabular}"
    open(os.path.join(TAB, "tab_sensitivity.tex"), "w").write(body)


OFAT_TEX = {"base": "baseline ($k=3$, $\\sigma_{\\mathrm{out}}=2.8$, $|T|=6$)",
            "k=1": "$k=1$", "k=2": "$k=2$", "k=4": "$k=4$", "k=5": "$k=5$",
            "sigma_out=1.5": "$\\sigma_{\\mathrm{out}}=1.5$", "sigma_out=4.0": "$\\sigma_{\\mathrm{out}}=4.0$",
            "subsample=200/400": "subsample $(200,400)$", "subsample=all": "subsample: all objects",
            "windows=4": "$|T|=4$", "windows=8": "$|T|=8$",
            "sharpness=0": "shift sharpness $0$", "sharpness=0.5": "shift sharpness $0.5$"}


def sens_results():
    p = os.path.join(RES, "sens.pkl")
    if not os.path.exists(p):
        return None
    df = pd.DataFrame(pickle.load(open(p, "rb")))
    tags = list(dict.fromkeys(df.tag))
    pa = paired_table(df, "tag", tags, "Static reduct",
                      ["RA-DSS myopic", "RA-DSS DP-exact", "NB-VOI greedy", "Cost-blind greedy"])
    pq = paired_table(df, "tag", tags, "Static reduct",
                      ["RA-DSS myopic", "RA-DSS DP-exact", "NB-VOI greedy"], value="q_test")
    lines = []
    for tg in tags:
        cells = []
        for mth in ("RA-DSS myopic", "RA-DSS DP-exact", "NB-VOI greedy", "Cost-blind greedy"):
            m, h, pv = pa[(tg, mth)]
            cells.append(f"${m:+.4f}\\pm{h:.4f}$")
        m, h, _ = pq[(tg, "RA-DSS DP-exact")]
        cells.append(f"${m:+.3f}\\pm{h:.3f}$")
        fa = df[(df.tag == tg) & (df.method == "RA-DSS DP-exact")].auroc.mean()
        cells.append(f"{fa:.3f}")
        lines.append(OFAT_TEX.get(tg, tg) +
                     " & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
            "& \\multicolumn{4}{c}{$\\Delta$AUROC vs static reduct} & $\\Delta Q$ held-out & AUROC \\\\\n"
            "\\cmidrule(lr){2-5}\n"
            "setting & myopic & DP-exact & NB-VOI & cost-blind & DP-exact & DP-exact \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    open(os.path.join(TAB, "tab_ofat.tex"), "w").write(body)
    mac("SensNRep", df.rep.nunique())
    pos = sum(1 for tg in tags if pa[(tg, "RA-DSS DP-exact")][0] > 0)
    mac("SensDpxPosCount", pos); mac("SensNConfig", len(tags))
    sig = sum(1 for tg in tags if pa[(tg, "RA-DSS DP-exact")][0] - pa[(tg, "RA-DSS DP-exact")][1] > 0)
    mac("SensDpxSigCount", sig)
    vb = sum(1 for tg in tags if pa[(tg, "NB-VOI greedy")][0] > pa[(tg, "RA-DSS DP-exact")][0])
    mac("SensVoiBeatsDpx", vb)
    qpos = sum(1 for tg in tags if pq[(tg, "RA-DSS DP-exact")][0] - pq[(tg, "RA-DSS DP-exact")][1] > 0)
    mac("SensDpxQSigCount", qpos)
    return df, pa, pq


def stat_results():
    p = os.path.join(RES, "stat.pkl")
    if not os.path.exists(p):
        return None
    raw = pickle.load(open(p, "rb"))
    viol = [r for r in raw if r.get("method") == "__violations__"]
    df = pd.DataFrame([r for r in raw if r.get("method") != "__violations__"])
    poc = [CS.KEYS.index(k) for k in ("ABG", "LAC")]
    routine = (0, 3)
    lines = []
    for mult in sorted(df.mult.unique()):
        s = df[df.mult == mult]
        for bf in sorted(s.budget_frac.unique()):
            ss = s[s.budget_frac == bf]
            w = ss.groupby(["method", "rep"])[["auroc", "q_test", "viol"]].mean()
            ref = w.xs("RA-DSS exact/window", level="method")
            cells = []
            for mth in ("Time-avg price + repair", "Static reduct"):
                x = w.xs(mth, level="method")
                m, h = tci((x.q_test - ref.q_test).values)
                cells.append(f"${m:+.3f}\\pm{h:.3f}$")
                m, h = tci((x.auroc - ref.auroc).values)
                cells.append(f"${m:+.4f}\\pm{h:.4f}$")
            vs = 100 * np.mean(ss[ss.method == "Static reduct"].viol > 0)
            ex = ss[ss.method == "RA-DSS exact/window"]
            nr = ns = pr_ = ps = 0
            for masks in ex.masks:
                for wdx, m in enumerate(masks):
                    k = sum((m >> j) & 1 for j in range(CS.N_PAR))
                    kp = sum((m >> j) & 1 for j in poc)
                    if wdx in routine:
                        nr += k; pr_ += kp
                    else:
                        ns += k; ps += kp
            lines.append(f"{mult:.1f} & {100 * bf:.0f}\\% & {100 * pr_ / max(nr, 1):.0f}\\% & "
                         f"{100 * ps / max(ns, 1):.0f}\\% & " + " & ".join(cells) + f" & {vs:.0f}\\% \\\\")
    body = ("\\begin{tabular}{llccccccc}\n\\toprule\n"
            "& & \\multicolumn{2}{c}{POC share of selection} & \\multicolumn{2}{c}{time-avg + repair} & "
            "\\multicolumn{2}{c}{static reduct} & static \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "$m$ & budget & routine & STAT & $\\Delta Q$ & $\\Delta$AUROC & $\\Delta Q$ & $\\Delta$AUROC & over budget \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    open(os.path.join(TAB, "tab_stat.tex"), "w").write(body)
    s = df[df.mult == 2.0]
    w = s.groupby(["method", "rep"])[["auroc", "q_test"]].mean()
    ref = w.xs("RA-DSS exact/window", level="method")
    for mth, nm in (("Time-avg price + repair", "Avg"), ("Static reduct", "Static")):
        x = w.xs(mth, level="method")
        d = (ref.q_test - x.q_test).values
        m, h = tci(d)
        mac(f"StatDQ{nm}", f"{m:+.3f}"); mac(f"StatDQ{nm}ci", h, ".3f"); mac(f"StatDQ{nm}p", fmt_pm(wilcox(d)))
        d = (ref.auroc - x.auroc).values
        m, h = tci(d)
        mac(f"StatDA{nm}", f"{m:+.4f}"); mac(f"StatDA{nm}ci", h, ".4f"); mac(f"StatDA{nm}p", fmt_pm(wilcox(d)))
    vv = [v for r in viol if r["mult"] == 2.0 for v in r["violations"]]
    mac("StatAvgViolPct", 100 * np.mean(np.array(vv) > 0), ".0f")
    sv = df[(df.mult == 2.0) & (df.method == "Static reduct")].viol
    mac("StatStaticViolFrac", 100 * np.mean(sv > 0), ".0f")
    mac("StatNRep", df.rep.nunique())
    return df


def inc_results():
    p = os.path.join(RES, "inc.pkl")
    if not os.path.exists(p):
        return None
    df = pd.DataFrame(pickle.load(open(p, "rb")))
    mac("IncAgree", 100 * df.agree.mean(), ".1f")
    mac("IncEvalsInc", df.evals_inc.mean(), ".1f")
    mac("IncEvalsFull", df.evals_full.mean(), ".1f")
    mac("IncSaving", 100 * (1 - df.evals_inc.mean() / df.evals_full.mean()), ".1f")
    mac("IncAgreeTrue", 100 * df.agree_true.mean(), ".1f")
    mac("IncNRep", df.rep.nunique())
    return df


def scaling_results():
    p = os.path.join(RES, "scaling.pkl")
    if not os.path.exists(p):
        return None
    df = pd.DataFrame(pickle.load(open(p, "rb")))
    sc = df[df.kind == "scratch"].groupby("n")[["t_plain", "t_lazy", "ev_plain", "ev_lazy", "same", "k_sel"]].mean()
    wm = df[df.kind != "scratch"].groupby(["kind", "n", "delta"])[["t_warm", "t_plain", "t_lazy", "ev_warm", "ev_plain", "ev_lazy", "same"]].mean()
    lines = []
    for n in sc.index:
        r = sc.loc[n]
        lines.append(f"{n} & scratch & -- & {r.t_plain:.2f} & {r.t_lazy:.2f} & -- & "
                     f"{r.ev_plain:,.0f} & {r.ev_lazy:,.0f} & -- \\\\")
        for kind in ("warm-unselected", "warm-random"):
            for dlt in (1, 20):
                if (kind, n, dlt) not in wm.index:
                    continue
                q = wm.loc[(kind, n, dlt)]
                lines.append(f"{n} & {kind.split('-')[1]} & {dlt} & {q.t_plain:.2f} & {q.t_lazy:.2f} & "
                             f"{q.t_warm:.2f} & {q.ev_plain:,.0f} & {q.ev_lazy:,.0f} & {q.ev_warm:,.0f} \\\\")
    body = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
            "& & & \\multicolumn{3}{c}{wall-clock (s)} & \\multicolumn{3}{c}{marginal evaluations} \\\\\n"
            "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
            "$|E|$ & change & $|\\Delta|$ & greedy & lazy & warm & greedy & lazy & warm \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    open(os.path.join(TAB, "tab_scaling.tex"), "w").write(body)
    mac("ScaleSameAll", int(df.same.min()))
    r = sc.loc[sc.index.max()]
    mac("ScaleNmax", int(sc.index.max()))
    mac("ScalePlainT", r.t_plain, ".2f"); mac("ScaleLazyT", r.t_lazy, ".2f")
    q = wm.loc[("warm-unselected", sc.index.max(), 1)]
    mac("ScaleWarmT", q.t_warm, ".3f"); mac("ScaleWarmEv", q.ev_warm, ".0f")
    mac("ScaleWarmPlainEv", q.ev_plain, ".0f")
    q = wm.loc[("warm-random", sc.index.max(), 20)]
    mac("ScaleWarmRandT", q.t_warm, ".2f")
    mac("ScaleWarmRandLazyT", q.t_lazy, ".2f")
    return df


def tab_parameters():
    windows = {"PCT": "$t_0,t_3$", "BNP": "$t_0,t_2,t_4$", "CULT": "$t_2$--$t_5$"}
    lines = []
    for t in CS.TESTS:
        lines.append(f"{t['key']} & {t['cpt']} & {t['name']} & {t['cost']:.2f} & "
                     f"{t['phase']} & {t['sal']:.2f} & {windows.get(t['key'], 'all')} \\\\")
    body = ("\\begin{tabular}{lllrlll}\n\\toprule\n"
            "key & CPT & assay & cost (\\$) & phase & $s_e$ & windows \\\\\n"
            " & OBS & OBS & OBS & ASM & ASM & ASM \\\\\n\\midrule\n" + "\n".join(lines) +
            f"\n\\midrule\n\\multicolumn{{3}}{{l}}{{full panel, one window}} & {CS.FULL_PANEL_COST:.2f} & & & \\\\\n"
            f"\\multicolumn{{3}}{{l}}{{full panel, 24\\,h (if all were available)}} & {6 * CS.FULL_PANEL_COST:,.2f} & & & \\\\\n"
            "\\bottomrule\n\\end{tabular}")
    open(os.path.join(TAB, "tab_parameters.tex"), "w").write(body)


def verification_table():
    old = json.load(open(os.path.join(ROOT, "results", "verification.json")))
    new_p = os.path.join(RES, "verification2.json")
    new = json.load(open(new_p)) if os.path.exists(new_p) else {}
    V4 = old["V4_greedy_ratio"]
    g, e2, e3 = (V4["stats"]["guarded greedy"], V4["stats"]["partial-enum (l=2)"],
                 V4["stats"]["partial-enum (l=3)"])
    rows = [
        ("Lemma~\\ref{lem:indep}(ii)", "power set $\\iff c_t(D_t)\\le B_t$", old["V1_non_collapse"]["trials"], old["V1_non_collapse"]["violations"]),
        ("Proposition~\\ref{prop:hierarchy}", "embedding of dynamic soft sets", old["V8_embedding"]["trials"], old["V8_embedding"]["failures"]),
        ("Theorem~\\ref{thm:submod}", "normalised, monotone, submodular", old["V2_submodularity"]["trials"], old["V2_submodularity"]["sub"] + old["V2_submodularity"]["mono"] + old["V2_submodularity"]["norm"]),
        ("Proposition~\\ref{prop:modular}(ii)", "modular $\\iff$ $w$-disjoint", old["V2_submodularity"]["trials"], old["V2_submodularity"]["modularity_iff"]),
        ("Proposition~\\ref{prop:graded}", "graded: normalised, monotone, submodular", new.get("V17_instances", "--"),
         (new["V17_norm_violations"] + new["V17_mono_violations"] + new["V17_submod_violations"] + new["V17_crisp_mismatch"]) if "V17_instances" in new else "--"),
        ("Proposition~\\ref{prop:graded}", (f"graded: Alg.~1 $\\ge0.427$, Alg.~2 $\\ge1-1/e$ (min.\\ {new['V17_alg1_min_ratio']:.3f}, {new['V17_alg2_min_ratio']:.3f})"
                                            if "V17_instances" in new else "graded: approximation guarantees"),
         new.get("V17_greedy_instances", "--"),
         (new["V17_alg1_violations"] + new["V17_alg2_violations"]) if "V17_instances" in new else "--"),
        ("Proposition~\\ref{prop:sandwich}", "sandwich of the truncated margin", new.get("V14_trials", "--"), new.get("V14_violations", "--")),
        ("Theorem~\\ref{thm:nphard}", "maximum-coverage embedding", old["V3_reduction"]["trials"], old["V3_reduction"]["mismatches"]),
        ("Theorem~\\ref{thm:greedy}(i)", f"Alg.~1 $\\ge0.427$ (min.\\ {g['min']:.3f})", V4["n"], int(g["min"] < 0.427)),
        ("Theorem~\\ref{thm:greedy}(ii)", f"Alg.~2, $\\ell=2$, $\\ge1-1/e$ (min.\\ {e2['min']:.3f})", V4["n"], int(e2["min"] < 1 - 1 / math.e)),
        ("Theorem~\\ref{thm:greedy}(ii)", f"Alg.~2, $\\ell=3$, $\\ge1-1/e$ (min.\\ {e3['min']:.3f})", V4["n"], int(e3["min"] < 1 - 1 / math.e)),
        ("Proposition~\\ref{prop:dd}", "data-dependent bound $\\ge$ optimum", new.get("V11_trials", "--"), new.get("V15_violations", "--")),
        ("Theorem~\\ref{thm:structure}(i)", "violation of submodularity found", old["V5_switch_not_submodular"]["random_trials"], f"found in {old['V5_switch_not_submodular']['random_violations']}"),
        ("Theorem~\\ref{thm:structure}(ii),(iii)", "$-J$ submodular; chains exact (modular $Q_t$)", new.get("V13_trials", "--"), (new.get("V13_submod_violations", 0) + new.get("V13_chain_mismatch", 0)) if new else "--"),
        ("Theorem~\\ref{thm:exactdp}", "distance-transform DP $=$ exhaustive search", new.get("V10_trials", "--"), new.get("V10_mismatch", "--")),
        ("Section~\\ref{sec:exactdp}", "pattern-transform lattice $=$ direct $Q$", 6 * 400, new.get("V16_lattice_mismatches", "--")),
        ("Theorem~\\ref{thm:certs}(iii)", "Lagrangian bound $\\ge$ optimum", new.get("V11_trials", "--"), new.get("V11_violations", "--")),
        ("Proposition~\\ref{prop:depleting}", "depleting-budget DP $=$ exhaustive search", new.get("V12_trials", "--"), new.get("V12_mismatch", "--")),
        ("Theorem~\\ref{thm:incremental}", "warm start reproduces greedy", old["V7_incremental"]["trials"], old["V7_incremental"]["mismatches"]),
    ]
    lines = [f"{a} & {b} & {tex_group(c)} & {tex_group(d)} \\\\" for a, b, c, d in rows]
    body = ("\\begin{tabular}{llrr}\n\\toprule\nresult & property checked & instances & violations \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    open(os.path.join(TAB, "tab_verification.tex"), "w").write(body)
    mac("VEngMasks", 6 * 400)
    mac("VEngAgree", new.get("V16_greedy_agree", "--"))
    mac("VEngCases", new.get("V16_cases", "--"))
    mac("VGradedN", new.get("V17_instances", "--"))
    mac("VGradedGreedyN", new.get("V17_greedy_instances", "--"))
    mac("VGradedCtrlN", new.get("V17_control_convex_trials", "--"))
    mac("VGradedCtrlFail", new.get("V17_control_convex_failed", "--"))
    mac("VLagTight", new.get("V11_ratio_mean", float("nan")), ".3f")
    mac("VSumUTight", new.get("V11_sumU_ratio_mean", float("nan")), ".3f")
    mac("VDPMean", old["V6_multiperiod"]["dp_ratio_mean"], ".4f")
    mac("VDPMin", old["V6_multiperiod"]["dp_ratio_min"], ".4f")
    mac("VGuardMin", g["min"], ".3f")
    mac("VEnumTwoMin", e2["min"], ".3f")


def paired_all(D):
    lines = []
    for meth in ORDER:
        if meth == REF:
            continue
        s = D[D.method == meth].sort_values("budget_frac")
        s = s[~s.d_auc.isna()]
        for i, r in enumerate(s.itertuples()):
            lab = LABEL[meth] if i == 0 else ""
            lines.append(f"{lab} & {100 * r.budget_frac:g}\\% & ${r.d_auc:+.4f}\\pm{r.d_auc_ci:.4f}$ & "
                         f"{fmt_p(r.p_auc)} & ${r.d_q:+.3f}\\pm{r.d_q_ci:.3f}$ & {fmt_p(r.p_q)} \\\\")
        lines.append("\\midrule")
    body = ("\\begin{tabular}{llcccc}\n\\toprule\npolicy & budget & $\\Delta$AUROC & $p$ & "
            "$\\Delta Q$ held-out & $p$ \\\\\n\\midrule\n" + "\n".join(lines[:-1]) +
            "\n\\bottomrule\n\\end{tabular}")
    open(os.path.join(TAB, "tab_paired_all.tex"), "w").write(body)
    open(os.path.join(TAB, "tab_paired_all_rows.tex"), "w").write("\n".join(lines[:-1]) + "\n")



# ---------------------------------------------------------------------------
# additional figures (Sections 5, 6 and 7)
# ---------------------------------------------------------------------------
def fig_multiperiod_schematic():
    """Section 5: the multi-period problem as a layered graph and the
    coordinate-wise distance transform of Theorem 5.3 (illustrative)."""
    from matplotlib.patches import FancyArrowPatch
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3),
                           gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    a.set_xlim(-0.6, 3.6); a.set_ylim(-0.4, 3.5); a.axis("off")
    cols = {1: 0.8, 2: 1.9, 3: 3.0}
    rows = {0: 2.4, 1: 1.4, 2: 0.4}
    feas = {(1, 0): True, (1, 1): True, (1, 2): False,
            (2, 0): True, (2, 1): True, (2, 2): True,
            (3, 0): False, (3, 1): True, (3, 2): True}
    best = [(1, 0), (2, 1), (3, 1)]
    a.scatter([-0.3], [1.4], s=260, color="white", edgecolor="black", zorder=3)
    a.text(-0.3, 1.4, r"$A_0$", ha="center", va="center", fontsize=10, zorder=4)
    for (t, r), ok in feas.items():
        x, y = cols[t], rows[r]
        a.scatter([x], [y], s=900, zorder=3,
                  color=("#dbe9f6" if ok else "#f0f0f0"),
                  edgecolor=("#08519c" if (t, r) in best else ("black" if ok else "#bdbdbd")),
                  linewidth=(2.2 if (t, r) in best else 1.0))
        a.text(x, y, r"$A^{(%d)}_{%d}$" % (r + 1, t), ha="center", va="center",
               fontsize=10, color=("black" if ok else "#9e9e9e"), zorder=4)
        if not ok:
            a.text(x, y - 0.38, r"$c_{%d}(A)>B_{%d}$" % (t, t), ha="center",
                   fontsize=8, color="#9e9e9e")
    for r in rows:
        if feas[(1, r)]:
            a.plot([-0.3, cols[1]], [1.4, rows[r]], color="#bdbdbd", lw=0.8, zorder=1)
    for t in (1, 2):
        for r1 in rows:
            for r2 in rows:
                if feas[(t, r1)] and feas[(t + 1, r2)]:
                    on = (t, r1) in best and (t + 1, r2) in best
                    a.plot([cols[t], cols[t + 1]], [rows[r1], rows[r2]],
                           color=("#08519c" if on else "#bdbdbd"),
                           lw=(2.2 if on else 0.8), zorder=1)
    a.plot([-0.3, cols[1]], [1.4, rows[0]], color="#08519c", lw=2.2, zorder=1)
    for t in (1, 2, 3):
        a.text(cols[t], 3.15, r"period $t=%d$" % t, ha="center", fontsize=10)
        a.text(cols[t], 2.85, r"$\mathcal{A}_{%d}$" % t, ha="center", fontsize=11)
    a.text(1.35, -0.2, r"each edge carries weight $-\lambda\,\kappa(A_{t-1},A_t)$; grey nodes are infeasible",
           ha="center", fontsize=9, color="#525252")
    a.set_title(r"(a) policies as paths: $V_t(A)=Q_t(A)+\max_{A'}\{V_{t-1}(A')-\lambda\kappa(A',A)\}$",
                fontsize=10)
    b = ax[1]
    b.axis("off"); b.set_xlim(-0.6, 2.4); b.set_ylim(-0.3, 2.2)
    # cube {0,1}^3 in oblique projection
    P = {}
    for m in range(8):
        x = (m & 1) * 1.2 + ((m >> 2) & 1) * 0.55
        y = ((m >> 1) & 1) * 1.2 + ((m >> 2) & 1) * 0.45
        P[m] = (x, y)
    names = {0: r"$\varnothing$", 1: r"$\{e_1\}$", 2: r"$\{e_2\}$", 3: r"$\{e_1,e_2\}$",
             4: r"$\{e_3\}$", 5: r"$\{e_1,e_3\}$", 6: r"$\{e_2,e_3\}$", 7: r"$E$"}
    colr = {0: "#d55e00", 1: "#009e73", 2: "#0072b2"}
    for m in range(8):
        for i in range(3):
            if not (m >> i) & 1:
                n2 = m | (1 << i)
                b.add_patch(FancyArrowPatch(P[m], P[n2], arrowstyle="<->", mutation_scale=9,
                                            color=colr[i], lw=1.4, shrinkA=5, shrinkB=5))
    for m in range(8):
        b.scatter(*P[m], s=60, color="black", zorder=3)
        dx = -0.09 if (m & 1) == 0 else 0.09
        b.text(P[m][0] + dx, P[m][1] + 0.1, names[m], ha=("right" if dx < 0 else "left"),
               va="bottom", fontsize=9, zorder=4,
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    for i in range(3):
        b.plot([], [], color=colr[i], lw=1.6,
               label=r"pass $i=%d$: flip $e_%d$ at cost $\lambda\kappa^{\pm}(e_%d)$" % (i + 1, i + 1, i + 1))
    b.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, frameon=False)
    b.set_title(r"(b) distance transform: $n$ relaxations over $\{0,1\}^n$", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_multiperiod.png"))
    fig.savefig(os.path.join(FIG, "fig_multiperiod.pdf"))
    plt.close(fig)


def fig_design():
    """Section 6: prices, availability and the time-varying loadings."""
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.0),
                           gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    order = sorted(range(CS.N_PAR), key=lambda j: CS.COST[j])
    cols = ["#0072b2" if CS.TESTS[j]["phase"] == "early" else "#d55e00" for j in order]
    ax[0].barh(range(CS.N_PAR), [CS.COST[j] for j in order], color=cols)
    ax[0].set_yticks(range(CS.N_PAR))
    ax[0].set_yticklabels([CS.KEYS[j] for j in order])
    ax[0].invert_yaxis()
    ax[0].set_xlabel(r"fee-schedule price $c(e)$ (US\$)")
    ax[0].set_title(r"(a) acquisition cost by assay")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color="#0072b2", label="early phase"),
                          Patch(color="#d55e00", label="late phase")],
                 loc="center right", fontsize=9)
    Av = CS.availability_matrix()
    M = Av[:, order].T.astype(float)
    from matplotlib.colors import ListedColormap
    ax[1].imshow(M, cmap=ListedColormap(["#fbe3d4", "#dbe9f6"]), vmin=0, vmax=1, aspect="auto")
    ax[1].set_yticks(range(CS.N_PAR)); ax[1].set_yticklabels([CS.KEYS[j] for j in order])
    ax[1].set_xticks(range(CS.N_WIN)); ax[1].set_xticklabels([f"$t_{w}$" for w in range(CS.N_WIN)])
    ax[1].set_xlabel(r"window $t$"); ax[1].grid(False)
    for (r, c), v in np.ndenumerate(M):
        if v == 0:
            ax[1].text(c, r, r"$\infty$", ha="center", va="center", fontsize=9, color="#d55e00")
    ax[1].set_title(r"(b) availability; $\infty$ marks $c_t(e)=\infty$")
    beta = CS.loadings()
    w = np.arange(CS.N_WIN)
    for j in range(CS.N_PAR):
        ph = CS.TESTS[j]["phase"]
        ax[2].plot(w, beta[:, j], color=("#0072b2" if ph == "early" else "#d55e00"),
                   lw=1.0, alpha=0.8)
    ax[2].set_xticks(w); ax[2].set_xticklabels([f"$t_{i}$" for i in w])
    ax[2].set_xlabel(r"window $t$"); ax[2].set_ylabel(r"loading $\beta_{e,t}=s_e\,\ell_t$")
    ax[2].set_title(r"(c) regime shift in the loadings")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_design.png"))
    fig.savefig(os.path.join(FIG, "fig_design.pdf"))
    plt.close(fig)


def fig_ofat(pa, tags):
    """Section 7: forest plot of the one-factor-at-a-time sensitivity."""
    meths = [("RA-DSS myopic", "RA-DSS myopic", "#0072b2", "o"),
             ("RA-DSS DP-exact", "RA-DSS DP-exact", "#3f007d", "s"),
             ("NB-VOI greedy", "NB-VOI greedy", "#009e73", "D")]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    n = len(tags)
    for k, (m, lab, col, mk) in enumerate(meths):
        ys = np.arange(n) + (k - 1) * 0.22
        mu = [pa[(tg, m)][0] for tg in tags]
        h = [pa[(tg, m)][1] for tg in tags]
        ax.errorbar(mu, ys, xerr=h, fmt=mk, color=col, ms=4.5, capsize=2, lw=1.2, label=lab)
    ax.axvline(0, color="#d55e00", lw=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([OFAT_FIG.get(t, t) for t in tags])
    ax.invert_yaxis()
    ax.set_xlabel(r"paired $\Delta$AUROC vs static reduct (mean and 95% CI)")
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], color=col, marker=mk, ms=5, lw=1.2, label=lab) for _, lab, col, mk in meths]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, frameon=False)
    ax.set_title(r"sensitivity to modelling choices, budgets from $7.5\%$ to $20\%$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ofat.png"))
    fig.savefig(os.path.join(FIG, "fig_ofat.pdf"))
    plt.close(fig)


OFAT_FIG = {"base": r"baseline ($k=3$, $\sigma_{\rm out}=2.8$, $|T|=6$)",
            "k=1": r"$k=1$", "k=2": r"$k=2$", "k=4": r"$k=4$", "k=5": r"$k=5$",
            "sigma_out=1.5": r"$\sigma_{\rm out}=1.5$", "sigma_out=4.0": r"$\sigma_{\rm out}=4.0$",
            "subsample=200/400": r"subsample $(200,400)$", "subsample=all": "subsample: all objects",
            "windows=4": r"$|T|=4$", "windows=8": r"$|T|=8$",
            "sharpness=0": r"shift sharpness $0$", "sharpness=0.5": r"shift sharpness $0.5$"}



# ---------------------------------------------------------------------------
# exact static panel and endpoint-weighted objective (experiment4.py)
# ---------------------------------------------------------------------------
def static_exact_results(M, pr, wide, opm, cost_at):
    if REF2 not in opm.columns:
        return None
    rows = []
    comps = [("RA-DSS DP-exact", "Dpx"), ("RA-DSS exact/window", "Exw"),
             ("RA-DSS myopic", "Myo"), ("NB-VOI greedy", "Voi"), ("GA", "Ga"),
             ("Static reduct", "Static"), ("RA-DSS DP-exact (w)", "Dpw")]
    for meth, nm in comps:
        dd = (opm[meth] - opm[REF2]).values
        md, hd = tci(dd)
        mac(f"VsSexOp{nm}", f"{md:+.4f}"); mac(f"VsSexOp{nm}ci", hd, ".4f")
        mac(f"VsSexOpP{nm}", fmt_pm(wilcox(dd)))
        for bf, bn in ((0.075, "Seven"), (0.10, "Ten"), (0.20, "Twenty")):
            A = wide["auroc"].loc[bf]
            d = (A[meth] - A[REF2]).values
            mm, hh = tci(d)
            mac(f"VsSex{nm}{bn}", f"{mm:+.4f}"); mac(f"VsSex{nm}{bn}ci", hh, ".4f")
            mac(f"VsSexP{nm}{bn}", fmt_pm(wilcox(d)))
            Q = wide["q_test"].loc[bf]
            dq = (Q[meth] - Q[REF2]).values
            mq, hq = tci(dq)
            mac(f"VsSexQ{nm}{bn}", f"{mq:+.3f}"); mac(f"VsSexQ{nm}{bn}ci", hq, ".3f")
        # budgets 5..30% at which the paired interval lies above / below zero
        pos = neg = 0
        for bf in sorted(wide["auroc"].index.get_level_values(0).unique()):
            if bf > 0.30 + 1e-9:
                continue
            A = wide["auroc"].loc[bf]
            mm, hh = tci((A[meth] - A[REF2]).values)
            pos += int(mm - hh > 0)
            neg += int(mm + hh < 0)
        mac(f"VsSexNPos{nm}", pos); mac(f"VsSexNNeg{nm}", neg)
        rows.append((meth, md, hd))
    # endpoint-weighted objective against the unweighted exact program
    dd = (opm["RA-DSS DP-exact (w)"] - opm["RA-DSS DP-exact"]).values
    md, hd = tci(dd)
    mac("WvsDpxOp", f"{md:+.4f}"); mac("WvsDpxOpci", hd, ".4f"); mac("WvsDpxOpP", fmt_pm(wilcox(dd)))
    for v, f in (("cost", ".1f"), ("switches", ".1f"), ("q_test", ".3f")):
        a = wide[v].loc[0.10]
        mm, hh = tci((a["RA-DSS DP-exact (w)"] - a["RA-DSS DP-exact"]).values)
        mac(f"WvsDpx{v.replace('_', '').capitalize()}Ten", f"{mm:+{f}}")
        mac(f"WvsDpx{v.replace('_', '').capitalize()}Tenci", hh, f)
    # paired cost gaps at 97% of full-panel AUROC (dollars per patient-day)
    for a, b, nm in (("Static reduct", "RA-DSS myopic", "StaticMyo"),
                     ("Static reduct", "RA-DSS DP-exact", "StaticDpx"),
                     ("Static exact", "RA-DSS DP-exact", "SexDpx"),
                     ("Static exact", "RA-DSS myopic", "SexMyo")):
        d = cost_at[(a, 0.97)] - cost_at[(b, 0.97)]
        mm, hh = tci(d)
        mac(f"CostGapNS{nm}", mm, ".0f"); mac(f"CostGapNS{nm}ci", hh, ".0f")
    Wt = M["weights"]
    for t in range(6):
        mac(f"WMean{'ABCDEF'[t]}", float(Wt[f"w{t}"].mean()), ".2f")   # t0..t5
    return rows


# ---------------------------------------------------------------------------
# multi-period methods at large n (scale_mp.py)
# ---------------------------------------------------------------------------
def scale_mp_results():
    p = os.path.join(RES, "scale_mp.pkl")
    if not os.path.exists(p):
        return None
    d = pd.DataFrame(pickle.load(open(p, "rb")))
    d["cert_rdp"] = d.J_restricted / d.UB
    d["cert_myo"] = d.J_myopic / d.UB
    d["ub_ratio"] = d.UB / d.J_restricted
    lines = []
    for n, g in d.groupby("n"):
        nm = {20: "Twenty", 50: "Fifty", 100: "Hundred"}[int(n)]
        ex = "J_exact" in g and g.J_exact.notna().all()
        cells = [f"{int(n)}", f"{len(g)}", f"{g.pool_size.mean():.0f}",
                 f"{g.cert_myo.mean():.3f} ({g.cert_myo.min():.3f})",
                 f"{g.cert_rdp.mean():.3f} ({g.cert_rdp.min():.3f})"]
        mac(f"ScMP{nm}CertRdp", g.cert_rdp.mean(), ".3f")
        mac(f"ScMP{nm}CertRdpMin", g.cert_rdp.min(), ".3f")
        mac(f"ScMP{nm}CertMyo", g.cert_myo.mean(), ".3f")
        mac(f"ScMP{nm}Tpool", g.t_pools.mean(), ".1f")
        mac(f"ScMP{nm}Tbound", g.t_bound.mean(), ".1f")
        mac(f"ScMP{nm}N", len(g))
        if ex:
            tr_r = (g.J_restricted / g.J_exact)
            tr_m = (g.J_myopic / g.J_exact)
            ubo = (g.UB / g.J_exact)
            cells += [f"{tr_m.mean():.3f}", f"{tr_r.mean():.3f} ({tr_r.min():.3f})", f"{ubo.mean():.2f}",
                      f"{g.t_exact.mean():.1f}"]
            mac(f"ScMP{nm}TrueRdp", tr_r.mean(), ".3f"); mac(f"ScMP{nm}TrueRdpMin", tr_r.min(), ".3f")
            mac(f"ScMP{nm}TrueMyo", tr_m.mean(), ".3f")
            mac(f"ScMP{nm}UbOpt", ubo.mean(), ".2f"); mac(f"ScMP{nm}Texact", g.t_exact.mean(), ".1f")
            mac(f"ScMP{nm}ExactAgree", int(np.sum(np.isclose(g.J_exact, g.J_exact_check))))
        else:
            cells += ["--", "--", "--", "--"]
        cells += [f"{g.t_pools.mean() + g.t_restricted.mean():.1f}", f"{g.t_bound.mean():.1f}"]
        lines.append(" & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{rrrcccccrrr}\n\\toprule\n"
            "& & & \\multicolumn{2}{c}{certified ratio $J/\\mathrm{UB}$} & \\multicolumn{3}{c}{against the exact optimum} & & \\multicolumn{2}{c}{time (s)} \\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\\cmidrule(lr){10-11}\n"
            "$n$ & inst. & pool & myopic & restricted DP & myopic & restricted DP & UB/OPT & exact (s) & pools+DP & bound \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    open(os.path.join(TAB, "tab_scale_mp.tex"), "w").write(body)
    return d


# ---------------------------------------------------------------------------
# graded (fuzzy) readings (experiment6.py)
# ---------------------------------------------------------------------------
def graded_results():
    files = sorted(glob.glob(os.path.join(RES, "grd_*.pkl")))
    if not files:
        return None
    rows = []
    for f in files:
        rows += [{k: v for k, v in r.items() if k != "masks"} for r in pickle.load(open(f, "rb"))["rows"]]
    d = pd.DataFrame(rows)
    pr = d.groupby(["method", "budget_frac", "rep"]).mean(numeric_only=True).reset_index()
    mac("GrdNRep", d.rep.nunique())
    lines = []
    bn = {0.075: "Seven", 0.10: "Ten", 0.15: "Fifteen", 0.20: "Twenty", 0.30: "Thirty"}
    for bf in sorted(pr.budget_frac.unique()):
        cells = [f"{100 * bf:g}\\%"]
        for kind in ("exact/window", "DP-exact"):
            g = pr[(pr.method == f"graded {kind}") & (pr.budget_frac == bf)].set_index("rep")
            c = pr[(pr.method == f"crisp {kind}") & (pr.budget_frac == bf)].set_index("rep")
            da, ha = tci((g.auroc_graded - c.auroc_graded).values)
            dq, hq = tci((g.qg_test - c.qg_test).values)
            dc, hc = tci((g.auroc_crisp - c.auroc_crisp).values)
            cells += [f"${da:+.4f}\\pm{ha:.4f}$", fmt_p(wilcox((g.auroc_graded - c.auroc_graded).values)),
                      f"${dq:+.3f}\\pm{hq:.3f}$", f"${dc:+.4f}\\pm{hc:.4f}$"]
            if kind == "exact/window":
                nm = bn.get(round(bf, 4), None)
                if nm:
                    mac(f"GrdDAuc{nm}", f"{da:+.4f}"); mac(f"GrdDAuc{nm}ci", ha, ".4f")
                    mac(f"GrdDQ{nm}", f"{dq:+.3f}"); mac(f"GrdDQ{nm}ci", hq, ".3f")
        lines.append(" & ".join(cells) + " \\\\")
    # pooled over budgets
    for kind, kn in (("exact/window", "Exw"), ("DP-exact", "Dpx")):
        g = pr[pr.method == f"graded {kind}"].groupby("rep").mean(numeric_only=True)
        c = pr[pr.method == f"crisp {kind}"].groupby("rep").mean(numeric_only=True)
        dd = (g.auroc_graded - c.auroc_graded).values
        m, h = tci(dd)
        mac(f"GrdOp{kn}", f"{m:+.4f}"); mac(f"GrdOp{kn}ci", h, ".4f"); mac(f"GrdOpP{kn}", fmt_pm(wilcox(dd)))
        dq = (g.qg_test - c.qg_test).values
        m, h = tci(dq)
        mac(f"GrdOpQ{kn}", f"{m:+.3f}"); mac(f"GrdOpQ{kn}ci", h, ".3f")
        dc = (g.auroc_crisp - c.auroc_crisp).values
        m, h = tci(dc)
        mac(f"GrdOpCrispFeat{kn}", f"{m:+.4f}"); mac(f"GrdOpCrispFeat{kn}ci", h, ".4f")
    # value of graded features themselves (same crisp panel, graded vs crisp features)
    c = pr[pr.method == "crisp exact/window"].groupby("rep").mean(numeric_only=True)
    m, h = tci((c.auroc_graded - c.auroc_crisp).values)
    mac("GrdFeatGain", f"{m:+.4f}"); mac("GrdFeatGainci", h, ".4f")
    body = ("\\begin{tabular}{lcccccccc}\n\\toprule\n"
            "& \\multicolumn{4}{c}{exact per window} & \\multicolumn{4}{c}{exact multi-period} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "budget & $\\Delta$AUROC & $p$ & $\\Delta Q^{\\mathrm S}$ held-out & $\\Delta$AUROC (crisp feat.) & "
            "$\\Delta$AUROC & $p$ & $\\Delta Q^{\\mathrm S}$ held-out & $\\Delta$AUROC (crisp feat.) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    open(os.path.join(TAB, "tab_graded.tex"), "w").write(body)
    return d


# ---------------------------------------------------------------------------
# eICU demo validation (eicu_study.py)
# ---------------------------------------------------------------------------
def eicu_results(n_boot=1000):
    files = sorted(glob.glob(os.path.join(RES, "eicu_0*.pkl")))
    cp = os.path.join(RES, "eicu_cohort.pkl")
    summ = os.path.join(RES, "eicu_summary.json")
    if not files or not os.path.exists(cp):
        # patient-level eICU outputs are not redistributed; fall back to the
        # aggregate summary written when eicu_study.py was run with the data
        if os.path.exists(summ):
            d = json.load(open(summ))
            MAC.update(d["macros"])
            open(os.path.join(TAB, "tab_eicu.tex"), "w").write(d["table"])
        return None
    before = set(MAC)
    import engine as E
    co = pickle.load(open(cp, "rb"))
    y, g = np.asarray(co["y"]), np.asarray(co["groups"])
    rows, oof = [], []
    for f in files:
        d = pickle.load(open(f, "rb"))
        rows += d["rows"]
        oof.append(d["oof"])
    R = pd.DataFrame(rows)
    nrep = len(oof)
    OP = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    meths = ["Static reduct", "Static exact", "SS-reduct", "TCS-reduct", "L1-cost",
             "NB-VOI greedy", "RA-DSS myopic", "RA-DSS exact/window", "RA-DSS DP-exact"]
    ug, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ug))]

    def op_auc(m, idx):
        # mean over repeats of the pooled out-of-fold AUROC, averaged over budgets
        return np.mean([np.mean([E.fast_auroc(y[idx], o[(bf, m)][idx]) for o in oof]) for bf in OP])

    def auc_at(m, bf, idx):
        return np.mean([E.fast_auroc(y[idx], o[(bf, m)][idx]) for o in oof])

    allidx = np.arange(len(y))
    rng = np.random.default_rng(2026)
    boots = [np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
             for _ in range(n_boot)]
    est = {m: op_auc(m, allidx) for m in meths}
    full = auc_at("Full panel", 1.0, allidx)
    ref = "Static exact"
    bref = np.array([op_auc(ref, b) for b in boots])
    lines = []
    nm = {"RA-DSS DP-exact": "Dpx", "RA-DSS exact/window": "Exw", "NB-VOI greedy": "Voi",
          "Static reduct": "Static", "RA-DSS myopic": "Myo", "L1-cost": "Lone",
          "SS-reduct": "Ssr", "TCS-reduct": "Tcs", "Static exact": "Sex"}
    at10 = R[R.budget_frac == 0.10].groupby("method")[["cost", "n_distinct", "switches"]].mean()
    for m in meths:
        a10 = auc_at(m, 0.10, allidx)
        if m == ref:
            dtxt = "--"
        else:
            bm = np.array([op_auc(m, b) for b in boots])
            dd = est[m] - est[ref]
            lo, hi = np.percentile(bm - bref, [2.5, 97.5])
            dtxt = f"${dd:+.4f}$ [${lo:+.4f}$, ${hi:+.4f}$]"
            mac(f"EicuD{nm[m]}", f"{dd:+.4f}"); mac(f"EicuD{nm[m]}lo", f"{lo:+.4f}")
            mac(f"EicuD{nm[m]}hi", f"{hi:+.4f}")
            if m == "Static reduct":      # the same interval, read as static exact minus static reduct
                mac("EicuSexVsStatic", f"{-dd:+.4f}"); mac("EicuSexVsStaticlo", f"{-hi:+.4f}")
                mac("EicuSexVsStatichi", f"{-lo:+.4f}")
        mac(f"EicuOp{nm[m]}", est[m], ".3f")
        mac(f"EicuDistinct{nm[m]}", at10.loc[m, "n_distinct"], ".1f")
        lines.append(f"{LABEL[m]} & ${est[m]:.3f}$ & {dtxt} & ${a10:.3f}$ & "
                     f"${at10.loc[m, 'cost']:.0f}$ & ${at10.loc[m, 'n_distinct']:.1f}$ & "
                     f"${at10.loc[m, 'switches']:.1f}$ \\\\")
    # DP-exact against the static reduct (the comparison of the simulated study)
    bs = np.array([op_auc("Static reduct", b) for b in boots])
    bd = np.array([op_auc("RA-DSS DP-exact", b) for b in boots])
    lo, hi = np.percentile(bd - bs, [2.5, 97.5])
    mac("EicuDpxVsStatic", f"{est['RA-DSS DP-exact'] - est['Static reduct']:+.4f}")
    mac("EicuDpxVsStaticlo", f"{lo:+.4f}"); mac("EicuDpxVsStatichi", f"{hi:+.4f}")
    mac("EicuFull", full, ".3f")
    mac("EicuN", int(co["n"])); mac("EicuDeaths", int(co["deaths"]))
    mac("EicuDeathPct", 100 * co["deaths"] / co["n"], ".1f")
    mac("EicuPatients", int(co["n_patients"])); mac("EicuHosp", int(co["n_hosp"]))
    mac("EicuNRep", nrep); mac("EicuNBoot", n_boot)
    mr = np.asarray(co["meas_rate"])
    import casestudy as CS
    k = {kk: i for i, kk in enumerate(CS.KEYS)}
    mac("EicuMeasBmpFirst", 100 * mr[0, k["BMP"]], ".0f"); mac("EicuMeasBmpLast", 100 * mr[-1, k["BMP"]], ".0f")
    mac("EicuMeasAny", 100 * mr[:, [i for i in range(len(CS.KEYS)) if CS.KEYS[i] not in ("DDIM", "PCT")]].mean(), ".0f")
    body = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
            "& \\multicolumn{2}{c}{mean AUROC, budgets 5--30\\%} & \\multicolumn{4}{c}{at a 10\\% budget} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\n"
            "policy & mean & $\\Delta$ vs static exact [95\\% CI] & AUROC & cost (\\$) & panels & changes \\\\\n"
            "\\midrule\n" + "\n".join(lines) +
            f"\n\\midrule\nFull panel & \\multicolumn{{2}}{{l}}{{AUROC ${full:.3f}$ at every budget}} & ${full:.3f}$ & "
            f"${at10.loc['Full panel', 'cost']:.0f}$ & $1.0$ & $12.0$ \\\\\n\\bottomrule\n\\end{{tabular}}")
    open(os.path.join(TAB, "tab_eicu.tex"), "w").write(body)
    json.dump(dict(macros={k: MAC[k] for k in sorted(set(MAC) - before)}, table=body),
              open(summ, "w"), indent=1)
    return est

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    M = load_main()
    mac("NPatients", 3000); mac("OutNoise", "2.8"); mac("GammaDefault", "0.20")
    mac("PanelWindow", f"\\${CS.FULL_PANEL_COST:.2f}")
    mac("PanelDay", f"\\${6 * CS.FULL_PANEL_COST:.2f}")
    mac("AlphaGS", "0.427")
    S, D, pr, wide, cost_at, opm, full, fullcost = main_results(M)
    paired_all(D)
    static_exact_results(M, pr, wide, opm, cost_at)
    scale_mp_results()
    graded_results()
    eicu_results()
    verification_table()
    tab_parameters()
    fig_frontier(pr, D, float(full.mean()))
    freqs = fig_selection(M)
    selection_stats(M, freqs)
    mp, win = mp_results(M)
    g = lam_results(M)
    cert = cert_results()
    fig_lambda_cert(g, cert)
    h = headroom_results()
    fig_headroom_gap(h, win)
    sat_results()
    pr_ = price_results()
    cr_ = corr_results()
    write_sens_table(pr_[1] if pr_ else None, cr_[1] if cr_ else None)
    sr = sens_results()
    if sr is not None:
        fig_ofat(sr[1], list(dict.fromkeys(sr[0].tag)))
    fig_multiperiod_schematic()
    fig_design()
    import figs_extra as FX
    FX.fig_stat(stat_results(), FIG, tci)
    FX.fig_dp_timing(RES, FIG, mac)
    FX.fig_cohort(FIG, mac, 3000, 2.8)
    inc_results()
    scaling_results()
    with open(os.path.join(PAP, "numbers2.tex"), "w") as f:
        f.write("% auto-generated by code/analysis2.py -- do not edit\n")
        for k in sorted(MAC):
            f.write(f"\\newcommand{{\\{k}}}{{{tex_group(MAC[k])}}}\n")
    json.dump({k: v for k, v in MAC.items()}, open(os.path.join(RES, "summary2.json"), "w"), indent=1)
    print(len(MAC), "macros;", M["n_rep"], "replicates")
