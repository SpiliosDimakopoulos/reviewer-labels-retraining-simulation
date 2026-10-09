"""Figures for the paper and its supplement. Usage (from the simulation folder): python figures.py
Reads results/raw/*.csv and results/results.json. Writes ../figures/*.pdf and *.png (300 dpi).
Palette: Okabe-Ito based, readable in greyscale; direct labels instead of legends where possible.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = os.path.join("results", "raw")
OUT = os.path.join("..", "figures")
os.makedirs(OUT, exist_ok=True)
EPS = 0.10
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
BLUES = ["#9ecae1", "#4292c6", "#08519c"]
ORANGE, GREEN, VERM, PURPLE = "#E69F00", "#009E73", "#D55E00", "#CC79A7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 120, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def band(ax, df, col, color, label=None, lw=1.8, alpha=0.18, ls="-", marker=None, every=3):
    g = df.groupby("t")[col]
    m, lo, hi = g.mean(), g.quantile(0.025), g.quantile(0.975)
    ax.fill_between(m.index, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(m.index, m.values, color=color, lw=lw, ls=ls, label=label, marker=marker, markevery=every, markersize=4.5, markerfacecolor="white" if marker else None, markeredgewidth=1.1)
    return m


def tolerance(ax, x1, xlab=None, label=True):
    ax.axhline(1 + EPS, color=MUTED, lw=1, ls=(0, (4, 3)))
    if label:
        ax.text(xlab if xlab is not None else x1, 1 + EPS - 0.006, "tolerance 1.10", color=MUTED, fontsize=8, ha="right", va="top")


# ----------------------------------------------------------------------------- Figure 1
def fig1():
    df = pd.read_csv(os.path.join(RAW, "E2.csv"))
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    spec = [("b0.0_l1.0", "reviewer penalty 0", MUTED, "o"), ("b0.05_l1.0", "0.05", BLUES[0], "s"),
            ("b0.1_l1.0", "0.10", BLUES[1], "^"), ("b0.15_l1.0", "0.15", BLUES[2], "D")]
    for tag, lab, col, mk in spec:
        g = df[df.tag == tag]
        m = band(ax, g, "R0", col, marker=mk)
        ax.text(23.3, m.iloc[-1], lab, color=col if col != BLUES[0] else "#5b9bc2", fontsize=8.5, va="center")
    tolerance(ax, 23, label=False)
    ax.axvline(1, color=MUTED, lw=0.8, ls=":")
    ax.annotate("first retraining on\nreviewer labels", (1, 1.30), xytext=(2.6, 1.34), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("deployment audit (cycle 0)\npasses in every case", (0, 1.0), xytext=(4.5, 0.86), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(0, 25.5)
    ax.set_ylim(0.8, 1.55)
    ax.set_xlabel("Retraining cycle")
    ax.set_ylabel("Escalation-rate ratio, group vs. rest")
    save(fig, "fig1_emergence")


# ----------------------------------------------------------------------------- Figure 2
def fig2():
    r = json.load(open(os.path.join("results", "results.json")))
    T = pd.DataFrame(r["E2_table"])
    bs, ls = sorted(T.b.unique()), sorted(T.lam.unique())
    M = np.array([[T[(T.b == b) & (T.lam == l)].R_late.iloc[0] for l in ls] for b in bs])
    fig, (a, c) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    im = a.imshow(M, origin="lower", cmap="Blues", vmin=0.98, vmax=2.0, aspect="auto")
    a.set_xticks(range(len(ls)), [f"{v:g}" for v in ls])
    a.set_yticks(range(len(bs)), [f"{v:.2f}" for v in bs])
    a.set_xlabel("Proxy strength")
    a.set_ylabel("Reviewer penalty")
    a.grid(False)
    for i in range(len(bs)):
        for j in range(len(ls)):
            v = M[i, j]
            exceeds = v > 1 + EPS
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                   color="white" if v > 1.55 else INK, fontweight="bold" if exceeds else "normal")
            if exceeds:
                a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
    a.set_title("(a)", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    P = pd.read_csv(os.path.join("results", "formula_points_E2_E9.csv"))
    M11 = pd.DataFrame(r["E11_misspecification"])
    M11o = pd.DataFrame(r["E11_outside_scope"])
    for lab, x, y, col, mk in (("single proxy", P[P.exp == "E2"].empirical, P[P.exp == "E2"].obs, BLUES[2], "o"),
                               ("several proxies", P[P.exp == "E9"].empirical, P[P.exp == "E9"].obs, ORANGE, "s"),
                               ("assumptions varied", M11.pred_empirical_hazard, M11.lnR_obs, VERM, "^"),
                               ("outside the derivation", M11o.pred_empirical_hazard, M11o.lnR_obs, MUTED, "X")):
        c.scatter(x, y, s=26, color=col, marker=mk, edgecolor="white", lw=0.5, zorder=3, label=lab)
    allx = np.concatenate([P.empirical.values, M11.pred_empirical_hazard.values, P.obs.values, M11.lnR_obs.values, M11o.pred_empirical_hazard.values, M11o.lnR_obs.values])
    lim = [min(allx.min(), 0) - 0.03, allx.max() + 0.05]
    c.plot(lim, lim, color=MUTED, lw=1, zorder=2)
    c.set_xlim(lim)
    c.set_ylim(lim)
    c.set_xlabel("Predicted log ratio, first cycle")
    c.set_ylabel("Simulated log ratio, first cycle")
    st = r["First_step_formula_variants"]["empirical_fitted_variance"]
    st11 = r["E11_fit_empirical"]
    c.text(0.04, 0.96, f"R$^2$ = {st['r2']:.3f} (n = {st['n']})" + "\n" + f"R$^2$ = {st11['r2']:.3f} (n = {st11['n']}, varied)", transform=c.transAxes, va="top", fontsize=8, color=INK)
    c.legend(frameon=False, loc="lower right", fontsize=8)
    c.set_title("(b)", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "fig2_grid_and_prediction")


# ----------------------------------------------------------------------------- Figure 3
def fig3():
    r = json.load(open(os.path.join("results", "results.json")))
    e4 = pd.read_csv(os.path.join(RAW, "E4D.csv"))
    fig, (a, c) = plt.subplots(1, 2, figsize=(7.8, 3.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    spec = [("w0.0", "0", MUTED, "o"), ("w0.25", "0.25", BLUES[0], "s"), ("w0.5", "0.5", BLUES[1], "^"), ("w0.75", "0.75", BLUES[2], "D")]
    for tag, lab, col, mk in spec:
        g = e4[(e4.tag == tag) & (e4.t <= 30)]
        m = band(a, g, "R0", col, alpha=0.0, marker=mk, every=4)
        a.text(30.6, m.iloc[-1] + {"w0.0": 0.02, "w0.25": -0.015, "w0.5": 0.0, "w0.75": -0.005}[tag], lab, color=col if col != BLUES[0] else "#5b9bc2", fontsize=8.5, va="center")
    tolerance(a, 30, label=False)
    a.set_xlim(0, 33.5)
    a.set_ylim(1.0, 1.4)
    a.set_xlabel("Retraining cycle")
    a.set_ylabel("Escalation-rate ratio")
    a.set_title("(a)", fontsize=9, loc="left")

    mon = pd.DataFrame(r["monitor_comparison"])
    short = ["Null deployment\n(no bias)", "Fair system,\n5-point base-rate gap", "Bias inherited\nfrom first model (0.10)",
             "Bias inherited\nfrom first model (0.15)", "Clean start, penalty\nreturns at retraining"]
    y = np.arange(len(mon))[::-1]
    h = 0.36
    for series, off, col, lab in (("level_monitor", h / 2, BLUES[2], "level monitor"), ("drift_large_baseline", -h / 2, ORANGE, "drift monitor, large reference sample")):
        vals = mon[series].values
        cis = np.array(mon["level_ci" if series == "level_monitor" else "drift_large_ci"].tolist())
        c.barh(y + off, vals, height=h, color=col, label=lab, hatch="////" if series != "level_monitor" else None, edgecolor="white" if series != "level_monitor" else col, linewidth=0)
        c.errorbar(vals, y + off, xerr=[vals - cis[:, 0], cis[:, 1] - vals], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        for yy, v, hi in zip(y, vals, cis[:, 1]):
            c.text(min(hi, 1.0) + 0.03, yy + off, f"{v:.0%}", va="center", fontsize=8)
    c.set_yticks(y, short, fontsize=8)
    c.set_xlim(0, 1.15)
    c.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    c.set_xlabel("Share of deployments declared (95% interval)")
    c.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.2))
    c.grid(axis="y", visible=False)
    c.set_title("(b)", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "fig3_coupling_and_monitors")


# ----------------------------------------------------------------------------- Figure 4
def fig4():
    df = pd.read_csv(os.path.join(RAW, "E10.csv"))
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    fail = [("none", "no action", MUTED, "o"), ("freeze", "stop retraining", VERM, "s"), ("rollback_continue", "roll back, keep retraining on reviewer labels", ORANGE, "^")]
    hold = [("rollback_freeze", "roll back, stop retraining", BLUES[0], "D"), ("drop_proxies", "remove the proxies", BLUES[1], "v"), ("verified_labels", "train on verified outcomes", BLUES[2], "P")]
    for k, (tag, lab, col, mk) in enumerate(fail + hold):
        g = df[df.tag == tag]
        m = g.groupby("t").R0.mean()
        ax.plot(m.index, m.values, color=col, lw=1.7, ls="-" if (tag, lab, col, mk) in hold or tag == "none" else (0, (5, 2)), label=lab,
                marker=mk, markevery=(k % 3, 3), markersize=4.5, markerfacecolor="white", markeredgewidth=1.1)
    ax.axvline(6, color=MUTED, lw=0.8, ls=":")
    ax.text(6.2, 1.43, "intervention", fontsize=8, color=MUTED, va="top")
    tolerance(ax, 23.8)
    ax.set_xlim(0, 24)
    ax.set_ylim(0.9, 1.45)
    ax.set_xlabel("Retraining cycle")
    ax.set_ylabel("Escalation-rate ratio")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save(fig, "fig4_interventions")


# ----------------------------------------------------------------------------- Supplement figures
def figS1():
    r = json.load(open(os.path.join("results", "results.json")))["E1_null_false_alarm"]
    ns = sorted(int(k) for k in r)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(ns, [r[str(n)]["p_any_in_24_cycles"] for n in ns], "o-", color=VERM, label="single exceedance")
    ax.plot(ns, [r[str(n)]["p_any_in_24_cycles_two_consecutive"] for n in ns], "o-", color=BLUES[2], label="two consecutive exceedances")
    ax.set_xscale("log")
    ax.set_xticks(ns, [f"{n:,}" for n in ns])
    ax.set_xlabel("Applicants per cycle")
    ax.set_ylabel("Probability of a false alarm in 24 cycles")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "figS1_false_alarm")


def figS2():
    r = json.load(open(os.path.join("results", "results.json")))["E5_blinding"]
    d = {x["reviewer"]: x["R_late"] for x in r}
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ss = [0.0, 0.25, 0.5, 1.0, 1.5]
    ax.plot(ss, [d[f"blind_s{s}"] for s in ss], "o-", color=BLUES[2], label="blinded reviewer leaning on a visible correlate")
    ax.axhline(d["unblinded"], color=ORANGE, lw=1.6)
    ax.text(1.5, d["unblinded"] + 0.03, "reviewer who sees the group", color=ORANGE, ha="right", fontsize=8)
    ax.axhline(1.10, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("Substitution strength")
    ax.set_ylabel("Plateau ratio")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "figS2_blinding")


def figS3():
    r = pd.DataFrame(json.load(open(os.path.join("results", "results.json")))["E7_window_capacity"])
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for cap, col in ((0.05, VERM), (0.15, BLUES[2]), (0.30, BLUES[0])):
        s = r[r.cap == cap].sort_values("k")
        ax.plot(s.k, s.R_t39, "o-", color=col, label=f"capacity {cap:.0%}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_xlabel("Training window (cycles)")
    ax.set_ylabel("Ratio at cycle 39")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "figS3_capacity_window")


def figS4():
    df = pd.read_csv(os.path.join(RAW, "E8.csv"))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for tag, col in (("b0.0", MUTED), ("b0.05", BLUES[0]), ("b0.1", BLUES[1]), ("b0.15", BLUES[2])):
        m = band(ax, df[df.tag == tag], "R0", col, alpha=0.12)
    tolerance(ax, 23)
    ax.set_xlabel("Retraining cycle")
    ax.set_ylabel("Escalation-rate ratio")
    save(fig, "figS4_inherited_bias")


def figS5():
    r = pd.DataFrame(json.load(open(os.path.join("results", "results.json")))["E6_intersection"])
    s = r[(r.gint == 0.0) & (r.n == 20000)].iloc[0]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    vals = [s.R_attr0_late, s.R_attr1_late, s.R_inter_late]
    ax.bar(["attribute 1", "attribute 2", "both"], vals, color=[BLUES[1], BLUES[1], ORANGE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.axhline(1.10, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_ylim(1, max(vals) + 0.1)
    ax.set_ylabel("Plateau ratio")
    ax.grid(axis="x", visible=False)
    save(fig, "figS5_intersection")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, figS1, figS2, figS3, figS4, figS5):
        f()
