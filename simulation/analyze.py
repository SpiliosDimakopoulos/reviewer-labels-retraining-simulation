"""Aggregate raw experiment output into results/results.json and CSV tables.

Every reported number comes from results.json.
Usage (from the simulation folder): python analyze.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from analyze_extra import wilson as _wilson

RAW = os.path.join("results", "raw")
EPS = 0.10
CAP = 0.15
SHARE = 0.3
PQ = SHARE * (1 - SHARE)
IOTA = float(norm.pdf(norm.ppf(1 - CAP)) / CAP)
R = {}  # results dict


def load(name):
    p = os.path.join(RAW, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def first_cross(vals, thr):
    """First index where vals > thr, or NaN."""
    idx = np.flatnonzero(np.asarray(vals) > thr)
    return float(idx[0]) if len(idx) else np.nan


def first_run2(vals, thr):
    """Cycle at which the monitor declares: the second of two consecutive exceedances of thr, or NaN."""
    v = np.asarray(vals) > thr
    for i in range(1, len(v)):
        if v[i] and v[i - 1]:
            return float(i)
    return np.nan


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def per_seed(df, col, fn):
    return df.groupby("seed").apply(lambda g: fn(g.sort_values("t")[col].values), include_groups=False)


def parse_bl(tag):
    b = float(tag.split("_")[0][1:])
    lam = float(tag.split("_")[1][1:])
    return b, lam


# ---------------------------------------------------------------- E1 null
def e1():
    df = load("E1")
    if df is None:
        return
    out = {}
    for tag, g in df.groupby("tag"):
        n = int(tag[1:])
        viol = g.assign(v=g.R0 > 1 + EPS)
        per_cycle = float(viol.v.mean())
        any_t = float(viol.groupby("seed").v.any().mean())
        any2 = float(g.groupby("seed").apply(lambda q: not np.isnan(first_run2(q.sort_values("t").R0.values, 1 + EPS)), include_groups=False).mean())
        out[str(n)] = dict(per_cycle_false_alarm=per_cycle, p_any_in_24_cycles=any_t, p_any_in_24_cycles_two_consecutive=any2,
                           mean_R=float(g.R0.mean()), sd_R=float(g.R0.std()))
    R["E1_null_false_alarm"] = out


# ---------------------------------------------------------------- E2 step grid
def e2():
    df = load("E2")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        b, lam = parse_bl(tag)
        t0 = g[g.t == 0]
        t1 = g[g.t == 1]
        late = g[(g.t >= 10) & (g.t <= 23)]
        dbar = t0.dbar.mean()
        coef_pred = b * lam * PQ / (1 + lam ** 2 * PQ) / (dbar * (1 - dbar)) if lam > 0 and b > 0 else 0.0
        lnR1 = np.log(t1.R0).mean()
        lnR_pred = IOTA * lam * coef_pred / t1.sd_z.mean()
        det = per_seed(g, "R0", lambda v: first_run2(v, 1 + EPS))
        a0 = per_seed(g, "R0", lambda v: float(v[0] > 1 + EPS))
        amid = per_seed(g, "R0", lambda v: float(v[12] > 1 + EPS))
        aend = per_seed(g, "R0", lambda v: float(v[23] > 1 + EPS))
        ffgap = g.assign(gap=g.ffS0 - g.ffR0)

        def harm_after(q):
            q = q.sort_values("t")
            dt = first_run2(q.R0.values, 1 + EPS)
            gap = (q.ffS0 - q.ffR0).values
            tot = gap[1:].sum()
            return np.nan if (np.isnan(dt) or tot <= 0) else gap[int(dt) + 1:].sum() / tot
        ha = g.groupby("seed").apply(harm_after, include_groups=False)
        if b == 0:
            ha = ha * np.nan
        rows.append(dict(b=b, lam=lam, dbar0=dbar, R0_t0=t0.R0.mean(), R_t1=t1.R0.mean(), R_t1_lo=ci(t1.R0)[0], R_t1_hi=ci(t1.R0)[1],
                         R_late=late.R0.mean(), lnR_t1=lnR1, lnR_t1_pred=lnR_pred, coef_t1=t1.coef_p0.mean(), coef_t1_pred=coef_pred,
                         sd_z_t1=t1.sd_z.mean(), detect_median=float(np.nanmedian(det)) if det.notna().any() else np.nan,
                         p_detect_in_24=float(det.notna().mean()), p_flag_audit_t0=float(a0.mean()),
                         p_flag_audit_t12=float(amid.mean()), p_flag_audit_t23=float(aend.mean()), p_flag_audit_t12_ci=_wilson(int(amid.sum()), len(amid)), p_detect_in_24_ci=_wilson(int(det.notna().sum()), len(det)),
                         excess_false_flag_per_1000_late=float(1000 * ffgap[(ffgap.t >= 1)].gap.mean()),
                         harm_share_after_detection=float(ha.mean()) if ha.notna().any() else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join("results", "E2_table.csv"), index=False)
    sub = T[(T.b > 0) & (T.lam > 0)]
    x, y = sub.lnR_t1.values, sub.lnR_t1_pred.values
    r2 = 1 - ((x - y) ** 2).sum() / ((x - x.mean()) ** 2).sum()
    xc, yc = sub.coef_t1.values, sub.coef_t1_pred.values
    r2c = 1 - ((xc - yc) ** 2).sum() / ((xc - xc.mean()) ** 2).sum()
    R["E2_step_prediction"] = dict(iota=IOTA, r2_lnR=float(r2), mean_abs_err_lnR=float(np.abs(x - y).mean()),
                                   slope_obs_on_pred=float(np.polyfit(y, x, 1)[0]), r2_coef=float(r2c),
                                   n_configs=int(len(sub)))
    R["E2_controls"] = dict(
        b0_all_lam_max_abs_lnR_t1=float(T[T.b == 0].lnR_t1.abs().max()),
        lam0_all_b_max_abs_lnR_t1=float(T[T.lam == 0].lnR_t1.abs().max()),
        R_t0_mean=float(T.R0_t0.mean()), R_t0_max_abs_dev=float((T.R0_t0 - 1).abs().max()))
    ref = T[(T.b == 0.15) & (T.lam == 1.0)].iloc[0]
    R["E2_reference_b0.15_lam1.0"] = {k: (v if isinstance(v, (list, tuple)) else float(v)) for k, v in ref.items()}
    R["E2_table"] = T.round(4).to_dict(orient="records")


# ---------------------------------------------------------------- E2L creep
def e2l():
    df = load("E2L")
    if df is None:
        return
    m = df.groupby("t").R0.mean()
    R["E2L_creep"] = dict(R_t1=float(m.loc[1]), R_t10=float(m.loc[10]), R_t30=float(m.loc[30]), R_t59=float(m.loc[59]),
                          creep_ratio_t59_over_t1=float(m.loc[59] / m.loc[1]))


# ---------------------------------------------------------------- E3 base rates
def e3():
    df = load("E3")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        dq = float(tag.split("_")[0][2:])
        b = float(tag.split("_")[1][1:])
        m = g.groupby("t").R0.mean()
        rows.append(dict(dq=dq, b=b, R_t0=float(m.loc[0]), R_t1=float(m.loc[1]), R_t23=float(m.loc[23]),
                         change_t0_to_t23=float(m.loc[23] - m.loc[0])))
    R["E3_base_rates"] = rows


# ---------------------------------------------------------------- E4 coupling
def e4():
    df = load("E4")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        a = float(tag[1:])
        m = g.groupby("t").mean(numeric_only=True)
        det = per_seed(g, "R0", lambda v: first_run2(v, 1 + EPS))
        rows.append(dict(anchor=a, R_t1=float(m.R0.loc[1]), R_t5=float(m.R0.loc[5]), R_t15=float(m.R0.loc[15]), R_t39=float(m.R0.loc[39]),
                         detect_median=float(np.nanmedian(det)) if det.notna().any() else np.nan,
                         p_detect_in_40=float(det.notna().mean()), sd_z_t39=float(m.sd_z.loc[39]), wx_norm_t39=float(m.wx_norm.loc[39])))
    R["E4_coupling"] = rows


# ---------------------------------------------------------------- E5 blinding
def e5():
    df = load("E5")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        m = g.groupby("t").R0.mean()
        rows.append(dict(reviewer=tag, R_t1=float(m.loc[1]), R_late=float(m.loc[10:23].mean())))
    R["E5_blinding"] = rows


# ---------------------------------------------------------------- E6 intersection
def e6():
    df = load("E6")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        gi = float(tag.split("_")[0][4:])
        n = int(tag.split("_")[1][1:])
        late = g[(g.t >= 10) & (g.t <= 23)]
        det_single = per_seed(g, "R0", lambda v: first_run2(v, 1 + EPS))
        det_x = per_seed(g, "Rx", lambda v: first_run2(v, 1 + EPS))
        det_single3 = float((det_single <= 3).mean())
        rows.append(dict(gint=gi, n=n, p_detect_single_t3=det_single3, R_attr0_late=float(late.R0.mean()), R_attr1_late=float(late.R1.mean()), R_inter_late=float(late.Rx.mean()),
                         cell_size=float(g.nX.mean()), p_detect_single=float(det_single.notna().mean()),
                         p_detect_inter=float(det_x.notna().mean()),
                         p_detect_inter_t3=float((det_x <= 3).mean())))
    R["E6_intersection"] = rows


# ---------------------------------------------------------------- E7 window and capacity
def e7():
    df = load("E7")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        k = int(tag.split("_")[0][1:])
        cap = float(tag.split("_")[1][3:])
        m = g.groupby("t").R0.mean()
        rows.append(dict(k=k, cap=cap, R_t1=float(m.loc[1]), R_t5=float(m.loc[5]), R_t39=float(m.loc[39]),
                         creep=float(m.loc[39] / m.loc[1])))
    R["E7_window_capacity"] = rows


# ---------------------------------------------------------------- E8 biased first model
def e8():
    df = load("E8")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        b = float(tag[1:])
        m = g.groupby("t").R0.mean()
        p0 = per_seed(g, "R0", lambda v: float(v[0] > 1 + EPS)).mean()
        rows.append(dict(b=b, R_t0=float(m.loc[0]), R_t1=float(m.loc[1]), R_t23=float(m.loc[23]), p_flag_static_audit_t0=float(p0)))
    R["E8_biased_first_model"] = rows



# ---------------------------------------------------------------- R-squared form of the step formula (E2 + E9)
def r2_form():
    """ln R(1) ~ iota * b * R2(G | model features) / (dbar (1 - dbar) sd_z), all inputs measured in the run."""
    rows = []
    for name in ("E2", "E9"):
        df = load(name)
        if df is None:
            continue
        for tag, g in df.groupby("tag"):
            b = float(tag.split("_b")[-1]) if name == "E9" else float(tag.split("_")[0][1:])
            t0, t1 = g[g.t == 0], g[g.t == 1]
            if b == 0:
                continue
            dbar = t0.dbar.mean()
            pred = IOTA * b * t0.r2G0.mean() / (dbar * (1 - dbar) * t1.sd_z.mean())
            rows.append(dict(exp=name, tag=tag, b=b, r2G=float(t0.r2G0.mean()), lnR_obs=float(np.log(t1.R0).mean()), lnR_pred=float(pred)))
    if not rows:
        return
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join("results", "R2form_table.csv"), index=False)

    def fit_stats(sub):
        x, y = sub.lnR_obs.values, sub.lnR_pred.values
        return dict(n=int(len(sub)), r2=float(1 - ((x - y) ** 2).sum() / ((x - x.mean()) ** 2).sum()),
                    mean_abs_err=float(np.abs(x - y).mean()), slope=float(np.polyfit(y, x, 1)[0]))
    R["R2form_all"] = fit_stats(T)
    R["R2form_E2_only"] = fit_stats(T[T.exp == "E2"])
    if (T.exp == "E9").any():
        R["R2form_E9_only"] = fit_stats(T[T.exp == "E9"])
        R["R2form_E9_table"] = T[T.exp == "E9"].round(4).to_dict(orient="records")


# ---------------------------------------------------------------- E4 diagnostics and timing separated from level
def e4_diag():
    out = []
    for name in ("E4", "E4C"):
        df = load(name)
        if df is None:
            continue
        for tag, g in df.groupby("tag"):
            m = g.groupby("t").mean(numeric_only=True)
            plateau = float(m.R0.loc[30:39].mean())
            target = 1 + 0.9 * (plateau - 1)
            t90 = next((int(t) for t, v in m.R0.items() if t >= 1 and v >= target), np.nan)
            out.append(dict(exp=name, tag=tag, plateau=plateau, t90=t90,
                            esc_share_mean=float((g.n_esc / 20000).mean()), esc_share_max=float((g.n_esc / 20000).max()),
                            sat_frac_mean=float(g.sat_frac.mean()), tie_frac_max=float(g.tie_frac.max()),
                            sd_z_t39=float(m.sd_z.loc[39]), R_t1=float(m.R0.loc[1])))
    if out:
        R["E4_diagnostics"] = out


# ---------------------------------------------------------------- harm attributable to the loop
def harm_attribution():
    T = pd.DataFrame(R.get("E2_table", []))
    if T.empty:
        return
    base = T[T.lam == 0].set_index("b").excess_false_flag_per_1000_late
    T["loop_attributable_ff_per_1000"] = [r.excess_false_flag_per_1000_late - base.loc[r.b] for r in T.itertuples()]
    R["E2_loop_attributable_harm"] = T[T.b > 0][["b", "lam", "excess_false_flag_per_1000_late", "loop_attributable_ff_per_1000"]].round(3).to_dict(orient="records")


# ---------------------------------------------------------------- level monitor vs drift monitor
def monitors():
    def declare_level(v):
        return not np.isnan(first_run2(v, 1 + EPS))

    def declare_drift(v):
        return not np.isnan(first_run2(v / v[0], 1 + EPS))

    scen = []
    def add(label, df, tag_filter):
        g = df[df.tag.isin(tag_filter)] if tag_filter else df
        lev = g.groupby("seed").apply(lambda q: declare_level(q.sort_values("t").R0.values), include_groups=False).mean()
        dri = g.groupby("seed").apply(lambda q: declare_drift(q.sort_values("t").R0.values), include_groups=False).mean()
        scen.append(dict(scenario=label, level_monitor=float(lev), drift_monitor=float(dri), runs=int(g.seed.nunique() * len(set(g.tag)))))
    e1, e2, e3, e8 = load("E1"), load("E2"), load("E3"), load("E8")
    if e1 is not None:
        add("null deployment, 20,000 applicants per cycle (no bias)", e1, ["n20000"])
    if e3 is not None:
        add("fair system, base-rate gap of 5 points (no reviewer penalty)", e3, ["dq0.05_b0.0"])
    if e8 is not None:
        add("bias inherited from first model, b = 0.10", e8, ["b0.1"])
        add("bias inherited from first model, b = 0.15", e8, ["b0.15"])
    if e2 is not None:
        add("clean first model, penalty returns at first retraining (b = 0.15, proxy 1.0)", e2, ["b0.15_l1.0"])
    R["monitor_comparison"] = scen


def main():
    import analyze_extra
    for f in (e1, e2, e2l, e3, e4, e5, e6, e7, e8, r2_form, e4_diag, harm_attribution, monitors):
        f()
    analyze_extra.run_all(R)
    R["settings"] = dict(eps=EPS, cap=CAP, share_B=SHARE, default_n_per_cycle=20000, default_T=24, default_k=4)
    with open(os.path.join("results", "results.json"), "w", encoding="utf8") as fh:
        json.dump(R, fh, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))
    print("keys:", list(R))


if __name__ == "__main__":
    main()
