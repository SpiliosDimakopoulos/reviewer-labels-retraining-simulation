"""Additional analyses: repairs, misspecification, nonlinear models, nulls and persistence, deference, penalty
estimators, formula variants, monitors. Imported by analyze.py; every function writes into the shared results dict R."""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

RAW = os.path.join("results", "raw")
EPS = 0.10
CAP = 0.15
IOTA = float(norm.pdf(norm.ppf(1 - CAP)) / CAP)


def load(name):
    p = os.path.join(RAW, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def first_run(vals, thr, k=2):
    """Index of the k-th consecutive exceedance of thr (the cycle at which the monitor declares), or NaN."""
    v = np.asarray(vals) > thr
    run = 0
    for i, x in enumerate(v):
        run = run + 1 if x else 0
        if run >= k:
            return float(i)
    return np.nan


def first_run2(vals, thr):
    return first_run(vals, thr, 2)


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [np.nan, np.nan]


def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, c - h)), float(min(1, c + h))]


def iota_for(cap):
    return float(norm.pdf(norm.ppf(1 - cap)) / cap)


def declared(q, col="R0", scale=1.0, k=2):
    return not np.isnan(first_run(q.sort_values("t")[col].values / scale, 1 + EPS, k))


def r2(x, y):
    return float(1 - ((x - y) ** 2).sum() / ((x - x.mean()) ** 2).sum())


# ------------------------------------------------------------------ repairs
def _repairs(name, key, R):
    df = load(name)
    if df is None:
        return
    base = df[df.tag == "none"]
    base_R = base[base.t >= 8].groupby("seed").R0.mean()
    base_ff = base[base.t >= 8].groupby("seed").apply(lambda q: 1000 * (q.ffS0 - q.ffR0).mean(), include_groups=False)
    base_tp = base[base.t >= 8].groupby("seed").esc_tp_per_1000.mean()
    rows = []
    for tag, g in df.groupby("tag"):
        after = g[g.t >= 8]
        pre = g[(g.t >= 1) & (g.t <= 5)]
        per_seed = after.groupby("seed").R0.mean()
        diff = (per_seed - base_R).dropna()
        ff_seed = after.groupby("seed").apply(lambda q: 1000 * (q.ffS0 - q.ffR0).mean(), include_groups=False)
        tp_seed = after.groupby("seed").esc_tp_per_1000.mean()
        se = diff.std() / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        rows.append(dict(intervention=tag, R_before=float(g[g.t == 5].R0.mean()), R_after=float(after.R0.mean()),
                         R_after_lo=ci(per_seed)[0], R_after_hi=ci(per_seed)[1],
                         paired_diff_R=float(diff.mean()), paired_diff_R_ci=[float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
                         excess_false_flags_per_1000_after=float(1000 * (after.ffS0 - after.ffR0).mean()),
                         excess_false_flags_per_1000_before=float(1000 * (pre.ffS0 - pre.ffR0).mean()),
                         paired_diff_ff=float((ff_seed - base_ff).dropna().mean()),
                         escalated_true_positives_per_1000_after=float(after.esc_tp_per_1000.mean()),
                         paired_diff_tp=float((tp_seed - base_tp).dropna().mean())))
    R[key] = rows


def e10(R):
    _repairs("E10", "E10_interventions", R)
    _repairs("E10b", "E10b_interventions", R)
    df = load("E10c")
    if df is not None:
        rows = []
        for tag, g in df.groupby("tag"):
            b, iv = tag[1:].split("_", 1)
            a = g[g.t >= 8]
            per = a.groupby("seed").R0.mean()
            rows.append(dict(b=float(b), intervention=iv, R_before=float(g[(g.t >= 2) & (g.t <= 5)].R0.mean()), R_after=float(per.mean()),
                             R_after_lo=ci(per)[0], R_after_hi=ci(per)[1], R_base=float(a.R_base.mean()),
                             excess_false_flags_per_1000_after=float(1000 * (a.ffS0 - a.ffR0).mean()),
                             escalated_true_positives_per_1000_after=float(a.esc_tp_per_1000.mean())))
        R["E10c_label_correction_base_rate_gap"] = rows


# ------------------------------------------------------------------ misspecification
def _pred_row(g, cap):
    t0, t1 = g[g.t == 0], g[g.t == 1]
    b_dir = float((t0.dflagS0 - t0.dflagR0).mean())
    r2l, dbar = float(t0.r2G0.mean()), float(t0.dbar.mean())
    nu = float(t1.dvar_fit.mean())
    obs = float(np.log(t1.R0).mean() - np.log(t0.R0).mean())
    normal = iota_for(cap) * b_dir * r2l / (dbar * (1 - dbar) * float(t1.sd_z.mean()))
    emp = float(t1.hz.mean()) * b_dir * r2l / nu
    return dict(b_measured=b_dir, r2G=r2l, lnR_obs=obs, pred_normal=float(normal), pred_empirical_hazard=float(emp),
                R_cycle0=float(t0.R0.mean()))


def e11(R):
    df = load("E11")
    if df is None:
        return
    inside, outside = [], []
    for tag, g in df.groupby("tag"):
        cap = float(tag.split("_")[1]) if tag.startswith("capacity_") else 0.15
        row = dict(config=tag, **_pred_row(g, cap))
        (outside if tag.startswith("outside_") else inside).append(row)
    R["E11_misspecification"] = inside
    R["E11_outside_scope"] = outside
    T = pd.DataFrame(inside)
    for key, col in (("normal", "pred_normal"), ("empirical", "pred_empirical_hazard")):
        x, y = T.lnR_obs.values, T[col].values
        R[f"E11_fit_{key}"] = dict(r2=r2(x, y), mean_abs_err=float(np.abs(x - y).mean()), max_abs_err=float(np.abs(x - y).max()), n=int(len(T)))
    rel = (T.pred_empirical_hazard - T.lnR_obs).abs() / T.lnR_obs.abs()
    R["E11_relative_error"] = dict(median=float(rel.median()), max=float(rel.max()), worst_config=str(T.config.iloc[int(rel.values.argmax())]))


def e14(R):
    """Hold-out test: configurations and seeds not used elsewhere; the formula is the one fixed before the run."""
    df = load("E14")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        cap = float(tag.split("_c")[1].split("_")[0])
        rows.append(dict(config=tag, cap=cap, **_pred_row(g, cap)))
    T = pd.DataFrame(rows)
    x, y = T.lnR_obs.values, T.pred_empirical_hazard.values
    rel = (T.pred_empirical_hazard - T.lnR_obs).abs() / T.lnR_obs.abs()
    R["E14_holdout"] = dict(rows=rows, n=int(len(T)), r2=r2(x, y), mean_abs_err=float(np.abs(x - y).mean()), max_abs_err=float(np.abs(x - y).max()),
                            slope=float(np.polyfit(y, x, 1)[0]), median_rel_err=float(rel.median()), max_rel_err=float(rel.max()),
                            r2_normal=r2(x, T.pred_normal.values), obs_min=float(x.min()), obs_max=float(x.max()))


def e13(R):
    df = load("E13")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        t0, t1 = g[g.t == 0], g[g.t == 1]
        b = float((t0.dflagS0 - t0.dflagR0).mean())
        nu = float(t1.dvar_fit.mean())
        hz = float(t1.hz.mean())
        obs = float(np.log(t1.R0).mean() - np.log(t0.R0).mean())
        rows.append(dict(config=tag, b_measured=b, r2_linear=float(t0.r2G0.mean()), r2_nonlinear=float(t0.r2G_nl0.mean()), lnR_obs=obs,
                         pred_linear=hz * b * float(t0.r2G0.mean()) / nu,
                         pred_nonlinear=hz * b * float(t0.r2G_nl0.mean()) / nu))
    R["E13_model_class"] = rows


# ------------------------------------------------------------------ nulls, persistence and a confidence-interval monitor
def e12(R):
    df = load("E12")
    if df is None:
        return
    rows, pers = [], []
    for tag, g in df.groupby("tag"):
        seeds = int(g.seed.nunique())
        if tag.startswith("cell_"):
            n = int(tag.split("_n")[1])
            single = int(g.groupby("seed").apply(lambda q: declared(q, "R0"), include_groups=False).sum())
            cell = int(g.groupby("seed").apply(lambda q: declared(q, "Rx"), include_groups=False).sum())
            rows.append(dict(setting=tag, n=n, single_attr_false_alarm=single / seeds, single_ci=wilson(single, seeds),
                             cell_false_alarm=cell / seeds, cell_ci=wilson(cell, seeds), cell_size=float(g.nX.mean()), runs=seeds))
            continue
        lam, cap = float(tag.split("_")[0][1:]), float(tag.split("_c")[1])
        one = int(g.groupby("seed").apply(lambda q: bool((q.R0.values > 1 + EPS).any()), include_groups=False).sum())
        two = int(g.groupby("seed").apply(lambda q: declared(q, "R0"), include_groups=False).sum())
        dri = int(g.groupby("seed").apply(lambda q: declared(q, "R0", float(q.R_base.iloc[0])), include_groups=False).sum())

        def ci_monitor(q):
            q = q.sort_values("t")
            nS = q.nS0.values
            nR = 20000 - nS
            rS, rR = q.rS0.values, q.rR0.values
            se = np.sqrt((1 - rS) / (nS * rS) + (1 - rR) / (nR * rR))
            lower = np.log(q.R0.values) - 1.645 * se
            return not np.isnan(first_run(np.exp(lower), 1 + EPS, 2))
        cim = int(g.groupby("seed").apply(ci_monitor, include_groups=False).sum())
        rows.append(dict(setting=tag, proxy=lam, capacity=cap, single_exceedance=one / seeds, single_ci=wilson(one, seeds),
                         two_consecutive=two / seeds, two_ci=wilson(two, seeds), drift_large_baseline=dri / seeds,
                         drift_ci=wilson(dri, seeds), binomial_ci_monitor=cim / seeds, binomial_ci_monitor_ci=wilson(cim, seeds),
                         sd_R=float(g.R0.std()), runs=seeds))
        for k in (1, 2, 3, 4):
            fa = int(g.groupby("seed").apply(lambda q, k=k: declared(q, "R0", 1.0, k), include_groups=False).sum())
            pers.append(dict(setting=tag, proxy=lam, capacity=cap, k=k, false_alarm=fa / seeds, ci=wilson(fa, seeds)))
    R["E12_nulls"] = rows
    R["E12_persistence_null"] = pers
    e2 = load("E2")
    out = []
    for tag in ("b0.15_l1.0", "b0.1_l1.0", "b0.05_l1.5"):
        g = e2[e2.tag == tag]
        for k in (1, 2, 3, 4):
            det = g.groupby("seed").apply(lambda q, k=k: first_run(q.sort_values("t").R0.values, 1 + EPS, k), include_groups=False)
            out.append(dict(config=tag, k=k, power=float(det.notna().mean()), median_cycle=float(np.nanmedian(det)) if det.notna().any() else np.nan))
    R["E12_persistence_power"] = out
    pw = []
    for tag in ("b0.15_l1.0", "b0.1_l1.0", "b0.05_l1.0", "b0.05_l1.5", "b0.15_l0.6"):
        g = e2[e2.tag == tag]

        def ci_det(q):
            q = q.sort_values("t")
            nS = q.nS0.values
            nR = 20000 - nS
            se = np.sqrt((1 - q.rS0.values) / (nS * q.rS0.values) + (1 - q.rR0.values) / (nR * q.rR0.values))
            return first_run(np.exp(np.log(q.R0.values) - 1.645 * se), 1 + EPS, 2)
        det = g.groupby("seed").apply(ci_det, include_groups=False)
        pw.append(dict(config=tag, power=float(det.notna().mean()), median_cycle=float(np.nanmedian(det)) if det.notna().any() else np.nan))
    R["E12_ci_monitor_power"] = pw


def e4d(R):
    df = load("E4D")
    if df is None:
        return
    rows = []
    for tag, g in df.groupby("tag"):
        m = g.groupby("t").mean(numeric_only=True)
        det = g.groupby("seed").apply(lambda q: first_run2(q.sort_values("t").R0.values, 1 + EPS), include_groups=False)
        t0 = g[g.t == 0]
        rows.append(dict(deference=float(tag[1:]), R_t1=float(m.R0.loc[1]), R_t39=float(m.R0.loc[39]), mean_R_30_39=float(m.R0.loc[30:39].mean()),
                         detect_median=float(np.nanmedian(det)) if det.notna().any() else np.nan, p_detect=float(det.notna().mean()),
                         sd_z_t39=float(m.sd_z.loc[39]), label_diff_t0=float((t0.dflagS0 - t0.dflagR0).mean())))
    R["E4D_deference"] = rows


def identification(R):
    df = load("E2")
    if df is None:
        return
    out = []
    for b, lam in ((0.15, 1.0), (0.10, 1.0), (0.05, 1.5)):
        g = df[df.tag == f"b{b}_l{lam}"]
        e_dir, e_inv, e_emp = [], [], []
        for seed, q in g.groupby("seed"):
            q = q.sort_values("t")
            t0, t1 = q.iloc[0], q.iloc[1]
            lnR1 = np.log(t1.R0)
            e_dir.append(t0.dflagS0 - t0.dflagR0)
            e_inv.append(lnR1 * t1.dvar_fit * t1.sd_z / (IOTA * t0.r2G0))
            e_emp.append(lnR1 * t1.dvar_fit / (t1.hz * t0.r2G0))

        def summ(x):
            x = np.asarray(x)
            return dict(mean=float(x.mean()), sd=float(x.std()), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                        share_within_30pct=float(np.mean(np.abs(x - b) <= 0.3 * b)))
        out.append(dict(true_b=b, proxy=lam, direct=summ(e_dir), inversion_normal=summ(e_inv), inversion_empirical_hazard=summ(e_emp)))
    R["Penalty_estimators"] = out


def formula_variants(R):
    rows = []
    for name in ("E2", "E9"):
        df = load(name)
        if df is None:
            continue
        for tag, g in df.groupby("tag"):
            b = float(tag.split("_b")[-1]) if name == "E9" else float(tag.split("_")[0][1:])
            if b == 0 or (name == "E2" and float(tag.split("_l")[-1]) == 0):
                continue
            t0, t1 = g[g.t == 0], g[g.t == 1]
            r2l, dbar, dvar = t0.r2G0.mean(), t0.dbar.mean(), t0.dvar.mean()
            sdz = t1.sd_z.mean()
            fv = dbar * (1 - dbar)
            rows.append(dict(exp=name, tag=tag, obs=float(np.log(t1.R0).mean()),
                             normal=float(IOTA * b * r2l / (fv * sdz)), normal_jensen=float(IOTA * b * r2l / (dvar * sdz)),
                             empirical=float(t1.hz.mean() * b * r2l / fv), empirical_jensen=float(t1.hz.mean() * b * r2l / dvar),
                             empirical_fitted_variance=float(t1.hz.mean() * b * r2l / t1.dvar_fit.mean()),
                             empirical_ex_ante=float(t0.hz.mean() * b * r2l / fv),
                             empirical_ex_ante_fv=float(t0.hz.mean() * b * r2l / t1.dvar_fit.mean()),
                             empirical_bw05=float(t1.hz05.mean() * b * r2l / t1.dvar_fit.mean()), empirical_bw20=float(t1.hz20.mean() * b * r2l / t1.dvar_fit.mean())))
    T = pd.DataFrame(rows)
    T.assign(empirical=T.empirical_fitted_variance)[["exp", "tag", "obs", "normal", "empirical"]].to_csv(os.path.join("results", "formula_points_E2_E9.csv"), index=False)
    out = {}
    for col in ("normal", "normal_jensen", "empirical", "empirical_jensen", "empirical_fitted_variance", "empirical_ex_ante", "empirical_ex_ante_fv", "empirical_bw05", "empirical_bw20"):
        x, y = T.obs.values, T[col].values
        out[col] = dict(r2=r2(x, y), mean_abs_err=float(np.abs(x - y).mean()), slope=float(np.polyfit(y, x, 1)[0]), n=int(len(T)))
    g = load("E2")
    g = g[g.tag == "b0.15_l1.0"]
    per = {"normal": [], "empirical": [], "ex_ante": []}
    for seed, q in g.groupby("seed"):
        q = q.sort_values("t")
        t0, t1 = q.iloc[0], q.iloc[1]
        lnr = float(np.log(t1.R0))
        fv = t0.dbar * (1 - t0.dbar)
        per["normal"].append(lnr - IOTA * 0.15 * t0.r2G0 / (fv * t1.sd_z))
        per["empirical"].append(lnr - t1.hz * 0.15 * t0.r2G0 / t1.dvar_fit)
        per["ex_ante"].append(lnr - t0.hz * 0.15 * t0.r2G0 / fv)
    for key, arr in per.items():
        arr = np.asarray(arr)
        name = {"normal": "per_run_error_reference", "empirical": "per_run_error_reference_empirical", "ex_ante": "per_run_error_reference_ex_ante"}[key]
        out[name] = dict(mean=float(arr.mean()), sd=float(arr.std()), within_0_10=float(np.mean(np.abs(arr) <= 0.10)))
    R["First_step_formula_variants"] = out


def monitors_extra(R):
    """Adds a drift monitor with a large reference sample and Wilson intervals to the monitor comparison."""
    spec = [("E1", ["n20000"]), ("E3", ["dq0.05_b0.0"]), ("E8", ["b0.1"]), ("E8", ["b0.15"]), ("E2", ["b0.15_l1.0"])]
    rows = R.get("monitor_comparison", [])
    for (name, tags), row in zip(spec, rows):
        df = load(name)
        g = df[df.tag.isin(tags)]
        n = int(g.seed.nunique())
        lev = int(g.groupby("seed").apply(lambda q: declared(q, "R0"), include_groups=False).sum())
        dri = int(g.groupby("seed").apply(lambda q: declared(q, "R0", float(q.sort_values("t").R0.iloc[0])), include_groups=False).sum())
        big = int(g.groupby("seed").apply(lambda q: declared(q, "R0", float(q.R_base.iloc[0])), include_groups=False).sum())
        row.update(runs=n, level_ci=wilson(lev, n), drift_ci=wilson(dri, n), drift_large_baseline=big / n, drift_large_ci=wilson(big, n))


def run_all(R):
    for f in (e10, e11, e13, e14, e12, e4d, identification, formula_variants, monitors_extra):
        f(R)
