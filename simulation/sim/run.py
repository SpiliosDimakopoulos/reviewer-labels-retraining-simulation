import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .model import fit, logit, prob
from .population import make_cohort
from .reviewer import review
from .routing import escalate

DEFAULT_GROUPS = [dict(share=0.3, lam=1.0, b=0.10)]
N_FEAT = 5
INTERVENTIONS = ("none", "freeze", "rollback_freeze", "rollback_continue", "drop_proxies",
                 "verified_labels", "group_offset", "label_correction")


def _rate(mask_group, esc):
    return esc[mask_group].mean() if mask_group.any() else np.nan


def r2_group(g, X):
    """R-squared of the group indicator on the model's feature matrix (ordinary least squares)."""
    A = np.column_stack([np.ones(len(g)), X])
    beta, *_ = np.linalg.lstsq(A, g.astype(float), rcond=None)
    resid = g - A @ beta
    return float(1 - resid.var() / g.var())


def r2_group_nonlinear(g, X, seed=0):
    """Out-of-sample R-squared of the group indicator on the features with gradient boosting (half/half split)."""
    h = len(g) // 2
    reg = HistGradientBoostingRegressor(max_iter=60, max_depth=3, learning_rate=0.1, random_state=seed)
    reg.fit(X[:h], g[:h].astype(float))
    pred = reg.predict(X[h:])
    gt = g[h:].astype(float)
    return float(1 - ((gt - pred) ** 2).sum() / ((gt - gt.mean()) ** 2).sum())


def _offset_for_parity(z, g, cap, iters=40):
    """Group-specific score offset d such that the group's escalation rate equals that of the others."""
    lo, hi = -3.0, 3.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        zz = z + mid * g
        esc = zz >= np.quantile(zz, 1 - cap)
        diff = esc[g == 1].mean() - esc[g == 0].mean()
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _hazard(z, cap, frac):
    thr = np.quantile(z, 1 - cap)
    h = frac * z.std()
    return float(np.mean(np.abs(z - thr) < h) / (2 * h) / cap)


def run(groups=None, q0=0.3, dq=0.0, anchor=0.0, n=20000, T=24, k=4, cap=0.15,
        seed=0, n0=20000, f0="clean", blind=False, subst=1.0, gint=0.0, mu=0.8, C=1.0,
        noise="gauss", reviewer_mode="additive", pi1=0.8, pi0=0.1, defer_w=0.0,
        intervene_at=None, intervention="none", base_mult=5, model_kind="logit",
        legit_shift=0.0, nl_r2=False):
    """One deployment of T retraining cycles. Returns a DataFrame with one row per cycle.

    f0 = 'clean': the first model is trained on the true flags of an unselected cohort. It represents a
    model that passed, or was repaired by, a pre-deployment audit.
    f0 = 'reviewed': the first model is trained on historic reviewer labels of an unselected cohort.
    intervention (applied at the start of cycle intervene_at):
      freeze             stop retraining, keep the current model
      rollback_freeze    restore the first model and stop retraining
      rollback_continue  restore the first model and keep retraining on reviewer labels
      drop_proxies       remove the proxy features and refit on the labels already collected
      verified_labels    from this cycle on, train on the verified true status of escalated cases
      group_offset       a one-time group-specific score offset that equalises escalation, kept fixed
      label_correction   from this cycle on, train on reviewer labels corrected for the measured group
                         differential in the flag rate (flags of group members are removed at random)
    """
    groups = groups or DEFAULT_GROUPS
    m = len(groups)
    rng = np.random.default_rng(seed)
    rng_extra = np.random.default_rng(10_000_000 + seed)  # separate stream: extras never change the main draws

    G0, y0, X0, fc0 = make_cohort(rng, n0, groups, q0, dq, mu, noise=noise, legit_shift=legit_shift)
    if f0 == "clean":
        labels0 = y0
    else:
        labels0 = review(rng, y0, G0, np.full(n0, 0.5), X0[:, fc0], groups, gint=gint, blind=blind, subst=subst,
                         pi1=pi1, pi0=pi0, mode=reviewer_mode)[0]
    n_all = X0.shape[1]
    all_cols = list(range(n_all))
    legit_cols = list(range(N_FEAT))
    cols = all_cols
    model = fit(X0[:, cols], labels0, C, model_kind)
    model0 = model

    # reference sample for a drift monitor: ratio of the first model on a large independent cohort
    Gb, yb, Xb, _ = make_cohort(rng_extra, base_mult * n, groups, q0, dq, mu, noise=noise, legit_shift=legit_shift)
    sb = prob(model0, Xb)
    eb = escalate(sb, cap)
    r_base = _rate(Gb[:, 0] == 1, eb) / _rate(Gb[:, 0] == 0, eb)

    window = []
    rows = []
    frozen = False
    label_mode = "reviewer"
    offset = 0.0
    for t in range(T):
        if intervene_at is not None and t == intervene_at and intervention != "none":
            if intervention == "freeze":
                frozen = True
            elif intervention == "rollback_freeze":
                model, cols, frozen = model0, all_cols, True
            elif intervention == "rollback_continue":
                model, cols = model0, all_cols
            elif intervention == "drop_proxies":
                cols = legit_cols
                Xw = np.vstack([w[0] for w in window])[:, cols]
                dw = np.concatenate([w[1] for w in window])
                model = fit(Xw, dw, C, model_kind)
            elif intervention == "verified_labels":
                label_mode, window = "verified", []
            elif intervention == "label_correction":
                label_mode, window = "corrected", []
        G, y, X, fc = make_cohort(rng, n, groups, q0, dq, mu, noise=noise, legit_shift=legit_shift)
        s = prob(model, X[:, cols])
        z = logit(model, X[:, cols])
        if intervention == "group_offset" and intervene_at is not None and t == intervene_at:
            offset = _offset_for_parity(z, G[:, 0], cap)
        zr = z + offset * G[:, 0] if offset else z
        esc = escalate(zr if offset else s, cap)
        d_rev, p_own = review(rng, y[esc], G[esc], s[esc], X[esc][:, fc], groups, gint=gint,
                              anchor=anchor, blind=blind, subst=subst, pi1=pi1, pi0=pi0,
                              mode=reviewer_mode, defer_w=defer_w)
        if label_mode == "verified":
            d_label = y[esc]
        elif label_mode == "corrected":
            ge = G[esc][:, 0] == 1
            delta = d_rev[ge].mean() - d_rev[~ge].mean() if ge.any() and (~ge).any() else 0.0
            d_label = d_rev.copy()
            if delta > 0 and d_rev[ge].mean() > 0:
                drop = rng_extra.random(len(d_rev)) < delta / d_rev[ge].mean()
                d_label[ge & drop & (d_rev == 1)] = 0
        else:
            d_label = d_rev
        row = dict(t=t, sd_z=float(z.std()),
                   wx_norm=float(np.linalg.norm(model.coef_[0][:N_FEAT])) if model.kind == "logit" else np.nan,
                   dbar=float(d_rev.mean()), dvar=float((p_own * (1 - p_own)).mean()), n_esc=int(esc.sum()),
                   sat_frac=float((s > 0.9999).mean()), tie_frac=float((s == s[esc].min()).mean()),
                   R_base=float(r_base), frozen=int(frozen), esc_tp_per_1000=float(1000 * (y[esc] == 1).sum() / n))
        row["hz"] = _hazard(zr, cap, 0.1)  # estimated hazard of the score at the cut-off, per logit unit
        row["hz05"] = _hazard(zr, cap, 0.05)
        row["hz20"] = _hazard(zr, cap, 0.2)
        if window:
            pf = prob(model, window[-1][0][:, cols])
            row["dvar_fit"] = float((pf * (1 - pf)).mean())
        else:
            row["dvar_fit"] = np.nan
        ff = np.zeros(n)
        ff[np.flatnonzero(esc)] = ((d_rev == 1) & (y[esc] == 0)).astype(float)
        for j in range(m):
            inj, out = G[:, j] == 1, G[:, j] == 0
            rS, rR = _rate(inj, esc), _rate(out, esc)
            row[f"R{j}"] = rS / rR
            row[f"rS{j}"] = rS
            row[f"rR{j}"] = rR
            used = fc[j] in cols and model.kind == "logit"
            row[f"coef_p{j}"] = float(model.coef_[0][cols.index(fc[j])]) if used else np.nan
            row[f"r2G{j}"] = r2_group(G[:, j], X[:, cols])
            if nl_r2:
                row[f"r2G_nl{j}"] = r2_group_nonlinear(G[:, j], X[:, cols], seed)
            row[f"ffS{j}"] = ff[inj].mean()
            row[f"ffR{j}"] = ff[out].mean()
            row[f"nS{j}"] = int(inj.sum())
            gm = G[esc][:, j] == 1
            row[f"dflagS{j}"] = float(d_rev[gm].mean()) if gm.any() else np.nan
            row[f"dflagR{j}"] = float(d_rev[~gm].mean()) if (~gm).any() else np.nan
        if m >= 2:
            both = (G[:, 0] == 1) & (G[:, 1] == 1)
            none = (G[:, 0] == 0) & (G[:, 1] == 0)
            rS, rR = _rate(both, esc), _rate(none, esc)
            row["Rx"] = rS / rR if rR else np.nan
            row["nX"] = int(both.sum())
        rows.append(row)
        window.append((X[esc], d_label))
        window = window[-k:]
        if not frozen:
            Xw = np.vstack([w[0] for w in window])[:, cols]
            dw = np.concatenate([w[1] for w in window])
            model = fit(Xw, dw, C, model_kind)
    return pd.DataFrame(rows)
