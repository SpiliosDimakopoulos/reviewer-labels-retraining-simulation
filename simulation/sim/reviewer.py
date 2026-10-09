import numpy as np


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def review(rng, y, G, s, P_first, groups, gint=0.0, anchor=0.0, blind=False, subst=1.0,
           pi1=0.8, pi0=0.1, mode="additive", defer_w=0.0):
    """Reviewer decision d (1 = flagged). Returns (d, p) where p is the reviewer's own flag probability.

    Default (mode = 'additive'):
      P(d=1) = pi(y*) + sum_j b_j * penalty_j + gint * G0 * G1 + anchor * (s - mean(s)), clipped to [0, 1].
    penalty_j is the member indicator G_j for a reviewer who sees the protected attribute. A blinded reviewer
    applies the penalty to the first visible proxy of the group, scaled so that the group-average penalty equals
    b_j * subst.
    Other penalty forms, used to test the derivation under misspecification:
      'logit'    the penalty shifts the log-odds of a flag by 4 b (so the probability gap depends on the base level)
      'neg_only' the penalty applies only when the true status is zero (a pure false-flag bias)
      'hetero'   the penalty of each case is drawn from an exponential distribution with mean b
    anchor couples the decision linearly to the model score (automation bias).
    defer_w is the probability that the reviewer copies the model's recommendation (flag the top share of
    scores, with the share equal to the reviewer's own average flag rate) instead of deciding.
    """
    base = np.where(y == 1, pi1, pi0).astype(float)
    p = base.copy()
    pen = np.zeros(len(y))
    for j, g in enumerate(groups):
        lam = g["lam"][0] if isinstance(g["lam"], (list, tuple)) else g["lam"]
        if blind:
            pen += g["b"] * subst * (P_first[:, j] / lam if lam > 0 else 0.0)
        elif mode == "hetero":
            pen += G[:, j] * rng.exponential(g["b"], len(y))
        elif mode == "neg_only":
            pen += g["b"] * G[:, j] * (y == 0)
        else:
            pen += g["b"] * G[:, j]
    if mode == "logit" and not blind:
        k = 4.0
        gm = sum(g["b"] * G[:, j] for j, g in enumerate(groups))
        p = _expit(np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6))) + k * gm)
    else:
        p = p + pen
    if gint and G.shape[1] >= 2:
        p = p + gint * G[:, 0] * G[:, 1]
    if anchor:
        p = p + anchor * (s - s.mean())
    p = np.clip(p, 0.0, 1.0)
    d = (rng.random(len(y)) < p).astype(int)
    if defer_w > 0:
        rate = float(p.mean())
        rec = (s >= np.quantile(s, 1 - rate)).astype(int)
        use = rng.random(len(y)) < defer_w
        d = np.where(use, rec, d)
    return d, p
