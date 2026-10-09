import numpy as np


def _noise(rng, n, kind):
    """Unit-variance noise for proxy features."""
    if kind == "gauss":
        return rng.standard_normal(n)
    if kind == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)
    if kind == "t3":
        return rng.standard_t(3, n) / np.sqrt(3.0)
    raise ValueError(kind)


def make_cohort(rng, n, groups, q0=0.3, dq=0.0, mu=0.8, n_feat=5, noise="gauss", legit_shift=0.0):
    """Draw one cohort.

    groups: list of dicts with keys
      share  P[member]
      lam    proxy shift, a float or a list of floats (one proxy feature per entry)
      rho    optional correlation between the noise terms of the proxies of the same group
      pvar   optional extra spread of the proxy for members (noise is multiplied by 1 + pvar for members);
             with lam = 0 this gives a proxy that is invisible to linear regression of the group on the features
    Group memberships are independent Bernoulli draws.
    The latent true flag y* has base rate q0 + dq * G[:, 0], so dq != 0 gives unequal base rates.
    Legitimate features depend on y* only, except that legit_shift adds a shift of the first legitimate feature
    for members of the first group (a legitimate feature that differs by group).
    The protected attributes are never returned as features.
    noise selects the proxy noise distribution (gauss, laplace, t3), always with unit variance.

    Returns G, y, X (legitimate features followed by all proxy features) and first_col, the column
    index of the first proxy of each group.
    """
    m = len(groups)
    G = np.column_stack([(rng.random(n) < g["share"]).astype(int) for g in groups])
    y = (rng.random(n) < (q0 + dq * G[:, 0])).astype(int)
    X = y[:, None] * mu + rng.standard_normal((n, n_feat))
    if legit_shift:
        X[:, 0] += legit_shift * G[:, 0]
    cols, first_col, offset = [], [], n_feat
    for j, g in enumerate(groups):
        lams = g["lam"] if isinstance(g["lam"], (list, tuple)) else [g["lam"]]
        rho = g.get("rho", 0.0)
        pvar = g.get("pvar", 0.0)
        shared = _noise(rng, n, noise) if noise != "gauss" else rng.standard_normal(n)
        first_col.append(offset)
        for lam in lams:
            e = _noise(rng, n, noise) if noise != "gauss" else rng.standard_normal(n)
            e = np.sqrt(1 - rho) * e + np.sqrt(rho) * shared
            cols.append(lam * G[:, j] + e * (1.0 + pvar * G[:, j]))
            offset += 1
    return G, y, np.column_stack([X] + cols), first_col
