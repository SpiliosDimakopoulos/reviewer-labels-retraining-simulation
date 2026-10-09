"""Tests for the simulation. Run from the simulation folder: python -m pytest tests -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sim.population import make_cohort  # noqa: E402
from sim.reviewer import review  # noqa: E402
from sim.routing import escalate  # noqa: E402
from sim.run import run  # noqa: E402

GROUPS = [dict(share=0.3, lam=1.0, b=0.10)]


def test_population_shares_and_shapes():
    rng = np.random.default_rng(0)
    G, y, X, fc = make_cohort(rng, 50000, GROUPS)
    assert G.shape == (50000, 1) and X.shape == (50000, 6) and fc == [5]
    assert abs(G.mean() - 0.3) < 0.01
    assert abs(y.mean() - 0.3) < 0.01


def test_protected_attribute_is_never_a_feature():
    rng = np.random.default_rng(1)
    G, y, X, fc = make_cohort(rng, 5000, GROUPS)
    for col in range(X.shape[1]):
        assert not np.array_equal(X[:, col], G[:, 0].astype(float))
    # the proxy is informative about the group but does not reproduce it
    c = np.corrcoef(X[:, fc[0]], G[:, 0])[0, 1]
    assert 0.1 < c < 0.9


def test_multi_proxy_columns():
    rng = np.random.default_rng(2)
    groups = [dict(share=0.3, lam=[0.6, 0.6, 0.3], b=0.1)]
    G, y, X, fc = make_cohort(rng, 2000, groups)
    assert X.shape[1] == 5 + 3


def test_routing_hits_capacity():
    rng = np.random.default_rng(3)
    s = rng.random(20000)
    esc = escalate(s, 0.15)
    assert abs(esc.mean() - 0.15) < 0.001


def test_reviewer_penalty_direction():
    rng = np.random.default_rng(4)
    n = 200000
    y = np.zeros(n, dtype=int)
    G = (rng.random(n) < 0.5).astype(int)[:, None]
    P = np.zeros((n, 1))
    d = review(rng, y, G, np.zeros(n), P, GROUPS)[0]
    assert d[G[:, 0] == 1].mean() - d[G[:, 0] == 0].mean() == pytest.approx(0.10, abs=0.01)


def test_null_deployment_has_no_group_gap():
    groups = [dict(share=0.3, lam=1.0, b=0.0)]
    R = np.mean([run(groups=groups, n=10000, T=6, seed=s).R0.values[2:].mean() for s in range(6)])
    assert abs(R - 1.0) < 0.04


def test_clean_first_model_passes_a_deployment_audit():
    R0 = np.mean([run(groups=GROUPS, n=10000, T=2, seed=s).R0.values[0] for s in range(10)])
    assert abs(R0 - 1.0) < 0.05


def test_no_proxy_no_step():
    groups = [dict(share=0.3, lam=0.0, b=0.15)]
    R = np.mean([run(groups=groups, n=10000, T=5, seed=s).R0.values[1:].mean() for s in range(8)])
    assert abs(R - 1.0) < 0.05


def test_step_appears_with_proxy_and_penalty():
    groups = [dict(share=0.3, lam=1.5, b=0.15)]
    R = np.mean([run(groups=groups, n=10000, T=4, seed=s).R0.values[1:].mean() for s in range(6)])
    assert R > 1.2


def test_run_is_reproducible():
    a = run(groups=GROUPS, n=3000, T=3, seed=7)
    b = run(groups=GROUPS, n=3000, T=3, seed=7)
    assert np.allclose(a.R0.values, b.R0.values)


def test_escalation_count_matches_capacity_in_run():
    df = run(groups=GROUPS, n=10000, T=4, seed=3)
    assert (abs(df.n_esc / 10000 - 0.15) < 0.002).all()


def test_stopping_retraining_does_not_remove_the_gap_but_rollback_does():
    g = [dict(share=0.3, lam=1.0, b=0.15)]
    freeze = np.mean([run(groups=g, n=10000, T=10, seed=s, intervene_at=4, intervention="freeze").R0.values[6:].mean() for s in range(4)])
    roll = np.mean([run(groups=g, n=10000, T=10, seed=s, intervene_at=4, intervention="rollback_freeze").R0.values[6:].mean() for s in range(4)])
    cont = np.mean([run(groups=g, n=10000, T=10, seed=s, intervene_at=4, intervention="rollback_continue").R0.values[6:].mean() for s in range(4)])
    assert freeze > 1.12 and cont > 1.12 and roll < 1.08


def test_first_step_formula_matches_simulation():
    g = [dict(share=0.3, lam=1.0, b=0.15)]
    df = pd.concat([run(groups=g, n=20000, T=2, seed=s) for s in range(8)])
    t0, t1 = df[df.t == 0], df[df.t == 1]
    b = (t0.dflagS0 - t0.dflagR0).mean()
    pred = t1.hz.mean() * b * t0.r2G0.mean() / (t0.dbar.mean() * (1 - t0.dbar.mean()))
    assert abs(np.log(t1.R0).mean() - pred) < 0.08
