"""Experiment runner. Usage (from the simulation folder): python experiments.py [E1 E2 ...]

Each experiment is a list of configurations; every configuration is run for several seeds in parallel.
Raw per-cycle output goes to results/raw/<exp>.csv. Aggregation happens in analyze.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from sim.run import run

OUT = os.path.join("results", "raw")
SHARE = 0.3


def g1(b, lam, share=SHARE):
    return [dict(share=share, lam=lam, b=b)]


def configs():
    E = {}
    # E1: null deployment, false-alarm rate of the continuous monitor against cohort size
    E["E1"] = dict(reps=100, cfgs=[dict(tag=f"n{n}", n=n, groups=g1(0.0, 0.6), T=24) for n in (2000, 5000, 10000, 20000)])
    # E2: step size grid (a = 0), including the lambda = 0 control and b = 0 control
    E["E2"] = dict(reps=30, cfgs=[dict(tag=f"b{b}_l{lam}", n=20000, groups=g1(b, lam), T=24)
                                  for b, lam in itertools.product((0.0, 0.05, 0.10, 0.15, 0.20), (0.0, 0.3, 0.6, 1.0, 1.5, 2.0))])
    # E2L: long horizon without reviewer-model coupling, to measure slow creep
    E["E2L"] = dict(reps=20, cfgs=[dict(tag="b0.15_l1.0", n=20000, groups=g1(0.15, 1.0), T=60)])
    # E3: unequal base rates, no reviewer penalty and with a penalty
    E["E3"] = dict(reps=30, cfgs=[dict(tag=f"dq{dq}_b{b}", n=20000, groups=g1(b, 0.6), dq=dq, T=24)
                                  for dq, b in itertools.product((0.0, 0.02, 0.05, 0.10), (0.0, 0.10))])
    # E4: reviewer-model coupling (automation bias)
    E["E4"] = dict(reps=20, cfgs=[dict(tag=f"a{a}", n=20000, groups=g1(0.15, 1.0), anchor=a, T=40)
                                  for a in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)])
    # E4C: does the inflation of score spread under strong coupling come from near-separation? vary regularisation
    E["E4C"] = dict(reps=15, cfgs=[dict(tag=f"a1.5_C{c}", n=20000, groups=g1(0.15, 1.0), anchor=1.5, C=c, T=40)
                                   for c in (1.0, 0.1, 0.01)])
    # E9: several partial proxies for one group, including correlated proxy noise, to test the R-squared form
    multi = {"two_0.6": dict(lam=[0.6, 0.6]), "three_1.0_0.5_0.3": dict(lam=[1.0, 0.5, 0.3]),
             "two_0.8_rho0.5": dict(lam=[0.8, 0.8], rho=0.5), "five_weak_0.3": dict(lam=[0.3] * 5)}
    E["E9"] = dict(reps=30, cfgs=[dict(tag=f"{name}_b{b}", n=20000, groups=[dict(share=SHARE, b=b, **kw)], T=24)
                                  for name, kw in multi.items() for b in (0.10, 0.15)])
    # E5: blinded reviewer with proxy substitution
    E["E5"] = dict(reps=30, cfgs=[dict(tag="unblinded", n=20000, groups=g1(0.15, 1.0), T=24)] +
                   [dict(tag=f"blind_s{s}", n=20000, groups=g1(0.15, 1.0), blind=True, subst=s, T=24)
                    for s in (0.0, 0.25, 0.5, 1.0, 1.5)])
    # E6: intersection of two attributes
    two = lambda gi: [dict(share=0.3, lam=1.0, b=0.10), dict(share=0.3, lam=1.0, b=0.10)]
    E["E6"] = dict(reps=40, cfgs=[dict(tag=f"gint{gi}_n{n}", n=n, groups=two(gi), gint=gi, T=24)
                                  for gi, n in itertools.product((-0.05, 0.0, 0.05, 0.10), (5000, 20000))])
    # E7: training window and routing capacity (selection strength)
    E["E7"] = dict(reps=30, cfgs=[dict(tag=f"k{k}_cap{c}", n=20000, groups=g1(0.15, 1.0), k=k, cap=c, T=40)
                                  for k, c in itertools.product((1, 2, 4, 8, 16), (0.05, 0.15, 0.30))])
    # E8: first model trained on historic reviewer labels instead of clean labels
    E["E8"] = dict(reps=30, cfgs=[dict(tag=f"b{b}", n=20000, groups=g1(b, 1.0), f0="reviewed", T=24) for b in (0.0, 0.05, 0.10, 0.15)])
    # E10: interventions at cycle 6 (first model clean, penalty 0.15, proxy 1.0)
    E["E10"] = dict(reps=30, cfgs=[dict(tag=iv, n=20000, groups=g1(0.15, 1.0), intervene_at=6, intervention=iv, T=24)
                                   for iv in ("none", "freeze", "rollback_freeze", "rollback_continue", "drop_proxies",
                                              "verified_labels", "group_offset", "label_correction")])
    E["E10b"] = dict(reps=30, cfgs=[dict(tag=iv, n=20000, groups=g1(0.10, 1.5), intervene_at=6, intervention=iv, T=24)
                                    for iv in ("none", "freeze", "rollback_freeze", "rollback_continue", "drop_proxies",
                                               "verified_labels", "group_offset", "label_correction")])
    # E10c: label correction by the measured differential when groups differ in base rates (5 points)
    E["E10c"] = dict(reps=30, cfgs=[dict(tag=f"b{b}_{iv}", n=20000, groups=g1(b, 1.0), dq=0.05, intervene_at=6, intervention=iv, T=24)
                                    for b in (0.0, 0.15) for iv in ("none", "label_correction", "verified_labels")])
    # E11: misspecification of the derivation (first two cycles only)
    mis = {"base": {}, "reviewer_logit": dict(reviewer_mode="logit"), "reviewer_false_flag_only": dict(reviewer_mode="neg_only"),
           "reviewer_heterogeneous": dict(reviewer_mode="hetero"), "proxy_laplace": dict(noise="laplace"),
           "proxy_t3": dict(noise="t3"), "capacity_0.05": dict(cap=0.05), "capacity_0.30": dict(cap=0.30),
           "signal_mu_0.4": dict(mu=0.4), "signal_mu_1.2": dict(mu=1.2), "flag_rates_0.6_0.3": dict(pi1=0.6, pi0=0.3),
           "flag_rates_0.9_0.05": dict(pi1=0.9, pi0=0.05), "base_rate_gap_0.05": dict(dq=0.05)}
    cfgs = []
    for name, kw in mis.items():
        b = 0.20 if name == "reviewer_logit" else 0.15
        cfgs.append(dict(tag=name, n=20000, groups=g1(b, 1.0), T=4, **kw))
    cfgs += [dict(tag="share_0.1", n=20000, groups=g1(0.15, 1.0, share=0.1), T=4),
             dict(tag="share_0.5", n=20000, groups=g1(0.15, 1.0, share=0.5), T=4),
             dict(tag="outside_inherited_first_model", n=20000, groups=g1(0.15, 1.0), f0="reviewed", T=4),
             dict(tag="outside_deference_0.5", n=20000, groups=g1(0.15, 1.0), defer_w=0.5, T=4),
             dict(tag="outside_legit_feature_shift_0.5", n=20000, groups=g1(0.15, 1.0), legit_shift=0.5, T=4)]
    E["E11"] = dict(reps=40, cfgs=cfgs)
    # E12: null deployments across proxy strength and capacity, and small intersection cells
    E["E12"] = dict(reps=100, cfgs=[dict(tag=f"l{lam}_c{c}", n=20000, groups=g1(0.0, lam), cap=c, T=24)
                                    for lam in (0.6, 1.0, 2.0) for c in (0.05, 0.15, 0.30)] +
                    [dict(tag=f"cell_n{n}", n=n, groups=[dict(share=0.3, lam=1.0, b=0.0), dict(share=0.3, lam=1.0, b=0.0)], T=24)
                     for n in (5000, 20000)])
    # E4D: deference mixture instead of linear anchoring
    E["E4D"] = dict(reps=30, cfgs=[dict(tag=f"w{w}", n=20000, groups=g1(0.15, 1.0), defer_w=w, T=40) for w in (0.0, 0.25, 0.5, 0.75)])
    # E13: model class and proxy structure (linear R-squared can miss nonlinear group information)
    cfgs = []
    for kind in ("logit", "gbm"):
        for name, grp in (("mean_shift", dict(share=SHARE, lam=1.0, b=0.15)),
                          ("variance_shift", dict(share=SHARE, lam=0.0, pvar=1.0, b=0.15)),
                          ("both", dict(share=SHARE, lam=1.0, pvar=1.0, b=0.15))):
            cfgs.append(dict(tag=f"{kind}_{name}", n=20000, groups=[grp], model_kind=kind, nl_r2=True, T=3))
    E["E13"] = dict(reps=20, cfgs=cfgs)
    # E14: hold-out test of the frozen formula. Configurations are drawn at random from the ranges of the derivation
    # (penalty, proxy strength, group share, capacity, signal) with a fixed generator, and seeds that no other experiment uses.
    import numpy as np
    rg = np.random.default_rng(20261008)
    cfgs = []
    for i in range(24):
        b, lam = float(rg.uniform(0.04, 0.22)), float(rg.uniform(0.2, 2.2))
        share, cap, mu = float(rg.uniform(0.15, 0.45)), float(rg.choice([0.10, 0.20, 0.25])), float(rg.uniform(0.5, 1.1))
        cfgs.append(dict(tag=f"h{i:02d}_b{b:.3f}_l{lam:.2f}_s{share:.2f}_c{cap:.2f}_m{mu:.2f}", n=20000, groups=g1(b, lam, share), cap=cap, mu=mu, T=4))
    E["E14"] = dict(reps=20, seed0=5000, cfgs=cfgs)
    return E


def job(args):
    exp, cfg, seed = args
    kw = {k: v for k, v in cfg.items() if k != "tag"}
    df = run(seed=seed, **kw)
    df.insert(0, "seed", seed)
    df.insert(0, "tag", cfg["tag"])
    df.insert(0, "exp", exp)
    return df


def main(names):
    os.makedirs(OUT, exist_ok=True)
    E = configs()
    names = names or list(E)
    for name in names:
        t0 = time.time()
        spec = E[name]
        jobs = [(name, c, s) for c in spec["cfgs"] for s in range(spec.get("seed0", 0), spec.get("seed0", 0) + spec["reps"])]
        with ProcessPoolExecutor(max_workers=4) as ex:
            parts = list(ex.map(job, jobs, chunksize=2))
        pd.concat(parts).to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"{name}: {len(jobs)} runs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
