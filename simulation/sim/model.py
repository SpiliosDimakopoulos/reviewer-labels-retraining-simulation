import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class Model:
    """Thin wrapper so that logistic regression and gradient boosting share one interface."""

    def __init__(self, kind, est):
        self.kind = kind
        self.est = est

    @property
    def coef_(self):
        if self.kind != "logit":
            raise AttributeError("no coefficients for this model kind")
        return self.est.coef_


def fit(X, d, C=1.0, kind="logit"):
    if kind == "logit":
        est = LogisticRegression(C=C, max_iter=300)
    elif kind == "gbm":
        est = HistGradientBoostingClassifier(max_iter=80, max_depth=3, learning_rate=0.1, random_state=0)
    else:
        raise ValueError(kind)
    est.fit(X, d)
    return Model(kind, est)


def prob(m, X):
    return m.est.predict_proba(X)[:, 1]


def logit(m, X):
    if m.kind == "logit":
        return m.est.decision_function(X)
    p = np.clip(prob(m, X), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
