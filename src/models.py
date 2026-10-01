"""Model zoo, imbalance strategies, calibration and cost-aware evaluation."""
from __future__ import annotations

import warnings

import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import FOCAL_ALPHA, FOCAL_GAMMA, N_JOBS, REVIEW_COST, SEED


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Focal loss as a custom XGBoost objective
# ---------------------------------------------------------------------------
def focal_loss(z, y, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """Mean focal loss, FL = -alpha_t (1 - p_t)^gamma log(p_t), as a function of the margin z."""
    p = np.clip(sigmoid(z), 1e-7, 1 - 1e-7)
    pt = np.where(y == 1, p, 1 - p)
    at = np.where(y == 1, alpha, 1 - alpha)
    return float(np.mean(-at * (1 - pt) ** gamma * np.log(pt)))


def focal_grad(z, y, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """Analytic d FL / d z."""
    p = np.clip(sigmoid(z), 1e-7, 1 - 1e-7)
    g_pos = alpha * (1 - p) ** gamma * (gamma * p * np.log(p) - (1 - p))
    g_neg = (1 - alpha) * p ** gamma * (p - gamma * (1 - p) * np.log(1 - p))
    return np.where(y == 1, g_pos, g_neg)


def focal_objective(preds, dtrain):
    y = dtrain.get_label()
    g = focal_grad(preds, y)
    eps = 1e-3                                              # numerical hessian, floored for stability
    h = (focal_grad(preds + eps, y) - focal_grad(preds - eps, y)) / (2 * eps)
    return g, np.clip(h, 1e-6, None)


# ---------------------------------------------------------------------------
# XGBoost wrapper (works for logistic and focal objectives alike)
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(max_depth=5, eta=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                  tree_method="hist", max_bin=128, nthread=N_JOBS, seed=SEED, verbosity=0)
N_ROUNDS = 250


class XGBModel:
    def __init__(self, booster: xgb.Booster, strategy: str):
        self.booster, self.strategy = booster, strategy

    def margin(self, X) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)), output_margin=True)

    def predict_proba(self, X) -> np.ndarray:
        return sigmoid(self.margin(X))


def train_xgb(X, y, strategy: str = "none", rounds: int = N_ROUNDS) -> XGBModel:
    """strategy in {none, class_weight, smote, focal}"""
    params = dict(XGB_PARAMS)
    cols = list(X.columns)
    if strategy == "smote":
        X, y = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=SEED).fit_resample(X, y)
        X = X.astype("float32")
    dtrain = xgb.DMatrix(X, label=np.asarray(y), feature_names=cols)
    if strategy == "focal":
        booster = xgb.train(params, dtrain, rounds, obj=focal_objective)
    else:
        params["objective"] = "binary:logistic"
        if strategy == "class_weight":
            params["scale_pos_weight"] = float((np.asarray(y) == 0).sum() / max((np.asarray(y) == 1).sum(), 1))
        booster = xgb.train(params, dtrain, rounds)
    return XGBModel(booster, strategy)


# ---------------------------------------------------------------------------
# Baselines and unsupervised detectors (all return margin-like scores)
# ---------------------------------------------------------------------------
def train_logreg(X, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, C=1.0))
    return m.fit(X, y)


def train_rf(X, y):
    return RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5, n_jobs=N_JOBS,
                                  class_weight="balanced_subsample", random_state=SEED).fit(X, y)


class IForestScorer:
    def __init__(self):
        self.model = IsolationForest(n_estimators=150, max_samples=256, random_state=SEED, n_jobs=1)
        self.train_scores = None

    def fit(self, X):
        self.model.fit(X)
        self.train_scores = np.sort(self.score(X))
        return self

    def score(self, X):
        return -self.model.score_samples(X)

    def percentile(self, s):
        return float(np.searchsorted(self.train_scores, s) / len(self.train_scores))


class AutoencoderScorer:
    """Small MLP autoencoder (sklearn, CPU-light) trained on LEGITIMATE transactions only.
    Anomaly score = log reconstruction error."""

    def __init__(self, max_rows=15000):
        self.scaler = StandardScaler()
        self.net = MLPRegressor(hidden_layer_sizes=(12, 6, 12), activation="relu", batch_size=256,
                                learning_rate_init=1e-3, max_iter=40, random_state=SEED)
        self.max_rows = max_rows

    def _z(self, X):
        return np.clip(self.scaler.transform(X), -5, 5)

    def fit(self, X_legit):
        X_legit = X_legit.iloc[: self.max_rows] if hasattr(X_legit, "iloc") else X_legit[: self.max_rows]
        self.scaler.fit(X_legit)
        Z = self._z(X_legit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.net.fit(Z, Z)
        return self

    def score(self, X):
        Z = self._z(X)
        return np.log1p(((self.net.predict(Z) - Z) ** 2).mean(axis=1))


# ---------------------------------------------------------------------------
# Calibration (Platt scaling on the validation set) -> risk score in [0, 1]
# ---------------------------------------------------------------------------
class Calibrator:
    def fit(self, s_val, y_val):
        s_val = np.asarray(s_val, float)
        self.mu, self.sd = float(s_val.mean()), float(s_val.std() + 1e-9)
        self.lr = LogisticRegression(C=1e3, max_iter=1000).fit(((s_val - self.mu) / self.sd).reshape(-1, 1), y_val)
        return self

    def transform(self, s):
        s = np.asarray(s, float)
        return self.lr.predict_proba(((s - self.mu) / self.sd).reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Metrics & cost
# ---------------------------------------------------------------------------
def total_cost(y, flag, amount, review_cost=REVIEW_COST):
    y, flag = np.asarray(y), np.asarray(flag)
    missed = amount[(y == 1) & (~flag)].sum()          # false negatives cost the full amount
    reviews = review_cost * ((y == 0) & flag).sum()    # false positives cost a manual review
    return float(missed + reviews)


GRID = np.round(np.arange(0.01, 1.0, 0.005), 3)


def best_threshold(p, y, amount):
    costs = [total_cost(y, p >= t, amount) for t in GRID]
    return float(GRID[int(np.argmin(costs))])


def threshold_metrics(p, y, amount, thr):
    y = np.asarray(y)
    flag = p >= thr
    tp = int(((y == 1) & flag).sum()); fp = int(((y == 0) & flag).sum()); fn = int(((y == 1) & ~flag).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(threshold=float(thr), precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn,
                flagged=int(flag.sum()), cost=total_cost(y, flag, amount))


def curve_table(p, y, amount):
    """Precision / recall / cost for each threshold in 0.01 .. 0.99 (for the dashboard slider)."""
    out = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        m = threshold_metrics(p, y, amount, t)
        m["precision"] = m["precision"] if m["flagged"] else None
        out.append(m)
    return out


def rank_metrics(y, score):
    return dict(pr_auc=float(average_precision_score(y, score)), roc_auc=float(roc_auc_score(y, score)))


def bootstrap_pr_auc(y, score, n=200, seed=SEED):
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(average_precision_score(y[i], score[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
