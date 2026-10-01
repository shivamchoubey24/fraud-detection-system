"""Robust comparison of imbalance strategies: repeat the whole experiment on several independent
synthetic datasets (different seeds) and report mean +/- std.  A single split with ~100 test frauds is
too noisy to rank strategies whose PR-AUC differs by 0.003, so this is the number to quote.

    python -m src.compare_imbalance            # 5 seeds, about 1-2 minutes
"""
from __future__ import annotations

import argparse

import pandas as pd

from config import N_TXN, REPORTS, TRAIN_FRAC, VAL_FRAC
from src.data import generate_synthetic
from src.features import build_feature_frame
from src.models import Calibrator, best_threshold, logit, rank_metrics, threshold_metrics, train_logreg, train_rf, train_xgb

STRATEGIES = {"none": "XGBoost (no handling)", "class_weight": "XGBoost + class weights",
              "smote": "XGBoost + SMOTE", "focal": "XGBoost + focal loss"}


def run(seeds=(11, 22, 33, 44, 55), n_txn=N_TXN):
    recs = []
    for seed in seeds:
        df = generate_synthetic(n_txn, seed=seed)
        X, _ = build_feature_frame(df)
        y, amt = df["is_fraud"].to_numpy(), df["amount"].to_numpy()
        n = len(df); i1, i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
        Xtr, Xva, Xte = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
        ytr, yva, yte = y[:i1], y[i1:i2], y[i2:]
        ava, ate = amt[i1:i2], amt[i2:]
        scorers = {"Logistic Regression": lambda m, X_: logit(m.predict_proba(X_)[:, 1]),
                   "Random Forest": lambda m, X_: logit(m.predict_proba(X_)[:, 1])}
        models = {"Logistic Regression": train_logreg(Xtr, ytr), "Random Forest": train_rf(Xtr, ytr)}
        results = {k: (m, scorers[k]) for k, m in models.items()}
        for key, name in STRATEGIES.items():
            results[name] = (train_xgb(Xtr, ytr, key), lambda m, X_: m.margin(X_))
        for name, (m, fn) in results.items():
            sv, st = fn(m, Xva), fn(m, Xte)
            cal = Calibrator().fit(sv, yva)
            thr = best_threshold(cal.transform(sv), yva, ava)
            tm = threshold_metrics(cal.transform(st), yte, ate, thr)
            recs.append({"seed": seed, "model": name, **rank_metrics(yte, st), "precision": tm["precision"],
                         "recall": tm["recall"], "cost": tm["cost"]})
        print(f"seed {seed} done", flush=True)
    d = pd.DataFrame(recs)
    d.to_csv(REPORTS / "imbalance_multiseed_raw.csv", index=False)
    agg = d.groupby("model")[["pr_auc", "roc_auc", "precision", "recall", "cost"]].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.sort_values("pr_auc_mean", ascending=False)
    agg.to_csv(REPORTS / "imbalance_multiseed.csv")
    print(agg.to_string())
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    run(tuple(range(11, 11 + 11 * a.seeds, 11)))
