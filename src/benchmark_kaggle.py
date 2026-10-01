"""Optional benchmark on the public Kaggle "Credit Card Fraud Detection" dataset (creditcard.csv).

    1. Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    2. Put it in ./data/creditcard.csv
    3. python -m src.benchmark_kaggle

That dataset has anonymised PCA features (V1..V28), Time, Amount and Class - no card/device/IP identifiers,
so graph features cannot be built; this script therefore compares the model families and the imbalance
strategies on the flat features only, with the same time-based split, calibration and cost-based threshold.
(It was tested on a mock file with the same schema; the real file was not available in the build environment,
so no Kaggle numbers are claimed in the README.)
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from config import DATA_DIR, REPORTS, TRAIN_FRAC, VAL_FRAC
from src.models import (Calibrator, IForestScorer, best_threshold, logit, rank_metrics, threshold_metrics,
                        train_logreg, train_rf, train_xgb)


def run(path, review_cost_note=True):
    df = pd.read_csv(path).sort_values("Time", kind="stable").reset_index(drop=True)
    need = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    if not need.issubset(df.columns):
        sys.exit(f"{path} is missing columns: {sorted(need - set(df.columns))}")
    X = df[[f"V{i}" for i in range(1, 29)] + ["Amount"]].astype("float32")
    y, amt = df["Class"].to_numpy(), df["Amount"].to_numpy()
    n = len(df); i1, i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    Xtr, Xva, Xte = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
    ytr, yva, yte, ava, ate = y[:i1], y[i1:i2], y[i2:], amt[i1:i2], amt[i2:]
    print(f"rows={n:,} frauds={int(y.sum())} (train {int(ytr.sum())}, val {int(yva.sum())}, test {int(yte.sum())})")

    models = {"Logistic Regression": (train_logreg(Xtr, ytr), lambda m, X_: logit(m.predict_proba(X_)[:, 1])),
              "Random Forest": (train_rf(Xtr, ytr), lambda m, X_: logit(m.predict_proba(X_)[:, 1]))}
    for key, name in {"none": "XGBoost (no handling)", "class_weight": "XGBoost + class weights",
                      "smote": "XGBoost + SMOTE", "focal": "XGBoost + focal loss"}.items():
        models[name] = (train_xgb(Xtr, ytr, key), lambda m, X_: m.margin(X_))
    iso = IForestScorer().fit(Xtr)
    models["Isolation Forest (unsupervised)"] = (iso, lambda m, X_: m.score(X_))

    rows = []
    for name, (m, fn) in models.items():
        sv, st = fn(m, Xva), fn(m, Xte)
        cal = Calibrator().fit(sv, yva)
        thr = best_threshold(cal.transform(sv), yva, ava)
        tm = threshold_metrics(cal.transform(st), yte, ate, thr)
        rows.append({"model": name, **rank_metrics(yte, st), "precision": tm["precision"], "recall": tm["recall"], "cost": tm["cost"]})
        print(f"{name:<34} PR-AUC={rows[-1]['pr_auc']:.3f} ROC-AUC={rows[-1]['roc_auc']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "kaggle_benchmark.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DATA_DIR / "creditcard.csv"))
    run(ap.parse_args().path)
