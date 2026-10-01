"""Per-transaction explanations with TreeSHAP.

Contributions come from XGBoost's native TreeSHAP (`pred_contribs=True`), which is the same algorithm as
`shap.TreeExplainer`, but without importing numba/shap at serving time (much lighter on an old laptop).
`tests/test_pipeline.py` verifies that both give identical values.
Values are in log-odds (margin) space: positive => pushes the transaction towards FRAUD.
"""
from __future__ import annotations

import math

import numpy as np
import xgboost as xgb

from config import CATEGORIES

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _dur(s):
    s = int(s)
    if s >= 7 * 86400:
        return "7+ days"
    if s >= 86400:
        return f"{s / 86400:.1f} days"
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{s // 60} min"
    return f"{s} s"


def describe(feature: str, v: float) -> str:
    if feature.startswith("cat_"):
        c = feature[4:].replace("_", " ")
        return f"Merchant category is {c}" if v else f"Merchant is not {c}"
    return {
        "amount": lambda: f"Amount is ${v:,.2f}",
        "log_amount": lambda: f"Amount is ${math.expm1(v):,.2f}",
        "amount_ratio": lambda: f"Amount is {v:.1f}x this card's recent average",
        "hour": lambda: f"Transaction at {int(v):02d}:00 (UTC)",
        "day_of_week": lambda: f"Transaction on a {DOW[int(v) % 7]}",
        "is_night": lambda: "Night-time transaction (00-06h)" if v else "Not a night-time transaction",
        "is_foreign": lambda: "Foreign transaction" if v else "Domestic transaction",
        "dist_from_home_km": lambda: f"{v:,.0f} km from the card's home area",
        "card_age_days": lambda: f"Card is {int(v)} days old",
        "is_new_device": lambda: "Card used on a device it has not used before" if v else "Card used on a known device",
        "card_txn_1h": lambda: f"{int(v)} transaction(s) on this card within 1 hour",
        "card_txn_24h": lambda: f"{int(v)} transaction(s) on this card within 24 hours",
        "secs_since_last": lambda: f"{_dur(v)} since this card's previous transaction",
        "device_n_cards": lambda: f"Device linked to {int(v)} different card(s)",
        "ip_n_cards": lambda: f"IP address linked to {int(v)} different card(s)",
        "card_n_devices": lambda: f"Card linked to {int(v)} different device(s)",
        "component_log_size": lambda: f"Part of a connected network of ~{int(round(math.expm1(v)))} cards/devices/IPs",
    }.get(feature, lambda: f"{feature} = {v:.3g}")()


def contributions(booster: xgb.Booster, X) -> np.ndarray:
    """(n_rows, n_features + 1) TreeSHAP matrix; the last column is the bias term."""
    return booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)


def top_reasons(columns, values, contrib, top_k: int = 6) -> list[dict]:
    """Top-k features by |SHAP| from one row of contributions (bias column already removed)."""
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [{"feature": columns[i], "value": float(values[i]), "shap": float(contrib[i]),
             "text": describe(columns[i], float(values[i])),
             "direction": "increases risk" if contrib[i] > 0 else "decreases risk"} for i in order]


def explain_row(booster: xgb.Booster, X_row, top_k: int = 6) -> list[dict]:
    contrib = contributions(booster, X_row)[0][:-1]
    return top_reasons(list(X_row.columns), X_row.iloc[0].to_numpy(dtype=float), contrib, top_k)


def global_importance(booster: xgb.Booster, X, max_rows: int = 3000) -> list[dict]:
    Xs = X.iloc[:max_rows] if len(X) > max_rows else X
    imp = np.abs(contributions(booster, Xs)[:, :-1]).mean(axis=0)
    return sorted(({"feature": f, "mean_abs_shap": float(v)} for f, v in zip(X.columns, imp)),
                  key=lambda d: -d["mean_abs_shap"])
