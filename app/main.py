"""FastAPI service: real-time transaction scoring + explanations + dashboard.

    uvicorn app.main:app --port 8000        ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from config import ARTIFACTS, CATEGORIES
from src.explain import top_reasons
from src.features import ALL_FEATURES, featurize, now_ts

STATIC = Path(__file__).parent / "static"
RAW_COLS = ["txn_id", "timestamp", "card_id", "merchant_id", "merchant_category", "amount", "device_id",
            "ip_id", "is_foreign", "dist_from_home_km", "card_age_days"]
_lock = threading.Lock()


class Transaction(BaseModel):
    txn_id: Optional[str] = None
    timestamp: Optional[int] = Field(None, description="unix seconds (UTC); default = now")
    card_id: str = Field(..., min_length=1, max_length=64)
    merchant_id: str = Field("M000", max_length=64)
    merchant_category: str
    amount: float = Field(..., gt=0, le=1_000_000)
    device_id: str = Field(..., min_length=1, max_length=64)
    ip_id: str = Field(..., min_length=1, max_length=64)
    is_foreign: int = Field(0, ge=0, le=1)
    dist_from_home_km: float = Field(0.0, ge=0, le=20_000)
    card_age_days: int = Field(365, ge=0, le=40_000)

    @field_validator("merchant_category")
    @classmethod
    def _cat(cls, v):
        if v not in CATEGORIES:
            raise ValueError(f"merchant_category must be one of {CATEGORIES}")
        return v


class Bundle:
    """Everything loaded once from ./artifacts."""

    def __init__(self):
        need = ["model.json", "calibrator.joblib", "store.joblib", "iforest.joblib", "meta.json", "replay.csv"]
        missing = [f for f in need if not (ARTIFACTS / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing artifacts {missing}. Run:  python -m src.train")
        self.meta = json.loads((ARTIFACTS / "meta.json").read_text())
        self.booster = xgb.Booster()
        self.booster.load_model(str(ARTIFACTS / "model.json"))
        self.booster.set_param({"nthread": 1})
        self.cal = joblib.load(ARTIFACTS / "calibrator.joblib")
        self.store = joblib.load(ARTIFACTS / "store.joblib")
        self.iso = joblib.load(ARTIFACTS / "iforest.joblib")
        self.replay = pd.read_csv(ARTIFACTS / "replay.csv")
        self.replay[ALL_FEATURES] = self.replay[ALL_FEATURES].astype("float32")
        self.threshold = float(self.meta["threshold"])
        self.fraud_idx = np.flatnonzero(self.replay["is_fraud"].to_numpy() == 1)
        self.legit_idx = np.flatnonzero(self.replay["is_fraud"].to_numpy() == 0)
        self.rng = np.random.default_rng()

    # ---------------------------------------------------------------
    def score_features(self, X: pd.DataFrame) -> dict:
        t0 = time.perf_counter()
        dm = xgb.DMatrix(X, feature_names=ALL_FEATURES)
        contrib = self.booster.predict(dm, pred_contribs=True)[0]       # TreeSHAP; last col = bias
        margin = float(contrib.sum())                                   # == model margin
        prob = float(self.cal.transform([margin])[0])
        anomaly = self.iso.percentile(float(self.iso.score(X)[0]))
        reasons = top_reasons(ALL_FEATURES, X.iloc[0].to_numpy(dtype=float), contrib[:-1], top_k=6)
        return {
            "risk_score": round(100 * prob, 1), "probability": prob, "threshold": self.threshold,
            "decision": "FLAG" if prob >= self.threshold else "APPROVE",
            "anomaly_percentile": round(100 * anomaly, 1), "reasons": reasons,
            "latency_ms": round(1000 * (time.perf_counter() - t0), 2),
        }

    def score_transaction(self, tx: dict, commit: bool = False) -> dict:
        tx = dict(tx)
        tx["timestamp"] = int(tx.get("timestamp") or now_ts())
        tx["txn_id"] = tx.get("txn_id") or f"LIVE-{tx['timestamp']}"
        with _lock:                                   # the store is shared mutable state
            feats = featurize(tx, self.store, commit=commit)
        X = pd.DataFrame([feats])[ALL_FEATURES].astype("float32")
        out = self.score_features(X)
        out["txn_id"] = tx["txn_id"]
        out["features"] = {k: round(float(v), 3) for k, v in feats.items()}
        return out

    def sample(self, fraud_ratio: float) -> dict:
        pool = self.fraud_idx if (self.rng.random() < fraud_ratio and len(self.fraud_idx)) else self.legit_idx
        i = int(self.rng.choice(pool))
        row = self.replay.iloc[i]
        X = self.replay[ALL_FEATURES].iloc[[i]].reset_index(drop=True)
        out = self.score_features(X)
        raw = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in RAW_COLS}
        out.update({"txn_id": raw["txn_id"], "transaction": raw, "label": int(row["is_fraud"]),
                    "features": {k: round(float(X.iloc[0][k]), 3) for k in ALL_FEATURES}})
        return out

    def presets(self) -> list[dict]:
        day0 = (now_ts() // 86400) * 86400
        r, store = self.replay, self.store
        home = r[(r["is_fraud"] == 0) & (r["is_new_device"] == 0) & (r["device_id"].str.startswith("DH"))].iloc[0]
        normal = dict(card_id=home["card_id"], merchant_id="M010", merchant_category="grocery", amount=42.5,
                      device_id=home["device_id"], ip_id=home["ip_id"], is_foreign=0, dist_from_home_km=3.0,
                      card_age_days=int(home["card_age_days"]), timestamp=day0 + 12 * 3600 + 600)
        stolen = dict(card_id=home["card_id"], merchant_id="M020", merchant_category="electronics", amount=950.0,
                      device_id="DNEW-4471", ip_id="INEW-4471", is_foreign=1, dist_from_home_km=480.0,
                      card_age_days=int(home["card_age_days"]), timestamp=day0 + 3 * 3600 + 720)
        ring_dev = max((d for d in store.device_cards if d.startswith("DX")), key=lambda d: len(store.device_cards[d]))
        ring_ip = max((i for i in store.ip_cards if i.startswith("IX")), key=lambda i: len(store.ip_cards[i]))
        ring_card = sorted(store.device_cards[ring_dev])[0]
        ring = dict(card_id=ring_card, merchant_id="M030", merchant_category="online_retail", amount=120.0,
                    device_id=ring_dev, ip_id=ring_ip, is_foreign=0, dist_from_home_km=15.0, card_age_days=75,
                    timestamp=day0 + 15 * 3600)
        return [{"name": "Everyday purchase", "hint": "Known card, known device, groceries at noon", "transaction": normal},
                {"name": "Stolen card", "hint": "Big foreign electronics purchase at 03:12 from a new device", "transaction": stolen},
                {"name": "Fraud ring", "hint": "Ordinary amount, but the device is shared by many cards", "transaction": ring}]


@lru_cache(maxsize=1)
def bundle() -> Bundle:
    return Bundle()


def get_bundle() -> Bundle:
    try:
        return bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


app = FastAPI(title="Real-Time Fraud Detection with Explainability", version="1.0.0")


@app.get("/health")
def health():
    b = get_bundle()
    return {"status": "ok", "model": b.meta["final_model"], "threshold": b.threshold,
            "features": len(ALL_FEATURES), "graph_cards_seen": b.store.n_seen}


@app.get("/api/meta")
def meta():
    m = dict(get_bundle().meta)
    for k in ("curve", "importance", "comparison"):
        m.pop(k, None)
    return m


@app.get("/api/metrics")
def metrics():
    m = get_bundle().meta
    return {"final_model": m["final_model"], "final_test": m["final_test"], "comparison": m["comparison"],
            "split": m["split"], "data": m["data"], "review_cost": m["review_cost"]}


@app.get("/api/curve")
def curve():
    b = get_bundle()
    return {"threshold_optimal": b.threshold, "review_cost": b.meta["review_cost"], "curve": b.meta["curve"],
            "cost_without_model": b.meta["final_test"]["cost_without_model"]}


@app.get("/api/importance")
def importance():
    return get_bundle().meta["importance"]


@app.get("/api/sample")
def sample(fraud_ratio: float = 0.15):
    """Replay a random held-out TEST transaction (the model never saw it in training)."""
    return get_bundle().sample(min(max(fraud_ratio, 0.0), 1.0))


@app.get("/api/presets")
def presets():
    return get_bundle().presets()


@app.post("/api/score")
def score(tx: Transaction, commit: bool = False):
    """Score one transaction.  commit=true also adds it to the live graph/velocity state."""
    return get_bundle().score_transaction(tx.model_dump(), commit=commit)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
