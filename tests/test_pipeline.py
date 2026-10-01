"""Run with:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.data import generate_synthetic
from src.features import ALL_FEATURES, FeatureStore, build_feature_frame, featurize
from src.models import Calibrator, focal_grad, focal_loss, sigmoid, train_xgb, best_threshold, total_cost


@pytest.fixture(scope="module")
def small():
    return generate_synthetic(6000, seed=3)


def test_data_properties(small):
    assert small["timestamp"].is_monotonic_increasing
    assert 0 < small["is_fraud"].mean() < 0.01 + 1e-9           # rare (<1 %)
    assert small["txn_id"].is_unique
    assert set(small.columns) >= {"card_id", "device_id", "ip_id", "amount", "is_fraud"}


def test_focal_gradient_matches_numeric():
    z = np.linspace(-6, 6, 31)
    for y in (0, 1):
        num = np.array([(focal_loss(np.array([a + 1e-5]), np.array([y])) - focal_loss(np.array([a - 1e-5]), np.array([y]))) / 2e-5 for a in z])
        assert np.allclose(num, focal_grad(z, np.full_like(z, y)), atol=1e-6)


def test_features_use_only_the_past(small):
    """Features of the first k rows must be identical whether or not later rows exist."""
    k = 2500
    X_full, _ = build_feature_frame(small)
    X_part, _ = build_feature_frame(small.iloc[:k])
    pd.testing.assert_frame_equal(X_full.iloc[:k].reset_index(drop=True), X_part.reset_index(drop=True))


def test_serving_peek_equals_training_features(small):
    """Online scoring (commit=False) must give the same features as the offline sequential pass."""
    X_full, _ = build_feature_frame(small)
    k = 3000
    store = FeatureStore()
    for t in small.iloc[:k].to_dict("records"):
        featurize(t, store, commit=True)
    peek = featurize(small.iloc[k].to_dict(), store, commit=False)
    assert np.allclose([peek[c] for c in ALL_FEATURES], X_full.iloc[k].to_numpy(), atol=1e-4)
    seen = store.n_seen
    featurize(small.iloc[k].to_dict(), store, commit=False)
    assert store.n_seen == seen                                 # peek must not mutate state


def test_union_find_matches_networkx(small):
    from src import graph_analysis
    _, store = build_feature_frame(small)
    assert graph_analysis.verify_against_store(graph_analysis.build_graph(small), store)


def test_native_treeshap_equals_shap_library(small):
    shap = pytest.importorskip("shap")
    X, _ = build_feature_frame(small)
    m = train_xgb(X, small["is_fraud"].to_numpy(), "none", rounds=40)
    sample = X.iloc[:50]
    native = m.booster.predict(xgb.DMatrix(sample, feature_names=list(sample.columns)), pred_contribs=True)[:, :-1]
    lib = shap.TreeExplainer(m.booster).shap_values(sample)
    assert np.allclose(native, lib, atol=1e-3)
    # contributions + bias reproduce the margin exactly
    full = m.booster.predict(xgb.DMatrix(sample, feature_names=list(sample.columns)), pred_contribs=True)
    assert np.allclose(full.sum(axis=1), m.margin(sample), atol=1e-3)


def test_calibrator_is_monotonic_and_cost_threshold_beats_extremes():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.05).astype(int)
    s = rng.normal(size=4000) + 2.5 * y
    amt = rng.uniform(10, 200, 4000)
    cal = Calibrator().fit(s, y)
    grid = np.linspace(-3, 6, 50)
    assert np.all(np.diff(cal.transform(grid)) >= 0)
    p = cal.transform(s)
    thr = best_threshold(p, y, amt)
    assert total_cost(y, p >= thr, amt) <= min(total_cost(y, np.zeros(4000, bool), amt), total_cost(y, np.ones(4000, bool), amt))


# ------------------------------------------------------------------ API
@pytest.fixture(scope="module")
def client():
    from config import ARTIFACTS
    if not (ARTIFACTS / "model.json").exists():
        pytest.skip("run `python -m src.train` first")
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_api_health_and_meta(client):
    assert client.get("/health").json()["status"] == "ok"
    assert 0 < client.get("/api/meta").json()["threshold"] < 1
    assert len(client.get("/api/curve").json()["curve"]) == 99
    assert client.get("/").status_code == 200


def test_api_presets_behave_sensibly(client):
    res = {p["name"]: client.post("/api/score", json=p["transaction"]).json() for p in client.get("/api/presets").json()}
    assert res["Everyday purchase"]["decision"] == "APPROVE"
    assert res["Stolen card"]["decision"] == "FLAG"
    assert res["Fraud ring"]["decision"] == "FLAG"
    for r in res.values():
        assert 0 <= r["risk_score"] <= 100 and len(r["reasons"]) == 6 and r["latency_ms"] < 500


def test_api_sample_has_label_and_reasons(client):
    j = client.get("/api/sample?fraud_ratio=1").json()
    assert j["label"] == 1 and "transaction" in j and j["reasons"]


def test_api_validation(client):
    bad = {"card_id": "x", "merchant_category": "nope", "amount": 5, "device_id": "d", "ip_id": "i"}
    assert client.post("/api/score", json=bad).status_code == 422
    bad.update(merchant_category="fuel", amount=-1)
    assert client.post("/api/score", json=bad).status_code == 422


def test_scoring_does_not_mutate_state_by_default(client):
    from app.main import bundle
    b = bundle(); before = b.store.n_seen
    p = client.get("/api/presets").json()[0]["transaction"]
    client.post("/api/score", json=p)
    assert b.store.n_seen == before
