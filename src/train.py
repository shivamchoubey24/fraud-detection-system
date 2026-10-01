"""End-to-end training / evaluation pipeline.

    python -m src.train            # ~2-4 minutes on a laptop CPU, < 1 GB RAM

Steps: generate data -> leakage-free streaming features -> time-based split -> train & compare
(baselines, imbalance strategies, unsupervised) -> graph ablation -> calibration -> cost-optimal threshold
-> SHAP importance -> save artifacts + reports + figures.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from config import (ARTIFACTS, DATA_DIR, FIGURES, N_TXN, REPORTS, REVIEW_COST, SEED, TRAIN_FRAC, VAL_FRAC,
                    FOCAL_ALPHA, FOCAL_GAMMA)
from src import graph_analysis
from src.data import generate_synthetic
from src.explain import global_importance
from src.features import ALL_FEATURES, NO_GRAPH_FEATURES, build_feature_frame
from src.models import (AutoencoderScorer, Calibrator, IForestScorer, best_threshold, bootstrap_pr_auc, curve_table,
                        logit, rank_metrics, threshold_metrics, total_cost, train_logreg, train_rf, train_xgb)

XGB_STRATEGIES = {"none": "XGBoost (no imbalance handling)",
                  "class_weight": "XGBoost + class weights",
                  "smote": "XGBoost + SMOTE",
                  "focal": "XGBoost + focal loss"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate(name, family, s_val, s_test, y_val, y_test, amt_val, amt_test):
    """Rank metrics on raw scores + calibrated, cost-optimal thresholding (threshold chosen on VAL only)."""
    cal = Calibrator().fit(s_val, y_val)
    p_val, p_test = cal.transform(s_val), cal.transform(s_test)
    thr = best_threshold(p_val, y_val, amt_val)
    row = {"model": name, "family": family, **rank_metrics(y_test, s_test),
           "val_pr_auc": rank_metrics(y_val, s_val)["pr_auc"]}
    row.update({k: v for k, v in threshold_metrics(p_test, y_test, amt_test, thr).items()})
    return row, cal, p_test


def main(n_txn: int = N_TXN, plots: bool = True):
    t_start = time.time()
    for d in (ARTIFACTS, REPORTS, FIGURES, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    log(f"Generating {n_txn:,} synthetic transactions ...")
    df = generate_synthetic(n_txn)
    df.to_csv(DATA_DIR / "transactions.csv", index=False)
    log(f"fraud rate = {100 * df.is_fraud.mean():.2f}%  ({int(df.is_fraud.sum())} frauds)")

    log("Building leakage-free streaming features (velocity + graph) ...")
    X, store = build_feature_frame(df)
    y, amount = df["is_fraud"].to_numpy(), df["amount"].to_numpy()

    n = len(df)
    i1, i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    sl = {"train": slice(0, i1), "val": slice(i1, i2), "test": slice(i2, n)}
    assert df["timestamp"].iloc[i1 - 1] <= df["timestamp"].iloc[i1] and df["timestamp"].iloc[i2 - 1] <= df["timestamp"].iloc[i2]
    Xtr, Xva, Xte = X.iloc[sl["train"]], X.iloc[sl["val"]], X.iloc[sl["test"]]
    ytr, yva, yte = y[sl["train"]], y[sl["val"]], y[sl["test"]]
    atr, ava, ate = amount[sl["train"]], amount[sl["val"]], amount[sl["test"]]
    split_info = {k: {"rows": int(v.stop - v.start), "frauds": int(y[v].sum()),
                      "fraud_rate_pct": round(100 * float(y[v].mean()), 3)} for k, v in sl.items()}
    log(f"split (by time): {split_info}")

    rows, scores_test, fitted = [], {}, {}

    def add(name, family, s_val, s_test):
        row, cal, p_test = evaluate(name, family, s_val, s_test, yva, yte, ava, ate)
        rows.append(row)
        scores_test[name] = np.asarray(s_test)
        log(f"{name:<44} PR-AUC={row['pr_auc']:.3f} ROC-AUC={row['roc_auc']:.3f} "
            f"P={row['precision']:.2f} R={row['recall']:.2f} cost=${row['cost']:,.0f}")
        return cal, p_test

    # ------------------------------------------------------------ baselines
    log("Training baselines ...")
    lr = train_logreg(Xtr, ytr)
    add("Logistic Regression (baseline)", "baseline",
        logit(lr.predict_proba(Xva)[:, 1]), logit(lr.predict_proba(Xte)[:, 1]))
    rf = train_rf(Xtr, ytr)
    add("Random Forest", "baseline", logit(rf.predict_proba(Xva)[:, 1]), logit(rf.predict_proba(Xte)[:, 1]))

    # --------------------------- XGBoost + imbalance strategy comparison
    log("Training XGBoost with 4 imbalance strategies (none / class weights / SMOTE / focal loss) ...")
    xgb_models = {}
    for key, name in XGB_STRATEGIES.items():
        m = train_xgb(Xtr, ytr, key)
        xgb_models[key] = m
        add(name, "imbalance", m.margin(Xva), m.margin(Xte))

    # ---------------------------------------------------------- unsupervised
    log("Training unsupervised detectors (Isolation Forest, Autoencoder) ...")
    iso = IForestScorer().fit(Xtr)
    add("Isolation Forest (unsupervised)", "unsupervised", iso.score(Xva), iso.score(Xte))
    ae = AutoencoderScorer().fit(Xtr[ytr == 0])
    add("Autoencoder (unsupervised)", "unsupervised", ae.score(Xva), ae.score(Xte))

    # -------------------------------------------- pick final (validation PR-AUC)
    best_key = max(XGB_STRATEGIES, key=lambda k: next(r["val_pr_auc"] for r in rows if r["model"] == XGB_STRATEGIES[k]))
    final = xgb_models[best_key]
    final_name = XGB_STRATEGIES[best_key]
    log(f"FINAL MODEL = {final_name} (best validation PR-AUC)")

    # ------------------------------------------------- graph feature ablation
    log("Ablation: final strategy WITHOUT graph features ...")
    m_ng = train_xgb(Xtr[NO_GRAPH_FEATURES], ytr, best_key)
    add("Ablation: same model without graph features", "ablation",
        m_ng.margin(Xva[NO_GRAPH_FEATURES]), m_ng.margin(Xte[NO_GRAPH_FEATURES]))

    # ------------------------------------------ final calibration + threshold
    s_val, s_test = final.margin(Xva), final.margin(Xte)
    cal = Calibrator().fit(s_val, yva)
    p_val, p_test = cal.transform(s_val), cal.transform(s_test)
    thr = best_threshold(p_val, yva, ava)
    final_metrics = threshold_metrics(p_test, yte, ate, thr)
    ci = bootstrap_pr_auc(yte, s_test)
    no_model_cost = float(ate[yte == 1].sum())
    flag_all_cost = total_cost(yte, np.ones(len(yte), bool), ate)
    curve = curve_table(p_test, yte, ate)
    importance = global_importance(final.booster, Xte)
    log(f"threshold (val cost-optimal) = {thr:.3f} | test: {final_metrics}")
    log(f"PR-AUC 95% bootstrap CI = [{ci[0]:.3f}, {ci[1]:.3f}] | cost without model = ${no_model_cost:,.0f}")

    # ------------------------------------------------ NetworkX ring analysis
    log("NetworkX fraud-ring analysis ...")
    g = graph_analysis.build_graph(df)
    ok = graph_analysis.verify_against_store(g, store)
    comps = graph_analysis.component_table(g)
    assert ok, "streaming union-find disagrees with NetworkX"
    comps.to_csv(REPORTS / "suspicious_components.csv", index=False)
    log(f"union-find == NetworkX components: {ok}; suspicious components found: {len(comps)}")

    # ----------------------------------------------------------- save artifacts
    final.booster.save_model(str(ARTIFACTS / "model.json"))
    reloaded = xgb.Booster(); reloaded.load_model(str(ARTIFACTS / "model.json"))
    chk = reloaded.predict(xgb.DMatrix(Xte.iloc[:200], feature_names=ALL_FEATURES), output_margin=True)
    assert np.allclose(chk, s_test[:200], atol=1e-4), "saved model does not reproduce in-memory predictions"
    joblib.dump(cal, ARTIFACTS / "calibrator.joblib")
    joblib.dump(store, ARTIFACTS / "store.joblib")
    joblib.dump(iso, ARTIFACTS / "iforest.joblib")

    replay = pd.concat([df.iloc[sl["test"]].reset_index(drop=True), Xte.reset_index(drop=True)], axis=1)
    replay.to_csv(ARTIFACTS / "replay.csv", index=False)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    meta = {
        "final_model": final_name, "final_strategy": best_key, "features": ALL_FEATURES,
        "threshold": thr, "review_cost": REVIEW_COST, "split": split_info, "seed": SEED,
        "focal": {"gamma": FOCAL_GAMMA, "alpha": FOCAL_ALPHA},
        "final_test": {**final_metrics, **rank_metrics(yte, s_test), "pr_auc_ci95": ci,
                       "cost_without_model": no_model_cost, "cost_flag_everything": flag_all_cost,
                       "savings_pct": 100 * (1 - final_metrics["cost"] / no_model_cost)},
        "comparison": rows, "curve": curve, "importance": importance,
        "data": {"n_txn": n, "fraud_rate_pct": round(100 * float(y.mean()), 3), "synthetic": True},
        "trained_seconds": round(time.time() - t_start, 1),
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))
    (REPORTS / "metrics.json").write_text(json.dumps({k: v for k, v in meta.items() if k not in ("curve",)}, indent=2))

    if plots:
        log("Saving figures ...")
        make_figures(scores_test, yte, curve, thr, importance, final, Xte, g)

    log(f"DONE in {time.time() - t_start:.0f}s. Artifacts in {ARTIFACTS}")
    return meta


def make_figures(scores_test, yte, curve, thr, importance, final, Xte, g):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores_test.items():
        p, r, _ = precision_recall_curve(yte, s)
        ax.plot(r, p, lw=1.6 if "XGBoost" in name else 1.0, label=name)
    ax.axhline(yte.mean(), color="k", ls=":", lw=0.8, label="random (fraud rate)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curves (test set)")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIGURES / "pr_curves.png", dpi=120); plt.close(fig)

    t = [c["threshold"] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, [c["cost"] for c in curve], color="#d62728")
    ax.axvline(thr, color="k", ls="--", label=f"cost-optimal threshold = {thr:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Total cost on test set ($)")
    ax.set_title("Business cost vs threshold (missed fraud = amount, review = $%g)" % REVIEW_COST)
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIGURES / "cost_vs_threshold.png", dpi=120); plt.close(fig)

    top = importance[:12][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([d["feature"] for d in top], [d["mean_abs_shap"] for d in top], color="#1f77b4")
    ax.set_xlabel("mean |SHAP| (log-odds)"); ax.set_title("Global feature importance (TreeSHAP)")
    fig.tight_layout(); fig.savefig(FIGURES / "shap_importance.png", dpi=120); plt.close(fig)

    try:                                      # optional beeswarm with the official shap package
        import shap
        sample = Xte.sample(min(1500, len(Xte)), random_state=SEED)
        sv = shap.TreeExplainer(final.booster).shap_values(sample)
        plt.figure()
        shap.summary_plot(sv, sample, show=False, max_display=12)
        plt.tight_layout(); plt.savefig(FIGURES / "shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()
    except Exception as e:                    # never fail the pipeline for an optional plot
        log(f"(optional shap beeswarm skipped: {type(e).__name__})")

    graph_analysis.plot_rings(g, FIGURES / "fraud_rings.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-txn", type=int, default=N_TXN)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    main(a.n_txn, plots=not a.no_plots)
