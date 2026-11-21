"""3.1 training loop + validation | 3.3 MLflow experiment tracking and model selection.

Run:  python -m src.train
Then: mlflow ui --backend-store-uri sqlite:///mlflow.db   (open http://127.0.0.1:5000)
"""
import json
import os
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, roc_auc_score

from . import config as C
from .features import FEATURES, LABEL, build_dataset, time_split
from .modeling import (best_threshold, build_pipeline, candidate_models, evaluate,
                       precision_at_k_per_day)

MODEL_VERSION = "surge-1.0.0"


def time_series_cv(pipe, df, n_folds=4):
    """Expanding-window CV on the training period, split by date with a 7-day gap."""
    dates = np.sort(df["dt"].unique())
    edges = np.linspace(len(dates) * 0.4, len(dates), n_folds + 1).astype(int)
    scores = []
    for i in range(n_folds):
        cut = dates[edges[i] - 1]
        tr = df[df["dt"] <= cut]
        va = df[(df["dt"] > cut + np.timedelta64(C.HORIZON_DAYS, "D")) &
                (df["dt"] <= dates[min(edges[i + 1], len(dates)) - 1])]
        if va[LABEL].nunique() < 2:
            continue
        m = clone(pipe).fit(tr[FEATURES], tr[LABEL])
        scores.append(roc_auc_score(va[LABEL], m.predict_proba(va[FEATURES])[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    C.MODEL_DIR.mkdir(exist_ok=True); C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    tr, va, te = time_split(df)
    print(f"rows train={len(tr)} val={len(va)} test={len(te)}  "
          f"surge rate train={tr[LABEL].mean():.3f} val={va[LABEL].mean():.3f} "
          f"test={te[LABEL].mean():.3f}")

    mlflow.set_tracking_uri(f"sqlite:///{C.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("tiktok_shop_sales_surge")
    pos_weight = float((tr[LABEL] == 0).sum() / (tr[LABEL] == 1).sum())

    rows, fitted = [], {}
    # Rule-based reference: "it is rising right now, so it will keep rising"
    rule = (va["momentum_3_7"] > 1.05).astype(float).values
    rows.append({"model": "v0_rule_momentum", **evaluate(va, va[LABEL], rule, 0.5),
                 "cv_auc_mean": np.nan, "cv_auc_std": np.nan, "threshold": 0.5,
                 "description": "Rule: last-3-day avg > 1.05 x last-7-day avg"})

    for name, (est, numeric, params, desc) in candidate_models(pos_weight).items():
        pipe = build_pipeline(est, numeric)
        with mlflow.start_run(run_name=name):
            cv_m, cv_s = time_series_cv(pipe, tr)
            pipe.fit(tr[FEATURES], tr[LABEL])
            p_va = pipe.predict_proba(va[FEATURES])[:, 1]
            thr = best_threshold(va[LABEL], p_va)
            m = evaluate(va, va[LABEL], p_va, thr)
            mlflow.log_params({**params, "model_family": type(est).__name__,
                               "n_features": len(numeric) + 1, "label": "surge>=1.2x next 7d",
                               "train_end": C.TRAIN_END})
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()} |
                               {"cv_auc_mean": cv_m, "cv_auc_std": cv_s, "threshold": thr})
            mlflow.set_tag("description", desc)
            fitted[name] = (pipe, thr, numeric)
            rows.append({"model": name, **m, "cv_auc_mean": cv_m, "cv_auc_std": cv_s,
                         "threshold": thr, "description": desc})
            print(f"{name:<24} cvAUC={cv_m:.3f}±{cv_s:.3f}  valAUC={m['roc_auc']:.3f}  "
                  f"valPR-AUC={m['pr_auc']:.3f}  F1={m['f1']:.3f}  P@10={m['precision_at_10']:.3f}")

    comp = pd.DataFrame(rows).set_index("model")
    comp.round(4).to_csv(C.REPORT_DIR / "model_comparison.csv")
    print("\n", comp.drop(columns="description").round(3).to_string())

    # ---- Model selection: highest validation PR-AUC (ranking quality on rare class) ----
    best = comp.drop(index="v0_rule_momentum")["pr_auc"].idxmax()
    pipe, thr, numeric = fitted[best]
    print(f"\nSelected: {best} (highest validation PR-AUC), threshold={thr:.2f}")

    # Refit on train+val (same hyper-parameters), evaluate ONCE on the untouched test period
    trva = pd.concat([tr, va])
    final = clone(pipe).fit(trva[FEATURES], trva[LABEL])
    p_te = final.predict_proba(te[FEATURES])[:, 1]
    test_m = evaluate(te, te[LABEL], p_te, thr)
    rule_te = evaluate(te, te[LABEL], (te["momentum_3_7"] > 1.05).astype(float).values, 0.5)
    print("Test metrics:", {k: round(v, 3) for k, v in test_m.items()})
    print("Rule on test:", {k: round(v, 3) for k, v in rule_te.items()})

    with mlflow.start_run(run_name=f"FINAL_{best}"):
        mlflow.log_params({"selected": best, "refit_on": "train+val", "threshold": thr})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.set_tag("model_version", MODEL_VERSION)

    # Figures for the report
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    PrecisionRecallDisplay.from_predictions(te[LABEL], p_te, ax=ax[0], name=best)
    ax[0].axhline(te[LABEL].mean(), ls="--", color="gray", lw=1, label="base rate")
    ax[0].legend(frameon=False); ax[0].set_title("Test precision–recall")
    ConfusionMatrixDisplay.from_predictions(te[LABEL], (p_te >= thr).astype(int), ax=ax[1],
                                            display_labels=["no surge", "surge"],
                                            colorbar=False, cmap="Blues")
    ax[1].set_title(f"Test confusion matrix (threshold {thr:.2f})")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "test_pr_confusion.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    plot = comp.drop(columns=["description"])[["roc_auc", "pr_auc", "f1", "precision_at_10"]]
    plot.plot.bar(ax=ax, width=.8, color=["#93c5fd", "#2563eb", "#f97316", "#16a34a"])
    ax.set_ylim(0, 1); ax.set_title("Validation metrics by model version"); ax.set_xlabel("")
    ax.legend(frameon=False, ncol=4, fontsize=8); plt.xticks(rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "model_comparison.png", dpi=130); plt.close(fig)

    bundle = {"pipeline": final, "threshold": thr, "features": FEATURES,
              "numeric": list(numeric), "model_name": best, "model_version": MODEL_VERSION,
              "train_end": str(trva["dt"].max().date()), "test_metrics": test_m}
    joblib.dump(bundle, C.MODEL_DIR / "surge_model.joblib")
    # Latest feature snapshot per product: lets the API score a product by its id
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    latest = build_dataset(drop_unlabelled=False)
    latest = latest.sort_values("dt").groupby("product_id").tail(1)
    latest[["product_id", "product_name", "dt", "sales_mean_7d"] + FEATURES].to_csv(
        C.PROCESSED_DIR / "latest_features.csv", index=False)
    # Test predictions for the explainability / fairness scripts
    te.assign(proba=p_te, pred=(p_te >= thr).astype(int)).to_csv(
        C.PROCESSED_DIR / "test_predictions.csv", index=False)
    json.dump({"selected": best, "threshold": thr, "model_version": MODEL_VERSION,
               "test": test_m, "rule_test": rule_te,
               "validation": comp.drop(columns="description").round(4).to_dict(orient="index")},
              open(C.REPORT_DIR / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
